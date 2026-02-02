"""
Batch processor for Klarna dataset ingestion.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from src.config.settings import get_settings
from src.database.chroma_manager import ChromaManager
from src.database.sqlite_manager import SQLiteManager
from src.pipeline.klarna_parser import KlarnaParser, ParsedProduct
from src.services.embedding_service import EmbeddingService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_files: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed / self.total_files) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate processing duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_files": self.total_files,
            "processed": self.processed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "error_count": len(self.errors),
        }


class BatchProcessor:
    """
    Batch processor for large-scale dataset ingestion.

    Handles:
    - Parallel file parsing
    - Batch embedding generation
    - Database insertion with transactions
    - Progress tracking and error handling
    """

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        settings = get_settings()

        self.sqlite = sqlite_manager or SQLiteManager()
        self.chroma = chroma_manager or ChromaManager()
        self.embeddings = embedding_service or EmbeddingService()
        self.parser = KlarnaParser()

        self.batch_size = batch_size or settings.batch_size
        self.max_workers = max_workers or settings.max_workers
        self.embedding_batch_size = settings.embedding_batch_size

        # Initialize database
        self.sqlite.initialize()

    async def process_dataset(
        self,
        source_path: str,
        progress_callback: Optional[Callable[[ProcessingStats], None]] = None,
    ) -> ProcessingStats:
        """
        Process entire Klarna dataset.

        Args:
            source_path: Path to dataset root directory
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingStats with final statistics
        """
        stats = ProcessingStats()
        stats.start_time = datetime.utcnow()

        # Discover product directories
        source = Path(source_path)
        product_dirs = self._discover_product_dirs(source)
        stats.total_files = len(product_dirs)

        logger.info(f"Found {stats.total_files} product directories to process")

        # Process in batches
        for i in range(0, len(product_dirs), self.batch_size):
            batch = product_dirs[i:i + self.batch_size]

            try:
                batch_stats = await self._process_batch(batch)
                stats.processed += batch_stats["processed"]
                stats.failed += batch_stats["failed"]
                stats.skipped += batch_stats["skipped"]
                stats.errors.extend(batch_stats.get("errors", []))

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                stats.failed += len(batch)
                stats.errors.append(str(e))

            # Progress callback
            if progress_callback:
                progress_callback(stats)

            # Log progress
            progress = (i + len(batch)) / stats.total_files * 100
            logger.info(
                f"Progress: {progress:.1f}% "
                f"({stats.processed} processed, {stats.failed} failed)"
            )

        stats.end_time = datetime.utcnow()
        return stats

    def _discover_product_dirs(self, source: Path) -> List[Path]:
        """
        Discover all product directories in dataset.

        Looks for directories containing .mhtml, .wtl, or elements_metadata.json files.
        """
        product_dirs = []

        for path in source.rglob("*"):
            if path.is_dir():
                # Check if directory contains product files
                has_mhtml = list(path.glob("*.mhtml")) or list(path.glob("*.mht"))
                has_wtl = list(path.glob("*.wtl"))
                has_metadata = (path / "metadata.json").exists()
                # Klarna WTL dataset format
                has_elements_metadata = (path / "elements_metadata.json").exists()

                if has_mhtml or has_wtl or has_metadata or has_elements_metadata:
                    product_dirs.append(path)

        return product_dirs

    async def _process_batch(self, batch: List[Path]) -> dict:
        """
        Process a batch of product directories.

        Args:
            batch: List of directory paths

        Returns:
            Dictionary with batch statistics
        """
        stats = {"processed": 0, "failed": 0, "skipped": 0, "errors": []}

        # Parse products concurrently
        semaphore = asyncio.Semaphore(self.max_workers)

        async def parse_with_semaphore(dir_path: Path) -> Optional[ParsedProduct]:
            async with semaphore:
                return await asyncio.to_thread(
                    self.parser.parse_directory, str(dir_path)
                )

        parsed_tasks = [parse_with_semaphore(d) for d in batch]
        parsed_products = await asyncio.gather(*parsed_tasks, return_exceptions=True)

        # Filter valid products
        valid_products = []
        for product in parsed_products:
            if isinstance(product, Exception):
                stats["failed"] += 1
                stats["errors"].append(str(product))
            elif isinstance(product, ParsedProduct):
                if self.parser.is_valid_product(product):
                    valid_products.append(product)
                else:
                    stats["skipped"] += 1
            else:
                stats["skipped"] += 1

        if not valid_products:
            return stats

        # Generate embeddings in sub-batches
        product_dicts = [self.parser.to_dict(p) for p in valid_products]
        embeddings = await self._generate_embeddings_batch(product_dicts)

        # Insert into databases
        try:
            await self._insert_batch(product_dicts, embeddings)
            stats["processed"] = len(valid_products)
        except Exception as e:
            stats["failed"] += len(valid_products)
            stats["errors"].append(f"Insert error: {e}")

        return stats

    async def _generate_embeddings_batch(
        self,
        products: List[dict]
    ) -> List[dict]:
        """
        Generate embeddings for a batch of products.

        Args:
            products: List of product dictionaries

        Returns:
            List of embedding dictionaries
        """
        all_embeddings = []

        for i in range(0, len(products), self.embedding_batch_size):
            sub_batch = products[i:i + self.embedding_batch_size]

            # Prepare texts for embedding
            names = [p.get("name", "") for p in sub_batch]

            try:
                name_embeddings = await self.embeddings.embed_texts_batch_async(names)

                for j, product in enumerate(sub_batch):
                    all_embeddings.append({
                        "name_embedding": name_embeddings[j] if j < len(name_embeddings) else None,
                    })

            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
                # Add empty embeddings for this sub-batch
                all_embeddings.extend([{"name_embedding": None} for _ in sub_batch])

        return all_embeddings

    async def _insert_batch(
        self,
        products: List[dict],
        embeddings: List[dict]
    ) -> None:
        """
        Insert products and embeddings into databases.

        Args:
            products: Product dictionaries
            embeddings: Embedding dictionaries
        """
        # Prepare product data
        products_to_insert = []
        for product in products:
            # Ensure required fields
            product.setdefault("name_normalized", product.get("name", "").lower().strip())
            products_to_insert.append(product)

        # Insert into SQLite and get back products with generated IDs
        created_products = await self.sqlite.bulk_create_products(
            products_to_insert, return_products=True
        )
        logger.debug(f"Inserted {len(created_products)} products into SQLite")

        # Insert embeddings into ChromaDB using actual product IDs
        valid_embeddings = [
            (i, emb)
            for i, emb in enumerate(embeddings)
            if emb.get("name_embedding")
        ]

        if valid_embeddings:
            product_ids = []
            names = []
            embedding_vectors = []
            metadatas = []

            for i, emb in valid_embeddings:
                product = created_products[i]
                product_ids.append(str(product.id))  # Use actual UUID from database
                names.append(product.name or "")
                embedding_vectors.append(emb["name_embedding"])
                metadatas.append({
                    "merchant": product.merchant or "",
                    "market": product.market or "",
                    "price": product.price or 0.0,
                })

            self.chroma.add_product_embeddings_batch(
                product_ids=product_ids,
                names=names,
                embeddings=embedding_vectors,
                metadatas=metadatas,
            )
            logger.debug(f"Inserted {len(valid_embeddings)} embeddings into ChromaDB")

    async def process_single_file(self, file_path: str) -> Optional[dict]:
        """
        Process a single file (useful for testing).

        Args:
            file_path: Path to MHTML or WTL file

        Returns:
            Product dictionary or None
        """
        path = Path(file_path)

        if path.is_dir():
            product = self.parser.parse_directory(str(path))
        elif path.suffix.lower() in (".mhtml", ".mht"):
            product = self.parser.parse_mhtml(str(path))
        else:
            return None

        if not self.parser.is_valid_product(product):
            return None

        product_dict = self.parser.to_dict(product)

        # Generate embedding
        if product_dict.get("name"):
            embedding = await self.embeddings.embed_text_async(product_dict["name"])
            product_dict["name_embedding"] = embedding

        return product_dict

    def get_stats(self) -> dict:
        """Get current database statistics."""
        return {
            "chroma": self.chroma.get_stats(),
        }


class IngestionJob:
    """
    Manages a dataset ingestion job.

    Tracks progress and allows for resumable processing.
    """

    def __init__(
        self,
        job_id: str,
        source_path: str,
        processor: Optional[BatchProcessor] = None,
    ):
        self.job_id = job_id
        self.source_path = source_path
        self.processor = processor or BatchProcessor()
        self.stats = ProcessingStats()
        self.status = "pending"
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the ingestion job."""
        self.status = "running"

        def update_callback(stats: ProcessingStats):
            self.stats = stats

        try:
            self.stats = await self.processor.process_dataset(
                self.source_path,
                progress_callback=update_callback,
            )
            self.status = "completed"
        except Exception as e:
            self.status = "failed"
            self.stats.errors.append(str(e))
            raise

    def get_progress(self) -> dict:
        """Get current job progress."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": {
                "total": self.stats.total_files,
                "processed": self.stats.processed,
                "failed": self.stats.failed,
                "percentage": self.stats.success_rate,
            },
        }
