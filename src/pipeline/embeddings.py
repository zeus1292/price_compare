"""
Embedding generation pipeline for product data.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.database.chroma_manager import ChromaManager
from src.database.sqlite_manager import SQLiteManager
from src.services.embedding_service import EmbeddingService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for embedding generation."""
    total_products: int = 0
    embedded: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


class EmbeddingPipeline:
    """
    Pipeline for generating and storing product embeddings.

    Can be used to:
    - Generate embeddings for new products
    - Re-embed products after model updates
    - Batch process existing database products
    """

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        batch_size: int = 50,
    ):
        self.sqlite = sqlite_manager or SQLiteManager()
        self.chroma = chroma_manager or ChromaManager()
        self.embeddings = embedding_service or EmbeddingService()
        self.batch_size = batch_size

        # Initialize database
        self.sqlite.initialize()

    async def embed_product(self, product: dict) -> dict:
        """
        Generate embeddings for a single product.

        Args:
            product: Product dictionary with 'name' field

        Returns:
            Dictionary with product_id and embeddings
        """
        name = product.get("name", "")
        if not name:
            return {"product_id": product.get("id"), "error": "No name field"}

        try:
            embedding = await self.embeddings.embed_text_async(name)

            # Store in ChromaDB
            product_id = str(product.get("id", ""))
            if product_id:
                self.chroma.add_product_embedding(
                    product_id=product_id,
                    name=name,
                    embedding=embedding,
                    metadata={
                        "merchant": product.get("merchant") or "",
                        "market": product.get("market") or "",
                        "price": product.get("price") or 0.0,
                    },
                )

            return {
                "product_id": product_id,
                "name_embedding": embedding,
            }

        except Exception as e:
            return {"product_id": product.get("id"), "error": str(e)}

    async def embed_products_batch(
        self,
        products: List[dict],
    ) -> EmbeddingStats:
        """
        Generate embeddings for multiple products.

        Args:
            products: List of product dictionaries

        Returns:
            EmbeddingStats with results
        """
        stats = EmbeddingStats(total_products=len(products))

        for i in range(0, len(products), self.batch_size):
            batch = products[i:i + self.batch_size]

            # Filter products with names
            valid_products = [p for p in batch if p.get("name")]
            stats.skipped += len(batch) - len(valid_products)

            if not valid_products:
                continue

            try:
                # Generate embeddings
                names = [p.get("name", "") for p in valid_products]
                embeddings = await self.embeddings.embed_texts_batch_async(names)

                # Store in ChromaDB
                product_ids = [str(p.get("id", i)) for i, p in enumerate(valid_products)]
                metadatas = [
                    {
                        "merchant": p.get("merchant") or "",
                        "market": p.get("market") or "",
                        "price": p.get("price") or 0.0,
                    }
                    for p in valid_products
                ]

                self.chroma.add_product_embeddings_batch(
                    product_ids=product_ids,
                    names=names,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                stats.embedded += len(valid_products)

            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                stats.failed += len(valid_products)
                stats.errors.append(str(e))

        return stats

    async def reembed_all_products(
        self,
        clear_existing: bool = False,
    ) -> EmbeddingStats:
        """
        Re-generate embeddings for all products in database.

        Args:
            clear_existing: Whether to clear existing embeddings first

        Returns:
            EmbeddingStats with results
        """
        if clear_existing:
            self.chroma.reset_all()
            logger.info("Cleared existing embeddings")

        stats = EmbeddingStats()
        page = 1
        page_size = 100

        while True:
            # Fetch products from SQLite
            products, total = await self.sqlite.list_products(
                page=page,
                limit=page_size,
            )

            if not products:
                break

            stats.total_products = total

            # Convert to dicts and embed
            product_dicts = [p.to_dict() for p in products]
            batch_stats = await self.embed_products_batch(product_dicts)

            stats.embedded += batch_stats.embedded
            stats.failed += batch_stats.failed
            stats.skipped += batch_stats.skipped
            stats.errors.extend(batch_stats.errors)

            logger.info(
                f"Re-embedded page {page}: "
                f"{stats.embedded}/{stats.total_products} products"
            )

            page += 1

        return stats

    async def update_product_embedding(
        self,
        product_id: str,
        product: dict,
    ) -> bool:
        """
        Update embedding for a specific product.

        Args:
            product_id: Product ID
            product: Updated product data

        Returns:
            True if successful
        """
        name = product.get("name", "")
        if not name:
            return False

        try:
            embedding = await self.embeddings.embed_text_async(name)

            self.chroma.update_product_embedding(
                product_id=product_id,
                embedding=embedding,
                document=name,
                metadata={
                    "merchant": product.get("merchant") or "",
                    "market": product.get("market") or "",
                    "price": product.get("price") or 0.0,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update embedding for {product_id}: {e}")
            return False

    async def delete_product_embedding(self, product_id: str) -> None:
        """
        Delete embedding for a product.

        Args:
            product_id: Product ID to delete
        """
        self.chroma.delete_product(product_id)

    def check_embedding_exists(self, product_id: str) -> bool:
        """
        Check if embedding exists for a product.

        Args:
            product_id: Product ID to check

        Returns:
            True if embedding exists
        """
        return self.chroma.product_exists(product_id)

    async def get_missing_embeddings(self) -> List[str]:
        """
        Find products that don't have embeddings.

        Returns:
            List of product IDs missing embeddings
        """
        missing = []
        page = 1
        page_size = 100

        while True:
            products, _ = await self.sqlite.list_products(
                page=page,
                limit=page_size,
            )

            if not products:
                break

            for product in products:
                if not self.chroma.product_exists(str(product.id)):
                    missing.append(str(product.id))

            page += 1

        return missing

    async def embed_missing_products(self) -> EmbeddingStats:
        """
        Generate embeddings for products that don't have them.

        Returns:
            EmbeddingStats with results
        """
        missing_ids = await self.get_missing_embeddings()

        if not missing_ids:
            return EmbeddingStats()

        logger.info(f"Found {len(missing_ids)} products missing embeddings")

        # Fetch and embed in batches
        stats = EmbeddingStats(total_products=len(missing_ids))

        for i in range(0, len(missing_ids), self.batch_size):
            batch_ids = missing_ids[i:i + self.batch_size]
            products = await self.sqlite.get_products_by_ids(batch_ids)
            product_dicts = [p.to_dict() for p in products]

            batch_stats = await self.embed_products_batch(product_dicts)

            stats.embedded += batch_stats.embedded
            stats.failed += batch_stats.failed
            stats.errors.extend(batch_stats.errors)

        return stats

    def get_stats(self) -> dict:
        """Get embedding pipeline statistics."""
        return {
            "chroma": self.chroma.get_stats(),
            "embedding_model": self.embeddings.model,
            "batch_size": self.batch_size,
        }
