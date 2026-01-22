"""
ChromaDB manager for vector store operations.
Handles product embeddings and semantic search.
"""
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings


class ChromaManager:
    """
    Manages ChromaDB collections for semantic product search.

    Collections:
    - product_names: Embeddings of product names
    - product_descriptions: Embeddings of full product descriptions
    """

    COLLECTION_NAMES = "product_names"
    COLLECTION_DESCRIPTIONS = "product_descriptions"

    def __init__(self, persist_path: Optional[str] = None):
        settings = get_settings()
        self.persist_path = persist_path or settings.chroma_path

        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        self._collections: Dict[str, chromadb.Collection] = {}

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[dict] = None
    ) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        if name not in self._collections:
            default_metadata = {
                "description": f"Product {name} embeddings",
                "hnsw:space": "cosine",
            }
            if metadata:
                default_metadata.update(metadata)

            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata=default_metadata,
            )
        return self._collections[name]

    @property
    def names_collection(self) -> chromadb.Collection:
        """Get the product names collection."""
        return self.get_or_create_collection(
            self.COLLECTION_NAMES,
            {"description": "Embeddings of product names for semantic search"}
        )

    @property
    def descriptions_collection(self) -> chromadb.Collection:
        """Get the product descriptions collection."""
        return self.get_or_create_collection(
            self.COLLECTION_DESCRIPTIONS,
            {"description": "Embeddings of full product descriptions"}
        )

    # Add Operations

    def add_product_embedding(
        self,
        product_id: str,
        name: str,
        embedding: List[float],
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a single product name embedding."""
        meta = metadata or {}
        self.names_collection.add(
            ids=[product_id],
            embeddings=[embedding],
            documents=[name],
            metadatas=[meta],
        )

    def add_product_embeddings_batch(
        self,
        product_ids: List[str],
        names: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Add multiple product name embeddings in batch."""
        metas = metadatas or [{} for _ in product_ids]
        self.names_collection.add(
            ids=product_ids,
            embeddings=embeddings,
            documents=names,
            metadatas=metas,
        )

    def add_description_embedding(
        self,
        product_id: str,
        description: str,
        embedding: List[float],
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a single product description embedding."""
        meta = metadata or {}
        self.descriptions_collection.add(
            ids=[product_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[meta],
        )

    def add_description_embeddings_batch(
        self,
        product_ids: List[str],
        descriptions: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Add multiple product description embeddings in batch."""
        metas = metadatas or [{} for _ in product_ids]
        self.descriptions_collection.add(
            ids=product_ids,
            embeddings=embeddings,
            documents=descriptions,
            metadatas=metas,
        )

    # Query Operations

    def query_by_embedding(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        filter_ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Query collection by embedding similarity.

        Args:
            collection_name: Name of collection to query
            query_embedding: Query embedding vector
            limit: Maximum results to return
            filter_ids: If provided, only search within these IDs
            where: ChromaDB filter conditions

        Returns:
            Query results with ids, distances, documents, and metadatas
        """
        collection = self.get_or_create_collection(collection_name)

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_ids:
            query_params["where"] = {"$and": [{"id": {"$in": filter_ids}}]}
            if where:
                query_params["where"]["$and"].append(where)
        elif where:
            query_params["where"] = where

        results = collection.query(**query_params)

        # Flatten results (ChromaDB returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }

    def query_names(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_ids: Optional[List[str]] = None,
        merchant: Optional[str] = None,
        market: Optional[str] = None,
    ) -> dict:
        """Query product names collection."""
        where = {}
        if merchant:
            where["merchant"] = merchant
        if market:
            where["market"] = market

        return self.query_by_embedding(
            collection_name=self.COLLECTION_NAMES,
            query_embedding=query_embedding,
            limit=limit,
            filter_ids=filter_ids,
            where=where if where else None,
        )

    def query_descriptions(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_ids: Optional[List[str]] = None,
    ) -> dict:
        """Query product descriptions collection."""
        return self.query_by_embedding(
            collection_name=self.COLLECTION_DESCRIPTIONS,
            query_embedding=query_embedding,
            limit=limit,
            filter_ids=filter_ids,
        )

    # Update Operations

    def update_product_metadata(
        self,
        product_id: str,
        metadata: dict,
        collection_name: str = COLLECTION_NAMES,
    ) -> None:
        """Update metadata for a product embedding."""
        collection = self.get_or_create_collection(collection_name)
        collection.update(
            ids=[product_id],
            metadatas=[metadata],
        )

    def update_product_embedding(
        self,
        product_id: str,
        embedding: List[float],
        document: Optional[str] = None,
        metadata: Optional[dict] = None,
        collection_name: str = COLLECTION_NAMES,
    ) -> None:
        """Update embedding for a product."""
        collection = self.get_or_create_collection(collection_name)
        update_params = {
            "ids": [product_id],
            "embeddings": [embedding],
        }
        if document:
            update_params["documents"] = [document]
        if metadata:
            update_params["metadatas"] = [metadata]

        collection.update(**update_params)

    # Delete Operations

    def delete_product(self, product_id: str) -> None:
        """Delete a product from all collections."""
        for collection_name in [self.COLLECTION_NAMES, self.COLLECTION_DESCRIPTIONS]:
            try:
                collection = self.get_or_create_collection(collection_name)
                collection.delete(ids=[product_id])
            except Exception:
                pass  # Product may not exist in this collection

    def delete_products_batch(self, product_ids: List[str]) -> None:
        """Delete multiple products from all collections."""
        for collection_name in [self.COLLECTION_NAMES, self.COLLECTION_DESCRIPTIONS]:
            try:
                collection = self.get_or_create_collection(collection_name)
                collection.delete(ids=product_ids)
            except Exception:
                pass

    # Utility Operations

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of items in a collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def get_stats(self) -> dict:
        """Get ChromaDB statistics."""
        return {
            "names_count": self.get_collection_count(self.COLLECTION_NAMES),
            "descriptions_count": self.get_collection_count(self.COLLECTION_DESCRIPTIONS),
            "persist_path": self.persist_path,
        }

    def reset_collection(self, collection_name: str) -> None:
        """Reset (clear) a collection."""
        try:
            self.client.delete_collection(collection_name)
            del self._collections[collection_name]
        except Exception:
            pass

    def reset_all(self) -> None:
        """Reset all collections."""
        self.reset_collection(self.COLLECTION_NAMES)
        self.reset_collection(self.COLLECTION_DESCRIPTIONS)
        self._collections.clear()

    def get_product_embedding(
        self,
        product_id: str,
        collection_name: str = COLLECTION_NAMES
    ) -> Optional[dict]:
        """Get a product's embedding and metadata."""
        collection = self.get_or_create_collection(collection_name)
        try:
            result = collection.get(
                ids=[product_id],
                include=["embeddings", "documents", "metadatas"]
            )
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "embedding": result["embeddings"][0] if result["embeddings"] else None,
                    "document": result["documents"][0] if result["documents"] else None,
                    "metadata": result["metadatas"][0] if result["metadatas"] else None,
                }
        except Exception:
            pass
        return None

    def product_exists(
        self,
        product_id: str,
        collection_name: str = COLLECTION_NAMES
    ) -> bool:
        """Check if a product exists in a collection."""
        collection = self.get_or_create_collection(collection_name)
        try:
            result = collection.get(ids=[product_id])
            return len(result["ids"]) > 0
        except Exception:
            return False
