"""
Embedding service for generating text embeddings using OpenAI.
"""
import asyncio
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI's API.

    Supports batch processing for efficient embedding generation.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key.get_secret_value()
        self.model = model or settings.embedding_model
        self.batch_size = settings.embedding_batch_size

        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        text = text.replace("\n", " ").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        response = self.sync_client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_text_async(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string asynchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        text = text.replace("\n", " ").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        response = await self.async_client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch = [t.replace("\n", " ").strip() for t in batch]
            batch = [t if t else "empty" for t in batch]

            response = self.sync_client.embeddings.create(
                input=batch,
                model=self.model,
            )

            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_texts_batch_async(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch = [t.replace("\n", " ").strip() for t in batch]
            batch = [t if t else "empty" for t in batch]

            response = await self.async_client.embeddings.create(
                input=batch,
                model=self.model,
            )

            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model, 1536)

    async def embed_product(self, product: dict) -> dict:
        """
        Generate embeddings for a product's name and description.

        Args:
            product: Product dictionary with 'name' and optional 'description'

        Returns:
            Dictionary with 'name_embedding' and 'description_embedding'
        """
        name = product.get("name", "")
        description = product.get("description", "")

        name_embedding = await self.embed_text_async(name) if name else None

        # Create description from available fields
        full_description = f"{name} {description}".strip()
        if product.get("category"):
            full_description += f" {product['category']}"
        if product.get("merchant"):
            full_description += f" from {product['merchant']}"

        description_embedding = (
            await self.embed_text_async(full_description) if full_description else None
        )

        return {
            "name_embedding": name_embedding,
            "description_embedding": description_embedding,
        }

    async def embed_products_batch(
        self,
        products: list[dict]
    ) -> list[dict]:
        """
        Generate embeddings for multiple products.

        Args:
            products: List of product dictionaries

        Returns:
            List of dictionaries with embeddings
        """
        names = [p.get("name", "") for p in products]

        descriptions = []
        for p in products:
            desc = f"{p.get('name', '')} {p.get('description', '')}".strip()
            if p.get("category"):
                desc += f" {p['category']}"
            if p.get("merchant"):
                desc += f" from {p['merchant']}"
            descriptions.append(desc)

        name_embeddings = await self.embed_texts_batch_async(names)
        desc_embeddings = await self.embed_texts_batch_async(descriptions)

        results = []
        for i, product in enumerate(products):
            results.append({
                "product_id": product.get("id"),
                "name_embedding": name_embeddings[i] if i < len(name_embeddings) else None,
                "description_embedding": desc_embeddings[i] if i < len(desc_embeddings) else None,
            })

        return results
