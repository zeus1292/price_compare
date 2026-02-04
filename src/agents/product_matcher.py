"""
Product Matching Agent for searching the database.

Supports:
- Text-based hybrid search (SQL + OpenAI embeddings)
- Image-based search (CLIP embeddings for direct visual similarity)
"""
import logging
from typing import List, Optional

from langsmith import traceable

from src.agents.base_agent import BaseAgent
from src.database.chroma_manager import ChromaManager
from src.database.sqlite_manager import SQLiteManager
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService
from src.services.search_service import ConfidenceScorer, HybridSearchService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductMatcherAgent(BaseAgent):
    """
    Agent for matching products in the database.

    Uses hybrid search combining:
    - Exact GTIN matching
    - SQL filtering
    - Vector similarity search
    - Reciprocal Rank Fusion
    - LLM-assisted ranking

    Output:
        {
            "database_matches": [...],
            "match_confidence": 0.95,
            "search_method": "hybrid"
        }
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        sqlite_manager: Optional[SQLiteManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        super().__init__(name="ProductMatcher", llm_service=llm_service)

        self.sqlite = sqlite_manager or SQLiteManager()
        self.chroma = chroma_manager or ChromaManager()
        self.embeddings = embedding_service or EmbeddingService()
        self.scorer = ConfidenceScorer()

        # Initialize search service
        self.search_service = HybridSearchService(
            sqlite_manager=self.sqlite,
            chroma_manager=self.chroma,
            embedding_service=self.embeddings,
            llm_service=self.llm,
        )

        # Initialize database
        self.sqlite.initialize()

    @traceable(name="product_matching", run_type="chain")
    async def _execute_impl(self, state: dict) -> dict:
        """
        Search for matching products.

        Args:
            state: Must contain 'extracted_properties'
                   Optional 'image_embedding' for CLIP-based image search

        Returns:
            State with 'database_matches' and 'match_confidence'
        """
        self.validate_state(state, ["extracted_properties"])

        properties = state["extracted_properties"]
        image_embedding = state.get("image_embedding")
        limit = state.get("limit", 10)
        confidence_threshold = state.get("confidence_threshold", 0.5)

        logger.info(f"Searching for products matching: {properties.get('name', 'unknown')}")

        try:
            # Check if we have a CLIP image embedding for visual search
            if image_embedding is not None:
                logger.info("Using CLIP image embedding for visual search")
                result = await self._image_similarity_search(
                    image_embedding=image_embedding,
                    limit=limit,
                    confidence_threshold=confidence_threshold,
                )
            else:
                # Execute standard hybrid search for text queries
                result = await self.search_service.search(
                    query_properties=properties,
                    limit=limit,
                    confidence_threshold=confidence_threshold,
                )

            # Get best match confidence
            best_confidence = (
                result.matches[0]["match_confidence"]
                if result.matches
                else 0.0
            )

            return self.update_state(state, {
                "database_matches": result.matches,
                "match_confidence": best_confidence,
                "search_method": result.method,
                "sql_candidates_count": result.sql_candidates_count,
                "vector_candidates_count": result.vector_candidates_count,
                "search_cached": result.cached,
            })

        except Exception as e:
            logger.error(f"Product matching failed: {e}")
            return self.update_state(state, {
                "database_matches": [],
                "match_confidence": 0.0,
                "search_error": str(e),
            })

    @traceable(name="image_similarity_search")
    async def _image_similarity_search(
        self,
        image_embedding: List[float],
        limit: int = 10,
        confidence_threshold: float = 0.5,
    ):
        """
        Search for products using CLIP image embedding.

        This is much faster than text extraction + hybrid search for images.

        Args:
            image_embedding: 512-dim CLIP embedding of query image
            limit: Maximum results to return
            confidence_threshold: Minimum confidence for results

        Returns:
            SearchResult with matches from image similarity
        """
        from src.services.search_service import SearchResult

        # Query the images collection in ChromaDB
        vector_results = self.chroma.query_images(
            query_embedding=image_embedding,
            limit=limit * 2,  # Over-fetch for filtering
        )

        matches = []
        product_ids = vector_results.get("ids", [])
        distances = vector_results.get("distances", [])
        metadatas = vector_results.get("metadatas", [])

        if product_ids:
            # Get full product data from SQLite
            products = await self.sqlite.get_products_by_ids(product_ids)
            product_map = {p.id: p.to_dict() for p in products}

            for i, product_id in enumerate(product_ids):
                if product_id in product_map:
                    product = product_map[product_id]
                    distance = distances[i] if distances else 1.0

                    # Convert cosine distance to confidence (0 = identical, 2 = opposite)
                    confidence = max(0, 1 - distance / 2)

                    # Include ALL results - don't filter by threshold
                    # Threshold filtering happens in orchestrator if needed
                    product["match_confidence"] = confidence
                    product["match_source"] = "clip_image"

                    # Add image URL from metadata if available
                    if metadatas and i < len(metadatas):
                        meta = metadatas[i]
                        if "image_url" in meta:
                            product["image_url"] = meta["image_url"]

                    matches.append(product)

            # Sort by confidence
            matches.sort(key=lambda x: x["match_confidence"], reverse=True)
            matches = matches[:limit]

        logger.info(f"CLIP image search found {len(matches)} matches")

        return SearchResult(
            matches=matches,
            method="clip_image",
            sql_candidates_count=0,
            vector_candidates_count=len(product_ids),
            confidence=matches[0]["match_confidence"] if matches else 0.0,
            cached=False,
        )

    @traceable(name="exact_gtin_search")
    async def exact_gtin_search(self, gtin: str) -> Optional[dict]:
        """
        Search for exact GTIN match.

        Returns product dict if found, None otherwise.
        """
        product = await self.sqlite.get_product_by_gtin(gtin)
        if product:
            result = product.to_dict()
            result["match_confidence"] = 1.0
            result["match_source"] = "exact_gtin"
            return result
        return None

    @traceable(name="sql_filter_search")
    async def sql_filter_search(
        self,
        name_pattern: Optional[str] = None,
        merchant: Optional[str] = None,
        market: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Search using SQL filters.

        Returns list of matching products.
        """
        products = await self.sqlite.search_products(
            name_pattern=name_pattern,
            merchant=merchant,
            market=market,
            price_min=price_min,
            price_max=price_max,
            limit=limit,
        )
        return [p.to_dict() for p in products]

    @traceable(name="vector_similarity_search")
    async def vector_similarity_search(
        self,
        query_text: str,
        limit: int = 20,
        filter_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search using vector similarity.

        Returns list of matching products with distances.
        """
        # Generate query embedding
        embedding = await self.embeddings.embed_text_async(query_text)

        # Query ChromaDB
        results = self.chroma.query_names(
            query_embedding=embedding,
            limit=limit,
            filter_ids=filter_ids,
        )

        # Format results
        matches = []
        for i, product_id in enumerate(results.get("ids", [])):
            matches.append({
                "product_id": product_id,
                "name": results["documents"][i] if results["documents"] else "",
                "distance": results["distances"][i] if results["distances"] else None,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })

        return matches

    @traceable(name="hybrid_search")
    async def hybrid_search(
        self,
        properties: dict,
        limit: int = 10,
    ) -> dict:
        """
        Execute full hybrid search.

        Combines SQL and vector search with RRF fusion.
        """
        result = await self.search_service.search(
            query_properties=properties,
            limit=limit,
        )

        return {
            "matches": result.matches,
            "confidence": result.confidence,
            "method": result.method,
        }

    async def calculate_match_confidence(
        self,
        query: dict,
        candidate: dict,
        vector_distance: Optional[float] = None,
    ) -> float:
        """
        Calculate confidence score for a candidate match.
        """
        return self.scorer.calculate_confidence(
            query, candidate, vector_distance
        )

    @traceable(name="rank_results")
    async def rank_results(
        self,
        query_properties: dict,
        candidates: List[dict],
        top_k: int = 10,
    ) -> List[dict]:
        """
        Use LLM to rank search candidates.
        """
        return await self.llm.rank_search_results(
            query_properties=query_properties,
            candidates=candidates,
            top_k=top_k,
        )

    async def get_product_details(self, product_id: str) -> Optional[dict]:
        """
        Get full product details by ID.
        """
        product = await self.sqlite.get_product(product_id)
        return product.to_dict() if product else None

    async def get_search_stats(self) -> dict:
        """
        Get search service statistics.
        """
        return await self.search_service.get_stats()
