"""
Product Matching Agent for searching the database.
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

        Returns:
            State with 'database_matches' and 'match_confidence'
        """
        self.validate_state(state, ["extracted_properties"])

        properties = state["extracted_properties"]
        limit = state.get("limit", 10)
        confidence_threshold = state.get("confidence_threshold", 0.5)

        logger.info(f"Searching for products matching: {properties.get('name', 'unknown')}")

        try:
            # Execute hybrid search
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
