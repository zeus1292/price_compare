"""
Hybrid search service combining SQL and vector search.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from src.config.settings import get_settings
from src.database.chroma_manager import ChromaManager
from src.database.sqlite_manager import SQLiteManager
from src.services.embedding_service import EmbeddingService
from src.services.encryption_service import EncryptionService
from src.services.llm_service import LLMService


@dataclass
class SearchResult:
    """Container for search results."""
    matches: List[dict] = field(default_factory=list)
    method: str = "hybrid"
    sql_candidates_count: int = 0
    vector_candidates_count: int = 0
    confidence: float = 0.0
    cached: bool = False
    live_search_triggered: bool = False


class ConfidenceScorer:
    """
    Calculate confidence scores for product matches.

    Uses weighted combination of multiple factors.
    """

    WEIGHTS = {
        "name_similarity": 0.35,
        "price_similarity": 0.20,
        "merchant_match": 0.15,
        "vector_similarity": 0.20,
        "attribute_match": 0.10,
    }

    def calculate_confidence(
        self,
        query: dict,
        candidate: dict,
        vector_distance: Optional[float] = None,
    ) -> float:
        """
        Calculate weighted confidence score.

        Args:
            query: Query properties
            candidate: Candidate product
            vector_distance: Optional vector distance score

        Returns:
            Confidence score between 0 and 1
        """
        scores = {}

        # Name similarity (fuzzy matching)
        query_name = query.get("name", "")
        candidate_name = candidate.get("name", "")
        if query_name and candidate_name:
            scores["name_similarity"] = fuzz.token_sort_ratio(
                query_name.lower(), candidate_name.lower()
            ) / 100
        else:
            scores["name_similarity"] = 0.0

        # Price similarity
        query_price = query.get("price")
        candidate_price = candidate.get("price")
        if query_price and candidate_price:
            scores["price_similarity"] = self._price_similarity(
                query_price, candidate_price, tolerance=0.15
            )
        else:
            scores["price_similarity"] = 0.5  # Neutral if missing

        # Merchant match
        query_merchant = query.get("merchant", "").lower()
        candidate_merchant = candidate.get("merchant", "").lower()
        if query_merchant and candidate_merchant:
            scores["merchant_match"] = 1.0 if query_merchant == candidate_merchant else 0.3
        else:
            scores["merchant_match"] = 0.5  # Neutral if missing

        # Vector similarity (converted from distance)
        if vector_distance is not None:
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            scores["vector_similarity"] = max(0, 1 - vector_distance / 2)
        else:
            scores["vector_similarity"] = 0.5

        # Attribute match (simplified)
        scores["attribute_match"] = self._attribute_match(query, candidate)

        # Weighted sum
        total = sum(
            scores.get(k, 0) * w
            for k, w in self.WEIGHTS.items()
        )

        return min(1.0, max(0.0, total))

    def _price_similarity(
        self,
        price1: float,
        price2: float,
        tolerance: float = 0.15,
    ) -> float:
        """Calculate price similarity within tolerance."""
        if price1 <= 0 or price2 <= 0:
            return 0.5

        diff_ratio = abs(price1 - price2) / max(price1, price2)

        if diff_ratio <= tolerance:
            return 1.0 - (diff_ratio / tolerance) * 0.5
        else:
            return max(0.0, 0.5 - (diff_ratio - tolerance))

    def _attribute_match(self, query: dict, candidate: dict) -> float:
        """Calculate attribute match score."""
        # Match on category
        query_category = (query.get("category") or "").lower()
        candidate_category = (candidate.get("category") or "").lower()

        if query_category and candidate_category:
            if query_category == candidate_category:
                return 1.0
            elif query_category in candidate_category or candidate_category in query_category:
                return 0.7

        return 0.5


# Bump this whenever scoring logic or result structure changes to bust stale cache
CACHE_VERSION = "v2"


class HybridSearchService:
    """
    Hybrid search combining SQL filtering with vector similarity.

    Strategy:
    1. Check for exact GTIN match (fastest path)
    2. SQL pre-filtering to narrow candidates
    3. Vector similarity search on filtered set
    4. Reciprocal Rank Fusion to combine results
    5. LLM-assisted confidence scoring
    """

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None,
        encryption_service: Optional[EncryptionService] = None,
    ):
        self.sqlite = sqlite_manager or SQLiteManager()
        self.chroma = chroma_manager or ChromaManager()
        self.embeddings = embedding_service or EmbeddingService()
        self.llm = llm_service or LLMService()
        self.encryption = encryption_service or EncryptionService()
        self.scorer = ConfidenceScorer()

        settings = get_settings()
        self.confidence_threshold = settings.confidence_threshold
        self.default_limit = settings.default_search_limit

        # Initialize SQLite
        self.sqlite.initialize()

    async def search(
        self,
        query_properties: dict,
        limit: int = 10,
        confidence_threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> SearchResult:
        """
        Execute hybrid search.

        Args:
            query_properties: Extracted properties from query
            limit: Maximum results to return
            confidence_threshold: Minimum confidence for results
            use_cache: Whether to use/update cache

        Returns:
            SearchResult with matches and metadata
        """
        threshold = confidence_threshold or self.confidence_threshold
        limit = limit or self.default_limit

        # Check cache
        if use_cache:
            cached_result = await self._check_cache(query_properties, limit)
            if cached_result:
                return cached_result

        # Stage 1: Exact GTIN match (fastest path)
        if gtin := query_properties.get("gtin"):
            exact_match = await self._exact_gtin_search(gtin)
            if exact_match:
                result = SearchResult(
                    matches=[exact_match],
                    method="exact_gtin",
                    confidence=1.0,
                )
                if use_cache:
                    await self._update_cache(query_properties, result, limit)
                return result

        # Stage 2: SQL pre-filtering
        sql_candidates = await self.sqlite.search_products(
            name_pattern=query_properties.get("name"),
            merchant=query_properties.get("merchant"),
            market=query_properties.get("market"),
            price_min=query_properties.get("price_min"),
            price_max=query_properties.get("price_max"),
            category=query_properties.get("category"),
            limit=limit * 10,  # Over-fetch for vector reranking
        )

        # Stage 3: Vector similarity search
        query_text = self._build_query_text(query_properties)
        query_embedding = await self.embeddings.embed_text_async(query_text)

        filter_ids = [str(p.id) for p in sql_candidates] if sql_candidates else None

        vector_results = self.chroma.query_names(
            query_embedding=query_embedding,
            limit=limit * 2,
            filter_ids=filter_ids,
        )

        # Stage 4: Reciprocal Rank Fusion
        fused_ids = self._reciprocal_rank_fusion(
            sql_results=sql_candidates,
            vector_results=vector_results,
        )

        # Stage 5: Score and rank final results
        if fused_ids:
            # Get full product data for top candidates
            top_ids = [item["id"] for item in fused_ids[:limit * 2]]
            products = await self.sqlite.get_products_by_ids(top_ids)
            product_map = {p.id: p.to_dict() for p in products}

            # Calculate confidence scores - include ALL results, don't filter by threshold
            # Threshold filtering happens in orchestrator's _combine_results if needed
            scored_results = []
            for item in fused_ids[:limit * 2]:
                product_id = item["id"]
                if product_id in product_map:
                    product = product_map[product_id]
                    confidence = self.scorer.calculate_confidence(
                        query_properties,
                        product,
                        vector_distance=item.get("distance"),
                    )
                    product["match_confidence"] = confidence
                    product["match_source"] = "database"
                    scored_results.append(product)

            # Sort by confidence
            scored_results.sort(key=lambda x: x["match_confidence"], reverse=True)
            matches = scored_results[:limit]
        else:
            matches = []

        result = SearchResult(
            matches=matches,
            method="hybrid",
            sql_candidates_count=len(sql_candidates),
            vector_candidates_count=len(vector_results.get("ids", [])),
            confidence=matches[0]["match_confidence"] if matches else 0.0,
        )

        if use_cache:
            await self._update_cache(query_properties, result, limit)

        return result

    async def _exact_gtin_search(self, gtin: str) -> Optional[dict]:
        """Search for exact GTIN match."""
        product = await self.sqlite.get_product_by_gtin(gtin)
        if product:
            result = product.to_dict()
            result["match_confidence"] = 1.0
            result["match_source"] = "database"
            return result
        return None

    def _build_query_text(self, properties: dict) -> str:
        """Build search text from properties."""
        parts = []
        if properties.get("name"):
            parts.append(properties["name"])
        if properties.get("brand"):
            parts.append(properties["brand"])
        if properties.get("category"):
            parts.append(properties["category"])
        return " ".join(parts) or "product"

    def _reciprocal_rank_fusion(
        self,
        sql_results: list,
        vector_results: dict,
        k: int = 60,
    ) -> List[dict]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        RRF(d) = sum(1 / (k + rank(d)))

        Args:
            sql_results: Products from SQL search
            vector_results: Results from vector search
            k: RRF parameter (default 60)

        Returns:
            Fused list of product IDs with scores
        """
        scores: Dict[str, dict] = {}

        # Score SQL results
        for rank, product in enumerate(sql_results, 1):
            product_id = str(product.id)
            if product_id not in scores:
                scores[product_id] = {"id": product_id, "rrf_score": 0, "distance": None}
            scores[product_id]["rrf_score"] += 1 / (k + rank)

        # Score vector results
        vector_ids = vector_results.get("ids", [])
        vector_distances = vector_results.get("distances", [])

        for rank, (product_id, distance) in enumerate(
            zip(vector_ids, vector_distances), 1
        ):
            if product_id not in scores:
                scores[product_id] = {"id": product_id, "rrf_score": 0, "distance": None}
            scores[product_id]["rrf_score"] += 1 / (k + rank)
            scores[product_id]["distance"] = distance

        # Sort by combined RRF score
        fused = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return fused

    async def _check_cache(self, query_properties: dict, limit: int) -> Optional[SearchResult]:
        """Check cache for existing results."""
        query_hash = self._hash_query(query_properties, limit)
        cache = await self.sqlite.get_cached_search(query_hash)

        if cache:
            try:
                results_data = self.encryption.decrypt_json(cache.results_json)
                return SearchResult(
                    matches=results_data.get("matches", []),
                    method=results_data.get("method", "cached"),
                    cached=True,
                    confidence=cache.confidence_score or 0.0,
                )
            except Exception:
                pass
        return None

    async def _update_cache(
        self,
        query_properties: dict,
        result: SearchResult,
        limit: int,
    ) -> None:
        """Update cache with search results."""
        query_hash = self._hash_query(query_properties, limit)

        results_data = {
            "matches": result.matches,
            "method": result.method,
        }
        encrypted_results = self.encryption.encrypt_json(results_data)

        await self.sqlite.set_cached_search(
            query_hash=query_hash,
            query_type="hybrid",
            results_json=encrypted_results,
            confidence_score=result.confidence,
        )

    def _hash_query(self, query_properties: dict, limit: int) -> str:
        """Generate cache key for query, limit, and cache version."""
        cache_input = {
            "props": query_properties,
            "limit": limit,
            "v": CACHE_VERSION,
        }
        serialised = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()

    async def get_stats(self) -> dict:
        """Get search service statistics."""
        sqlite_stats = await self.sqlite.get_stats()
        chroma_stats = self.chroma.get_stats()

        return {
            "sqlite": sqlite_stats,
            "chromadb": chroma_stats,
        }
