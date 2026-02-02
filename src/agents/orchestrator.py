"""
Orchestrator Agent using LangGraph for coordinating the product matching workflow.
"""
import logging
import time
from typing import Any, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.agents.live_searcher import LiveSearcherAgent
from src.agents.product_matcher import ProductMatcherAgent
from src.agents.property_extractor import PropertyExtractorAgent
from src.config.settings import get_settings
from src.services.metrics_service import get_metrics_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict, total=False):
    """
    State schema for the orchestrator workflow.
    """
    # Input
    user_input: str
    input_type: Literal["text", "url", "image"]
    limit: int
    confidence_threshold: float
    enable_live_search: bool
    fast_mode: bool  # Use fast heuristics instead of LLM for live search

    # Extraction phase
    extracted_properties: Optional[dict]
    extraction_confidence: float
    extraction_error: Optional[str]

    # Matching phase
    database_matches: List[dict]
    match_confidence: float
    search_method: Optional[str]
    sql_candidates_count: int
    vector_candidates_count: int

    # Live search phase
    live_search_triggered: bool
    live_search_results: List[dict]
    live_search_query: Optional[str]
    search_time_ms: int

    # Output
    final_results: List[dict]
    processing_time_ms: int
    error: Optional[str]
    trace_id: Optional[str]


class Orchestrator:
    """
    Central orchestrator using LangGraph supervisor pattern.

    Coordinates:
    - PropertyExtractorAgent: Extract product properties from input
    - ProductMatcherAgent: Search database for matches
    - LiveSearcherAgent: Search web when database confidence is low

    Workflow:
    1. Extract properties from user input
    2. Search database for matches
    3. If confidence < threshold, trigger live search
    4. Combine and synthesize results
    """

    def __init__(
        self,
        property_extractor: Optional[PropertyExtractorAgent] = None,
        product_matcher: Optional[ProductMatcherAgent] = None,
        live_searcher: Optional[LiveSearcherAgent] = None,
    ):
        self.extractor = property_extractor or PropertyExtractorAgent()
        self.matcher = product_matcher or ProductMatcherAgent()
        self.searcher = live_searcher or LiveSearcherAgent()

        self.settings = get_settings()
        self.default_confidence_threshold = self.settings.confidence_threshold
        self.default_limit = self.settings.default_search_limit

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        """
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("extraction", self._extraction_node)
        graph.add_node("database_matching", self._matching_node)
        graph.add_node("live_search", self._live_search_node)
        graph.add_node("result_synthesis", self._synthesis_node)

        # Add edges
        graph.add_edge("extraction", "database_matching")
        graph.add_conditional_edges(
            "database_matching",
            self._should_trigger_live_search,
            {
                "live_search": "live_search",
                "synthesis": "result_synthesis",
            },
        )
        graph.add_edge("live_search", "result_synthesis")
        graph.add_edge("result_synthesis", END)

        # Set entry point
        graph.set_entry_point("extraction")

        return graph.compile()

    async def _extraction_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node for property extraction.
        """
        logger.info("Executing extraction node")

        result = await self.extractor.execute({
            "user_input": state["user_input"],
            "input_type": state.get("input_type", "text"),
        })

        return {
            **state,
            "extracted_properties": result.get("extracted_properties"),
            "extraction_confidence": result.get("extraction_confidence", 0.0),
            "extraction_error": result.get("extraction_error"),
        }

    async def _matching_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node for database matching.
        """
        logger.info("Executing matching node")

        result = await self.matcher.execute({
            "extracted_properties": state.get("extracted_properties", {}),
            "limit": state.get("limit", self.default_limit),
            "confidence_threshold": state.get(
                "confidence_threshold", self.default_confidence_threshold
            ),
        })

        return {
            **state,
            "database_matches": result.get("database_matches", []),
            "match_confidence": result.get("match_confidence", 0.0),
            "search_method": result.get("search_method"),
            "sql_candidates_count": result.get("sql_candidates_count", 0),
            "vector_candidates_count": result.get("vector_candidates_count", 0),
        }

    def _should_trigger_live_search(
        self, state: OrchestratorState
    ) -> Literal["live_search", "synthesis"]:
        """
        Decide whether to trigger live search.

        Always triggers live search when enabled to supplement database results.
        Database results will still be included regardless of live search.
        """
        enable_live_search = state.get("enable_live_search", True)

        if not enable_live_search:
            logger.info("Live search disabled")
            return "synthesis"

        if not self.searcher.is_available():
            logger.info("Live search not available (no API key)")
            return "synthesis"

        # Always run live search to supplement database results
        db_matches = len(state.get("database_matches", []))
        logger.info(
            f"Triggering live search to supplement {db_matches} database matches"
        )
        return "live_search"

    async def _live_search_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node for live web search.
        """
        logger.info("Executing live search node")

        result = await self.searcher.execute({
            "extracted_properties": state.get("extracted_properties", {}),
            "fast_mode": state.get("fast_mode", True),  # Default to fast mode
        })

        return {
            **state,
            "live_search_triggered": True,
            "live_search_results": result.get("live_search_results", []),
            "live_search_query": result.get("live_search_query"),
            "search_time_ms": result.get("search_time_ms", 0),
        }

    async def _synthesis_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node for combining and synthesizing results.
        """
        logger.info("Executing synthesis node")

        database_matches = state.get("database_matches", [])
        live_results = state.get("live_search_results", [])
        confidence_threshold = state.get(
            "confidence_threshold", self.default_confidence_threshold
        )
        limit = state.get("limit", self.default_limit)

        # Combine and filter results by confidence threshold
        final_results = self._combine_results(
            database_matches,
            live_results,
            confidence_threshold=confidence_threshold,
            limit=limit,
        )

        return {
            **state,
            "final_results": final_results,
        }

    def _combine_results(
        self,
        database_matches: List[dict],
        live_results: List[dict],
        confidence_threshold: float = 0.5,
        limit: int = 10,
    ) -> List[dict]:
        """
        Combine database and live search results without strict filtering.

        Args:
            database_matches: Results from database search
            live_results: Results from live web search
            confidence_threshold: Not used for filtering (kept for API compatibility)
            limit: Maximum number of results to return

        Returns:
            Combined results sorted by confidence, deduplicated by URL
        """
        combined = []
        seen_urls = set()

        # Add ALL database matches (from local Klarna data)
        for match in database_matches:
            url = match.get("source_url")
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    combined.append(match)
            else:
                # No URL - add anyway (dedup by product_id not possible here)
                combined.append(match)

        # Add live search results (supplementary)
        for result in live_results:
            url = result.get("source_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

        # Sort by confidence (highest first) but don't filter
        combined.sort(
            key=lambda x: x.get("match_confidence", 0),
            reverse=True,
        )

        # Log combining stats
        logger.info(
            f"Combined results: {len(database_matches)} database + "
            f"{len(live_results)} live = {len(combined)} unique -> "
            f"returning top {min(len(combined), limit)}"
        )

        return combined[:limit]

    @traceable(name="orchestrator_search", run_type="chain")
    async def search(
        self,
        query: str,
        input_type: Literal["text", "url", "image"] = "text",
        limit: int = 10,
        confidence_threshold: Optional[float] = None,
        enable_live_search: bool = True,
        fast_mode: bool = True,
    ) -> dict:
        """
        Execute the full search workflow.

        Args:
            query: User query (text, URL, or base64 image)
            input_type: Type of input
            limit: Maximum results to return
            confidence_threshold: Minimum confidence for results
            enable_live_search: Whether to enable live search fallback
            fast_mode: Use fast heuristics for live search (default True)

        Returns:
            Dictionary with results and metadata
        """
        start_time = time.perf_counter()

        # Initialize state
        initial_state: OrchestratorState = {
            "user_input": query,
            "input_type": input_type,
            "limit": limit,
            "confidence_threshold": confidence_threshold or self.default_confidence_threshold,
            "enable_live_search": enable_live_search,
            "fast_mode": fast_mode,
            "database_matches": [],
            "live_search_triggered": False,
            "live_search_results": [],
            "final_results": [],
        }

        try:
            logger.info(f"Starting orchestrator search: input_type={input_type}, query_length={len(query)}")

            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Calculate total processing time
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                f"Orchestrator completed: "
                f"extracted_props={bool(final_state.get('extracted_properties'))}, "
                f"db_matches={len(final_state.get('database_matches', []))}, "
                f"live_results={len(final_state.get('live_search_results', []))}, "
                f"time_ms={processing_time_ms}"
            )

            if final_state.get("extraction_error"):
                logger.warning(f"Extraction error: {final_state.get('extraction_error')}")

            # Record metrics (results already limited in _combine_results)
            final_results = final_state.get("final_results", [])
            metrics_service = get_metrics_service()
            search_metrics = metrics_service.record_search(
                query=query[:100],  # Truncate for storage
                input_type=input_type,
                results=final_results,
                live_search_triggered=final_state.get("live_search_triggered", False),
                processing_time_ms=processing_time_ms,
                extraction_confidence=final_state.get("extraction_confidence", 0.0),
            )

            return {
                "results": final_results,
                "query_info": {
                    "extracted_properties": final_state.get("extracted_properties"),
                    "search_method": final_state.get("search_method"),
                    "live_search_triggered": final_state.get("live_search_triggered", False),
                    "live_search_query": final_state.get("live_search_query"),
                },
                "stats": {
                    "database_matches": len(final_state.get("database_matches", [])),
                    "live_search_results": len(final_state.get("live_search_results", [])),
                    "sql_candidates": final_state.get("sql_candidates_count", 0),
                    "vector_candidates": final_state.get("vector_candidates_count", 0),
                },
                "metrics": search_metrics.to_dict(),
                "processing_time_ms": processing_time_ms,
                "error": final_state.get("error"),
            }

        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            return {
                "results": [],
                "query_info": {
                    "extracted_properties": None,
                    "search_method": None,
                    "live_search_triggered": False,
                },
                "stats": {},
                "processing_time_ms": processing_time_ms,
                "error": str(e),
            }


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """
    Get or create the orchestrator singleton.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
