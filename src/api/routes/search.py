"""
Search API routes.
"""
import base64
import logging
import uuid
from typing import List, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agents.orchestrator import get_orchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class SearchFilters(BaseModel):
    """Search filters."""
    merchant: Optional[str] = None
    market: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., description="Search query (text, URL, or base64 image)")
    input_type: Literal["text", "url", "image"] = Field(
        default="text",
        description="Type of input",
    )
    filters: Optional[SearchFilters] = None
    limit: int = Field(default=10, ge=1, le=100)
    enable_live_search: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class ProductResult(BaseModel):
    """Product result."""
    product_id: Optional[str] = None
    name: str
    price: Optional[float] = None
    currency: Optional[str] = "USD"
    merchant: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    match_confidence: float = 0.0
    match_source: Optional[str] = None


class QueryInfo(BaseModel):
    """Query information."""
    extracted_properties: Optional[dict] = None
    search_method: Optional[str] = None
    live_search_triggered: bool = False
    live_search_query: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: List[dict]
    query_info: QueryInfo
    processing_time_ms: int
    trace_id: Optional[str] = None


@router.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Search for products.

    Accepts text queries, URLs, or base64-encoded images.
    Returns matching products from database and optionally live search.
    """
    trace_id = str(uuid.uuid4())
    logger.info(f"Search request [{trace_id}]: {request.input_type} query")

    try:
        orchestrator = get_orchestrator()

        result = await orchestrator.search(
            query=request.query,
            input_type=request.input_type,
            limit=request.limit,
            confidence_threshold=request.confidence_threshold,
            enable_live_search=request.enable_live_search,
        )

        if result.get("error"):
            logger.error(f"Search error [{trace_id}]: {result['error']}")

        return SearchResponse(
            results=result.get("results", []),
            query_info=QueryInfo(
                extracted_properties=result.get("query_info", {}).get("extracted_properties"),
                search_method=result.get("query_info", {}).get("search_method"),
                live_search_triggered=result.get("query_info", {}).get("live_search_triggered", False),
                live_search_query=result.get("query_info", {}).get("live_search_query"),
            ),
            processing_time_ms=result.get("processing_time_ms", 0),
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"Search failed [{trace_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(...),
    limit: int = Form(default=10),
    enable_live_search: bool = Form(default=True),
    confidence_threshold: float = Form(default=0.9),
):
    """
    Search for products using an image.

    Upload a product image to find matching products.
    """
    trace_id = str(uuid.uuid4())
    logger.info(f"Image search request [{trace_id}]: {image.filename}")

    try:
        # Read and encode image
        image_content = await image.read()
        image_b64 = base64.b64encode(image_content).decode()

        orchestrator = get_orchestrator()

        result = await orchestrator.search(
            query=image_b64,
            input_type="image",
            limit=limit,
            confidence_threshold=confidence_threshold,
            enable_live_search=enable_live_search,
        )

        return SearchResponse(
            results=result.get("results", []),
            query_info=QueryInfo(
                extracted_properties=result.get("query_info", {}).get("extracted_properties"),
                search_method=result.get("query_info", {}).get("search_method"),
                live_search_triggered=result.get("query_info", {}).get("live_search_triggered", False),
                live_search_query=result.get("query_info", {}).get("live_search_query"),
            ),
            processing_time_ms=result.get("processing_time_ms", 0),
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"Image search failed [{trace_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/stats")
async def get_search_stats():
    """
    Get search statistics including cache info.
    """
    try:
        from src.agents.live_searcher import get_tavily_cache
        from src.services.search_service import HybridSearchService

        search_service = HybridSearchService()
        stats = await search_service.get_stats()

        # Get Tavily cache stats
        tavily_cache = get_tavily_cache()
        cache_stats = tavily_cache.get_stats()

        return {
            "database": stats,
            "tavily_cache": cache_stats,
        }

    except Exception as e:
        logger.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/metrics")
async def get_search_metrics():
    """
    Get search quality metrics (precision/recall proxies).

    Returns aggregate statistics and recent search performance.
    """
    from src.services.metrics_service import get_metrics_service

    metrics_service = get_metrics_service()

    return {
        "aggregate": metrics_service.get_aggregate_report(),
        "recent_searches": metrics_service.get_recent_searches(limit=20),
        "performance_summary": metrics_service.get_performance_summary(),
    }


@router.get("/search/debug")
async def debug_llm():
    """
    Debug endpoint to test LLM connectivity and diagnose issues.
    """
    from src.config.settings import get_settings
    from src.services.llm_service import LLMService

    results = {
        "api_keys": {},
        "llm_test": {},
        "embedding_test": {},
        "errors": [],
    }

    # Check API keys
    try:
        settings = get_settings()
        openai_key = settings.openai_api_key.get_secret_value()
        anthropic_key = settings.anthropic_api_key.get_secret_value()
        tavily_key = settings.tavily_api_key.get_secret_value()

        results["api_keys"] = {
            "openai": "configured" if openai_key and len(openai_key) > 10 else "missing or invalid",
            "openai_prefix": openai_key[:7] + "..." if openai_key and len(openai_key) > 10 else "N/A",
            "anthropic": "configured" if anthropic_key and len(anthropic_key) > 10 else "missing",
            "tavily": "configured" if tavily_key and len(tavily_key) > 10 else "missing",
        }
    except Exception as e:
        results["errors"].append(f"Settings error: {str(e)}")

    # Test LLM call
    try:
        llm = LLMService()
        logger.info("Testing LLM with simple prompt...")

        response = await llm.complete(
            prompt="What is 2+2? Reply with just the number.",
            system_prompt="You are a helpful assistant. Be brief.",
            provider="openai",
            max_tokens=10,
        )

        results["llm_test"] = {
            "status": "success",
            "response": response.strip(),
            "model": "gpt-4o-mini",
        }
    except Exception as e:
        results["llm_test"] = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        results["errors"].append(f"LLM test failed: {str(e)}")
        logger.error(f"LLM test failed: {e}", exc_info=True)

    # Test embedding
    try:
        from src.services.embedding_service import EmbeddingService
        embeddings = EmbeddingService()

        embedding = await embeddings.embed_text_async("test product")

        results["embedding_test"] = {
            "status": "success",
            "dimensions": len(embedding),
        }
    except Exception as e:
        results["embedding_test"] = {
            "status": "failed",
            "error": str(e),
        }
        results["errors"].append(f"Embedding test failed: {str(e)}")

    # Test property extraction (full flow)
    try:
        from src.agents.property_extractor import PropertyExtractorAgent

        extractor = PropertyExtractorAgent()
        state = {
            "user_input": "iPhone 15 Pro Max",
            "input_type": "text",
        }

        result = await extractor.execute(state)

        results["extraction_test"] = {
            "status": "success" if not result.get("extraction_error") else "failed",
            "extracted_properties": result.get("extracted_properties"),
            "confidence": result.get("extraction_confidence"),
            "error": result.get("extraction_error"),
        }
    except Exception as e:
        results["extraction_test"] = {
            "status": "failed",
            "error": str(e),
        }
        results["errors"].append(f"Extraction test failed: {str(e)}")

    return results
