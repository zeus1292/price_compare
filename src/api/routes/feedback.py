"""
Feedback API routes for collecting user ratings on search results.
"""
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.database.sqlite_manager import SQLiteManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback."""
    query: str = Field(..., description="The search query that was executed")
    query_type: str = Field(..., description="Type of query: text, url, or image")
    rating: int = Field(..., ge=-1, le=1, description="Rating: 1 (thumbs up) or -1 (thumbs down)")
    trace_id: Optional[str] = Field(None, description="LangSmith trace ID for correlation")
    result_product_id: Optional[str] = Field(None, description="Product ID that was rated")
    result_name: Optional[str] = Field(None, description="Product name for reference")
    result_merchant: Optional[str] = Field(None, description="Merchant name")
    result_confidence: Optional[float] = Field(None, description="Match confidence when shown")
    comment: Optional[str] = Field(None, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    id: int
    message: str
    trace_id: Optional[str] = None


class FeedbackItem(BaseModel):
    """Single feedback item."""
    id: int
    trace_id: Optional[str]
    query: str
    query_type: str
    result_product_id: Optional[str]
    result_name: Optional[str]
    result_merchant: Optional[str]
    result_confidence: Optional[float]
    rating: int
    comment: Optional[str]
    created_at: str


class FeedbackStats(BaseModel):
    """Feedback statistics."""
    total_feedback: int
    positive: int
    negative: int
    satisfaction_rate: float
    by_query_type: List[dict]
    avg_confidence_positive: Optional[float]
    avg_confidence_negative: Optional[float]


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a search result.

    Use rating=1 for thumbs up, rating=-1 for thumbs down.
    """
    logger.info(f"Feedback received: rating={request.rating}, query_type={request.query_type}")

    try:
        db = SQLiteManager()
        db.initialize()

        feedback = await db.create_feedback(
            query=request.query,
            query_type=request.query_type,
            rating=request.rating,
            trace_id=request.trace_id,
            result_product_id=request.result_product_id,
            result_name=request.result_name,
            result_merchant=request.result_merchant,
            result_confidence=request.result_confidence,
            comment=request.comment,
        )

        # Optionally send to LangSmith if trace_id is provided
        if request.trace_id:
            await _send_to_langsmith(
                trace_id=request.trace_id,
                score=1.0 if request.rating == 1 else 0.0,
                comment=request.comment,
            )

        return FeedbackResponse(
            id=feedback.id,
            message="Feedback recorded successfully",
            trace_id=request.trace_id,
        )

    except Exception as e:
        logger.error(f"Failed to record feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback", response_model=List[FeedbackItem])
async def get_feedback(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    rating: Optional[int] = Query(default=None, ge=-1, le=1),
    query_type: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None, description="ISO format date"),
    end_date: Optional[str] = Query(default=None, description="ISO format date"),
):
    """
    Query feedback entries with optional filters.
    """
    try:
        db = SQLiteManager()
        db.initialize()

        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        feedback_list = await db.get_feedback(
            limit=limit,
            offset=offset,
            rating_filter=rating,
            query_type=query_type,
            start_date=start_dt,
            end_date=end_dt,
        )

        return [
            FeedbackItem(
                id=f.id,
                trace_id=f.trace_id,
                query=f.query,
                query_type=f.query_type,
                result_product_id=f.result_product_id,
                result_name=f.result_name,
                result_merchant=f.result_merchant,
                result_confidence=f.result_confidence,
                rating=f.rating,
                comment=f.comment,
                created_at=f.created_at.isoformat() if f.created_at else "",
            )
            for f in feedback_list
        ]

    except Exception as e:
        logger.error(f"Failed to query feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """
    Get aggregate feedback statistics.

    Useful for understanding search quality and user satisfaction.
    """
    try:
        db = SQLiteManager()
        db.initialize()

        stats = await db.get_feedback_stats()

        return FeedbackStats(**stats)

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/export")
async def export_feedback(
    format: str = Query(default="json", description="Export format: json or csv"),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
):
    """
    Export feedback data for analysis.

    Returns all feedback within the date range.
    """
    try:
        db = SQLiteManager()
        db.initialize()

        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        feedback_list = await db.get_feedback(
            limit=10000,  # Large limit for export
            start_date=start_dt,
            end_date=end_dt,
        )

        data = [f.to_dict() for f in feedback_list]

        if format == "csv":
            import csv
            import io
            from fastapi.responses import StreamingResponse

            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=feedback_export.csv"}
            )

        return {"feedback": data, "count": len(data)}

    except Exception as e:
        logger.error(f"Failed to export feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _send_to_langsmith(
    trace_id: str,
    score: float,
    comment: Optional[str] = None,
) -> None:
    """
    Send feedback to LangSmith for trace correlation.

    This allows viewing feedback directly in LangSmith traces.
    """
    try:
        from langsmith import Client

        client = Client()
        client.create_feedback(
            run_id=trace_id,
            key="user_rating",
            score=score,
            comment=comment,
        )
        logger.info(f"Feedback sent to LangSmith for trace {trace_id}")

    except ImportError:
        logger.debug("LangSmith not available, skipping feedback sync")
    except Exception as e:
        # Don't fail the request if LangSmith sync fails
        logger.warning(f"Failed to send feedback to LangSmith: {e}")
