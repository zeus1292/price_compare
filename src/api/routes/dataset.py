"""
Dataset management API routes.
"""
import asyncio
import logging
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.batch_processor import BatchProcessor, IngestionJob


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Store for active ingestion jobs
active_jobs: Dict[str, IngestionJob] = {}


class IngestRequest(BaseModel):
    """Dataset ingestion request."""
    source_path: str = Field(..., description="Path to dataset directory")
    batch_size: int = Field(default=100, ge=1, le=1000)
    overwrite: bool = Field(default=False)


class IngestResponse(BaseModel):
    """Dataset ingestion response."""
    job_id: str
    status: str
    message: str


class JobProgress(BaseModel):
    """Job progress information."""
    total: int
    processed: int
    failed: int
    percentage: float


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: Optional[JobProgress] = None
    error: Optional[str] = None


async def run_ingestion_job(job: IngestionJob):
    """Background task to run ingestion."""
    try:
        await job.start()
    except Exception as e:
        logger.error(f"Ingestion job {job.job_id} failed: {e}")


@router.post("/dataset/ingest", response_model=IngestResponse)
async def start_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start dataset ingestion.

    Processes Klarna dataset files from the specified path.
    Returns a job ID for tracking progress.
    """
    import os

    # Validate path exists
    if not os.path.exists(request.source_path):
        raise HTTPException(
            status_code=400,
            detail=f"Source path does not exist: {request.source_path}",
        )

    if not os.path.isdir(request.source_path):
        raise HTTPException(
            status_code=400,
            detail=f"Source path is not a directory: {request.source_path}",
        )

    # Create job
    job_id = str(uuid.uuid4())
    processor = BatchProcessor(batch_size=request.batch_size)
    job = IngestionJob(
        job_id=job_id,
        source_path=request.source_path,
        processor=processor,
    )

    # Store job
    active_jobs[job_id] = job

    # Start background task
    background_tasks.add_task(run_ingestion_job, job)

    logger.info(f"Started ingestion job {job_id} for {request.source_path}")

    return IngestResponse(
        job_id=job_id,
        status="started",
        message=f"Ingestion started for {request.source_path}",
    )


@router.get("/dataset/ingest/{job_id}", response_model=JobStatusResponse)
async def get_ingestion_status(job_id: str):
    """
    Get ingestion job status.
    """
    job = active_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = job.get_progress()

    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        progress=JobProgress(
            total=progress["progress"]["total"],
            processed=progress["progress"]["processed"],
            failed=progress["progress"]["failed"],
            percentage=progress["progress"]["percentage"],
        ) if progress["progress"]["total"] > 0 else None,
        error=job.stats.errors[-1] if job.stats.errors else None,
    )


@router.delete("/dataset/ingest/{job_id}")
async def cancel_ingestion(job_id: str):
    """
    Cancel an ingestion job.

    Note: This only removes the job from tracking.
    Currently running operations may complete.
    """
    job = active_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Mark as cancelled
    job.status = "cancelled"

    # Remove from active jobs
    del active_jobs[job_id]

    return {"message": f"Job {job_id} cancelled"}


@router.get("/dataset/jobs")
async def list_ingestion_jobs():
    """
    List all ingestion jobs.
    """
    jobs = []
    for job_id, job in active_jobs.items():
        progress = job.get_progress()
        jobs.append({
            "job_id": job_id,
            "status": job.status,
            "source_path": job.source_path,
            "progress": progress["progress"],
        })

    return {"jobs": jobs}


@router.get("/dataset/stats")
async def get_dataset_stats():
    """
    Get dataset statistics.
    """
    from src.database.chroma_manager import ChromaManager
    from src.database.sqlite_manager import SQLiteManager

    try:
        sqlite = SQLiteManager()
        sqlite.initialize()
        sqlite_stats = await sqlite.get_stats()

        chroma = ChromaManager()
        chroma_stats = chroma.get_stats()

        return {
            "sqlite": sqlite_stats,
            "chromadb": chroma_stats,
        }

    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/reembed")
async def reembed_products(
    background_tasks: BackgroundTasks,
    clear_existing: bool = False,
):
    """
    Re-generate embeddings for all products.

    Useful after model updates or to fix embedding issues.
    """
    from src.pipeline.embeddings import EmbeddingPipeline

    job_id = str(uuid.uuid4())

    async def run_reembedding():
        pipeline = EmbeddingPipeline()
        stats = await pipeline.reembed_all_products(clear_existing=clear_existing)
        logger.info(
            f"Re-embedding complete: {stats.embedded} embedded, "
            f"{stats.failed} failed"
        )

    background_tasks.add_task(run_reembedding)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Re-embedding started in background",
    }


@router.delete("/dataset/clear")
async def clear_dataset(confirm: bool = False):
    """
    Clear all data from the database.

    WARNING: This is destructive and cannot be undone.
    Requires confirm=true parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to confirm data deletion",
        )

    from src.database.chroma_manager import ChromaManager
    from src.database.models import Base
    from src.database.sqlite_manager import SQLiteManager

    try:
        # Clear ChromaDB
        chroma = ChromaManager()
        chroma.reset_all()
        logger.info("ChromaDB cleared")

        # Clear SQLite - drop and recreate tables
        sqlite = SQLiteManager()
        Base.metadata.drop_all(sqlite.sync_engine)
        sqlite.initialize()
        logger.info("SQLite cleared")

        return {"message": "All data cleared successfully"}

    except Exception as e:
        logger.error(f"Failed to clear dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
