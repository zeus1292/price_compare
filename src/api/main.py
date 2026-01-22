"""
FastAPI application entry point.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware.tracing import TracingMiddleware
from src.api.routes import dataset, products, search
from src.config.settings import get_settings
from src.database.sqlite_manager import SQLiteManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global instances
sqlite_manager: Optional[SQLiteManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    global sqlite_manager

    logger.info("Starting Price Compare API...")

    # Initialize settings and directories
    settings = get_settings()
    settings.ensure_directories()

    # Initialize database
    sqlite_manager = SQLiteManager()
    sqlite_manager.initialize()

    logger.info("Database initialized")

    yield

    # Cleanup
    logger.info("Shutting down Price Compare API...")
    if sqlite_manager:
        sqlite_manager.close()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title="Price Compare API",
        description="Multi-agent product matching tool for price comparison",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add tracing middleware
    app.add_middleware(TracingMiddleware)

    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Include routers
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(products.router, prefix="/api/v1", tags=["Products"])
    app.include_router(dataset.router, prefix="/api/v1", tags=["Dataset"])

    # Health check endpoints
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check."""
        return {"status": "healthy"}

    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with component status."""
        from src.database.chroma_manager import ChromaManager

        components = {}

        # Check SQLite
        try:
            if sqlite_manager:
                stats = await sqlite_manager.get_stats()
                components["sqlite"] = {
                    "status": "healthy",
                    "product_count": stats.get("total_products", 0),
                }
            else:
                components["sqlite"] = {"status": "not_initialized"}
        except Exception as e:
            components["sqlite"] = {"status": "unhealthy", "error": str(e)}

        # Check ChromaDB
        try:
            chroma = ChromaManager()
            stats = chroma.get_stats()
            components["chromadb"] = {
                "status": "healthy",
                "collections": {
                    "product_names": stats.get("names_count", 0),
                    "product_descriptions": stats.get("descriptions_count", 0),
                },
            }
        except Exception as e:
            components["chromadb"] = {"status": "unhealthy", "error": str(e)}

        # Check API keys
        try:
            settings = get_settings()
            components["api_keys"] = {
                "openai": "configured" if settings.openai_api_key.get_secret_value() else "missing",
                "anthropic": "configured" if settings.anthropic_api_key.get_secret_value() else "missing",
                "tavily": "configured" if settings.tavily_api_key.get_secret_value() else "missing",
            }
        except Exception as e:
            components["api_keys"] = {"status": "error", "error": str(e)}

        # Overall status
        all_healthy = all(
            c.get("status") == "healthy"
            for c in components.values()
            if isinstance(c, dict) and "status" in c
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
        }

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Static files and web UI
    static_dir = Path(__file__).parent.parent / "web" / "static"
    templates_dir = Path(__file__).parent.parent / "web" / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", tags=["Web UI"])
    async def serve_index():
        """Serve the web UI."""
        index_path = templates_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(
            status_code=404,
            content={"detail": "Web UI not found"},
        )

    return app


# Create app instance
app = create_app()


def get_sqlite_manager() -> SQLiteManager:
    """
    Get the SQLite manager instance.

    For use as a FastAPI dependency.
    """
    global sqlite_manager
    if sqlite_manager is None:
        sqlite_manager = SQLiteManager()
        sqlite_manager.initialize()
    return sqlite_manager


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
