"""
LangSmith tracing middleware for FastAPI.
"""
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracing with LangSmith integration.

    Adds:
    - Unique trace ID to each request
    - Request timing
    - LangSmith run tracking (when enabled)
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.langsmith_client = None

        # Initialize LangSmith client if enabled
        if self.settings.langsmith_tracing:
            try:
                from langsmith import Client
                langsmith_key = self.settings.langsmith_api_key.get_secret_value()
                if langsmith_key:
                    self.langsmith_client = Client(api_key=langsmith_key)
                    logger.info("LangSmith tracing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request with tracing.
        """
        # Generate trace ID
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        # Record start time
        start_time = time.perf_counter()

        # Log request
        logger.info(
            f"[{trace_id}] {request.method} {request.url.path} - Started"
        )

        # Create LangSmith run if enabled
        run_id = None
        if self.langsmith_client:
            try:
                run_id = self._create_langsmith_run(request, trace_id)
            except Exception as e:
                logger.warning(f"Failed to create LangSmith run: {e}")

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Add headers
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Process-Time-Ms"] = str(duration_ms)

            # Log response
            logger.info(
                f"[{trace_id}] {request.method} {request.url.path} - "
                f"{response.status_code} ({duration_ms}ms)"
            )

            # Update LangSmith run
            if self.langsmith_client and run_id:
                self._update_langsmith_run(
                    run_id,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Log error
            logger.error(
                f"[{trace_id}] {request.method} {request.url.path} - "
                f"Error: {e} ({duration_ms}ms)"
            )

            # Update LangSmith run with error
            if self.langsmith_client and run_id:
                self._update_langsmith_run(
                    run_id,
                    status_code=500,
                    duration_ms=duration_ms,
                    error=str(e),
                )

            raise

    def _create_langsmith_run(self, request: Request, trace_id: str) -> str:
        """
        Create a LangSmith run for the request.
        """
        if not self.langsmith_client:
            return None

        try:
            run = self.langsmith_client.create_run(
                name=f"api_request:{request.url.path}",
                run_type="chain",
                inputs={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                },
                project_name=self.settings.langsmith_project,
                run_id=trace_id,
            )
            return trace_id

        except Exception as e:
            logger.warning(f"Failed to create LangSmith run: {e}")
            return None

    def _update_langsmith_run(
        self,
        run_id: str,
        status_code: int,
        duration_ms: int,
        error: str = None,
    ) -> None:
        """
        Update a LangSmith run with results.
        """
        if not self.langsmith_client:
            return

        try:
            outputs = {
                "status_code": status_code,
                "duration_ms": duration_ms,
            }

            if error:
                self.langsmith_client.update_run(
                    run_id=run_id,
                    outputs=outputs,
                    error=error,
                    end_time=time.time(),
                )
            else:
                self.langsmith_client.update_run(
                    run_id=run_id,
                    outputs=outputs,
                    end_time=time.time(),
                )

        except Exception as e:
            logger.warning(f"Failed to update LangSmith run: {e}")


def get_trace_id(request: Request) -> str:
    """
    Get the trace ID from request state.

    For use in route handlers to include trace ID in responses.
    """
    return getattr(request.state, "trace_id", str(uuid.uuid4()))
