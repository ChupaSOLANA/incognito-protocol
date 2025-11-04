#!/usr/bin/env python3
"""
Logging middleware for FastAPI
Logs all requests and responses with timing information
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from services.api.logging_config import get_request_logger, api_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all HTTP requests and responses
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response"""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get request details
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        client_host = request.client.host if request.client else "unknown"

        # Create logger with request context
        logger = get_request_logger(request_id, path)

        # Log incoming request
        logger.info(
            f"{method} {path}",
            extra={
                "event": "request_started",
                "method": method,
                "path": path,
                "query_params": query_params,
                "client_host": client_host,
                "user_agent": request.headers.get("user-agent"),
            }
        )

        # Start timer
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                f"{method} {path} -> {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "event": "request_completed",
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                f"{method} {path} -> ERROR ({duration_ms:.2f}ms): {str(e)}",
                exc_info=True,
                extra={
                    "event": "request_failed",
                    "method": method,
                    "path": path,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

            # Re-raise to let FastAPI handle it
            raise


class SlowRequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs slow requests (performance monitoring)
    """

    def __init__(self, app: ASGIApp, threshold_ms: float = 1000.0):
        """
        Args:
            app: ASGI application
            threshold_ms: Threshold in milliseconds above which to log as slow
        """
        super().__init__(app)
        self.threshold_ms = threshold_ms

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log slow requests"""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log if slow
        if duration_ms > self.threshold_ms:
            request_id = getattr(request.state, "request_id", "unknown")
            logger = get_request_logger(request_id, request.url.path)

            logger.warning(
                f"SLOW REQUEST: {request.method} {request.url.path} took {duration_ms:.2f}ms",
                extra={
                    "event": "slow_request",
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "threshold_ms": self.threshold_ms,
                }
            )

        return response
