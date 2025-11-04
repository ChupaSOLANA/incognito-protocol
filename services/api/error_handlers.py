#!/usr/bin/env python3
"""
Global exception handlers for FastAPI
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from services.api.exceptions import IncognitoException, RateLimitExceededException
from services.api.logging_config import get_logger
from services.api.validators import ValidationError

logger = get_logger("api.error_handler")


async def incognito_exception_handler(request: Request, exc: IncognitoException) -> JSONResponse:
    """
    Handle all custom Incognito exceptions

    Args:
        request: FastAPI request
        exc: IncognitoException instance

    Returns:
        JSONResponse with error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    logger.error(
        f"IncognitoException: {exc.message}",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "endpoint": request.url.path,
            "details": exc.details,
        },
        exc_info=exc.status_code >= 500  # Only log traceback for 5xx errors
    )

    # Build response
    response = exc.to_dict()
    response["request_id"] = request_id

    # Add Retry-After header for rate limiting
    headers = {}
    if isinstance(exc, RateLimitExceededException):
        retry_after = exc.details.get("retry_after_seconds", 60)
        headers["Retry-After"] = str(retry_after)

    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=headers
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle validation errors from our validators module

    Args:
        request: FastAPI request
        exc: ValidationError instance

    Returns:
        JSONResponse with validation error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"ValidationError: {str(exc)}",
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
        }
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": "VALIDATION_ERROR",
            "message": str(exc),
            "request_id": request_id,
        }
    )


async def pydantic_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors (from request body validation)

    Args:
        request: FastAPI request
        exc: RequestValidationError instance

    Returns:
        JSONResponse with validation error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Extract field-level errors
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(
        f"Request validation failed: {len(errors)} error(s)",
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
            "validation_errors": errors,
        }
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "errors": errors,
            "request_id": request_id,
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle standard HTTP exceptions

    Args:
        request: FastAPI request
        exc: HTTPException instance

    Returns:
        JSONResponse with error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Log based on status code
    if exc.status_code >= 500:
        logger.error(
            f"HTTPException {exc.status_code}: {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "endpoint": request.url.path,
            }
        )
    else:
        logger.warning(
            f"HTTPException {exc.status_code}: {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "endpoint": request.url.path,
            }
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "request_id": request_id,
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions (catch-all)

    Args:
        request: FastAPI request
        exc: Exception instance

    Returns:
        JSONResponse with generic error message
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error with full traceback
    logger.critical(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
            "exception_type": type(exc).__name__,
        }
    )

    # Return generic error (don't leak internal details)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal error occurred. Please contact support.",
            "request_id": request_id,
        }
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with FastAPI app

    Args:
        app: FastAPI application instance
    """
    # Custom Incognito exceptions
    app.add_exception_handler(IncognitoException, incognito_exception_handler)

    # Validation errors
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(RequestValidationError, pydantic_validation_error_handler)

    # HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered successfully")
