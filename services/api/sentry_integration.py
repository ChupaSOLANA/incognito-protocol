#!/usr/bin/env python3
"""
Sentry integration for error tracking and performance monitoring
"""
import os
from typing import Optional
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from services.api.logging_config import get_logger

logger = get_logger("sentry")


def init_sentry(
    dsn: Optional[str] = None,
    environment: str = "development",
    release: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    enable_logging_integration: bool = True
) -> bool:
    """
    Initialize Sentry error tracking

    Args:
        dsn: Sentry DSN (Data Source Name). If None, will try to get from SENTRY_DSN env var
        environment: Environment name (development, staging, production)
        release: Release version (e.g., "1.0.0" or git commit hash)
        traces_sample_rate: Percentage of transactions to trace (0.0 to 1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
        enable_logging_integration: Whether to capture logs

    Returns:
        True if initialized successfully, False otherwise
    """
    # Get DSN from parameter or environment
    dsn = dsn or os.getenv("SENTRY_DSN")

    if not dsn:
        logger.warning("Sentry DSN not configured. Error tracking disabled.")
        return False

    # Get environment from parameter or env var
    environment = environment or os.getenv("SENTRY_ENVIRONMENT", "development")

    # Get release from parameter or env var (use git commit hash if available)
    if not release:
        release = os.getenv("SENTRY_RELEASE")
        if not release:
            try:
                import subprocess
                release = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except:
                release = "unknown"

    try:
        # Configure integrations
        integrations = [
            FastApiIntegration(transaction_style="endpoint"),
            AsyncioIntegration(),
        ]

        # Add logging integration if enabled
        if enable_logging_integration:
            integrations.append(
                LoggingIntegration(
                    level=None,  # Capture all logs
                    event_level="ERROR"  # Only send ERROR+ to Sentry
                )
            )

        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            integrations=integrations,
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            send_default_pii=False,  # Don't send personally identifiable information
            attach_stacktrace=True,  # Attach stack traces to messages
            max_breadcrumbs=50,  # Keep last 50 breadcrumbs
            before_send=before_send_filter,  # Filter sensitive data
        )

        logger.info(
            f"Sentry initialized: environment={environment}, release={release}, "
            f"traces_sample_rate={traces_sample_rate}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False


def before_send_filter(event, hint):
    """
    Filter sensitive data before sending to Sentry

    Args:
        event: Sentry event dict
        hint: Additional context

    Returns:
        Modified event or None to drop the event
    """
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        sensitive_headers = ["authorization", "cookie", "x-api-key"]

        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[Filtered]"

    # Remove sensitive query parameters
    if "request" in event and "query_string" in event["request"]:
        query_string = event["request"]["query_string"]
        sensitive_params = ["secret", "private_key", "password", "token"]

        for param in sensitive_params:
            if param in query_string.lower():
                event["request"]["query_string"] = "[Filtered]"
                break

    # Remove sensitive data from exception messages
    if "exception" in event and "values" in event["exception"]:
        for exc in event["exception"]["values"]:
            if "value" in exc:
                # Redact hex strings that look like secrets (64 hex chars)
                import re
                exc["value"] = re.sub(r'\b[0-9a-fA-F]{64}\b', '[SECRET]', exc["value"])

    return event


def capture_exception(
    error: Exception,
    context: Optional[dict] = None,
    level: str = "error"
):
    """
    Capture an exception and send to Sentry

    Args:
        error: Exception to capture
        context: Additional context to include
        level: Error level (error, warning, info)
    """
    try:
        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add context
            if context:
                for key, value in context.items():
                    scope.set_context(key, value)

            # Capture exception
            sentry_sdk.capture_exception(error)

    except Exception as e:
        logger.error(f"Failed to capture exception in Sentry: {e}")


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[dict] = None
):
    """
    Capture a message and send to Sentry

    Args:
        message: Message to capture
        level: Message level (info, warning, error)
        context: Additional context to include
    """
    try:
        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add context
            if context:
                for key, value in context.items():
                    scope.set_context(key, value)

            # Capture message
            sentry_sdk.capture_message(message)

    except Exception as e:
        logger.error(f"Failed to capture message in Sentry: {e}")


def set_user_context(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    ip_address: Optional[str] = None
):
    """
    Set user context for Sentry events

    Args:
        user_id: User ID (e.g., public key)
        username: Username
        ip_address: Client IP address
    """
    try:
        sentry_sdk.set_user({
            "id": user_id,
            "username": username,
            "ip_address": ip_address
        })
    except Exception as e:
        logger.error(f"Failed to set user context in Sentry: {e}")


def set_transaction_context(name: str, op: str = "http.server"):
    """
    Start a new Sentry transaction

    Args:
        name: Transaction name (e.g., "POST /deposit")
        op: Operation type (e.g., "http.server", "blockchain.transaction")

    Returns:
        Transaction context manager
    """
    try:
        return sentry_sdk.start_transaction(name=name, op=op)
    except Exception as e:
        logger.error(f"Failed to start Sentry transaction: {e}")
        return None


def add_breadcrumb(
    message: str,
    category: str = "default",
    level: str = "info",
    data: Optional[dict] = None
):
    """
    Add a breadcrumb to Sentry

    Breadcrumbs are a trail of events that happened before an error

    Args:
        message: Breadcrumb message
        category: Category (e.g., "http", "database", "blockchain")
        level: Level (debug, info, warning, error)
        data: Additional data
    """
    try:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except Exception as e:
        logger.error(f"Failed to add breadcrumb in Sentry: {e}")


def flush_sentry(timeout: int = 2):
    """
    Flush Sentry events (wait for all events to be sent)

    Args:
        timeout: Maximum time to wait in seconds
    """
    try:
        sentry_sdk.flush(timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to flush Sentry events: {e}")
