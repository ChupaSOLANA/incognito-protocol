#!/usr/bin/env python3
"""
Structured logging configuration for Incognito Protocol API
"""
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import traceback


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON-structured logs
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "transaction_signature"):
            log_data["transaction_signature"] = record.transaction_signature
        if hasattr(record, "error_code"):
            log_data["error_code"] = record.error_code

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Formatter for human-readable console output (development)
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for terminal"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:8s}{reset}"
        location = f"{record.module}:{record.funcName}:{record.lineno}"
        message = record.getMessage()

        log_line = f"{timestamp} | {level} | {location:40s} | {message}"

        # Add request ID if present
        if hasattr(record, "request_id"):
            log_line += f" [req_id={record.request_id}]"

        # Add exception traceback if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False
) -> None:
    """
    Configure structured logging for the application

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: If True, use JSON format (production). If False, use human-readable (development)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent logs from growing too large
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log startup message
    root_logger.info(
        f"Logging configured: level={log_level}, json_format={json_format}, log_file={log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds contextual information to all logs
    """

    def process(self, msg, kwargs):
        """Add extra context to log records"""
        # Ensure 'extra' dict exists
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        # Add context from adapter
        kwargs['extra'].update(self.extra)

        return msg, kwargs


def get_request_logger(request_id: str, endpoint: str = None) -> LoggerAdapter:
    """
    Get a logger adapter with request context

    Args:
        request_id: Unique request ID
        endpoint: API endpoint path

    Returns:
        LoggerAdapter with request context
    """
    logger = get_logger("api.request")
    extra = {"request_id": request_id}
    if endpoint:
        extra["endpoint"] = endpoint
    return LoggerAdapter(logger, extra)


# Predefined logger instances for common use cases
api_logger = get_logger("api")
db_logger = get_logger("database")
blockchain_logger = get_logger("blockchain")
security_logger = get_logger("security")
encryption_logger = get_logger("encryption")
