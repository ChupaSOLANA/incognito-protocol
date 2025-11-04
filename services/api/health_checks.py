#!/usr/bin/env python3
"""
Health check endpoints and system monitoring
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import os

from services.api.logging_config import get_logger

logger = get_logger("health")

# Track API startup time
API_START_TIME = time.time()


async def check_database_health() -> Dict[str, Any]:
    """
    Check database connectivity and performance

    Returns:
        dict with status, response_time_ms, and error (if any)
    """
    try:
        from services.database.config import test_connection_async

        start = time.time()
        await test_connection_async()
        response_time = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_rpc_health(rpc_url: str) -> Dict[str, Any]:
    """
    Check Solana RPC connectivity

    Args:
        rpc_url: Solana RPC endpoint URL

    Returns:
        dict with status, response_time_ms, and error (if any)
    """
    try:
        import httpx

        start = time.time()

        # Make a simple RPC call (getHealth)
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
            )
            response.raise_for_status()

        response_time = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "rpc_url": rpc_url
        }
    except Exception as e:
        logger.error(f"RPC health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "rpc_url": rpc_url
        }


def get_system_metrics() -> Dict[str, Any]:
    """
    Get system resource metrics

    Returns:
        dict with CPU, memory, and disk usage
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)

        return {
            "cpu": {
                "usage_percent": round(cpu_percent, 2)
            },
            "memory": {
                "usage_percent": round(memory_percent, 2),
                "used_mb": round(memory_used_mb, 2),
                "total_mb": round(memory_total_mb, 2)
            },
            "disk": {
                "usage_percent": round(disk_percent, 2),
                "used_gb": round(disk_used_gb, 2),
                "total_gb": round(disk_total_gb, 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"error": str(e)}


def get_uptime() -> Dict[str, Any]:
    """
    Get API uptime

    Returns:
        dict with uptime_seconds and uptime_formatted
    """
    uptime_seconds = time.time() - API_START_TIME
    uptime_minutes = uptime_seconds / 60
    uptime_hours = uptime_minutes / 60
    uptime_days = uptime_hours / 24

    if uptime_days >= 1:
        uptime_str = f"{int(uptime_days)}d {int(uptime_hours % 24)}h"
    elif uptime_hours >= 1:
        uptime_str = f"{int(uptime_hours)}h {int(uptime_minutes % 60)}m"
    else:
        uptime_str = f"{int(uptime_minutes)}m {int(uptime_seconds % 60)}s"

    return {
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_formatted": uptime_str
    }


async def comprehensive_health_check(
    database_enabled: bool,
    rpc_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive health check of all services

    Args:
        database_enabled: Whether database is enabled
        rpc_url: Solana RPC URL to check

    Returns:
        dict with overall status and component statuses
    """
    checks = {}

    # Check database
    if database_enabled:
        checks["database"] = await check_database_health()
    else:
        checks["database"] = {"status": "disabled"}

    # Check RPC
    if rpc_url:
        checks["rpc"] = await check_rpc_health(rpc_url)
    else:
        checks["rpc"] = {"status": "not_configured"}

    # Get system metrics
    checks["system"] = get_system_metrics()

    # Get uptime
    checks["uptime"] = get_uptime()

    # Determine overall status
    component_statuses = [
        checks["database"].get("status"),
        checks["rpc"].get("status"),
    ]

    # Overall healthy if all enabled components are healthy
    if all(s in ["healthy", "disabled", "not_configured"] for s in component_statuses):
        overall_status = "healthy"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": checks
    }


async def readiness_check(
    database_enabled: bool,
    rpc_url: Optional[str] = None
) -> bool:
    """
    Check if API is ready to serve requests

    Args:
        database_enabled: Whether database is enabled
        rpc_url: Solana RPC URL

    Returns:
        True if ready, False otherwise
    """
    try:
        # Check critical dependencies
        checks = []

        # Database (if enabled)
        if database_enabled:
            db_check = await check_database_health()
            checks.append(db_check["status"] == "healthy")

        # RPC
        if rpc_url:
            rpc_check = await check_rpc_health(rpc_url)
            checks.append(rpc_check["status"] == "healthy")

        # All critical checks must pass
        return all(checks) if checks else True

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False


async def liveness_check() -> bool:
    """
    Check if API is alive (basic health check)

    Returns:
        True if alive, False otherwise
    """
    try:
        # Simple check - just verify we can execute code
        return True
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return False
