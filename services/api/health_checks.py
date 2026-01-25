"""
Health check utilities for Incognito Protocol API

Provides comprehensive, readiness, and liveness health checks
for Kubernetes probes and general monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

api_logger = logging.getLogger("incognito.api")


async def liveness_check() -> bool:
    """
    Simple liveness check - just verify the process is running.
    This should always return True unless the app has crashed.
    """
    return True


async def readiness_check(
    database_enabled: bool = False,
    rpc_url: Optional[str] = None
) -> bool:
    """
    Check if the service is ready to accept traffic.
    
    Returns True if all critical dependencies are available.
    """
    try:
        # Check database if enabled
        if database_enabled:
            from services.database.config import test_connection_async
            try:
                await test_connection_async()
            except Exception:
                return False
        
        # Check RPC connection if configured
        if rpc_url:
            import httpx
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        rpc_url,
                        json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
                    )
                    if response.status_code != 200:
                        return False
            except Exception:
                # RPC not available is not a blocker for readiness
                pass
        
        return True
    except Exception as e:
        api_logger.error(f"Readiness check failed: {e}")
        return False


async def comprehensive_health_check(
    database_enabled: bool = False,
    rpc_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive health check for monitoring dashboards.
    
    Returns detailed status of all components.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    checks = {}
    overall_healthy = True
    
    # Check 1: Database (if enabled)
    if database_enabled:
        try:
            from services.database.config import test_connection_async
            await test_connection_async()
            checks["database"] = {"status": "healthy", "message": "Connection successful"}
        except Exception as e:
            checks["database"] = {"status": "unhealthy", "message": str(e)}
            overall_healthy = False
    else:
        checks["database"] = {"status": "disabled", "message": "Using JSON file storage"}
    
    # Check 2: RPC connection
    if rpc_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    rpc_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
                )
                if response.status_code == 200:
                    checks["solana_rpc"] = {"status": "healthy", "url": rpc_url}
                else:
                    checks["solana_rpc"] = {
                        "status": "degraded", 
                        "url": rpc_url,
                        "message": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            checks["solana_rpc"] = {
                "status": "unhealthy",
                "url": rpc_url,
                "message": str(e)
            }
            # RPC being down doesn't make the whole API unhealthy
            # Just degraded functionality
    else:
        checks["solana_rpc"] = {"status": "not_configured"}
    
    # Check 3: Merkle state files
    try:
        from pathlib import Path
        import os
        
        repo_root = Path(__file__).parent.parent.parent
        pool_state_path = repo_root / "pool_merkle_state.json"
        
        if pool_state_path.exists():
            checks["pool_merkle_state"] = {"status": "healthy", "exists": True}
        else:
            checks["pool_merkle_state"] = {"status": "warning", "exists": False, "message": "No deposits yet"}
    except Exception as e:
        checks["pool_merkle_state"] = {"status": "error", "message": str(e)}
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": timestamp,
        "checks": checks
    }
