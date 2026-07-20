"""FastAPI router for the integration orchestrator (US-125).

Exposes ``GET /api/v1/integration/status`` — orchestrator status, retry
queue depth, per-channel success/failure counts, last error per channel,
and error-rate health alerts.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from fastapi import APIRouter

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


def create_integration_router(orchestrator: Optional[Any] = None) -> Any:
    """Create the integration-status API router."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the integration router. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/integration", tags=["integration"])

    @router.get("/status", summary="Integration orchestrator status")
    async def integration_status() -> Any:
        """Return orchestrator status, queue depth, and per-channel health."""
        if orchestrator is None:
            return {
                "status": "not_configured",
                "processed": 0,
                "queue_depth": 0,
                "channels": {},
                "health_alerts": [],
            }
        return orchestrator.status()

    return router
