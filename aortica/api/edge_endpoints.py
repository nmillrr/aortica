"""FastAPI router for edge site monitoring (US-061b).

Exposes ``GET /edge/status`` on the local edge server, returning the current
:class:`~aortica.edge.site_monitor.SiteStatus` for remote pilot monitoring:
daily inference count, error rate, sync status, storage utilisation, and the
last-sync timestamp.

Usage::

    from aortica.api.edge_endpoints import create_edge_router
    from aortica.edge.site_monitor import SiteMonitor

    router = create_edge_router(SiteMonitor("/var/lib/aortica/data"))
    app.include_router(router)
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class EdgeStatusResponse(BaseModel):
    """Response body for ``GET /edge/status``."""

    site_id: str = Field(..., description="Edge site identifier")
    timestamp: float = Field(..., description="Unix time the status was generated")
    daily_inference_count: int = Field(
        ..., description="Successful + failed inferences in the last 24 h"
    )
    daily_error_count: int = Field(
        ..., description="Failed inferences in the last 24 h"
    )
    total_inference_count: int = Field(
        ..., description="All inferences ever recorded"
    )
    error_rate: float = Field(..., description="Daily error rate in [0, 1]")
    sync_status: str = Field(
        ..., description="'synced', 'pending', or 'unknown'"
    )
    pending_sync_count: int = Field(
        ..., description="Results awaiting upload to the central server"
    )
    last_sync_timestamp: Optional[float] = Field(
        default=None, description="Unix time of the last successful sync"
    )
    storage_total_bytes: int = Field(..., description="Total data-volume bytes")
    storage_used_bytes: int = Field(..., description="Used data-volume bytes")
    storage_free_bytes: int = Field(..., description="Free data-volume bytes")
    storage_utilization_pct: float = Field(
        ..., description="Percentage of the data volume in use"
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_edge_router(site_monitor: Optional[Any] = None) -> Any:
    """Create the edge monitoring API router.

    Parameters
    ----------
    site_monitor:
        A :class:`~aortica.edge.site_monitor.SiteMonitor` instance. If
        ``None``, a temporary in-memory monitor is created (useful for tests
        and for servers that have not been configured with a data directory).
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the edge router. "
            "Install with: pip install aortica[api]"
        )

    if site_monitor is None:
        import tempfile

        from aortica.edge.site_monitor import SiteMonitor

        site_monitor = SiteMonitor(tempfile.mkdtemp(prefix="aortica-edge-"))

    router = APIRouter(prefix="/edge", tags=["edge"])

    @router.get(
        "/status",
        response_model=EdgeStatusResponse,
        summary="Edge site operational status",
    )
    async def edge_status() -> EdgeStatusResponse:
        """Return the current operational status of this edge site."""
        status = site_monitor.status()
        return EdgeStatusResponse(**status.to_dict())

    return router
