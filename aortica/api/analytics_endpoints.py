"""Edge-sync receiver and cross-site analytics endpoints (US-128).

* ``POST /api/v1/sync/receive``   — accept a batch of synced edge results
* ``GET  /api/v1/analytics/sites`` — per-site metrics, anomalies, fleet view
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Header, HTTPException, Request

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


class SyncBatchRequest(BaseModel):
    """Body for POST /api/v1/sync/receive."""

    device_id: str = Field(..., description="Source edge device identifier")
    site_id: str = Field(..., description="Deployment site identifier")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Batch of synced result dicts"
    )
    expected_count: Optional[int] = Field(
        default=None, description="Device-reported total for reconciliation"
    )


def create_analytics_router(
    aggregator: Optional[Any] = None,
    *,
    require_device_key: bool = False,
    device_keys: Optional[Dict[str, str]] = None,
) -> Any:
    """Create the sync-receive + site-analytics router.

    Args:
        aggregator: A :class:`CentralAggregator`.  If ``None``, a temporary
            in-memory aggregator is created.
        require_device_key: When True, ``POST /sync/receive`` requires a
            valid ``X-Device-Key`` header matching *device_keys*.
        device_keys: ``{device_id: api_key}`` for per-device auth.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the analytics router. "
            "Install with: pip install aortica[api]"
        )

    if aggregator is None:
        from aortica.sync.central_aggregator import CentralAggregator

        aggregator = CentralAggregator()

    keys = device_keys or {}
    router = APIRouter(prefix="/api/v1", tags=["analytics"])

    @router.post("/sync/receive")
    async def sync_receive(
        body: SyncBatchRequest,
        x_device_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Any:
        """Accept a batch of synced results from an edge device."""
        if require_device_key:
            expected = keys.get(body.device_id)
            if expected is None or x_device_key != expected:
                raise HTTPException(status_code=401, detail="Invalid device key")

        summary = aggregator.receive_batch(
            body.device_id, body.site_id, body.results
        )
        if body.expected_count is not None:
            summary["reconciliation"] = aggregator.reconcile(
                body.device_id, body.expected_count
            )
        return summary

    @router.get("/analytics/sites")
    async def analytics_sites(request: Request) -> Any:  # type: ignore[valid-type]
        """Return per-site metrics, anomalies, and a fleet summary."""
        sites = aggregator.site_metrics()
        anomalies = aggregator.detect_anomalies()
        total = sum(s.total_ecgs for s in sites)
        return {
            "sites": [s.to_dict() for s in sites],
            "anomalies": [a.to_dict() for a in anomalies],
            "fleet": {
                "total_sites": len(sites),
                "total_ecgs": total,
                "total_devices": len(
                    {d for s in sites for d in s.device_ids}
                ),
            },
        }

    return router
