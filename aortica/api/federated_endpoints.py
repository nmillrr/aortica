"""FastAPI endpoints for the federated learning monitoring dashboard (US-113).

Provides three API routes consumed by the ``FLDashboard`` React page:

- ``GET /api/v1/federated/status`` — current campaign state
- ``GET /api/v1/federated/rounds`` — per-round aggregated metrics
- ``GET /api/v1/federated/sites``  — anonymised per-site participation

All endpoints require admin authentication when auth is enabled.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, Query, Request
    from fastapi.responses import JSONResponse

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

from aortica.federated.fl_metrics_store import (
    CampaignStatus,
    ConvergenceIndicators,
    FLMetricsStore,
    RoundRecord,
    SiteRecord,
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class CampaignStatusResponse(BaseModel):
    """Response for ``GET /api/v1/federated/status``."""

    campaign_name: str = Field(default="default", description="Campaign identifier")
    current_round: int = Field(default=0, description="Latest completed round")
    total_rounds: int = Field(default=0, description="Total configured rounds")
    strategy: str = Field(default="fedavg", description="Aggregation strategy")
    start_timestamp: float = Field(default=0.0, description="Campaign start (unix epoch)")
    elapsed_seconds: float = Field(default=0.0, description="Wall-clock elapsed time")
    status: str = Field(default="idle", description="Campaign lifecycle status")
    convergence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Convergence health indicators",
    )


class RoundMetricsResponse(BaseModel):
    """Single round entry in the rounds response."""

    round_number: int
    loss: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    num_clients: int = 0
    gradient_norm: Optional[float] = None
    timestamp: float = 0.0


class RoundsListResponse(BaseModel):
    """Response for ``GET /api/v1/federated/rounds``."""

    rounds: List[RoundMetricsResponse] = Field(default_factory=list)
    total: int = 0


class SiteParticipationResponse(BaseModel):
    """Single site entry in the sites response."""

    site_id: str
    status: str = "offline"
    samples_contributed: int = 0
    last_communication: float = 0.0
    local_training_time_ms: float = 0.0
    epsilon_spent: float = 0.0
    epsilon_budget_pct: float = Field(
        default=0.0,
        description="Percentage of total ε budget consumed",
    )


class SitesListResponse(BaseModel):
    """Response for ``GET /api/v1/federated/sites``."""

    sites: List[SiteParticipationResponse] = Field(default_factory=list)
    total: int = 0
    epsilon_budget: float = 1.0


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_federated_router(
    metrics_store: Optional[FLMetricsStore] = None,
) -> "APIRouter":
    """Create the ``/api/v1/federated/`` router.

    Args:
        metrics_store: Pre-built :class:`FLMetricsStore` instance.  If
            *None* a temporary in-memory store is created (useful for
            testing and when no FL campaign has been run yet).

    Returns:
        A FastAPI ``APIRouter`` with the three federated monitoring
        endpoints.
    """
    if metrics_store is None:
        metrics_store = FLMetricsStore()

    router = APIRouter(prefix="/api/v1/federated", tags=["federated"])

    # -- GET /status -------------------------------------------------------

    @router.get(
        "/status",
        response_model=CampaignStatusResponse,
        summary="Current FL campaign status",
    )
    async def federated_status() -> CampaignStatusResponse:
        """Return the current federated learning campaign state."""
        campaign = metrics_store.get_campaign_status()
        convergence = metrics_store.get_convergence_indicators()

        return CampaignStatusResponse(
            campaign_name=campaign.campaign_name,
            current_round=campaign.current_round,
            total_rounds=campaign.total_rounds,
            strategy=campaign.strategy,
            start_timestamp=campaign.start_timestamp,
            elapsed_seconds=campaign.elapsed_seconds,
            status=campaign.status,
            convergence=convergence.to_dict(),
        )

    # -- GET /rounds -------------------------------------------------------

    @router.get(
        "/rounds",
        response_model=RoundsListResponse,
        summary="Per-round aggregated metrics",
    )
    async def federated_rounds() -> RoundsListResponse:
        """Return per-round aggregated loss and per-task metrics."""
        rounds = metrics_store.get_rounds()
        items = [
            RoundMetricsResponse(
                round_number=r.round_number,
                loss=r.loss,
                metrics=r.metrics,
                num_clients=r.num_clients,
                gradient_norm=r.gradient_norm,
                timestamp=r.timestamp,
            )
            for r in rounds
        ]
        return RoundsListResponse(rounds=items, total=len(items))

    # -- GET /sites --------------------------------------------------------

    @router.get(
        "/sites",
        response_model=SitesListResponse,
        summary="Per-site participation stats",
    )
    async def federated_sites() -> SitesListResponse:
        """Return anonymised per-site participation data."""
        sites = metrics_store.get_sites()
        budget = metrics_store.get_epsilon_budget()

        items = [
            SiteParticipationResponse(
                site_id=s.site_id,
                status=s.status,
                samples_contributed=s.samples_contributed,
                last_communication=s.last_communication,
                local_training_time_ms=s.local_training_time_ms,
                epsilon_spent=s.epsilon_spent,
                epsilon_budget_pct=round(
                    (s.epsilon_spent / budget) * 100, 1
                ) if budget > 0 else 0.0,
            )
            for s in sites
        ]
        return SitesListResponse(
            sites=items,
            total=len(items),
            epsilon_budget=budget,
        )

    return router
