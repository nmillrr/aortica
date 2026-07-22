"""FastAPI router for FHIR subscriptions and webhook notifications (US-117).

Exposes CRUD over FHIR ``rest-hook`` Subscriptions plus per-subscription
delivery history:

* ``POST   /api/v1/subscriptions``            — create a subscription
* ``GET    /api/v1/subscriptions``            — list subscriptions
* ``GET    /api/v1/subscriptions/{id}``       — fetch one
* ``DELETE /api/v1/subscriptions/{id}``       — remove one
* ``GET    /api/v1/subscriptions/{id}/notifications`` — delivery history

Usage::

    from aortica.api.subscription_endpoints import create_subscription_router
    from aortica.integration.fhir_subscription import SubscriptionManager

    app.include_router(create_subscription_router(SubscriptionManager()))
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CriteriaModel(BaseModel):
    """Subscription criteria filter."""

    min_severity: str = Field(
        default="critical", description="info | warning | critical"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Optional condition allow-list (e.g. STEMI, VT, VF).",
    )
    urgency_threshold: int = Field(
        default=0, ge=0, le=100, description="Minimum urgency score (0-100)."
    )


class CreateSubscriptionRequest(BaseModel):
    """Body for ``POST /api/v1/subscriptions``."""

    webhook_url: str = Field(..., description="Destination rest-hook URL.")
    criteria: CriteriaModel = Field(default_factory=CriteriaModel)
    channel_type: str = Field(default="rest-hook")


class SubscriptionResponse(BaseModel):
    """A subscription resource."""

    id: str
    webhook_url: str
    criteria: CriteriaModel
    channel_type: str
    active: bool
    created_at: str


class NotificationResponse(BaseModel):
    """A single delivery record."""

    id: str
    subscription_id: str
    status: str
    ecg_id: str
    urgency_score: int
    matched_findings: List[dict]
    attempts: int
    created_at: str
    delivered_at: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_subscription_router(manager: Optional[Any] = None) -> Any:
    """Create the subscription API router backed by *manager*."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the subscription router. "
            "Install with: pip install aortica[api]"
        )

    from aortica.integration.fhir_subscription import (
        SubscriptionCriteria,
        SubscriptionManager,
    )

    if manager is None:
        manager = SubscriptionManager()

    router = APIRouter(prefix="/api/v1/subscriptions", tags=["subscriptions"])

    @router.post("", response_model=SubscriptionResponse, status_code=201)
    async def create_subscription(
        body: CreateSubscriptionRequest,
    ) -> SubscriptionResponse:
        """Register a new webhook subscription."""
        try:
            criteria = SubscriptionCriteria(
                min_severity=body.criteria.min_severity,
                conditions=list(body.criteria.conditions),
                urgency_threshold=body.criteria.urgency_threshold,
            )
            sub = manager.create_subscription(
                body.webhook_url, criteria, body.channel_type
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SubscriptionResponse(**sub.to_dict())

    @router.get("", response_model=List[SubscriptionResponse])
    async def list_subscriptions() -> List[SubscriptionResponse]:
        """List all registered subscriptions."""
        return [SubscriptionResponse(**s.to_dict()) for s in manager.list_subscriptions()]

    @router.get("/{sub_id}", response_model=SubscriptionResponse)
    async def get_subscription(sub_id: str) -> SubscriptionResponse:
        """Fetch a single subscription."""
        sub = manager.get_subscription(sub_id)
        if sub is None:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return SubscriptionResponse(**sub.to_dict())

    @router.delete("/{sub_id}", status_code=204)
    async def delete_subscription(sub_id: str) -> None:
        """Remove a subscription."""
        if not manager.delete_subscription(sub_id):
            raise HTTPException(status_code=404, detail="Subscription not found")

    @router.get(
        "/{sub_id}/notifications",
        response_model=List[NotificationResponse],
    )
    async def get_notifications(sub_id: str) -> List[NotificationResponse]:
        """Return delivery history for a subscription."""
        if manager.get_subscription(sub_id) is None:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return [
            NotificationResponse(**n.to_dict())
            for n in manager.get_notifications(sub_id)
        ]

    return router
