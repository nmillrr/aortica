"""FastAPI router for urgent-finding notifications (US-126).

Exposes ``GET /api/v1/notifications`` returning delivery history with
per-notification status (sent / failed).
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel

try:
    from fastapi import APIRouter, Query

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


class NotificationModel(BaseModel):
    """A single notification delivery record."""

    id: str
    patient_id: Optional[str] = None
    ecg_id: str
    finding: str
    confidence: float
    urgency_score: int
    channel: str
    status: str
    created_at: float
    error: Optional[str] = None


def create_notification_router(notifier: Optional[Any] = None) -> Any:
    """Create the notifications API router backed by *notifier*."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the notifications router. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

    @router.get("", response_model=List[NotificationModel])
    async def list_notifications(
        patient_id: Optional[str] = Query(default=None),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> List[NotificationModel]:
        """Return notification delivery history (newest first)."""
        if notifier is None:
            return []
        return [
            NotificationModel(**r.to_dict())
            for r in notifier.history(patient_id=patient_id, limit=limit)
        ]

    return router
