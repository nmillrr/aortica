"""FastAPI router for the stateful worklist (US-119).

* ``GET   /api/v1/worklist``            — prioritized worklist + summary
* ``PATCH /api/v1/worklist/{ecg_id}``   — update review status / assignee

Backed by :class:`~aortica.integration.worklist_store.WorklistStore`.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException, Query

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


class WorklistEntryModel(BaseModel):
    """A single worklist entry."""

    ecg_id: str
    urgency_score: int
    urgency_tier: str
    top_finding: str
    recommended_action: str
    patient_id: Optional[str] = None
    acquired_at: Optional[str] = None
    review_status: str
    assignee: Optional[str] = None
    created_at: Optional[str] = None
    reviewed_at: Optional[str] = None
    active_findings: List[dict] = Field(default_factory=list)


class WorklistSummaryModel(BaseModel):
    """Summary metrics for the worklist bar."""

    total: int
    total_pending: int
    critical_count: int
    completed_count: int
    avg_time_to_review_seconds: Optional[float] = None


class WorklistResponse(BaseModel):
    """Response body for ``GET /api/v1/worklist``."""

    items: List[WorklistEntryModel]
    summary: WorklistSummaryModel


class WorklistPatchRequest(BaseModel):
    """Body for ``PATCH /api/v1/worklist/{ecg_id}``."""

    review_status: Optional[str] = Field(
        default=None, description="pending | in-progress | completed"
    )
    assignee: Optional[str] = Field(default=None, description="Assigned clinician")


def create_worklist_router(store: Optional[Any] = None) -> Any:
    """Create the worklist API router backed by *store*."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the worklist router. "
            "Install with: pip install aortica[api]"
        )

    from aortica.integration.worklist_store import WorklistStore

    if store is None:
        store = WorklistStore()

    router = APIRouter(prefix="/api/v1/worklist", tags=["worklist"])

    @router.get("", response_model=WorklistResponse)
    async def get_worklist(
        status: Optional[str] = Query(default=None),
        tier: Optional[str] = Query(default=None),
        finding: Optional[str] = Query(default=None),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
    ) -> WorklistResponse:
        """Return the prioritized worklist with optional filters."""
        entries = store.list_entries(
            status=status,
            tier=tier,
            finding=finding,
            date_from=date_from,
            date_to=date_to,
        )
        return WorklistResponse(
            items=[WorklistEntryModel(**e.to_dict()) for e in entries],
            summary=WorklistSummaryModel(**store.summary()),
        )

    @router.patch("/{ecg_id}", response_model=WorklistEntryModel)
    async def patch_worklist(
        ecg_id: str, body: WorklistPatchRequest
    ) -> WorklistEntryModel:
        """Update review status and/or assignee for an ECG."""
        try:
            entry = store.update_entry(
                ecg_id,
                review_status=body.review_status,
                assignee=body.assignee,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if entry is None:
            raise HTTPException(status_code=404, detail="ECG not in worklist")
        return WorklistEntryModel(**entry.to_dict())

    return router
