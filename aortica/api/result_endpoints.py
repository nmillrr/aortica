"""Result browser API endpoints.

Provides ``GET /api/v1/results`` (paginated listing with filters) and
``GET /api/v1/results/{result_id}`` (single result detail).  These
endpoints query the local ``ResultStore`` SQLite database that stores
all processed ECG analyses.

Also provides ``POST /api/v1/results/export/csv`` for bulk CSV export of
selected results.
"""

from __future__ import annotations

import csv
import io
import json
import time
from typing import Any, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, Query, Request
    from fastapi.responses import JSONResponse, Response

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ResultSummary(BaseModel):
    """Summary view of a stored ECG result for the list endpoint."""

    id: int = Field(..., description="Database row ID")
    ecg_hash: str = Field(..., description="ECG file hash")
    timestamp: float = Field(..., description="Unix timestamp when result was stored")
    synced: bool = Field(..., description="Whether the result has been synced")
    patient_id: Optional[str] = Field(default=None, description="Patient identifier from metadata")
    quality_score: Optional[float] = Field(default=None, description="Overall quality score")
    quality_class: Optional[str] = Field(
        default=None, description="Quality classification: good/marginal/poor"
    )
    top_finding: Optional[str] = Field(default=None, description="Highest-confidence finding")
    top_finding_prob: Optional[float] = Field(
        default=None, description="Probability of top finding"
    )
    urgency_tier: Optional[str] = Field(
        default=None, description="Urgency tier: critical/urgent/routine/normal"
    )


class ResultDetail(BaseModel):
    """Full detail view of a stored ECG result."""

    id: int
    ecg_hash: str
    timestamp: float
    synced: bool
    predictions: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultListResponse(BaseModel):
    """Paginated result list response."""

    results: List[ResultSummary] = Field(default_factory=list)
    total: int = Field(..., description="Total number of matching results")
    page: int = Field(..., description="Current page number (1-indexed)")
    per_page: int = Field(..., description="Results per page")
    total_pages: int = Field(..., description="Total number of pages")


class BulkExportRequest(BaseModel):
    """Request body for bulk CSV export."""

    result_ids: List[int] = Field(..., description="List of result IDs to export")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Urgency classification thresholds
_CRITICAL_CONDITIONS = frozenset({
    "stemi", "vt", "vf", "complete_heart_block", "torsades",
    "ventricular_tachycardia", "ventricular_fibrillation",
    "st_elevation", "brugada",
})

_URGENT_CONDITIONS = frozenset({
    "atrial_fibrillation", "atrial_flutter", "wpw",
    "lvh", "rvh", "mi", "ischaemia", "wellens",
    "long_qt", "short_qt", "de_winter",
})


def _classify_urgency(predictions: dict[str, Any]) -> str:
    """Derive urgency tier from predictions."""
    # Flatten all prediction values
    all_findings: dict[str, float] = {}
    for task_preds in predictions.values():
        if isinstance(task_preds, dict):
            for name, prob in task_preds.items():
                if isinstance(prob, (int, float)):
                    all_findings[name.lower().replace(" ", "_")] = float(prob)

    # Check for critical conditions with high probability
    for name, prob in all_findings.items():
        for critical in _CRITICAL_CONDITIONS:
            if critical in name and prob > 0.5:
                return "critical"

    # Check for urgent conditions
    for name, prob in all_findings.items():
        for urgent in _URGENT_CONDITIONS:
            if urgent in name and prob > 0.5:
                return "urgent"

    # Check for any moderate findings
    has_moderate = any(p > 0.3 for p in all_findings.values())
    if has_moderate:
        return "routine"

    return "normal"


def _extract_top_finding(predictions: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """Extract the highest-probability finding across all task heads."""
    best_name: Optional[str] = None
    best_prob: float = 0.0

    for task_name, task_preds in predictions.items():
        if isinstance(task_preds, dict):
            for name, prob in task_preds.items():
                if isinstance(prob, (int, float)) and float(prob) > best_prob:
                    best_prob = float(prob)
                    best_name = name

    if best_name is None:
        return None, None
    return best_name, round(best_prob, 4)


def _stored_to_summary(result: Any) -> ResultSummary:
    """Convert a StoredResult to a ResultSummary."""
    # Extract patient ID from metadata
    patient_id = result.metadata.get("patient_id") or result.metadata.get("patient_identifier")

    # Quality info
    quality_score = result.quality.get("overall")
    quality_class = result.quality.get("classification")

    # Top finding
    top_finding, top_finding_prob = _extract_top_finding(result.predictions)

    # Urgency tier
    urgency_tier = _classify_urgency(result.predictions)

    return ResultSummary(
        id=result.id,
        ecg_hash=result.ecg_hash,
        timestamp=result.timestamp,
        synced=result.synced,
        patient_id=patient_id,
        quality_score=quality_score,
        quality_class=quality_class,
        top_finding=top_finding,
        top_finding_prob=top_finding_prob,
        urgency_tier=urgency_tier,
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_result_browser_router() -> Any:
    """Create the result browser API router.

    Returns
    -------
    APIRouter
        Router with ``GET /api/v1/results`` and
        ``GET /api/v1/results/{result_id}`` endpoints.
    """
    router = APIRouter(prefix="/api/v1", tags=["results"])

    @router.get(
        "/results",
        response_model=ResultListResponse,
        summary="List stored ECG results with filters and pagination",
    )
    async def list_results(
        request: Request,  # type: ignore[arg-type]
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        per_page: int = Query(
            default=25,
            ge=1,
            le=100,
            description="Results per page (max 100)",
        ),
        date_from: Optional[float] = Query(
            default=None,
            description="Filter: minimum timestamp (Unix epoch)",
        ),
        date_to: Optional[float] = Query(
            default=None,
            description="Filter: maximum timestamp (Unix epoch)",
        ),
        finding: Optional[str] = Query(
            default=None,
            description="Filter: finding name (case-insensitive substring match)",
        ),
        quality: Optional[str] = Query(
            default=None,
            description="Filter: quality classification (good/marginal/poor)",
        ),
        urgency: Optional[str] = Query(
            default=None,
            description="Filter: urgency tier (critical/urgent/routine/normal)",
        ),
        search: Optional[str] = Query(
            default=None,
            description="Free-text search on patient metadata",
        ),
        sort_by: Optional[str] = Query(
            default="timestamp",
            description="Column to sort by (timestamp/quality_score/urgency_tier)",
        ),
        sort_order: Optional[str] = Query(
            default="desc",
            description="Sort order (asc/desc)",
        ),
    ) -> Any:
        """List stored ECG results with filters, search, sorting, and pagination.

        All query parameters are optional.  Results are returned newest-first
        by default.
        """
        store = getattr(request.app.state, "result_store", None)
        if store is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "Result store not available"},
            )

        # Fetch all results (the store does basic pagination but we need
        # to filter in Python for the rich query params)
        all_results = store.list_results(limit=10000, offset=0)

        # Convert to summaries
        summaries = [_stored_to_summary(r) for r in all_results]

        # Apply filters
        if date_from is not None:
            summaries = [s for s in summaries if s.timestamp >= date_from]
        if date_to is not None:
            summaries = [s for s in summaries if s.timestamp <= date_to]
        if quality is not None:
            q_lower = quality.lower()
            summaries = [s for s in summaries if s.quality_class == q_lower]
        if urgency is not None:
            u_lower = urgency.lower()
            summaries = [s for s in summaries if s.urgency_tier == u_lower]
        if finding is not None:
            f_lower = finding.lower()
            summaries = [
                s for s in summaries
                if s.top_finding and f_lower in s.top_finding.lower()
            ]
        if search is not None:
            s_lower = search.lower()
            filtered = []
            for s in summaries:
                # Search in patient_id and ecg_hash
                searchable = " ".join(
                    str(v) for v in [s.patient_id, s.ecg_hash] if v
                ).lower()
                if s_lower in searchable:
                    filtered.append(s)
            summaries = filtered

        # Sort
        sort_field = sort_by or "timestamp"
        reverse = (sort_order or "desc").lower() == "desc"

        def _sort_key(s: ResultSummary) -> Any:
            val = getattr(s, sort_field, None)
            if val is None:
                return (1, "")  # Nones last
            return (0, val)

        summaries.sort(key=_sort_key, reverse=reverse)

        # Paginate
        total = len(summaries)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        end = start + per_page
        page_results = summaries[start:end]

        return ResultListResponse(
            results=page_results,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )

    @router.get(
        "/results/{result_id}",
        response_model=ResultDetail,
        summary="Get full details of a stored ECG result",
    )
    async def get_result(
        request: Request,  # type: ignore[arg-type]
        result_id: int = 0,
    ) -> Any:
        """Return the full stored result including predictions, quality, and XAI data."""
        store = getattr(request.app.state, "result_store", None)
        if store is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "Result store not available"},
            )

        result = store.get_result_by_id(result_id)
        if result is None:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Result {result_id} not found"},
            )

        return ResultDetail(
            id=result.id,
            ecg_hash=result.ecg_hash,
            timestamp=result.timestamp,
            synced=result.synced,
            predictions=result.predictions,
            quality=result.quality,
            metadata=result.metadata,
        )

    @router.post(
        "/results/export/csv",
        summary="Export selected results as CSV",
    )
    async def export_results_csv(
        request: Request,  # type: ignore[arg-type]
    ) -> Any:
        """Export selected results as a downloadable CSV file.

        Accepts a JSON body with ``result_ids`` (list of int).
        """
        store = getattr(request.app.state, "result_store", None)
        if store is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "Result store not available"},
            )

        body = await request.json()
        result_ids = body.get("result_ids", [])

        if not result_ids:
            return JSONResponse(
                status_code=422,
                content={"detail": "No result IDs provided for export"},
            )

        # Fetch results
        results = []
        for rid in result_ids:
            r = store.get_result_by_id(rid)
            if r is not None:
                results.append(r)

        if not results:
            return JSONResponse(
                status_code=404,
                content={"detail": "No results found for the given IDs"},
            )

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "ecg_hash", "timestamp", "synced",
            "patient_id", "quality_score", "quality_class",
            "top_finding", "top_finding_prob", "urgency_tier",
            "predictions_json",
        ])

        for r in results:
            summary = _stored_to_summary(r)
            writer.writerow([
                summary.id,
                summary.ecg_hash,
                summary.timestamp,
                summary.synced,
                summary.patient_id or "",
                summary.quality_score or "",
                summary.quality_class or "",
                summary.top_finding or "",
                summary.top_finding_prob or "",
                summary.urgency_tier or "",
                json.dumps(r.predictions),
            ])

        csv_content = output.getvalue()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=ecg_results_export.csv",
            },
        )

    return router
