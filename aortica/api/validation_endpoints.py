"""FastAPI router for prospective validation endpoints.

Provides ``POST /api/v1/validation/submit`` for sites to submit
ECG + outcome pairs to the prospective data collector.

Also provides adverse event reporting endpoints:
- ``POST /api/v1/validation/adverse-event``
- ``GET /api/v1/validation/adverse-events``
- ``GET /api/v1/validation/adverse-events/summary``

Usage::

    from aortica.api.validation_endpoints import create_validation_router

    router = create_validation_router(collector)
    app.include_router(router)
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ValidationSubmitRequest(BaseModel):
    """Request body for POST /api/v1/validation/submit."""

    ecg_hash: str = Field(
        ..., description="Unique identifier for the ECG recording"
    )
    site_id: str = Field(
        ..., description="Identifier for the contributing site"
    )
    predictions: dict[str, Any] = Field(
        default_factory=dict,
        description="AI multi-task prediction results",
    )
    quality: dict[str, Any] = Field(
        default_factory=dict,
        description="Signal quality report",
    )
    ground_truth: dict[str, Any] = Field(
        default_factory=dict,
        description="Clinician-verified ground-truth labels",
    )
    clinician_id: str = Field(
        default="",
        description="ID of the clinician providing ground truth",
    )
    outcome: dict[str, Any] = Field(
        default_factory=dict,
        description="Follow-up outcome data",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (demographics, device info)",
    )


class ValidationSubmitResponse(BaseModel):
    """Response body for POST /api/v1/validation/submit."""

    record_id: int = Field(..., description="Database row ID of the record")
    status: str = Field(
        default="accepted", description="Submission status"
    )
    linked: bool = Field(
        default=False,
        description="Whether ground-truth was linked to the record",
    )


class MonitorStatusResponse(BaseModel):
    """Response body for GET /api/v1/validation/monitor/status."""

    task_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Current rolling metrics per task",
    )
    drift_alerts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Active drift alerts",
    )
    window_days: int = Field(
        default=30,
        description="Monitoring window in days",
    )
    total_predictions: int = Field(
        default=0,
        description="Total predictions recorded",
    )
    total_labeled: int = Field(
        default=0,
        description="Total predictions with ground-truth labels",
    )
    last_updated: float = Field(
        default=0.0,
        description="Timestamp of last recorded prediction",
    )
    has_drift: bool = Field(
        default=False,
        description="Whether any drift alerts are active",
    )


# -- Adverse event response models ----------------------------------------


class AdverseEventSubmitRequest(BaseModel):
    """Request body for POST /api/v1/validation/adverse-event."""

    reporter_id: str = Field(
        ..., description="Identifier for the reporting clinician"
    )
    ecg_reference: str = Field(
        ..., description="ECG recording reference or hash"
    )
    event_description: str = Field(
        ..., description="Free-text description of the adverse event"
    )
    severity: str = Field(
        ...,
        description="Event severity: minor, moderate, serious, or critical",
    )
    ai_finding: str = Field(
        ...,
        description="The AI finding that contributed to the adverse event",
    )
    patient_outcome: str = Field(
        default="",
        description="Description of the patient outcome",
    )


class AdverseEventSubmitResponse(BaseModel):
    """Response body for POST /api/v1/validation/adverse-event."""

    id: str = Field(..., description="Assigned event record ID")
    status: str = Field(default="recorded", description="Submission status")


class AdverseEventRecordResponse(BaseModel):
    """An adverse event record in API responses."""

    id: str = Field(..., description="Unique event record ID")
    reporter_id: str = Field(..., description="Reporting clinician ID")
    ecg_reference: str = Field(..., description="ECG recording reference")
    event_description: str = Field(..., description="Event description")
    severity: str = Field(..., description="Event severity level")
    ai_finding: str = Field(..., description="Contributing AI finding")
    patient_outcome: str = Field(default="", description="Patient outcome")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class SeverityCountResponse(BaseModel):
    """Count of events by severity level."""

    severity: str = Field(..., description="Severity level")
    count: int = Field(..., description="Number of events")


class FindingCountResponse(BaseModel):
    """Count of events by AI finding."""

    ai_finding: str = Field(..., description="AI finding name")
    count: int = Field(..., description="Number of events")


class AdverseEventSummaryResponse(BaseModel):
    """Aggregate adverse event statistics."""

    total_events: int = Field(..., description="Total reported events")
    by_severity: List[SeverityCountResponse] = Field(
        ..., description="Event counts by severity"
    )
    most_reported_findings: List[FindingCountResponse] = Field(
        ..., description="Top findings associated with events"
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_validation_router(
    collector: Optional[Any] = None,
    monitor: Optional[Any] = None,
    adverse_event_store: Optional[Any] = None,
) -> Any:
    """Create the validation API router.

    Parameters
    ----------
    collector:
        A :class:`~aortica.validation.ProspectiveCollector` instance.
        If ``None``, a temporary in-memory collector is created (useful
        for testing).
    monitor:
        A :class:`~aortica.validation.PerformanceMonitor` instance.
        If ``None``, the monitor status endpoint returns a no-op response.
    adverse_event_store:
        A :class:`~aortica.validation.AdverseEventStore` instance.
        If ``None``, the adverse event endpoints return no-op responses.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the validation router. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/validation", tags=["validation"])

    @router.post(
        "/submit",
        response_model=ValidationSubmitResponse,
        summary="Submit ECG + outcome pair",
    )
    async def submit_validation(
        body: ValidationSubmitRequest,
    ) -> ValidationSubmitResponse:
        """Submit an ECG with AI predictions and optional ground-truth.

        If ``predictions`` are provided, a new record is ingested.
        If ``ground_truth`` is provided alongside ``predictions``, the
        ground-truth is linked to the record immediately.

        For linking outcomes to an existing record (submitted later),
        use the record_id from a previous submission.
        """
        if collector is None:
            return ValidationSubmitResponse(
                record_id=0,
                status="no_collector_configured",
                linked=False,
            )

        # Ingest the ECG
        record_id = collector.ingest_ecg(
            ecg_hash=body.ecg_hash,
            site_id=body.site_id,
            predictions=body.predictions,
            quality=body.quality,
            metadata=body.metadata,
        )

        # Link ground truth if provided
        linked = False
        if body.ground_truth:
            linked = collector.add_outcome(
                record_id=record_id,
                ground_truth=body.ground_truth,
                clinician_id=body.clinician_id,
                outcome=body.outcome,
            )

        return ValidationSubmitResponse(
            record_id=record_id,
            status="accepted",
            linked=linked,
        )

    @router.get(
        "/monitor/status",
        response_model=MonitorStatusResponse,
        summary="Get performance monitoring status",
    )
    async def monitor_status() -> MonitorStatusResponse:
        """Return current performance monitoring metrics and drift flags.

        Returns rolling AUC, F1, and ECE metrics per task over the
        configured monitoring window.  Includes drift alerts if any
        metric has dropped below its threshold or deviated from baseline.
        """
        if monitor is None:
            return MonitorStatusResponse(
                task_metrics={},
                drift_alerts=[],
                has_drift=False,
            )

        from dataclasses import asdict

        status = monitor.get_status()
        status_dict = status.to_dict()

        return MonitorStatusResponse(
            task_metrics=status_dict.get("task_metrics", {}),
            drift_alerts=status_dict.get("drift_alerts", []),
            window_days=status.window_days,
            total_predictions=status.total_predictions,
            total_labeled=status.total_labeled,
            last_updated=status.last_updated,
            has_drift=status.has_drift(),
        )

    # -- Adverse event endpoints ------------------------------------------

    @router.post(
        "/adverse-event",
        response_model=AdverseEventSubmitResponse,
        summary="Report a voluntary adverse event",
    )
    async def report_adverse_event(
        body: AdverseEventSubmitRequest,
    ) -> Any:
        """Report an adverse event related to AI findings.

        Captures the reporter, ECG reference, event description,
        severity, contributing AI finding, and patient outcome.
        Events are stored in an append-only audit trail.

        Returns ``422`` for invalid severity values.
        """
        if adverse_event_store is None:
            return AdverseEventSubmitResponse(
                id="",
                status="no_store_configured",
            )

        from aortica.validation.adverse_events import AdverseEventSubmission

        try:
            submission = AdverseEventSubmission(
                reporter_id=body.reporter_id,
                ecg_reference=body.ecg_reference,
                event_description=body.event_description,
                severity=body.severity,
                ai_finding=body.ai_finding,
                patient_outcome=body.patient_outcome,
            )
            record = adverse_event_store.store_submission(submission)
        except ValueError as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        return AdverseEventSubmitResponse(
            id=record.id,
            status="recorded",
        )

    @router.get(
        "/adverse-events",
        response_model=List[AdverseEventRecordResponse],
        summary="List adverse event reports",
    )
    async def list_adverse_events(
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[AdverseEventRecordResponse]:
        """Return reported adverse events (admin-only).

        Events are returned in reverse chronological order.
        Optionally filter by severity level.
        """
        if adverse_event_store is None:
            return []

        events = adverse_event_store.list_events(
            severity=severity,
            limit=limit,
        )

        return [
            AdverseEventRecordResponse(
                id=e.id,
                reporter_id=e.reporter_id,
                ecg_reference=e.ecg_reference,
                event_description=e.event_description,
                severity=e.severity,
                ai_finding=e.ai_finding,
                patient_outcome=e.patient_outcome,
                timestamp=e.timestamp,
            )
            for e in events
        ]

    @router.get(
        "/adverse-events/summary",
        response_model=AdverseEventSummaryResponse,
        summary="Get adverse event summary statistics",
    )
    async def adverse_events_summary() -> AdverseEventSummaryResponse:
        """Return aggregate adverse event statistics.

        Includes total count, breakdown by severity, and the most-
        reported AI findings (top 10).
        """
        if adverse_event_store is None:
            return AdverseEventSummaryResponse(
                total_events=0,
                by_severity=[],
                most_reported_findings=[],
            )

        summary = adverse_event_store.get_summary()

        return AdverseEventSummaryResponse(
            total_events=summary.total_events,
            by_severity=[
                SeverityCountResponse(
                    severity=sc.severity,
                    count=sc.count,
                )
                for sc in summary.by_severity
            ],
            most_reported_findings=[
                FindingCountResponse(
                    ai_finding=fc.ai_finding,
                    count=fc.count,
                )
                for fc in summary.most_reported_findings
            ],
        )

    return router


