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


# -- Site validation registry models (US-114) ---------------------------


class SiteRegistrationRequest(BaseModel):
    """Request body for POST /api/v1/validation/sites."""

    site_id: str = Field(
        ..., description="Unique identifier for the validation site"
    )
    region: str = Field(
        ..., description="Free-text region (e.g. 'South Asia', 'Western Europe')"
    )
    benchmark_report: dict = Field(
        default_factory=dict,
        description="Benchmark report JSON (from BenchmarkReport.as_dict())",
    )
    dataset_size: Optional[int] = Field(
        default=None,
        description="Number of samples in the validation dataset",
    )


class SiteRegistrationResponse(BaseModel):
    """Response body for POST /api/v1/validation/sites."""

    site_id: str = Field(..., description="Registered site ID")
    region_class: str = Field(
        ..., description="Computed classification: 'western' or 'non-western'"
    )
    status: str = Field(default="registered", description="Registration status")


class SiteValidationRecordResponse(BaseModel):
    """A single site validation record in API responses."""

    site_id: str = Field(..., description="Site identifier")
    region: str = Field(..., description="Region string")
    region_class: str = Field(..., description="'western' or 'non-western'")
    dataset_size: int = Field(default=0, description="Validation dataset size")
    benchmark_summary: dict = Field(
        default_factory=dict, description="Benchmark report summary"
    )
    timestamp: str = Field(default="", description="ISO-8601 registration time")
    overall_pass: bool = Field(
        default=True, description="Overall pass/fail from benchmark"
    )


class SiteValidationListResponse(BaseModel):
    """Response body for GET /api/v1/validation/sites."""

    sites: List[SiteValidationRecordResponse] = Field(default_factory=list)
    total: int = 0


class ReleaseReadinessResponse(BaseModel):
    """Response body for GET /api/v1/validation/readiness."""

    ready: bool = Field(
        default=False, description="Whether v-stable release gate is satisfied"
    )
    total_validations: int = Field(default=0, description="Total registered sites")
    western_count: int = Field(default=0, description="Western site count")
    non_western_count: int = Field(default=0, description="Non-Western site count")
    min_non_western: int = Field(
        default=2, description="Required minimum non-Western validations"
    )
    non_western_sites: List[str] = Field(
        default_factory=list, description="Non-Western site IDs"
    )
    western_sites: List[str] = Field(
        default_factory=list, description="Western site IDs"
    )


def create_validation_router(
    collector: Optional[Any] = None,
    monitor: Optional[Any] = None,
    adverse_event_store: Optional[Any] = None,
    site_registry: Optional[Any] = None,
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
    site_registry:
        A :class:`~aortica.evaluation.site_validation.SiteValidationRegistry`
        instance.  If ``None``, the site validation endpoints return
        no-op responses.
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

    # -- Site validation registry endpoints (US-114) ----------------------

    @router.post(
        "/sites",
        response_model=SiteRegistrationResponse,
        summary="Register a new site validation",
    )
    async def register_site_validation(
        body: SiteRegistrationRequest,
    ) -> SiteRegistrationResponse:
        """Register a site validation with benchmark report.

        If a validation for the same site_id already exists, it is
        replaced (latest result wins).
        """
        if site_registry is None:
            return SiteRegistrationResponse(
                site_id=body.site_id,
                region_class="unknown",
                status="no_registry_configured",
            )

        from aortica.evaluation.site_validation import classify_region

        record = site_registry.register_validation(
            site_id=body.site_id,
            region=body.region,
            benchmark_report=body.benchmark_report,
            dataset_size=body.dataset_size,
        )

        return SiteRegistrationResponse(
            site_id=record.site_id,
            region_class=record.region_class,
            status="registered",
        )

    @router.get(
        "/sites",
        response_model=SiteValidationListResponse,
        summary="List all registered site validations",
    )
    async def list_site_validations() -> SiteValidationListResponse:
        """Return all registered site validations."""
        if site_registry is None:
            return SiteValidationListResponse(sites=[], total=0)

        from dataclasses import asdict

        validations = site_registry.get_validations()
        items = [
            SiteValidationRecordResponse(
                site_id=v.site_id,
                region=v.region,
                region_class=v.region_class,
                dataset_size=v.dataset_size,
                benchmark_summary=v.benchmark_summary,
                timestamp=v.timestamp,
                overall_pass=v.benchmark_summary.get("overall_pass", True),
            )
            for v in validations
        ]
        return SiteValidationListResponse(sites=items, total=len(items))

    @router.get(
        "/readiness",
        response_model=ReleaseReadinessResponse,
        summary="Check release readiness status",
    )
    async def release_readiness() -> ReleaseReadinessResponse:
        """Check whether the v-stable release gate is satisfied.

        Requires ≥2 non-Western site validations.
        """
        if site_registry is None:
            return ReleaseReadinessResponse(
                ready=False,
                total_validations=0,
                western_count=0,
                non_western_count=0,
                min_non_western=2,
                non_western_sites=[],
                western_sites=[],
            )

        readiness = site_registry.check_release_readiness()

        return ReleaseReadinessResponse(
            ready=readiness.ready,
            total_validations=readiness.total_validations,
            western_count=readiness.western_count,
            non_western_count=readiness.non_western_count,
            min_non_western=readiness.min_non_western,
            non_western_sites=readiness.non_western_sites,
            western_sites=readiness.western_sites,
        )

    return router
