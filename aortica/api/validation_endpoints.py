"""FastAPI router for prospective validation endpoints.

Provides ``POST /api/v1/validation/submit`` for sites to submit
ECG + outcome pairs to the prospective data collector.

Usage::

    from aortica.api.validation_endpoints import create_validation_router

    router = create_validation_router(collector)
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


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_validation_router(
    collector: Optional[Any] = None,
    monitor: Optional[Any] = None,
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

    return router

