"""Single ECG inference endpoint — ``POST /api/v1/predict``.

Exposes a full inference pipeline:  ``read_ecg`` → ``denoise`` →
``score_quality`` → model inference → (optional) uncertainty estimation.

The endpoint accepts a single ECG file via multipart/form-data upload
and returns a JSON response containing the quality report, per-task
predictions, and uncertainty report.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from aortica.io import UnsupportedFormatError, read_ecg
from aortica.signal import denoise, score_quality


# ---------------------------------------------------------------------------
# Pydantic response / request models
# ---------------------------------------------------------------------------


class LeadQualityResponse(BaseModel):
    """Quality assessment for a single ECG lead."""

    lead_name: str = Field(..., description="Name of the lead")
    score: float = Field(..., description="Quality score 0-100")
    classification: str = Field(
        ..., description="Quality classification: good, marginal, or poor"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Detected artefact flags",
    )


class QualityReportResponse(BaseModel):
    """Overall signal quality report."""

    per_lead: List[LeadQualityResponse] = Field(
        ..., description="Per-lead quality assessments"
    )
    overall_score: float = Field(..., description="Overall quality score 0-100")
    overall_classification: str = Field(
        ..., description="Overall quality classification"
    )
    recommendation: str = Field(
        ..., description="Action recommendation: accept, review, or reject"
    )


class TaskPrediction(BaseModel):
    """Prediction results for a single task head."""

    task: str = Field(..., description="Task head name")
    class_names: List[str] = Field(..., description="Class/output label names")
    probabilities: List[float] = Field(
        ..., description="Predicted probabilities per class"
    )


class UncertaintyResponse(BaseModel):
    """Uncertainty information accompanying predictions."""

    prediction_sets: Dict[str, List[List[int]]] = Field(
        default_factory=dict,
        description="Conformal prediction sets per task (classification only)",
    )
    confidence_intervals: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=dict,
        description="Confidence intervals per task (risk only)",
    )
    ood_flag: bool = Field(
        default=False,
        description="Whether the input is flagged as out-of-distribution",
    )
    entropy_score: Optional[float] = Field(
        default=None,
        description="Mean prediction entropy across tasks",
    )


class PredictResponse(BaseModel):
    """Full response from the single ECG inference endpoint."""

    quality_report: QualityReportResponse = Field(
        ..., description="Signal quality assessment"
    )
    predictions: List[TaskPrediction] = Field(
        ..., description="Per-task prediction results"
    )
    uncertainty: Optional[UncertaintyResponse] = Field(
        default=None,
        description="Uncertainty estimation (when model supports it)",
    )


# ---------------------------------------------------------------------------
# Class-name lookups (lazy imports to avoid hard torch dep at module level)
# ---------------------------------------------------------------------------

_TASK_CLASS_NAMES: Dict[str, List[str]] = {}


def _get_task_class_names() -> Dict[str, List[str]]:
    """Lazily load class names from the model head modules."""
    if _TASK_CLASS_NAMES:
        return _TASK_CLASS_NAMES

    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.risk_head import RISK_OUTPUTS
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    _TASK_CLASS_NAMES["rhythm"] = list(RHYTHM_CLASSES)
    _TASK_CLASS_NAMES["structural"] = list(STRUCTURAL_CLASSES)
    _TASK_CLASS_NAMES["ischaemia"] = list(ISCHAEMIA_CLASSES)
    _TASK_CLASS_NAMES["risk"] = list(RISK_OUTPUTS)
    return _TASK_CLASS_NAMES


# ---------------------------------------------------------------------------
# Inference pipeline helper
# ---------------------------------------------------------------------------


def _quality_report_to_response(
    report: Any,
) -> QualityReportResponse:
    """Convert a :class:`QualityReport` to its Pydantic response model."""
    per_lead = [
        LeadQualityResponse(
            lead_name=lq.lead_name,
            score=lq.score,
            classification=lq.classification,
            flags=sorted(lq.flags),
        )
        for lq in report.per_lead
    ]
    return QualityReportResponse(
        per_lead=per_lead,
        overall_score=report.overall_score,
        overall_classification=report.overall_classification,
        recommendation=report.recommendation,
    )


def run_inference_pipeline(
    file_bytes: bytes,
    filename: str,
    *,
    format_override: Optional[str] = None,
    model: Any = None,
    conformal_predictor: Any = None,
    enabled_tasks: Optional[List[str]] = None,
) -> PredictResponse:
    """Execute the full ECG inference pipeline on uploaded file bytes.

    Steps:
      1. Write bytes to a temp file (preserving the original extension).
      2. ``read_ecg()`` — auto-detect / explicit format.
      3. ``denoise()`` — clean the signal.
      4. ``score_quality()`` — assess signal quality.
      5. Model inference (if a model is loaded).
      6. Optionally wrap with conformal prediction / uncertainty.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded ECG file.
    filename:
        Original filename (used for extension-based format detection).
    format_override:
        Explicit format string (``wfdb``, ``dicom``, etc.).
    model:
        A loaded ``AorticaModel`` instance, or ``None``.
    conformal_predictor:
        A fitted ``ConformalPredictor``, or ``None``.
    enabled_tasks:
        Task heads to run.  Defaults to model's ``enabled_tasks``.

    Returns
    -------
    PredictResponse
        JSON-serialisable response with quality, predictions, and
        uncertainty information.
    """
    # ── 1. Persist to temp file (auto-detect needs a real path) ──────
    suffix = Path(filename).suffix or ""
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── 2. Read ECG ──────────────────────────────────────────────
        read_kwargs: Dict[str, Any] = {}
        if format_override:
            read_kwargs["format"] = format_override

        ecg_record = read_ecg(tmp_path, **read_kwargs)

        # ── 3. Denoise ───────────────────────────────────────────────
        ecg_record = denoise(ecg_record)

        # ── 4. Quality scoring ───────────────────────────────────────
        quality = score_quality(ecg_record)
        quality_resp = _quality_report_to_response(quality)

        # ── 5. Model inference ───────────────────────────────────────
        predictions: List[TaskPrediction] = []
        uncertainty_resp: Optional[UncertaintyResponse] = None

        if model is not None:
            import torch

            class_names = _get_task_class_names()
            tasks_to_run = enabled_tasks or model.enabled_tasks

            # Prepare input tensor: [1, leads, samples]
            x = torch.from_numpy(
                ecg_record.signals.astype(np.float32)
            ).unsqueeze(0)

            if conformal_predictor is not None and conformal_predictor.is_fitted:
                # Run with conformal prediction
                preds_dict, u_report = conformal_predictor.predict(
                    x, tasks=tasks_to_run
                )

                # Build uncertainty response
                uncertainty_resp = UncertaintyResponse(
                    prediction_sets=u_report.prediction_sets,
                    ood_flag=bool(u_report.ood_flags[0].item())
                    if u_report.ood_flags is not None
                    else False,
                    entropy_score=float(u_report.entropy_scores[0].item())
                    if u_report.entropy_scores is not None
                    else None,
                )
                # Confidence intervals
                for task_name, (lower, upper) in u_report.confidence_intervals.items():
                    uncertainty_resp.confidence_intervals[task_name] = {
                        "lower": lower[0].tolist(),
                        "upper": upper[0].tolist(),
                    }
            else:
                # Run plain model inference
                model.eval()
                with torch.no_grad():
                    output = model(x, tasks=tasks_to_run)
                preds_dict = {
                    k: v for k, v in output.as_dict().items() if v is not None
                }

            # Convert to response
            for task_name, probs_tensor in preds_dict.items():
                probs_list = probs_tensor[0].tolist()
                names = class_names.get(task_name, [])
                predictions.append(
                    TaskPrediction(
                        task=task_name,
                        class_names=names,
                        probabilities=probs_list,
                    )
                )

        return PredictResponse(
            quality_report=quality_resp,
            predictions=predictions,
            uncertainty=uncertainty_resp,
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
