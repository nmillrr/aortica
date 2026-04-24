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


class XAIFeatureContribution(BaseModel):
    """A single named-feature contribution to a prediction."""

    feature_name: str = Field(..., description="ECG segment name")
    lead: str = Field(..., description="Lead name")
    delta_score: float = Field(..., description="Summed attribution score")


class XAISegmentAttribution(BaseModel):
    """Per-lead segment attribution scores."""

    lead: str = Field(..., description="Lead name")
    segments: Dict[str, float] = Field(
        ..., description="Segment name → attribution score"
    )


class XAIAttributionResponse(BaseModel):
    """XAI attribution response for a single task."""

    task: str = Field(..., description="Task head name")
    per_lead_attributions: Dict[str, List[float]] = Field(
        ..., description="Lead name → per-sample attribution values"
    )
    segment_attributions: List[XAISegmentAttribution] = Field(
        ..., description="Per-lead segment attribution scores"
    )
    top_features: List[XAIFeatureContribution] = Field(
        ..., description="Top-3 contributing features"
    )
    segment_boundaries: Dict[str, List[Dict[str, int]]] = Field(
        default_factory=dict,
        description="Per-lead segment boundaries (sample indices)",
    )


class SuggestionEntry(BaseModel):
    """A single clinical suggestion included in the predict response."""

    condition: str = Field(..., description="Condition class name")
    prompt: str = Field(..., description="Short clinical cue")
    urgency: str = Field(..., description="Urgency level")
    rationale: str = Field(..., description="Clinical justification")


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
    xai: Optional[List[XAIAttributionResponse]] = Field(
        default=None,
        description="XAI attribution data (when include_xai=true)",
    )
    suggestions: Optional[List[SuggestionEntry]] = Field(
        default=None,
        description="Clinical suggestions for active findings (when include_suggestions=true)",
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
    include_xai: bool = False,
    include_suggestions: bool = False,
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

        # ── 6. XAI attribution (optional) ────────────────────────────
        xai_results: Optional[List[XAIAttributionResponse]] = None

        if include_xai and model is not None:
            from aortica.xai import FeatureAttribution, delineate_segments, explain

            xai_results = []
            tasks_to_explain = enabled_tasks or model.enabled_tasks
            for task_name in tasks_to_explain:
                try:
                    attr: FeatureAttribution = explain(
                        model, ecg_record, task=task_name, n_steps=30
                    )

                    # Serialise per-lead attributions => list of floats
                    per_lead_attrs: Dict[str, List[float]] = {}
                    for lead_name, arr in attr.per_lead_attributions.items():
                        per_lead_attrs[lead_name] = arr.tolist()

                    # Segment attributions
                    seg_attrs = [
                        XAISegmentAttribution(lead=ln, segments=segs)
                        for ln, segs in attr.segment_attributions.items()
                    ]

                    # Top features
                    top_feats = [
                        XAIFeatureContribution(
                            feature_name=fc.feature_name,
                            lead=fc.lead,
                            delta_score=fc.delta_score,
                        )
                        for fc in attr.top_features
                    ]

                    # Segment boundaries per lead
                    seg_bounds: Dict[str, List[Dict[str, int]]] = {}
                    for lead_name in ecg_record.lead_names:
                        i = ecg_record.lead_names.index(lead_name)
                        lead_signal = ecg_record.signals[i].astype(np.float64)
                        bounds = delineate_segments(
                            lead_signal, ecg_record.sample_rate
                        )
                        seg_bounds[lead_name] = [
                            {
                                "p_start": b.p_start,
                                "p_end": b.p_end,
                                "qrs_start": b.qrs_start,
                                "qrs_end": b.qrs_end,
                                "t_start": b.t_start,
                                "t_end": b.t_end,
                            }
                            for b in bounds
                        ]

                    xai_results.append(
                        XAIAttributionResponse(
                            task=task_name,
                            per_lead_attributions=per_lead_attrs,
                            segment_attributions=seg_attrs,
                            top_features=top_feats,
                            segment_boundaries=seg_bounds,
                        )
                    )
                except Exception:
                    pass  # XAI is best-effort; skip task on failure

        # ── 7. Clinical suggestions (optional) ──────────────────────
        suggestion_entries: Optional[List[SuggestionEntry]] = None

        if include_suggestions and predictions:
            from aortica.api.clinical_suggestions import get_suggestion

            suggestion_entries = []
            for task_pred in predictions:
                for class_name, prob in zip(
                    task_pred.class_names, task_pred.probabilities
                ):
                    if prob >= 0.30:  # Only suggest for positive findings
                        suggestion = get_suggestion(class_name)
                        if suggestion is not None:
                            suggestion_entries.append(
                                SuggestionEntry(
                                    condition=class_name,
                                    prompt=suggestion.prompt,
                                    urgency=suggestion.urgency,
                                    rationale=suggestion.rationale,
                                )
                            )
            if not suggestion_entries:
                suggestion_entries = None

        return PredictResponse(
            quality_report=quality_resp,
            predictions=predictions,
            uncertainty=uncertainty_resp,
            xai=xai_results if xai_results else None,
            suggestions=suggestion_entries,
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
