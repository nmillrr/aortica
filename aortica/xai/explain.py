"""Integrated Gradient XAI with Named ECG Features.

Provides :func:`explain`, which computes integrated gradient attributions
for an ECG input with respect to a chosen task head, then maps those
attributions onto named ECG segments (P wave, PR interval, QRS complex,
ST segment, T wave, QT/QTc).

Usage::

    from aortica.xai import explain
    result = explain(model, ecg_record, task="rhythm")
    # result.top_features  -> list of (feature_name, lead, delta_score)
    # result.per_lead_attributions  -> {lead: ndarray}
    # result.segment_attributions  -> {lead: {segment: float}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn = types.ModuleType("nn")  # type: ignore[assignment]
    nn.Module = _DummyModule  # type: ignore[attr-defined]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for XAI explain. "
            "Install with: pip install aortica[torch]"
        )


# ── Named ECG Segments ─────────────────────────────────────────────
ECG_SEGMENTS: list[str] = [
    "P wave",
    "PR interval",
    "QRS complex",
    "ST segment",
    "T wave",
    "QT/QTc",
]


# ── Data Classes ───────────────────────────────────────────────────


@dataclass
class SegmentBoundaries:
    """Sample-index boundaries for one cardiac cycle's ECG segments.

    All indices are relative to the signal's sample array.
    A value of ``-1`` indicates the boundary could not be determined.
    """

    p_start: int = -1
    p_end: int = -1
    qrs_start: int = -1
    qrs_end: int = -1
    t_start: int = -1
    t_end: int = -1


@dataclass
class FeatureContribution:
    """A single named-feature contribution to a prediction.

    Attributes:
        feature_name: ECG segment name (e.g. ``"QRS complex"``).
        lead: Lead name (e.g. ``"V1"``).
        delta_score: Summed attribution over this segment in this lead.
    """

    feature_name: str
    lead: str
    delta_score: float


@dataclass
class FeatureAttribution:
    """Result of :func:`explain` for one ECG record and task head.

    Attributes:
        per_lead_attributions: Mapping from lead name to a 1-D
            attribution array of the same length as the input signal.
        segment_attributions: Mapping from lead name to a dict of
            ECG segment name → summed attribution score.
        top_features: Top-3 contributing features (across all leads),
            sorted by absolute delta-contribution score (descending).
        task: Task head that was explained.
        integrated_gradients_raw: Raw integrated gradient tensor with
            shape ``[leads, samples]``.
    """

    per_lead_attributions: dict[str, NDArray[np.float64]]
    segment_attributions: dict[str, dict[str, float]]
    top_features: list[FeatureContribution] = field(default_factory=list)
    task: str = "rhythm"
    integrated_gradients_raw: Optional[NDArray[np.float64]] = None


# ── R-Peak Detection (internal) ───────────────────────────────────


def _detect_r_peaks(
    signal_1d: NDArray[np.float64],
    sample_rate: float,
) -> NDArray[np.intp]:
    """Detect R-peaks using Pan-Tompkins for segment delineation.

    Uses the same Pan-Tompkins implementation from :mod:`aortica.signal`
    but operates on a single 1-D array directly.
    """
    from aortica.signal.qrs_detection import _detect_pan_tompkins

    return _detect_pan_tompkins(signal_1d, sample_rate)


# ── ECG Segment Delineation ───────────────────────────────────────


def delineate_segments(
    signal_1d: NDArray[np.float64],
    sample_rate: float,
    r_peaks: NDArray[np.intp] | None = None,
) -> list[SegmentBoundaries]:
    """Delineate ECG segments using R-peak + interval heuristics.

    For each detected R-peak (cardiac cycle) the following rule-based
    boundaries are estimated:

    * **QRS complex**: R-peak ± ~40 ms (total ~80 ms).
    * **P wave**: 200–80 ms before R-peak.
    * **PR interval**: 200–0 ms before QRS onset (overlaps P wave).
    * **ST segment**: QRS end to QRS end + 120 ms.
    * **T wave**: ST end to ST end + 160 ms.
    * **QT/QTc**: QRS onset to T-wave end.

    Parameters
    ----------
    signal_1d:
        Single-lead signal, shape ``(samples,)``.
    sample_rate:
        Sampling rate in Hz.
    r_peaks:
        Pre-computed R-peak indices.  If *None*, peaks are detected
        automatically via Pan-Tompkins.

    Returns
    -------
    list[SegmentBoundaries]
        One :class:`SegmentBoundaries` per detected R-peak, in
        chronological order.
    """
    n_samples = len(signal_1d)

    # Guard: signal too short for the filters in Pan-Tompkins
    # (scipy.signal.sosfiltfilt needs len > 3*ntaps, typically >=16 samples)
    if n_samples < 16:
        return []

    if r_peaks is None:
        r_peaks = _detect_r_peaks(signal_1d, sample_rate)

    if len(r_peaks) == 0:
        return []

    def _ms_to_samp(ms: float) -> int:
        return max(int(round(ms * sample_rate / 1000.0)), 1)

    boundaries: list[SegmentBoundaries] = []

    for r_idx in r_peaks:
        r: int = int(r_idx)

        # QRS complex: R-peak ± 40 ms
        qrs_half = _ms_to_samp(40)
        qrs_start = max(r - qrs_half, 0)
        qrs_end = min(r + qrs_half, n_samples - 1)

        # P wave: starts ~200 ms before R, ends ~80 ms before R
        p_start = max(r - _ms_to_samp(200), 0)
        p_end = max(r - _ms_to_samp(80), 0)

        # ST segment: from QRS end to QRS end + 120 ms
        st_end = min(qrs_end + _ms_to_samp(120), n_samples - 1)

        # T wave: from ST end to ST end + 160 ms
        t_start = st_end
        t_end = min(st_end + _ms_to_samp(160), n_samples - 1)

        boundaries.append(
            SegmentBoundaries(
                p_start=p_start,
                p_end=p_end,
                qrs_start=qrs_start,
                qrs_end=qrs_end,
                t_start=t_start,
                t_end=t_end,
            )
        )

    return boundaries


# ── Integrated Gradients ──────────────────────────────────────────


def _compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    task: str,
    n_steps: int = 50,
    target_class: int | None = None,
) -> torch.Tensor:
    """Compute integrated gradients for *input_tensor* w.r.t. *task*.

    Uses a straight-line path from a zero baseline to the actual input,
    accumulating gradients along *n_steps* interpolation points.

    Parameters
    ----------
    model:
        The :class:`AorticaModel` (or equivalent ``nn.Module``).
    input_tensor:
        ECG input tensor of shape ``[1, leads, samples]``.
    task:
        Task head name (``'rhythm'``, ``'structural'``, etc.).
    n_steps:
        Number of interpolation steps (higher = more accurate).
    target_class:
        If provided, compute gradients w.r.t. this output index only.
        Otherwise, sum across all outputs.

    Returns
    -------
    torch.Tensor
        Attribution tensor of shape ``[leads, samples]``.
    """
    _check_torch()

    model.eval()
    baseline = torch.zeros_like(input_tensor)

    # Accumulate scaled gradients
    scaled_grads = torch.zeros_like(input_tensor)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        output = model(interpolated, tasks=[task])

        # Select the right task output
        task_output = getattr(output, task)
        if task_output is None:
            raise ValueError(
                f"Task '{task}' produced no output. "
                f"Is it enabled in the model?"
            )

        # Target: either a specific class index or sum of all outputs
        if target_class is not None:
            target_val = task_output[0, target_class]
        else:
            target_val = task_output.sum()

        target_val.backward()

        if interpolated.grad is not None:
            scaled_grads += interpolated.grad
            interpolated.grad = None

    # Riemann sum approximation: average gradient × (input - baseline)
    avg_grads = scaled_grads / (n_steps + 1)
    attributions = (input_tensor - baseline) * avg_grads

    # Remove batch dimension → [leads, samples]
    attr_np: torch.Tensor = attributions.squeeze(0).detach()
    return attr_np


# ── Segment Attribution Mapping ───────────────────────────────────


def _map_attributions_to_segments(
    attributions_1d: NDArray[np.float64],
    boundaries_list: list[SegmentBoundaries],
) -> dict[str, float]:
    """Aggregate per-sample attributions into named ECG segments.

    For each segment, the absolute attribution values within that
    segment's boundaries are summed and averaged across all detected
    cardiac cycles.

    Returns a dict mapping segment name → average absolute attribution.
    """
    # Accumulate segment contributions across all beats
    segment_sums: dict[str, list[float]] = {name: [] for name in ECG_SEGMENTS}

    for bounds in boundaries_list:
        # P wave
        if bounds.p_start >= 0 and bounds.p_end > bounds.p_start:
            segment_sums["P wave"].append(
                float(np.sum(np.abs(attributions_1d[bounds.p_start : bounds.p_end])))
            )

        # PR interval: from P start to QRS start
        if bounds.p_start >= 0 and bounds.qrs_start > bounds.p_start:
            segment_sums["PR interval"].append(
                float(
                    np.sum(
                        np.abs(attributions_1d[bounds.p_start : bounds.qrs_start])
                    )
                )
            )

        # QRS complex
        if bounds.qrs_start >= 0 and bounds.qrs_end > bounds.qrs_start:
            segment_sums["QRS complex"].append(
                float(
                    np.sum(
                        np.abs(attributions_1d[bounds.qrs_start : bounds.qrs_end])
                    )
                )
            )

        # ST segment: from QRS end to T start
        if bounds.qrs_end >= 0 and bounds.t_start > bounds.qrs_end:
            segment_sums["ST segment"].append(
                float(
                    np.sum(
                        np.abs(attributions_1d[bounds.qrs_end : bounds.t_start])
                    )
                )
            )

        # T wave
        if bounds.t_start >= 0 and bounds.t_end > bounds.t_start:
            segment_sums["T wave"].append(
                float(
                    np.sum(np.abs(attributions_1d[bounds.t_start : bounds.t_end]))
                )
            )

        # QT/QTc: from QRS onset to T-wave end
        if bounds.qrs_start >= 0 and bounds.t_end > bounds.qrs_start:
            segment_sums["QT/QTc"].append(
                float(
                    np.sum(
                        np.abs(attributions_1d[bounds.qrs_start : bounds.t_end])
                    )
                )
            )

    # Average across beats
    result: dict[str, float] = {}
    for name in ECG_SEGMENTS:
        values = segment_sums[name]
        result[name] = float(np.mean(values)) if values else 0.0

    return result


# ── Public API ────────────────────────────────────────────────────


def explain(
    model: nn.Module,
    ecg_record: ECGRecord,
    task: str = "rhythm",
    n_steps: int = 50,
    target_class: int | None = None,
) -> FeatureAttribution:
    """Compute integrated gradient attributions mapped to ECG features.

    Parameters
    ----------
    model:
        An :class:`~aortica.models.AorticaModel` instance (or any
        ``nn.Module`` that accepts ``(x, tasks=)`` and returns a
        :class:`~aortica.models.MultiTaskOutput`).
    ecg_record:
        The ECG recording to explain.
    task:
        Which task head to compute attributions for.  Must be one of
        ``'rhythm'``, ``'structural'``, ``'ischaemia'``, ``'risk'``.
    n_steps:
        Number of interpolation steps for integrated gradients.
    target_class:
        If provided, compute attributions w.r.t. this specific output
        index only.  Otherwise attributions are computed w.r.t. the sum
        of all task outputs.

    Returns
    -------
    FeatureAttribution
        Attribution results including per-lead attributions, segment
        attributions, and top-3 contributing features.
    """
    _check_torch()

    # Convert ECGRecord to tensor: [1, leads, samples]
    signals = ecg_record.signals.astype(np.float64)
    input_tensor = torch.tensor(
        signals, dtype=torch.float32
    ).unsqueeze(0)  # [1, leads, samples]

    # Compute integrated gradients
    ig_tensor = _compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        task=task,
        n_steps=n_steps,
        target_class=target_class,
    )

    # Convert to numpy: [leads, samples]
    ig_numpy: NDArray[np.float64] = ig_tensor.cpu().numpy().astype(np.float64)

    # ── Per-lead attributions ────────────────────────────────────
    per_lead: dict[str, NDArray[np.float64]] = {}
    for i, lead_name in enumerate(ecg_record.lead_names):
        per_lead[lead_name] = ig_numpy[i]

    # ── Segment delineation & attribution mapping ────────────────
    segment_attrs: dict[str, dict[str, float]] = {}
    for i, lead_name in enumerate(ecg_record.lead_names):
        lead_signal = signals[i].astype(np.float64)
        boundaries = delineate_segments(
            lead_signal, ecg_record.sample_rate
        )
        segment_attrs[lead_name] = _map_attributions_to_segments(
            ig_numpy[i], boundaries
        )

    # ── Top-3 contributing features ──────────────────────────────
    all_contributions: list[FeatureContribution] = []
    for lead_name, seg_dict in segment_attrs.items():
        for seg_name, score in seg_dict.items():
            # Skip QT/QTc for ranking (it's a superset of other segments)
            if seg_name == "QT/QTc":
                continue
            all_contributions.append(
                FeatureContribution(
                    feature_name=seg_name,
                    lead=lead_name,
                    delta_score=score,
                )
            )

    # Sort by absolute delta_score descending, take top 3
    all_contributions.sort(key=lambda c: abs(c.delta_score), reverse=True)
    top_3 = all_contributions[:3]

    return FeatureAttribution(
        per_lead_attributions=per_lead,
        segment_attributions=segment_attrs,
        top_features=top_3,
        task=task,
        integrated_gradients_raw=ig_numpy,
    )
