"""Signal quality scoring module.

Provides per-lead and overall ECG signal quality assessment.  Detects
common artefacts — lead-off / flatline, excessive baseline wander,
motion artifact, and saturation / clipping — and produces a
:class:`QualityReport` with numeric scores and an actionable
recommendation.

Usage::

    from aortica.signal import score_quality
    report = score_quality(ecg_record)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

# ─────────────────────────────────────────────────────────────────
# Public data structures
# ─────────────────────────────────────────────────────────────────

QualityClass = Literal["good", "marginal", "poor"]
Recommendation = Literal["accept", "review", "reject"]


@dataclass
class LeadQuality:
    """Quality assessment for a single lead.

    Attributes:
        lead_name: Name of the lead.
        score: Numeric quality score in 0–100.
        classification: ``'good'``, ``'marginal'``, or ``'poor'``.
        flags: Set of detected artefact flags.
    """

    lead_name: str
    score: float
    classification: QualityClass
    flags: set[str] = field(default_factory=set)


@dataclass
class QualityReport:
    """Aggregate quality report for an :class:`ECGRecord`.

    Attributes:
        per_lead: List of :class:`LeadQuality` results, one per lead.
        overall_score: Mean of per-lead scores.
        overall_classification: Classification derived from the overall
            score using the configured thresholds.
        recommendation: ``'accept'`` (good), ``'review'`` (marginal),
            or ``'reject'`` (poor).
        scan_origin: ``True`` if the record was digitised from a
            PDF/image scan.  Scan-derived records are automatically
            capped to ``'marginal'`` quality (max 69/100).
    """

    per_lead: list[LeadQuality]
    overall_score: float
    overall_classification: QualityClass
    recommendation: Recommendation
    scan_origin: bool = False


# ─────────────────────────────────────────────────────────────────
# Default thresholds
# ─────────────────────────────────────────────────────────────────

_DEFAULT_GOOD_THRESHOLD = 70.0
_DEFAULT_MARGINAL_THRESHOLD = 40.0
_SCAN_QUALITY_CEILING = 69.0  # scan-derived records cannot exceed this

# Penalty weights for each artefact category (deducted from 100).
_PENALTY_FLATLINE = 40.0
_PENALTY_CLIPPING = 30.0
_PENALTY_BASELINE_WANDER = 20.0
_PENALTY_MOTION_ARTIFACT = 20.0

# Detection parameters
_FLATLINE_FRACTION_THRESHOLD = 0.10  # ≥10 % flat → flag
_FLATLINE_TOLERANCE = 1e-6  # absolute amplitude tolerance
_CLIPPING_FRACTION_THRESHOLD = 0.02  # ≥2 % clipped → flag
_CLIPPING_PERCENTILE = 99.5  # samples near min/max boundary
_BASELINE_WANDER_RATIO_THRESHOLD = 0.30  # low-freq energy ratio
_MOTION_ARTIFACT_RATIO_THRESHOLD = 0.25  # high-freq energy ratio


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def score_quality(
    ecg_record: ECGRecord,
    *,
    good_threshold: float = _DEFAULT_GOOD_THRESHOLD,
    marginal_threshold: float = _DEFAULT_MARGINAL_THRESHOLD,
) -> QualityReport:
    """Score signal quality for each lead and overall.

    Parameters
    ----------
    ecg_record:
        The ECG record to assess.
    good_threshold:
        Minimum score to be classified ``'good'``.  Default ``70``.
    marginal_threshold:
        Minimum score to be classified ``'marginal'``; below this is
        ``'poor'``.  Default ``40``.

    Returns
    -------
    QualityReport
        A quality report with per-lead scores, overall score, and a
        recommendation.
    """
    if good_threshold <= marginal_threshold:
        raise ValueError(
            f"good_threshold ({good_threshold}) must be greater than "
            f"marginal_threshold ({marginal_threshold})"
        )

    per_lead: list[LeadQuality] = []
    for i, lead_name in enumerate(ecg_record.lead_names):
        lead_sig: NDArray[np.float64] = ecg_record.signals[i].astype(np.float64)
        lq = _assess_lead(lead_sig, lead_name, ecg_record.sample_rate,
                          good_threshold, marginal_threshold)
        per_lead.append(lq)

    overall_score = float(np.mean([lq.score for lq in per_lead]))

    # Scan-origin quality ceiling: records digitised from PDF/image
    # scans cannot exceed 'marginal' quality (max 69/100).
    is_scan = False
    if ecg_record.patient_metadata and ecg_record.patient_metadata.get("scan_origin"):
        is_scan = True
        overall_score = min(overall_score, _SCAN_QUALITY_CEILING)
        # Also cap per-lead scores
        for lq in per_lead:
            if lq.score > _SCAN_QUALITY_CEILING:
                lq.score = _SCAN_QUALITY_CEILING
                lq.classification = _classify(
                    lq.score, good_threshold, marginal_threshold
                )

    overall_class = _classify(overall_score, good_threshold, marginal_threshold)
    recommendation = _recommend(overall_class)

    return QualityReport(
        per_lead=per_lead,
        overall_score=overall_score,
        overall_classification=overall_class,
        recommendation=recommendation,
        scan_origin=is_scan,
    )


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────


def _assess_lead(
    sig: NDArray[np.float64],
    lead_name: str,
    sample_rate: float,
    good_threshold: float,
    marginal_threshold: float,
) -> LeadQuality:
    """Compute quality score and flags for a single lead."""
    flags: set[str] = set()
    penalty = 0.0

    # 1) Flatline / lead-off detection
    if _detect_flatline(sig):
        flags.add("flatline")
        penalty += _PENALTY_FLATLINE

    # 2) Saturation / clipping detection
    if _detect_clipping(sig):
        flags.add("clipping")
        penalty += _PENALTY_CLIPPING

    # 3) Excessive baseline wander
    if _detect_baseline_wander(sig, sample_rate):
        flags.add("baseline_wander")
        penalty += _PENALTY_BASELINE_WANDER

    # 4) Motion artifact (high-frequency energy)
    if _detect_motion_artifact(sig, sample_rate):
        flags.add("motion_artifact")
        penalty += _PENALTY_MOTION_ARTIFACT

    score = max(0.0, 100.0 - penalty)
    classification = _classify(score, good_threshold, marginal_threshold)

    return LeadQuality(
        lead_name=lead_name,
        score=score,
        classification=classification,
        flags=flags,
    )


def _classify(
    score: float,
    good_threshold: float,
    marginal_threshold: float,
) -> QualityClass:
    """Map a numeric score to a quality class."""
    if score >= good_threshold:
        return "good"
    if score >= marginal_threshold:
        return "marginal"
    return "poor"


def _recommend(classification: QualityClass) -> Recommendation:
    """Map a quality class to a clinical recommendation."""
    if classification == "good":
        return "accept"
    if classification == "marginal":
        return "review"
    return "reject"


# ── artefact detectors ──────────────────────────────────────────


def _detect_flatline(sig: NDArray[np.float64]) -> bool:
    """Return ``True`` if ≥ 10 % of samples are practically constant.

    A sample is "flat" when its absolute change from the previous
    sample is below ``_FLATLINE_TOLERANCE``.
    """
    if len(sig) < 2:
        return False
    diffs = np.abs(np.diff(sig))
    flat_count = int(np.sum(diffs < _FLATLINE_TOLERANCE))
    return (flat_count / len(diffs)) >= _FLATLINE_FRACTION_THRESHOLD


def _detect_clipping(sig: NDArray[np.float64]) -> bool:
    """Return ``True`` if a notable fraction of samples sit at the
    min/max rail (saturation).

    Identifies samples within 0.5 % of the total amplitude range from
    the signal extremes.
    """
    if len(sig) < 2:
        return False
    sig_min = float(np.min(sig))
    sig_max = float(np.max(sig))
    amplitude = sig_max - sig_min
    if amplitude < _FLATLINE_TOLERANCE:
        # Entire signal is constant — handled by flatline detector.
        return False

    margin = amplitude * (1.0 - _CLIPPING_PERCENTILE / 100.0)
    at_min = int(np.sum(sig <= sig_min + margin))
    at_max = int(np.sum(sig >= sig_max - margin))
    clipped = at_min + at_max
    return (clipped / len(sig)) >= _CLIPPING_FRACTION_THRESHOLD


def _detect_baseline_wander(sig: NDArray[np.float64], fs: float) -> bool:
    """Return ``True`` if low-frequency (< 0.5 Hz) energy dominates.

    Uses the ratio of spectral energy below 0.5 Hz to total energy.
    """
    n = len(sig)
    if n < 4:
        return False

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    total_energy = float(np.sum(fft_mag**2))
    if total_energy < _FLATLINE_TOLERANCE:
        return False

    low_mask = freqs < 0.5
    low_energy = float(np.sum(fft_mag[low_mask] ** 2))
    ratio = low_energy / total_energy
    return ratio >= _BASELINE_WANDER_RATIO_THRESHOLD


def _detect_motion_artifact(sig: NDArray[np.float64], fs: float) -> bool:
    """Return ``True`` if high-frequency energy (> 40 Hz) dominates.

    Uses the ratio of spectral energy above 40 Hz to total energy.
    A high ratio indicates EMG / motion contamination.
    """
    n = len(sig)
    if n < 4:
        return False

    nyq = fs / 2.0
    if nyq <= 40.0:
        # Cannot assess high-frequency content — skip.
        return False

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    total_energy = float(np.sum(fft_mag**2))
    if total_energy < _FLATLINE_TOLERANCE:
        return False

    high_mask = freqs > 40.0
    high_energy = float(np.sum(fft_mag[high_mask] ** 2))
    ratio = high_energy / total_energy
    return ratio >= _MOTION_ARTIFACT_RATIO_THRESHOLD
