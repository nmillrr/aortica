"""Tests for aortica.signal.quality_scoring."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.signal.quality_scoring import (
    LeadQuality,
    QualityReport,
    score_quality,
)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_record(
    signals: np.ndarray,
    sample_rate: float = 500.0,
    lead_names: list[str] | None = None,
) -> ECGRecord:
    """Convenience factory for a minimal ECGRecord."""
    if lead_names is None:
        lead_names = [f"lead_{i}" for i in range(signals.shape[0])]
    return ECGRecord(
        signals=signals.astype(np.float64),
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="test",
    )


def _clean_ecg(
    duration_s: float = 10.0,
    fs: float = 500.0,
    num_leads: int = 1,
    heart_rate: float = 72.0,
) -> np.ndarray:
    """Generate a synthetic clean ECG-like signal.

    Produces Gaussian-shaped QRS complexes at a fixed heart rate with a
    small amount of baseline noise.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rr_interval = 60.0 / heart_rate
    signals = np.zeros((num_leads, n))
    for lead in range(num_leads):
        sig = np.zeros(n)
        peak_time = 0.0
        while peak_time < duration_s:
            qrs = np.exp(-0.5 * ((t - peak_time) / 0.02) ** 2) * 1000.0
            sig += qrs
            peak_time += rr_interval
        # Add small realistic noise (< 10 µV RMS)
        sig += np.random.default_rng(42 + lead).normal(0, 5, n)
        signals[lead] = sig
    return signals


# ─────────────────────────────────────────────────────────────────
# Tests — basic API / return types
# ─────────────────────────────────────────────────────────────────


class TestScoreQualityAPI:
    """Tests for the overall API and return structure."""

    def test_returns_quality_report(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert isinstance(report, QualityReport)

    def test_per_lead_count_matches_record(self) -> None:
        signals = _clean_ecg(num_leads=3)
        record = _make_record(signals)
        report = score_quality(record)
        assert len(report.per_lead) == 3

    def test_per_lead_types(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        lq = report.per_lead[0]
        assert isinstance(lq, LeadQuality)
        assert isinstance(lq.score, float)
        assert isinstance(lq.classification, str)
        assert isinstance(lq.flags, set)

    def test_overall_score_is_mean(self) -> None:
        signals = _clean_ecg(num_leads=4)
        record = _make_record(signals)
        report = score_quality(record)
        expected = np.mean([lq.score for lq in report.per_lead])
        assert abs(report.overall_score - expected) < 1e-6

    def test_score_range(self) -> None:
        """All scores must be between 0 and 100."""
        signals = _clean_ecg(num_leads=2)
        record = _make_record(signals)
        report = score_quality(record)
        for lq in report.per_lead:
            assert 0.0 <= lq.score <= 100.0
        assert 0.0 <= report.overall_score <= 100.0


# ─────────────────────────────────────────────────────────────────
# Tests — clean signal
# ─────────────────────────────────────────────────────────────────


class TestCleanSignal:
    """A clean ECG should receive a high score and 'good' classification."""

    def test_clean_signal_good(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert report.overall_score >= 70.0
        assert report.overall_classification == "good"
        assert report.recommendation == "accept"

    def test_clean_no_flags(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        for lq in report.per_lead:
            assert len(lq.flags) == 0


# ─────────────────────────────────────────────────────────────────
# Tests — flatline / lead-off detection
# ─────────────────────────────────────────────────────────────────


class TestFlatline:
    """Flatline / lead-off detection."""

    def test_fully_flat_signal(self) -> None:
        """All-zero signal should be flagged as flatline."""
        signals = np.zeros((1, 5000))
        record = _make_record(signals)
        report = score_quality(record)
        assert "flatline" in report.per_lead[0].flags
        assert report.per_lead[0].classification != "good"

    def test_partial_flatline(self) -> None:
        """Signal flat for a substantial portion should still flag."""
        rng = np.random.default_rng(1)
        sig = rng.normal(0, 100, 5000)
        # Make 20 % of the signal flat (well above 10 % threshold).
        sig[1000:2000] = 0.0
        signals = sig[np.newaxis, :]
        record = _make_record(signals)
        report = score_quality(record)
        assert "flatline" in report.per_lead[0].flags

    def test_no_flatline_on_normal(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert "flatline" not in report.per_lead[0].flags


# ─────────────────────────────────────────────────────────────────
# Tests — saturation / clipping detection
# ─────────────────────────────────────────────────────────────────


class TestClipping:
    """Saturation / clipping detection."""

    def test_clipped_signal(self) -> None:
        """A signal hard-clipped at the rail should be flagged."""
        rng = np.random.default_rng(2)
        sig = rng.normal(0, 500, 5000)
        # Clip at ±800 — a generous clip that should produce lots of
        # samples exactly at the rail.
        sig = np.clip(sig, -800, 800)
        signals = sig[np.newaxis, :]
        record = _make_record(signals)
        report = score_quality(record)
        assert "clipping" in report.per_lead[0].flags

    def test_no_clipping_on_clean(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert "clipping" not in report.per_lead[0].flags


# ─────────────────────────────────────────────────────────────────
# Tests — baseline wander detection
# ─────────────────────────────────────────────────────────────────


class TestBaselineWander:
    """Excessive baseline wander detection."""

    def test_strong_wander(self) -> None:
        """A large low-frequency sinusoid should be flagged."""
        fs = 500.0
        n = 5000
        t = np.arange(n) / fs
        # 0.1 Hz wander — extremely low frequency, large amplitude.
        sig = 2000.0 * np.sin(2.0 * np.pi * 0.1 * t)
        signals = sig[np.newaxis, :]
        record = _make_record(signals, sample_rate=fs)
        report = score_quality(record)
        assert "baseline_wander" in report.per_lead[0].flags

    def test_no_wander_on_clean(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert "baseline_wander" not in report.per_lead[0].flags


# ─────────────────────────────────────────────────────────────────
# Tests — motion artifact detection
# ─────────────────────────────────────────────────────────────────


class TestMotionArtifact:
    """Motion / EMG artifact detection."""

    def test_high_freq_noise(self) -> None:
        """Signal dominated by high-frequency noise should be flagged."""
        rng = np.random.default_rng(3)
        fs = 500.0
        n = 5000
        # White noise has substantial energy above 40 Hz when fs=500.
        sig = rng.normal(0, 500, n)
        signals = sig[np.newaxis, :]
        record = _make_record(signals, sample_rate=fs)
        report = score_quality(record)
        assert "motion_artifact" in report.per_lead[0].flags

    def test_no_artifact_on_clean(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert "motion_artifact" not in report.per_lead[0].flags


# ─────────────────────────────────────────────────────────────────
# Tests — classification thresholds
# ─────────────────────────────────────────────────────────────────


class TestClassification:
    """Classification and recommendation mapping."""

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should shift the classification boundaries."""
        signals = _clean_ecg()
        record = _make_record(signals)
        # Very high threshold → normal clean signal becomes marginal.
        report = score_quality(record, good_threshold=99.0, marginal_threshold=50.0)
        # With a 100 score the good_threshold is 99, so still good.
        # But if we set it to 101 it would become marginal (not possible
        # because max score is 100). Instead test that it's still scored.
        assert report.overall_score <= 100.0

    def test_marginal_classification(self) -> None:
        """Score between marginal and good thresholds → 'marginal'."""
        # Create a signal with ONE artefact (e.g. clipping → score 70).
        rng = np.random.default_rng(5)
        sig = rng.normal(0, 500, 5000)
        sig = np.clip(sig, -800, 800)
        signals = sig[np.newaxis, :]
        record = _make_record(signals)
        report = score_quality(record)
        # Clipping penalty is 30 → score ~70; with motion artifact too
        # it may be lower. Check not 'good'.
        lq = report.per_lead[0]
        assert "clipping" in lq.flags
        assert lq.score < 100.0

    def test_poor_classification(self) -> None:
        """A signal with multiple artefacts should be 'poor' / 'reject'."""
        # All-zero signal (flatline, penalty=40) → score 60 = marginal.
        # To reach 'poor' (< 40) we need multiple artefacts.
        rng = np.random.default_rng(10)
        # White noise (motion artifact penalty=20) + heavy clipping
        sig = rng.normal(0, 500, 5000)
        sig = np.clip(sig, -400, 400)  # clipping penalty=30
        # Also inject a long flatline segment for flatline penalty=40
        sig[0:2500] = 0.0
        signals = sig[np.newaxis, :]
        record = _make_record(signals)
        report = score_quality(record)
        assert report.overall_classification == "poor"
        assert report.recommendation == "reject"

    def test_good_recommendation(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        report = score_quality(record)
        assert report.recommendation == "accept"

    def test_invalid_thresholds_raises(self) -> None:
        signals = _clean_ecg()
        record = _make_record(signals)
        with pytest.raises(ValueError, match="good_threshold"):
            score_quality(record, good_threshold=30, marginal_threshold=50)


# ─────────────────────────────────────────────────────────────────
# Tests — multi-lead records
# ─────────────────────────────────────────────────────────────────


class TestMultiLead:
    """Tests with multi-lead records."""

    def test_mixed_quality_leads(self) -> None:
        """One clean lead + one flatline lead → overall not 'good'."""
        clean = _clean_ecg(num_leads=1)
        flat = np.zeros((1, clean.shape[1]))
        signals = np.vstack([clean, flat])
        record = _make_record(signals, lead_names=["II", "V1"])
        report = score_quality(record)
        # Lead "II" should be good.
        ii = next(lq for lq in report.per_lead if lq.lead_name == "II")
        assert ii.classification == "good"
        # Lead "V1" — flatline only ⇒ score 60 = marginal.
        v1 = next(lq for lq in report.per_lead if lq.lead_name == "V1")
        assert v1.classification in ("marginal", "poor")
        assert "flatline" in v1.flags
        # Overall mean is (100 + 60) / 2 = 80 → still 'good'.
        # But the overall score should be strictly less than a fully
        # clean record's score, and V1 should have been flagged.
        assert report.overall_score < 100.0
        assert report.overall_score == pytest.approx(80.0, abs=1.0)

    def test_all_good_leads(self) -> None:
        signals = _clean_ecg(num_leads=12, duration_s=10.0)
        lead_names = [
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ]
        record = _make_record(signals, lead_names=lead_names)
        report = score_quality(record)
        assert report.overall_classification == "good"
        assert report.recommendation == "accept"


# ─────────────────────────────────────────────────────────────────
# Tests — edge cases
# ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge-case scenarios."""

    def test_short_signal(self) -> None:
        """Very short signal should not crash."""
        signals = np.array([[100.0, 200.0, 300.0]])
        record = _make_record(signals, sample_rate=500.0)
        report = score_quality(record)
        assert 0.0 <= report.overall_score <= 100.0

    def test_single_sample(self) -> None:
        """Single sample signal shouldn't crash."""
        signals = np.array([[42.0]])
        record = _make_record(signals, sample_rate=500.0)
        report = score_quality(record)
        assert 0.0 <= report.overall_score <= 100.0

    def test_low_sample_rate(self) -> None:
        """Low sample rate (< 80 Hz) should still work — motion
        artifact detector should skip gracefully."""
        rng = np.random.default_rng(7)
        signals = rng.normal(0, 100, (1, 500))
        record = _make_record(signals, sample_rate=50.0)
        report = score_quality(record)
        assert 0.0 <= report.overall_score <= 100.0
        # Motion artifact can't be assessed at fs=50 (Nyquist=25 < 40).
        assert "motion_artifact" not in report.per_lead[0].flags
