"""Tests for aortica.xai.explain — Integrated Gradient XAI."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.xai.explain import (
    ECG_SEGMENTS,
    FeatureAttribution,
    FeatureContribution,
    SegmentBoundaries,
    _map_attributions_to_segments,
    delineate_segments,
)

torch = pytest.importorskip("torch")

from aortica.models.aortica_model import AorticaModel  # noqa: E402, I001


# ── Fixtures ──────────────────────────────────────────────────────


def _make_synthetic_ecg(
    n_leads: int = 12,
    duration_s: float = 2.0,
    sample_rate: float = 500.0,
    heart_rate: float = 75.0,
) -> ECGRecord:
    """Create a synthetic ECG record with Gaussian-shaped QRS complexes."""
    n_samples = int(duration_s * sample_rate)
    signals = np.zeros((n_leads, n_samples), dtype=np.float64)

    beat_interval = sample_rate * 60.0 / heart_rate
    beat_positions = np.arange(
        beat_interval / 2, n_samples, beat_interval
    ).astype(int)

    t = np.arange(n_samples)
    for pos in beat_positions:
        # QRS complex (narrow Gaussian)
        qrs = np.exp(-0.5 * ((t - pos) / (sample_rate * 0.01)) ** 2)
        # T wave (wider Gaussian, ~200ms after R)
        t_pos = pos + int(0.2 * sample_rate)
        t_wave = 0.3 * np.exp(-0.5 * ((t - t_pos) / (sample_rate * 0.04)) ** 2)
        # P wave (small Gaussian, ~150ms before R)
        p_pos = pos - int(0.15 * sample_rate)
        p_wave = 0.15 * np.exp(-0.5 * ((t - p_pos) / (sample_rate * 0.02)) ** 2)

        for lead in range(n_leads):
            scale = 1.0 if lead < 6 else 0.8  # limb vs chest leads
            signals[lead] += scale * (qrs + t_wave + p_wave)

    all_leads = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    lead_names = all_leads[:n_leads]
    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="synthetic",
    )


def _make_small_model() -> AorticaModel:
    """Create a small AorticaModel for testing."""
    return AorticaModel(
        in_channels=12,
        feature_dim=252,  # divisible by 12
        num_leads=12,
        num_heads=4,
        head_dim=16,
        head_hidden_dim=32,
        head_dropout=0.0,
        attention_dropout=0.0,
        enabled_tasks=["rhythm"],
    )


# ── Constants Tests ───────────────────────────────────────────────


class TestConstants:
    """Test ECG segment constants."""

    def test_ecg_segments_count(self) -> None:
        assert len(ECG_SEGMENTS) == 6

    def test_ecg_segments_contents(self) -> None:
        expected = [
            "P wave",
            "PR interval",
            "QRS complex",
            "ST segment",
            "T wave",
            "QT/QTc",
        ]
        assert ECG_SEGMENTS == expected


# ── SegmentBoundaries Tests ───────────────────────────────────────


class TestSegmentBoundaries:
    """Test SegmentBoundaries dataclass."""

    def test_default_values(self) -> None:
        sb = SegmentBoundaries()
        assert sb.p_start == -1
        assert sb.p_end == -1
        assert sb.qrs_start == -1
        assert sb.qrs_end == -1
        assert sb.t_start == -1
        assert sb.t_end == -1

    def test_custom_values(self) -> None:
        sb = SegmentBoundaries(
            p_start=10, p_end=30, qrs_start=40, qrs_end=60,
            t_start=80, t_end=120,
        )
        assert sb.p_start == 10
        assert sb.qrs_end == 60
        assert sb.t_end == 120


# ── FeatureContribution Tests ────────────────────────────────────


class TestFeatureContribution:
    """Test FeatureContribution dataclass."""

    def test_construction(self) -> None:
        fc = FeatureContribution(
            feature_name="QRS complex", lead="V1", delta_score=0.42
        )
        assert fc.feature_name == "QRS complex"
        assert fc.lead == "V1"
        assert fc.delta_score == pytest.approx(0.42)


# ── FeatureAttribution Tests ─────────────────────────────────────


class TestFeatureAttribution:
    """Test FeatureAttribution dataclass."""

    def test_construction(self) -> None:
        attr = FeatureAttribution(
            per_lead_attributions={"I": np.zeros(100)},
            segment_attributions={"I": {"QRS complex": 0.5}},
            task="rhythm",
        )
        assert "I" in attr.per_lead_attributions
        assert attr.task == "rhythm"
        assert len(attr.top_features) == 0  # empty default

    def test_default_task(self) -> None:
        attr = FeatureAttribution(
            per_lead_attributions={},
            segment_attributions={},
        )
        assert attr.task == "rhythm"


# ── Delineate Segments Tests ─────────────────────────────────────


class TestDelineateSegments:
    """Test ECG segment delineation."""

    def test_returns_list(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        result = delineate_segments(lead_signal, ecg.sample_rate)
        assert isinstance(result, list)

    def test_detects_beats(self) -> None:
        ecg = _make_synthetic_ecg(duration_s=3.0, heart_rate=60.0)
        lead_signal = ecg.signals[0]
        result = delineate_segments(lead_signal, ecg.sample_rate)
        # At 60 bpm over 3 seconds, expect ~3 beats
        assert len(result) >= 2

    def test_boundaries_are_segment_boundaries(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        result = delineate_segments(lead_signal, ecg.sample_rate)
        for sb in result:
            assert isinstance(sb, SegmentBoundaries)

    def test_boundaries_non_negative(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        result = delineate_segments(lead_signal, ecg.sample_rate)
        for sb in result:
            assert sb.p_start >= 0
            assert sb.qrs_start >= 0
            assert sb.qrs_end >= 0
            assert sb.t_start >= 0

    def test_boundaries_ordered(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        result = delineate_segments(lead_signal, ecg.sample_rate)
        for sb in result:
            assert sb.p_start <= sb.p_end
            assert sb.qrs_start <= sb.qrs_end
            assert sb.t_start <= sb.t_end

    def test_boundaries_within_signal(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        n = len(lead_signal)
        result = delineate_segments(lead_signal, ecg.sample_rate)
        for sb in result:
            assert 0 <= sb.p_start < n
            assert 0 <= sb.qrs_end < n
            assert 0 <= sb.t_end < n

    def test_empty_signal(self) -> None:
        sig = np.zeros(10)
        result = delineate_segments(sig, 500.0)
        assert result == []

    def test_with_explicit_r_peaks(self) -> None:
        ecg = _make_synthetic_ecg()
        lead_signal = ecg.signals[0]
        r_peaks = np.array([250, 750], dtype=np.intp)
        result = delineate_segments(
            lead_signal, ecg.sample_rate, r_peaks=r_peaks
        )
        assert len(result) == 2


# ── Attribution Mapping Tests ─────────────────────────────────────


class TestAttributionMapping:
    """Test _map_attributions_to_segments."""

    def test_zero_attributions(self) -> None:
        attr = np.zeros(1000)
        bounds = [
            SegmentBoundaries(
                p_start=10, p_end=50, qrs_start=60, qrs_end=100,
                t_start=120, t_end=200,
            )
        ]
        result = _map_attributions_to_segments(attr, bounds)
        for name in ECG_SEGMENTS:
            assert name in result
            assert result[name] == 0.0

    def test_qrs_attribution_dominant(self) -> None:
        attr = np.zeros(1000)
        # Put large values in QRS region
        attr[60:100] = 10.0
        bounds = [
            SegmentBoundaries(
                p_start=10, p_end=50, qrs_start=60, qrs_end=100,
                t_start=120, t_end=200,
            )
        ]
        result = _map_attributions_to_segments(attr, bounds)
        assert result["QRS complex"] > result["P wave"]
        assert result["QRS complex"] > result["T wave"]

    def test_all_segments_present(self) -> None:
        attr = np.random.default_rng(42).standard_normal(1000)
        bounds = [
            SegmentBoundaries(
                p_start=10, p_end=50, qrs_start=60, qrs_end=100,
                t_start=120, t_end=200,
            )
        ]
        result = _map_attributions_to_segments(attr, bounds)
        for name in ECG_SEGMENTS:
            assert name in result

    def test_empty_boundaries_list(self) -> None:
        attr = np.ones(1000)
        result = _map_attributions_to_segments(attr, [])
        for name in ECG_SEGMENTS:
            assert result[name] == 0.0

    def test_multiple_beats_averaged(self) -> None:
        attr = np.ones(2000) * 0.1
        # Two beats with different QRS magnitudes
        attr[60:100] = 1.0  # beat 1 QRS
        attr[560:600] = 3.0  # beat 2 QRS
        bounds = [
            SegmentBoundaries(
                p_start=10, p_end=50, qrs_start=60, qrs_end=100,
                t_start=120, t_end=200,
            ),
            SegmentBoundaries(
                p_start=510, p_end=550, qrs_start=560, qrs_end=600,
                t_start=620, t_end=700,
            ),
        ]
        result = _map_attributions_to_segments(attr, bounds)
        # Average of 40*1.0=40 and 40*3.0=120 → 80
        assert result["QRS complex"] == pytest.approx(80.0)


# ── Integrated Explain Tests ─────────────────────────────────────


class TestExplain:
    """Test the full explain() pipeline."""

    def test_explain_returns_feature_attribution(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert isinstance(result, FeatureAttribution)

    def test_per_lead_attributions_shape(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert len(result.per_lead_attributions) == ecg.num_leads
        for lead_name in ecg.lead_names:
            assert lead_name in result.per_lead_attributions
            assert result.per_lead_attributions[lead_name].shape == (
                ecg.num_samples,
            )

    def test_segment_attributions_keys(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert len(result.segment_attributions) == ecg.num_leads
        for lead_name in ecg.lead_names:
            assert lead_name in result.segment_attributions
            seg_dict = result.segment_attributions[lead_name]
            for seg_name in ECG_SEGMENTS:
                assert seg_name in seg_dict

    def test_top_features_at_most_3(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert len(result.top_features) <= 3

    def test_top_features_are_feature_contributions(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        for fc in result.top_features:
            assert isinstance(fc, FeatureContribution)
            assert fc.feature_name in ECG_SEGMENTS
            assert fc.lead in ecg.lead_names

    def test_top_features_sorted_descending(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        scores = [abs(fc.delta_score) for fc in result.top_features]
        assert scores == sorted(scores, reverse=True)

    def test_task_field_set(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert result.task == "rhythm"

    def test_integrated_gradients_raw_shape(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        assert result.integrated_gradients_raw is not None
        assert result.integrated_gradients_raw.shape == (
            ecg.num_leads,
            ecg.num_samples,
        )

    def test_explain_with_target_class(self) -> None:
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(
            model, ecg, task="rhythm", n_steps=5, target_class=0
        )
        assert isinstance(result, FeatureAttribution)
        assert result.integrated_gradients_raw is not None

    def test_top_features_exclude_qt_qtc(self) -> None:
        """Top features should NOT include QT/QTc (it's a superset)."""
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        for fc in result.top_features:
            assert fc.feature_name != "QT/QTc"

    def test_segment_attributions_non_negative(self) -> None:
        """Segment attributions use absolute values, so should be >= 0."""
        from aortica.xai import explain

        model = _make_small_model()
        ecg = _make_synthetic_ecg()
        result = explain(model, ecg, task="rhythm", n_steps=5)
        for seg_dict in result.segment_attributions.values():
            for score in seg_dict.values():
                assert score >= 0.0

    def test_explain_few_leads(self) -> None:
        """Explain works with fewer than 12 leads."""
        from aortica.xai import explain

        model = AorticaModel(
            in_channels=3,
            feature_dim=252,
            num_leads=3,
            num_heads=3,
            head_dim=16,
            head_hidden_dim=32,
            head_dropout=0.0,
            enabled_tasks=["rhythm"],
        )
        ecg = _make_synthetic_ecg(n_leads=3)
        result = explain(model, ecg, task="rhythm", n_steps=3)
        assert len(result.per_lead_attributions) == 3
        assert len(result.top_features) <= 3
