"""Tests for aortica.federated.data_quality (US-115)."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.federated.data_quality import (
    _TASK_NUM_OUTPUTS,
    DQ_POLICIES,
    DataQualityGate,
    DataQualityReport,
    QualityCheck,
    site_admitted,
)
from aortica.io.ecg_record import ECGRecord

_LABEL_WIDTH = sum(_TASK_NUM_OUTPUTS.values())  # 28 + 19 + 19 + 6 = 72
_SLICES = {}
_offset = 0
for _task in ["rhythm", "structural", "ischaemia", "risk"]:
    _SLICES[_task] = (_offset, _offset + _TASK_NUM_OUTPUTS[_task])
    _offset += _TASK_NUM_OUTPUTS[_task]


# ─────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────


# Scaled-down thresholds keep the synthetic datasets (and thus per-record
# signal-quality scoring) small enough to run fast while exercising every
# threshold boundary.  Ratios mirror the production defaults:
#   min=50 (prod 500), warn=20 (prod 200), block=10 (prod 100).
def _gate(**overrides: object) -> DataQualityGate:
    params: dict = {
        "min_sample_size": 50,
        "warn_sample_size": 20,
        "block_sample_size": 10,
    }
    params.update(overrides)
    return DataQualityGate(**params)  # type: ignore[arg-type]


def _clean_ecg(
    duration_s: float = 2.0,
    fs: float = 250.0,
    num_leads: int = 2,
    heart_rate: float = 72.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic clean, high-quality ECG-like signal."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rr = 60.0 / heart_rate
    signals = np.zeros((num_leads, n))
    for lead in range(num_leads):
        sig = np.zeros(n)
        peak = 0.0
        while peak < duration_s:
            sig += np.exp(-0.5 * ((t - peak) / 0.02) ** 2) * 1000.0
            peak += rr
        sig += np.random.default_rng(seed + lead).normal(0, 5, n)
        signals[lead] = sig
    return signals


def _poor_ecg(
    num_leads: int = 2, fs: float = 250.0, seed: int = 0
) -> np.ndarray:
    """Generate a poor-quality signal (heavy clipping + flat segment).

    Scores well below the 40 "marginal" threshold in
    :func:`aortica.signal.quality_scoring.score_quality`.
    """
    n = int(2 * fs)
    rng = np.random.default_rng(seed)
    sig = np.clip(rng.normal(0, 1000, (num_leads, n)), -10, 10)
    sig[:, : n // 5] = 0.0  # flat segment
    return sig


def _make_record(signals: np.ndarray, sample_rate: float = 500.0) -> ECGRecord:
    lead_names = [f"lead_{i}" for i in range(signals.shape[0])]
    return ECGRecord(
        signals=signals.astype(np.float64),
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="test",
    )


def _label_vector(superclasses: list[str]) -> np.ndarray:
    """Build a label vector with a positive in each named superclass."""
    vec = np.zeros(_LABEL_WIDTH, dtype=np.float64)
    for sc in superclasses:
        start, _ = _SLICES[sc]
        vec[start] = 1.0
    return vec


def _build_dataset(
    n: int,
    *,
    quality: str = "good",
    labeled_fraction: float = 1.0,
    superclasses: list[str] | None = None,
) -> tuple[list[ECGRecord], np.ndarray]:
    """Construct a synthetic (records, labels) dataset.

    Args:
        n: number of ECGs.
        quality: "good" or "flat" (poor).
        labeled_fraction: fraction of records that carry a label.
        superclasses: which task superclasses get positives (default all 4).
    """
    if superclasses is None:
        superclasses = ["rhythm", "structural", "ischaemia", "risk"]
    records: list[ECGRecord] = []
    labels = np.zeros((n, _LABEL_WIDTH), dtype=np.float64)
    num_labeled = int(round(labeled_fraction * n))
    for i in range(n):
        sig = _clean_ecg(seed=i) if quality == "good" else _poor_ecg()
        records.append(_make_record(sig))
        if i < num_labeled:
            labels[i] = _label_vector(superclasses)
    return records, labels


class _FakeDataset:
    """Duck-typed dataset exposing .records and .labels."""

    def __init__(self, records: list[ECGRecord], labels: np.ndarray) -> None:
        self.records = records
        self.labels = labels


# ─────────────────────────────────────────────────────────────────
# API / return structure
# ─────────────────────────────────────────────────────────────────


class TestGateApi:
    def test_validate_returns_report(self) -> None:
        gate = DataQualityGate()
        report = gate.validate(_build_dataset(60))
        assert isinstance(report, DataQualityReport)
        assert all(isinstance(c, QualityCheck) for c in report.checks)

    def test_accepts_duck_typed_dataset(self) -> None:
        records, labels = _build_dataset(60)
        report = _gate().validate(_FakeDataset(records, labels))
        assert report.passed

    def test_report_to_dict_json_serialisable(self) -> None:
        import json

        report = _gate().validate(_build_dataset(60))
        json.dumps(report.to_dict())  # must not raise

    def test_all_five_checks_present(self) -> None:
        report = _gate().validate(_build_dataset(60))
        names = {c.name for c in report.checks}
        assert names == {
            "sample_size",
            "format_consistency",
            "signal_quality",
            "label_completeness",
            "label_diversity",
        }

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            DataQualityGate(block_sample_size=300, warn_sample_size=200)

    def test_bad_dataset_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _gate().validate(object())

    def test_label_count_mismatch_raises(self) -> None:
        records, labels = _build_dataset(10)
        with pytest.raises(ValueError):
            _gate().validate((records, labels[:5]))


# ─────────────────────────────────────────────────────────────────
# Passing dataset
# ─────────────────────────────────────────────────────────────────


class TestPassingDataset:
    def test_good_dataset_passes(self) -> None:
        report = _gate().validate(_build_dataset(60))
        assert report.passed
        assert not report.blocking
        for c in report.checks:
            assert c.passed, c.name

    def test_admitted_under_all_policies(self) -> None:
        report = _gate().validate(_build_dataset(60))
        for policy in DQ_POLICIES:
            assert site_admitted(report, policy)


# ─────────────────────────────────────────────────────────────────
# Sample-size thresholds
# ─────────────────────────────────────────────────────────────────


class TestSampleSize:
    def test_below_block_threshold_blocks(self) -> None:
        report = _gate().validate(_build_dataset(5))
        size = report.check("sample_size")
        assert size is not None and not size.passed and size.blocking
        assert report.blocking

    def test_between_block_and_warn_warns_not_blocks(self) -> None:
        report = _gate().validate(_build_dataset(15))
        size = report.check("sample_size")
        assert size is not None and not size.passed
        assert not size.blocking

    def test_between_warn_and_min_soft_fail(self) -> None:
        report = _gate().validate(_build_dataset(30))
        size = report.check("sample_size")
        assert size is not None and not size.passed and not size.blocking

    def test_at_min_passes(self) -> None:
        report = _gate().validate(_build_dataset(50))
        size = report.check("sample_size")
        assert size is not None and size.passed


# ─────────────────────────────────────────────────────────────────
# Signal quality distribution
# ─────────────────────────────────────────────────────────────────


class TestSignalQuality:
    def test_flatline_dataset_fails_quality(self) -> None:
        report = _gate().validate(
            _build_dataset(60, quality="flat")
        )
        q = report.check("signal_quality")
        assert q is not None and not q.passed and q.blocking

    def test_mostly_good_passes(self) -> None:
        # 80% good, 20% flat → above the 70% requirement.
        good_r, good_l = _build_dataset(48, quality="good")
        bad_r, bad_l = _build_dataset(12, quality="flat")
        records = good_r + bad_r
        labels = np.vstack([good_l, bad_l])
        report = _gate().validate((records, labels))
        q = report.check("signal_quality")
        assert q is not None and q.passed


# ─────────────────────────────────────────────────────────────────
# Label completeness
# ─────────────────────────────────────────────────────────────────


class TestLabelCompleteness:
    def test_low_completeness_fails(self) -> None:
        report = _gate().validate(
            _build_dataset(60, labeled_fraction=0.5)
        )
        c = report.check("label_completeness")
        assert c is not None and not c.passed and c.blocking

    def test_high_completeness_passes(self) -> None:
        report = _gate().validate(
            _build_dataset(60, labeled_fraction=0.9)
        )
        c = report.check("label_completeness")
        assert c is not None and c.passed


# ─────────────────────────────────────────────────────────────────
# Label diversity
# ─────────────────────────────────────────────────────────────────


class TestLabelDiversity:
    def test_all_four_superclasses_passes(self) -> None:
        report = _gate().validate(_build_dataset(60))
        d = report.check("label_diversity")
        assert d is not None and d.passed

    def test_two_superclasses_fails(self) -> None:
        report = _gate().validate(
            _build_dataset(60, superclasses=["rhythm", "structural"])
        )
        d = report.check("label_diversity")
        assert d is not None and not d.passed and d.blocking

    def test_three_superclasses_passes(self) -> None:
        report = _gate().validate(
            _build_dataset(
                600, superclasses=["rhythm", "structural", "ischaemia"]
            )
        )
        d = report.check("label_diversity")
        assert d is not None and d.passed

    def test_positives_per_superclass_counted(self) -> None:
        report = _gate().validate(_build_dataset(60))
        per = report.statistics["positive_per_superclass"]
        assert per["rhythm"] == 60
        assert per["risk"] == 60


# ─────────────────────────────────────────────────────────────────
# Format consistency
# ─────────────────────────────────────────────────────────────────


class _BrokenRecord:
    """Object that raises when quality-scored."""


class TestFormatConsistency:
    def test_unreadable_record_blocks(self) -> None:
        records, labels = _build_dataset(60)
        records[0] = _BrokenRecord()  # type: ignore[assignment]
        report = _gate().validate((records, labels))
        f = report.check("format_consistency")
        assert f is not None and not f.passed and f.blocking
        assert report.statistics["num_format_errors"] == 1

    def test_clean_records_pass_format(self) -> None:
        report = _gate().validate(_build_dataset(60))
        f = report.check("format_consistency")
        assert f is not None and f.passed


# ─────────────────────────────────────────────────────────────────
# Admission policy
# ─────────────────────────────────────────────────────────────────


class TestAdmissionPolicy:
    def test_strict_excludes_blocking(self) -> None:
        report = _gate().validate(_build_dataset(5))
        assert report.blocking
        assert not site_admitted(report, "strict")
        assert not site_admitted(report, "warn")
        assert site_admitted(report, "permissive")

    def test_warn_only_admitted(self) -> None:
        # 300 samples: sample_size soft-fails (non-blocking) but no blocking
        # check fails, so the site is admitted under strict too.
        report = _gate().validate(_build_dataset(30))
        assert not report.blocking
        assert site_admitted(report, "strict")

    def test_unknown_policy_raises(self) -> None:
        report = _gate().validate(_build_dataset(60))
        with pytest.raises(ValueError):
            site_admitted(report, "nonsense")


# ─────────────────────────────────────────────────────────────────
# Summary / recommendations
# ─────────────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_contains_verdict(self) -> None:
        report = _gate().validate(_build_dataset(5))
        assert "BLOCKED" in report.summary()

    def test_recommendations_present_on_failure(self) -> None:
        report = _gate().validate(_build_dataset(5))
        assert report.recommendations


# ─────────────────────────────────────────────────────────────────
# FL client integration (US-063 hook)
# ─────────────────────────────────────────────────────────────────


class _FakeLoader:
    """Minimal stand-in for a torch DataLoader exposing `.dataset`."""

    def __init__(self, dataset: _FakeDataset) -> None:
        self.dataset = dataset


class TestClientIntegration:
    def test_client_runs_gate_and_reports_metrics(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient

        records, labels = _build_dataset(60)
        loader = _FakeLoader(_FakeDataset(records, labels))
        client = AorticaFlowerClient(train_loader=loader)

        report = client.check_data_quality(gate=_gate())
        assert isinstance(report, DataQualityReport)
        assert report.passed

        metrics = client._data_quality_metrics()
        assert metrics["dq_passed"] == 1.0
        assert metrics["dq_blocking"] == 0.0
        assert metrics["dq_num_ecgs"] == 60.0

    def test_client_reports_blocking_small_site(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient

        records, labels = _build_dataset(5)
        loader = _FakeLoader(_FakeDataset(records, labels))
        client = AorticaFlowerClient(train_loader=loader)

        client.check_data_quality(gate=_gate())
        metrics = client._data_quality_metrics()
        assert metrics["dq_blocking"] == 1.0

    def test_client_without_dataset_skips_gate(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient

        client = AorticaFlowerClient()
        assert client.check_data_quality() is None
        assert client._data_quality_metrics() == {}
