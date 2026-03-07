"""Tests for aortica.io.ecg_record — ECGRecord dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_record(
    n_leads: int = 3,
    n_samples: int = 1000,
    sample_rate: float = 500.0,
    units: str = "µV",
    lead_names: list[str] | None = None,
    **kwargs: object,
) -> ECGRecord:
    """Helper to build a minimal ECGRecord for tests."""
    if lead_names is None:
        lead_names = [f"L{i}" for i in range(n_leads)]
    signals = np.random.default_rng(42).standard_normal((n_leads, n_samples))
    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        units=units,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------

class TestECGRecordConstruction:
    """Basic construction and automatic field derivation."""

    def test_basic_construction(self) -> None:
        rec = _make_record()
        assert rec.num_leads == 3
        assert rec.num_samples == 1000
        assert rec.sample_rate == 500.0
        assert rec.units == "µV"
        assert rec.source_format == "unknown"
        assert rec.patient_metadata is None

    def test_duration_computed(self) -> None:
        rec = _make_record(n_samples=2500, sample_rate=500.0)
        assert rec.duration_seconds == pytest.approx(5.0)

    def test_duration_explicit(self) -> None:
        rec = _make_record(n_samples=2500, sample_rate=500.0, duration_seconds=10.0)
        assert rec.duration_seconds == pytest.approx(10.0)

    def test_patient_metadata(self) -> None:
        meta = {"age": 55, "sex": "M"}
        rec = _make_record(patient_metadata=meta)
        assert rec.patient_metadata == meta

    def test_source_format(self) -> None:
        rec = _make_record(source_format="wfdb")
        assert rec.source_format == "wfdb"


class TestECGRecordValidation:
    """Validation errors raised during __post_init__."""

    def test_negative_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            _make_record(sample_rate=-1.0)

    def test_zero_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            _make_record(sample_rate=0.0)

    def test_1d_signals_rejected(self) -> None:
        with pytest.raises(ValueError, match="signals must be 2D"):
            ECGRecord(
                signals=np.zeros(100),
                sample_rate=500.0,
                lead_names=["I"],
            )

    def test_3d_signals_rejected(self) -> None:
        with pytest.raises(ValueError, match="signals must be 2D"):
            ECGRecord(
                signals=np.zeros((2, 3, 4)),
                sample_rate=500.0,
                lead_names=["I", "II"],
            )

    def test_lead_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="lead_names length"):
            ECGRecord(
                signals=np.zeros((3, 100)),
                sample_rate=500.0,
                lead_names=["I", "II"],
            )


# ---------------------------------------------------------------
# Properties
# ---------------------------------------------------------------

class TestECGRecordProperties:
    def test_num_leads(self) -> None:
        rec = _make_record(n_leads=12)
        assert rec.num_leads == 12

    def test_num_samples(self) -> None:
        rec = _make_record(n_samples=5000)
        assert rec.num_samples == 5000


# ---------------------------------------------------------------
# resample
# ---------------------------------------------------------------

class TestResample:
    def test_upsample(self) -> None:
        rec = _make_record(n_samples=500, sample_rate=250.0)
        up = rec.resample(500.0)
        assert up.sample_rate == 500.0
        assert up.num_samples == 1000
        assert up.num_leads == rec.num_leads

    def test_downsample(self) -> None:
        rec = _make_record(n_samples=1000, sample_rate=500.0)
        down = rec.resample(250.0)
        assert down.sample_rate == 250.0
        assert down.num_samples == 500
        assert down.num_leads == rec.num_leads

    def test_same_rate_copies(self) -> None:
        rec = _make_record()
        same = rec.resample(rec.sample_rate)
        assert same.sample_rate == rec.sample_rate
        assert same.num_samples == rec.num_samples
        np.testing.assert_array_equal(same.signals, rec.signals)
        # Verify it is a copy, not the same object
        assert same.signals is not rec.signals

    def test_preserves_metadata(self) -> None:
        rec = _make_record(patient_metadata={"age": 42}, source_format="csv")
        up = rec.resample(1000.0)
        assert up.patient_metadata == {"age": 42}
        assert up.source_format == "csv"
        assert up.units == rec.units

    def test_invalid_target_hz(self) -> None:
        rec = _make_record()
        with pytest.raises(ValueError, match="target_hz must be positive"):
            rec.resample(0)
        with pytest.raises(ValueError, match="target_hz must be positive"):
            rec.resample(-100)

    def test_duration_recomputed(self) -> None:
        rec = _make_record(n_samples=500, sample_rate=500.0)
        assert rec.duration_seconds == pytest.approx(1.0)
        up = rec.resample(1000.0)
        # duration should remain ~1 second
        assert up.duration_seconds == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------
# select_leads
# ---------------------------------------------------------------

class TestSelectLeads:
    def test_select_subset(self) -> None:
        rec = _make_record(
            n_leads=4,
            lead_names=["I", "II", "III", "aVR"],
        )
        sub = rec.select_leads(["II", "aVR"])
        assert sub.num_leads == 2
        assert sub.lead_names == ["II", "aVR"]
        np.testing.assert_array_equal(sub.signals[0], rec.signals[1])
        np.testing.assert_array_equal(sub.signals[1], rec.signals[3])

    def test_reorder_leads(self) -> None:
        rec = _make_record(n_leads=3, lead_names=["A", "B", "C"])
        reordered = rec.select_leads(["C", "A"])
        assert reordered.lead_names == ["C", "A"]
        np.testing.assert_array_equal(reordered.signals[0], rec.signals[2])
        np.testing.assert_array_equal(reordered.signals[1], rec.signals[0])

    def test_single_lead(self) -> None:
        rec = _make_record(n_leads=3, lead_names=["X", "Y", "Z"])
        single = rec.select_leads(["Y"])
        assert single.num_leads == 1
        assert single.lead_names == ["Y"]

    def test_missing_lead_raises(self) -> None:
        rec = _make_record(n_leads=2, lead_names=["I", "II"])
        with pytest.raises(ValueError, match="Lead 'V1' not found"):
            rec.select_leads(["I", "V1"])

    def test_preserves_metadata(self) -> None:
        rec = _make_record(
            n_leads=2,
            lead_names=["I", "II"],
            patient_metadata={"id": "P1"},
            source_format="mat",
        )
        sub = rec.select_leads(["I"])
        assert sub.patient_metadata == {"id": "P1"}
        assert sub.source_format == "mat"
        assert sub.sample_rate == rec.sample_rate
        assert sub.duration_seconds == rec.duration_seconds

    def test_returns_copy(self) -> None:
        rec = _make_record(n_leads=2, lead_names=["A", "B"])
        sub = rec.select_leads(["A"])
        sub.signals[0, 0] = 999.0
        assert rec.signals[0, 0] != 999.0


# ---------------------------------------------------------------
# to_millivolts
# ---------------------------------------------------------------

class TestToMillivolts:
    def test_from_microvolts(self) -> None:
        rec = _make_record(units="µV")
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_allclose(mv.signals, rec.signals * 1e-3)

    def test_from_uv_ascii(self) -> None:
        rec = _make_record(units="uV")
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_allclose(mv.signals, rec.signals * 1e-3)

    def test_from_millivolts_noop(self) -> None:
        rec = _make_record(units="mV")
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_allclose(mv.signals, rec.signals)

    def test_from_volts(self) -> None:
        rec = _make_record(units="V")
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_allclose(mv.signals, rec.signals * 1e3)

    def test_unknown_units_raises(self) -> None:
        rec = _make_record(units="ADC")
        with pytest.raises(ValueError, match="Cannot convert units"):
            rec.to_millivolts()

    def test_preserves_metadata(self) -> None:
        rec = _make_record(
            units="µV",
            patient_metadata={"name": "test"},
            source_format="dicom",
        )
        mv = rec.to_millivolts()
        assert mv.patient_metadata == {"name": "test"}
        assert mv.source_format == "dicom"
        assert mv.sample_rate == rec.sample_rate
        assert mv.lead_names == rec.lead_names
