"""Tests for the WFDB format reader (``aortica.io.wfdb_reader``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wfdb

from aortica.io.ecg_record import ECGRecord
from aortica.io.wfdb_reader import read_wfdb

# ──────────────────────────────────────────────────────────────────
# Fixtures — synthetic WFDB records written to a temp directory
# ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def single_segment_record(tmp_path: Path) -> Path:
    """Create a minimal single-segment WFDB record (2 leads, 1 s @ 500 Hz)."""
    n_leads = 2
    fs = 500
    duration_s = 1
    n_samples = fs * duration_s
    rng = np.random.default_rng(42)
    # Signals in mV
    signals = rng.standard_normal((n_samples, n_leads)) * 0.5

    record_name = "synth_single"
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV", "mV"],
        sig_name=["II", "V5"],
        p_signal=signals,
        write_dir=str(tmp_path),
    )
    return tmp_path / record_name


@pytest.fixture()
def three_lead_record(tmp_path: Path) -> Path:
    """Create a 3-lead record with mixed units for normalisation tests."""
    fs = 250
    n_samples = 500
    rng = np.random.default_rng(99)
    signals = rng.standard_normal((n_samples, 3)) * 1.0

    record_name = "synth_three"
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV", "mV", "mV"],
        sig_name=["I", "II", "III"],
        p_signal=signals,
        write_dir=str(tmp_path),
    )
    return tmp_path / record_name


# ──────────────────────────────────────────────────────────────────
# Tests — core functionality
# ──────────────────────────────────────────────────────────────────


class TestReadWfdbBasic:
    """Basic reading and ECGRecord construction."""

    def test_returns_ecg_record(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert isinstance(rec, ECGRecord)

    def test_signal_shape(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        # [leads, samples]
        assert rec.signals.shape == (2, 500)

    def test_sample_rate(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.sample_rate == 500.0

    def test_lead_names(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.lead_names == ["II", "V5"]

    def test_source_format(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.source_format == "wfdb"

    def test_units_normalised_to_uv(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.units == "µV"

    def test_duration_seconds(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.duration_seconds == pytest.approx(1.0, abs=0.01)

    def test_signals_dtype(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.signals.dtype == np.float64

    def test_three_lead_shape(self, three_lead_record: Path) -> None:
        rec = read_wfdb(three_lead_record)
        assert rec.signals.shape == (3, 500)
        assert rec.lead_names == ["I", "II", "III"]


class TestReadWfdbUnitConversion:
    """Unit normalisation from mV to µV."""

    def test_mv_signals_converted_to_uv(self, single_segment_record: Path) -> None:
        """WFDB record stored in mV should be scaled by 1e3 to µV."""
        # Read the raw wfdb record to get original mV values
        raw = wfdb.rdrecord(str(single_segment_record), physical=True)
        raw_mv = raw.p_signal.T  # [leads, samples]

        rec = read_wfdb(single_segment_record)
        # Our reader converts mV → µV (× 1000)
        np.testing.assert_allclose(rec.signals, raw_mv * 1e3, rtol=1e-5)


class TestReadWfdbEdgeCases:
    """Edge cases: file-not-found, extension stripping, etc."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_wfdb(tmp_path / "nonexistent")

    def test_accepts_path_with_hea_extension(self, single_segment_record: Path) -> None:
        """Caller can pass path ending in .hea — we strip it."""
        rec = read_wfdb(single_segment_record.with_suffix(".hea"))
        assert isinstance(rec, ECGRecord)

    def test_accepts_path_with_dat_extension(self, single_segment_record: Path) -> None:
        """Caller can pass path ending in .dat — we strip it."""
        rec = read_wfdb(single_segment_record.with_suffix(".dat"))
        assert isinstance(rec, ECGRecord)

    def test_accepts_string_path(self, single_segment_record: Path) -> None:
        rec = read_wfdb(str(single_segment_record))
        assert isinstance(rec, ECGRecord)

    def test_metadata_contains_record_name(self, single_segment_record: Path) -> None:
        rec = read_wfdb(single_segment_record)
        assert rec.patient_metadata is not None
        assert rec.patient_metadata["record_name"] == "synth_single"


# ──────────────────────────────────────────────────────────────────
# Tests — real PhysioNet WFDB files
# ──────────────────────────────────────────────────────────────────


class TestReadWfdbPhysioNet:
    """Integration tests using real PhysioNet records.

    These tests download small records from PhysioNet and are marked
    ``@pytest.mark.slow`` so they can be skipped during quick local runs.
    """

    @pytest.mark.slow
    def test_mitbih_record_100(self, tmp_path: Path) -> None:
        """Read MIT-BIH Arrhythmia record 100 (2 leads, 360 Hz)."""
        # Download record 100 from MIT-BIH via wfdb
        wfdb.dl_database("mitdb", str(tmp_path), records=["100"])
        rec = read_wfdb(tmp_path / "100")

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 2
        assert rec.sample_rate == 360.0
        # MIT-BIH 100 has leads MLII and V5
        assert "MLII" in rec.lead_names
        assert "V5" in rec.lead_names
        assert rec.source_format == "wfdb"
        assert rec.units == "µV"
        # Should have ~30 minutes of data at 360 Hz
        assert rec.num_samples > 0

    @pytest.mark.slow
    def test_ptbxl_sample(self, tmp_path: Path) -> None:
        """Read a PTB-XL record (12 leads, 500 Hz)."""
        # PTB-XL record 00001 from the ptb-xl database
        wfdb.dl_database(
            "ptb-xl/records500/00000",
            str(tmp_path),
            records=["00001_hr"],
        )
        rec = read_wfdb(tmp_path / "00001_hr")

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 12
        assert rec.sample_rate == 500.0
        assert rec.source_format == "wfdb"
        assert rec.units == "µV"

    @pytest.mark.slow
    def test_mitbih_record_100_ecg_record_methods(self, tmp_path: Path) -> None:
        """Verify ECGRecord utility methods work on a real WFDB record."""
        wfdb.dl_database("mitdb", str(tmp_path), records=["100"])
        rec = read_wfdb(tmp_path / "100")

        # Resample to 500 Hz
        resampled = rec.resample(500.0)
        assert resampled.sample_rate == 500.0
        assert resampled.num_leads == rec.num_leads

        # Select a single lead
        single = rec.select_leads(["MLII"])
        assert single.num_leads == 1
        assert single.lead_names == ["MLII"]
