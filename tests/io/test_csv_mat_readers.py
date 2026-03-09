"""Tests for CSV and MAT format readers.

All tests use synthetic data created via temporary files — no external
downloads are required.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy import io as sio

from aortica.io.csv_reader import CSVConfig, read_csv
from aortica.io.ecg_record import ECGRecord
from aortica.io.mat_reader import MATConfig, read_mat


# =====================================================================
# CSV Reader Tests
# =====================================================================


class TestReadCSVColumnPerLead:
    """CSV files where each column is a lead (default orientation)."""

    def test_basic_column_per_lead(self, tmp_path: Path) -> None:
        """Load a simple 3-lead CSV with column-per-lead layout."""
        csv_file = tmp_path / "ecg.csv"
        csv_file.write_text(
            textwrap.dedent("""\
            Lead_I,Lead_II,Lead_III
            1.0,2.0,3.0
            4.0,5.0,6.0
            7.0,8.0,9.0
            10.0,11.0,12.0
            """)
        )
        config = CSVConfig(sample_rate=500.0)
        rec = read_csv(csv_file, config)

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 3
        assert rec.num_samples == 4
        assert rec.sample_rate == 500.0
        assert rec.lead_names == ["Lead_I", "Lead_II", "Lead_III"]
        assert rec.source_format == "csv"
        assert rec.units == "µV"
        np.testing.assert_array_equal(rec.signals[0], [1.0, 4.0, 7.0, 10.0])
        np.testing.assert_array_equal(rec.signals[1], [2.0, 5.0, 8.0, 11.0])

    def test_with_time_column(self, tmp_path: Path) -> None:
        """Time column is excluded from signal data."""
        csv_file = tmp_path / "ecg_time.csv"
        csv_file.write_text(
            textwrap.dedent("""\
            time,I,II
            0.0,100.0,200.0
            0.002,110.0,210.0
            0.004,120.0,220.0
            """)
        )
        config = CSVConfig(sample_rate=500.0, time_column="time")
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 2
        assert rec.num_samples == 3
        assert rec.lead_names == ["I", "II"]
        np.testing.assert_array_equal(rec.signals[0], [100.0, 110.0, 120.0])

    def test_select_lead_columns(self, tmp_path: Path) -> None:
        """Only selected lead columns are loaded, in the specified order."""
        csv_file = tmp_path / "ecg_select.csv"
        csv_file.write_text(
            textwrap.dedent("""\
            I,II,III,aVR
            1.0,2.0,3.0,4.0
            5.0,6.0,7.0,8.0
            """)
        )
        config = CSVConfig(sample_rate=250.0, lead_columns=["III", "I"])
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 2
        assert rec.lead_names == ["III", "I"]
        np.testing.assert_array_equal(rec.signals[0], [3.0, 7.0])
        np.testing.assert_array_equal(rec.signals[1], [1.0, 5.0])

    def test_custom_delimiter(self, tmp_path: Path) -> None:
        """Tab-separated CSV is handled with custom delimiter."""
        csv_file = tmp_path / "ecg.tsv"
        csv_file.write_text("A\tB\n1.0\t2.0\n3.0\t4.0\n")
        config = CSVConfig(sample_rate=100.0, delimiter="\t")
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 2
        assert rec.lead_names == ["A", "B"]

    def test_skip_rows(self, tmp_path: Path) -> None:
        """Extra header lines are skipped."""
        csv_file = tmp_path / "ecg_extra.csv"
        csv_file.write_text("# comment line\n# another\nX,Y\n1,2\n3,4\n")
        config = CSVConfig(sample_rate=100.0, skip_rows=2)
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 2
        assert rec.num_samples == 2

    def test_custom_units(self, tmp_path: Path) -> None:
        """Custom units are preserved."""
        csv_file = tmp_path / "ecg_mv.csv"
        csv_file.write_text("V1\n0.5\n0.6\n")
        config = CSVConfig(sample_rate=500.0, units="mV")
        rec = read_csv(csv_file, config)

        assert rec.units == "mV"

    def test_patient_metadata(self, tmp_path: Path) -> None:
        """Patient metadata is passed through."""
        csv_file = tmp_path / "ecg_meta.csv"
        csv_file.write_text("V1\n1.0\n2.0\n")
        meta = {"patient_id": "P001", "age": 55}
        config = CSVConfig(sample_rate=500.0, patient_metadata=meta)
        rec = read_csv(csv_file, config)

        assert rec.patient_metadata == meta


class TestReadCSVRowPerLead:
    """CSV files where each row is a lead."""

    def test_basic_row_per_lead(self, tmp_path: Path) -> None:
        """Load a 2-lead CSV with row-per-lead layout."""
        csv_file = tmp_path / "ecg_row.csv"
        csv_file.write_text(
            textwrap.dedent("""\
            s0,s1,s2,s3
            1.0,2.0,3.0,4.0
            5.0,6.0,7.0,8.0
            """)
        )
        config = CSVConfig(
            sample_rate=250.0,
            orientation="row_per_lead",
            lead_columns=["I", "II"],
        )
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 2
        assert rec.num_samples == 4
        assert rec.lead_names == ["I", "II"]
        np.testing.assert_array_equal(rec.signals[0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(rec.signals[1], [5.0, 6.0, 7.0, 8.0])

    def test_row_per_lead_auto_names(self, tmp_path: Path) -> None:
        """Without lead_columns, generic names are generated."""
        csv_file = tmp_path / "ecg_row_auto.csv"
        csv_file.write_text("s0,s1\n10,20\n30,40\n50,60\n")
        config = CSVConfig(sample_rate=100.0, orientation="row_per_lead")
        rec = read_csv(csv_file, config)

        assert rec.num_leads == 3
        assert rec.lead_names == ["Lead_0", "Lead_1", "Lead_2"]


class TestReadCSVErrors:
    """Error handling for the CSV reader."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        config = CSVConfig(sample_rate=500.0)
        with pytest.raises(FileNotFoundError):
            read_csv(tmp_path / "nonexistent.csv", config)

    def test_missing_time_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("A,B\n1,2\n")
        config = CSVConfig(sample_rate=500.0, time_column="missing")
        with pytest.raises(ValueError, match="time_column"):
            read_csv(csv_file, config)

    def test_missing_lead_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "bad2.csv"
        csv_file.write_text("A,B\n1,2\n")
        config = CSVConfig(sample_rate=500.0, lead_columns=["A", "MISSING"])
        with pytest.raises(ValueError, match="lead_column"):
            read_csv(csv_file, config)

    def test_row_per_lead_wrong_count(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "bad3.csv"
        csv_file.write_text("s0,s1\n1,2\n3,4\n")
        config = CSVConfig(
            sample_rate=100.0,
            orientation="row_per_lead",
            lead_columns=["I", "II", "III"],  # 3 names but only 2 rows
        )
        with pytest.raises(ValueError, match="lead_columns length"):
            read_csv(csv_file, config)


# =====================================================================
# MAT Reader Tests
# =====================================================================


class TestReadMATBasic:
    """Basic MAT file reading."""

    def test_leads_first(self, tmp_path: Path) -> None:
        """Load signals with [leads, samples] orientation."""
        mat_file = tmp_path / "ecg.mat"
        signals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        sio.savemat(str(mat_file), {"ecg": signals})

        config = MATConfig(sample_rate=500.0, signal_variable="ecg")
        rec = read_mat(mat_file, config)

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 2
        assert rec.num_samples == 3
        assert rec.sample_rate == 500.0
        assert rec.source_format == "mat"
        assert rec.units == "µV"
        np.testing.assert_array_almost_equal(rec.signals, signals)

    def test_samples_first(self, tmp_path: Path) -> None:
        """Load signals with [samples, leads] orientation (transposed)."""
        mat_file = tmp_path / "ecg_t.mat"
        # 4 samples × 2 leads
        signals_st = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
            dtype=np.float64,
        )
        sio.savemat(str(mat_file), {"data": signals_st})

        config = MATConfig(
            sample_rate=250.0,
            signal_variable="data",
            orientation="samples_first",
        )
        rec = read_mat(mat_file, config)

        assert rec.num_leads == 2
        assert rec.num_samples == 4
        np.testing.assert_array_almost_equal(rec.signals, signals_st.T)

    def test_single_lead(self, tmp_path: Path) -> None:
        """1-D signal array is promoted to [1, samples]."""
        mat_file = tmp_path / "single.mat"
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sio.savemat(str(mat_file), {"sig": sig})

        config = MATConfig(sample_rate=100.0, signal_variable="sig")
        rec = read_mat(mat_file, config)

        assert rec.num_leads == 1
        assert rec.num_samples == 5

    def test_custom_units(self, tmp_path: Path) -> None:
        mat_file = tmp_path / "ecg_mv.mat"
        sio.savemat(str(mat_file), {"s": np.ones((2, 10))})
        config = MATConfig(sample_rate=500.0, signal_variable="s", units="mV")
        rec = read_mat(mat_file, config)
        assert rec.units == "mV"


class TestReadMATMetadata:
    """MAT files with embedded metadata variables."""

    def test_lead_names_variable(self, tmp_path: Path) -> None:
        """Lead names are read from a separate variable."""
        mat_file = tmp_path / "ecg_names.mat"
        signals = np.random.randn(3, 100)
        lead_names = np.array(["I", "II", "III"], dtype=object)
        sio.savemat(str(mat_file), {"ecg": signals, "leads": lead_names})

        config = MATConfig(
            sample_rate=500.0,
            signal_variable="ecg",
            lead_names_variable="leads",
        )
        rec = read_mat(mat_file, config)

        assert rec.lead_names == ["I", "II", "III"]

    def test_sample_rate_variable(self, tmp_path: Path) -> None:
        """Sample rate overridden by a variable in the MAT file."""
        mat_file = tmp_path / "ecg_fs.mat"
        sio.savemat(str(mat_file), {"ecg": np.ones((2, 50)), "fs": np.array([1000.0])})

        config = MATConfig(
            sample_rate=500.0,  # will be overridden
            signal_variable="ecg",
            sample_rate_variable="fs",
        )
        rec = read_mat(mat_file, config)

        assert rec.sample_rate == 1000.0

    def test_missing_lead_names_variable(self, tmp_path: Path) -> None:
        """If the lead names variable isn't in the file, generic names are used."""
        mat_file = tmp_path / "ecg_no_names.mat"
        sio.savemat(str(mat_file), {"ecg": np.ones((4, 20))})

        config = MATConfig(
            sample_rate=500.0,
            signal_variable="ecg",
            lead_names_variable="does_not_exist",
        )
        rec = read_mat(mat_file, config)

        assert rec.lead_names == ["Lead_0", "Lead_1", "Lead_2", "Lead_3"]

    def test_patient_metadata(self, tmp_path: Path) -> None:
        mat_file = tmp_path / "ecg_pm.mat"
        sio.savemat(str(mat_file), {"s": np.ones((1, 10))})
        meta = {"id": "X123"}
        config = MATConfig(sample_rate=500.0, signal_variable="s", patient_metadata=meta)
        rec = read_mat(mat_file, config)
        assert rec.patient_metadata == meta


class TestReadMATErrors:
    """Error handling for the MAT reader."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        config = MATConfig(sample_rate=500.0, signal_variable="ecg")
        with pytest.raises(FileNotFoundError):
            read_mat(tmp_path / "missing.mat", config)

    def test_missing_variable(self, tmp_path: Path) -> None:
        mat_file = tmp_path / "ecg_bad.mat"
        sio.savemat(str(mat_file), {"other": np.ones((2, 10))})
        config = MATConfig(sample_rate=500.0, signal_variable="ecg")
        with pytest.raises(KeyError, match="ecg"):
            read_mat(mat_file, config)

    def test_3d_signal_raises(self, tmp_path: Path) -> None:
        mat_file = tmp_path / "ecg_3d.mat"
        sio.savemat(str(mat_file), {"ecg": np.ones((2, 3, 4))})
        config = MATConfig(sample_rate=500.0, signal_variable="ecg")
        with pytest.raises(ValueError, match="2-D"):
            read_mat(mat_file, config)


# =====================================================================
# Integration: round-trip through ECGRecord
# =====================================================================


class TestReadersProduceValidECGRecord:
    """Both readers produce records with working ECGRecord methods."""

    def test_csv_record_resample(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "rt.csv"
        csv_file.write_text("I,II\n" + "\n".join(f"{i},{i * 2}" for i in range(100)))
        rec = read_csv(csv_file, CSVConfig(sample_rate=500.0))
        resampled = rec.resample(250.0)
        assert resampled.sample_rate == 250.0
        assert resampled.num_samples == 50

    def test_csv_record_select_leads(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "sl.csv"
        csv_file.write_text("A,B,C\n1,2,3\n4,5,6\n")
        rec = read_csv(csv_file, CSVConfig(sample_rate=100.0))
        sub = rec.select_leads(["C", "A"])
        assert sub.lead_names == ["C", "A"]

    def test_mat_record_to_millivolts(self, tmp_path: Path) -> None:
        mat_file = tmp_path / "mv.mat"
        sio.savemat(str(mat_file), {"s": np.ones((2, 50)) * 1000.0})
        rec = read_mat(mat_file, MATConfig(sample_rate=500.0, signal_variable="s"))
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_array_almost_equal(mv.signals, np.ones((2, 50)))
