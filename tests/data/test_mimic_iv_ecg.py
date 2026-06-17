"""Tests for MIMIC-IV-ECG dataset loader (US-103)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import wfdb

from aortica.data.mimic_iv_ecg import (
    MIMICDataNotFoundError,
    _map_icd_to_taxonomy,
    _split_data,
    load_combined,
    load_mimic_iv_ecg,
)
from aortica.io.ecg_record import ECGRecord


# ---------------------------------------------------------------------------
# Fixtures: synthetic MIMIC-IV-ECG dataset
# ---------------------------------------------------------------------------

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

NUM_SUBJECTS = 10
NUM_STUDIES = 10


def _create_synthetic_mimic(base_dir: Path, num_records: int = NUM_STUDIES) -> None:
    """Create a synthetic MIMIC-IV-ECG directory structure."""

    # Create record_list.csv
    rows = []
    for i in range(num_records):
        subject_id = 10000 + i
        study_id = 50000 + i
        rec_name = f"p{subject_id:05d}/s{study_id:05d}/rec_{i:05d}"
        rows.append({
            "study_id": study_id,
            "subject_id": subject_id,
            "path": rec_name,
        })

    record_df = pd.DataFrame(rows)
    record_df.to_csv(base_dir / "record_list.csv", index=False)

    # Create WFDB records
    for _, row in record_df.iterrows():
        rec_path = base_dir / str(row["path"])
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        sig = np.random.randn(5000, 12).astype(np.float64)
        wfdb.wrsamp(
            rec_path.name,
            fs=500,
            units=["mV"] * 12,
            sig_name=LEAD_NAMES,
            p_signal=sig,
            fmt=["16"] * 12,
            write_dir=str(rec_path.parent),
        )


def _create_machine_measurements(base_dir: Path, num_records: int = NUM_STUDIES) -> None:
    """Create a synthetic machine_measurements.csv."""
    rows = []
    for i in range(num_records):
        study_id = 50000 + i
        rows.append({
            "study_id": study_id,
            "ventricular_rate": 72 + i,
            "p_onset": 10,
            "qrs_duration": 90 + i,
        })
    pd.DataFrame(rows).to_csv(base_dir / "machine_measurements.csv", index=False)


def _create_diagnoses_icd(diag_path: Path, num_subjects: int = NUM_SUBJECTS) -> None:
    """Create a synthetic diagnoses_icd.csv."""
    rows = []
    # Subject 10000: AF (rhythm)
    rows.append({"subject_id": 10000, "icd_code": "I48.0", "icd_version": 10})
    # Subject 10001: HCM (structural)
    rows.append({"subject_id": 10001, "icd_code": "I42.1", "icd_version": 10})
    # Subject 10002: AMI (ischaemia)
    rows.append({"subject_id": 10002, "icd_code": "I21.0", "icd_version": 10})
    # Subject 10003: ICD-9 AF
    rows.append({"subject_id": 10003, "icd_code": "427.31", "icd_version": 9})
    # Subject 10004: No cardiac ICD codes (should be [0,0,0])
    rows.append({"subject_id": 10004, "icd_code": "E11.9", "icd_version": 10})
    # Subjects 10005-10009: no diagnoses
    pd.DataFrame(rows).to_csv(diag_path, index=False)


# ---------------------------------------------------------------------------
# Tests: ICD mapping
# ---------------------------------------------------------------------------


class TestICDMapping:
    """Test ICD code to Aortica taxonomy mapping."""

    def test_icd10_rhythm(self) -> None:
        label = _map_icd_to_taxonomy(["I48.0"], icd_version=10)
        np.testing.assert_array_equal(label, [1, 0, 0])

    def test_icd10_structural(self) -> None:
        label = _map_icd_to_taxonomy(["I42.1"], icd_version=10)
        np.testing.assert_array_equal(label, [0, 1, 0])

    def test_icd10_ischaemia(self) -> None:
        label = _map_icd_to_taxonomy(["I21.0"], icd_version=10)
        np.testing.assert_array_equal(label, [0, 0, 1])

    def test_icd9_rhythm(self) -> None:
        label = _map_icd_to_taxonomy(["427.31"], icd_version=9)
        np.testing.assert_array_equal(label, [1, 0, 0])

    def test_icd9_ischaemia(self) -> None:
        label = _map_icd_to_taxonomy(["410.1"], icd_version=9)
        np.testing.assert_array_equal(label, [0, 0, 1])

    def test_multiple_codes_multi_class(self) -> None:
        """Multiple codes from different classes should set all."""
        label = _map_icd_to_taxonomy(
            ["I48.0", "I21.0", "I42.1"], icd_version=10
        )
        np.testing.assert_array_equal(label, [1, 1, 1])

    def test_unknown_code_returns_zeros(self) -> None:
        label = _map_icd_to_taxonomy(["Z99.99"], icd_version=10)
        np.testing.assert_array_equal(label, [0, 0, 0])

    def test_empty_codes_returns_zeros(self) -> None:
        label = _map_icd_to_taxonomy([], icd_version=10)
        np.testing.assert_array_equal(label, [0, 0, 0])

    def test_prefix_matching(self) -> None:
        """Codes like '410.11' should match prefix '410.1'."""
        label = _map_icd_to_taxonomy(["410.11"], icd_version=9)
        np.testing.assert_array_equal(label, [0, 0, 1])


# ---------------------------------------------------------------------------
# Tests: Split strategies
# ---------------------------------------------------------------------------


class TestSplitData:
    """Test data splitting strategies."""

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "study_id": list(range(n)),
            "subject_id": list(range(n)),
            "path": [f"p{i:05d}/s{i:05d}/rec_{i:05d}" for i in range(n)],
        })

    def test_random_split_sizes(self) -> None:
        df = self._make_df(20)
        train, val, test = _split_data(df, strategy="random")
        assert len(train) + len(val) + len(test) == 20
        assert len(train) == 14  # 0.7 * 20
        assert len(val) == 3     # 0.15 * 20
        assert len(test) == 3    # remainder

    def test_patient_split_no_overlap(self) -> None:
        df = self._make_df(20)
        train, val, test = _split_data(df, strategy="patient")
        train_subjects = set(train["subject_id"])
        val_subjects = set(val["subject_id"])
        test_subjects = set(test["subject_id"])
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)

    def test_temporal_split_ordering(self) -> None:
        df = self._make_df(20)
        train, val, test = _split_data(df, strategy="temporal")
        # In temporal split, train should have the lowest study_ids
        assert train["study_id"].max() <= val["study_id"].min()
        assert val["study_id"].max() <= test["study_id"].min()

    def test_split_reproducibility(self) -> None:
        df = self._make_df(20)
        t1, v1, te1 = _split_data(df, strategy="random", seed=42)
        t2, v2, te2 = _split_data(df, strategy="random", seed=42)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
        pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))

    def test_all_records_present(self) -> None:
        df = self._make_df(20)
        train, val, test = _split_data(df, strategy="random")
        all_ids = set(train["study_id"]) | set(val["study_id"]) | set(test["study_id"])
        assert all_ids == set(range(20))


# ---------------------------------------------------------------------------
# Tests: MIMICDataNotFoundError
# ---------------------------------------------------------------------------


class TestMIMICDataNotFoundError:
    """Test error handling for missing data."""

    def test_error_message_content(self) -> None:
        err = MIMICDataNotFoundError(Path("/foo/bar"), "record_list.csv")
        msg = str(err)
        assert "PhysioNet" in msg
        assert "credentialed" in msg.lower() or "credential" in msg.lower()
        assert "record_list.csv" in msg
        assert "/foo/bar" in msg

    def test_is_file_not_found_error(self) -> None:
        err = MIMICDataNotFoundError(Path("/x"), "y")
        assert isinstance(err, FileNotFoundError)


# ---------------------------------------------------------------------------
# Tests: load_mimic_iv_ecg
# ---------------------------------------------------------------------------


class TestLoadMIMICIVECG:
    """Test the main MIMIC-IV-ECG loader."""

    def test_missing_data_raises_with_instructions(self, tmp_path: Path) -> None:
        with pytest.raises(MIMICDataNotFoundError, match="PhysioNet"):
            load_mimic_iv_ecg(tmp_path)

    def test_basic_load(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, val, test = load_mimic_iv_ecg(tmp_path)
        train_records, train_labels = train
        val_records, val_labels = val
        test_records, test_labels = test

        total = len(train_records) + len(val_records) + len(test_records)
        assert total == NUM_STUDIES

        # Check label shapes
        assert train_labels.shape[1] == 3
        assert val_labels.shape[1] == 3
        assert test_labels.shape[1] == 3

    def test_records_are_ecg_records(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, _, _ = load_mimic_iv_ecg(tmp_path)
        for r in train[0]:
            assert isinstance(r, ECGRecord)
            assert r.source_format == "mimic_iv_ecg"

    def test_record_metadata_tagging(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, _, _ = load_mimic_iv_ecg(tmp_path)
        for r in train[0]:
            assert r.patient_metadata is not None
            assert r.patient_metadata["source_dataset"] == "mimic_iv_ecg"
            assert "mimic_study_id" in r.patient_metadata
            assert "mimic_subject_id" in r.patient_metadata

    def test_sampling_rate_default(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, _, _ = load_mimic_iv_ecg(tmp_path, sampling_rate=500)
        for r in train[0]:
            assert r.sample_rate == 500

    def test_signal_shape(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, _, _ = load_mimic_iv_ecg(tmp_path)
        for r in train[0]:
            assert r.signals.shape[0] == 12  # 12 leads
            assert r.signals.shape[1] > 0

    def test_labels_without_diagnoses(self, tmp_path: Path) -> None:
        """Without diagnoses_path, all labels should be [0, 0, 0]."""
        _create_synthetic_mimic(tmp_path)
        train, val, test = load_mimic_iv_ecg(tmp_path)
        for labels in [train[1], val[1], test[1]]:
            np.testing.assert_array_equal(labels, np.zeros_like(labels))

    def test_labels_with_diagnoses(self, tmp_path: Path) -> None:
        """With diagnoses, mapped subjects should have non-zero labels."""
        _create_synthetic_mimic(tmp_path)
        diag_path = tmp_path / "diagnoses_icd.csv"
        _create_diagnoses_icd(diag_path)

        train, val, test = load_mimic_iv_ecg(
            tmp_path, diagnoses_path=diag_path
        )

        # Collect all records and labels
        all_records = list(train[0]) + list(val[0]) + list(test[0])
        all_labels = np.concatenate([train[1], val[1], test[1]], axis=0)

        # Find record with subject_id 10000 (AF -> rhythm)
        for i, r in enumerate(all_records):
            sid = r.patient_metadata["mimic_subject_id"]  # type: ignore[index]
            if sid == 10000:
                assert all_labels[i][0] == 1.0  # rhythm
                break

    def test_patient_split_strategy(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, val, test = load_mimic_iv_ecg(
            tmp_path, split_strategy="patient"
        )
        # No subject should appear in multiple splits
        train_sids = {r.patient_metadata["mimic_subject_id"] for r in train[0]}  # type: ignore[index]
        val_sids = {r.patient_metadata["mimic_subject_id"] for r in val[0]}  # type: ignore[index]
        test_sids = {r.patient_metadata["mimic_subject_id"] for r in test[0]}  # type: ignore[index]
        assert train_sids.isdisjoint(val_sids)
        assert train_sids.isdisjoint(test_sids)
        assert val_sids.isdisjoint(test_sids)

    def test_temporal_split_strategy(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        train, val, test = load_mimic_iv_ecg(
            tmp_path, split_strategy="temporal"
        )
        total = len(train[0]) + len(val[0]) + len(test[0])
        assert total == NUM_STUDIES

    def test_invalid_split_strategy(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        with pytest.raises(ValueError, match="split_strategy"):
            load_mimic_iv_ecg(tmp_path, split_strategy="invalid")  # type: ignore[arg-type]

    def test_machine_measurements_attached(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        _create_machine_measurements(tmp_path)
        train, val, test = load_mimic_iv_ecg(tmp_path)
        all_records = list(train[0]) + list(val[0]) + list(test[0])
        found_meas = False
        for r in all_records:
            if "machine_measurements" in (r.patient_metadata or {}):
                found_meas = True
                meas = r.patient_metadata["machine_measurements"]  # type: ignore[index]
                assert "ventricular_rate" in meas
                break
        assert found_meas, "No records had machine_measurements attached"

    def test_reproducibility(self, tmp_path: Path) -> None:
        _create_synthetic_mimic(tmp_path)
        t1, v1, te1 = load_mimic_iv_ecg(tmp_path, seed=42)
        t2, v2, te2 = load_mimic_iv_ecg(tmp_path, seed=42)
        assert len(t1[0]) == len(t2[0])
        assert len(v1[0]) == len(v2[0])
        assert len(te1[0]) == len(te2[0])
        np.testing.assert_array_equal(t1[1], t2[1])

    def test_label_vectors_compatible_shape(self, tmp_path: Path) -> None:
        """Labels should be (N, 3) arrays compatible with PyTorch/TF."""
        _create_synthetic_mimic(tmp_path)
        train, val, test = load_mimic_iv_ecg(tmp_path)
        for split_records, split_labels in [train, val, test]:
            assert split_labels.ndim == 2
            assert split_labels.shape[1] == 3
            assert split_labels.shape[0] == len(split_records)
            assert split_labels.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: load_combined
# ---------------------------------------------------------------------------


def _create_synthetic_ptbxl(base_dir: Path) -> None:
    """Create a minimal synthetic PTB-XL dataset (reuse from test_ptbxl.py)."""
    scp_df = pd.DataFrame({
        "Unnamed: 0": ["NORM", "SR", "AFIB", "STEMI", "LVH", "LBBB"],
        "diagnostic": [1, 0, 0, 1, 1, 1],
        "form": [0, 0, 0, 0, 0, 0],
        "rhythm": [0, 1, 1, 0, 0, 0],
        "diagnostic_class": ["NORM", np.nan, np.nan, "MI", "HYP", "CD"],
        "diagnostic_subclass": ["NORM", np.nan, np.nan, "AMI", "LVH", "LBBB"],
        "Statement Category": ["Normal", "Rhythm", "Rhythm", "MI", "HYP", "Conduction"],
        "SCP-ECG Statement Description": ["Normal", "Sinus Rhythm", "AF", "STEMI", "LVH", "LBBB"],
        "AHA code": ["1", "2", "3", "4", "5", "6"],
    })
    scp_df.to_csv(base_dir / "scp_statements.csv", index=False)

    db_df = pd.DataFrame({
        "ecg_id": [1, 2, 3],
        "patient_id": [101, 102, 103],
        "strat_fold": [1, 9, 10],
        "filename_lr": [
            "records100/00000/00001_lr",
            "records100/00000/00002_lr",
            "records100/00000/00003_lr",
        ],
        "filename_hr": [
            "records500/00000/00001_hr",
            "records500/00000/00002_hr",
            "records500/00000/00003_hr",
        ],
        "scp_codes": [
            "{'SR': 100.0, 'NORM': 100.0}",
            "{'AFIB': 100.0, 'LBBB': 100.0}",
            "{'STEMI': 100.0, 'LVH': 100.0}",
        ],
    })
    db_df.to_csv(base_dir / "ptbxl_database.csv", index=False)

    for rmp100, rm500 in zip(db_df["filename_lr"], db_df["filename_hr"]):
        p100 = base_dir / rmp100
        p100.parent.mkdir(parents=True, exist_ok=True)
        sig100 = np.random.randn(1000, 12)
        wfdb.wrsamp(
            p100.name, fs=100, units=["mV"] * 12, sig_name=LEAD_NAMES,
            p_signal=sig100, fmt=["16"] * 12, write_dir=str(p100.parent),
        )
        p500 = base_dir / rm500
        p500.parent.mkdir(parents=True, exist_ok=True)
        sig500 = np.random.randn(5000, 12)
        wfdb.wrsamp(
            p500.name, fs=500, units=["mV"] * 12, sig_name=LEAD_NAMES,
            p_signal=sig500, fmt=["16"] * 12, write_dir=str(p500.parent),
        )


class TestLoadCombined:
    """Test the combined PTB-XL + MIMIC-IV-ECG loader."""

    def test_combined_merge(self, tmp_path: Path) -> None:
        ptbxl_dir = tmp_path / "ptbxl"
        ptbxl_dir.mkdir()
        _create_synthetic_ptbxl(ptbxl_dir)

        mimic_dir = tmp_path / "mimic"
        mimic_dir.mkdir()
        _create_synthetic_mimic(mimic_dir)

        train, val, test = load_combined(ptbxl_dir, mimic_dir)

        # PTB-XL has 3 records (1 train, 1 val, 1 test)
        # MIMIC has 10 records split across train/val/test
        total = len(train[0]) + len(val[0]) + len(test[0])
        assert total == 3 + NUM_STUDIES

    def test_source_tagging(self, tmp_path: Path) -> None:
        ptbxl_dir = tmp_path / "ptbxl"
        ptbxl_dir.mkdir()
        _create_synthetic_ptbxl(ptbxl_dir)

        mimic_dir = tmp_path / "mimic"
        mimic_dir.mkdir()
        _create_synthetic_mimic(mimic_dir)

        train, val, test = load_combined(ptbxl_dir, mimic_dir)

        all_records = list(train[0]) + list(val[0]) + list(test[0])
        sources = {r.patient_metadata["source_dataset"] for r in all_records}  # type: ignore[index]
        assert "ptbxl" in sources
        assert "mimic_iv_ecg" in sources

    def test_combined_label_shapes(self, tmp_path: Path) -> None:
        ptbxl_dir = tmp_path / "ptbxl"
        ptbxl_dir.mkdir()
        _create_synthetic_ptbxl(ptbxl_dir)

        mimic_dir = tmp_path / "mimic"
        mimic_dir.mkdir()
        _create_synthetic_mimic(mimic_dir)

        train, val, test = load_combined(ptbxl_dir, mimic_dir)

        for records, labels in [train, val, test]:
            assert labels.shape[0] == len(records)
            assert labels.shape[1] == 3
            assert labels.dtype == np.float32

    def test_combined_label_values_preserved(self, tmp_path: Path) -> None:
        """Labels from PTB-XL and MIMIC should be preserved in merge."""
        ptbxl_dir = tmp_path / "ptbxl"
        ptbxl_dir.mkdir()
        _create_synthetic_ptbxl(ptbxl_dir)

        mimic_dir = tmp_path / "mimic"
        mimic_dir.mkdir()
        _create_synthetic_mimic(mimic_dir)

        train, val, test = load_combined(ptbxl_dir, mimic_dir)

        # Verify that PTB-XL records exist in the combined output
        all_records = list(train[0]) + list(val[0]) + list(test[0])
        all_labels = np.concatenate([train[1], val[1], test[1]], axis=0)

        ptbxl_found = False
        mimic_found = False
        for i, r in enumerate(all_records):
            src = r.patient_metadata.get("source_dataset")  # type: ignore[union-attr]
            if src == "ptbxl":
                ptbxl_found = True
                # All PTB-XL labels should be valid 3-vectors
                assert all_labels[i].shape == (3,)
            elif src == "mimic_iv_ecg":
                mimic_found = True

        assert ptbxl_found, "No PTB-XL records found in combined output"
        assert mimic_found, "No MIMIC records found in combined output"


# ---------------------------------------------------------------------------
# Tests: Module imports and interface
# ---------------------------------------------------------------------------


class TestModuleInterface:
    """Test that public API is accessible from aortica.data."""

    def test_import_load_mimic_iv_ecg(self) -> None:
        from aortica.data import load_mimic_iv_ecg as fn
        assert callable(fn)

    def test_import_load_combined(self) -> None:
        from aortica.data import load_combined as fn
        assert callable(fn)

    def test_import_error_class(self) -> None:
        from aortica.data import MIMICDataNotFoundError as cls
        assert issubclass(cls, FileNotFoundError)
