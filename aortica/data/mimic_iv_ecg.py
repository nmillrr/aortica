"""MIMIC-IV-ECG dataset loader.

Provides :func:`load_mimic_iv_ecg` for loading the MIMIC-IV-ECG dataset
(PhysioNet credentialed access) into Aortica's standard ECGRecord format,
and :func:`load_combined` for merging PTB-XL and MIMIC-IV-ECG datasets
with source tagging for cross-dataset evaluation.

MIMIC-IV-ECG uses WFDB format with its own metadata schema linked to
MIMIC-IV clinical tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from aortica.io.ecg_record import ECGRecord

# ---------------------------------------------------------------------------
# ICD code -> Aortica taxonomy mapping
# ---------------------------------------------------------------------------

# ICD-9 codes mapped to (rhythm, structural, ischaemia)
_ICD9_MAP: dict[str, tuple[int, int, int]] = {
    # Rhythm / conduction
    "427.31": (1, 0, 0),  # AF
    "427.32": (1, 0, 0),  # AFL
    "427.0": (1, 0, 0),   # PSVT
    "427.1": (1, 0, 0),   # VT
    "427.41": (1, 0, 0),  # VF
    "427.5": (1, 0, 0),   # Cardiac arrest
    "427.81": (1, 0, 0),  # SSS
    "427.89": (1, 0, 0),  # Other rhythm
    "426.0": (1, 0, 0),   # AV block complete
    "426.11": (1, 0, 0),  # 1st deg AV block
    "426.12": (1, 0, 0),  # Mobitz II
    "426.13": (1, 0, 0),  # Other 2nd deg AV block
    "426.3": (1, 0, 0),   # LBBB
    "426.4": (1, 0, 0),   # RBBB
    "426.7": (1, 0, 0),   # WPW
    "427.61": (1, 0, 0),  # SVE beats
    "427.69": (1, 0, 0),  # PVCs
    # Structural / hypertrophy
    "429.3": (0, 1, 0),   # Cardiomegaly
    "402.90": (0, 1, 0),  # Hypertensive heart disease
    "402.91": (0, 1, 0),  # Hypertensive heart disease w/ CHF
    "425.1": (0, 1, 0),   # HCM
    "425.4": (0, 1, 0),   # Cardiomyopathy
    "416.0": (0, 1, 0),   # Pulm HTN
    "424.0": (0, 1, 0),   # Mitral valve
    "424.1": (0, 1, 0),   # Aortic valve
    "428.0": (0, 1, 0),   # CHF
    "428.1": (0, 1, 0),   # Left heart failure
    # Ischaemia / MI
    "410": (0, 0, 1),     # AMI (prefix)
    "410.0": (0, 0, 1),
    "410.1": (0, 0, 1),
    "410.2": (0, 0, 1),
    "410.3": (0, 0, 1),
    "410.4": (0, 0, 1),
    "410.5": (0, 0, 1),
    "410.6": (0, 0, 1),
    "410.7": (0, 0, 1),
    "410.8": (0, 0, 1),
    "410.9": (0, 0, 1),
    "411": (0, 0, 1),     # Other acute IHD
    "411.1": (0, 0, 1),   # Intermediate coronary syndrome
    "412": (0, 0, 1),     # Old MI
    "413": (0, 0, 1),     # Angina pectoris
    "414": (0, 0, 1),     # Chronic IHD
    "414.0": (0, 0, 1),
    "414.01": (0, 0, 1),
}

# ICD-10 codes mapped to (rhythm, structural, ischaemia)
_ICD10_MAP: dict[str, tuple[int, int, int]] = {
    # Rhythm
    "I48": (1, 0, 0),     # AF/AFL
    "I48.0": (1, 0, 0),
    "I48.1": (1, 0, 0),
    "I48.2": (1, 0, 0),
    "I48.91": (1, 0, 0),
    "I47": (1, 0, 0),     # PSVT/VT
    "I47.0": (1, 0, 0),
    "I47.1": (1, 0, 0),
    "I47.2": (1, 0, 0),
    "I49.0": (1, 0, 0),   # VF
    "I49.1": (1, 0, 0),   # SVE beats
    "I49.3": (1, 0, 0),   # PVCs
    "I44.0": (1, 0, 0),   # 1st deg AV block
    "I44.1": (1, 0, 0),   # 2nd deg AV block
    "I44.2": (1, 0, 0),   # Complete AV block
    "I44.7": (1, 0, 0),   # LBBB
    "I45.0": (1, 0, 0),   # RBBB
    "I45.6": (1, 0, 0),   # WPW
    # Structural
    "I51.7": (0, 1, 0),   # Cardiomegaly
    "I42": (0, 1, 0),     # Cardiomyopathy
    "I42.0": (0, 1, 0),   # DCM
    "I42.1": (0, 1, 0),   # HCM
    "I42.2": (0, 1, 0),   # HCM obstructive
    "I27.0": (0, 1, 0),   # Pulm HTN
    "I34": (0, 1, 0),     # Mitral valve
    "I35": (0, 1, 0),     # Aortic valve
    "I50": (0, 1, 0),     # Heart failure
    "I50.1": (0, 1, 0),
    "I50.2": (0, 1, 0),
    "I50.9": (0, 1, 0),
    "I11": (0, 1, 0),     # Hypertensive heart disease
    "I11.0": (0, 1, 0),
    "I11.9": (0, 1, 0),
    # Ischaemia
    "I21": (0, 0, 1),     # AMI
    "I21.0": (0, 0, 1),
    "I21.1": (0, 0, 1),
    "I21.2": (0, 0, 1),
    "I21.3": (0, 0, 1),
    "I21.4": (0, 0, 1),
    "I21.9": (0, 0, 1),
    "I22": (0, 0, 1),     # Subsequent MI
    "I24": (0, 0, 1),     # Other acute IHD
    "I25": (0, 0, 1),     # Chronic IHD
    "I25.1": (0, 0, 1),
    "I25.10": (0, 0, 1),
    "I25.2": (0, 0, 1),   # Old MI
    "I20": (0, 0, 1),     # Angina
    "I20.0": (0, 0, 1),
}


class MIMICDataNotFoundError(FileNotFoundError):
    """Raised when MIMIC-IV-ECG data files are not found.

    Provides download instructions referencing PhysioNet credentialing
    requirements.
    """

    _INSTRUCTIONS = (
        "MIMIC-IV-ECG data files not found at: {path}\n\n"
        "MIMIC-IV-ECG requires PhysioNet credentialed access.\n"
        "To obtain the data:\n"
        "  1. Create a PhysioNet account: https://physionet.org/register/\n"
        "  2. Complete the CITI training: "
        "https://physionet.org/settings/credentialing/\n"
        "  3. Sign the data use agreement for MIMIC-IV-ECG:\n"
        "     https://physionet.org/content/mimic-iv-ecg/\n"
        "  4. Download and extract the dataset to a local directory.\n"
        "  5. Pass the path to load_mimic_iv_ecg(path=...).\n\n"
        "Missing file: {missing}"
    )

    def __init__(self, path: Path, missing: str) -> None:
        self.data_path = path
        self.missing_file = missing
        super().__init__(
            self._INSTRUCTIONS.format(path=path, missing=missing)
        )


def _map_icd_to_taxonomy(
    icd_codes: list[str],
    icd_version: int,
) -> np.ndarray:
    """Map a list of ICD codes to Aortica's [rhythm, structural, ischaemia] taxonomy.

    Args:
        icd_codes: List of ICD-9 or ICD-10 codes.
        icd_version: 9 or 10.

    Returns:
        Label vector of shape (3,) with binary flags.
    """
    label = np.zeros(3, dtype=np.float32)
    code_map = _ICD9_MAP if icd_version == 9 else _ICD10_MAP

    for code in icd_codes:
        code_str = str(code).strip()
        if code_str in code_map:
            flags = code_map[code_str]
            label = np.maximum(label, np.array(flags, dtype=np.float32))
        else:
            # Try prefix matching (e.g., "410.11" matches "410.1" or "410")
            for prefix_len in range(len(code_str) - 1, 0, -1):
                prefix = code_str[:prefix_len]
                if prefix in code_map:
                    flags = code_map[prefix]
                    label = np.maximum(
                        label, np.array(flags, dtype=np.float32)
                    )
                    break

    return label


def _load_wfdb_record(
    record_path: str, target_rate: float
) -> ECGRecord:
    """Load a WFDB record and return an ECGRecord.

    Uses aortica's read_ecg dispatcher for consistency with the rest
    of the codebase.
    """
    from aortica.io.dispatcher import read_ecg

    return read_ecg(record_path, target_rate=target_rate)


def _validate_mimic_path(path: Path) -> None:
    """Validate that required MIMIC-IV-ECG files exist."""
    record_list = path / "record_list.csv"
    if not record_list.exists():
        raise MIMICDataNotFoundError(path, "record_list.csv")


def _split_data(
    records_df: pd.DataFrame,
    strategy: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records into train/val/test according to the chosen strategy.

    Args:
        records_df: DataFrame with at least ``study_id`` and ``subject_id`` columns.
        strategy: One of ``"random"``, ``"patient"``, ``"temporal"``.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    rng = np.random.RandomState(seed)

    if strategy == "patient":
        # Patient-level split: no patient appears in >1 split
        patient_ids = records_df["subject_id"].unique()
        rng.shuffle(patient_ids)
        n_total = len(patient_ids)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        train_patients = set(patient_ids[:n_train])
        val_patients = set(patient_ids[n_train : n_train + n_val])
        test_patients = set(patient_ids[n_train + n_val :])

        train_df = records_df[
            records_df["subject_id"].isin(train_patients)
        ].copy()
        val_df = records_df[
            records_df["subject_id"].isin(val_patients)
        ].copy()
        test_df = records_df[
            records_df["subject_id"].isin(test_patients)
        ].copy()

    elif strategy == "temporal":
        # Temporal split by admission date (or study_id as proxy)
        if "admittime" in records_df.columns:
            sorted_df = records_df.sort_values("admittime").reset_index(
                drop=True
            )
        else:
            # Fall back to study_id ordering
            sorted_df = records_df.sort_values("study_id").reset_index(
                drop=True
            )
        n = len(sorted_df)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_df = sorted_df.iloc[:n_train].copy()
        val_df = sorted_df.iloc[n_train : n_train + n_val].copy()
        test_df = sorted_df.iloc[n_train + n_val :].copy()

    else:
        # Random split (default)
        shuffled = records_df.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_df = shuffled.iloc[:n_train].copy()
        val_df = shuffled.iloc[n_train : n_train + n_val].copy()
        test_df = shuffled.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df


def load_mimic_iv_ecg(
    path: str | Path,
    sampling_rate: int = 500,
    split_strategy: Literal["random", "patient", "temporal"] = "random",
    diagnoses_path: Optional[str | Path] = None,
    seed: int = 42,
) -> Tuple[
    Tuple[List[ECGRecord], np.ndarray],
    Tuple[List[ECGRecord], np.ndarray],
    Tuple[List[ECGRecord], np.ndarray],
]:
    """Load the MIMIC-IV-ECG dataset into train, validation, and test splits.

    Args:
        path: Path to the MIMIC-IV-ECG dataset directory containing
            ``record_list.csv`` and WFDB record subdirectories.
        sampling_rate: Target sampling rate in Hz (default 500).
        split_strategy: Split method — ``"random"`` (default),
            ``"patient"`` (no patient in both train and test), or
            ``"temporal"`` (by admission date / study_id order).
        diagnoses_path: Optional path to a ``diagnoses_icd.csv`` file from
            MIMIC-IV clinical tables. When provided, ICD codes are mapped to
            Aortica's label taxonomy. When ``None``, labels default to
            ``[0, 0, 0]`` (unlabelled).
        seed: Random seed for split reproducibility.

    Returns:
        A tuple of ``(train, val, test)`` where each is a tuple of
        ``(records, labels)``.  Records are lists of :class:`ECGRecord`
        objects.  Labels are numpy arrays of shape ``(N, 3)`` for
        ``[rhythm, structural, ischaemia]``.

    Raises:
        MIMICDataNotFoundError: If required data files are missing, with
            download instructions referencing PhysioNet credentialing.
    """
    path = Path(path)
    _validate_mimic_path(path)

    if split_strategy not in ("random", "patient", "temporal"):
        raise ValueError(
            f"split_strategy must be 'random', 'patient', or 'temporal', "
            f"got '{split_strategy}'"
        )

    # ------------------------------------------------------------------
    # Load record list
    # ------------------------------------------------------------------
    record_list = pd.read_csv(path / "record_list.csv")

    # Ensure required columns
    for col in ("study_id", "subject_id", "path"):
        if col not in record_list.columns:
            raise MIMICDataNotFoundError(
                path,
                f"record_list.csv (missing required column: {col})",
            )

    # ------------------------------------------------------------------
    # Load diagnoses if available
    # ------------------------------------------------------------------
    diag_map: dict[int, list[tuple[str, int]]] = {}  # subject_id -> [(code, version)]
    if diagnoses_path is not None:
        diag_path = Path(diagnoses_path)
        if diag_path.exists():
            diag_df = pd.read_csv(diag_path)
            if "icd_code" in diag_df.columns and "subject_id" in diag_df.columns:
                icd_version_col = (
                    "icd_version"
                    if "icd_version" in diag_df.columns
                    else None
                )
                for _, row in diag_df.iterrows():
                    sid = int(row["subject_id"])
                    code = str(row["icd_code"])
                    version = int(row[icd_version_col]) if icd_version_col else 9
                    diag_map.setdefault(sid, []).append((code, version))

    # ------------------------------------------------------------------
    # Load machine measurements if available
    # ------------------------------------------------------------------
    meas_path = path / "machine_measurements.csv"
    meas_map: dict[int, dict[str, object]] = {}  # study_id -> measurements
    if meas_path.exists():
        meas_df = pd.read_csv(meas_path)
        if "study_id" in meas_df.columns:
            for _, row in meas_df.iterrows():
                study_id = int(row["study_id"])
                meas_map[study_id] = row.to_dict()

    # ------------------------------------------------------------------
    # Split data
    # ------------------------------------------------------------------
    train_df, val_df, test_df = _split_data(
        record_list, strategy=split_strategy, seed=seed
    )

    # ------------------------------------------------------------------
    # Build records and labels
    # ------------------------------------------------------------------
    def _build_split(
        split_df: pd.DataFrame,
    ) -> Tuple[List[ECGRecord], np.ndarray]:
        records: List[ECGRecord] = []
        labels: list[np.ndarray] = []

        for _, row in split_df.iterrows():
            record_path_str = str(path / str(row["path"]))
            try:
                record = _load_wfdb_record(
                    record_path_str, target_rate=float(sampling_rate)
                )
            except Exception:
                # Skip unreadable records
                continue

            # Attach metadata
            meta = dict(record.patient_metadata or {})
            meta["mimic_study_id"] = int(row["study_id"])
            meta["mimic_subject_id"] = int(row["subject_id"])
            meta["source_dataset"] = "mimic_iv_ecg"

            # Merge machine measurements if available
            study_id = int(row["study_id"])
            if study_id in meas_map:
                meta["machine_measurements"] = meas_map[study_id]

            record = ECGRecord(
                signals=record.signals,
                sample_rate=record.sample_rate,
                lead_names=record.lead_names,
                duration_seconds=record.duration_seconds,
                patient_metadata=meta,
                source_format="mimic_iv_ecg",
                units=record.units,
            )
            records.append(record)

            # Build label vector
            subject_id = int(row["subject_id"])
            if subject_id in diag_map:
                label_vec = np.zeros(3, dtype=np.float32)
                for code, version in diag_map[subject_id]:
                    mapped = _map_icd_to_taxonomy([code], version)
                    label_vec = np.maximum(label_vec, mapped)
                labels.append(label_vec)
            else:
                labels.append(np.zeros(3, dtype=np.float32))

        label_array = (
            np.array(labels, dtype=np.float32)
            if labels
            else np.empty((0, 3), dtype=np.float32)
        )
        return records, label_array

    train = _build_split(train_df)
    val = _build_split(val_df)
    test = _build_split(test_df)

    return train, val, test


def load_combined(
    ptbxl_path: str | Path,
    mimic_path: str | Path,
    sampling_rate: int = 500,
    mimic_split_strategy: Literal["random", "patient", "temporal"] = "random",
    mimic_diagnoses_path: Optional[str | Path] = None,
    seed: int = 42,
) -> Tuple[
    Tuple[List[ECGRecord], np.ndarray],
    Tuple[List[ECGRecord], np.ndarray],
    Tuple[List[ECGRecord], np.ndarray],
]:
    """Merge PTB-XL and MIMIC-IV-ECG datasets with source tagging.

    Loads both datasets using their respective loaders and concatenates
    the splits.  Each :class:`ECGRecord` carries a ``source_dataset``
    key in ``patient_metadata`` (``"ptbxl"`` or ``"mimic_iv_ecg"``) for
    cross-dataset evaluation.

    Args:
        ptbxl_path: Path to the PTB-XL dataset directory.
        mimic_path: Path to the MIMIC-IV-ECG dataset directory.
        sampling_rate: Target sampling rate in Hz.
        mimic_split_strategy: Split strategy for MIMIC data.
        mimic_diagnoses_path: Optional path to MIMIC diagnoses_icd.csv.
        seed: Random seed for reproducibility.

    Returns:
        Combined ``(train, val, test)`` tuples with source tagging.
    """
    from aortica.data.ptbxl import load_ptbxl

    # Load PTB-XL
    ptb_train, ptb_val, ptb_test = load_ptbxl(
        ptbxl_path, sampling_rate=sampling_rate
    )

    # Tag PTB-XL records
    def _tag_records(
        records: List[ECGRecord], tag: str
    ) -> List[ECGRecord]:
        tagged = []
        for r in records:
            meta = dict(r.patient_metadata or {})
            meta["source_dataset"] = tag
            tagged.append(
                ECGRecord(
                    signals=r.signals,
                    sample_rate=r.sample_rate,
                    lead_names=r.lead_names,
                    duration_seconds=r.duration_seconds,
                    patient_metadata=meta,
                    source_format=r.source_format,
                    units=r.units,
                )
            )
        return tagged

    ptb_train_records = _tag_records(ptb_train[0], "ptbxl")
    ptb_val_records = _tag_records(ptb_val[0], "ptbxl")
    ptb_test_records = _tag_records(ptb_test[0], "ptbxl")

    # Load MIMIC-IV-ECG (already tagged by load_mimic_iv_ecg)
    mimic_train, mimic_val, mimic_test = load_mimic_iv_ecg(
        mimic_path,
        sampling_rate=sampling_rate,
        split_strategy=mimic_split_strategy,
        diagnoses_path=mimic_diagnoses_path,
        seed=seed,
    )

    # Merge splits
    def _merge(
        a: Tuple[List[ECGRecord], np.ndarray],
        b: Tuple[List[ECGRecord], np.ndarray],
        a_tagged: List[ECGRecord],
    ) -> Tuple[List[ECGRecord], np.ndarray]:
        merged_records = list(a_tagged) + list(b[0])
        merged_labels = np.concatenate([a[1], b[1]], axis=0)
        return merged_records, merged_labels

    return (
        _merge(ptb_train, mimic_train, ptb_train_records),
        _merge(ptb_val, mimic_val, ptb_val_records),
        _merge(ptb_test, mimic_test, ptb_test_records),
    )
