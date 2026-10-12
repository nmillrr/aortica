"""Tests for the Georgia external-validation loader and label map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aortica.data.chapman import build_labels
from aortica.data.georgia import GeorgiaDataNotFoundError, load_georgia
from aortica.data.label_mapping import load_label_map
from aortica.data.ptbxl import _output_columns

NAMES, INDEX = _output_columns(True)


def _labels(codes: list[str]) -> dict[str, float]:
    row = build_labels([codes], multitask=True, label_map=load_label_map("georgia_snomed"))[0]
    return {n: row[i] for n, i in INDEX.items() if row[i] > 0}


def test_label_map_loads_and_is_evaluation_only() -> None:
    m = load_label_map("georgia_snomed")
    assert m.split_policy["fractions"] == {"train": 0.0, "val": 0.0, "test": 1.0}
    assert len(m.labelable_outputs) == 25


@pytest.mark.parametrize(
    "code, output",
    [
        ("713427006", "RBBB"),      # complete RBBB scored as RBBB by the challenge
        ("733534002", "LBBB"),      # complete LBBB
        ("63593006", "PAC"),        # supraventricular premature beats
        ("427172004", "PVC"),       # premature ventricular contractions
        ("17338001", "PVC"),        # ventricular premature beats
        ("253352002", "LA_enlargement"),   # left atrial abnormality
        ("253339007", "RA_enlargement"),   # right atrial abnormality
        ("10370003", "pacemaker_rhythm"),
    ],
)
def test_challenge_equivalences_are_mapped(code: str, output: str) -> None:
    assert output in _labels([code])


@pytest.mark.parametrize("code", ["713426002", "251120003", "57054005", "164931005"])
def test_deliberately_unmapped_codes(code: str) -> None:
    # Incomplete BBB, acute MI, ST elevation: not equivalent to an output.
    assert _labels([code]) == {}


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(GeorgiaDataNotFoundError):
        load_georgia(tmp_path)


def test_loads_wfdb_record_with_dx_header(tmp_path: Path) -> None:
    wfdb = pytest.importorskip("wfdb")
    folder = tmp_path / "g1"
    folder.mkdir()
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = np.arange(5000) / 500.0
    sig = np.stack([np.sin(2 * np.pi * 1.2 * t) * (0.5 + 0.05 * i) for i in range(12)], axis=1)
    wfdb.wrsamp("E99999", fs=500, units=["mV"] * 12, sig_name=leads, p_signal=sig,
                fmt=["16"] * 12, write_dir=str(folder),
                comments=["Age: 71", "Sex: Female", "Dx: 59118001,426783006"])
    records, labels, dropped = load_georgia(tmp_path, sampling_rate=100)
    assert len(records) == 1 and sum(len(v) for v in dropped.values()) == 0
    rec = records[0]
    assert rec.signals.shape == (12, 1000) and rec.sample_rate == 100.0
    assert rec.patient_metadata["sex"] == "female" and rec.patient_metadata["age"] == 71
    assert labels[0, INDEX["RBBB"]] == 1 and labels[0, INDEX["normal_sinus_rhythm"]] == 1
    assert labels.sum() == 2
