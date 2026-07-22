"""Tests for `aortica federated validate-data` CLI (US-115)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")
from click.testing import CliRunner  # noqa: E402

from aortica.cli.federated_cmd import federated_group  # noqa: E402
from aortica.federated.data_quality import _TASK_NUM_OUTPUTS  # noqa: E402

_LABEL_WIDTH = sum(_TASK_NUM_OUTPUTS.values())


def _clean_signal(n: int = 500, fs: float = 250.0, seed: int = 0) -> np.ndarray:
    """Two-lead clean ECG-like signal, shape [samples, leads]."""
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2))
    for lead in range(2):
        sig = np.zeros(n)
        peak = 0.0
        while peak < n / fs:
            sig += np.exp(-0.5 * ((t - peak) / 0.02) ** 2) * 1.0
            peak += 60.0 / 72.0
        out[:, lead] = sig + rng.normal(0, 0.005, n)
    return out


def _write_site(tmp: Path, num_records: int) -> Path:
    data_dir = tmp / "ecgs"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_records):
        wfdb.wrsamp(
            f"rec{i}",
            fs=250,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            p_signal=_clean_signal(seed=i),
            write_dir=str(data_dir),
        )
    return data_dir


def _write_labels(tmp: Path, num_records: int) -> Path:
    labels = np.zeros((num_records, _LABEL_WIDTH), dtype=np.float64)
    # Positive in each of the four superclass blocks.
    offset = 0
    for task in ["rhythm", "structural", "ischaemia", "risk"]:
        labels[:, offset] = 1.0
        offset += _TASK_NUM_OUTPUTS[task]
    path = tmp / "labels.npy"
    np.save(path, labels)
    return path


def test_validate_data_json_passes(tmp_path: Path) -> None:
    data_dir = _write_site(tmp_path, 60)
    labels = _write_labels(tmp_path, 60)
    result = CliRunner().invoke(
        federated_group,
        [
            "validate-data",
            str(data_dir),
            "--labels",
            str(labels),
            "--min-samples",
            "50",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["passed"] is True
    assert report["statistics"]["num_ecgs"] == 60


def test_validate_data_blocks_small_site(tmp_path: Path) -> None:
    data_dir = _write_site(tmp_path, 5)
    labels = _write_labels(tmp_path, 5)
    result = CliRunner().invoke(
        federated_group,
        [
            "validate-data",
            str(data_dir),
            "--labels",
            str(labels),
            "--min-samples",
            "50",
            "--format",
            "json",
        ],
    )
    # Too few samples → blocking failure → non-zero exit.
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["blocking"] is True


def test_validate_data_without_labels(tmp_path: Path) -> None:
    data_dir = _write_site(tmp_path, 60)
    result = CliRunner().invoke(
        federated_group,
        ["validate-data", str(data_dir), "--min-samples", "50", "--format", "json"],
    )
    # No labels supplied → completeness/diversity fail (blocking).
    assert result.exit_code == 1
    report = json.loads(result.output)
    completeness = next(
        c for c in report["checks"] if c["name"] == "label_completeness"
    )
    assert completeness["passed"] is False
