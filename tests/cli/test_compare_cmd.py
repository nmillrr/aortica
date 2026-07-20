"""Tests for `aortica compare` CLI (US-116, npz mode)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from aortica.cli.compare_cmd import compare_cmd

_N = 200
_RHYTHM = 28


def _targets(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((_N, _RHYTHM)) < 0.3).astype(np.float64)


def _preds(tg: np.ndarray, correct: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ok = rng.random(tg.shape) < correct
    p = np.where(tg == 1, 0.9, 0.1)
    p = np.where(ok, p, 1.0 - p)
    return np.clip(p + rng.normal(0, 0.02, tg.shape), 0.0, 1.0)


def _write_bundle(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def test_compare_npz_upgrade(tmp_path: Path) -> None:
    tg = _targets()
    _write_bundle(tmp_path / "a.npz", {"rhythm": _preds(tg, 0.6, 1)})
    _write_bundle(tmp_path / "b.npz", {"rhythm": _preds(tg, 0.95, 2)})
    _write_bundle(tmp_path / "t.npz", {"rhythm": tg})
    out = tmp_path / "MODEL_COMPARISON.md"

    result = CliRunner().invoke(
        compare_cmd,
        [
            "--predictions-a", str(tmp_path / "a.npz"),
            "--predictions-b", str(tmp_path / "b.npz"),
            "--targets", str(tmp_path / "t.npz"),
            "--n-bootstrap", "100",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Model Comparison Report" in out.read_text()
    assert "UPGRADE" in result.output


def test_compare_npz_regression_exit_code(tmp_path: Path) -> None:
    tg = _targets()
    _write_bundle(tmp_path / "a.npz", {"rhythm": _preds(tg, 0.95, 1)})
    _write_bundle(tmp_path / "b.npz", {"rhythm": _preds(tg, 0.55, 2)})
    _write_bundle(tmp_path / "t.npz", {"rhythm": tg})

    result = CliRunner().invoke(
        compare_cmd,
        [
            "--predictions-a", str(tmp_path / "a.npz"),
            "--predictions-b", str(tmp_path / "b.npz"),
            "--targets", str(tmp_path / "t.npz"),
            "--n-bootstrap", "200",
            "--output", str(tmp_path / "out.md"),
        ],
    )
    # Regression → non-zero exit for CI gating.
    assert result.exit_code == 1
    assert "Regressions" in result.output


def test_compare_json_output(tmp_path: Path) -> None:
    tg = _targets()
    _write_bundle(tmp_path / "a.npz", {"rhythm": _preds(tg, 0.7, 1)})
    _write_bundle(tmp_path / "b.npz", {"rhythm": _preds(tg, 0.8, 2)})
    _write_bundle(tmp_path / "t.npz", {"rhythm": tg})

    result = CliRunner().invoke(
        compare_cmd,
        [
            "--predictions-a", str(tmp_path / "a.npz"),
            "--predictions-b", str(tmp_path / "b.npz"),
            "--targets", str(tmp_path / "t.npz"),
            "--n-bootstrap", "50",
            "--output", str(tmp_path / "out.md"),
            "--format", "json",
        ],
    )
    assert result.exit_code in (0, 1)
    payload = json.loads(result.output)
    assert payload["version_a"].endswith("a.npz")
    assert "task_deltas" in payload


def test_compare_requires_inputs(tmp_path: Path) -> None:
    result = CliRunner().invoke(compare_cmd, [])
    assert result.exit_code != 0
    assert "Provide either" in result.output
