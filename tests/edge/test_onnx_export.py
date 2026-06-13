"""Tests for US-037: ONNX Export Pipeline.

Tests cover:
  - Module structure and imports
  - _OnnxWrapper (flattening dataclass to tuple)
  - export_onnx (file creation, dynamic axes, opset, structural validity)
  - validate_onnx (PyTorch vs ONNX Runtime output comparison)
  - Subset task export (e.g. rhythm-only)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aortica.edge.onnx_export import (  # noqa: E402
    _OnnxWrapper,
    export_onnx,
    validate_onnx,
)
from aortica.models.aortica_model import AorticaModel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    enabled_tasks: list[str] | None = None,
    feature_dim: int = 252,
) -> AorticaModel:
    """Create a small AorticaModel for testing."""
    return AorticaModel(
        in_channels=12,
        feature_dim=feature_dim,
        num_leads=12,
        enabled_tasks=enabled_tasks,
    )


# ---------------------------------------------------------------------------
# Module / Import Tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify edge subpackage and exports are importable."""

    def test_edge_package_importable(self) -> None:
        import aortica.edge  # noqa: F401

    def test_export_onnx_importable(self) -> None:
        from aortica.edge import export_onnx as _fn  # noqa: F401

    def test_validate_onnx_importable(self) -> None:
        from aortica.edge import validate_onnx as _fn  # noqa: F401


# ---------------------------------------------------------------------------
# _OnnxWrapper Tests
# ---------------------------------------------------------------------------


class TestOnnxWrapper:
    """Tests for the dataclass-to-tuple wrapper."""

    def test_output_is_tuple(self) -> None:
        model = _make_model()
        wrapper = _OnnxWrapper(model)
        x = torch.randn(2, 12, 5000)
        out = wrapper(x)
        assert isinstance(out, tuple)

    def test_output_count_all_tasks(self) -> None:
        model = _make_model()
        wrapper = _OnnxWrapper(model)
        x = torch.randn(2, 12, 5000)
        out = wrapper(x)
        assert len(out) == 4  # rhythm, structural, ischaemia, risk

    def test_output_count_subset(self) -> None:
        model = _make_model(enabled_tasks=["rhythm", "risk"])
        wrapper = _OnnxWrapper(model)
        x = torch.randn(2, 12, 5000)
        out = wrapper(x)
        assert len(out) == 2

    def test_output_shapes(self) -> None:
        model = _make_model()
        wrapper = _OnnxWrapper(model)
        x = torch.randn(3, 12, 5000)
        out = wrapper(x)
        # rhythm=28, structural=19, ischaemia=19, risk=6
        expected_dims = [28, 19, 19, 6]
        for tensor, dim in zip(out, expected_dims):
            assert tensor.shape == (3, dim)

    def test_output_order_matches_enabled_tasks(self) -> None:
        model = _make_model(enabled_tasks=["risk", "rhythm"])
        wrapper = _OnnxWrapper(model)
        x = torch.randn(1, 12, 5000)
        out = wrapper(x)
        # risk = 6 outputs, rhythm = 28 outputs
        assert out[0].shape[1] == 6   # risk first
        assert out[1].shape[1] == 28  # rhythm second


# ---------------------------------------------------------------------------
# export_onnx Tests
# ---------------------------------------------------------------------------


class TestExportOnnx:
    """Tests for ONNX model export."""

    def test_creates_file(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        result = export_onnx(model, onnx_path)
        assert result.exists()
        assert result.suffix == ".onnx"

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        onnx_path = tmp_path / "sub" / "dir" / "model.onnx"
        model = _make_model()
        result = export_onnx(model, onnx_path)
        assert result.exists()

    def test_returns_resolved_path(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        result = export_onnx(model, onnx_path)
        assert result.is_absolute()
        assert result == onnx_path.resolve()

    def test_string_path_accepted(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = str(tmp_path / "model.onnx")
        result = export_onnx(model, onnx_path)
        assert result.exists()

    def test_onnx_model_valid(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    def test_input_name(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        input_names = [inp.name for inp in onnx_model.graph.input]
        assert "ecg_input" in input_names

    def test_output_names_all_tasks(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        output_names = [out.name for out in onnx_model.graph.output]
        assert "rhythm_output" in output_names
        assert "structural_output" in output_names
        assert "ischaemia_output" in output_names
        assert "risk_output" in output_names

    def test_output_names_subset(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        output_names = [out.name for out in onnx_model.graph.output]
        assert output_names == ["rhythm_output"]

    def test_dynamic_batch_axis(self, tmp_path: pathlib.Path) -> None:
        """ONNX model should accept different batch sizes."""
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)

        sess = ort.InferenceSession(str(onnx_path))
        # Batch 1
        out1 = sess.run(None, {"ecg_input": np.random.randn(1, 12, 5000).astype(np.float32)})
        assert out1[0].shape[0] == 1
        # Batch 4
        out4 = sess.run(None, {"ecg_input": np.random.randn(4, 12, 5000).astype(np.float32)})
        assert out4[0].shape[0] == 4

    def test_dynamic_signal_length(self, tmp_path: pathlib.Path) -> None:
        """ONNX model should accept different signal lengths."""
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)

        sess = ort.InferenceSession(str(onnx_path))
        # Short signal (2500 samples = 5s at 500 Hz)
        out_short = sess.run(
            None, {"ecg_input": np.random.randn(1, 12, 2500).astype(np.float32)},
        )
        assert out_short[0].shape == (1, 28)
        # Long signal (10000 samples = 20s at 500 Hz)
        out_long = sess.run(
            None, {"ecg_input": np.random.randn(1, 12, 10000).astype(np.float32)},
        )
        assert out_long[0].shape == (1, 28)

    def test_custom_opset_version(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path, opset_version=14)
        onnx_model = onnx.load(str(onnx_path))
        opset = onnx_model.opset_import[0].version
        assert opset == 14


# ---------------------------------------------------------------------------
# validate_onnx Tests
# ---------------------------------------------------------------------------


class TestValidateOnnx:
    """Tests for PyTorch vs ONNX Runtime output validation."""

    def test_returns_dict(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(model, onnx_path)
        assert isinstance(result, dict)

    def test_dict_keys_match_tasks(self, tmp_path: pathlib.Path) -> None:
        tasks = ["rhythm", "structural", "ischaemia", "risk"]
        model = _make_model(enabled_tasks=tasks)
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(model, onnx_path)
        assert set(result.keys()) == set(tasks)

    def test_diffs_within_tolerance(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(model, onnx_path, atol=1e-4)
        for task, diff in result.items():
            assert diff <= 1e-4, f"Task {task} diff {diff} exceeds tolerance"

    def test_reproducible_with_seed(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        r1 = validate_onnx(model, onnx_path, seed=123)
        r2 = validate_onnx(model, onnx_path, seed=123)
        for task in r1:
            assert r1[task] == r2[task]

    def test_different_seeds_may_differ(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        r1 = validate_onnx(model, onnx_path, seed=1)
        r2 = validate_onnx(model, onnx_path, seed=2)
        # Diffs should both be small but may not be identical
        for task in r1:
            assert r1[task] <= 1e-4
            assert r2[task] <= 1e-4

    def test_subset_tasks(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["risk"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(model, onnx_path)
        assert list(result.keys()) == ["risk"]
        assert result["risk"] <= 1e-4

    def test_custom_batch_size(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(model, onnx_path, batch_size=8)
        assert result["rhythm"] <= 1e-4

    def test_custom_signal_length(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["rhythm"])
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        result = validate_onnx(
            model, onnx_path, sample_length=2500,
        )
        assert result["rhythm"] <= 1e-4


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full export → validate pipeline."""

    def test_all_tasks_roundtrip(self, tmp_path: pathlib.Path) -> None:
        model = _make_model()
        model.eval()
        onnx_path = tmp_path / "full.onnx"
        export_onnx(model, onnx_path)
        diffs = validate_onnx(model, onnx_path)
        assert len(diffs) == 4
        for task, diff in diffs.items():
            assert diff <= 1e-4, f"{task}: {diff}"

    def test_rhythm_only_roundtrip(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["rhythm"])
        model.eval()
        onnx_path = tmp_path / "rhythm.onnx"
        export_onnx(model, onnx_path)
        diffs = validate_onnx(model, onnx_path)
        assert list(diffs.keys()) == ["rhythm"]

    def test_two_task_roundtrip(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(enabled_tasks=["structural", "ischaemia"])
        model.eval()
        onnx_path = tmp_path / "two_task.onnx"
        export_onnx(model, onnx_path)
        diffs = validate_onnx(model, onnx_path)
        assert set(diffs.keys()) == {"structural", "ischaemia"}

    def test_onnx_runtime_inference(self, tmp_path: pathlib.Path) -> None:
        """Standalone ONNX Runtime inference produces valid output shapes."""
        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)

        import numpy as np

        sess = ort.InferenceSession(str(onnx_path))
        inp = np.random.randn(2, 12, 5000).astype(np.float32)
        outputs = sess.run(None, {"ecg_input": inp})

        # 4 outputs: rhythm[28], structural[19], ischaemia[19], risk[6]
        assert len(outputs) == 4
        assert outputs[0].shape == (2, 28)
        assert outputs[1].shape == (2, 19)
        assert outputs[2].shape == (2, 19)
        assert outputs[3].shape == (2, 6)

    def test_output_values_in_range(self, tmp_path: pathlib.Path) -> None:
        """ONNX outputs should be sigmoid-scaled (0, 1)."""
        import numpy as np

        model = _make_model()
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)

        sess = ort.InferenceSession(str(onnx_path))
        inp = np.random.randn(4, 12, 5000).astype(np.float32)
        outputs = sess.run(None, {"ecg_input": inp})

        for out in outputs:
            assert np.all(out >= 0.0)
            assert np.all(out <= 1.0)
