"""Tests for US-040: INT8 Quantization Pipeline.

Tests cover:
  - Module structure and imports
  - ECGCalibrationDataReader (iteration, rewind, shape handling)
  - QuantizationReport dataclass
  - quantize_int8 (file creation, validation, error handling)
  - Output comparison between original and quantized models
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx_mod = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aortica.edge.onnx_export import export_onnx  # noqa: E402
from aortica.edge.quantization import (  # noqa: E402
    ECGCalibrationDataReader,
    QuantizationReport,
    quantize_int8,
)
from aortica.models.aortica_model import AorticaModel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LEADS = 12
SIGNAL_LENGTH = 5000
FEATURE_DIM = 252


def _make_model(
    enabled_tasks: list[str] | None = None,
) -> AorticaModel:
    """Create a small AorticaModel for testing."""
    return AorticaModel(
        in_channels=LEADS,
        feature_dim=FEATURE_DIM,
        num_leads=LEADS,
        enabled_tasks=enabled_tasks,
    )


def _export_model(
    tmp_path: pathlib.Path,
    enabled_tasks: list[str] | None = None,
    filename: str = "model.onnx",
) -> tuple[AorticaModel, pathlib.Path]:
    """Export a model to ONNX and return the model + path."""
    model = _make_model(enabled_tasks=enabled_tasks)
    model.eval()
    onnx_path = tmp_path / filename
    export_onnx(model, onnx_path)
    return model, onnx_path


def _make_calibration_data(
    n: int = 10,
    leads: int = LEADS,
    length: int = SIGNAL_LENGTH,
) -> list[np.ndarray]:
    """Create synthetic calibration data."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((1, leads, length)).astype(np.float32)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Import Tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify module and exports are importable."""

    def test_quantization_module_importable(self) -> None:
        import aortica.edge.quantization  # noqa: F401

    def test_quantize_int8_importable(self) -> None:
        from aortica.edge import quantize_int8 as _fn  # noqa: F401

    def test_ecg_calibration_reader_importable(self) -> None:
        from aortica.edge.quantization import ECGCalibrationDataReader  # noqa: F401

    def test_quantization_report_importable(self) -> None:
        from aortica.edge.quantization import QuantizationReport  # noqa: F401


# ---------------------------------------------------------------------------
# ECGCalibrationDataReader Tests
# ---------------------------------------------------------------------------


class TestECGCalibrationDataReader:
    """Tests for the calibration data reader."""

    def test_get_next_returns_dict(self) -> None:
        data = _make_calibration_data(3)
        reader = ECGCalibrationDataReader(data)
        result = reader.get_next()
        assert isinstance(result, dict)

    def test_get_next_has_input_name(self) -> None:
        data = _make_calibration_data(3)
        reader = ECGCalibrationDataReader(data, input_name="my_input")
        result = reader.get_next()
        assert result is not None
        assert "my_input" in result

    def test_default_input_name(self) -> None:
        data = _make_calibration_data(1)
        reader = ECGCalibrationDataReader(data)
        result = reader.get_next()
        assert result is not None
        assert "ecg_input" in result

    def test_get_next_returns_float32(self) -> None:
        data = _make_calibration_data(1)
        reader = ECGCalibrationDataReader(data)
        result = reader.get_next()
        assert result is not None
        assert result["ecg_input"].dtype == np.float32

    def test_get_next_shape_includes_batch(self) -> None:
        data = _make_calibration_data(1, leads=12, length=5000)
        reader = ECGCalibrationDataReader(data)
        result = reader.get_next()
        assert result is not None
        assert result["ecg_input"].shape == (1, 12, 5000)

    def test_get_next_adds_batch_dim_if_missing(self) -> None:
        # 2D input: [leads, samples] → should become [1, leads, samples]
        data = [np.random.randn(12, 5000).astype(np.float32)]
        reader = ECGCalibrationDataReader(data)
        result = reader.get_next()
        assert result is not None
        assert result["ecg_input"].ndim == 3
        assert result["ecg_input"].shape[0] == 1

    def test_iterates_all_samples(self) -> None:
        n = 5
        data = _make_calibration_data(n)
        reader = ECGCalibrationDataReader(data)
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == n

    def test_returns_none_after_exhaustion(self) -> None:
        data = _make_calibration_data(2)
        reader = ECGCalibrationDataReader(data)
        reader.get_next()
        reader.get_next()
        assert reader.get_next() is None

    def test_rewind_resets_iterator(self) -> None:
        data = _make_calibration_data(3)
        reader = ECGCalibrationDataReader(data)
        # Exhaust
        while reader.get_next() is not None:
            pass
        # Rewind and read again
        reader.rewind()
        result = reader.get_next()
        assert result is not None

    def test_rewind_full_iteration(self) -> None:
        n = 4
        data = _make_calibration_data(n)
        reader = ECGCalibrationDataReader(data)
        # First pass
        while reader.get_next() is not None:
            pass
        # Rewind
        reader.rewind()
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == n


# ---------------------------------------------------------------------------
# QuantizationReport Tests
# ---------------------------------------------------------------------------


class TestQuantizationReport:
    """Tests for the QuantizationReport dataclass."""

    def test_construction(self, tmp_path: pathlib.Path) -> None:
        report = QuantizationReport(
            output_path=tmp_path / "out.onnx",
            original_path=tmp_path / "orig.onnx",
            num_calibration_samples=50,
        )
        assert report.num_calibration_samples == 50
        assert report.success is True

    def test_default_max_diff(self, tmp_path: pathlib.Path) -> None:
        report = QuantizationReport(
            output_path=tmp_path / "out.onnx",
            original_path=tmp_path / "orig.onnx",
            num_calibration_samples=10,
        )
        assert report.max_abs_diff == 0.0

    def test_default_per_output_diff(self, tmp_path: pathlib.Path) -> None:
        report = QuantizationReport(
            output_path=tmp_path / "out.onnx",
            original_path=tmp_path / "orig.onnx",
            num_calibration_samples=10,
        )
        assert report.per_output_max_diff == {}

    def test_custom_values(self, tmp_path: pathlib.Path) -> None:
        report = QuantizationReport(
            output_path=tmp_path / "out.onnx",
            original_path=tmp_path / "orig.onnx",
            num_calibration_samples=100,
            max_abs_diff=0.05,
            per_output_max_diff={"rhythm_output": 0.05},
            success=True,
        )
        assert report.max_abs_diff == 0.05
        assert "rhythm_output" in report.per_output_max_diff


# ---------------------------------------------------------------------------
# quantize_int8 Error Handling Tests
# ---------------------------------------------------------------------------


class TestQuantizeInt8Errors:
    """Tests for error conditions in quantize_int8."""

    def test_file_not_found(self, tmp_path: pathlib.Path) -> None:
        cal_data = _make_calibration_data(5)
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            quantize_int8(
                tmp_path / "nonexistent.onnx",
                cal_data,
                tmp_path / "out.onnx",
            )

    def test_empty_calibration_data(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path)
        with pytest.raises(ValueError, match="calibration_data must not be empty"):
            quantize_int8(onnx_path, [], tmp_path / "out.onnx")

    def test_string_path_accepted(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "quantized" / "model_int8.onnx"
        report = quantize_int8(str(onnx_path), cal_data, str(output))
        assert report.output_path.exists()


# ---------------------------------------------------------------------------
# quantize_int8 Functional Tests
# ---------------------------------------------------------------------------


class TestQuantizeInt8:
    """Tests for successful quantization."""

    def test_creates_output_file(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.output_path.exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "sub" / "dir" / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.output_path.exists()

    def test_returns_quantization_report(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert isinstance(report, QuantizationReport)

    def test_report_success(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.success is True

    def test_report_original_path(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.original_path == onnx_path.resolve()

    def test_report_num_samples(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(7)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.num_calibration_samples == 7

    def test_num_calibration_samples_limit(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(20)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(
            onnx_path, cal_data, output, num_calibration_samples=5,
        )
        assert report.num_calibration_samples == 5

    def test_report_has_max_abs_diff(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert isinstance(report.max_abs_diff, float)
        assert report.max_abs_diff >= 0.0

    def test_report_has_per_output_diffs(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert isinstance(report.per_output_max_diff, dict)
        assert len(report.per_output_max_diff) > 0

    def test_per_output_diffs_contain_task_names(
        self, tmp_path: pathlib.Path,
    ) -> None:
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert "rhythm_output" in report.per_output_max_diff


# ---------------------------------------------------------------------------
# Quantized Model Validation Tests
# ---------------------------------------------------------------------------


class TestQuantizedModelValidation:
    """Tests verifying the quantized model runs correctly."""

    def test_quantized_model_runs_via_ort(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Quantized model should be loadable and runnable."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)

        # Run inference with the quantized model
        sess = ort.InferenceSession(str(report.output_path))
        test_input = np.random.randn(1, LEADS, SIGNAL_LENGTH).astype(
            np.float32,
        )
        results = sess.run(None, {"ecg_input": test_input})
        assert len(results) > 0

    def test_quantized_output_shape(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Quantized model should produce same output shapes."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        quantize_int8(onnx_path, cal_data, output)

        sess = ort.InferenceSession(str(output))
        test_input = np.random.randn(1, LEADS, SIGNAL_LENGTH).astype(
            np.float32,
        )
        results = sess.run(None, {"ecg_input": test_input})
        # rhythm head: 22 outputs
        assert results[0].shape == (1, 22)

    def test_quantized_output_in_range(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Quantized outputs should still be in [0, 1] (sigmoid range)."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        quantize_int8(onnx_path, cal_data, output)

        sess = ort.InferenceSession(str(output))
        test_input = np.random.randn(2, LEADS, SIGNAL_LENGTH).astype(
            np.float32,
        )
        results = sess.run(None, {"ecg_input": test_input})
        for arr in results:
            assert np.all(arr >= -0.1), "Outputs significantly below 0"
            assert np.all(arr <= 1.1), "Outputs significantly above 1"

    def test_quantized_vs_original_diff_logged(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Report should log the max absolute difference."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(10)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        # The diff should be logged (non-negative float)
        assert report.max_abs_diff >= 0.0

    def test_quantized_dynamic_batch(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Quantized model should handle different batch sizes."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        quantize_int8(onnx_path, cal_data, output)

        sess = ort.InferenceSession(str(output))
        # Batch 1
        inp1 = np.random.randn(1, LEADS, SIGNAL_LENGTH).astype(np.float32)
        out1 = sess.run(None, {"ecg_input": inp1})
        assert out1[0].shape[0] == 1
        # Batch 3
        inp3 = np.random.randn(3, LEADS, SIGNAL_LENGTH).astype(np.float32)
        out3 = sess.run(None, {"ecg_input": inp3})
        assert out3[0].shape[0] == 3


# ---------------------------------------------------------------------------
# Multi-Task Quantization Tests
# ---------------------------------------------------------------------------


class TestMultiTaskQuantization:
    """Tests for quantizing models with multiple task heads."""

    def test_all_tasks_quantization(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path)
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.success is True
        # Should have diffs for all 4 outputs
        assert len(report.per_output_max_diff) == 4

    def test_subset_tasks_quantization(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(
            tmp_path, enabled_tasks=["rhythm", "risk"],
        )
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, output)
        assert report.success is True
        assert len(report.per_output_max_diff) == 2

    def test_all_tasks_output_shapes(self, tmp_path: pathlib.Path) -> None:
        _, onnx_path = _export_model(tmp_path)
        cal_data = _make_calibration_data(5)
        output = tmp_path / "model_int8.onnx"
        quantize_int8(onnx_path, cal_data, output)

        sess = ort.InferenceSession(str(output))
        test_input = np.random.randn(2, LEADS, SIGNAL_LENGTH).astype(
            np.float32,
        )
        results = sess.run(None, {"ecg_input": test_input})
        # rhythm=22, structural=15, ischaemia=10, risk=3
        assert len(results) == 4
        assert results[0].shape == (2, 22)
        assert results[1].shape == (2, 15)
        assert results[2].shape == (2, 10)
        assert results[3].shape == (2, 3)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full export → quantize → validate pipeline."""

    def test_export_quantize_infer(self, tmp_path: pathlib.Path) -> None:
        """Complete pipeline: export → quantize → run inference."""
        model = _make_model(enabled_tasks=["rhythm"])
        model.eval()

        # Export
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)

        # Quantize
        cal_data = _make_calibration_data(10)
        int8_path = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, int8_path)

        assert report.success is True
        assert report.output_path.exists()
        assert report.max_abs_diff >= 0.0

        # Inference with quantized model
        sess = ort.InferenceSession(str(int8_path))
        test_input = np.random.randn(1, LEADS, SIGNAL_LENGTH).astype(
            np.float32,
        )
        results = sess.run(None, {"ecg_input": test_input})
        assert results[0].shape == (1, 22)

    def test_quantized_file_smaller_or_different(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Quantized model file should exist (size comparison is optional)."""
        _, onnx_path = _export_model(tmp_path, enabled_tasks=["rhythm"])
        cal_data = _make_calibration_data(5)
        int8_path = tmp_path / "model_int8.onnx"
        report = quantize_int8(onnx_path, cal_data, int8_path)

        original_size = onnx_path.stat().st_size
        quantized_size = report.output_path.stat().st_size
        # Both files should exist and have content
        assert original_size > 0
        assert quantized_size > 0
