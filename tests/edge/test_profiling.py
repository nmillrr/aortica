"""Tests for aortica.edge.profiling — inference profiling and power estimation."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports (skip if onnxruntime not available)
# ---------------------------------------------------------------------------

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")

from aortica.edge.profiling import (
    DEFAULT_HARDWARE,
    DEFAULT_N_RUNS,
    HARDWARE_TDP,
    WARMUP_RUNS,
    InferenceProfile,
    _estimate_peak_memory,
    _estimate_power,
    profile_inference,
)


# ---------------------------------------------------------------------------
# Helpers — create a minimal ONNX model for testing
# ---------------------------------------------------------------------------


def _create_minimal_onnx_model(path: str, input_shape: tuple[int, ...] = (1, 12, 5000)) -> str:
    """Create a minimal ONNX model that accepts the given input shape."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx package required to create test models")

    # Simple model: input → identity → output
    input_name = "ecg_input"
    output_name = "output"

    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, list(input_shape),
    )
    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, list(input_shape),
    )

    identity_node = helper.make_node("Identity", [input_name], [output_name])

    graph = helper.make_graph(
        [identity_node],
        "test_model",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.save(model, path)
    return path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_hardware_tdp_contains_rpi4(self) -> None:
        assert "rpi4" in HARDWARE_TDP

    def test_hardware_tdp_contains_rpi5(self) -> None:
        assert "rpi5" in HARDWARE_TDP

    def test_hardware_tdp_contains_jetson_nano(self) -> None:
        assert "jetson_nano" in HARDWARE_TDP

    def test_hardware_tdp_contains_jetson_orin_nano(self) -> None:
        assert "jetson_orin_nano" in HARDWARE_TDP

    def test_hardware_tdp_values_positive(self) -> None:
        for hw, tdp in HARDWARE_TDP.items():
            assert tdp > 0, f"{hw} TDP must be positive"

    def test_default_hardware(self) -> None:
        assert DEFAULT_HARDWARE == "rpi4"

    def test_default_n_runs(self) -> None:
        assert DEFAULT_N_RUNS == 100

    def test_warmup_runs(self) -> None:
        assert WARMUP_RUNS == 5


# ---------------------------------------------------------------------------
# InferenceProfile dataclass
# ---------------------------------------------------------------------------


class TestInferenceProfile:
    """Test InferenceProfile dataclass."""

    def test_defaults(self) -> None:
        p = InferenceProfile()
        assert p.mean_latency_ms == 0.0
        assert p.p50_latency_ms == 0.0
        assert p.p95_latency_ms == 0.0
        assert p.min_latency_ms == 0.0
        assert p.max_latency_ms == 0.0
        assert p.std_latency_ms == 0.0
        assert p.peak_memory_bytes == 0
        assert p.model_size_bytes == 0
        assert p.n_runs == 0
        assert p.hardware_profile == DEFAULT_HARDWARE
        assert p.latencies_ms == []
        assert p.input_shape == ()
        assert p.model_path == ""

    def test_custom_construction(self) -> None:
        p = InferenceProfile(
            mean_latency_ms=10.5,
            p50_latency_ms=9.8,
            p95_latency_ms=15.2,
            min_latency_ms=8.1,
            max_latency_ms=20.0,
            std_latency_ms=2.3,
            peak_memory_bytes=1024 * 1024 * 50,
            model_size_bytes=1024 * 1024 * 10,
            n_runs=100,
            hardware_profile="rpi5",
            tdp_watts=6.0,
            energy_per_inference_mj=63.0,
            power_draw_watts=0.063,
            latencies_ms=[10.0, 11.0],
            input_shape=(1, 12, 5000),
            model_path="/tmp/model.onnx",
        )
        assert p.mean_latency_ms == 10.5
        assert p.hardware_profile == "rpi5"
        assert p.n_runs == 100
        assert p.input_shape == (1, 12, 5000)

    def test_model_size_mb(self) -> None:
        p = InferenceProfile(model_size_bytes=10 * 1024 * 1024)
        assert abs(p.model_size_mb - 10.0) < 0.01

    def test_peak_memory_mb(self) -> None:
        p = InferenceProfile(peak_memory_bytes=50 * 1024 * 1024)
        assert abs(p.peak_memory_mb - 50.0) < 0.01

    def test_summary_table(self) -> None:
        p = InferenceProfile(
            mean_latency_ms=10.5,
            p50_latency_ms=9.8,
            p95_latency_ms=15.2,
            min_latency_ms=8.1,
            max_latency_ms=20.0,
            std_latency_ms=2.3,
            peak_memory_bytes=1024 * 1024 * 50,
            model_size_bytes=1024 * 1024 * 10,
            n_runs=100,
            hardware_profile="rpi4",
            tdp_watts=4.0,
            energy_per_inference_mj=42.0,
            power_draw_watts=0.042,
            input_shape=(1, 12, 5000),
            model_path="/tmp/model.onnx",
        )
        table = p.summary_table()
        assert "Inference Profile" in table
        assert "Latency" in table
        assert "Memory" in table
        assert "Power Estimation" in table
        assert "10.50 ms" in table
        assert "rpi4" in table
        assert "4.0 W" in table

    def test_summary_table_contains_model_path(self) -> None:
        p = InferenceProfile(model_path="/tmp/test.onnx")
        assert "/tmp/test.onnx" in p.summary_table()

    def test_to_dict(self) -> None:
        p = InferenceProfile(
            mean_latency_ms=10.0,
            model_size_bytes=1024,
            n_runs=50,
            input_shape=(1, 12, 5000),
            model_path="/tmp/model.onnx",
        )
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["mean_latency_ms"] == 10.0
        assert d["model_size_bytes"] == 1024
        assert d["n_runs"] == 50
        assert d["input_shape"] == [1, 12, 5000]
        assert d["model_path"] == "/tmp/model.onnx"

    def test_to_dict_includes_derived_fields(self) -> None:
        p = InferenceProfile(
            model_size_bytes=10 * 1024 * 1024,
            peak_memory_bytes=50 * 1024 * 1024,
        )
        d = p.to_dict()
        assert "model_size_mb" in d
        assert "peak_memory_mb" in d
        assert abs(d["model_size_mb"] - 10.0) < 0.01
        assert abs(d["peak_memory_mb"] - 50.0) < 0.01

    def test_to_dict_json_serialisable(self) -> None:
        p = InferenceProfile(
            mean_latency_ms=5.0,
            input_shape=(1, 12, 5000),
            latencies_ms=[4.0, 5.0, 6.0],
        )
        d = p.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


class TestEstimatePeakMemory:
    """Test _estimate_peak_memory helper."""

    def test_returns_positive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"x" * 1000)
            f.flush()
            try:
                data = np.zeros((1, 12, 5000), dtype=np.float32)
                mem = _estimate_peak_memory(f.name, data)
                assert mem > 0
            finally:
                os.unlink(f.name)

    def test_scales_with_model_size(self) -> None:
        data = np.zeros((1, 12, 5000), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f1:
            f1.write(b"x" * 1000)
            f1.flush()
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f2:
                f2.write(b"x" * 10000)
                f2.flush()
                try:
                    mem1 = _estimate_peak_memory(f1.name, data)
                    mem2 = _estimate_peak_memory(f2.name, data)
                    assert mem2 > mem1
                finally:
                    os.unlink(f1.name)
                    os.unlink(f2.name)

    def test_includes_input_size(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"x" * 1000)
            f.flush()
            try:
                small = np.zeros((1, 1, 100), dtype=np.float32)
                large = np.zeros((1, 12, 5000), dtype=np.float32)
                mem_small = _estimate_peak_memory(f.name, small)
                mem_large = _estimate_peak_memory(f.name, large)
                assert mem_large > mem_small
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# Power estimation
# ---------------------------------------------------------------------------


class TestEstimatePower:
    """Test _estimate_power helper."""

    def test_returns_tuple_of_three(self) -> None:
        result = _estimate_power(10.0, "rpi4")
        assert len(result) == 3

    def test_rpi4_tdp(self) -> None:
        tdp, _, _ = _estimate_power(10.0, "rpi4")
        assert tdp == 4.0

    def test_rpi5_tdp(self) -> None:
        tdp, _, _ = _estimate_power(10.0, "rpi5")
        assert tdp == 6.0

    def test_jetson_nano_tdp(self) -> None:
        tdp, _, _ = _estimate_power(10.0, "jetson_nano")
        assert tdp == 5.0

    def test_jetson_orin_nano_tdp(self) -> None:
        tdp, _, _ = _estimate_power(10.0, "jetson_orin_nano")
        assert tdp == 7.0

    def test_unknown_hardware_defaults_to_rpi4(self) -> None:
        tdp, _, _ = _estimate_power(10.0, "unknown_device")
        assert tdp == HARDWARE_TDP[DEFAULT_HARDWARE]

    def test_energy_scales_with_latency(self) -> None:
        _, energy_short, _ = _estimate_power(10.0, "rpi4")
        _, energy_long, _ = _estimate_power(100.0, "rpi4")
        assert energy_long > energy_short

    def test_energy_calculation(self) -> None:
        # 10ms latency on 4W device: 4 * 0.01 * 1000 = 40 mJ
        _, energy_mj, _ = _estimate_power(10.0, "rpi4")
        assert abs(energy_mj - 40.0) < 0.01

    def test_power_draw_positive(self) -> None:
        _, _, power = _estimate_power(10.0, "rpi4")
        assert power > 0


# ---------------------------------------------------------------------------
# profile_inference with synthetic ONNX model
# ---------------------------------------------------------------------------


class TestProfileInference:
    """Test profile_inference with a real (minimal) ONNX model."""

    @pytest.fixture()
    def onnx_model_path(self, tmp_path: Any) -> str:
        """Create a minimal ONNX model in a temp directory."""
        path = str(tmp_path / "test_model.onnx")
        return _create_minimal_onnx_model(path, input_shape=(1, 12, 500))

    def test_returns_inference_profile(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=5, warmup_runs=1,
        )
        assert isinstance(result, InferenceProfile)

    def test_latency_statistics_populated(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=10, warmup_runs=1,
        )
        assert result.mean_latency_ms > 0
        assert result.p50_latency_ms > 0
        assert result.p95_latency_ms > 0
        assert result.min_latency_ms > 0
        assert result.max_latency_ms > 0
        assert result.min_latency_ms <= result.p50_latency_ms
        assert result.p50_latency_ms <= result.p95_latency_ms
        assert result.p95_latency_ms <= result.max_latency_ms

    def test_n_runs_matches(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=7, warmup_runs=1,
        )
        assert result.n_runs == 7
        assert len(result.latencies_ms) == 7

    def test_model_size_positive(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.model_size_bytes > 0

    def test_peak_memory_positive(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.peak_memory_bytes > 0

    def test_input_shape_recorded(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.input_shape == (1, 12, 500)

    def test_model_path_recorded(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.model_path == onnx_model_path

    def test_hardware_profile_recorded(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3,
            warmup_runs=0, hardware_profile="rpi5",
        )
        assert result.hardware_profile == "rpi5"
        assert result.tdp_watts == 6.0

    def test_energy_positive(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.energy_per_inference_mj > 0

    def test_power_draw_positive(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.power_draw_watts > 0

    def test_2d_input_expanded(self, onnx_model_path: str) -> None:
        """Test that 2D input (leads, samples) gets batch dim added."""
        data = np.random.randn(12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
        )
        assert result.input_shape == (1, 12, 500)

    def test_file_not_found(self) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        with pytest.raises(FileNotFoundError, match="not found"):
            profile_inference("/nonexistent/model.onnx", data, n_runs=3)

    def test_n_runs_zero_raises(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        with pytest.raises(ValueError, match="n_runs must be positive"):
            profile_inference(onnx_model_path, data, n_runs=0)

    def test_n_runs_negative_raises(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        with pytest.raises(ValueError, match="n_runs must be positive"):
            profile_inference(onnx_model_path, data, n_runs=-5)

    def test_session_options_applied(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=3, warmup_runs=0,
            session_options={"intra_op_num_threads": 2, "inter_op_num_threads": 1},
        )
        assert isinstance(result, InferenceProfile)

    def test_latencies_ordered(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=20, warmup_runs=2,
        )
        # All latencies should be positive
        for lat in result.latencies_ms:
            assert lat > 0

    def test_std_nonnegative(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=10, warmup_runs=1,
        )
        assert result.std_latency_ms >= 0

    def test_summary_table_after_profiling(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=5, warmup_runs=1,
        )
        table = result.summary_table()
        assert "Inference Profile" in table
        assert "Latency" in table

    def test_to_dict_after_profiling(self, onnx_model_path: str) -> None:
        data = np.random.randn(1, 12, 500).astype(np.float32)
        result = profile_inference(
            onnx_model_path, data, n_runs=5, warmup_runs=1,
        )
        d = result.to_dict()
        assert d["n_runs"] == 5
        assert d["mean_latency_ms"] > 0


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestProfileCLI:
    """Test the `aortica profile` CLI command."""

    @pytest.fixture()
    def onnx_model_path(self, tmp_path: Any) -> str:
        path = str(tmp_path / "test_model.onnx")
        return _create_minimal_onnx_model(path, input_shape=(1, 12, 500))

    def test_help(self) -> None:
        click = pytest.importorskip("click")
        from click.testing import CliRunner

        from aortica.cli.profile_cmd import profile_cmd

        runner = CliRunner()
        result = runner.invoke(profile_cmd, ["--help"])
        assert result.exit_code == 0
        assert "Profile inference" in result.output or "MODEL_PATH" in result.output

    def test_table_output(self, onnx_model_path: str) -> None:
        pytest.importorskip("click")
        pytest.importorskip("rich")
        from click.testing import CliRunner

        from aortica.cli.profile_cmd import profile_cmd

        runner = CliRunner()
        result = runner.invoke(
            profile_cmd,
            [onnx_model_path, "--n-runs", "3", "--warmup", "1",
             "--duration-samples", "500"],
        )
        assert result.exit_code == 0

    def test_json_output(self, onnx_model_path: str) -> None:
        pytest.importorskip("click")
        pytest.importorskip("rich")
        from click.testing import CliRunner

        from aortica.cli.profile_cmd import profile_cmd

        runner = CliRunner()
        result = runner.invoke(
            profile_cmd,
            [onnx_model_path, "--format", "json", "--n-runs", "3",
             "--warmup", "1", "--duration-samples", "500"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "mean_latency_ms" in data
        assert "model_size_bytes" in data

    def test_nonexistent_model(self) -> None:
        pytest.importorskip("click")
        from click.testing import CliRunner

        from aortica.cli.profile_cmd import profile_cmd

        runner = CliRunner()
        result = runner.invoke(profile_cmd, ["/nonexistent/model.onnx"])
        assert result.exit_code != 0

    def test_custom_hardware(self, onnx_model_path: str) -> None:
        pytest.importorskip("click")
        pytest.importorskip("rich")
        from click.testing import CliRunner

        from aortica.cli.profile_cmd import profile_cmd

        runner = CliRunner()
        result = runner.invoke(
            profile_cmd,
            [onnx_model_path, "--format", "json", "--n-runs", "3",
             "--warmup", "1", "--duration-samples", "500",
             "--hardware", "jetson_nano"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["hardware_profile"] == "jetson_nano"
        assert data["tdp_watts"] == 5.0

    def test_cli_group_registration(self) -> None:
        pytest.importorskip("click")
        pytest.importorskip("rich")
        try:
            from aortica.cli import _build_cli

            cli = _build_cli()
            command_names = list(cli.commands.keys())
            assert "profile" in command_names
        except (ImportError, AttributeError, NameError):
            # Pre-existing issue: train_multitask.py imports torch at module
            # level, causing cascade failure when torch is not installed.
            # Verify profile_cmd is importable independently.
            from aortica.cli.profile_cmd import profile_cmd
            assert profile_cmd.name == "profile"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Test that modules can be imported."""

    def test_import_profiling_module(self) -> None:
        from aortica.edge import profiling
        assert hasattr(profiling, "profile_inference")
        assert hasattr(profiling, "InferenceProfile")

    def test_import_from_edge_package(self) -> None:
        from aortica.edge import InferenceProfile, profile_inference
        assert InferenceProfile is not None
        assert profile_inference is not None

    def test_import_profile_cmd(self) -> None:
        pytest.importorskip("click")
        from aortica.cli.profile_cmd import profile_cmd
        assert profile_cmd is not None
