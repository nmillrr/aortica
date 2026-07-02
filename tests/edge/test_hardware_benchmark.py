"""Tests for aortica.edge.hardware_benchmark — Cross-hardware benchmark suite."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aortica.edge.hardware_benchmark import (
    HardwareBenchmarkReport,
    MetricResult,
    PlatformProfile,
    _input_shape_for_mode,
    benchmark_all_platforms,
    consolidated_csv,
    consolidated_markdown_table,
    hardware_benchmark,
    load_platform_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_profiles() -> Dict[str, PlatformProfile]:
    """Platform profiles for testing without YAML dependency."""
    return {
        "test_fast": PlatformProfile(
            name="test_fast",
            arch="amd64",
            accelerator="none",
            latency_p95_ms=500.0,
            peak_memory_mb=4096.0,
            min_throughput_ips=1.0,
            input_mode="12-lead",
            description="Test fast platform",
        ),
        "test_strict": PlatformProfile(
            name="test_strict",
            arch="arm64",
            accelerator="none",
            latency_p95_ms=0.001,  # impossibly tight
            peak_memory_mb=1.0,  # impossibly tight
            min_throughput_ips=1000000.0,  # impossibly high
            input_mode="single-lead",
            description="Test strict platform (will fail)",
        ),
        "test_6lead": PlatformProfile(
            name="test_6lead",
            arch="arm64",
            accelerator="nnapi",
            latency_p95_ms=300.0,
            peak_memory_mb=512.0,
            input_mode="6-lead",
            description="Test 6-lead platform",
        ),
    }


@pytest.fixture()
def sample_reports() -> List[HardwareBenchmarkReport]:
    """Sample benchmark reports for rendering tests."""
    return [
        HardwareBenchmarkReport(
            platform_name="server_cpu",
            model_variant="edge_int8",
            mean_latency_ms=45.0,
            p50_latency_ms=42.0,
            p95_latency_ms=80.0,
            p99_latency_ms=95.0,
            peak_memory_mb=256.0,
            throughput_ips=22.2,
            model_size_bytes=5_000_000,
            metric_results=[
                MetricResult(
                    metric_name="latency_p95",
                    measured=80.0,
                    target=100.0,
                    passed=True,
                    unit=" ms",
                ),
                MetricResult(
                    metric_name="peak_memory",
                    measured=256.0,
                    target=2048.0,
                    passed=True,
                    unit=" MB",
                ),
                MetricResult(
                    metric_name="throughput",
                    measured=22.2,
                    target=10.0,
                    passed=True,
                    unit=" ips",
                    comparison=">=",
                ),
            ],
            overall_pass=True,
            n_runs=50,
        ),
        HardwareBenchmarkReport(
            platform_name="rpi4",
            model_variant="edge_int8",
            mean_latency_ms=300.0,
            p50_latency_ms=290.0,
            p95_latency_ms=400.0,
            p99_latency_ms=450.0,
            peak_memory_mb=320.0,
            throughput_ips=3.3,
            model_size_bytes=5_000_000,
            metric_results=[
                MetricResult(
                    metric_name="latency_p95",
                    measured=400.0,
                    target=350.0,
                    passed=False,
                    unit=" ms",
                ),
                MetricResult(
                    metric_name="peak_memory",
                    measured=320.0,
                    target=512.0,
                    passed=True,
                    unit=" MB",
                ),
            ],
            overall_pass=False,
            n_runs=50,
        ),
    ]


# ---------------------------------------------------------------------------
# PlatformProfile
# ---------------------------------------------------------------------------


class TestPlatformProfile:
    """Tests for PlatformProfile dataclass."""

    def test_construction(self) -> None:
        p = PlatformProfile(name="test", arch="arm64")
        assert p.name == "test"
        assert p.arch == "arm64"

    def test_defaults(self) -> None:
        p = PlatformProfile(name="test")
        assert p.accelerator == "none"
        assert p.latency_p95_ms == 100.0
        assert p.peak_memory_mb == 2048.0
        assert p.min_throughput_ips == 0.0
        assert p.input_mode == "12-lead"


# ---------------------------------------------------------------------------
# Load platform profiles from YAML
# ---------------------------------------------------------------------------


class TestLoadPlatformProfiles:
    """Tests for loading platform profiles from YAML."""

    def test_load_bundled_profiles(self) -> None:
        profiles = load_platform_profiles()
        assert len(profiles) >= 5
        assert "rpi4" in profiles
        assert "server_cpu" in profiles
        assert "jetson_nano" in profiles

    def test_rpi4_targets(self) -> None:
        profiles = load_platform_profiles()
        rpi4 = profiles["rpi4"]
        assert rpi4.latency_p95_ms == 350
        assert rpi4.peak_memory_mb == 512
        assert rpi4.arch == "arm64"

    def test_server_cpu_targets(self) -> None:
        profiles = load_platform_profiles()
        scpu = profiles["server_cpu"]
        assert scpu.latency_p95_ms == 100
        assert scpu.min_throughput_ips == 10.0

    def test_server_gpu_targets(self) -> None:
        profiles = load_platform_profiles()
        sgpu = profiles["server_gpu"]
        assert sgpu.latency_p95_ms == 30
        assert sgpu.min_throughput_ips == 50.0

    def test_custom_yaml(self, tmp_path: Any) -> None:
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "my_device:\n"
            "  arch: arm64\n"
            "  latency_p95_ms: 200\n"
            "  peak_memory_mb: 256\n"
        )
        profiles = load_platform_profiles(str(yaml_file))
        assert "my_device" in profiles
        assert profiles["my_device"].latency_p95_ms == 200


# ---------------------------------------------------------------------------
# Input shape helper
# ---------------------------------------------------------------------------


class TestInputShape:
    """Tests for input shape determination from mode."""

    def test_12_lead(self) -> None:
        assert _input_shape_for_mode("12-lead") == (1, 12, 5000)

    def test_6_lead(self) -> None:
        assert _input_shape_for_mode("6-lead") == (1, 6, 5000)

    def test_single_lead(self) -> None:
        assert _input_shape_for_mode("single-lead") == (1, 1, 5000)

    def test_default_is_12_lead(self) -> None:
        assert _input_shape_for_mode("unknown") == (1, 12, 5000)


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_pass(self) -> None:
        mr = MetricResult(
            metric_name="latency_p95",
            measured=50.0,
            target=100.0,
            passed=True,
            unit=" ms",
        )
        assert mr.passed is True
        assert "PASS" in str(mr)

    def test_fail(self) -> None:
        mr = MetricResult(
            metric_name="latency_p95",
            measured=150.0,
            target=100.0,
            passed=False,
            unit=" ms",
        )
        assert mr.passed is False
        assert "FAIL" in str(mr)


# ---------------------------------------------------------------------------
# HardwareBenchmarkReport
# ---------------------------------------------------------------------------


class TestHardwareBenchmarkReport:
    """Tests for HardwareBenchmarkReport dataclass."""

    def test_construction(self) -> None:
        report = HardwareBenchmarkReport(
            platform_name="test",
            model_variant="edge_int8",
            n_runs=50,
        )
        assert report.platform_name == "test"
        assert report.n_runs == 50

    def test_model_size_mb(self) -> None:
        report = HardwareBenchmarkReport(
            model_size_bytes=10 * 1024 * 1024,
        )
        assert abs(report.model_size_mb - 10.0) < 0.01

    def test_summary_table(self, sample_reports: List[HardwareBenchmarkReport]) -> None:
        table = sample_reports[0].summary_table()
        assert "server_cpu" in table
        assert "edge_int8" in table
        assert "PASS" in table or "✅" in table

    def test_summary_table_failure(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        table = sample_reports[1].summary_table()
        assert "FAIL" in table or "❌" in table

    def test_to_dict(self, sample_reports: List[HardwareBenchmarkReport]) -> None:
        d = sample_reports[0].to_dict()
        assert d["platform_name"] == "server_cpu"
        assert d["overall_pass"] is True
        assert "metric_results" in d
        assert len(d["metric_results"]) == 3

    def test_to_dict_serializable(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        """Verify to_dict output is JSON-serializable."""
        d = sample_reports[0].to_dict()
        json_str = json.dumps(d)
        assert json_str is not None


# ---------------------------------------------------------------------------
# Hardware benchmark function (with mock ONNX session)
# ---------------------------------------------------------------------------


class TestHardwareBenchmark:
    """Tests for the hardware_benchmark function using mock ONNX sessions."""

    def _make_mock_session(self) -> MagicMock:
        """Create a mock onnxruntime InferenceSession."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 22))]
        return mock_session

    def test_benchmark_pass(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        # Create a dummy model file
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_fast",
                n_runs=10,
                profiles=mock_profiles,
            )

        assert report.platform_name == "test_fast"
        assert report.n_runs == 10
        assert report.overall_pass is True
        assert len(report.metric_results) >= 2

    def test_benchmark_fail_strict(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_strict",
                n_runs=10,
                profiles=mock_profiles,
            )

        # test_strict has impossibly tight targets
        assert report.overall_pass is False
        assert any(not mr.passed for mr in report.metric_results)

    def test_benchmark_unknown_profile(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        with pytest.raises(ValueError, match="Unknown platform profile"):
            hardware_benchmark(
                model_path=str(model_file),
                platform_profile="nonexistent",
                n_runs=5,
                profiles=mock_profiles,
            )

    def test_benchmark_missing_model(
        self, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            hardware_benchmark(
                model_path="/nonexistent/model.onnx",
                platform_profile="test_fast",
                profiles=mock_profiles,
            )

    def test_benchmark_model_variant_detection(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        model_file = tmp_path / "aortica_edge_int8_v0.3.0.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_fast",
                n_runs=5,
                profiles=mock_profiles,
            )

        assert report.model_variant == "edge_int8"

    def test_benchmark_throughput_metric(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        """test_fast has min_throughput_ips=1.0, so should include throughput metric."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_fast",
                n_runs=5,
                profiles=mock_profiles,
            )

        metric_names = [mr.metric_name for mr in report.metric_results]
        assert "throughput" in metric_names

    def test_benchmark_no_throughput_metric(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        """test_6lead has no min_throughput_ips, so no throughput metric."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_6lead",
                n_runs=5,
                profiles=mock_profiles,
            )

        metric_names = [mr.metric_name for mr in report.metric_results]
        assert "throughput" not in metric_names

    def test_benchmark_latencies_populated(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = self._make_mock_session()

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            report = hardware_benchmark(
                model_path=str(model_file),
                platform_profile="test_fast",
                n_runs=15,
                profiles=mock_profiles,
            )

        assert len(report.latencies_ms) == 15
        assert report.mean_latency_ms > 0 or report.mean_latency_ms == 0


# ---------------------------------------------------------------------------
# Benchmark all platforms
# ---------------------------------------------------------------------------


class TestBenchmarkAllPlatforms:
    """Tests for benchmark_all_platforms."""

    def test_runs_all(
        self, tmp_path: Any, mock_profiles: Dict[str, PlatformProfile]
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 22))]

        with patch(
            "aortica.edge.hardware_benchmark.HAS_ONNXRUNTIME", True
        ), patch("aortica.edge.hardware_benchmark.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session

            reports = benchmark_all_platforms(
                model_path=str(model_file),
                n_runs=5,
                profiles=mock_profiles,
            )

        assert len(reports) == len(mock_profiles)
        # Should be sorted by name
        names = [r.platform_name for r in reports]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Consolidated reports
# ---------------------------------------------------------------------------


class TestConsolidatedMarkdownTable:
    """Tests for markdown table generation."""

    def test_markdown_contains_platforms(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        md = consolidated_markdown_table(sample_reports)
        assert "server_cpu" in md
        assert "rpi4" in md

    def test_markdown_header(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        md = consolidated_markdown_table(sample_reports)
        assert "Platform" in md
        assert "p95" in md

    def test_markdown_pass_fail_indicators(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        md = consolidated_markdown_table(sample_reports)
        assert "✅" in md
        assert "❌" in md

    def test_markdown_all_pass(self) -> None:
        reports = [
            HardwareBenchmarkReport(
                platform_name="test",
                overall_pass=True,
                model_variant="v1",
            )
        ]
        md = consolidated_markdown_table(reports)
        assert "All platform targets met" in md

    def test_markdown_some_fail(
        self, sample_reports: List[HardwareBenchmarkReport]
    ) -> None:
        md = consolidated_markdown_table(sample_reports)
        assert "Targets missed" in md
        assert "rpi4" in md


class TestConsolidatedCSV:
    """Tests for CSV report generation."""

    def test_csv_header(self, sample_reports: List[HardwareBenchmarkReport]) -> None:
        csv_text = consolidated_csv(sample_reports)
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        assert "platform" in header
        assert "p95_ms" in header
        assert "overall_pass" in header

    def test_csv_rows(self, sample_reports: List[HardwareBenchmarkReport]) -> None:
        csv_text = consolidated_csv(sample_reports)
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 reports

    def test_csv_values(self, sample_reports: List[HardwareBenchmarkReport]) -> None:
        csv_text = consolidated_csv(sample_reports)
        assert "server_cpu" in csv_text
        assert "True" in csv_text
        assert "False" in csv_text


# ---------------------------------------------------------------------------
# CLI command (Click runner)
# ---------------------------------------------------------------------------


class TestCLICommand:
    """Tests for the benchmark-hardware CLI command."""

    def test_cli_registered(self) -> None:
        from aortica.cli.hardware_benchmark_cmd import benchmark_hardware_cmd

        assert benchmark_hardware_cmd.name == "benchmark-hardware"

    def test_cli_help(self) -> None:
        from click.testing import CliRunner

        from aortica.cli.hardware_benchmark_cmd import benchmark_hardware_cmd

        runner = CliRunner()
        result = runner.invoke(benchmark_hardware_cmd, ["--help"])
        assert result.exit_code == 0
        assert "--platform" in result.output
        assert "--model" in result.output
        assert "--all" in result.output
        assert "--format" in result.output

    def test_cli_no_platform_no_all(self) -> None:
        from click.testing import CliRunner

        from aortica.cli.hardware_benchmark_cmd import benchmark_hardware_cmd

        runner = CliRunner()
        result = runner.invoke(
            benchmark_hardware_cmd,
            ["--model", "/dev/null"],  # model exists but won't be used
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API imports work."""

    def test_import_from_edge(self) -> None:
        from aortica.edge import (
            HardwareBenchmarkReport,
            PlatformProfile,
            hardware_benchmark,
            load_platform_profiles,
        )

        assert callable(hardware_benchmark)
        assert callable(load_platform_profiles)

    def test_import_metric_result(self) -> None:
        from aortica.edge.hardware_benchmark import MetricResult

        mr = MetricResult(
            metric_name="test", measured=1.0, target=2.0, passed=True
        )
        assert mr.passed is True

    def test_import_consolidated_functions(self) -> None:
        from aortica.edge.hardware_benchmark import (
            benchmark_all_platforms,
            consolidated_csv,
            consolidated_markdown_table,
        )

        assert callable(benchmark_all_platforms)
        assert callable(consolidated_csv)
        assert callable(consolidated_markdown_table)
