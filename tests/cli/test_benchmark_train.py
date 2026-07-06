"""Tests for ``aortica benchmark`` and ``aortica train`` CLI commands.

Uses Click's CliRunner for isolated CLI invocation testing and mocks
the underlying pipeline functions to avoid requiring real datasets or
model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

click = pytest.importorskip("click")
rich = pytest.importorskip("rich")

from click.testing import CliRunner  # noqa: E402

from aortica.cli import _build_cli  # noqa: E402
from aortica.cli.benchmark import (  # noqa: E402
    _VALID_FORMATS,
    _VALID_TASKS,
    _render_benchmark_csv,
    _render_benchmark_json,
    _render_benchmark_table,
    benchmark_cmd,
)
from aortica.cli.train import train_cmd  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_report() -> Dict[str, Any]:
    """Build a mock benchmark report result dict."""
    return {
        "as_dict": {
            "overall": {
                "rhythm": {
                    "task_name": "rhythm",
                    "macro_f1": 0.8800,
                    "ece": 0.0500,
                    "per_class": [
                        {"name": "AF", "auc": 0.95, "sensitivity": 0.90,
                         "specificity": 0.92, "f1": 0.91},
                    ],
                },
            },
            "subgroups": [],
            "n_samples": 100,
            "tasks_evaluated": ["rhythm"],
        },
        "summary": (
            "Benchmark Report (100 samples)\n"
            "============================================================\n"
            "\n--- RHYTHM ---\n"
            "  Macro-F1: 0.8800\n"
            "  ECE:      0.0500\n"
        ),
        "csv": "task,class,auc,sensitivity,specificity,f1,macro_f1,ece,c_index,brier_score\n"
               "rhythm,AF,0.9500,0.9000,0.9200,0.9100,0.8800,0.0500,,\n",
    }


@pytest.fixture()
def runner() -> CliRunner:
    """Click CliRunner for isolated CLI testing."""
    return CliRunner()


@pytest.fixture()
def dummy_dataset_dir(tmp_path: Path) -> str:
    """Create a dummy dataset directory for CLI path validation."""
    (tmp_path / "ptbxl_database.csv").write_text("ecg_id\n1\n")
    return str(tmp_path)


@pytest.fixture()
def dummy_config_yaml(tmp_path: Path) -> str:
    """Create a minimal YAML config file for the train command."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data_path: /tmp/ptbxl\n"
        "epochs: 1\n"
        "lr: 0.001\n"
        "batch_size: 4\n"
        "backend: pytorch\n"
    )
    return str(config_file)


# ---------------------------------------------------------------------------
# Tests: module structure and constants
# ---------------------------------------------------------------------------


class TestBenchmarkConstants:
    """Benchmark CLI constants and imports."""

    def test_valid_tasks(self) -> None:
        assert set(_VALID_TASKS) == {"rhythm", "structural", "ischaemia", "risk"}

    def test_valid_formats(self) -> None:
        assert set(_VALID_FORMATS) == {"table", "json", "csv"}

    def test_benchmark_cmd_is_click_command(self) -> None:
        assert isinstance(benchmark_cmd, click.Command)

    def test_train_cmd_is_click_command(self) -> None:
        assert isinstance(train_cmd, click.Command)


# ---------------------------------------------------------------------------
# Tests: CLI group registration
# ---------------------------------------------------------------------------


class TestCliGroupRegistration:
    """benchmark and train commands are registered in the CLI group."""

    def test_benchmark_registered(self) -> None:
        cli = _build_cli()
        assert "benchmark" in cli.commands

    def test_train_registered(self) -> None:
        cli = _build_cli()
        assert "train" in cli.commands

    def test_all_three_commands_registered(self) -> None:
        cli = _build_cli()
        assert "predict" in cli.commands
        assert "benchmark" in cli.commands
        assert "train" in cli.commands


# ---------------------------------------------------------------------------
# Tests: benchmark command help
# ---------------------------------------------------------------------------


class TestBenchmarkHelp:
    """Benchmark command help and usage."""

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["--help"])
        assert result.exit_code == 0
        assert "Evaluate model performance" in result.output

    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["--help"])
        assert "--format" in result.output
        assert "--tasks" in result.output
        assert "--model" in result.output
        assert "--csv-export" in result.output
        assert "--sampling-rate" in result.output
        assert "--batch-size" in result.output
        assert "--seed" in result.output

    def test_missing_dataset_path(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, [])
        assert result.exit_code != 0

    def test_nonexistent_dataset_path(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["/nonexistent/path"])
        assert result.exit_code != 0

    def test_invalid_task_name(
        self, runner: CliRunner, dummy_dataset_dir: str,
    ) -> None:
        result = runner.invoke(
            benchmark_cmd, [dummy_dataset_dir, "--tasks", "bogus"],
        )
        assert result.exit_code == 1
        assert "invalid task" in result.stderr.lower()

    def test_help_via_group(self, runner: CliRunner) -> None:
        cli = _build_cli()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Evaluate model performance" in result.output


# ---------------------------------------------------------------------------
# Tests: benchmark command execution (mocked)
# ---------------------------------------------------------------------------


class TestBenchmarkExecution:
    """Benchmark command execution with mocked pipeline."""

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_table_output(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        result = runner.invoke(benchmark_cmd, [dummy_dataset_dir])
        assert result.exit_code == 0
        assert "Benchmark Report" in result.output

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_json_output(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        result = runner.invoke(
            benchmark_cmd, [dummy_dataset_dir, "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "overall" in parsed
        assert parsed["n_samples"] == 100

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_csv_output_to_stdout(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        result = runner.invoke(
            benchmark_cmd, [dummy_dataset_dir, "--format", "csv"],
        )
        assert result.exit_code == 0
        assert "task,class,auc" in result.output

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_csv_export_to_file(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
        tmp_path: Path,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        csv_file = str(tmp_path / "results.csv")
        result = runner.invoke(
            benchmark_cmd,
            [dummy_dataset_dir, "--format", "csv", "--csv-export", csv_file],
        )
        assert result.exit_code == 0
        assert Path(csv_file).exists()
        content = Path(csv_file).read_text()
        assert "task,class,auc" in content

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_tasks_filter_passed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        runner.invoke(
            benchmark_cmd,
            [dummy_dataset_dir, "--tasks", "rhythm,ischaemia"],
        )
        call_kwargs = mock_bench.call_args.kwargs
        assert "rhythm" in call_kwargs["tasks"]
        assert "ischaemia" in call_kwargs["tasks"]
        assert "structural" not in call_kwargs["tasks"]

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_sampling_rate_passed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        runner.invoke(
            benchmark_cmd,
            [dummy_dataset_dir, "--sampling-rate", "100"],
        )
        call_kwargs = mock_bench.call_args.kwargs
        assert call_kwargs["sampling_rate"] == 100

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_batch_size_passed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        runner.invoke(
            benchmark_cmd,
            [dummy_dataset_dir, "--batch-size", "32"],
        )
        call_kwargs = mock_bench.call_args.kwargs
        assert call_kwargs["batch_size"] == 32

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_seed_passed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        runner.invoke(
            benchmark_cmd,
            [dummy_dataset_dir, "--seed", "123"],
        )
        call_kwargs = mock_bench.call_args.kwargs
        assert call_kwargs["seed"] == 123

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_pipeline_error_exits_1(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.side_effect = RuntimeError("Dataset corrupt")
        result = runner.invoke(benchmark_cmd, [dummy_dataset_dir])
        assert result.exit_code == 1
        assert "Dataset corrupt" in result.stderr

    @patch("aortica.cli.benchmark._run_benchmark")
    def test_via_cli_group(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        dummy_dataset_dir: str,
    ) -> None:
        mock_bench.return_value = _make_mock_report()
        cli = _build_cli()
        result = runner.invoke(
            cli, ["benchmark", dummy_dataset_dir, "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "overall" in parsed


# ---------------------------------------------------------------------------
# Tests: benchmark render helpers
# ---------------------------------------------------------------------------


class TestBenchmarkRenderHelpers:
    """Benchmark rendering functions."""

    def test_render_table_no_crash(self) -> None:
        result = _make_mock_report()
        _render_benchmark_table(result)

    def test_render_json_valid(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        result = _make_mock_report()
        _render_benchmark_json(result)
        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert "overall" in parsed

    def test_render_csv_to_stdout(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        result = _make_mock_report()
        _render_benchmark_csv(result, csv_path=None)
        captured = capsys.readouterr().out
        assert "task,class,auc" in captured

    def test_render_csv_to_file(self, tmp_path: Path) -> None:
        result = _make_mock_report()
        csv_path = str(tmp_path / "out.csv")
        _render_benchmark_csv(result, csv_path=csv_path)
        assert Path(csv_path).exists()
        assert "task,class,auc" in Path(csv_path).read_text()


# ---------------------------------------------------------------------------
# Tests: train command help
# ---------------------------------------------------------------------------


class TestTrainHelp:
    """Train command help and usage."""

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(train_cmd, ["--help"])
        assert result.exit_code == 0
        assert "Train the multi-task AorticaModel" in result.output

    def test_help_shows_backend_option(self, runner: CliRunner) -> None:
        result = runner.invoke(train_cmd, ["--help"])
        assert "--backend" in result.output

    def test_missing_config_path(self, runner: CliRunner) -> None:
        result = runner.invoke(train_cmd, [])
        assert result.exit_code != 0

    def test_nonexistent_config_path(self, runner: CliRunner) -> None:
        result = runner.invoke(train_cmd, ["/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_help_via_group(self, runner: CliRunner) -> None:
        cli = _build_cli()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train the multi-task AorticaModel" in result.output


# ---------------------------------------------------------------------------
# Tests: train command execution (mocked)
# ---------------------------------------------------------------------------


class TestTrainExecution:
    """Train command execution with mocked training functions."""

    @patch("aortica.cli.train.train_multitask")
    @patch("aortica.cli.train.load_config")
    def test_train_calls_load_config(
        self,
        mock_load: MagicMock,
        mock_train: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "pytorch"
        mock_load.return_value = mock_config
        mock_train.return_value = []

        result = runner.invoke(train_cmd, [dummy_config_yaml])
        assert result.exit_code == 0
        mock_load.assert_called_once_with(dummy_config_yaml)

    @patch("aortica.cli.train.train_multitask")
    @patch("aortica.cli.train.load_config")
    def test_train_pytorch_backend(
        self,
        mock_load: MagicMock,
        mock_train: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "pytorch"
        mock_load.return_value = mock_config
        mock_train.return_value = []

        result = runner.invoke(train_cmd, [dummy_config_yaml])
        assert result.exit_code == 0
        mock_train.assert_called_once_with(mock_config)
        assert "Training complete" in result.output

    @patch("aortica.cli.train.train_multitask_tf")
    @patch("aortica.cli.train.load_config")
    def test_train_tensorflow_backend(
        self,
        mock_load: MagicMock,
        mock_train_tf: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "tensorflow"
        mock_load.return_value = mock_config
        mock_train_tf.return_value = []

        result = runner.invoke(train_cmd, [dummy_config_yaml])
        assert result.exit_code == 0
        mock_train_tf.assert_called_once_with(mock_config)

    @patch("aortica.cli.train.train_multitask")
    @patch("aortica.cli.train.load_config")
    def test_backend_override(
        self,
        mock_load: MagicMock,
        mock_train: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "tensorflow"  # will be overridden
        mock_load.return_value = mock_config
        mock_train.return_value = []

        result = runner.invoke(
            train_cmd, [dummy_config_yaml, "--backend", "pytorch"],
        )
        assert result.exit_code == 0
        # Backend should be overridden to pytorch
        assert mock_config.backend == "pytorch"
        mock_train.assert_called_once()

    @patch("aortica.cli.train.load_config")
    def test_config_load_error(
        self,
        mock_load: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_load.side_effect = ValueError("Invalid YAML")
        result = runner.invoke(train_cmd, [dummy_config_yaml])
        assert result.exit_code == 1
        assert "Error loading config" in result.stderr

    @patch("aortica.cli.train.train_multitask")
    @patch("aortica.cli.train.load_config")
    def test_training_error(
        self,
        mock_load: MagicMock,
        mock_train: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "pytorch"
        mock_load.return_value = mock_config
        mock_train.side_effect = RuntimeError("OOM")

        result = runner.invoke(train_cmd, [dummy_config_yaml])
        assert result.exit_code == 1
        assert "Error during training" in result.stderr

    @patch("aortica.cli.train.train_multitask")
    @patch("aortica.cli.train.load_config")
    def test_via_cli_group(
        self,
        mock_load: MagicMock,
        mock_train: MagicMock,
        runner: CliRunner,
        dummy_config_yaml: str,
    ) -> None:
        mock_config = MagicMock()
        mock_config.backend = "pytorch"
        mock_load.return_value = mock_config
        mock_train.return_value = []

        cli = _build_cli()
        result = runner.invoke(cli, ["train", dummy_config_yaml])
        assert result.exit_code == 0
        assert "Training complete" in result.output
