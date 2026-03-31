"""Tests for ``aortica predict`` CLI command (US-034)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner  # noqa: E402

from aortica.cli.main import (  # noqa: E402
    ALL_TASKS,
    _severity_badge,
    _severity_style,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click CliRunner for testing."""
    return CliRunner()


@pytest.fixture()
def sample_ecg_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic CSV ECG file for testing."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 12))
    csv_path = tmp_path / "test_ecg.csv"
    header = ",".join(
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    return csv_path


def _make_mock_predict_response() -> Any:
    """Build a mock PredictResponse-like object with ``model_dump()``."""

    class FakeResponse:
        def model_dump(self) -> dict[str, Any]:
            return {
                "quality_report": {
                    "per_lead": [
                        {
                            "lead_name": "II",
                            "score": 85.0,
                            "classification": "good",
                            "flags": [],
                        }
                    ],
                    "overall_score": 85.0,
                    "overall_classification": "good",
                    "recommendation": "accept",
                },
                "predictions": [
                    {
                        "task": "rhythm",
                        "class_names": ["AF", "Normal"],
                        "probabilities": [0.92, 0.08],
                    },
                    {
                        "task": "risk",
                        "class_names": ["mortality", "hf_hosp", "af_onset"],
                        "probabilities": [0.35, 0.62, 0.12],
                    },
                ],
                "uncertainty": {
                    "prediction_sets": {},
                    "confidence_intervals": {},
                    "ood_flag": False,
                    "entropy_score": 0.1234,
                },
            }

    return FakeResponse()


def _make_empty_predict_response() -> Any:
    """Build a mock PredictResponse with no predictions (no model loaded)."""

    class FakeResponse:
        def model_dump(self) -> dict[str, Any]:
            return {
                "quality_report": {
                    "per_lead": [],
                    "overall_score": 90.0,
                    "overall_classification": "good",
                    "recommendation": "accept",
                },
                "predictions": [],
                "uncertainty": None,
            }

    return FakeResponse()


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify CLI constants."""

    def test_all_tasks_list(self) -> None:
        assert ALL_TASKS == ["rhythm", "structural", "ischaemia", "risk"]

    def test_all_tasks_length(self) -> None:
        assert len(ALL_TASKS) == 4


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityHelpers:
    """Tests for severity colouring and badge helpers."""

    def test_high_severity_style(self) -> None:
        assert _severity_style(0.95) == "bold red"

    def test_high_boundary_style(self) -> None:
        assert _severity_style(0.80) == "bold red"

    def test_medium_severity_style(self) -> None:
        assert _severity_style(0.65) == "yellow"

    def test_medium_boundary_style(self) -> None:
        assert _severity_style(0.50) == "yellow"

    def test_low_severity_style(self) -> None:
        assert _severity_style(0.30) == "green"

    def test_zero_severity_style(self) -> None:
        assert _severity_style(0.0) == "green"

    def test_high_badge(self) -> None:
        assert _severity_badge(0.90) == "●"

    def test_medium_badge(self) -> None:
        assert _severity_badge(0.60) == "◐"

    def test_low_badge(self) -> None:
        assert _severity_badge(0.20) == "○"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """Tests for the top-level CLI group (``aortica``)."""

    def test_main_group_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "predict" in result.output

    def test_main_group_no_args(self, runner: CliRunner) -> None:
        result = runner.invoke(main, [])
        # Click groups return exit code 2 when invoked without a subcommand
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "predict" in result.output


# ---------------------------------------------------------------------------
# `aortica predict` command
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """Tests for ``aortica predict <file>``."""

    def test_predict_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--tasks" in result.output
        assert "--model" in result.output

    def test_predict_missing_file(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["predict", "/nonexistent/file.csv"])
        assert result.exit_code != 0

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_table_output(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(main, ["predict", str(sample_ecg_file)])
        assert result.exit_code == 0
        assert "Signal Quality" in result.output or "quality" in result.output.lower()

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_json_output(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(
            main, ["predict", str(sample_ecg_file), "--format", "json"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "quality_report" in parsed
        assert "predictions" in parsed

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_json_has_predictions(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(
            main, ["predict", str(sample_ecg_file), "--format", "json"]
        )
        parsed = json.loads(result.output)
        assert len(parsed["predictions"]) == 2
        assert parsed["predictions"][0]["task"] == "rhythm"

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_default_format_is_table(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(main, ["predict", str(sample_ecg_file)])
        assert result.exit_code == 0
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_tasks_flag(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(
            main,
            ["predict", str(sample_ecg_file), "--tasks", "rhythm,risk", "--format", "json"],
        )
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("enabled_tasks") == ["rhythm", "risk"]

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_invalid_task(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        result = runner.invoke(
            main,
            ["predict", str(sample_ecg_file), "--tasks", "rhythm,bogus"],
        )
        assert result.exit_code != 0
        assert "bogus" in result.output

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_no_model_loaded(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_empty_predict_response()
        result = runner.invoke(
            main, ["predict", str(sample_ecg_file), "--format", "json"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["predictions"] == []

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_passes_file_bytes(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_empty_predict_response()
        result = runner.invoke(
            main, ["predict", str(sample_ecg_file), "--format", "json"]
        )
        assert result.exit_code == 0
        call_args = mock_pipeline.call_args
        assert isinstance(call_args.args[0], bytes)
        assert len(call_args.args[0]) > 0

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_passes_filename(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_empty_predict_response()
        runner.invoke(main, ["predict", str(sample_ecg_file), "--format", "json"])
        call_args = mock_pipeline.call_args
        assert call_args.args[1] == sample_ecg_file.name

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_no_model_flag_passes_none(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_empty_predict_response()
        runner.invoke(main, ["predict", str(sample_ecg_file), "--format", "json"])
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs.get("model") is None

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_pipeline_error(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.side_effect = ValueError("bad format")
        result = runner.invoke(main, ["predict", str(sample_ecg_file)])
        assert result.exit_code != 0
        assert "Inference failed" in result.output

    def test_predict_nonexistent_model(
        self, runner: CliRunner, sample_ecg_file: Path
    ) -> None:
        result = runner.invoke(
            main,
            ["predict", str(sample_ecg_file), "--model", "/no/such/model.pt"],
        )
        assert result.exit_code != 0

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_table_shows_uncertainty(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        mock_pipeline.return_value = _make_mock_predict_response()
        result = runner.invoke(main, ["predict", str(sample_ecg_file)])
        assert result.exit_code == 0
        assert "Uncertainty" in result.output or "distribution" in result.output.lower()

    @patch("aortica.cli.main.run_inference_pipeline")
    def test_predict_table_shows_ood_flag(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        sample_ecg_file: Path,
    ) -> None:
        resp = _make_mock_predict_response()
        original_dump = resp.model_dump

        def patched_dump() -> dict[str, Any]:
            d = original_dump()
            d["uncertainty"]["ood_flag"] = True
            return d

        resp.model_dump = patched_dump  # type: ignore[method-assign]
        mock_pipeline.return_value = resp
        result = runner.invoke(main, ["predict", str(sample_ecg_file)])
        assert result.exit_code == 0
        assert "OUT-OF-DISTRIBUTION" in result.output


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestOutputFormatters:
    """Tests for the formatting helper functions."""

    def test_json_round_trip(self) -> None:
        data = {"foo": "bar", "num": 42}
        output = json.dumps(data, indent=2)
        parsed = json.loads(output)
        assert parsed == data


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that the CLI module is importable."""

    def test_import_cli_init(self) -> None:
        import aortica.cli

        assert hasattr(aortica.cli, "HAS_CLICK")
        assert hasattr(aortica.cli, "HAS_RICH")

    def test_import_cli_main(self) -> None:
        import aortica.cli.main

        assert hasattr(aortica.cli.main, "main")
        assert hasattr(aortica.cli.main, "predict")

    def test_main_is_click_group(self) -> None:
        from aortica.cli.main import main

        assert isinstance(main, click.Group)

    def test_predict_is_click_command(self) -> None:
        from aortica.cli.main import predict

        assert isinstance(predict, click.Command)
