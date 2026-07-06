"""Tests for ``aortica predict`` CLI command.

Uses Click's CliRunner for isolated CLI invocation testing and
mocks the inference pipeline to avoid requiring real ECG files or
model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

click = pytest.importorskip("click")
rich = pytest.importorskip("rich")

from click.testing import CliRunner  # noqa: E402

from aortica.cli import _build_cli, _check_cli_deps  # noqa: E402
from aortica.cli.predict import (  # noqa: E402
    _VALID_TASKS,
    _classify_severity,
    _render_json,
    _render_table,
    _run_pipeline,
    predict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_result(
    *,
    overall_score: float = 85.0,
    overall_classification: str = "good",
    recommendation: str = "accept",
    predictions: Optional[List[Dict[str, Any]]] = None,
    uncertainty: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a mock prediction result dict."""
    return {
        "quality_report": {
            "overall_score": overall_score,
            "overall_classification": overall_classification,
            "recommendation": recommendation,
            "per_lead": [
                {
                    "lead_name": "II",
                    "score": 90.0,
                    "classification": "good",
                    "flags": [],
                },
                {
                    "lead_name": "V1",
                    "score": 65.0,
                    "classification": "marginal",
                    "flags": ["baseline_wander"],
                },
            ],
        },
        "predictions": predictions or [],
        "uncertainty": uncertainty,
    }


@pytest.fixture()
def runner() -> CliRunner:
    """Click CliRunner for isolated CLI testing."""
    return CliRunner()


@pytest.fixture()
def dummy_ecg_file(tmp_path: Path) -> str:
    """Create a tiny dummy file for CLI path validation."""
    p = tmp_path / "test.csv"
    p.write_text("lead_I,lead_II\n0.1,0.2\n")
    return str(p)


# ---------------------------------------------------------------------------
# Tests: module structure and imports
# ---------------------------------------------------------------------------


class TestImports:
    """CLI module imports and structure."""

    def test_check_cli_deps_succeeds(self) -> None:
        """_check_cli_deps() succeeds when click and rich are available."""
        _check_cli_deps()  # should not raise

    def test_build_cli_returns_click_group(self) -> None:
        """_build_cli() returns a Click group."""
        cli = _build_cli()
        assert isinstance(cli, click.Group)

    def test_predict_registered(self) -> None:
        """predict command is registered in the CLI group."""
        cli = _build_cli()
        assert "predict" in cli.commands

    def test_valid_tasks_constant(self) -> None:
        """_VALID_TASKS contains all four task heads."""
        assert set(_VALID_TASKS) == {"rhythm", "structural", "ischaemia", "risk"}


# ---------------------------------------------------------------------------
# Tests: severity classification
# ---------------------------------------------------------------------------


class TestSeverityClassification:
    """_classify_severity helper."""

    def test_critical(self) -> None:
        assert _classify_severity(0.95) == "critical"

    def test_critical_boundary(self) -> None:
        assert _classify_severity(0.9) == "critical"

    def test_high(self) -> None:
        assert _classify_severity(0.8) == "high"

    def test_high_boundary(self) -> None:
        assert _classify_severity(0.7) == "high"

    def test_moderate(self) -> None:
        assert _classify_severity(0.6) == "moderate"

    def test_moderate_boundary(self) -> None:
        assert _classify_severity(0.5) == "moderate"

    def test_low(self) -> None:
        assert _classify_severity(0.3) == "low"

    def test_low_zero(self) -> None:
        assert _classify_severity(0.0) == "low"


# ---------------------------------------------------------------------------
# Tests: predict command via CliRunner
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """CLI invocation of ``aortica predict``."""

    def test_help_flag(self, runner: CliRunner) -> None:
        """--help shows usage text."""
        result = runner.invoke(predict, ["--help"])
        assert result.exit_code == 0
        assert "Run AI inference on an ECG file" in result.output

    def test_missing_file_argument(self, runner: CliRunner) -> None:
        """Invoking without a file argument fails."""
        result = runner.invoke(predict, [])
        assert result.exit_code != 0

    def test_nonexistent_file(self, runner: CliRunner) -> None:
        """Invoking with a nonexistent file fails."""
        result = runner.invoke(predict, ["/nonexistent/file.csv"])
        assert result.exit_code != 0

    @patch("aortica.cli.predict._run_pipeline")
    def test_table_output_default(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """Default output format is 'table' and renders without error."""
        mock_pipeline.return_value = _make_mock_result()

        result = runner.invoke(predict, [dummy_ecg_file])
        assert result.exit_code == 0
        # Should contain quality-related output
        assert "Signal Quality" in result.output or "quality" in result.output.lower()

    @patch("aortica.cli.predict._run_pipeline")
    def test_json_output(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """--format json prints valid JSON to stdout."""
        mock_pipeline.return_value = _make_mock_result()

        result = runner.invoke(predict, [dummy_ecg_file, "--format", "json"])
        assert result.exit_code == 0

        parsed = json.loads(result.output)
        assert "quality_report" in parsed
        assert parsed["quality_report"]["overall_score"] == 85.0

    @patch("aortica.cli.predict._run_pipeline")
    def test_tasks_filter(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """--tasks filters which tasks are passed to the pipeline."""
        mock_pipeline.return_value = _make_mock_result()

        result = runner.invoke(
            predict, [dummy_ecg_file, "--tasks", "rhythm,ischaemia", "--format", "json"]
        )
        assert result.exit_code == 0

        # Verify the pipeline was called with the correct tasks
        call_kwargs = mock_pipeline.call_args
        assert "rhythm" in call_kwargs.kwargs.get("tasks", call_kwargs[1].get("tasks", []))

    def test_invalid_task_name(
        self,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """Invalid task name prints error and exits with code 1."""
        result = runner.invoke(predict, [dummy_ecg_file, "--tasks", "bogus"])
        assert result.exit_code == 1
        assert "invalid task" in result.stderr.lower()

    @patch("aortica.cli.predict._run_pipeline")
    def test_ecg_format_override(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """--ecg-format is passed through to the pipeline."""
        mock_pipeline.return_value = _make_mock_result()

        result = runner.invoke(
            predict,
            [dummy_ecg_file, "--ecg-format", "csv", "--format", "json"],
        )
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get(
            "format_override", call_kwargs[1].get("format_override")
        ) == "csv"

    @patch("aortica.cli.predict._run_pipeline")
    def test_pipeline_error_exits_1(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """Pipeline exceptions cause exit code 1 and stderr error message."""
        mock_pipeline.side_effect = ValueError("Bad ECG file")

        result = runner.invoke(predict, [dummy_ecg_file])
        assert result.exit_code == 1
        assert "Bad ECG file" in result.stderr

    @patch("aortica.cli.predict._run_pipeline")
    def test_no_model_shows_message(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """When predictions are empty, table output notes no model loaded."""
        mock_pipeline.return_value = _make_mock_result(predictions=[])

        result = runner.invoke(predict, [dummy_ecg_file])
        assert result.exit_code == 0
        out_lower = result.output.lower()
        assert "predictions unavailable" in out_lower or "no model" in out_lower

    @patch("aortica.cli.predict._run_pipeline")
    def test_with_predictions(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """Table output renders predictions when present."""
        predictions = [
            {
                "task": "rhythm",
                "class_names": ["AF", "NSR"],
                "probabilities": [0.85, 0.95],
            }
        ]
        mock_pipeline.return_value = _make_mock_result(predictions=predictions)

        result = runner.invoke(predict, [dummy_ecg_file])
        assert result.exit_code == 0
        assert "Rhythm" in result.output

    @patch("aortica.cli.predict._run_pipeline")
    def test_with_uncertainty(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """Table output renders uncertainty section when present."""
        uncertainty = {
            "ood_flag": True,
            "entropy_score": 0.4321,
            "prediction_sets": {},
            "confidence_intervals": {},
        }
        mock_pipeline.return_value = _make_mock_result(uncertainty=uncertainty)

        result = runner.invoke(predict, [dummy_ecg_file])
        assert result.exit_code == 0
        assert "Uncertainty" in result.output

    @patch("aortica.cli.predict._run_pipeline")
    def test_json_with_predictions(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """JSON output includes predictions."""
        predictions = [
            {
                "task": "rhythm",
                "class_names": ["AF", "NSR"],
                "probabilities": [0.85, 0.95],
            }
        ]
        mock_pipeline.return_value = _make_mock_result(predictions=predictions)

        result = runner.invoke(predict, [dummy_ecg_file, "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["predictions"]) == 1
        assert parsed["predictions"][0]["task"] == "rhythm"


# ---------------------------------------------------------------------------
# Tests: render helpers
# ---------------------------------------------------------------------------


class TestRenderHelpers:
    """Rendering functions."""

    def test_render_json_valid(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_render_json produces valid JSON."""
        result = _make_mock_result()
        _render_json(result)
        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["quality_report"]["overall_score"] == 85.0

    def test_render_table_no_crash(self) -> None:
        """_render_table does not crash on a typical result dict."""
        result = _make_mock_result(
            predictions=[
                {
                    "task": "rhythm",
                    "class_names": ["AF", "NSR"],
                    "probabilities": [0.85, 0.15],
                }
            ]
        )
        # Should complete without raising
        _render_table(result)

    def test_render_table_empty_predictions(self) -> None:
        """_render_table handles empty predictions gracefully."""
        result = _make_mock_result(predictions=[])
        _render_table(result)

    def test_render_table_quality_marginal(self) -> None:
        """_render_table handles marginal quality classification."""
        result = _make_mock_result(
            overall_score=55.0,
            overall_classification="marginal",
            recommendation="review",
        )
        _render_table(result)

    def test_render_table_quality_poor(self) -> None:
        """_render_table handles poor quality classification."""
        result = _make_mock_result(
            overall_score=25.0,
            overall_classification="poor",
            recommendation="reject",
        )
        _render_table(result)

    def test_render_table_uncertainty(self) -> None:
        """_render_table renders uncertainty section."""
        uncertainty = {
            "ood_flag": False,
            "entropy_score": 0.123,
            "prediction_sets": {},
            "confidence_intervals": {},
        }
        result = _make_mock_result(uncertainty=uncertainty)
        _render_table(result)


# ---------------------------------------------------------------------------
# Tests: CLI group integration
# ---------------------------------------------------------------------------


class TestCliGroup:
    """Top-level CLI group."""

    def test_cli_group_help(self, runner: CliRunner) -> None:
        """CLI group shows help text."""
        cli = _build_cli()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "predict" in result.output

    def test_predict_subcommand_accessible(self, runner: CliRunner) -> None:
        """predict subcommand is accessible through the group."""
        cli = _build_cli()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Run AI inference" in result.output

    @patch("aortica.cli.predict._run_pipeline")
    def test_cli_predict_via_group(
        self,
        mock_pipeline: MagicMock,
        runner: CliRunner,
        dummy_ecg_file: str,
    ) -> None:
        """aortica predict <file> works through the group."""
        mock_pipeline.return_value = _make_mock_result()
        cli = _build_cli()

        result = runner.invoke(cli, ["predict", dummy_ecg_file, "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "quality_report" in parsed


# ---------------------------------------------------------------------------
# Tests: _run_pipeline integration (mocked read/denoise/quality)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """_run_pipeline integration with mocked inference pipeline."""

    @patch("aortica.api.predict.run_inference_pipeline")
    def test_calls_inference_pipeline(
        self,
        mock_infer: MagicMock,
        dummy_ecg_file: str,
    ) -> None:
        """_run_pipeline calls run_inference_pipeline with correct args."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _make_mock_result()
        mock_infer.return_value = mock_response

        result = _run_pipeline(
            dummy_ecg_file,
            format_override="csv",
            model=None,
            tasks=["rhythm"],
        )

        mock_infer.assert_called_once()
        call_kwargs = mock_infer.call_args
        assert call_kwargs.kwargs["format_override"] == "csv"
        assert call_kwargs.kwargs["enabled_tasks"] == ["rhythm"]
        assert "quality_report" in result

    @patch("aortica.api.predict.run_inference_pipeline")
    def test_reads_file_bytes(
        self,
        mock_infer: MagicMock,
        dummy_ecg_file: str,
    ) -> None:
        """_run_pipeline reads bytes from the file path."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _make_mock_result()
        mock_infer.return_value = mock_response

        _run_pipeline(
            dummy_ecg_file,
            format_override=None,
            model=None,
            tasks=list(_VALID_TASKS),
        )

        call_args = mock_infer.call_args
        file_bytes = call_args[0][0]
        assert isinstance(file_bytes, bytes)
        assert len(file_bytes) > 0
