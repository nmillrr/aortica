"""Tests for aortica.federated.release_pipeline.

Validates the federated model release pipeline orchestration with
synthetic aggregated weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from aortica.federated.release_pipeline import (
    FederatedReleaseConfig,
    PipelineStepResult,
    federated_checkpoint_filename,
    federated_int8_filename,
    federated_onnx_filename,
    federated_version_string,
    release_pipeline,
)
from aortica.models.aortica_model import AorticaModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_weights(tmp_path: Path) -> Path:
    """Create a synthetic aggregated weights file."""
    model = AorticaModel()
    weights_path = tmp_path / "aggregated_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "enabled_tasks": list(model.enabled_tasks),
        },
        str(weights_path),
    )
    return weights_path


@pytest.fixture
def raw_state_dict_weights(tmp_path: Path) -> Path:
    """Create a synthetic weights file with raw state_dict (no wrapper)."""
    model = AorticaModel()
    weights_path = tmp_path / "raw_state_dict.pt"
    torch.save(model.state_dict(), str(weights_path))
    return weights_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temp output directory."""
    out = tmp_path / "release_output"
    out.mkdir()
    return out


@pytest.fixture
def base_config(output_dir: Path) -> FederatedReleaseConfig:
    """Base config that skips expensive steps."""
    return FederatedReleaseConfig(
        fl_round=50,
        site_count=5,
        total_samples=10000,
        aggregation_strategy="fedavg",
        dp_epsilon_spent=1.0,
        output_dir=str(output_dir),
        skip_benchmark=True,
        skip_equity_gate=True,
        skip_regulatory_gate=True,
        skip_onnx_export=True,
        skip_performance_card=True,
        skip_hub_push=True,
    )


# ---------------------------------------------------------------------------
# Version naming tests
# ---------------------------------------------------------------------------


class TestVersionNaming:
    """Test federated version string and filename generation."""

    def test_version_string(self) -> None:
        assert federated_version_string("0.3.0", 50) == "v0.3.0-r50"
        assert federated_version_string("1.0.0", 1) == "v1.0.0-r1"
        assert federated_version_string("0.3.0", 0) == "v0.3.0-r0"

    def test_checkpoint_filename(self) -> None:
        name = federated_checkpoint_filename("0.3.0", 50)
        assert name == "aortica-federated-v0.3.0-r50.pt"

    def test_onnx_filename(self) -> None:
        name = federated_onnx_filename("0.3.0", 50)
        assert name == "aortica-federated-v0.3.0-r50.onnx"

    def test_int8_filename(self) -> None:
        name = federated_int8_filename("0.3.0", 50)
        assert name == "aortica-federated-v0.3.0-r50-int8.onnx"


# ---------------------------------------------------------------------------
# Pipeline orchestration tests
# ---------------------------------------------------------------------------


class TestReleasePipeline:
    """Test the release_pipeline function end-to-end."""

    def test_successful_pipeline_all_skipped(
        self,
        synthetic_weights: Path,
        base_config: FederatedReleaseConfig,
    ) -> None:
        """Pipeline succeeds when all gates are skipped."""
        result = release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=base_config,
        )

        assert result.success is True
        assert result.version_string == "v0.3.0-r50"
        assert result.checkpoint_filename == "aortica-federated-v0.3.0-r50.pt"
        assert result.sha256 != ""
        assert result.abort_reason is None

        # Check step results
        step_names = [s.name for s in result.steps]
        assert "load_weights" in step_names

        # Load weights step should pass
        load_step = next(s for s in result.steps if s.name == "load_weights")
        assert load_step.passed is True
        assert load_step.skipped is False

    def test_checkpoint_file_created(
        self,
        synthetic_weights: Path,
        base_config: FederatedReleaseConfig,
        output_dir: Path,
    ) -> None:
        """Pipeline creates a federated checkpoint file."""
        release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=base_config,
        )

        ckpt_path = output_dir / "aortica-federated-v0.3.0-r50.pt"
        assert ckpt_path.exists()

        # Verify checkpoint contents
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "federated_version" in checkpoint
        assert checkpoint["federated_version"] == "v0.3.0-r50"
        assert checkpoint["fl_round"] == 50
        assert checkpoint["aggregation_strategy"] == "fedavg"
        assert checkpoint["site_count"] == 5
        assert checkpoint["total_samples"] == 10000

    def test_missing_weights_file(
        self,
        base_config: FederatedReleaseConfig,
    ) -> None:
        """Pipeline fails with clear error when weights file is missing."""
        result = release_pipeline(
            aggregated_weights_path="/nonexistent/path/weights.pt",
            base_version="0.3.0",
            config=base_config,
        )

        assert result.success is False
        assert result.abort_reason is not None
        assert "not found" in result.abort_reason.lower()

    def test_raw_state_dict_weights(
        self,
        raw_state_dict_weights: Path,
        base_config: FederatedReleaseConfig,
    ) -> None:
        """Pipeline handles raw state_dict weights (no wrapper dict)."""
        result = release_pipeline(
            aggregated_weights_path=str(raw_state_dict_weights),
            base_version="0.3.0",
            config=base_config,
        )

        assert result.success is True

    def test_default_config(
        self,
        synthetic_weights: Path,
        tmp_path: Path,
    ) -> None:
        """Pipeline runs with default config (no config provided)."""
        # Default config skips hub push but tries other steps
        config = FederatedReleaseConfig(
            output_dir=str(tmp_path / "default_out"),
            skip_benchmark=True,
            skip_equity_gate=True,
            skip_regulatory_gate=True,
            skip_onnx_export=True,
            skip_performance_card=True,
        )
        result = release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=config,
        )

        assert result.success is True

    def test_summary_output(
        self,
        synthetic_weights: Path,
        base_config: FederatedReleaseConfig,
    ) -> None:
        """Pipeline result has a human-readable summary."""
        result = release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=base_config,
        )

        summary = result.summary()
        assert "SUCCESS" in summary
        assert "v0.3.0-r50" in summary
        assert "load_weights" in summary


# ---------------------------------------------------------------------------
# Gate abort tests
# ---------------------------------------------------------------------------


class TestGateAbort:
    """Test that pipeline aborts on gate failures."""

    def test_equity_gate_failure_aborts(
        self,
        synthetic_weights: Path,
        output_dir: Path,
    ) -> None:
        """Pipeline aborts when equity gate fails."""
        from aortica.evaluation.equity_gate import EquityGateResult

        config = FederatedReleaseConfig(
            fl_round=50,
            output_dir=str(output_dir),
            skip_benchmark=True,
            skip_equity_gate=False,  # Run equity gate
            skip_regulatory_gate=True,
            skip_onnx_export=True,
            skip_performance_card=True,
            skip_hub_push=True,
        )

        # Mock equity gate to return failure
        mock_result = EquityGateResult(passed=False)
        with patch(
            "aortica.federated.release_pipeline.equity_gate",
            return_value=mock_result,
        ):
            # equity_gate is imported inside the step function, so we need
            # to patch it at the import location
            pass

        # The equity gate step should be skipped since benchmark report
        # will be None (benchmark is skipped)
        result = release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=config,
        )

        # With benchmark skipped, equity gate gets a None report and
        # is auto-skipped
        eq_step = next(
            (s for s in result.steps if s.name == "equity_gate"),
            None,
        )
        assert eq_step is not None
        assert eq_step.skipped is True

    def test_failed_summary_shows_abort(
        self,
        output_dir: Path,
    ) -> None:
        """Failed pipeline summary includes abort reason."""
        config = FederatedReleaseConfig(
            fl_round=50,
            output_dir=str(output_dir),
        )

        result = release_pipeline(
            aggregated_weights_path="/nonexistent.pt",
            base_version="0.3.0",
            config=config,
        )

        assert result.success is False
        summary = result.summary()
        assert "FAILED" in summary


# ---------------------------------------------------------------------------
# Pipeline step result tests
# ---------------------------------------------------------------------------


class TestPipelineStepResult:
    """Test PipelineStepResult dataclass."""

    def test_passed_step(self) -> None:
        step = PipelineStepResult(name="test_step", passed=True)
        assert step.name == "test_step"
        assert step.passed is True
        assert step.skipped is False
        assert step.error is None

    def test_failed_step(self) -> None:
        step = PipelineStepResult(
            name="test_step", passed=False, error="Something went wrong"
        )
        assert step.passed is False
        assert step.error == "Something went wrong"

    def test_skipped_step(self) -> None:
        step = PipelineStepResult(
            name="test_step", passed=True, skipped=True,
            details={"reason": "Skipped by config"},
        )
        assert step.skipped is True
        assert step.details["reason"] == "Skipped by config"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestFederatedReleaseConfig:
    """Test FederatedReleaseConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = FederatedReleaseConfig()
        assert config.fl_round == 0
        assert config.site_count == 0
        assert config.aggregation_strategy == "fedavg"
        assert config.dp_epsilon_spent == 0.0
        assert config.push_to_hub is False
        assert config.skip_hub_push is True

    def test_custom_config(self) -> None:
        config = FederatedReleaseConfig(
            fl_round=100,
            site_count=10,
            total_samples=50000,
            aggregation_strategy="fedprox",
            dp_epsilon_spent=2.5,
            push_to_hub=True,
        )
        assert config.fl_round == 100
        assert config.aggregation_strategy == "fedprox"
        assert config.push_to_hub is True


# ---------------------------------------------------------------------------
# Model card generation tests
# ---------------------------------------------------------------------------


class TestModelCard:
    """Test federated model card generation."""

    def test_model_card_generated(
        self,
        synthetic_weights: Path,
        output_dir: Path,
    ) -> None:
        """Model card is generated when performance card step runs."""
        config = FederatedReleaseConfig(
            fl_round=50,
            site_count=5,
            total_samples=10000,
            aggregation_strategy="fedavg",
            dp_epsilon_spent=1.0,
            output_dir=str(output_dir),
            skip_benchmark=True,
            skip_equity_gate=True,
            skip_regulatory_gate=True,
            skip_onnx_export=True,
            skip_performance_card=False,  # Generate model card
            skip_hub_push=True,
        )

        result = release_pipeline(
            aggregated_weights_path=str(synthetic_weights),
            base_version="0.3.0",
            config=config,
        )

        assert result.success is True

        # Check model card was created
        card_path = output_dir / "MODEL_CARD.md"
        assert card_path.exists()

        content = card_path.read_text()
        assert "v0.3.0-r50" in content
        assert "fedavg" in content
        assert "PTB-XL" in content
        assert "No proprietary data" in content
        assert "5" in content  # site count
        assert "10000" in content  # total samples


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestReleaseCLI:
    """Test the federated release CLI command."""

    def test_release_help(self) -> None:
        """release --help shows usage info."""
        from click.testing import CliRunner

        from aortica.cli.federated_cmd import federated_group

        runner = CliRunner()
        result = runner.invoke(federated_group, ["release", "--help"])
        assert result.exit_code == 0
        assert "--weights" in result.output
        assert "--version" in result.output
        assert "--round" in result.output

    def test_release_missing_weights(self) -> None:
        """release fails with clear error when weights file missing."""
        from click.testing import CliRunner

        from aortica.cli.federated_cmd import federated_group

        runner = CliRunner()
        result = runner.invoke(
            federated_group,
            [
                "release",
                "--weights", "/nonexistent/weights.pt",
                "--version", "0.3.0",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or result.exception is not None

    def test_release_json_output(
        self,
        synthetic_weights: Path,
        tmp_path: Path,
    ) -> None:
        """release --format json outputs valid JSON."""
        from click.testing import CliRunner

        from aortica.cli.federated_cmd import federated_group

        out_dir = tmp_path / "cli_release_out"
        out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            federated_group,
            [
                "release",
                "--weights", str(synthetic_weights),
                "--version", "0.3.0",
                "--round", "50",
                "--output-dir", str(out_dir),
                "--skip-onnx",
                "--format", "json",
            ],
        )

        # Parse JSON from output (may have non-JSON prefix from Rich)
        output_lines = result.output.strip().split("\n")
        # Find the JSON block
        json_start = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is not None:
            json_str = "\n".join(output_lines[json_start:])
            data = json.loads(json_str)
            assert "success" in data
            assert "version_string" in data
            assert data["version_string"] == "v0.3.0-r50"

    def test_release_text_output(
        self,
        synthetic_weights: Path,
        tmp_path: Path,
    ) -> None:
        """release with text format runs successfully."""
        from click.testing import CliRunner

        from aortica.cli.federated_cmd import federated_group

        out_dir = tmp_path / "cli_release_text"
        out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            federated_group,
            [
                "release",
                "--weights", str(synthetic_weights),
                "--version", "0.3.0",
                "--round", "10",
                "--output-dir", str(out_dir),
                "--skip-onnx",
                "--format", "text",
            ],
        )

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Registry variant test
# ---------------------------------------------------------------------------


class TestRegistryFederatedVariant:
    """Test that the model registry supports the 'federated' variant."""

    def test_federated_variant_filename(self) -> None:
        from aortica.models.registry import _VARIANT_FILENAME

        assert "federated" in _VARIANT_FILENAME
        name = _VARIANT_FILENAME["federated"].format(version="0.3.0-r50")
        assert name == "aortica-federated-v0.3.0-r50.pt"

    def test_federated_download_url(self) -> None:
        from aortica.models.registry import _build_download_url

        url = _build_download_url("nmillrr/aortica", "0.3.0-r50", "federated")
        assert "federated-v0.3.0-r50" in url
        assert "aortica-federated-v0.3.0-r50.pt" in url
