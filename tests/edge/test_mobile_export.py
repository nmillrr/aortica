"""Tests for aortica.edge.mobile_export — Android mobile model export utility."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aortica.edge.mobile_export import (
    MobileExportResult,
    MobileModelMetadata,
    export_mobile_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_onnx_model(tmp_path: Path) -> Path:
    """Create a minimal fake ONNX file for testing (not a valid model)."""
    model_path = tmp_path / "test_model.onnx"
    # Write minimal ONNX magic bytes + some content
    model_path.write_bytes(b"\x08\x09" * 512)  # 1 KB of data
    return model_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a clean output directory."""
    out = tmp_path / "mobile_bundle"
    return out


# ---------------------------------------------------------------------------
# Tests: MobileModelMetadata
# ---------------------------------------------------------------------------


class TestMobileModelMetadata:
    """Tests for MobileModelMetadata dataclass."""

    def test_default_construction(self) -> None:
        metadata = MobileModelMetadata(version="1.0.0")
        assert metadata.version == "1.0.0"
        assert metadata.model_filename == "model.onnx"
        assert metadata.sample_rate_hz == 500
        assert metadata.quantization == "INT8"
        assert len(metadata.supported_lead_configs) == 4

    def test_to_dict(self) -> None:
        metadata = MobileModelMetadata(version="0.2.0")
        d = metadata.to_dict()
        assert isinstance(d, dict)
        assert d["version"] == "0.2.0"
        assert d["model_filename"] == "model.onnx"
        assert "input_spec" in d
        assert "task_heads" in d

    def test_from_dict_roundtrip(self) -> None:
        original = MobileModelMetadata(
            version="1.2.3",
            sha256="abc123",
            file_size_bytes=42,
        )
        d = original.to_dict()
        restored = MobileModelMetadata.from_dict(d)
        assert restored.version == "1.2.3"
        assert restored.sha256 == "abc123"
        assert restored.file_size_bytes == 42

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = {"version": "1.0.0", "unknown_field": "ignored"}
        metadata = MobileModelMetadata.from_dict(d)
        assert metadata.version == "1.0.0"

    def test_default_task_heads(self) -> None:
        metadata = MobileModelMetadata(version="1.0.0")
        assert len(metadata.task_heads) == 4
        head_names = {h["name"] for h in metadata.task_heads}
        assert head_names == {"rhythm", "structural", "ischaemia", "risk"}

    def test_default_input_spec(self) -> None:
        metadata = MobileModelMetadata(version="1.0.0")
        assert metadata.input_spec["name"] == "ecg_input"
        assert metadata.input_spec["shape"] == [1, 12, 5000]
        assert metadata.input_spec["dtype"] == "float32"


# ---------------------------------------------------------------------------
# Tests: export_mobile_model
# ---------------------------------------------------------------------------


class TestExportMobileModel:
    """Tests for the export_mobile_model function."""

    def test_basic_export(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
            version="0.3.0",
        )

        assert isinstance(result, MobileExportResult)
        assert result.success is True
        assert result.error is None

        # Check files exist
        assert (output_dir / "model.onnx").exists()
        assert (output_dir / "model_metadata.json").exists()

    def test_metadata_json_valid(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
            version="0.5.0",
        )

        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path) as f:
            data = json.load(f)

        assert data["version"] == "0.5.0"
        assert data["model_filename"] == "model.onnx"
        assert data["quantization"] == "INT8"
        assert len(data["sha256"]) == 64  # SHA-256 hex string
        assert data["file_size_bytes"] > 0
        assert data["sample_rate_hz"] == 500

    def test_sha256_integrity(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
        )

        assert len(result.metadata.sha256) == 64
        # SHA-256 should be deterministic
        result2 = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
        )
        assert result.metadata.sha256 == result2.metadata.sha256

    def test_file_not_found(self, output_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            export_mobile_model(
                model_path="/nonexistent/model.onnx",
                output_path=output_dir,
            )

    def test_non_onnx_extension(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        bad_file = tmp_path / "model.pt"
        bad_file.write_bytes(b"\x00" * 100)

        with pytest.raises(ValueError, match="Expected .onnx file"):
            export_mobile_model(
                model_path=bad_file,
                output_path=output_dir,
            )

    def test_custom_task_heads(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        custom_heads = [
            {"name": "rhythm", "output_name": "rhythm_out", "num_classes": 10, "type": "classification"}
        ]
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
            task_heads=custom_heads,
        )

        assert len(result.metadata.task_heads) == 1
        assert result.metadata.task_heads[0]["name"] == "rhythm"

    def test_custom_sample_rate(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
            sample_rate_hz=250,
        )

        assert result.metadata.sample_rate_hz == 250

    def test_model_file_copied(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
        )

        # Verify the model file content matches
        original_content = dummy_onnx_model.read_bytes()
        copied_content = (output_dir / "model.onnx").read_bytes()
        assert original_content == copied_content

    def test_creates_output_directory(
        self, dummy_onnx_model: Path, tmp_path: Path
    ) -> None:
        nested_output = tmp_path / "deep" / "nested" / "output"
        assert not nested_output.exists()

        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=nested_output,
        )

        assert nested_output.exists()
        assert (nested_output / "model.onnx").exists()

    def test_min_app_version(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
            min_app_version="2.0.0",
        )

        assert result.metadata.min_app_version == "2.0.0"

    def test_result_paths_are_absolute(
        self, dummy_onnx_model: Path, output_dir: Path
    ) -> None:
        result = export_mobile_model(
            model_path=dummy_onnx_model,
            output_path=output_dir,
        )

        assert result.output_dir.is_absolute()
        assert result.model_path.is_absolute()
        assert result.metadata_path.is_absolute()
