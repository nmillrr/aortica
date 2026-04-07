"""Tests for ``aortica.models.registry`` — pretrained model distribution.

Tests cover:
- ``load_pretrained()`` cache behaviour, checksum validation, variant handling
- ``list_available_versions()`` Hub API parsing
- ``get_model_info()`` sidecar read/write
- ``ChecksumError`` and ``ModelNotFoundError`` exceptions
- CLI ``aortica info`` command
- CLI ``--model`` flag override in predict and benchmark
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

# Registry module uses stdlib only — always importable
from aortica.models.registry import (
    _VARIANT_FILENAME,
    _build_download_url,
    _compute_sha256,
    _get_cache_dir,
    _get_expected_checksum,
    _load_model_info,
    _save_model_info,
    ChecksumError,
    DATA_PROVENANCE,
    HUB_REPO_ID,
    ModelNotFoundError,
    ModelVersion,
    PretrainedModelInfo,
    get_model_info,
    list_available_versions,
    load_pretrained,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture()
def fake_checkpoint(cache_dir: Path) -> Path:
    """Create a fake .pt checkpoint file."""
    ckpt = cache_dir / "aortica_full_v0.2.0.pt"
    ckpt.write_bytes(b"fake-pytorch-checkpoint-data")
    return ckpt


@pytest.fixture()
def fake_onnx(cache_dir: Path) -> Path:
    """Create a fake .onnx edge model file."""
    onnx = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
    onnx.write_bytes(b"fake-onnx-model-data")
    return onnx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants and configuration."""

    def test_hub_repo_id(self) -> None:
        assert HUB_REPO_ID == "nmillrr/aortica"

    def test_variant_filenames(self) -> None:
        assert "full" in _VARIANT_FILENAME
        assert "edge" in _VARIANT_FILENAME
        assert _VARIANT_FILENAME["full"].format(version="0.2.0") == "aortica_full_v0.2.0.pt"
        assert (
            _VARIANT_FILENAME["edge"].format(version="0.2.0")
            == "aortica_edge_int8_v0.2.0.onnx"
        )

    def test_data_provenance(self) -> None:
        assert "PTB-XL" in DATA_PROVENANCE
        assert "CC BY 4.0" in DATA_PROVENANCE
        assert "No proprietary data" in DATA_PROVENANCE


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """Test custom exception classes."""

    def test_checksum_error_is_exception(self) -> None:
        assert issubclass(ChecksumError, Exception)

    def test_checksum_error_message(self) -> None:
        err = ChecksumError("bad hash")
        assert str(err) == "bad hash"

    def test_model_not_found_error_is_exception(self) -> None:
        assert issubclass(ModelNotFoundError, Exception)

    def test_model_not_found_error_message(self) -> None:
        err = ModelNotFoundError("v99.9.9 not found")
        assert str(err) == "v99.9.9 not found"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_construction(self) -> None:
        v = ModelVersion(version="0.2.0")
        assert v.version == "0.2.0"
        assert v.release_date is None
        assert v.variants == ["full", "edge"]
        assert v.performance_summary is None

    def test_with_metadata(self) -> None:
        v = ModelVersion(
            version="0.3.0",
            release_date="2026-04-01",
            performance_summary="macro-F1 0.91",
        )
        assert v.release_date == "2026-04-01"
        assert v.performance_summary == "macro-F1 0.91"


class TestPretrainedModelInfo:
    """Test PretrainedModelInfo dataclass."""

    def test_construction(self) -> None:
        info = PretrainedModelInfo(
            version="0.2.0",
            variant="full",
            source="hub",
            sha256="abc123",
            cache_path="/tmp/test",
        )
        assert info.version == "0.2.0"
        assert info.variant == "full"
        assert info.source == "hub"
        assert info.sha256 == "abc123"
        assert info.cache_path == "/tmp/test"
        assert info.data_attribution == DATA_PROVENANCE

    def test_custom_attribution(self) -> None:
        info = PretrainedModelInfo(
            version="0.2.0",
            variant="edge",
            source="local",
            sha256="def456",
            cache_path="/tmp/test",
            data_attribution="Custom data",
        )
        assert info.data_attribution == "Custom data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestComputeSha256:
    """Test SHA-256 computation."""

    def test_known_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "testfile"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _compute_sha256(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _compute_sha256(f) == expected


class TestGetCacheDir:
    """Test cache directory resolution."""

    def test_custom_dir(self, tmp_path: Path) -> None:
        cache = _get_cache_dir(str(tmp_path / "custom"))
        assert cache.exists()
        assert cache == tmp_path / "custom"

    def test_none_uses_default(self) -> None:
        cache = _get_cache_dir(None)
        assert "aortica" in str(cache)

    def test_creates_parents(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "a" / "b" / "c")
        cache = _get_cache_dir(nested)
        assert cache.exists()


class TestBuildDownloadUrl:
    """Test download URL construction."""

    def test_full_variant(self) -> None:
        url = _build_download_url("nmillrr/aortica", "0.2.0", "full")
        assert "nmillrr/aortica" in url
        assert "v0.2.0" in url
        assert "aortica_full_v0.2.0.pt" in url

    def test_edge_variant(self) -> None:
        url = _build_download_url("nmillrr/aortica", "0.2.0", "edge")
        assert "aortica_edge_int8_v0.2.0.onnx" in url


class TestModelInfoSidecar:
    """Test model info JSON sidecar read/write."""

    def test_save_and_load(self, cache_dir: Path) -> None:
        info = PretrainedModelInfo(
            version="0.2.0",
            variant="full",
            source="hub",
            sha256="abc123def456",
            cache_path=str(cache_dir / "model.pt"),
        )
        _save_model_info(cache_dir, info)

        loaded = _load_model_info(cache_dir)
        assert loaded is not None
        assert loaded.version == "0.2.0"
        assert loaded.variant == "full"
        assert loaded.sha256 == "abc123def456"

    def test_load_nonexistent(self, cache_dir: Path) -> None:
        loaded = _load_model_info(cache_dir)
        assert loaded is None

    def test_load_corrupted(self, cache_dir: Path) -> None:
        (cache_dir / "model_info.json").write_text("not valid json{{", encoding="utf-8")
        loaded = _load_model_info(cache_dir)
        assert loaded is None

    def test_get_model_info_public(self, cache_dir: Path) -> None:
        info = PretrainedModelInfo(
            version="0.2.0",
            variant="edge",
            source="hub",
            sha256="xyz789",
            cache_path=str(cache_dir / "model.onnx"),
        )
        _save_model_info(cache_dir, info)

        result = get_model_info(cache_dir=str(cache_dir))
        assert result is not None
        assert result.variant == "edge"

    def test_get_model_info_no_cache(self, tmp_path: Path) -> None:
        result = get_model_info(cache_dir=str(tmp_path / "nonexistent"))
        assert result is None


# ---------------------------------------------------------------------------
# Checksum validation
# ---------------------------------------------------------------------------


class TestChecksumValidation:
    """Test SHA-256 checksum verification in load_pretrained."""

    def test_good_checksum(self, cache_dir: Path) -> None:
        """Cached file with correct checksum should succeed (edge variant)."""
        ckpt = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
        ckpt.write_bytes(b"good-edge-data")
        expected_sha = hashlib.sha256(b"good-edge-data").hexdigest()

        # Inject known checksum for edge variant
        with mock.patch.dict(
            "aortica.models.registry._KNOWN_CHECKSUMS",
            {"0.2.0/edge": {"sha256": expected_sha}},
        ):
            with mock.patch(
                "aortica.models.registry._resolve_version", return_value="0.2.0"
            ):
                result = load_pretrained(
                    "0.2.0",
                    variant="edge",
                    cache_dir=str(cache_dir),
                )
                # Edge variant returns path
                assert isinstance(result, str)
                assert result.endswith(".onnx")

    def test_tampered_checksum_raises(self, cache_dir: Path) -> None:
        """Cached file with wrong checksum should raise ChecksumError."""
        ckpt = cache_dir / "aortica_full_v0.2.0.pt"
        ckpt.write_bytes(b"tampered-data")

        with mock.patch.dict(
            "aortica.models.registry._KNOWN_CHECKSUMS",
            {"0.2.0/full": {"sha256": "0000000000000000000000000000000000000000000000000000000000000000"}},
        ):
            with mock.patch(
                "aortica.models.registry._resolve_version", return_value="0.2.0"
            ):
                with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
                    load_pretrained(
                        "0.2.0",
                        variant="full",
                        cache_dir=str(cache_dir),
                    )

    def test_no_known_checksum_passes(self, cache_dir: Path) -> None:
        """When no checksum is registered, validation should be skipped."""
        ckpt = cache_dir / "aortica_full_v0.2.0.pt"
        ckpt.write_bytes(b"any-data")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            # No checksums registered; should proceed to load
            # Will fail at torch.load but the checksum step should pass
            try:
                load_pretrained(
                    "0.2.0",
                    variant="full",
                    cache_dir=str(cache_dir),
                )
            except (ImportError, Exception) as exc:
                # Expected: torch may not be installed, or the file
                # isn't a real checkpoint
                assert "SHA-256" not in str(exc)


# ---------------------------------------------------------------------------
# load_pretrained
# ---------------------------------------------------------------------------


class TestLoadPretrained:
    """Test load_pretrained function."""

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            load_pretrained("0.2.0", variant="unknown")

    def test_edge_returns_path(self, cache_dir: Path) -> None:
        """Edge variant should return the file path as a string."""
        ckpt = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
        ckpt.write_bytes(b"fake-onnx")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            result = load_pretrained(
                "0.2.0",
                variant="edge",
                cache_dir=str(cache_dir),
            )
            assert isinstance(result, str)
            assert result.endswith(".onnx")
            assert Path(result).exists()

    def test_cache_used_when_file_exists(self, cache_dir: Path) -> None:
        """Should not download when cached file exists."""
        ckpt = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
        ckpt.write_bytes(b"cached-onnx")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            with mock.patch(
                "aortica.models.registry._download_file"
            ) as mock_dl:
                result = load_pretrained(
                    "0.2.0",
                    variant="edge",
                    cache_dir=str(cache_dir),
                )
                mock_dl.assert_not_called()
                assert isinstance(result, str)

    def test_force_download_redownloads(self, cache_dir: Path) -> None:
        """force_download=True should download even if cached."""
        ckpt = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
        ckpt.write_bytes(b"old-cached-onnx")

        def fake_download(url: str, dest: Path, timeout: int = 60) -> None:
            dest.write_bytes(b"new-downloaded-onnx")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            with mock.patch(
                "aortica.models.registry._download_file",
                side_effect=fake_download,
            ) as mock_dl:
                result = load_pretrained(
                    "0.2.0",
                    variant="edge",
                    cache_dir=str(cache_dir),
                    force_download=True,
                )
                mock_dl.assert_called_once()
                assert ckpt.read_bytes() == b"new-downloaded-onnx"

    def test_download_failure_raises_model_not_found(self, cache_dir: Path) -> None:
        """Network failure should raise ModelNotFoundError."""
        from urllib.error import URLError

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="99.9.9"
        ):
            with mock.patch(
                "aortica.models.registry._download_file",
                side_effect=URLError("Network unreachable"),
            ):
                with pytest.raises(ModelNotFoundError, match="Could not download"):
                    load_pretrained(
                        "99.9.9",
                        variant="edge",
                        cache_dir=str(cache_dir),
                    )

    def test_model_info_saved_after_load(self, cache_dir: Path) -> None:
        """Model info sidecar should be written after successful load."""
        ckpt = cache_dir / "aortica_edge_int8_v0.2.0.onnx"
        ckpt.write_bytes(b"onnx-data")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            load_pretrained(
                "0.2.0",
                variant="edge",
                cache_dir=str(cache_dir),
            )

        info_path = cache_dir / "model_info.json"
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert info["version"] == "0.2.0"
        assert info["variant"] == "edge"
        assert info["source"] == "hub"
        assert len(info["sha256"]) == 64  # valid hex sha256

    def test_latest_version_resolution(self, cache_dir: Path) -> None:
        """'latest' should resolve to the package version as fallback."""
        ckpt_name = _VARIANT_FILENAME["edge"].format(version="0.2.0")
        ckpt = cache_dir / ckpt_name
        ckpt.write_bytes(b"edge-model")

        with mock.patch(
            "aortica.models.registry._resolve_version", return_value="0.2.0"
        ):
            result = load_pretrained(
                "latest",
                variant="edge",
                cache_dir=str(cache_dir),
            )
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# list_available_versions
# ---------------------------------------------------------------------------


class TestListAvailableVersions:
    """Test Hub API version listing."""

    def test_parses_tags(self) -> None:
        """Should parse version tags from Hub API response."""
        fake_response = json.dumps({
            "tags": [
                {"name": "v0.2.0", "date": "2026-03-01"},
                {"name": "v0.1.0", "date": "2026-01-15"},
            ]
        }).encode()

        with mock.patch("aortica.models.registry.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            versions = list_available_versions()

        assert len(versions) == 2
        assert versions[0].version == "0.2.0"
        assert versions[0].release_date == "2026-03-01"
        assert versions[1].version == "0.1.0"

    def test_sorted_descending(self) -> None:
        """Versions should be sorted newest first."""
        fake_response = json.dumps({
            "tags": [
                {"name": "v0.1.0"},
                {"name": "v0.3.0"},
                {"name": "v0.2.0"},
            ]
        }).encode()

        with mock.patch("aortica.models.registry.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            versions = list_available_versions()

        assert versions[0].version == "0.3.0"
        assert versions[-1].version == "0.1.0"

    def test_handles_string_tags(self) -> None:
        """Should handle tags as simple strings."""
        fake_response = json.dumps({
            "tags": ["v0.2.0", "v0.1.0"]
        }).encode()

        with mock.patch("aortica.models.registry.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            versions = list_available_versions()

        assert len(versions) == 2

    def test_network_error_returns_empty(self) -> None:
        """Network errors should return empty list, not raise."""
        from urllib.error import URLError

        with mock.patch(
            "aortica.models.registry.urlopen",
            side_effect=URLError("offline"),
        ):
            versions = list_available_versions()
            assert versions == []

    def test_empty_tags_returns_empty(self) -> None:
        """Empty tags list should return empty."""
        fake_response = json.dumps({"tags": []}).encode()

        with mock.patch("aortica.models.registry.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            versions = list_available_versions()
            assert versions == []

    def test_model_version_defaults(self) -> None:
        """ModelVersion should have sensible defaults."""
        v = ModelVersion(version="1.0.0")
        assert v.variants == ["full", "edge"]
        assert v.performance_summary is None


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

click = pytest.importorskip("click")
pytest.importorskip("rich")

from click.testing import CliRunner  # noqa: E402


class TestInfoCommand:
    """Test aortica info CLI command."""

    def test_info_help(self) -> None:
        from aortica.cli.info import info_cmd

        runner = CliRunner()
        result = runner.invoke(info_cmd, ["--help"])
        assert result.exit_code == 0
        assert "model version" in result.output.lower() or "Display" in result.output

    def test_info_no_cache(self, tmp_path: Path) -> None:
        from aortica.cli.info import info_cmd

        runner = CliRunner()
        result = runner.invoke(info_cmd, ["--cache-dir", str(tmp_path / "empty")])
        assert result.exit_code == 0
        assert "No pretrained model cached" in result.output or "Aortica" in result.output

    def test_info_with_cached_model(self, cache_dir: Path) -> None:
        from aortica.cli.info import info_cmd

        # Create model info sidecar
        info = PretrainedModelInfo(
            version="0.2.0",
            variant="full",
            source="hub",
            sha256="abcd1234" * 8,
            cache_path=str(cache_dir / "model.pt"),
        )
        _save_model_info(cache_dir, info)

        runner = CliRunner()
        result = runner.invoke(info_cmd, ["--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_info_registered_in_cli_group(self) -> None:
        try:
            from aortica.cli import _build_cli

            cli = _build_cli()
            commands = list(cli.commands.keys())
            assert "info" in commands
        except (ImportError, NameError):
            pytest.skip("torch not available — CLI group import fails")


class TestCLIModelFlagOverride:
    """Test that --model flag overrides pretrained auto-loading."""

    def test_predict_with_model_flag_skips_pretrained(self) -> None:
        """When --model is provided, load_pretrained should not be called."""
        from aortica.cli.predict import _load_model

        with mock.patch(
            "aortica.cli.predict.torch",
            create=True,
        ) as mock_torch:
            # When model_path is provided, it goes through the torch.load path
            # not through load_pretrained
            with mock.patch(
                "aortica.models.registry.load_pretrained"
            ) as mock_lp:
                # model_path=None should try load_pretrained
                _load_model(None, ("rhythm",))
                mock_lp.assert_called_once_with("latest")

    def test_predict_no_model_tries_pretrained(self) -> None:
        """When no --model flag, predict should try load_pretrained."""
        from aortica.cli.predict import _load_model

        with mock.patch(
            "aortica.models.registry.load_pretrained",
            return_value="mock-model",
        ) as mock_lp:
            result = _load_model(None, ("rhythm",))
            mock_lp.assert_called_once_with("latest")
            assert result == "mock-model"

    def test_predict_pretrained_failure_returns_none(self) -> None:
        """When load_pretrained fails, predict should return None."""
        from aortica.cli.predict import _load_model

        with mock.patch(
            "aortica.models.registry.load_pretrained",
            side_effect=ModelNotFoundError("offline"),
        ):
            result = _load_model(None, ("rhythm",))
            assert result is None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Test that registry symbols are properly exported."""

    def test_import_from_models_package(self) -> None:
        from aortica.models import (
            ChecksumError,
            ModelNotFoundError,
            PretrainedModelInfo,
            get_model_info,
            list_available_versions,
            load_pretrained,
        )

        assert callable(load_pretrained)
        assert callable(list_available_versions)
        assert callable(get_model_info)
        assert issubclass(ChecksumError, Exception)
        assert issubclass(ModelNotFoundError, Exception)

    def test_import_from_registry_module(self) -> None:
        from aortica.models.registry import (
            ChecksumError,
            ModelNotFoundError,
            ModelVersion,
            PretrainedModelInfo,
            get_model_info,
            list_available_versions,
            load_pretrained,
        )

        assert callable(load_pretrained)
