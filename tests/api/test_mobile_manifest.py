"""Tests for aortica.api.mobile_manifest — mobile model manifest endpoint."""

from __future__ import annotations

import hashlib
import os
import tempfile
from unittest.mock import patch

import pytest

from aortica.api.mobile_manifest import (
    ModelManifestRequest,
    ModelManifestResponse,
    build_model_manifest,
)

# ---------------------------------------------------------------------------
# build_model_manifest tests
# ---------------------------------------------------------------------------


class TestBuildModelManifest:
    """Tests for the manifest builder function."""

    def test_default_manifest(self) -> None:
        """build_model_manifest returns a valid manifest with defaults."""
        manifest = build_model_manifest()
        assert isinstance(manifest, ModelManifestResponse)
        assert manifest.latest_version
        assert manifest.download_url
        assert manifest.min_app_version == "1.0.0"

    def test_explicit_version(self) -> None:
        """Explicit version parameter takes precedence."""
        manifest = build_model_manifest(version="0.5.0")
        assert manifest.latest_version == "0.5.0"
        assert "0.5.0" in manifest.download_url

    def test_explicit_download_url(self) -> None:
        """Explicit download_url overrides the HF template."""
        url = "https://custom.example.com/model.onnx"
        manifest = build_model_manifest(download_url=url)
        assert manifest.download_url == url

    def test_explicit_min_app_version(self) -> None:
        """Explicit min_app_version overrides the default."""
        manifest = build_model_manifest(min_app_version="2.0.0")
        assert manifest.min_app_version == "2.0.0"

    def test_env_var_version(self) -> None:
        """AORTICA_MOBILE_MODEL_VERSION env var sets the version."""
        with patch.dict(os.environ, {"AORTICA_MOBILE_MODEL_VERSION": "0.9.0"}):
            manifest = build_model_manifest()
        assert manifest.latest_version == "0.9.0"

    def test_env_var_download_url(self) -> None:
        """AORTICA_MOBILE_MODEL_DOWNLOAD_URL env var sets download URL."""
        custom_url = "https://env.example.com/edge.onnx"
        with patch.dict(os.environ, {"AORTICA_MOBILE_MODEL_DOWNLOAD_URL": custom_url}):
            manifest = build_model_manifest()
        assert manifest.download_url == custom_url

    def test_env_var_sha256(self) -> None:
        """AORTICA_MOBILE_MODEL_SHA256 env var provides the hash."""
        with patch.dict(os.environ, {"AORTICA_MOBILE_MODEL_SHA256": "deadbeef"}):
            manifest = build_model_manifest()
        assert manifest.sha256 == "deadbeef"

    def test_env_var_min_app_version(self) -> None:
        """AORTICA_MOBILE_MIN_APP_VERSION env var sets min app version."""
        with patch.dict(os.environ, {"AORTICA_MOBILE_MIN_APP_VERSION": "3.0.0"}):
            manifest = build_model_manifest()
        assert manifest.min_app_version == "3.0.0"

    def test_model_file_path_computes_sha_and_size(self) -> None:
        """When a local model file exists, compute SHA-256 and file size."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake onnx model content")
            f.flush()
            path = f.name

        try:
            manifest = build_model_manifest(model_file_path=path)

            expected_sha = hashlib.sha256(b"fake onnx model content").hexdigest()
            assert manifest.sha256 == expected_sha
            assert manifest.file_size_bytes == len(b"fake onnx model content")
        finally:
            os.unlink(path)

    def test_model_file_path_nonexistent(self) -> None:
        """Non-existent model file path produces empty sha/zero size."""
        manifest = build_model_manifest(model_file_path="/nonexistent/model.onnx")
        assert manifest.sha256 == ""
        assert manifest.file_size_bytes == 0

    def test_env_model_file_path(self) -> None:
        """AORTICA_MOBILE_MODEL_FILE_PATH env var triggers file inspection."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"env model")
            f.flush()
            path = f.name

        try:
            with patch.dict(os.environ, {"AORTICA_MOBILE_MODEL_FILE_PATH": path}):
                manifest = build_model_manifest()
            assert manifest.sha256 == hashlib.sha256(b"env model").hexdigest()
            assert manifest.file_size_bytes == 9
        finally:
            os.unlink(path)

    def test_explicit_params_override_env(self) -> None:
        """Explicit function args take precedence over env vars."""
        with patch.dict(os.environ, {"AORTICA_MOBILE_MODEL_VERSION": "0.1.0"}):
            manifest = build_model_manifest(version="0.9.9")
        assert manifest.latest_version == "0.9.9"

    def test_download_url_template_includes_version(self) -> None:
        """Default HuggingFace download URL includes the version string."""
        manifest = build_model_manifest(version="0.4.0")
        assert "0.4.0" in manifest.download_url
        assert "huggingface.co" in manifest.download_url


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestModelManifestResponse:
    """Tests for the Pydantic response model."""

    def test_serialization(self) -> None:
        """ModelManifestResponse serializes to expected JSON keys."""
        resp = ModelManifestResponse(
            latest_version="0.3.0",
            download_url="https://example.com/model.onnx",
            sha256="abc123",
            min_app_version="1.0.0",
            file_size_bytes=15000000,
        )
        data = resp.model_dump()
        assert data["latest_version"] == "0.3.0"
        assert data["download_url"] == "https://example.com/model.onnx"
        assert data["sha256"] == "abc123"
        assert data["min_app_version"] == "1.0.0"
        assert data["file_size_bytes"] == 15000000

    def test_default_file_size(self) -> None:
        """file_size_bytes defaults to 0."""
        resp = ModelManifestResponse(
            latest_version="0.1.0",
            download_url="https://example.com/model.onnx",
            sha256="hash",
            min_app_version="1.0.0",
        )
        assert resp.file_size_bytes == 0


class TestModelManifestRequest:
    """Tests for the optional request model."""

    def test_all_fields_optional(self) -> None:
        """All request fields are optional."""
        req = ModelManifestRequest()
        assert req.current_model_version is None
        assert req.app_version is None
        assert req.device_id is None

    def test_populated_request(self) -> None:
        """Request can be populated with all fields."""
        req = ModelManifestRequest(
            current_model_version="0.2.0",
            app_version="1.0.0",
            device_id="device-123",
        )
        assert req.current_model_version == "0.2.0"
        assert req.app_version == "1.0.0"
        assert req.device_id == "device-123"


# ---------------------------------------------------------------------------
# API endpoint integration test
# ---------------------------------------------------------------------------


class TestMobileManifestEndpoint:
    """Tests for the FastAPI endpoint via TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_client(self) -> None:
        """Create a TestClient for the endpoint tests."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        self.client = TestClient(app)

    def test_model_manifest_endpoint_returns_200(self) -> None:
        """POST /api/v1/mobile/model-manifest returns 200."""
        response = self.client.post("/api/v1/mobile/model-manifest")
        assert response.status_code == 200

    def test_model_manifest_response_structure(self) -> None:
        """Response has all required fields."""
        response = self.client.post("/api/v1/mobile/model-manifest")
        data = response.json()
        assert "latest_version" in data
        assert "download_url" in data
        assert "sha256" in data
        assert "min_app_version" in data
        assert "file_size_bytes" in data

    def test_model_manifest_with_request_body(self) -> None:
        """Endpoint accepts optional request body."""
        body = {
            "current_model_version": "0.2.0",
            "app_version": "1.0.0",
            "device_id": "test-device",
        }
        response = self.client.post(
            "/api/v1/mobile/model-manifest",
            json=body,
        )
        assert response.status_code == 200

    def test_model_manifest_empty_body(self) -> None:
        """Endpoint works with empty JSON body."""
        response = self.client.post(
            "/api/v1/mobile/model-manifest",
            json={},
        )
        assert response.status_code == 200

    def test_model_manifest_version_in_response(self) -> None:
        """Response version matches expected format."""
        response = self.client.post("/api/v1/mobile/model-manifest")
        data = response.json()
        # Version should be a non-empty string
        assert isinstance(data["latest_version"], str)
        assert len(data["latest_version"]) > 0

    def test_model_manifest_download_url_format(self) -> None:
        """Download URL should be a valid-looking URL."""
        response = self.client.post("/api/v1/mobile/model-manifest")
        data = response.json()
        assert data["download_url"].startswith("https://")
