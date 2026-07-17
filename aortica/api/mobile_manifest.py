"""Mobile model manifest API endpoint for OTA updates.

Provides ``POST /api/v1/mobile/model-manifest`` returning the latest
model version, download URL, SHA-256 hash, and minimum app version
required.  The Android app calls this on startup to check for OTA
model updates.

Example response::

    {
        "latest_version": "0.3.0",
        "download_url": "https://huggingface.co/nmillrr/aortica/resolve/v0.3.0/aortica_edge_int8_v0.3.0.onnx",
        "sha256": "abc123...",
        "min_app_version": "1.0.0",
        "file_size_bytes": 15000000
    }
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Environment variable overrides for model manifest fields.
ENV_MODEL_VERSION = "AORTICA_MOBILE_MODEL_VERSION"
ENV_MODEL_DOWNLOAD_URL = "AORTICA_MOBILE_MODEL_DOWNLOAD_URL"
ENV_MODEL_SHA256 = "AORTICA_MOBILE_MODEL_SHA256"
ENV_MODEL_MIN_APP_VERSION = "AORTICA_MOBILE_MIN_APP_VERSION"
ENV_MODEL_FILE_PATH = "AORTICA_MOBILE_MODEL_FILE_PATH"

#: Default HuggingFace Hub download URL template.
HF_DOWNLOAD_URL_TEMPLATE = (
    "https://huggingface.co/nmillrr/aortica/resolve/v{version}/"
    "aortica_edge_int8_v{version}.onnx"
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelManifestResponse(BaseModel):
    """Response schema for the mobile model manifest endpoint."""

    latest_version: str = Field(
        ...,
        description="Semantic version of the latest model (e.g. '0.3.0')",
    )
    download_url: str = Field(
        ...,
        description="URL to download the ONNX model file",
    )
    sha256: str = Field(
        ...,
        description="SHA-256 hex digest of the model file",
    )
    min_app_version: str = Field(
        ...,
        description="Minimum Android app version required for this model",
    )
    file_size_bytes: int = Field(
        default=0,
        description="Size of the model file in bytes",
    )


class ModelManifestRequest(BaseModel):
    """Optional request body for the model manifest endpoint.

    The Android app may send its current model version and app version
    so the server can make smarter decisions.
    """

    current_model_version: Optional[str] = Field(
        default=None,
        description="Client's currently installed model version",
    )
    app_version: Optional[str] = Field(
        default=None,
        description="Client's app version",
    )
    device_id: Optional[str] = Field(
        default=None,
        description="Optional device identifier for analytics",
    )


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_package_version() -> str:
    """Get the current aortica package version."""
    try:
        import aortica

        return aortica.__version__
    except (ImportError, AttributeError):
        return "0.2.0"


def build_model_manifest(
    model_file_path: Optional[str] = None,
    version: Optional[str] = None,
    download_url: Optional[str] = None,
    min_app_version: Optional[str] = None,
) -> ModelManifestResponse:
    """Build a model manifest from configuration and/or a local model file.

    Resolution order for each field:

    1. Explicit function arguments
    2. Environment variable overrides
    3. Computed defaults (version from package, URL from HF template, etc.)

    Args:
        model_file_path: Path to the local ONNX model file for computing
            SHA-256 and file size.  Falls back to ``AORTICA_MOBILE_MODEL_FILE_PATH``
            env var.
        version: Model version string.  Falls back to
            ``AORTICA_MOBILE_MODEL_VERSION`` env var or package version.
        download_url: Download URL.  Falls back to
            ``AORTICA_MOBILE_MODEL_DOWNLOAD_URL`` env var or HF template.
        min_app_version: Minimum app version.  Falls back to
            ``AORTICA_MOBILE_MIN_APP_VERSION`` env var or ``"1.0.0"``.

    Returns:
        Populated :class:`ModelManifestResponse`.
    """
    # Resolve version
    resolved_version = (
        version
        or os.environ.get(ENV_MODEL_VERSION)
        or _get_package_version()
    )

    # Resolve download URL
    resolved_url = (
        download_url
        or os.environ.get(ENV_MODEL_DOWNLOAD_URL)
        or HF_DOWNLOAD_URL_TEMPLATE.format(version=resolved_version)
    )

    # Resolve min app version
    resolved_min_app = (
        min_app_version
        or os.environ.get(ENV_MODEL_MIN_APP_VERSION)
        or "1.0.0"
    )

    # Resolve SHA-256 and file size from a local model file
    resolved_sha256 = os.environ.get(ENV_MODEL_SHA256, "")
    file_size = 0

    file_path = model_file_path or os.environ.get(ENV_MODEL_FILE_PATH)
    if file_path:
        path = Path(file_path)
        if path.exists():
            resolved_sha256 = _compute_sha256(path)
            file_size = path.stat().st_size

    return ModelManifestResponse(
        latest_version=resolved_version,
        download_url=resolved_url,
        sha256=resolved_sha256,
        min_app_version=resolved_min_app,
        file_size_bytes=file_size,
    )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------


def create_mobile_router():  # type: ignore[no-untyped-def]
    """Create and return a FastAPI router for mobile endpoints.

    Returns:
        ``fastapi.APIRouter`` with the mobile model manifest endpoint.
    """
    try:
        from fastapi import APIRouter
    except ImportError:
        raise ImportError(
            "FastAPI is required for mobile API endpoints. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/mobile", tags=["mobile"])

    @router.post(
        "/model-manifest",
        response_model=ModelManifestResponse,
        summary="Get latest mobile model manifest for OTA updates",
        description=(
            "Returns the latest ONNX edge model version, download URL, "
            "SHA-256 hash, and minimum app version required. The Android "
            "app calls this on startup to check for model updates."
        ),
    )
    async def get_model_manifest(
        request: Optional[ModelManifestRequest] = None,
    ) -> ModelManifestResponse:
        """Return the current mobile model manifest."""
        if request and request.device_id:
            logger.info(
                "Model manifest requested by device %s "
                "(current model: %s, app: %s)",
                request.device_id,
                request.current_model_version,
                request.app_version,
            )

        return build_model_manifest()

    return router
