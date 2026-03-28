"""FastAPI application factory with health and info endpoints."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

import aortica

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = Field(..., description="Service health status", examples=["ok"])


class InfoResponse(BaseModel):
    """Response model for GET /info."""

    name: str = Field(
        default="aortica",
        description="Application name",
    )
    version: str = Field(
        ...,
        description="Application version",
    )
    supported_formats: List[str] = Field(
        ...,
        description="ECG file formats the API can accept",
    )
    enabled_task_heads: List[str] = Field(
        ...,
        description="Currently enabled model task heads",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether a model checkpoint is currently loaded",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS: List[str] = [
    "wfdb",
    "dicom",
    "scp-ecg",
    "csv",
    "mat",
    "hl7-aecg",
    "xml",
]

DEFAULT_TASK_HEADS: List[str] = [
    "rhythm",
    "structural",
    "ischaemia",
    "risk",
]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _check_fastapi() -> None:
    """Raise *ImportError* if fastapi is not installed."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the Aortica API. "
            "Install it with: pip install aortica[api]"
        )


def create_app(
    *,
    cors_origins: Optional[Sequence[str]] = None,
    enabled_tasks: Optional[Sequence[str]] = None,
    model_loaded: bool = False,
) -> Any:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    cors_origins:
        Allowed CORS origins.  Defaults to ``["*"]`` (allow all).
    enabled_tasks:
        List of enabled task heads.  Defaults to all four heads.
    model_loaded:
        Whether a model checkpoint is loaded at startup.

    Returns
    -------
    FastAPI
        The configured application instance.
    """
    _check_fastapi()

    if cors_origins is None:
        cors_origins = ["*"]
    if enabled_tasks is None:
        enabled_tasks = list(DEFAULT_TASK_HEADS)

    app = FastAPI(
        title="Aortica ECG Analysis API",
        description="AI-powered multi-task ECG analysis platform",
        version=aortica.__version__,
    )

    # ---- CORS middleware ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- shared state ----
    app.state.enabled_tasks = list(enabled_tasks)  # type: ignore[attr-defined]
    app.state.model_loaded = model_loaded  # type: ignore[attr-defined]

    # ---- routes ----------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Service health check",
    )
    async def health() -> HealthResponse:
        """Return service health status."""
        return HealthResponse(status="ok")

    @app.get(
        "/info",
        response_model=InfoResponse,
        tags=["system"],
        summary="Service information",
    )
    async def info() -> InfoResponse:
        """Return model version, supported formats, and enabled task heads."""
        return InfoResponse(
            version=aortica.__version__,
            supported_formats=list(SUPPORTED_FORMATS),
            enabled_task_heads=list(app.state.enabled_tasks),  # type: ignore[attr-defined]
            model_loaded=app.state.model_loaded,  # type: ignore[attr-defined]
        )

    return app
