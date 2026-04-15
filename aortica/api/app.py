"""FastAPI application factory with health, info, and predict endpoints."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, File, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

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
    model: Any = None,
    conformal_predictor: Any = None,
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
    model:
        An optional pre-loaded ``AorticaModel`` instance.
    conformal_predictor:
        An optional fitted ``ConformalPredictor`` instance.

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
    app.state.model = model  # type: ignore[attr-defined]
    app.state.conformal_predictor = conformal_predictor  # type: ignore[attr-defined]

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

    # ---- POST /api/v1/predict -------------------------------------------

    @app.post(
        "/api/v1/predict",
        tags=["inference"],
        summary="Single ECG inference",
    )
    async def predict(
        file: UploadFile = File(..., description="ECG file to analyse"),
        format: Optional[str] = Query(  # noqa: A002
            default=None,
            description="Explicit format override (e.g. wfdb, dicom, csv)",
        ),
        include_xai: bool = Query(
            default=False,
            description="Include XAI attribution data in response",
        ),
    ) -> Any:
        """Upload a single ECG file and receive multi-task AI predictions.

        Runs the full pipeline: read_ecg → denoise → score_quality →
        model inference → uncertainty estimation.

        When ``include_xai=true``, integrated gradient attributions are
        computed for each active task head and included in the response.

        Returns ``422`` for unsupported or unparseable formats.
        """
        from aortica.api.predict import PredictResponse, run_inference_pipeline
        from aortica.io.dispatcher import UnsupportedFormatError

        file_bytes = await file.read()
        filename = file.filename or "upload.dat"

        try:
            result: PredictResponse = run_inference_pipeline(
                file_bytes,
                filename,
                format_override=format,
                model=app.state.model,  # type: ignore[attr-defined]
                conformal_predictor=app.state.conformal_predictor,  # type: ignore[attr-defined]
                enabled_tasks=list(app.state.enabled_tasks),  # type: ignore[attr-defined]
                include_xai=include_xai,
            )
        except UnsupportedFormatError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )
        except (ValueError, OSError) as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        return result

    # ---- POST /api/v1/predict/batch -------------------------------------

    @app.post(
        "/api/v1/predict/batch",
        tags=["inference"],
        summary="Batch ECG inference",
    )
    async def predict_batch(
        files: List[UploadFile] = File(
            ..., description="ECG files to analyse"
        ),
        format: Optional[str] = Query(  # noqa: A002
            default=None,
            description="Explicit format override applied to all files",
        ),
    ) -> Any:
        """Upload multiple ECG files and receive per-file AI predictions.

        Runs the full pipeline for each file:
        read_ecg → denoise → score_quality → model inference.

        Returns per-file status (success/error) with error messages for
        any files that fail processing.  Maximum batch size is
        configurable (default 50).
        """
        from aortica.api.batch_predict import (
            BatchPredictResponse,
            run_batch_inference,
        )

        max_batch_size: int = getattr(
            app.state, "max_batch_size", 50
        )

        # Read all file contents
        file_data: List[tuple[bytes, str]] = []
        for f in files:
            content = await f.read()
            file_data.append((content, f.filename or "upload.dat"))

        if len(file_data) > max_batch_size:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": (
                        f"Batch size {len(file_data)} exceeds maximum "
                        f"allowed batch size of {max_batch_size}"
                    )
                },
            )

        result: BatchPredictResponse = run_batch_inference(
            file_data,
            format_override=format,
            model=app.state.model,  # type: ignore[attr-defined]
            conformal_predictor=app.state.conformal_predictor,  # type: ignore[attr-defined]
            enabled_tasks=list(app.state.enabled_tasks),  # type: ignore[attr-defined]
            max_batch_size=max_batch_size,
        )

        return result

    return app
