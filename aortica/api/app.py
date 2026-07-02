"""FastAPI application factory with health, info, and predict endpoints."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile
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
    enable_auth: bool = True,
    rate_limit_config: Any = None,
    rate_limit_config_path: Optional[str] = None,
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
    enable_auth:
        Whether to enforce authentication on ``/api/v1/`` endpoints.
        Defaults to ``True``.
    rate_limit_config:
        An optional ``RateLimitConfig`` instance for rate limiting.
        If *None*, rate limiting is configured from environment variables.
    rate_limit_config_path:
        Optional path to a ``rate_limits.yaml`` configuration file.

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

    # ---- Rate limiting middleware ----
    from aortica.api.rate_limiter import add_rate_limiting

    add_rate_limiting(
        app,
        config=rate_limit_config,
        config_path=rate_limit_config_path,
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
    app.state.enable_auth = enable_auth  # type: ignore[attr-defined]

    # ---- auth setup ----
    from aortica.api.auth import APIKeyStore, UserInfo, create_auth_router, require_auth

    api_key_store = APIKeyStore()
    app.state.api_key_store = api_key_store  # type: ignore[attr-defined]

    # Mount auth router (login/callback/token/refresh)
    auth_router = create_auth_router(api_key_store)
    app.include_router(auth_router)

    # Mount clinical suggestions router
    from aortica.api.clinical_suggestions import create_suggestions_router

    suggestions_router = create_suggestions_router()
    app.include_router(suggestions_router)

    # Mount clinician feedback router
    from aortica.api.feedback import FeedbackStore, create_feedback_router

    feedback_store = FeedbackStore()
    app.state.feedback_store = feedback_store  # type: ignore[attr-defined]
    feedback_router = create_feedback_router(feedback_store)
    app.include_router(feedback_router)

    # Mount SMART on FHIR router
    from aortica.api.smart_on_fhir import create_smart_router

    smart_router = create_smart_router()
    app.include_router(smart_router)

    # Mount report generation router
    from aortica.api.report_endpoints import create_report_router

    report_router = create_report_router()
    app.include_router(report_router)

    # Mount validation endpoints router
    from aortica.api.validation_endpoints import create_validation_router

    validation_router = create_validation_router()
    app.include_router(validation_router)

    # Optional OAuth providers (best-effort — only if authlib installed
    # and env vars are set)
    try:
        from aortica.api.auth import create_oauth

        oauth = create_oauth()
        app.state.oauth = oauth  # type: ignore[attr-defined]
    except ImportError:
        app.state.oauth = None  # type: ignore[attr-defined]

    # Auth dependency (no-op when disabled)
    async def _auth_dependency(
        request: Request,  # type: ignore[arg-type]
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Optional[Any]:
        if not app.state.enable_auth:  # type: ignore[attr-defined]
            return None
        return await require_auth(request, x_api_key=x_api_key)

    app.state.auth_dependency = _auth_dependency  # type: ignore[attr-defined]

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
        include_suggestions: bool = Query(
            default=False,
            description="Include clinical suggestions for active findings",
        ),
        _user: Any = Depends(_auth_dependency),
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
                include_suggestions=include_suggestions,
                retrieval_index_path=getattr(app.state, "retrieval_index_path", None),
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
        _user: Any = Depends(_auth_dependency),
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

    # ---- POST /api/v1/compare -------------------------------------------

    @app.post(
        "/api/v1/compare",
        tags=["comparison"],
        summary="Second Reader comparison",
    )
    async def compare(
        request: Request,  # type: ignore[arg-type]
        _user: Any = Depends(_auth_dependency),
    ) -> Any:
        """Compare a cardiologist's interpretation against AI predictions.

        Accepts a JSON body with the clinician's selected findings and
        AI prediction probabilities.  Returns a categorised diff:
        green (agreement), red (AI found but clinician missed), and
        yellow (clinician found but AI didn't), ranked by clinical
        importance.
        """
        from aortica.api.compare import (
            CompareRequest,
            CompareResponse,
            compare_interpretations,
        )

        body = await request.json()
        req = CompareRequest(**body)

        result: CompareResponse = compare_interpretations(
            interpretation=req.interpretation,
            ai_predictions=req.ai_predictions,
            threshold=req.threshold,
        )

        return result

    # ---- POST /api/v1/worklist/prioritize --------------------------------

    @app.post(
        "/api/v1/worklist/prioritize",
        tags=["worklist"],
        summary="Prioritize ECGs by urgency",
    )
    async def worklist_prioritize(
        request: Request,  # type: ignore[arg-type]
        _user: Any = Depends(_auth_dependency),
    ) -> Any:
        """Prioritize a batch of multi-task ECG results by clinical urgency.

        Accepts a JSON body with ``results`` (list of per-ECG prediction
        dicts) and optional ``ecg_ids`` (list of string identifiers).
        Returns a prioritized worklist sorted by urgency score (descending).

        Each result dict should contain task-level prediction dicts:
        ``rhythm``, ``structural``, ``ischaemia``, ``risk``.

        Optional ``rules_yaml`` field specifies a custom urgency rules file.
        """
        from aortica.integration.worklist import (
            PrioritizedWorklist,
            WorklistPrioritizer,
        )

        body = await request.json()

        results = body.get("results", [])
        ecg_ids = body.get("ecg_ids", None)
        rules_yaml = body.get("rules_yaml", None)

        if not results:
            return JSONResponse(
                status_code=422,
                content={"detail": "No results provided for prioritization"},
            )

        try:
            prioritizer = WorklistPrioritizer(rules_yaml=rules_yaml)
            worklist: PrioritizedWorklist = prioritizer.prioritize(
                results, ecg_ids=ecg_ids,
            )
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        return worklist.to_dict()

    # ---- POST /api/v1/export/csv -----------------------------------------

    @app.post(
        "/api/v1/export/csv",
        tags=["export"],
        summary="Batch CSV analytics export",
    )
    async def export_csv_endpoint(
        request: Request,  # type: ignore[arg-type]
        _user: Any = Depends(_auth_dependency),
    ) -> Any:
        """Export batch multi-task ECG results as a CSV download.

        Accepts a JSON body with ``results`` (list of per-ECG prediction
        dicts), optional ``filenames``, ``quality_scores``, ``urgency_scores``,
        and ``ood_flags`` lists.

        Returns the CSV content as a downloadable file response.
        """
        from fastapi.responses import Response

        from aortica.reports.csv_export import export_csv_string

        body = await request.json()

        results = body.get("results", [])
        if not results:
            return JSONResponse(
                status_code=422,
                content={"detail": "No results provided for CSV export"},
            )

        filenames_list = body.get("filenames", None)
        quality_scores_list = body.get("quality_scores", None)
        urgency_scores_list = body.get("urgency_scores", None)
        ood_flags_list = body.get("ood_flags", None)

        try:
            csv_content = export_csv_string(
                results,
                filenames=filenames_list,
                quality_scores=quality_scores_list,
                urgency_scores=urgency_scores_list,
                ood_flags=ood_flags_list,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=batch_results.csv",
            },
        )

    # ---- POST /api/v1/retrieve/similar ----------------------------------

    @app.post(
        "/api/v1/retrieve/similar",
        tags=["retrieval"],
        summary="Retrieve similar historical ECGs",
    )
    async def retrieve_similar_endpoint(
        file: UploadFile = File(..., description="ECG file to find similar cases for"),
        k: int = Query(
            default=3,
            description="Number of similar cases to retrieve",
            ge=1,
            le=20,
        ),
        format: Optional[str] = Query(  # noqa: A002
            default=None,
            description="Explicit format override (e.g. wfdb, dicom, csv)",
        ),
        _user: Any = Depends(_auth_dependency),
    ) -> Any:
        """Retrieve the top-K most similar historical ECGs for a query ECG.

        Encodes the uploaded ECG through the model backbone and queries
        the pre-built latent-space index for phenotypically similar cases
        with verified diagnoses.

        Returns similar cases with similarity scores, diagnoses, and
        demographic information.  Returns ``422`` if no index is loaded
        or the ECG file cannot be parsed.
        """
        from dataclasses import asdict

        from aortica.io.dispatcher import UnsupportedFormatError

        # Check index is loaded
        index_path = getattr(app.state, "retrieval_index_path", None)
        if index_path is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "No retrieval index loaded. Build one with aortica build-index."},
            )

        # Check model is loaded
        model = app.state.model  # type: ignore[attr-defined]
        if model is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "No model loaded for feature extraction."},
            )

        # Read and parse ECG
        file_bytes = await file.read()
        filename = file.filename or "upload.dat"

        try:
            from aortica.io.dispatcher import read_ecg
            import tempfile
            import os

            # Write to temp file for read_ecg
            suffix = os.path.splitext(filename)[1] or ".dat"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                ecg_record = read_ecg(tmp_path, format=format)
            finally:
                os.unlink(tmp_path)

        except (UnsupportedFormatError, ValueError, OSError) as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        # Retrieve similar cases
        try:
            from aortica.retrieval import retrieve_similar

            results = retrieve_similar(
                model=model,
                ecg_record=ecg_record,
                index_path=index_path,
                k=k,
            )
        except (FileNotFoundError, ImportError) as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        return {
            "similar_cases": [asdict(r) for r in results],
            "k": k,
            "query_file": filename,
        }

    return app
