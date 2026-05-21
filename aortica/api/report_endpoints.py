"""Report generation API endpoints.

Provides endpoints for generating clinical reports from stored ECG
analysis results:

- ``POST /api/v1/report/pdf/{result_id}`` — PDF clinical report
- ``POST /api/v1/report/jsonld/{result_id}`` — JSON-LD machine-readable report
- ``POST /api/v1/report/fhir/{result_id}`` — FHIR R4 DiagnosticReport bundle
- ``POST /api/v1/report/hl7/{result_id}`` — HL7 v2.x ORU^R01 message

All endpoints require authentication via the existing auth system
(JWT or API key).

Requires a :class:`~aortica.sync.result_store.ResultStore` to be
attached to ``app.state.result_store``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse, Response

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ReportErrorResponse(BaseModel):
    """Error response for report endpoints."""

    detail: str = Field(..., description="Error message")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_result_store(request: Any) -> Any:
    """Retrieve the ResultStore from app state, or raise 422."""
    store = getattr(request.app.state, "result_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Result store not configured on this server",
        )
    return store


def _get_stored_result(store: Any, result_id: int) -> Any:
    """Retrieve a stored result by ID, or raise 404."""
    result = store.get_result_by_id(result_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result with id {result_id} not found",
        )
    return result


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_report_router() -> Any:
    """Build a FastAPI APIRouter with report generation endpoints.

    Returns an ``APIRouter`` with prefix ``/api/v1/report``.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for report endpoints. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/report", tags=["reports"])

    # ------------------------------------------------------------------
    # POST /api/v1/report/pdf/{result_id}
    # ------------------------------------------------------------------

    @router.post(
        "/pdf/{result_id}",
        summary="Generate PDF clinical report",
        responses={
            404: {"model": ReportErrorResponse, "description": "Result not found"},
            422: {"model": ReportErrorResponse, "description": "Invalid parameters"},
        },
    )
    async def generate_pdf_report(
        result_id: int,
        request: Request,  # type: ignore[arg-type]
        model_version: str = Query(
            default="unknown",
            description="Model version string for report header",
        ),
        finding_threshold: float = Query(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Confidence threshold for including findings (0–1)",
        ),
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Any:
        """Generate a PDF clinical report for a stored result.

        Returns the PDF as a downloadable binary response with
        ``Content-Type: application/pdf``.
        """
        # Auth check via app-level dependency
        auth_dep = getattr(request.app.state, "auth_dependency", None)
        if auth_dep is not None:
            await auth_dep(request, x_api_key=x_api_key)

        store = _get_result_store(request)
        result = _get_stored_result(store, result_id)

        try:
            from aortica.reports.pdf_report import generate_pdf as _gen_pdf

            import os
            import tempfile

            # Create a minimal ECGRecord for the report
            import numpy as np

            from aortica.io.ecg_record import ECGRecord

            # Build a stub ECG record from stored metadata
            metadata = result.metadata or {}
            num_leads = metadata.get("num_leads", 12)
            sample_rate = metadata.get("sample_rate", 500)
            duration = metadata.get("duration_seconds", 10.0)
            num_samples = int(sample_rate * duration)
            lead_names = metadata.get(
                "lead_names",
                ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"][:num_leads],
            )
            source_format = metadata.get("source_format", "stored")

            ecg_record = ECGRecord(
                signals=np.zeros((num_leads, num_samples), dtype=np.float64),
                sample_rate=sample_rate,
                lead_names=lead_names,
                duration_seconds=duration,
                source_format=source_format,
                patient_metadata=metadata.get("patient_metadata"),
            )

            # Generate PDF to a temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp_path = tmp.name

            try:
                _gen_pdf(
                    result.predictions,
                    ecg_record,
                    output_path=tmp_path,
                    model_version=model_version,
                    finding_threshold=finding_threshold,
                )
                with open(tmp_path, "rb") as f:
                    pdf_bytes = f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=report_{result_id}.pdf"
                    ),
                },
            )

        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF generation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # POST /api/v1/report/jsonld/{result_id}
    # ------------------------------------------------------------------

    @router.post(
        "/jsonld/{result_id}",
        summary="Generate JSON-LD machine-readable report",
        responses={
            404: {"model": ReportErrorResponse, "description": "Result not found"},
            422: {"model": ReportErrorResponse, "description": "Invalid parameters"},
        },
    )
    async def generate_jsonld_report(
        result_id: int,
        request: Request,  # type: ignore[arg-type]
        model_version: str = Query(
            default="unknown",
            description="Model version for provenance metadata",
        ),
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Any:
        """Generate a JSON-LD machine-readable report for a stored result.

        Returns a JSON-LD document linked to SNOMED CT and LOINC
        ontologies with provenance metadata.
        """
        auth_dep = getattr(request.app.state, "auth_dependency", None)
        if auth_dep is not None:
            await auth_dep(request, x_api_key=x_api_key)

        store = _get_result_store(request)
        result = _get_stored_result(store, result_id)

        try:
            from aortica.reports.jsonld_report import generate_jsonld

            metadata = result.metadata or {}
            ecg_metadata = {
                "sample_rate": metadata.get("sample_rate", 500),
                "duration_seconds": metadata.get("duration_seconds", 10.0),
                "num_leads": metadata.get("num_leads", 12),
                "source_format": metadata.get("source_format", "stored"),
            }

            jsonld_doc = generate_jsonld(
                result.predictions,
                ecg_metadata=ecg_metadata,
                model_version=model_version,
            )

            return JSONResponse(content=jsonld_doc)

        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"JSON-LD generation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # POST /api/v1/report/fhir/{result_id}
    # ------------------------------------------------------------------

    @router.post(
        "/fhir/{result_id}",
        summary="Generate FHIR R4 DiagnosticReport bundle",
        responses={
            404: {"model": ReportErrorResponse, "description": "Result not found"},
            422: {"model": ReportErrorResponse, "description": "Invalid parameters"},
        },
    )
    async def generate_fhir_report(
        result_id: int,
        request: Request,  # type: ignore[arg-type]
        patient_ref: Optional[str] = Query(
            default=None,
            description="FHIR Patient reference (e.g. Patient/123)",
        ),
        confidence_threshold: float = Query(
            default=0.30,
            ge=0.0,
            le=1.0,
            description="Minimum confidence for including findings",
        ),
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Any:
        """Generate a FHIR R4 DiagnosticReport bundle for a stored result.

        Returns a FHIR Bundle (type=collection) containing a
        DiagnosticReport, child Observations for positive findings,
        and RiskAssessment resources.
        """
        auth_dep = getattr(request.app.state, "auth_dependency", None)
        if auth_dep is not None:
            await auth_dep(request, x_api_key=x_api_key)

        store = _get_result_store(request)
        result = _get_stored_result(store, result_id)

        try:
            from aortica.integration.fhir import to_diagnostic_report

            metadata = result.metadata or {}
            ecg_metadata = {
                "acquisition_datetime": metadata.get("acquisition_datetime"),
            }

            fhir_output = to_diagnostic_report(
                result.predictions,
                patient_ref=patient_ref,
                ecg_metadata=ecg_metadata,
                confidence_threshold=confidence_threshold,
            )

            import json

            return JSONResponse(content=json.loads(fhir_output.bundle_json))

        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"FHIR report generation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # POST /api/v1/report/hl7/{result_id}
    # ------------------------------------------------------------------

    @router.post(
        "/hl7/{result_id}",
        summary="Generate HL7 v2.x ORU^R01 message",
        responses={
            404: {"model": ReportErrorResponse, "description": "Result not found"},
            422: {"model": ReportErrorResponse, "description": "Invalid parameters"},
        },
    )
    async def generate_hl7_report(
        result_id: int,
        request: Request,  # type: ignore[arg-type]
        patient_id: Optional[str] = Query(
            default=None,
            description="Patient identifier for PID segment",
        ),
        order_id: Optional[str] = Query(
            default=None,
            description="Order/accession number for OBR segment",
        ),
        confidence_threshold: float = Query(
            default=0.30,
            ge=0.0,
            le=1.0,
            description="Minimum confidence for including findings",
        ),
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> Any:
        """Generate an HL7 v2.x ORU^R01 message for a stored result.

        Returns the ER7-encoded HL7 message as ``text/plain``.
        """
        auth_dep = getattr(request.app.state, "auth_dependency", None)
        if auth_dep is not None:
            await auth_dep(request, x_api_key=x_api_key)

        store = _get_result_store(request)
        result = _get_stored_result(store, result_id)

        try:
            from aortica.integration.hl7v2 import to_oru_r01

            hl7_message = to_oru_r01(
                result.predictions,
                patient_id=patient_id,
                order_id=order_id,
                confidence_threshold=confidence_threshold,
            )

            return Response(
                content=hl7_message,
                media_type="text/plain",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=report_{result_id}.hl7"
                    ),
                },
            )

        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"HL7 message generation failed: {exc}",
            )

    return router
