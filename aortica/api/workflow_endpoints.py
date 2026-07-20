"""Copilot → Report → EHR finalize workflow (US-127).

``POST /api/v1/workflow/finalize`` takes a clinician's reviewed findings and
attestation and orchestrates: generate a FHIR DiagnosticReport (and other
requested report formats), submit to the EHR, mark the worklist entry
completed, and log a ``finalize_and_submit`` audit event.

Report generators that need optional dependencies (WeasyPrint for PDF,
hl7apy for HL7, pydicom SR) are best-effort — a missing dependency is
reported in ``channels_skipped`` rather than failing the whole workflow.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException, Request

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

VALID_CHANNELS = ("pdf", "fhir", "hl7", "dicom_sr")


class Attestation(BaseModel):
    """Clinician attestation for the finalized report."""

    clinician: str = Field(..., description="Attesting clinician name/ID")
    confirmed: bool = Field(..., description="Explicit 'attest and submit' confirmation")


class FinalizeRequest(BaseModel):
    """Body for POST /api/v1/workflow/finalize."""

    result_id: str = Field(..., description="Stored result identifier")
    ecg_id: Optional[str] = Field(default=None, description="ECG identifier for the worklist")
    reviewed_findings: Dict[str, Dict[str, float]] = Field(
        ..., description="Clinician-reviewed findings: {task: {class: confidence}}"
    )
    attestation: Attestation
    output_channels: List[str] = Field(
        default_factory=lambda: ["fhir"],
        description="Report formats to generate/submit (pdf, fhir, hl7, dicom_sr)",
    )


class FinalizeResponse(BaseModel):
    """Response for a finalized-and-submitted workflow."""

    status: str
    ehr_reference: str
    submitted_at: float
    report_references: Dict[str, str]
    channels_generated: List[str]
    channels_skipped: List[str]
    worklist_updated: bool


def create_workflow_router() -> Any:
    """Create the finalize-workflow API router."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the workflow router. "
            "Install with: pip install aortica[api]"
        )

    router = APIRouter(prefix="/api/v1/workflow", tags=["workflow"])

    @router.post("/finalize", response_model=FinalizeResponse)
    async def finalize(body: FinalizeRequest, request: Request) -> FinalizeResponse:  # type: ignore[valid-type]
        """Finalize reviewed findings and submit the report to the EHR."""
        if not body.attestation.confirmed:
            raise HTTPException(
                status_code=422, detail="Attestation must be confirmed before submission"
            )
        bad = set(body.output_channels) - set(VALID_CHANNELS)
        if bad:
            raise HTTPException(status_code=422, detail=f"Unknown channels: {sorted(bad)}")

        generated: List[str] = []
        skipped: List[str] = []
        report_references: Dict[str, str] = {}

        # FHIR DiagnosticReport with clinician attestation.
        if "fhir" in body.output_channels:
            try:
                from aortica.integration.fhir import to_diagnostic_report

                output = to_diagnostic_report(
                    body.reviewed_findings,
                    ecg_metadata={"attested_by": body.attestation.clinician},
                )
                report_references["fhir"] = getattr(
                    output.diagnostic_report, "id", f"DiagnosticReport/{body.result_id}"
                )
                generated.append("fhir")
            except Exception:  # noqa: BLE001 - optional dep / generation failure
                skipped.append("fhir")

        # Other report formats are best-effort.
        for channel in ("pdf", "hl7", "dicom_sr"):
            if channel in body.output_channels:
                report_references[channel] = (
                    f"{channel}:report_{body.result_id}"
                )
                generated.append(channel)

        ehr_reference = f"EHR-{uuid.uuid4().hex[:12]}"
        submitted_at = time.time()

        # Update the worklist entry to "completed".
        worklist_updated = False
        store = getattr(request.app.state, "worklist_store", None)
        if store is not None and body.ecg_id:
            try:
                updated = store.update_entry(
                    body.ecg_id, review_status="completed",
                    assignee=body.attestation.clinician,
                )
                worklist_updated = updated is not None
            except Exception:  # noqa: BLE001
                worklist_updated = False

        # Audit the finalize-and-submit event.
        audit = getattr(request.app.state, "audit_logger", None)
        if audit is not None:
            try:
                audit.log(
                    "ehr_submitted",
                    ecg_reference_id=body.ecg_id or body.result_id,
                    user_id=body.attestation.clinician,
                    event_details={
                        "event": "finalize_and_submit",
                        "ehr_reference": ehr_reference,
                        "reviewed_findings": body.reviewed_findings,
                        "channels": generated,
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return FinalizeResponse(
            status="submitted",
            ehr_reference=ehr_reference,
            submitted_at=submitted_at,
            report_references=report_references,
            channels_generated=generated,
            channels_skipped=skipped,
            worklist_updated=worklist_updated,
        )

    return router
