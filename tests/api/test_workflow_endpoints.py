"""Tests for the finalize workflow endpoint (US-127)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.workflow_endpoints import create_workflow_router  # noqa: E402
from aortica.audit import AuditLogger  # noqa: E402
from aortica.integration.worklist import WorklistPrioritizer  # noqa: E402
from aortica.integration.worklist_store import WorklistStore  # noqa: E402


def _app(worklist_store=None, audit_logger=None) -> FastAPI:
    app = FastAPI()
    app.state.worklist_store = worklist_store
    app.state.audit_logger = audit_logger
    app.include_router(create_workflow_router())
    return app


def _body(**overrides) -> dict:
    body = {
        "result_id": "42",
        "ecg_id": "ecg_1",
        "reviewed_findings": {"ischaemia": {"STEMI": 0.95}},
        "attestation": {"clinician": "dr_smith", "confirmed": True},
        "output_channels": ["fhir", "pdf"],
    }
    body.update(overrides)
    return body


def test_route_registered() -> None:
    app = _app()
    assert "/api/v1/workflow/finalize" in set(app.openapi()["paths"].keys())


def test_finalize_success() -> None:
    client = TestClient(_app())
    resp = client.post("/api/v1/workflow/finalize", json=_body())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "submitted"
    assert data["ehr_reference"].startswith("EHR-")
    assert "pdf" in data["channels_generated"]


def test_unconfirmed_attestation_422() -> None:
    client = TestClient(_app())
    body = _body(attestation={"clinician": "dr_smith", "confirmed": False})
    resp = client.post("/api/v1/workflow/finalize", json=body)
    assert resp.status_code == 422


def test_unknown_channel_422() -> None:
    client = TestClient(_app())
    resp = client.post("/api/v1/workflow/finalize", json=_body(output_channels=["bogus"]))
    assert resp.status_code == 422


def test_updates_worklist_to_completed() -> None:
    store = WorklistStore()
    wl = WorklistPrioritizer().prioritize(
        [{"ischaemia": {"STEMI": 0.95}}], ecg_ids=["ecg_1"]
    )
    store.add_from_prioritized(wl)
    client = TestClient(_app(worklist_store=store))

    resp = client.post("/api/v1/workflow/finalize", json=_body())
    assert resp.json()["worklist_updated"] is True
    assert store.get("ecg_1").review_status == "completed"
    assert store.get("ecg_1").assignee == "dr_smith"


def test_logs_audit_event(tmp_path) -> None:
    audit = AuditLogger(str(tmp_path / "audit.db"), hmac_key="k")
    client = TestClient(_app(audit_logger=audit))

    client.post("/api/v1/workflow/finalize", json=_body())
    events = audit.query(event_type="ehr_submitted")
    assert len(events) == 1
    assert events[0].event_details["event"] == "finalize_and_submit"
    assert events[0].user_id == "dr_smith"
