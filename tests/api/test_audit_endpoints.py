"""Tests for aortica.api.audit_endpoints (US-121)."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.audit_endpoints import (  # noqa: E402
    AuditMiddleware,
    _match_auto_event,
    create_audit_router,
)
from aortica.audit import AuditLogger  # noqa: E402


def _client(tmp_path: Path) -> tuple[TestClient, AuditLogger]:
    logger = AuditLogger(str(tmp_path / "audit.db"), hmac_key="k")
    app = FastAPI()
    app.include_router(create_audit_router(logger))
    return TestClient(app), logger


def test_routes_registered(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(create_audit_router(AuditLogger(str(tmp_path / "a.db"))))
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/audit/events" in paths
    assert "/api/v1/audit/verify" in paths


def test_get_events(tmp_path: Path) -> None:
    client, logger = _client(tmp_path)
    logger.log("ecg_ingested", ecg_reference_id="e1", user_id="u1")
    logger.log("prediction_generated", ecg_reference_id="e1")
    resp = client.get("/api/v1/audit/events")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_events_filtered(tmp_path: Path) -> None:
    client, logger = _client(tmp_path)
    logger.log("ecg_ingested", ecg_reference_id="e1")
    logger.log("prediction_generated", ecg_reference_id="e1")
    resp = client.get("/api/v1/audit/events", params={"event_type": "ecg_ingested"})
    assert len(resp.json()) == 1
    assert resp.json()[0]["event_type"] == "ecg_ingested"


def test_verify_endpoint_valid(tmp_path: Path) -> None:
    client, logger = _client(tmp_path)
    logger.log("ecg_ingested", ecg_reference_id="e1")
    resp = client.get("/api/v1/audit/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["total_rows"] == 1


def test_match_auto_event() -> None:
    assert _match_auto_event("POST", "/api/v1/predict") == "prediction_generated"
    assert _match_auto_event("POST", "/api/v1/report/pdf/1") == "report_generated"
    assert _match_auto_event("POST", "/api/v1/ehr/submit") == "ehr_submitted"
    assert _match_auto_event("GET", "/api/v1/predict") is None
    assert _match_auto_event("POST", "/api/v1/other") is None


def test_middleware_auto_logs(tmp_path: Path) -> None:
    logger = AuditLogger(str(tmp_path / "audit.db"), hmac_key="k")
    app = FastAPI()
    app.state.audit_logger = logger
    app.add_middleware(AuditMiddleware)

    @app.post("/api/v1/predict")
    async def predict() -> dict:
        return {"ok": True}

    client = TestClient(app)
    client.post("/api/v1/predict")
    events = logger.query(event_type="prediction_generated")
    assert len(events) == 1
    assert events[0].event_details["path"] == "/api/v1/predict"


def test_middleware_ignores_non_matching(tmp_path: Path) -> None:
    logger = AuditLogger(str(tmp_path / "audit.db"), hmac_key="k")
    app = FastAPI()
    app.state.audit_logger = logger
    app.add_middleware(AuditMiddleware)

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    TestClient(app).get("/health")
    assert logger.count() == 0
