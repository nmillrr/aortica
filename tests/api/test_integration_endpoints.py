"""Tests for aortica.api.integration_endpoints (US-125)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.integration_endpoints import create_integration_router  # noqa: E402
from aortica.integration.orchestrator import (  # noqa: E402
    IntegrationConfig,
    IntegrationOrchestrator,
)


def test_status_route_registered() -> None:
    app = FastAPI()
    app.include_router(create_integration_router())
    assert "/api/v1/integration/status" in set(app.openapi()["paths"].keys())


def test_status_not_configured() -> None:
    app = FastAPI()
    app.include_router(create_integration_router(None))
    resp = TestClient(app).get("/api/v1/integration/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_configured"


def test_status_with_orchestrator() -> None:
    orch = IntegrationOrchestrator(
        IntegrationConfig(enabled_channels=["storage"]),
        processor=lambda p: {"rhythm": {"AF": 0.9}},
    )
    app = FastAPI()
    app.include_router(create_integration_router(orch))
    resp = TestClient(app).get("/api/v1/integration/status")
    body = resp.json()
    assert body["status"] == "running"
    assert body["enabled_channels"] == ["storage"]
    assert "channels" in body
