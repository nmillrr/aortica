"""Tests for aortica.api.edge_endpoints — GET /edge/status (US-061b)."""

from __future__ import annotations

import tempfile

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.edge_endpoints import EdgeStatusResponse, create_edge_router  # noqa: E402
from aortica.edge.site_monitor import SiteMonitor  # noqa: E402


def _client_with_monitor(monitor: SiteMonitor) -> TestClient:
    app = FastAPI()
    app.include_router(create_edge_router(monitor))
    return TestClient(app)


def test_edge_status_route_registered() -> None:
    app = FastAPI()
    app.include_router(create_edge_router())
    paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
    assert "/edge/status" in paths


def test_edge_status_returns_monitor_data() -> None:
    monitor = SiteMonitor(tempfile.mkdtemp(), site_id="unit-site")
    monitor.record_inference(success=True)
    monitor.record_inference(success=False, error_type="x")
    client = _client_with_monitor(monitor)

    resp = client.get("/edge/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["site_id"] == "unit-site"
    assert body["daily_inference_count"] == 2
    assert body["daily_error_count"] == 1
    assert body["sync_status"] == "unknown"
    assert "storage_utilization_pct" in body
    # Response validates against the declared schema.
    EdgeStatusResponse(**body)
    monitor.close()


def test_edge_status_default_monitor() -> None:
    # No monitor passed → router creates a temporary one and still responds.
    app = FastAPI()
    app.include_router(create_edge_router())
    client = TestClient(app)
    resp = client.get("/edge/status")
    assert resp.status_code == 200
    assert resp.json()["daily_inference_count"] == 0


def test_edge_status_in_full_app() -> None:
    # The endpoint is wired into the main application factory.
    from aortica.api.app import create_app

    client = TestClient(create_app(model_loaded=False))
    resp = client.get("/edge/status")
    assert resp.status_code == 200
    assert "site_id" in resp.json()
