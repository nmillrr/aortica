"""Tests for the edge-sync + analytics endpoints (US-128)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.analytics_endpoints import create_analytics_router  # noqa: E402
from aortica.sync.central_aggregator import CentralAggregator  # noqa: E402


def _client(agg=None, **kw) -> TestClient:
    app = FastAPI()
    app.include_router(create_analytics_router(agg, **kw))
    return TestClient(app)


def _result(ecg_id: str, stemi: float = 0.1, quality: float = 80.0) -> dict:
    return {
        "ecg_id": ecg_id,
        "predictions": {"ischaemia": {"STEMI": stemi}},
        "quality_score": quality,
    }


def test_routes_registered() -> None:
    app = FastAPI()
    app.include_router(create_analytics_router(CentralAggregator()))
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/sync/receive" in paths
    assert "/api/v1/analytics/sites" in paths


def test_sync_receive_and_analytics() -> None:
    agg = CentralAggregator()
    client = _client(agg)
    resp = client.post("/api/v1/sync/receive", json={
        "device_id": "dev1", "site_id": "site_a",
        "results": [_result("e1", stemi=0.9), _result("e2")],
    })
    assert resp.status_code == 200
    assert resp.json()["received"] == 2

    sites = client.get("/api/v1/analytics/sites").json()
    assert sites["fleet"]["total_ecgs"] == 2
    assert sites["sites"][0]["site_id"] == "site_a"


def test_reconciliation_in_response() -> None:
    client = _client(CentralAggregator())
    resp = client.post("/api/v1/sync/receive", json={
        "device_id": "dev1", "site_id": "site_a",
        "results": [_result("e1")], "expected_count": 3,
    })
    body = resp.json()
    assert body["reconciliation"]["gap"] == 2


def test_device_key_auth() -> None:
    agg = CentralAggregator()
    client = _client(agg, require_device_key=True, device_keys={"dev1": "secret"})

    # Missing / wrong key → 401.
    bad = client.post("/api/v1/sync/receive", json={
        "device_id": "dev1", "site_id": "site_a", "results": [],
    })
    assert bad.status_code == 401

    # Correct key → 200.
    ok = client.post(
        "/api/v1/sync/receive",
        json={"device_id": "dev1", "site_id": "site_a", "results": [_result("e1")]},
        headers={"X-Device-Key": "secret"},
    )
    assert ok.status_code == 200


def test_anomalies_in_analytics() -> None:
    agg = CentralAggregator(anomaly_z_threshold=1.5)
    client = _client(agg)
    for site in ("a", "b", "c", "d"):
        client.post("/api/v1/sync/receive", json={
            "device_id": f"dev_{site}", "site_id": f"site_{site}",
            "results": [_result(f"{site}1", quality=85)],
        })
    client.post("/api/v1/sync/receive", json={
        "device_id": "dev_x", "site_id": "site_x",
        "results": [_result("x1", quality=5)],
    })
    anomalies = client.get("/api/v1/analytics/sites").json()["anomalies"]
    assert any(a["site_id"] == "site_x" for a in anomalies)
