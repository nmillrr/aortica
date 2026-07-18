"""Tests for aortica.api.federated_endpoints — FL monitoring dashboard API (US-113)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.federated_endpoints import (  # noqa: E402
    CampaignStatusResponse,
    RoundsListResponse,
    SitesListResponse,
    create_federated_router,
)
from aortica.federated.fl_metrics_store import FLMetricsStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(store: FLMetricsStore | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(create_federated_router(store))
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/v1/federated/status
# ---------------------------------------------------------------------------


class TestFederatedStatus:
    """Tests for the campaign status endpoint."""

    def test_idle_status(self) -> None:
        client = _client()
        resp = client.get("/api/v1/federated/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"
        assert body["current_round"] == 0
        CampaignStatusResponse(**body)

    def test_running_campaign(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(name="test", total_rounds=10, strategy="fedprox")
        store.record_round(round_number=1, loss=0.5, num_clients=3)

        client = _client(store)
        resp = client.get("/api/v1/federated/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["campaign_name"] == "test"
        assert body["current_round"] == 1
        assert body["total_rounds"] == 10
        assert body["strategy"] == "fedprox"
        assert body["status"] == "running"
        assert body["start_timestamp"] > 0
        assert body["elapsed_seconds"] >= 0
        assert body["convergence"] is not None

    def test_completed_campaign(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(name="done", total_rounds=5)
        for i in range(1, 6):
            store.record_round(round_number=i, loss=0.5 - i * 0.05)
        store.complete_campaign()

        client = _client(store)
        body = client.get("/api/v1/federated/status").json()
        assert body["status"] == "completed"
        assert body["current_round"] == 5

    def test_convergence_in_status(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(total_rounds=10)
        # 7 rounds with flat loss => plateau
        for i in range(1, 8):
            store.record_round(round_number=i, loss=0.5, gradient_norm=0.1)

        client = _client(store)
        body = client.get("/api/v1/federated/status").json()
        assert body["convergence"]["plateau_detected"] is True


# ---------------------------------------------------------------------------
# GET /api/v1/federated/rounds
# ---------------------------------------------------------------------------


class TestFederatedRounds:
    """Tests for the per-round metrics endpoint."""

    def test_empty_rounds(self) -> None:
        client = _client()
        resp = client.get("/api/v1/federated/rounds")
        assert resp.status_code == 200
        body = resp.json()
        assert body["rounds"] == []
        assert body["total"] == 0
        RoundsListResponse(**body)

    def test_populated_rounds(self) -> None:
        store = FLMetricsStore()
        store.record_round(
            round_number=1,
            loss=0.5,
            metrics={"rhythm_f1": 0.87},
            num_clients=3,
            gradient_norm=1.2,
        )
        store.record_round(
            round_number=2,
            loss=0.4,
            metrics={"rhythm_f1": 0.89},
            num_clients=4,
        )

        client = _client(store)
        body = client.get("/api/v1/federated/rounds").json()
        assert body["total"] == 2
        assert len(body["rounds"]) == 2
        assert body["rounds"][0]["round_number"] == 1
        assert body["rounds"][0]["loss"] == 0.5
        assert body["rounds"][0]["metrics"]["rhythm_f1"] == 0.87
        assert body["rounds"][0]["gradient_norm"] == 1.2
        assert body["rounds"][1]["round_number"] == 2
        assert body["rounds"][1]["loss"] == 0.4


# ---------------------------------------------------------------------------
# GET /api/v1/federated/sites
# ---------------------------------------------------------------------------


class TestFederatedSites:
    """Tests for the per-site participation endpoint."""

    def test_empty_sites(self) -> None:
        client = _client()
        resp = client.get("/api/v1/federated/sites")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sites"] == []
        assert body["total"] == 0
        SitesListResponse(**body)

    def test_populated_sites(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(epsilon_budget=2.0)
        store.update_site("site-1", status="online", samples_contributed=500, epsilon_spent=0.5)
        store.update_site("site-2", status="offline", samples_contributed=300, epsilon_spent=1.0)

        client = _client(store)
        body = client.get("/api/v1/federated/sites").json()
        assert body["total"] == 2
        assert body["epsilon_budget"] == 2.0

        sites_by_id = {s["site_id"]: s for s in body["sites"]}
        assert sites_by_id["site-1"]["status"] == "online"
        assert sites_by_id["site-1"]["samples_contributed"] == 500
        assert sites_by_id["site-1"]["epsilon_budget_pct"] == 25.0  # 0.5/2.0*100

        assert sites_by_id["site-2"]["status"] == "offline"
        assert sites_by_id["site-2"]["epsilon_budget_pct"] == 50.0  # 1.0/2.0*100


# ---------------------------------------------------------------------------
# Full app integration
# ---------------------------------------------------------------------------


class TestFederatedInFullApp:
    """Verify endpoints are wired into the main app factory."""

    def test_routes_registered(self) -> None:
        from aortica.api.app import create_app

        client = TestClient(create_app(model_loaded=False, enable_auth=False))

        resp_status = client.get("/api/v1/federated/status")
        assert resp_status.status_code == 200

        resp_rounds = client.get("/api/v1/federated/rounds")
        assert resp_rounds.status_code == 200

        resp_sites = client.get("/api/v1/federated/sites")
        assert resp_sites.status_code == 200
