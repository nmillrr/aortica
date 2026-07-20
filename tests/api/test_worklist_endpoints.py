"""Tests for aortica.api.worklist_endpoints (US-119)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.worklist_endpoints import create_worklist_router  # noqa: E402
from aortica.integration.worklist import WorklistPrioritizer  # noqa: E402
from aortica.integration.worklist_store import WorklistStore  # noqa: E402


def _seed(store: WorklistStore) -> None:
    wl = WorklistPrioritizer().prioritize(
        [{"ischaemia": {"STEMI": 0.95}}, {"rhythm": {"normal_sinus_rhythm": 0.99}}],
        ecg_ids=["crit", "routine"],
    )
    store.add_from_prioritized(wl)


def _client() -> tuple[TestClient, WorklistStore]:
    store = WorklistStore()
    _seed(store)
    app = FastAPI()
    app.include_router(create_worklist_router(store))
    return TestClient(app), store


def test_routes_registered() -> None:
    app = FastAPI()
    app.include_router(create_worklist_router())
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/worklist" in paths
    assert "/api/v1/worklist/{ecg_id}" in paths


def test_get_worklist_sorted_with_summary() -> None:
    client, _ = _client()
    resp = client.get("/api/v1/worklist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["ecg_id"] == "crit"  # highest urgency first
    assert body["summary"]["total"] == 2
    assert body["summary"]["critical_count"] == 1


def test_filter_by_tier() -> None:
    client, _ = _client()
    resp = client.get("/api/v1/worklist", params={"tier": "critical"})
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["ecg_id"] == "crit"


def test_patch_updates_status() -> None:
    client, _ = _client()
    resp = client.patch(
        "/api/v1/worklist/crit",
        json={"review_status": "completed", "assignee": "dr_smith"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["review_status"] == "completed"
    assert body["assignee"] == "dr_smith"
    assert body["reviewed_at"] is not None


def test_patch_missing_404() -> None:
    client, _ = _client()
    resp = client.patch("/api/v1/worklist/nope", json={"review_status": "completed"})
    assert resp.status_code == 404


def test_patch_invalid_status_400() -> None:
    client, _ = _client()
    resp = client.patch("/api/v1/worklist/crit", json={"review_status": "bogus"})
    assert resp.status_code == 400


def test_filter_by_status_after_patch() -> None:
    client, _ = _client()
    client.patch("/api/v1/worklist/crit", json={"review_status": "completed"})
    pending = client.get("/api/v1/worklist", params={"status": "pending"}).json()
    assert {i["ecg_id"] for i in pending["items"]} == {"routine"}
