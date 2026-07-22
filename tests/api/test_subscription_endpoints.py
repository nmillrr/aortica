"""Tests for aortica.api.subscription_endpoints (US-117)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.subscription_endpoints import create_subscription_router  # noqa: E402
from aortica.integration.fhir_subscription import SubscriptionManager  # noqa: E402


def _client() -> tuple[TestClient, SubscriptionManager]:
    manager = SubscriptionManager(
        http_post=lambda url, payload, timeout: 200, sleep=lambda _s: None
    )
    app = FastAPI()
    app.include_router(create_subscription_router(manager))
    return TestClient(app), manager


def test_routes_registered() -> None:
    app = FastAPI()
    app.include_router(create_subscription_router())
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/subscriptions" in paths
    assert "/api/v1/subscriptions/{sub_id}" in paths
    assert "/api/v1/subscriptions/{sub_id}/notifications" in paths


def test_create_and_list() -> None:
    client, _ = _client()
    resp = client.post(
        "/api/v1/subscriptions",
        json={
            "webhook_url": "https://ehr/hook",
            "criteria": {"min_severity": "critical", "conditions": ["STEMI"]},
        },
    )
    assert resp.status_code == 201, resp.text
    sub_id = resp.json()["id"]

    listed = client.get("/api/v1/subscriptions")
    assert listed.status_code == 200
    assert any(s["id"] == sub_id for s in listed.json())


def test_get_single() -> None:
    client, _ = _client()
    sub_id = client.post(
        "/api/v1/subscriptions", json={"webhook_url": "https://a"}
    ).json()["id"]
    resp = client.get(f"/api/v1/subscriptions/{sub_id}")
    assert resp.status_code == 200
    assert resp.json()["webhook_url"] == "https://a"


def test_get_missing_404() -> None:
    client, _ = _client()
    assert client.get("/api/v1/subscriptions/does-not-exist").status_code == 404


def test_delete() -> None:
    client, _ = _client()
    sub_id = client.post(
        "/api/v1/subscriptions", json={"webhook_url": "https://a"}
    ).json()["id"]
    assert client.delete(f"/api/v1/subscriptions/{sub_id}").status_code == 204
    assert client.get(f"/api/v1/subscriptions/{sub_id}").status_code == 404


def test_delete_missing_404() -> None:
    client, _ = _client()
    assert client.delete("/api/v1/subscriptions/nope").status_code == 404


def test_invalid_criteria_400() -> None:
    client, _ = _client()
    resp = client.post(
        "/api/v1/subscriptions",
        json={"webhook_url": "https://a", "criteria": {"min_severity": "bogus"}},
    )
    assert resp.status_code == 400


def test_notifications_history() -> None:
    client, manager = _client()
    sub_id = client.post(
        "/api/v1/subscriptions",
        json={"webhook_url": "https://ehr/hook", "criteria": {"min_severity": "critical"}},
    ).json()["id"]

    # Trigger a matching result → a delivered notification.
    manager.process_result(
        {"ischaemia": {"STEMI": 0.95}}, "ecg_1", sync=True
    )
    resp = client.get(f"/api/v1/subscriptions/{sub_id}/notifications")
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) == 1
    assert history[0]["ecg_id"] == "ecg_1"
    assert history[0]["status"] == "sent"


def test_notifications_missing_subscription_404() -> None:
    client, _ = _client()
    assert client.get("/api/v1/subscriptions/nope/notifications").status_code == 404
