"""Tests for aortica.api.notification_endpoints (US-126)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.notification_endpoints import create_notification_router  # noqa: E402
from aortica.integration.notifications import (  # noqa: E402
    NotificationRules,
    UrgentFindingNotifier,
)


def test_route_registered() -> None:
    app = FastAPI()
    app.include_router(create_notification_router())
    assert "/api/v1/notifications" in set(app.openapi()["paths"].keys())


def test_empty_without_notifier() -> None:
    app = FastAPI()
    app.include_router(create_notification_router(None))
    resp = TestClient(app).get("/api/v1/notifications")
    assert resp.status_code == 200
    assert resp.json() == []


def test_history_returned() -> None:
    notifier = UrgentFindingNotifier(
        NotificationRules(channels=["webhook"]),
        db_path=":memory:",
        senders={"webhook": lambda p: True},
    )
    notifier.notify({"ischaemia": {"STEMI": 0.95}}, "ecg_1", patient_id="P1")

    app = FastAPI()
    app.include_router(create_notification_router(notifier))
    resp = TestClient(app).get("/api/v1/notifications")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["finding"] == "STEMI"
    assert body[0]["status"] == "sent"
