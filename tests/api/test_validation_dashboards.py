"""Tests for prospective-progress and monitor dashboard endpoints (US-122/123)."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("cryptography")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.validation_endpoints import create_validation_router  # noqa: E402
from aortica.validation.performance_monitor import PerformanceMonitor  # noqa: E402
from aortica.validation.prospective_collector import ProspectiveCollector  # noqa: E402


def _client(collector=None, monitor=None) -> TestClient:
    app = FastAPI()
    app.include_router(create_validation_router(collector=collector, monitor=monitor))
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────
# US-122: prospective/progress
# ─────────────────────────────────────────────────────────────────


def test_progress_noop_without_collector() -> None:
    resp = _client().get("/api/v1/validation/prospective/progress")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_progress_counts(tmp_path: Path) -> None:
    collector = ProspectiveCollector(db_dir=str(tmp_path))
    rid = collector.ingest_ecg(
        ecg_hash="h1", site_id="site_a",
        predictions={"rhythm": {"AF": 0.9}}, quality={"overall": 85},
    )
    collector.ingest_ecg(
        ecg_hash="h2", site_id="site_a",
        predictions={"rhythm": {"AF": 0.1}}, quality={"overall": 85},
    )
    collector.add_outcome(record_id=rid, ground_truth={"AF": 1})

    resp = _client(collector=collector).get("/api/v1/validation/prospective/progress")
    body = resp.json()
    assert body["total"] == 2
    assert body["linked"] == 1
    assert body["unlinked"] == 1
    assert abs(body["completion_rate"] - 0.5) < 1e-9


def test_progress_site_filter(tmp_path: Path) -> None:
    collector = ProspectiveCollector(db_dir=str(tmp_path))
    collector.ingest_ecg(ecg_hash="h1", site_id="site_a", predictions={"rhythm": {}}, quality={})
    collector.ingest_ecg(ecg_hash="h2", site_id="site_b", predictions={"rhythm": {}}, quality={})
    resp = _client(collector=collector).get(
        "/api/v1/validation/prospective/progress", params={"site_id": "site_a"}
    )
    assert resp.json()["total"] == 1


# ─────────────────────────────────────────────────────────────────
# US-123: monitor/metrics and monitor/alerts
# ─────────────────────────────────────────────────────────────────


def test_metrics_noop_without_monitor() -> None:
    resp = _client().get("/api/v1/validation/monitor/metrics")
    assert resp.status_code == 200
    assert resp.json()["task_metrics"] == {}


def test_metrics_with_data(tmp_path: Path) -> None:
    monitor = PerformanceMonitor(db_dir=str(tmp_path))
    for i in range(20):
        monitor.record_prediction(
            ecg_id=f"e{i}",
            task="rhythm",
            predictions={"AF": 0.9 if i % 2 else 0.1},
            ground_truth={"AF": 1 if i % 2 else 0},
        )
    monitor.set_baseline("rhythm", "f1", 0.5)

    resp = _client(monitor=monitor).get("/api/v1/validation/monitor/metrics")
    body = resp.json()
    assert "rhythm" in body["task_metrics"]
    rhythm = body["task_metrics"]["rhythm"]
    assert "baseline" in rhythm and "trend" in rhythm
    assert rhythm["baseline"]["f1"] == 0.5
    assert rhythm["trend"]["f1"] in {"up", "down", "flat"}
    assert body["volume"]["total_predictions"] > 0


def test_alerts_noop_without_monitor() -> None:
    resp = _client().get("/api/v1/validation/monitor/alerts")
    assert resp.status_code == 200
    assert resp.json()["alerts"] == []
    assert resp.json()["has_drift"] is False


def test_alerts_shape_with_monitor(tmp_path: Path) -> None:
    monitor = PerformanceMonitor(db_dir=str(tmp_path))
    monitor.record_prediction(
        ecg_id="e1", task="rhythm", predictions={"AF": 0.9}, ground_truth={"AF": 1}
    )
    resp = _client(monitor=monitor).get("/api/v1/validation/monitor/alerts")
    body = resp.json()
    assert "alerts" in body
    assert isinstance(body["alerts"], list)
