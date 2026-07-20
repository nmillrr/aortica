"""Tests for the report-listing endpoint GET /api/v1/reports/{id} (US-120)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402


class _StoredResult:
    def __init__(self, rid: int) -> None:
        self.id = rid
        self.predictions = {
            "rhythm": {"AF": 0.9, "normal_sinus_rhythm": 0.05},
            "structural": {"LVH": 0.7},
            "ischaemia": {"STEMI": 0.03},
            "risk": {"mortality_1y": 0.12},
        }
        self.quality = {"overall": 85}
        self.metadata: Dict[str, Any] = {
            "sample_rate": 500,
            "duration_seconds": 10.0,
            "num_leads": 12,
            "source_format": "stored",
        }
        self.timestamp = 1700000000.0
        self.synced = False


class _Store:
    def __init__(self, ids: list[int]) -> None:
        self._r = {i: _StoredResult(i) for i in ids}

    def get_result_by_id(self, rid: int) -> Optional[_StoredResult]:
        return self._r.get(rid)


@pytest.fixture()
def app_and_client() -> tuple[Any, TestClient]:
    app = create_app(enable_auth=False)
    app.state.result_store = _Store([1])  # type: ignore[attr-defined]
    return app, TestClient(app)


@pytest.fixture()
def client(app_and_client: tuple[Any, TestClient]) -> TestClient:
    return app_and_client[1]


def test_listing_route_registered() -> None:
    app = create_app(enable_auth=False)
    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/reports/{result_id}" in paths


def test_list_reports_returns_formats(client: TestClient) -> None:
    resp = client.get("/api/v1/reports/1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result_id"] == 1
    formats = {f["format"] for f in body["available_formats"]}
    assert formats == {"pdf", "fhir", "hl7", "jsonld"}
    assert body["history"] == []


def test_list_reports_unknown_id_404(client: TestClient) -> None:
    assert client.get("/api/v1/reports/999").status_code == 404


def test_history_reflected_in_listing(
    app_and_client: tuple[Any, TestClient],
) -> None:
    # A generated report is recorded via the app's report_history store and
    # surfaced by the listing endpoint. (Report generators themselves need
    # optional deps, so record directly to test the endpoint↔history wiring.)
    app, client = app_and_client
    app.state.report_history.record(1, "pdf", "report_1.pdf")
    app.state.report_history.record(1, "fhir", "report_1.json")

    listing = client.get("/api/v1/reports/1").json()
    assert len(listing["history"]) == 2
    assert {h["format"] for h in listing["history"]} == {"pdf", "fhir"}
    assert all("generated_at" in h for h in listing["history"])


def test_record_report_helper_records() -> None:
    from aortica.api.report_endpoints import ReportHistoryStore

    store = ReportHistoryStore()
    store.record(5, "hl7", "report_5.hl7")
    entries = store.list(5)
    assert len(entries) == 1
    assert entries[0]["format"] == "hl7"
    assert store.list(999) == []


def test_list_reports_no_store_422() -> None:
    app = create_app(enable_auth=False)
    resp = TestClient(app).get("/api/v1/reports/1")
    assert resp.status_code == 422
