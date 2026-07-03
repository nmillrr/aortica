"""Tests for aortica.api.result_endpoints — Result browser API."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402
from aortica.api.result_endpoints import (  # noqa: E402
    ResultDetail,
    ResultListResponse,
    ResultSummary,
    _classify_urgency,
    _extract_top_finding,
    _stored_to_summary,
)


# ---------------------------------------------------------------------------
# Fake StoredResult for testing
# ---------------------------------------------------------------------------


class FakeStoredResult:
    """Mimics aortica.sync.result_store.StoredResult."""

    def __init__(
        self,
        id: int = 1,
        ecg_hash: str = "abc123",
        predictions: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        timestamp: float | None = None,
        synced: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.ecg_hash = ecg_hash
        self.predictions = predictions or {"rhythm": {"af": 0.85, "nsr": 0.15}}
        self.quality = quality or {"overall": 90, "classification": "good"}
        self.timestamp = timestamp or time.time()
        self.synced = synced
        self.metadata = metadata or {}


class FakeResultStore:
    """In-memory fake result store for testing."""

    def __init__(self, results: list[FakeStoredResult] | None = None) -> None:
        self._results = results or []

    def list_results(self, limit: int = 100, offset: int = 0) -> list[FakeStoredResult]:
        return self._results[offset : offset + limit]

    def get_result_by_id(self, result_id: int) -> FakeStoredResult | None:
        for r in self._results:
            if r.id == result_id:
                return r
        return None

    def count(self, synced: bool | None = None) -> int:
        if synced is not None:
            return sum(1 for r in self._results if r.synced == synced)
        return len(self._results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    FakeStoredResult(
        id=1,
        ecg_hash="hash_001",
        predictions={"rhythm": {"atrial_fibrillation": 0.92, "nsr": 0.08}},
        quality={"overall": 95, "classification": "good"},
        timestamp=1700000000.0,
        synced=True,
        metadata={"patient_id": "P-4821"},
    ),
    FakeStoredResult(
        id=2,
        ecg_hash="hash_002",
        predictions={"rhythm": {"nsr": 0.97, "af": 0.03}},
        quality={"overall": 88, "classification": "good"},
        timestamp=1700001000.0,
        synced=False,
        metadata={"patient_id": "P-4822"},
    ),
    FakeStoredResult(
        id=3,
        ecg_hash="hash_003",
        predictions={"ischaemia": {"stemi": 0.78, "nstemi": 0.12}},
        quality={"overall": 45, "classification": "poor"},
        timestamp=1700002000.0,
        synced=False,
        metadata={"patient_id": "P-4823"},
    ),
    FakeStoredResult(
        id=4,
        ecg_hash="hash_004",
        predictions={"structural": {"lvh": 0.65, "rvh": 0.10}},
        quality={"overall": 72, "classification": "marginal"},
        timestamp=1700003000.0,
        synced=True,
        metadata={},
    ),
    FakeStoredResult(
        id=5,
        ecg_hash="hash_005",
        predictions={"rhythm": {"nsr": 0.99}},
        quality={"overall": 98, "classification": "good"},
        timestamp=1700004000.0,
        synced=True,
        metadata={"patient_id": "P-4824"},
    ),
]


@pytest.fixture()
def fake_store() -> FakeResultStore:
    return FakeResultStore(list(SAMPLE_RESULTS))


@pytest.fixture()
def app(fake_store: FakeResultStore) -> fastapi.FastAPI:
    application = create_app(enable_auth=False)
    application.state.result_store = fake_store  # type: ignore[attr-defined]
    return application


@pytest.fixture()
def client(app: fastapi.FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def app_no_store() -> fastapi.FastAPI:
    return create_app(enable_auth=False)


@pytest.fixture()
def client_no_store(app_no_store: fastapi.FastAPI) -> TestClient:
    return TestClient(app_no_store)


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestClassifyUrgency:
    """Tests for _classify_urgency helper."""

    def test_critical_stemi(self) -> None:
        preds = {"ischaemia": {"stemi": 0.85}}
        assert _classify_urgency(preds) == "critical"

    def test_critical_vt(self) -> None:
        preds = {"rhythm": {"ventricular_tachycardia": 0.75}}
        assert _classify_urgency(preds) == "critical"

    def test_urgent_af(self) -> None:
        preds = {"rhythm": {"atrial_fibrillation": 0.80}}
        assert _classify_urgency(preds) == "urgent"

    def test_urgent_lvh(self) -> None:
        preds = {"structural": {"lvh": 0.60}}
        assert _classify_urgency(preds) == "urgent"

    def test_routine_moderate_finding(self) -> None:
        preds = {"rhythm": {"some_finding": 0.35}}
        assert _classify_urgency(preds) == "routine"

    def test_normal_low_probs(self) -> None:
        preds = {"rhythm": {"nsr": 0.20, "af": 0.05}}
        assert _classify_urgency(preds) == "normal"

    def test_empty_predictions(self) -> None:
        assert _classify_urgency({}) == "normal"

    def test_non_dict_values_skipped(self) -> None:
        preds = {"rhythm": "invalid", "structural": {"lvh": 0.1}}
        assert _classify_urgency(preds) == "normal"


class TestExtractTopFinding:
    """Tests for _extract_top_finding helper."""

    def test_single_task(self) -> None:
        preds = {"rhythm": {"af": 0.85, "nsr": 0.15}}
        name, prob = _extract_top_finding(preds)
        assert name == "af"
        assert prob == 0.85

    def test_multi_task(self) -> None:
        preds = {
            "rhythm": {"nsr": 0.90},
            "ischaemia": {"stemi": 0.95},
        }
        name, prob = _extract_top_finding(preds)
        assert name == "stemi"
        assert prob == 0.95

    def test_empty_predictions(self) -> None:
        name, prob = _extract_top_finding({})
        assert name is None
        assert prob is None

    def test_non_numeric_skipped(self) -> None:
        preds = {"rhythm": {"af": "high", "nsr": 0.5}}
        name, prob = _extract_top_finding(preds)
        assert name == "nsr"
        assert prob == 0.5


class TestStoredToSummary:
    """Tests for _stored_to_summary helper."""

    def test_basic_conversion(self) -> None:
        result = FakeStoredResult(
            id=42,
            ecg_hash="test_hash",
            predictions={"rhythm": {"af": 0.90}},
            quality={"overall": 85, "classification": "good"},
            timestamp=1700000000.0,
            synced=True,
            metadata={"patient_id": "P-1234"},
        )
        summary = _stored_to_summary(result)
        assert summary.id == 42
        assert summary.ecg_hash == "test_hash"
        assert summary.patient_id == "P-1234"
        assert summary.quality_score == 85
        assert summary.quality_class == "good"
        assert summary.top_finding == "af"
        assert summary.top_finding_prob == 0.9
        assert summary.synced is True

    def test_no_patient_id(self) -> None:
        result = FakeStoredResult(metadata={})
        summary = _stored_to_summary(result)
        assert summary.patient_id is None

    def test_patient_identifier_fallback(self) -> None:
        result = FakeStoredResult(metadata={"patient_identifier": "Alt-ID"})
        summary = _stored_to_summary(result)
        assert summary.patient_id == "Alt-ID"


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestResultSummaryModel:
    """Tests for ResultSummary pydantic model."""

    def test_construction(self) -> None:
        s = ResultSummary(
            id=1,
            ecg_hash="abc",
            timestamp=1700000000.0,
            synced=True,
        )
        assert s.id == 1
        assert s.ecg_hash == "abc"

    def test_optional_fields_default_none(self) -> None:
        s = ResultSummary(
            id=1,
            ecg_hash="abc",
            timestamp=1700000000.0,
            synced=False,
        )
        assert s.patient_id is None
        assert s.quality_score is None
        assert s.urgency_tier is None

    def test_json_roundtrip(self) -> None:
        s = ResultSummary(
            id=1,
            ecg_hash="abc",
            timestamp=1700000000.0,
            synced=True,
            patient_id="P-100",
            quality_score=95.0,
            quality_class="good",
            top_finding="af",
            top_finding_prob=0.9,
            urgency_tier="urgent",
        )
        data = s.model_dump()
        rebuilt = ResultSummary(**data)
        assert rebuilt == s


class TestResultDetailModel:
    """Tests for ResultDetail pydantic model."""

    def test_construction(self) -> None:
        d = ResultDetail(
            id=1,
            ecg_hash="abc",
            timestamp=1700000000.0,
            synced=True,
            predictions={"rhythm": {"af": 0.9}},
            quality={"overall": 90},
            metadata={"patient_id": "P-100"},
        )
        assert d.predictions["rhythm"]["af"] == 0.9


class TestResultListResponse:
    """Tests for ResultListResponse pydantic model."""

    def test_construction(self) -> None:
        r = ResultListResponse(
            results=[],
            total=0,
            page=1,
            per_page=25,
            total_pages=1,
        )
        assert r.total == 0
        assert r.total_pages == 1


# ---------------------------------------------------------------------------
# API endpoint tests: GET /api/v1/results
# ---------------------------------------------------------------------------


class TestListResultsEndpoint:
    """Tests for GET /api/v1/results."""

    def test_basic_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["results"]) == 5
        assert data["page"] == 1
        assert data["per_page"] == 25

    def test_pagination(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?page=1&per_page=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["total"] == 5
        assert data["total_pages"] == 3

    def test_page_2(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?page=2&per_page=2")
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["page"] == 2

    def test_last_page(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?page=3&per_page=2")
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["page"] == 3

    def test_filter_by_quality(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?quality=poor")
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["quality_class"] == "poor"

    def test_filter_by_quality_good(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?quality=good")
        data = resp.json()
        assert data["total"] == 3
        for r in data["results"]:
            assert r["quality_class"] == "good"

    def test_filter_by_urgency(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?urgency=urgent")
        data = resp.json()
        # AF result should be urgent
        assert data["total"] >= 1
        for r in data["results"]:
            assert r["urgency_tier"] == "urgent"

    def test_filter_by_finding(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?finding=atrial_fibrillation")
        data = resp.json()
        assert data["total"] >= 1

    def test_filter_by_search(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?search=P-4821")
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["patient_id"] == "P-4821"

    def test_filter_by_date_range(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?date_from=1700000500&date_to=1700002500")
        data = resp.json()
        assert data["total"] == 2  # hash_002 and hash_003

    def test_sort_by_timestamp_asc(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?sort_by=timestamp&sort_order=asc")
        data = resp.json()
        timestamps = [r["timestamp"] for r in data["results"]]
        assert timestamps == sorted(timestamps)

    def test_sort_by_timestamp_desc(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?sort_by=timestamp&sort_order=desc")
        data = resp.json()
        timestamps = [r["timestamp"] for r in data["results"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_combined_filters(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?quality=good&urgency=normal")
        data = resp.json()
        for r in data["results"]:
            assert r["quality_class"] == "good"
            assert r["urgency_tier"] == "normal"

    def test_empty_result_set(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results?search=nonexistent_patient")
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_store_not_available(self, client_no_store: TestClient) -> None:
        resp = client_no_store.get("/api/v1/results")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# API endpoint tests: GET /api/v1/results/{result_id}
# ---------------------------------------------------------------------------


class TestGetResultEndpoint:
    """Tests for GET /api/v1/results/{result_id}."""

    def test_get_existing_result(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["ecg_hash"] == "hash_001"
        assert "predictions" in data
        assert "quality" in data
        assert "metadata" in data

    def test_get_result_predictions(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results/1")
        data = resp.json()
        assert data["predictions"]["rhythm"]["atrial_fibrillation"] == 0.92

    def test_get_nonexistent_result(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results/999")
        assert resp.status_code == 404

    def test_get_result_metadata(self, client: TestClient) -> None:
        resp = client.get("/api/v1/results/1")
        data = resp.json()
        assert data["metadata"]["patient_id"] == "P-4821"

    def test_store_not_available(self, client_no_store: TestClient) -> None:
        resp = client_no_store.get("/api/v1/results/1")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# API endpoint tests: POST /api/v1/results/export/csv
# ---------------------------------------------------------------------------


class TestExportCSVEndpoint:
    """Tests for POST /api/v1/results/export/csv."""

    def test_export_single_result(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": [1]},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "ecg_results_export.csv" in resp.headers.get("content-disposition", "")

        lines = resp.text.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "hash_001" in lines[1]

    def test_export_multiple_results(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": [1, 2, 3]},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) == 4  # header + 3 data rows

    def test_export_csv_headers(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": [1]},
        )
        header_line = resp.text.strip().split("\n")[0]
        expected_cols = [
            "id", "ecg_hash", "timestamp", "synced",
            "patient_id", "quality_score", "quality_class",
            "top_finding", "top_finding_prob", "urgency_tier",
            "predictions_json",
        ]
        for col in expected_cols:
            assert col in header_line

    def test_export_empty_ids(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": []},
        )
        assert resp.status_code == 422

    def test_export_nonexistent_ids(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": [998, 999]},
        )
        assert resp.status_code == 404

    def test_store_not_available(self, client_no_store: TestClient) -> None:
        resp = client_no_store.post(
            "/api/v1/results/export/csv",
            json={"result_ids": [1]},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestResultBrowserIntegration:
    """Integration tests combining listing and detail."""

    def test_list_then_detail(self, client: TestClient) -> None:
        """List results, pick first, then fetch its detail."""
        list_resp = client.get("/api/v1/results?per_page=1")
        assert list_resp.status_code == 200
        first_id = list_resp.json()["results"][0]["id"]

        detail_resp = client.get(f"/api/v1/results/{first_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["id"] == first_id

    def test_pagination_covers_all(self, client: TestClient) -> None:
        """Verify that paginating through all pages yields all results."""
        all_ids: set[int] = set()
        page = 1
        while True:
            resp = client.get(f"/api/v1/results?page={page}&per_page=2")
            data = resp.json()
            if not data["results"]:
                break
            for r in data["results"]:
                all_ids.add(r["id"])
            if page >= data["total_pages"]:
                break
            page += 1

        assert len(all_ids) == 5

    def test_filter_then_export(self, client: TestClient) -> None:
        """Filter results, then export the filtered set."""
        list_resp = client.get("/api/v1/results?quality=good")
        ids = [r["id"] for r in list_resp.json()["results"]]
        assert len(ids) >= 1

        export_resp = client.post(
            "/api/v1/results/export/csv",
            json={"result_ids": ids},
        )
        assert export_resp.status_code == 200
        lines = export_resp.text.strip().split("\n")
        assert len(lines) == len(ids) + 1  # header + data rows
