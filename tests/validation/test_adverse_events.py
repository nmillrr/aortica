"""Tests for aortica.validation.adverse_events — US-102.

Covers:
- Adverse event CRUD operations (append-only)
- Severity validation
- Summary aggregation (by severity, most-reported findings)
- Immutable audit trail (no update/delete)
- API endpoint registration
- Module imports
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aortica.validation.adverse_events import (
    VALID_SEVERITIES,
    AdverseEventRecord,
    AdverseEventStore,
    AdverseEventSubmission,
    AdverseEventSummary,
    FindingCount,
    SeverityCount,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> AdverseEventStore:
    """Fresh adverse event store backed by a temp SQLite file."""
    return AdverseEventStore(db_path=str(tmp_path / "adverse_events.db"))


@pytest.fixture
def populated_store(store: AdverseEventStore) -> AdverseEventStore:
    """Store pre-populated with diverse events."""
    store.report_event(
        reporter_id="dr_alpha",
        ecg_reference="ecg_001",
        event_description="AI missed inferior STEMI",
        severity="serious",
        ai_finding="normal_sinus_rhythm",
        patient_outcome="delayed cath lab activation",
    )
    store.report_event(
        reporter_id="dr_beta",
        ecg_reference="ecg_002",
        event_description="False positive AF triggered unnecessary anticoagulation",
        severity="moderate",
        ai_finding="AF",
        patient_outcome="medication discontinued, no harm",
    )
    store.report_event(
        reporter_id="dr_gamma",
        ecg_reference="ecg_003",
        event_description="Missed Wellens type A",
        severity="critical",
        ai_finding="normal_sinus_rhythm",
        patient_outcome="emergency PCI required",
    )
    store.report_event(
        reporter_id="dr_alpha",
        ecg_reference="ecg_004",
        event_description="Minor: low confidence finding confused junior resident",
        severity="minor",
        ai_finding="pvc",
        patient_outcome="no clinical impact",
    )
    store.report_event(
        reporter_id="dr_delta",
        ecg_reference="ecg_005",
        event_description="False VT alert caused unnecessary defibrillation prep",
        severity="serious",
        ai_finding="VT",
        patient_outcome="patient stressed, no physical harm",
    )
    return store


# ---------------------------------------------------------------------------
# Report event (create)
# ---------------------------------------------------------------------------


class TestReportEvent:
    """Test event creation."""

    def test_creates_event(self, store: AdverseEventStore) -> None:
        record = store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg_100",
            event_description="Test event",
            severity="minor",
            ai_finding="AF",
        )
        assert isinstance(record, AdverseEventRecord)
        assert record.id  # UUID assigned
        assert record.reporter_id == "dr_test"
        assert record.severity == "minor"
        assert record.timestamp  # non-empty

    def test_all_severities_accepted(self, store: AdverseEventStore) -> None:
        for sev in VALID_SEVERITIES:
            record = store.report_event(
                reporter_id="dr_test",
                ecg_reference="ecg",
                event_description=f"Test {sev}",
                severity=sev,
                ai_finding="AF",
            )
            assert record.severity == sev

    def test_invalid_severity_raises(self, store: AdverseEventStore) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            store.report_event(
                reporter_id="dr_test",
                ecg_reference="ecg",
                event_description="Bad severity",
                severity="catastrophic",
                ai_finding="AF",
            )

    def test_patient_outcome_optional(self, store: AdverseEventStore) -> None:
        record = store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg",
            event_description="No outcome recorded",
            severity="minor",
            ai_finding="AF",
        )
        assert record.patient_outcome == ""

    def test_patient_outcome_stored(self, store: AdverseEventStore) -> None:
        record = store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg",
            event_description="With outcome",
            severity="moderate",
            ai_finding="VT",
            patient_outcome="patient recovered fully",
        )
        assert record.patient_outcome == "patient recovered fully"

    def test_unique_ids(self, store: AdverseEventStore) -> None:
        r1 = store.report_event(
            reporter_id="dr", ecg_reference="e1",
            event_description="d1", severity="minor", ai_finding="AF",
        )
        r2 = store.report_event(
            reporter_id="dr", ecg_reference="e2",
            event_description="d2", severity="minor", ai_finding="AF",
        )
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# Store submission via Pydantic model
# ---------------------------------------------------------------------------


class TestStoreSubmission:
    """Test the Pydantic submission path."""

    def test_store_submission(self, store: AdverseEventStore) -> None:
        sub = AdverseEventSubmission(
            reporter_id="dr_pydantic",
            ecg_reference="ecg_pyd",
            event_description="Pydantic path test",
            severity="moderate",
            ai_finding="LBBB",
        )
        record = store.store_submission(sub)
        assert record.reporter_id == "dr_pydantic"
        assert record.ai_finding == "LBBB"

    def test_store_submission_invalid_severity(
        self, store: AdverseEventStore
    ) -> None:
        sub = AdverseEventSubmission(
            reporter_id="dr",
            ecg_reference="ecg",
            event_description="Bad",
            severity="extreme",
            ai_finding="AF",
        )
        with pytest.raises(ValueError, match="Invalid severity"):
            store.store_submission(sub)


# ---------------------------------------------------------------------------
# Read events
# ---------------------------------------------------------------------------


class TestReadEvents:
    """Test event retrieval (read-only)."""

    def test_get_event_by_id(
        self, populated_store: AdverseEventStore
    ) -> None:
        events = populated_store.list_events(limit=1)
        assert len(events) == 1
        fetched = populated_store.get_event(events[0].id)
        assert fetched is not None
        assert fetched.id == events[0].id

    def test_get_nonexistent_event(self, store: AdverseEventStore) -> None:
        assert store.get_event("nonexistent-id") is None

    def test_list_events_all(
        self, populated_store: AdverseEventStore
    ) -> None:
        events = populated_store.list_events(limit=100)
        assert len(events) == 5

    def test_list_events_by_severity(
        self, populated_store: AdverseEventStore
    ) -> None:
        serious_events = populated_store.list_events(severity="serious")
        assert len(serious_events) == 2
        for e in serious_events:
            assert e.severity == "serious"

    def test_list_events_limit(
        self, populated_store: AdverseEventStore
    ) -> None:
        events = populated_store.list_events(limit=2)
        assert len(events) == 2

    def test_list_events_reverse_chronological(
        self, populated_store: AdverseEventStore
    ) -> None:
        events = populated_store.list_events(limit=100)
        for i in range(len(events) - 1):
            assert events[i].timestamp >= events[i + 1].timestamp


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


class TestSummary:
    """Test aggregate statistics."""

    def test_summary_total(
        self, populated_store: AdverseEventStore
    ) -> None:
        summary = populated_store.get_summary()
        assert summary.total_events == 5

    def test_summary_by_severity(
        self, populated_store: AdverseEventStore
    ) -> None:
        summary = populated_store.get_summary()
        sev_map = {s.severity: s.count for s in summary.by_severity}
        assert sev_map["serious"] == 2
        assert sev_map["moderate"] == 1
        assert sev_map["critical"] == 1
        assert sev_map["minor"] == 1

    def test_summary_most_reported(
        self, populated_store: AdverseEventStore
    ) -> None:
        summary = populated_store.get_summary()
        finding_map = {f.ai_finding: f.count for f in summary.most_reported_findings}
        assert finding_map["normal_sinus_rhythm"] == 2
        assert finding_map["AF"] == 1

    def test_summary_empty_store(self, store: AdverseEventStore) -> None:
        summary = store.get_summary()
        assert summary.total_events == 0
        assert summary.by_severity == []
        assert summary.most_reported_findings == []


# ---------------------------------------------------------------------------
# Immutable audit trail
# ---------------------------------------------------------------------------


class TestImmutability:
    """Verify append-only behavior (no update/delete methods)."""

    def test_no_delete_method(self, store: AdverseEventStore) -> None:
        assert not hasattr(store, "delete_event")

    def test_no_update_method(self, store: AdverseEventStore) -> None:
        assert not hasattr(store, "update_event")

    def test_timestamp_is_set(self, store: AdverseEventStore) -> None:
        record = store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg",
            event_description="test",
            severity="minor",
            ai_finding="AF",
        )
        assert "T" in record.timestamp  # ISO 8601 format
        assert "+" in record.timestamp or "Z" in record.timestamp  # has timezone


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """Test adverse event API endpoint registration."""

    def test_validation_router_has_adverse_event_endpoint(self) -> None:
        from aortica.api.validation_endpoints import create_validation_router

        router = create_validation_router()
        routes = [r.path for r in router.routes]
        assert "/api/v1/validation/adverse-event" in routes

    def test_validation_router_has_adverse_events_list(self) -> None:
        from aortica.api.validation_endpoints import create_validation_router

        router = create_validation_router()
        routes = [r.path for r in router.routes]
        assert "/api/v1/validation/adverse-events" in routes

    def test_validation_router_has_adverse_events_summary(self) -> None:
        from aortica.api.validation_endpoints import create_validation_router

        router = create_validation_router()
        routes = [r.path for r in router.routes]
        assert "/api/v1/validation/adverse-events/summary" in routes

    def test_adverse_event_endpoint_post(self, tmp_path: Path) -> None:
        """Test the POST endpoint via TestClient."""
        from fastapi.testclient import TestClient

        from aortica.api.validation_endpoints import create_validation_router

        ae_store = AdverseEventStore(
            db_path=str(tmp_path / "ae_test.db")
        )
        router = create_validation_router(adverse_event_store=ae_store)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/api/v1/validation/adverse-event",
            json={
                "reporter_id": "dr_test",
                "ecg_reference": "ecg_001",
                "event_description": "Test event via API",
                "severity": "moderate",
                "ai_finding": "AF",
                "patient_outcome": "no harm",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert len(data["id"]) > 0

    def test_adverse_events_list_endpoint(self, tmp_path: Path) -> None:
        """Test the GET list endpoint."""
        from fastapi.testclient import TestClient

        from aortica.api.validation_endpoints import create_validation_router

        ae_store = AdverseEventStore(
            db_path=str(tmp_path / "ae_test.db")
        )
        ae_store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg",
            event_description="Listed event",
            severity="minor",
            ai_finding="LBBB",
        )

        router = create_validation_router(adverse_event_store=ae_store)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/api/v1/validation/adverse-events")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["severity"] == "minor"

    def test_adverse_events_summary_endpoint(self, tmp_path: Path) -> None:
        """Test the GET summary endpoint."""
        from fastapi.testclient import TestClient

        from aortica.api.validation_endpoints import create_validation_router

        ae_store = AdverseEventStore(
            db_path=str(tmp_path / "ae_test.db")
        )
        ae_store.report_event(
            reporter_id="dr_test",
            ecg_reference="ecg",
            event_description="Summary event",
            severity="critical",
            ai_finding="VT",
        )

        router = create_validation_router(adverse_event_store=ae_store)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/api/v1/validation/adverse-events/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 1
        assert len(data["by_severity"]) == 1
        assert data["by_severity"][0]["severity"] == "critical"

    def test_no_store_returns_no_op(self) -> None:
        """Endpoints should return no-op responses when no store is configured."""
        from fastapi.testclient import TestClient

        from aortica.api.validation_endpoints import create_validation_router

        router = create_validation_router()

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # POST returns empty id
        response = client.post(
            "/api/v1/validation/adverse-event",
            json={
                "reporter_id": "dr_test",
                "ecg_reference": "ecg",
                "event_description": "No store",
                "severity": "minor",
                "ai_finding": "AF",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "no_store_configured"

        # GET list returns empty
        response = client.get("/api/v1/validation/adverse-events")
        assert response.status_code == 200
        assert response.json() == []

        # GET summary returns zeros
        response = client.get("/api/v1/validation/adverse-events/summary")
        assert response.status_code == 200
        assert response.json()["total_events"] == 0

    def test_invalid_severity_returns_422(self, tmp_path: Path) -> None:
        """POST with invalid severity should return 422."""
        from fastapi.testclient import TestClient

        from aortica.api.validation_endpoints import create_validation_router

        ae_store = AdverseEventStore(
            db_path=str(tmp_path / "ae_test.db")
        )
        router = create_validation_router(adverse_event_store=ae_store)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/api/v1/validation/adverse-event",
            json={
                "reporter_id": "dr_test",
                "ecg_reference": "ecg",
                "event_description": "Bad severity",
                "severity": "catastrophic",
                "ai_finding": "AF",
            },
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify adverse events module exports are accessible."""

    def test_import_from_validation(self) -> None:
        from aortica.validation import (
            AdverseEventRecord,
            AdverseEventStore,
            AdverseEventSummary,
        )

    def test_import_from_submodule(self) -> None:
        from aortica.validation.adverse_events import (
            VALID_SEVERITIES,
            AdverseEventRecord,
            AdverseEventStore,
            AdverseEventSubmission,
            AdverseEventSummary,
            FindingCount,
            SeverityCount,
        )

    def test_import_api_models(self) -> None:
        from aortica.api.validation_endpoints import (
            AdverseEventRecordResponse,
            AdverseEventSubmitRequest,
            AdverseEventSubmitResponse,
            AdverseEventSummaryResponse,
        )
