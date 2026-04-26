"""Tests for ``aortica.api.feedback`` — Clinician Feedback Collection API.

Covers:
  • Pydantic model construction and serialization
  • FeedbackStore CRUD operations (store, get, list, delete)
  • FeedbackStore stats aggregation (agreement rate, most-rejected)
  • API endpoint integration (POST /feedback, GET /feedback/stats,
    GET /feedback, DELETE /feedback/{id})
  • Validation and error handling
  • Import structure
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from aortica.api.feedback import (
    VALID_ACTIONS,
    VALID_TASKS,
    FeedbackRecord,
    FeedbackResponse,
    FeedbackStatsResponse,
    FeedbackStore,
    FeedbackSubmission,
    FindingStats,
    create_feedback_router,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db() -> str:
    """Return a path to a temporary SQLite database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path  # type: ignore[misc]
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def store(temp_db: str) -> FeedbackStore:
    """Return a FeedbackStore backed by a temporary database."""
    return FeedbackStore(db_path=temp_db)


@pytest.fixture
def sample_submission() -> FeedbackSubmission:
    """Return a valid feedback submission."""
    return FeedbackSubmission(
        ecg_reference_id="ecg-123",
        finding_name="Atrial Fibrillation",
        task="rhythm",
        action="accept",
        comment="Agree with finding",
        clinician_id="dr-smith",
        ai_confidence=0.92,
    )


# ===================================================================
# Pydantic model tests
# ===================================================================


class TestFeedbackSubmission:
    """FeedbackSubmission Pydantic model tests."""

    def test_construction_minimal(self) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="accept",
        )
        assert sub.ecg_reference_id == "ecg-1"
        assert sub.action == "accept"
        assert sub.comment is None
        assert sub.clinician_id is None
        assert sub.ai_confidence is None

    def test_construction_full(self) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="reject",
            comment="Disagree — likely AFL",
            clinician_id="dr-jones",
            ai_confidence=0.75,
        )
        assert sub.comment == "Disagree — likely AFL"
        assert sub.clinician_id == "dr-jones"
        assert sub.ai_confidence == 0.75

    def test_serialization_roundtrip(self) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="modify",
            comment="Changed to AFL",
        )
        data = json.loads(sub.model_dump_json())
        restored = FeedbackSubmission(**data)
        assert restored.action == "modify"
        assert restored.comment == "Changed to AFL"


class TestFeedbackRecord:
    """FeedbackRecord Pydantic model tests."""

    def test_construction(self) -> None:
        rec = FeedbackRecord(
            id="abc-123",
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="accept",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert rec.id == "abc-123"
        assert rec.timestamp == "2026-01-01T00:00:00Z"

    def test_roundtrip(self) -> None:
        rec = FeedbackRecord(
            id="abc-123",
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="reject",
            comment="Wrong",
            clinician_id="dr-x",
            ai_confidence=0.3,
            timestamp="2026-01-01T00:00:00Z",
        )
        data = json.loads(rec.model_dump_json())
        restored = FeedbackRecord(**data)
        assert restored.id == rec.id
        assert restored.ai_confidence == 0.3


class TestFeedbackResponse:
    """FeedbackResponse Pydantic model tests."""

    def test_construction(self) -> None:
        resp = FeedbackResponse(id="abc-123", status="recorded")
        assert resp.id == "abc-123"
        assert resp.status == "recorded"

    def test_default_status(self) -> None:
        resp = FeedbackResponse(id="xyz")
        assert resp.status == "recorded"


class TestFindingStats:
    """FindingStats Pydantic model tests."""

    def test_construction(self) -> None:
        stats = FindingStats(
            finding_name="AF",
            task="rhythm",
            total=10,
            accept_count=7,
            reject_count=2,
            modify_count=1,
            agreement_rate=0.7,
        )
        assert stats.total == 10
        assert stats.agreement_rate == 0.7


class TestFeedbackStatsResponse:
    """FeedbackStatsResponse Pydantic model tests."""

    def test_construction(self) -> None:
        resp = FeedbackStatsResponse(
            total_feedback=5,
            overall_agreement_rate=0.6,
            per_finding=[],
            most_rejected=[],
        )
        assert resp.total_feedback == 5
        assert resp.per_finding == []


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_valid_actions(self) -> None:
        assert VALID_ACTIONS == {"accept", "reject", "modify"}

    def test_valid_tasks(self) -> None:
        assert "rhythm" in VALID_TASKS
        assert "structural" in VALID_TASKS
        assert "ischaemia" in VALID_TASKS
        assert "risk" in VALID_TASKS


# ===================================================================
# FeedbackStore CRUD tests
# ===================================================================


class TestFeedbackStoreCRUD:
    """SQLite-backed CRUD operations."""

    def test_store_and_get(
        self, store: FeedbackStore, sample_submission: FeedbackSubmission
    ) -> None:
        record = store.store_feedback(sample_submission)
        assert isinstance(record, FeedbackRecord)
        assert record.finding_name == "Atrial Fibrillation"
        assert record.action == "accept"
        assert record.id  # non-empty

        retrieved = store.get_feedback(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.ecg_reference_id == "ecg-123"

    def test_get_nonexistent(self, store: FeedbackStore) -> None:
        result = store.get_feedback("nonexistent-id")
        assert result is None

    def test_store_reject(self, store: FeedbackStore) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-2",
            finding_name="VT",
            task="rhythm",
            action="reject",
            comment="Artifact, not VT",
        )
        record = store.store_feedback(sub)
        assert record.action == "reject"
        assert record.comment == "Artifact, not VT"

    def test_store_modify(self, store: FeedbackStore) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-3",
            finding_name="LBBB",
            task="rhythm",
            action="modify",
            comment="Rate-related LBBB",
        )
        record = store.store_feedback(sub)
        assert record.action == "modify"

    def test_store_invalid_action(self, store: FeedbackStore) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-4",
            finding_name="AF",
            task="rhythm",
            action="invalid",
        )
        with pytest.raises(ValueError, match="Invalid action"):
            store.store_feedback(sub)

    def test_list_all(
        self, store: FeedbackStore, sample_submission: FeedbackSubmission
    ) -> None:
        store.store_feedback(sample_submission)
        sub2 = FeedbackSubmission(
            ecg_reference_id="ecg-2",
            finding_name="VT",
            task="rhythm",
            action="reject",
        )
        store.store_feedback(sub2)
        records = store.list_feedback()
        assert len(records) == 2

    def test_list_filter_by_ecg_ref(
        self, store: FeedbackStore
    ) -> None:
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-A",
                finding_name="AF",
                task="rhythm",
                action="accept",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-B",
                finding_name="VT",
                task="rhythm",
                action="reject",
            )
        )
        results = store.list_feedback(ecg_reference_id="ecg-A")
        assert len(results) == 1
        assert results[0].ecg_reference_id == "ecg-A"

    def test_list_filter_by_clinician(
        self, store: FeedbackStore
    ) -> None:
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="AF",
                task="rhythm",
                action="accept",
                clinician_id="dr-a",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="VT",
                task="rhythm",
                action="reject",
                clinician_id="dr-b",
            )
        )
        results = store.list_feedback(clinician_id="dr-a")
        assert len(results) == 1
        assert results[0].clinician_id == "dr-a"

    def test_list_with_limit(self, store: FeedbackStore) -> None:
        for i in range(5):
            store.store_feedback(
                FeedbackSubmission(
                    ecg_reference_id=f"ecg-{i}",
                    finding_name="AF",
                    task="rhythm",
                    action="accept",
                )
            )
        results = store.list_feedback(limit=3)
        assert len(results) == 3

    def test_list_empty(self, store: FeedbackStore) -> None:
        results = store.list_feedback()
        assert results == []

    def test_delete_existing(
        self, store: FeedbackStore, sample_submission: FeedbackSubmission
    ) -> None:
        record = store.store_feedback(sample_submission)
        assert store.delete_feedback(record.id) is True
        assert store.get_feedback(record.id) is None

    def test_delete_nonexistent(self, store: FeedbackStore) -> None:
        assert store.delete_feedback("nonexistent") is False

    def test_timestamp_format(
        self, store: FeedbackStore, sample_submission: FeedbackSubmission
    ) -> None:
        record = store.store_feedback(sample_submission)
        # Should be a valid ISO 8601 timestamp
        assert "T" in record.timestamp
        assert "+" in record.timestamp or "Z" in record.timestamp or record.timestamp.endswith("+00:00")

    def test_unique_ids(
        self, store: FeedbackStore, sample_submission: FeedbackSubmission
    ) -> None:
        r1 = store.store_feedback(sample_submission)
        r2 = store.store_feedback(sample_submission)
        assert r1.id != r2.id

    def test_ai_confidence_stored(self, store: FeedbackStore) -> None:
        sub = FeedbackSubmission(
            ecg_reference_id="ecg-1",
            finding_name="AF",
            task="rhythm",
            action="accept",
            ai_confidence=0.92,
        )
        record = store.store_feedback(sub)
        retrieved = store.get_feedback(record.id)
        assert retrieved is not None
        assert retrieved.ai_confidence == 0.92


# ===================================================================
# FeedbackStore stats tests
# ===================================================================


class TestFeedbackStoreStats:
    """Aggregate statistics from the feedback store."""

    def test_empty_stats(self, store: FeedbackStore) -> None:
        stats = store.get_stats()
        assert stats.total_feedback == 0
        assert stats.overall_agreement_rate == 0.0
        assert stats.per_finding == []
        assert stats.most_rejected == []

    def test_all_accepted(self, store: FeedbackStore) -> None:
        for _ in range(3):
            store.store_feedback(
                FeedbackSubmission(
                    ecg_reference_id="ecg-1",
                    finding_name="AF",
                    task="rhythm",
                    action="accept",
                )
            )
        stats = store.get_stats()
        assert stats.total_feedback == 3
        assert stats.overall_agreement_rate == 1.0
        assert len(stats.per_finding) == 1
        assert stats.per_finding[0].accept_count == 3
        assert stats.per_finding[0].reject_count == 0
        assert stats.most_rejected == []

    def test_mixed_actions(self, store: FeedbackStore) -> None:
        for action in ["accept", "reject", "modify", "accept"]:
            store.store_feedback(
                FeedbackSubmission(
                    ecg_reference_id="ecg-1",
                    finding_name="AF",
                    task="rhythm",
                    action=action,
                )
            )
        stats = store.get_stats()
        assert stats.total_feedback == 4
        assert stats.overall_agreement_rate == 0.5  # 2/4
        assert len(stats.per_finding) == 1
        pf = stats.per_finding[0]
        assert pf.accept_count == 2
        assert pf.reject_count == 1
        assert pf.modify_count == 1

    def test_most_rejected_sorted(self, store: FeedbackStore) -> None:
        # AF: 1 reject / 2 total = 50% rejection
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="AF",
                task="rhythm",
                action="reject",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="AF",
                task="rhythm",
                action="accept",
            )
        )
        # VT: 2 reject / 2 total = 100% rejection
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="VT",
                task="rhythm",
                action="reject",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="VT",
                task="rhythm",
                action="reject",
            )
        )
        stats = store.get_stats()
        assert len(stats.most_rejected) == 2
        # VT should be first (higher rejection rate)
        assert stats.most_rejected[0].finding_name == "VT"
        assert stats.most_rejected[1].finding_name == "AF"

    def test_most_rejected_top5_limit(self, store: FeedbackStore) -> None:
        # Create 7 findings, all rejected
        for i in range(7):
            store.store_feedback(
                FeedbackSubmission(
                    ecg_reference_id="ecg-1",
                    finding_name=f"Finding-{i}",
                    task="rhythm",
                    action="reject",
                )
            )
        stats = store.get_stats()
        assert len(stats.most_rejected) == 5

    def test_multi_finding_per_finding_stats(self, store: FeedbackStore) -> None:
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="AF",
                task="rhythm",
                action="accept",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="LVH",
                task="structural",
                action="reject",
            )
        )
        stats = store.get_stats()
        assert stats.total_feedback == 2
        assert len(stats.per_finding) == 2
        names = {pf.finding_name for pf in stats.per_finding}
        assert names == {"AF", "LVH"}

    def test_agreement_rate_per_finding(self, store: FeedbackStore) -> None:
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-1",
                finding_name="AF",
                task="rhythm",
                action="accept",
            )
        )
        store.store_feedback(
            FeedbackSubmission(
                ecg_reference_id="ecg-2",
                finding_name="AF",
                task="rhythm",
                action="reject",
            )
        )
        stats = store.get_stats()
        af_stats = stats.per_finding[0]
        assert af_stats.finding_name == "AF"
        assert af_stats.agreement_rate == 0.5


# ===================================================================
# API endpoint tests (FastAPI TestClient)
# ===================================================================

try:
    from starlette.testclient import TestClient

    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


@pytest.fixture
def api_client(temp_db: str) -> Any:
    """Create a FastAPI test client with feedback router.

    Builds a minimal FastAPI app with the temp-backed store injected
    directly into ``create_feedback_router`` so that the router closure
    references the correct (isolated) database.
    """
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from aortica.api.feedback import create_feedback_router

    app = fastapi.FastAPI()
    temp_store = FeedbackStore(db_path=temp_db)
    feedback_router = create_feedback_router(temp_store)
    app.include_router(feedback_router)

    return TestClient(app)


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="starlette not installed")
class TestFeedbackEndpoints:
    """API endpoint integration tests."""

    def test_post_feedback_200(self, api_client: Any) -> None:
        resp = api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "recorded"
        assert "id" in data

    def test_post_feedback_with_comment(self, api_client: Any) -> None:
        resp = api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "VT",
                "task": "rhythm",
                "action": "reject",
                "comment": "Artifact, not VT",
                "clinician_id": "dr-smith",
            },
        )
        assert resp.status_code == 200

    def test_post_feedback_invalid_action(self, api_client: Any) -> None:
        resp = api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "invalid_action",
            },
        )
        assert resp.status_code == 422

    def test_post_feedback_missing_fields(self, api_client: Any) -> None:
        resp = api_client.post(
            "/api/v1/feedback",
            json={"ecg_reference_id": "ecg-1"},
        )
        assert resp.status_code == 422

    def test_get_feedback_stats_empty(self, api_client: Any) -> None:
        resp = api_client.get("/api/v1/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_feedback"] == 0
        assert data["overall_agreement_rate"] == 0.0

    def test_get_feedback_stats_with_data(self, api_client: Any) -> None:
        # Submit some feedback
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "VT",
                "task": "rhythm",
                "action": "reject",
            },
        )

        resp = api_client.get("/api/v1/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_feedback"] == 2
        assert data["overall_agreement_rate"] == 0.5

    def test_get_feedback_stats_most_rejected(self, api_client: Any) -> None:
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "VT",
                "task": "rhythm",
                "action": "reject",
            },
        )
        resp = api_client.get("/api/v1/feedback/stats")
        data = resp.json()
        assert len(data["most_rejected"]) == 1
        assert data["most_rejected"][0]["finding_name"] == "VT"

    def test_list_feedback(self, api_client: Any) -> None:
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        resp = api_client.get("/api/v1/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["finding_name"] == "AF"

    def test_list_feedback_filter_ecg_ref(self, api_client: Any) -> None:
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-A",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-B",
                "finding_name": "VT",
                "task": "rhythm",
                "action": "reject",
            },
        )
        resp = api_client.get("/api/v1/feedback?ecg_reference_id=ecg-A")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ecg_reference_id"] == "ecg-A"

    def test_delete_feedback(self, api_client: Any) -> None:
        # Create
        post_resp = api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        feedback_id = post_resp.json()["id"]

        # Delete
        del_resp = api_client.delete(f"/api/v1/feedback/{feedback_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # Verify deleted
        list_resp = api_client.get("/api/v1/feedback")
        assert len(list_resp.json()) == 0

    def test_delete_nonexistent(self, api_client: Any) -> None:
        resp = api_client.delete("/api/v1/feedback/nonexistent-id")
        assert resp.status_code == 404

    def test_content_type_json(self, api_client: Any) -> None:
        resp = api_client.post(
            "/api/v1/feedback",
            json={
                "ecg_reference_id": "ecg-1",
                "finding_name": "AF",
                "task": "rhythm",
                "action": "accept",
            },
        )
        assert "application/json" in resp.headers["content-type"]

    def test_stats_content_type(self, api_client: Any) -> None:
        resp = api_client.get("/api/v1/feedback/stats")
        assert "application/json" in resp.headers["content-type"]


# ===================================================================
# Import tests
# ===================================================================


class TestImports:
    """Verify module structure and imports."""

    def test_feedback_module_imports(self) -> None:
        from aortica.api import feedback  # noqa: F401

    def test_pydantic_models_importable(self) -> None:
        from aortica.api.feedback import (
            FeedbackRecord,
            FeedbackResponse,
            FeedbackStatsResponse,
            FeedbackSubmission,
            FindingStats,
        )

        assert FeedbackSubmission is not None
        assert FeedbackRecord is not None
        assert FeedbackResponse is not None
        assert FeedbackStatsResponse is not None
        assert FindingStats is not None

    def test_store_importable(self) -> None:
        from aortica.api.feedback import FeedbackStore

        assert FeedbackStore is not None

    def test_router_factory_importable(self) -> None:
        from aortica.api.feedback import create_feedback_router

        assert create_feedback_router is not None
