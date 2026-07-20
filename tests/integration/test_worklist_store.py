"""Tests for aortica.integration.worklist_store (US-119)."""

from __future__ import annotations

import pytest

from aortica.integration.worklist import WorklistPrioritizer
from aortica.integration.worklist_store import WorklistEntry, WorklistStore


def _critical() -> dict:
    return {"ischaemia": {"STEMI": 0.95}}


def _routine() -> dict:
    return {"rhythm": {"normal_sinus_rhythm": 0.99}}


def _prioritized(results: list, ids: list):
    return WorklistPrioritizer().prioritize(results, ecg_ids=ids)


def test_add_from_prioritized_and_list() -> None:
    store = WorklistStore()
    wl = _prioritized([_critical(), _routine()], ["a", "b"])
    store.add_from_prioritized(wl)
    entries = store.list_entries()
    assert len(entries) == 2
    # Critical should sort first (higher urgency).
    assert entries[0].ecg_id == "a"
    assert entries[0].urgency_tier == "critical"


def test_default_status_pending() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    assert store.get("a").review_status == "pending"


def test_update_status_and_assignee() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    updated = store.update_entry("a", review_status="completed", assignee="dr_smith")
    assert updated is not None
    assert updated.review_status == "completed"
    assert updated.assignee == "dr_smith"
    assert updated.reviewed_at is not None


def test_update_missing_returns_none() -> None:
    store = WorklistStore()
    assert store.update_entry("nope", review_status="completed") is None


def test_invalid_status_raises() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    with pytest.raises(ValueError):
        store.update_entry("a", review_status="bogus")


def test_reverting_from_completed_clears_reviewed_at() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    store.update_entry("a", review_status="completed")
    reverted = store.update_entry("a", review_status="in-progress")
    assert reverted.reviewed_at is None


def test_filter_by_status() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical(), _routine()], ["a", "b"]))
    store.update_entry("a", review_status="completed")
    pending = store.list_entries(status="pending")
    assert {e.ecg_id for e in pending} == {"b"}


def test_filter_by_tier() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical(), _routine()], ["a", "b"]))
    crit = store.list_entries(tier="critical")
    assert {e.ecg_id for e in crit} == {"a"}


def test_filter_by_finding() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    assert len(store.list_entries(finding="STEMI")) == 1
    assert store.list_entries(finding="VF") == []


def test_add_preserves_review_state_on_reingest() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    store.update_entry("a", review_status="in-progress", assignee="dr_a")
    # Re-ingest same ECG (e.g. re-scored).
    store.add_from_prioritized(_prioritized([_critical()], ["a"]))
    entry = store.get("a")
    assert entry.review_status == "in-progress"
    assert entry.assignee == "dr_a"


def test_summary_metrics() -> None:
    store = WorklistStore()
    store.add_from_prioritized(_prioritized([_critical(), _routine()], ["a", "b"]))
    store.update_entry("a", review_status="completed")
    summary = store.summary()
    assert summary["total"] == 2
    assert summary["critical_count"] == 1
    assert summary["completed_count"] == 1
    assert summary["total_pending"] == 1
    assert summary["avg_time_to_review_seconds"] is not None


def test_add_entry_direct() -> None:
    store = WorklistStore()
    entry = WorklistEntry(
        ecg_id="x",
        urgency_score=85,
        urgency_tier="critical",
        top_finding="VT",
        recommended_action="Immediate review",
        patient_id="P1",
    )
    store.add_entry(entry)
    got = store.get("x")
    assert got.patient_id == "P1"
    assert got.urgency_score == 85
