"""Stateful worklist store with review tracking (US-119).

Builds on the stateless :class:`~aortica.integration.worklist.WorklistPrioritizer`
(US-086) by persisting prioritized ECG entries together with mutable review
state (pending / in-progress / completed), an optional assignee, and
timestamps used for the "time-to-review" summary metric.

The store is thread-safe (an ``RLock`` guards all access) so it can back a
threaded FastAPI server.  It is in-memory; entries live for the process
lifetime, which is sufficient for a review session.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REVIEW_STATUSES = ("pending", "in-progress", "completed")
URGENCY_TIERS = ("critical", "high", "moderate", "low")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


@dataclass
class WorklistEntry:
    """A single reviewable ECG in the worklist."""

    ecg_id: str
    urgency_score: int
    urgency_tier: str
    top_finding: str
    recommended_action: str
    patient_id: Optional[str] = None
    acquired_at: Optional[str] = None
    review_status: str = "pending"
    assignee: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    reviewed_at: Optional[datetime] = None
    active_findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecg_id": self.ecg_id,
            "urgency_score": self.urgency_score,
            "urgency_tier": self.urgency_tier,
            "top_finding": self.top_finding,
            "recommended_action": self.recommended_action,
            "patient_id": self.patient_id,
            "acquired_at": self.acquired_at,
            "review_status": self.review_status,
            "assignee": self.assignee,
            "created_at": _iso(self.created_at),
            "reviewed_at": _iso(self.reviewed_at),
            "active_findings": list(self.active_findings),
        }


class WorklistStore:
    """In-memory, thread-safe store of prioritized worklist entries."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, WorklistEntry] = {}

    # -- Ingestion ----------------------------------------------------------

    def add_entry(self, entry: WorklistEntry) -> WorklistEntry:
        """Insert or replace *entry* keyed by ``ecg_id``."""
        with self._lock:
            self._entries[entry.ecg_id] = entry
        return entry

    def add_from_item(
        self,
        item: Any,
        *,
        patient_id: Optional[str] = None,
        acquired_at: Optional[str] = None,
    ) -> WorklistEntry:
        """Add an entry from a :class:`WorklistItem`.

        If an entry with the same ``ecg_id`` already exists, its review
        state (status/assignee/timestamps) is preserved and only the
        scoring fields are refreshed.
        """
        with self._lock:
            existing = self._entries.get(item.ecg_id)
            entry = WorklistEntry(
                ecg_id=item.ecg_id,
                urgency_score=item.urgency_score,
                urgency_tier=item.urgency_tier,
                top_finding=item.top_finding,
                recommended_action=item.recommended_action,
                patient_id=patient_id,
                acquired_at=acquired_at,
                active_findings=list(item.active_findings),
            )
            if existing is not None:
                entry.review_status = existing.review_status
                entry.assignee = existing.assignee
                entry.created_at = existing.created_at
                entry.reviewed_at = existing.reviewed_at
                if patient_id is None:
                    entry.patient_id = existing.patient_id
                if acquired_at is None:
                    entry.acquired_at = existing.acquired_at
            self._entries[item.ecg_id] = entry
        return entry

    def add_from_prioritized(
        self,
        worklist: Any,
        *,
        patient_ids: Optional[Dict[str, str]] = None,
    ) -> List[WorklistEntry]:
        """Bulk-add every item from a ``PrioritizedWorklist``."""
        added: List[WorklistEntry] = []
        for item in worklist.items:
            pid = (patient_ids or {}).get(item.ecg_id)
            added.append(self.add_from_item(item, patient_id=pid))
        return added

    # -- Queries ------------------------------------------------------------

    def get(self, ecg_id: str) -> Optional[WorklistEntry]:
        with self._lock:
            return self._entries.get(ecg_id)

    def list_entries(
        self,
        *,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        finding: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[WorklistEntry]:
        """Return filtered entries sorted by urgency score (desc)."""
        with self._lock:
            entries = list(self._entries.values())

        def _keep(e: WorklistEntry) -> bool:
            if status and e.review_status != status:
                return False
            if tier and e.urgency_tier != tier:
                return False
            if finding and not self._has_finding(e, finding):
                return False
            if date_from and (e.acquired_at or "") < date_from:
                return False
            if date_to and (e.acquired_at or "") > date_to:
                return False
            return True

        filtered = [e for e in entries if _keep(e)]
        filtered.sort(key=lambda e: (-e.urgency_score, e.ecg_id))
        return filtered

    @staticmethod
    def _has_finding(entry: WorklistEntry, finding: str) -> bool:
        if entry.top_finding == finding:
            return True
        return any(
            f.get("class_name") == finding for f in entry.active_findings
        )

    # -- Mutation -----------------------------------------------------------

    def update_entry(
        self,
        ecg_id: str,
        *,
        review_status: Optional[str] = None,
        assignee: Optional[str] = None,
    ) -> Optional[WorklistEntry]:
        """Update review status and/or assignee for *ecg_id*.

        Returns the updated entry, or ``None`` if it does not exist.
        Transitioning to ``"completed"`` stamps ``reviewed_at``.
        """
        if review_status is not None and review_status not in REVIEW_STATUSES:
            raise ValueError(
                f"review_status must be one of {REVIEW_STATUSES}"
            )
        with self._lock:
            entry = self._entries.get(ecg_id)
            if entry is None:
                return None
            if review_status is not None:
                entry.review_status = review_status
                if review_status == "completed" and entry.reviewed_at is None:
                    entry.reviewed_at = _now()
                elif review_status != "completed":
                    entry.reviewed_at = None
            if assignee is not None:
                entry.assignee = assignee or None
            return entry

    # -- Summary ------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return worklist summary metrics."""
        with self._lock:
            entries = list(self._entries.values())

        total_pending = sum(1 for e in entries if e.review_status == "pending")
        critical_count = sum(
            1 for e in entries if e.urgency_tier == "critical"
        )
        review_times = [
            (e.reviewed_at - e.created_at).total_seconds()
            for e in entries
            if e.reviewed_at is not None
        ]
        avg_time = (
            sum(review_times) / len(review_times) if review_times else None
        )
        return {
            "total": len(entries),
            "total_pending": total_pending,
            "critical_count": critical_count,
            "completed_count": sum(
                1 for e in entries if e.review_status == "completed"
            ),
            "avg_time_to_review_seconds": avg_time,
        }
