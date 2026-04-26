"""Clinician Feedback Collection API — ``POST /api/v1/feedback``.

Provides CRUD operations for collecting clinician feedback on AI
findings, backed by a local SQLite database.  Feedback is stored
per-finding with an action (accept / reject / modify), optional
comment, and clinician identifier.  Aggregate statistics are
available via ``GET /api/v1/feedback/stats``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse as _JSONResponse

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FeedbackSubmission(BaseModel):
    """A single piece of clinician feedback on an AI finding."""

    ecg_reference_id: str = Field(
        ..., description="Unique identifier for the ECG analysis session"
    )
    finding_name: str = Field(
        ..., description="Name of the AI finding being reviewed"
    )
    task: str = Field(
        ..., description="Task head name (rhythm, structural, ischaemia, risk)"
    )
    action: str = Field(
        ...,
        description="Clinician action: 'accept', 'reject', or 'modify'",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Optional free-text comment from the clinician",
    )
    clinician_id: Optional[str] = Field(
        default=None,
        description="Optional clinician identifier",
    )
    ai_confidence: Optional[float] = Field(
        default=None,
        description="AI confidence for the finding at the time of feedback",
    )


class FeedbackRecord(BaseModel):
    """Stored feedback record with server-assigned metadata."""

    id: str = Field(..., description="Unique feedback record ID")
    ecg_reference_id: str = Field(
        ..., description="ECG analysis session identifier"
    )
    finding_name: str = Field(..., description="Finding name")
    task: str = Field(..., description="Task head name")
    action: str = Field(..., description="Clinician action taken")
    comment: Optional[str] = Field(default=None, description="Clinician comment")
    clinician_id: Optional[str] = Field(
        default=None, description="Clinician identifier"
    )
    ai_confidence: Optional[float] = Field(
        default=None, description="AI confidence at time of feedback"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    id: str = Field(..., description="Assigned feedback record ID")
    status: str = Field(
        default="recorded", description="Submission status"
    )


class FindingStats(BaseModel):
    """Aggregated statistics for a single finding."""

    finding_name: str = Field(..., description="Finding name")
    task: str = Field(..., description="Task head name")
    total: int = Field(..., description="Total feedback count")
    accept_count: int = Field(default=0, description="Number of accepts")
    reject_count: int = Field(default=0, description="Number of rejects")
    modify_count: int = Field(default=0, description="Number of modifications")
    agreement_rate: float = Field(
        ...,
        description="Accept / total ratio (0–1)",
    )


class FeedbackStatsResponse(BaseModel):
    """Aggregate feedback statistics across all findings."""

    total_feedback: int = Field(..., description="Total feedback entries")
    overall_agreement_rate: float = Field(
        ..., description="Global accept / total ratio"
    )
    per_finding: List[FindingStats] = Field(
        ..., description="Per-finding breakdown"
    )
    most_rejected: List[FindingStats] = Field(
        ...,
        description="Findings with highest rejection rates (top 5)",
    )


# ---------------------------------------------------------------------------
# Valid actions
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"accept", "reject", "modify"}

VALID_TASKS = {"rhythm", "structural", "ischaemia", "risk"}


# ---------------------------------------------------------------------------
# SQLite-backed feedback store
# ---------------------------------------------------------------------------


class FeedbackStore:
    """SQLite-backed CRUD store for clinician feedback.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created automatically
        if it does not exist.  Defaults to ``feedback.db`` in the
        current working directory.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path.cwd() / "feedback.db")
        self._db_path = db_path
        self._ensure_schema()

    # ---- schema ----------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the ``feedback`` table if it does not exist."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    ecg_reference_id TEXT NOT NULL,
                    finding_name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    action TEXT NOT NULL,
                    comment TEXT,
                    clinician_id TEXT,
                    ai_confidence REAL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ---- CRUD ------------------------------------------------------------

    def store_feedback(self, submission: FeedbackSubmission) -> FeedbackRecord:
        """Persist a feedback submission and return the full record."""
        if submission.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{submission.action}'. "
                f"Must be one of: {', '.join(sorted(VALID_ACTIONS))}"
            )

        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = FeedbackRecord(
            id=record_id,
            ecg_reference_id=submission.ecg_reference_id,
            finding_name=submission.finding_name,
            task=submission.task,
            action=submission.action,
            comment=submission.comment,
            clinician_id=submission.clinician_id,
            ai_confidence=submission.ai_confidence,
            timestamp=timestamp,
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO feedback
                    (id, ecg_reference_id, finding_name, task, action,
                     comment, clinician_id, ai_confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.ecg_reference_id,
                    record.finding_name,
                    record.task,
                    record.action,
                    record.comment,
                    record.clinician_id,
                    record.ai_confidence,
                    record.timestamp,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return record

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Retrieve a single feedback record by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return FeedbackRecord(
            id=row["id"],
            ecg_reference_id=row["ecg_reference_id"],
            finding_name=row["finding_name"],
            task=row["task"],
            action=row["action"],
            comment=row["comment"],
            clinician_id=row["clinician_id"],
            ai_confidence=row["ai_confidence"],
            timestamp=row["timestamp"],
        )

    def list_feedback(
        self,
        *,
        ecg_reference_id: Optional[str] = None,
        clinician_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        """List feedback records with optional filters."""
        query = "SELECT * FROM feedback"
        params: List[Any] = []
        conditions: List[str] = []

        if ecg_reference_id is not None:
            conditions.append("ecg_reference_id = ?")
            params.append(ecg_reference_id)
        if clinician_id is not None:
            conditions.append("clinician_id = ?")
            params.append(clinician_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        return [
            FeedbackRecord(
                id=row["id"],
                ecg_reference_id=row["ecg_reference_id"],
                finding_name=row["finding_name"],
                task=row["task"],
                action=row["action"],
                comment=row["comment"],
                clinician_id=row["clinician_id"],
                ai_confidence=row["ai_confidence"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a single feedback record. Returns True if deleted."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM feedback WHERE id = ?", (feedback_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # ---- stats -----------------------------------------------------------

    def get_stats(self) -> FeedbackStatsResponse:
        """Compute aggregate feedback statistics."""
        conn = self._get_connection()
        try:
            # Total count
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback"
            ).fetchone()
            total: int = total_row["cnt"] if total_row else 0

            # Overall agreement rate
            accept_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback WHERE action = 'accept'"
            ).fetchone()
            accept_total: int = accept_row["cnt"] if accept_row else 0
            overall_agreement = accept_total / total if total > 0 else 0.0

            # Per-finding stats
            rows = conn.execute(
                """
                SELECT
                    finding_name,
                    task,
                    COUNT(*) as total,
                    SUM(CASE WHEN action = 'accept' THEN 1 ELSE 0 END) as accept_count,
                    SUM(CASE WHEN action = 'reject' THEN 1 ELSE 0 END) as reject_count,
                    SUM(CASE WHEN action = 'modify' THEN 1 ELSE 0 END) as modify_count
                FROM feedback
                GROUP BY finding_name, task
                ORDER BY total DESC
                """
            ).fetchall()
        finally:
            conn.close()

        per_finding: List[FindingStats] = []
        for row in rows:
            row_total = int(row["total"])
            per_finding.append(
                FindingStats(
                    finding_name=row["finding_name"],
                    task=row["task"],
                    total=row_total,
                    accept_count=int(row["accept_count"]),
                    reject_count=int(row["reject_count"]),
                    modify_count=int(row["modify_count"]),
                    agreement_rate=int(row["accept_count"]) / row_total
                    if row_total > 0
                    else 0.0,
                )
            )

        # Most rejected: sort by rejection rate descending, top 5
        most_rejected = sorted(
            [f for f in per_finding if f.reject_count > 0],
            key=lambda f: f.reject_count / f.total if f.total > 0 else 0.0,
            reverse=True,
        )[:5]

        return FeedbackStatsResponse(
            total_feedback=total,
            overall_agreement_rate=overall_agreement,
            per_finding=per_finding,
            most_rejected=most_rejected,
        )


# ---------------------------------------------------------------------------
# Router factory for FastAPI integration
# ---------------------------------------------------------------------------


def create_feedback_router(store: Optional[FeedbackStore] = None) -> Any:
    """Create a FastAPI APIRouter with feedback endpoints.

    Parameters
    ----------
    store:
        A :class:`FeedbackStore` instance.  If ``None`` a default
        store is created using ``feedback.db`` in the cwd.
    """
    if not HAS_FASTAPI:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for the feedback router. "
            "Install with: pip install aortica[api]"
        )

    JSONResponse = _JSONResponse  # local alias for readability

    if store is None:
        store = FeedbackStore()

    router = APIRouter(prefix="/api/v1", tags=["feedback"])

    @router.post(
        "/feedback",
        response_model=FeedbackResponse,
        summary="Submit clinician feedback on a finding",
    )
    async def submit_feedback(request: Request) -> Any:  # type: ignore[arg-type]
        """Record clinician feedback on an AI finding.

        Accepts JSON with ``ecg_reference_id``, ``finding_name``,
        ``task``, ``action`` (accept/reject/modify), and optional
        ``comment`` and ``clinician_id`` fields.
        """
        body = await request.json()

        try:
            submission = FeedbackSubmission(**body)
        except Exception as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid feedback submission: {exc}"},
            )

        if submission.action not in VALID_ACTIONS:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": (
                        f"Invalid action '{submission.action}'. "
                        f"Must be one of: {', '.join(sorted(VALID_ACTIONS))}"
                    )
                },
            )

        record = store.store_feedback(submission)
        return FeedbackResponse(id=record.id, status="recorded")

    @router.get(
        "/feedback/stats",
        response_model=FeedbackStatsResponse,
        summary="Get aggregate feedback statistics",
    )
    async def feedback_stats() -> FeedbackStatsResponse:
        """Return aggregate feedback statistics.

        Includes overall agreement rate, per-finding breakdowns, and
        the most-rejected findings.
        """
        return store.get_stats()

    @router.get(
        "/feedback",
        response_model=List[FeedbackRecord],
        summary="List feedback records",
    )
    async def list_feedback(
        ecg_reference_id: Optional[str] = None,
        clinician_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        """List feedback records with optional filters."""
        return store.list_feedback(
            ecg_reference_id=ecg_reference_id,
            clinician_id=clinician_id,
            limit=limit,
        )

    @router.delete(
        "/feedback/{feedback_id}",
        summary="Delete a feedback record",
    )
    async def delete_feedback(feedback_id: str) -> Any:
        """Delete a specific feedback record by ID."""
        deleted = store.delete_feedback(feedback_id)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Feedback record '{feedback_id}' not found"},
            )
        return {"status": "deleted", "id": feedback_id}

    return router
