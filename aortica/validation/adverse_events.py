"""Voluntary adverse event reporting — US-102.

Provides an append-only SQLite-backed store for capturing clinician-reported
adverse events linked to AI findings, with aggregate summary statistics.

Usage::

    from aortica.validation.adverse_events import AdverseEventStore

    store = AdverseEventStore("/path/to/db")
    event_id = store.report_event(
        reporter_id="dr_smith",
        ecg_reference="ecg_12345",
        event_description="AI missed subtle inferior STEMI",
        severity="serious",
        ai_finding="normal_sinus_rhythm",
        patient_outcome="delayed_cath_lab_activation",
    )
    summary = store.get_summary()
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SEVERITIES = {"minor", "moderate", "serious", "critical"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AdverseEventSubmission(BaseModel):
    """Request body for submitting an adverse event report."""

    reporter_id: str = Field(
        ..., description="Identifier for the reporting clinician"
    )
    ecg_reference: str = Field(
        ..., description="ECG recording reference or hash"
    )
    event_description: str = Field(
        ..., description="Free-text description of the adverse event"
    )
    severity: str = Field(
        ...,
        description="Event severity: minor, moderate, serious, or critical",
    )
    ai_finding: str = Field(
        ...,
        description="The AI finding that contributed to the adverse event",
    )
    patient_outcome: str = Field(
        default="",
        description="Description of the patient outcome",
    )


class AdverseEventRecord(BaseModel):
    """Stored adverse event record with server-assigned metadata."""

    id: str = Field(..., description="Unique event record ID (UUID)")
    reporter_id: str = Field(..., description="Reporting clinician ID")
    ecg_reference: str = Field(..., description="ECG recording reference")
    event_description: str = Field(..., description="Event description")
    severity: str = Field(..., description="Event severity level")
    ai_finding: str = Field(..., description="Contributing AI finding")
    patient_outcome: str = Field(default="", description="Patient outcome")
    timestamp: str = Field(..., description="ISO 8601 timestamp (immutable)")


class AdverseEventResponse(BaseModel):
    """Response after submitting an adverse event."""

    id: str = Field(..., description="Assigned event record ID")
    status: str = Field(default="recorded", description="Submission status")


class SeverityCount(BaseModel):
    """Count of events by severity level."""

    severity: str = Field(..., description="Severity level")
    count: int = Field(..., description="Number of events at this severity")


class FindingCount(BaseModel):
    """Count of events by AI finding."""

    ai_finding: str = Field(..., description="AI finding name")
    count: int = Field(..., description="Number of events for this finding")


class AdverseEventSummary(BaseModel):
    """Aggregate statistics for adverse events."""

    total_events: int = Field(..., description="Total reported events")
    by_severity: List[SeverityCount] = Field(
        ..., description="Event counts by severity"
    )
    most_reported_findings: List[FindingCount] = Field(
        ..., description="Top findings associated with events (top 10)"
    )


# ---------------------------------------------------------------------------
# SQLite-backed adverse event store
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS adverse_events (
    id TEXT PRIMARY KEY,
    reporter_id TEXT NOT NULL,
    ecg_reference TEXT NOT NULL,
    event_description TEXT NOT NULL,
    severity TEXT NOT NULL,
    ai_finding TEXT NOT NULL,
    patient_outcome TEXT DEFAULT '',
    timestamp TEXT NOT NULL
);
"""

_CREATE_IDX_SEVERITY = """
CREATE INDEX IF NOT EXISTS idx_ae_severity
ON adverse_events (severity);
"""

_CREATE_IDX_TIMESTAMP = """
CREATE INDEX IF NOT EXISTS idx_ae_timestamp
ON adverse_events (timestamp);
"""


class AdverseEventStore:
    """Append-only SQLite-backed store for adverse event reports.

    Events are immutable once written — there is no UPDATE or DELETE
    operation, ensuring a tamper-resistant audit trail.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created automatically if
        it does not exist.  Defaults to ``adverse_events.db`` in the
        current working directory.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path.cwd() / "adverse_events.db")
        self._db_path = db_path
        self._ensure_schema()

    # ---- schema ----------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the ``adverse_events`` table if it does not exist."""
        conn = self._get_connection()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_IDX_SEVERITY)
            conn.execute(_CREATE_IDX_TIMESTAMP)
            conn.commit()
        finally:
            conn.close()

    # ---- append-only write -----------------------------------------------

    def report_event(
        self,
        reporter_id: str,
        ecg_reference: str,
        event_description: str,
        severity: str,
        ai_finding: str,
        patient_outcome: str = "",
    ) -> AdverseEventRecord:
        """Record a new adverse event report.

        Parameters
        ----------
        reporter_id:
            Clinician identifier.
        ecg_reference:
            ECG recording reference or hash.
        event_description:
            Free-text event description.
        severity:
            One of: minor, moderate, serious, critical.
        ai_finding:
            The AI finding that contributed.
        patient_outcome:
            Description of the patient outcome.

        Returns
        -------
        AdverseEventRecord
            The stored record with server-assigned ID and timestamp.

        Raises
        ------
        ValueError
            If severity is not one of the valid values.
        """
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
            )

        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = AdverseEventRecord(
            id=record_id,
            reporter_id=reporter_id,
            ecg_reference=ecg_reference,
            event_description=event_description,
            severity=severity,
            ai_finding=ai_finding,
            patient_outcome=patient_outcome,
            timestamp=timestamp,
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO adverse_events
                    (id, reporter_id, ecg_reference, event_description,
                     severity, ai_finding, patient_outcome, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.reporter_id,
                    record.ecg_reference,
                    record.event_description,
                    record.severity,
                    record.ai_finding,
                    record.patient_outcome,
                    record.timestamp,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return record

    def store_submission(
        self, submission: AdverseEventSubmission
    ) -> AdverseEventRecord:
        """Store an adverse event from a Pydantic submission model."""
        return self.report_event(
            reporter_id=submission.reporter_id,
            ecg_reference=submission.ecg_reference,
            event_description=submission.event_description,
            severity=submission.severity,
            ai_finding=submission.ai_finding,
            patient_outcome=submission.patient_outcome,
        )

    # ---- read (no update/delete — append only) ---------------------------

    def get_event(self, event_id: str) -> Optional[AdverseEventRecord]:
        """Retrieve a single adverse event record by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM adverse_events WHERE id = ?", (event_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_events(
        self,
        *,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[AdverseEventRecord]:
        """List adverse event records, optionally filtered by severity.

        Events are returned in reverse chronological order (newest first).
        """
        query = "SELECT * FROM adverse_events"
        params: List[Any] = []

        if severity is not None:
            query += " WHERE severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        return [self._row_to_record(row) for row in rows]

    # ---- aggregate statistics --------------------------------------------

    def get_summary(self) -> AdverseEventSummary:
        """Compute aggregate adverse event statistics.

        Returns
        -------
        AdverseEventSummary
            Total count, breakdown by severity, and most-reported
            AI findings (top 10).
        """
        conn = self._get_connection()
        try:
            # Total count
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM adverse_events"
            ).fetchone()
            total: int = total_row["cnt"] if total_row else 0

            # By severity
            severity_rows = conn.execute(
                """
                SELECT severity, COUNT(*) as cnt
                FROM adverse_events
                GROUP BY severity
                ORDER BY cnt DESC
                """
            ).fetchall()

            # Most reported findings
            finding_rows = conn.execute(
                """
                SELECT ai_finding, COUNT(*) as cnt
                FROM adverse_events
                GROUP BY ai_finding
                ORDER BY cnt DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            conn.close()

        by_severity = [
            SeverityCount(severity=row["severity"], count=int(row["cnt"]))
            for row in severity_rows
        ]

        most_reported = [
            FindingCount(ai_finding=row["ai_finding"], count=int(row["cnt"]))
            for row in finding_rows
        ]

        return AdverseEventSummary(
            total_events=total,
            by_severity=by_severity,
            most_reported_findings=most_reported,
        )

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AdverseEventRecord:
        """Convert a database row to an AdverseEventRecord."""
        return AdverseEventRecord(
            id=row["id"],
            reporter_id=row["reporter_id"],
            ecg_reference=row["ecg_reference"],
            event_description=row["event_description"],
            severity=row["severity"],
            ai_finding=row["ai_finding"],
            patient_outcome=row["patient_outcome"],
            timestamp=row["timestamp"],
        )
