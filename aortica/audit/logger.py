"""Tamper-evident audit trail (US-121).

Records clinical decision-support interactions to an append-only SQLite
table where each row carries an HMAC that chains to the previous row's
HMAC.  Any modification to a past row invalidates every subsequent HMAC,
making tampering detectable by :func:`verify_integrity`.

The logger is thread-safe (a lock guards every write) so it can back a
threaded FastAPI server.

Example::

    from aortica.audit import AuditLogger

    audit = AuditLogger("/var/lib/aortica/audit.db")
    audit.log("prediction_generated", ecg_reference_id="ecg_1",
              model_version="0.2.0", user_id="dr_a")
    report = audit.verify()
    assert report.valid
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Canonical, ordered set of auditable event types.
EVENT_TYPES = (
    "ecg_ingested",
    "prediction_generated",
    "xai_computed",
    "finding_accepted",
    "finding_rejected",
    "finding_modified",
    "report_generated",
    "report_exported",
    "ehr_submitted",
    "adverse_event_reported",
    "model_loaded",
    "model_updated",
)

_GENESIS_HASH = "0" * 64
_ENV_KEY = "AORTICA_AUDIT_HMAC_KEY"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditEvent:
    """A single audit-trail entry."""

    id: int
    timestamp: str
    event_type: str
    user_id: Optional[str]
    ecg_reference_id: Optional[str]
    model_version: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    event_details: Dict[str, Any]
    prev_hash: str
    hmac: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "ecg_reference_id": self.ecg_reference_id,
            "model_version": self.model_version,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "event_details": self.event_details,
            "prev_hash": self.prev_hash,
            "hmac": self.hmac,
        }


@dataclass
class IntegrityReport:
    """Result of an HMAC-chain verification."""

    valid: bool
    total_rows: int
    broken_links: List[int] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "total_rows": self.total_rows,
            "broken_links": list(self.broken_links),
            "detail": self.detail,
        }


def _resolve_key(explicit: Optional[str]) -> bytes:
    """Resolve the HMAC key from an explicit value or the environment."""
    key = explicit or os.environ.get(_ENV_KEY) or "aortica-default-audit-key"
    return key.encode("utf-8")


def _row_payload(
    timestamp: str,
    event_type: str,
    user_id: Optional[str],
    ecg_reference_id: Optional[str],
    model_version: Optional[str],
    session_id: Optional[str],
    ip_address: Optional[str],
    event_details_json: str,
    prev_hash: str,
) -> str:
    """Build the canonical string that a row's HMAC signs."""
    return "\x1f".join(
        [
            timestamp,
            event_type,
            user_id or "",
            ecg_reference_id or "",
            model_version or "",
            session_id or "",
            ip_address or "",
            event_details_json,
            prev_hash,
        ]
    )


def _compute_hmac(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


class AuditLogger:
    """Append-only, HMAC-chained audit logger backed by SQLite."""

    def __init__(self, db_path: str, *, hmac_key: Optional[str] = None) -> None:
        self._db_path = db_path
        self._key = _resolve_key(hmac_key)
        self._lock = threading.RLock()
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                ecg_reference_id TEXT,
                model_version TEXT,
                session_id TEXT,
                ip_address TEXT,
                event_details TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hmac TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- Writing ------------------------------------------------------------

    def log(
        self,
        event_type: str,
        *,
        user_id: Optional[str] = None,
        ecg_reference_id: Optional[str] = None,
        model_version: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        event_details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> AuditEvent:
        """Append an event to the audit trail and return it.

        Raises:
            ValueError: If *event_type* is not a known audit event type.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event_type!r}. Valid: {EVENT_TYPES}"
            )
        ts = timestamp or _now_iso()
        details_json = json.dumps(event_details or {}, sort_keys=True)

        with self._lock:
            prev_hash = self._last_hmac()
            payload = _row_payload(
                ts, event_type, user_id, ecg_reference_id, model_version,
                session_id, ip_address, details_json, prev_hash,
            )
            row_hmac = _compute_hmac(self._key, payload)
            cur = self._conn.execute(
                """
                INSERT INTO audit_events (
                    timestamp, event_type, user_id, ecg_reference_id,
                    model_version, session_id, ip_address, event_details,
                    prev_hash, hmac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, event_type, user_id, ecg_reference_id, model_version,
                    session_id, ip_address, details_json, prev_hash, row_hmac,
                ),
            )
            self._conn.commit()
            row_id = int(cur.lastrowid or 0)

        return AuditEvent(
            id=row_id,
            timestamp=ts,
            event_type=event_type,
            user_id=user_id,
            ecg_reference_id=ecg_reference_id,
            model_version=model_version,
            session_id=session_id,
            ip_address=ip_address,
            event_details=event_details or {},
            prev_hash=prev_hash,
            hmac=row_hmac,
        )

    def _last_hmac(self) -> str:
        row = self._conn.execute(
            "SELECT hmac FROM audit_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else _GENESIS_HASH

    # -- Reading ------------------------------------------------------------

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        ecg_reference_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Return audit events matching the given filters (newest first)."""
        clauses: List[str] = []
        params: List[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if ecg_reference_id:
            clauses.append("ecg_reference_id = ?")
            params.append(ecg_reference_id)
        if date_from:
            clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("timestamp <= ?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_events {where} "
                f"ORDER BY id DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM audit_events"
                ).fetchone()[0]
            )

    @staticmethod
    def _row_to_event(row: Any) -> AuditEvent:
        return AuditEvent(
            id=row[0],
            timestamp=row[1],
            event_type=row[2],
            user_id=row[3],
            ecg_reference_id=row[4],
            model_version=row[5],
            session_id=row[6],
            ip_address=row[7],
            event_details=json.loads(row[8]) if row[8] else {},
            prev_hash=row[9],
            hmac=row[10],
        )

    # -- Verification -------------------------------------------------------

    def verify(self) -> IntegrityReport:
        """Verify the HMAC chain of this logger's database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_events ORDER BY id ASC"
            ).fetchall()
        return _verify_rows(rows, self._key)

    # -- Rotation -----------------------------------------------------------

    def rotate_if_needed(self, max_bytes: int) -> Optional[str]:
        """Archive & truncate the log if the DB file exceeds *max_bytes*.

        Archives every current row to a gzip-compressed JSON file next to
        the database, then clears the table and restarts the HMAC chain.
        Returns the archive path if rotation happened, else ``None``.
        """
        try:
            size = os.path.getsize(self._db_path)
        except OSError:
            return None
        if size <= max_bytes:
            return None

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_events ORDER BY id ASC"
            ).fetchall()
            if not rows:
                return None
            events = [self._row_to_event(r).to_dict() for r in rows]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            archive_path = f"{self._db_path}.{stamp}.json.gz"
            with gzip.open(archive_path, "wt", encoding="utf-8") as fh:
                json.dump(events, fh)
            self._conn.execute("DELETE FROM audit_events")
            self._conn.execute(
                "DELETE FROM sqlite_sequence WHERE name = 'audit_events'"
            )
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._conn.commit()
        return archive_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------


def _verify_rows(rows: List[Any], key: bytes) -> IntegrityReport:
    # Chain on the RECOMPUTED hmac (not the stored one) so that tampering
    # with any row cascades: every subsequent row's expected prev_hash no
    # longer matches what was stored, invalidating the rest of the chain.
    prev_expected = _GENESIS_HASH
    broken: List[int] = []
    for row in rows:
        (
            row_id, ts, event_type, user_id, ecg_ref, model_version,
            session_id, ip_address, details_json, stored_prev, stored_hmac,
        ) = row
        payload = _row_payload(
            ts, event_type, user_id, ecg_ref, model_version, session_id,
            ip_address, details_json, prev_expected,
        )
        expected = _compute_hmac(key, payload)
        if stored_prev != prev_expected or stored_hmac != expected:
            broken.append(row_id)
        prev_expected = expected
    valid = not broken
    return IntegrityReport(
        valid=valid,
        total_rows=len(rows),
        broken_links=broken,
        detail="chain intact" if valid else f"{len(broken)} broken link(s)",
    )


def verify_integrity(
    audit_db_path: str, *, hmac_key: Optional[str] = None
) -> IntegrityReport:
    """Validate the HMAC chain of an audit database at *audit_db_path*."""
    key = _resolve_key(hmac_key)
    conn = sqlite3.connect(audit_db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM audit_events ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    return _verify_rows(rows, key)
