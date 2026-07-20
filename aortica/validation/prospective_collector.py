"""Prospective data collection pipeline for multi-site validation studies.

Manages ECG ingestion, AI prediction storage, ground-truth outcome entry,
and outcome linkage (prediction ↔ ground truth pairing) for prospective
validation of the Aortica model.

Uses encrypted SQLite storage (reuses encryption pattern from
:class:`~aortica.sync.result_store.ResultStore`).

Usage::

    from aortica.validation import ProspectiveCollector, export_study_data

    collector = ProspectiveCollector("/path/to/study_data")
    record_id = collector.ingest_ecg(
        ecg_hash="abc123",
        site_id="site_A",
        predictions={"rhythm": [0.1, 0.9]},
        quality={"overall": 85},
    )
    collector.add_outcome(
        record_id=record_id,
        ground_truth={"AF": 1, "STEMI": 0},
        clinician_id="dr_smith",
    )
    export_study_data(collector, "/output/study_data.csv")
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional dependency: cryptography
# ---------------------------------------------------------------------------
try:
    from cryptography.fernet import Fernet  # type: ignore[import-untyped]

    HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    HAS_CRYPTOGRAPHY = False


def _check_cryptography() -> None:
    if not HAS_CRYPTOGRAPHY:
        raise ImportError(
            "The 'cryptography' package is required for ProspectiveCollector. "
            "Install with: pip install aortica[sync]"
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StudyRecord:
    """A single prospective study record.

    Attributes:
        id: Database row ID.
        ecg_hash: Unique identifier for the source ECG.
        site_id: Identifier for the contributing site.
        predictions: AI multi-task predictions (decrypted).
        quality: Signal quality report.
        ground_truth: Clinician-verified ground-truth labels (if linked).
        clinician_id: ID of the clinician who provided ground truth.
        outcome: Follow-up outcome data (if available).
        timestamp: Unix timestamp of ECG ingestion.
        outcome_timestamp: Unix timestamp of outcome entry.
        metadata: Additional metadata (patient demographics, device info).
    """

    id: int = 0
    ecg_hash: str = ""
    site_id: str = ""
    predictions: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    clinician_id: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    outcome_timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prospective_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_hash TEXT NOT NULL,
    site_id TEXT NOT NULL,
    predictions_json BLOB NOT NULL,
    quality_json TEXT NOT NULL,
    ground_truth_json TEXT NOT NULL DEFAULT '{}',
    clinician_id TEXT NOT NULL DEFAULT '',
    outcome_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    outcome_timestamp REAL NOT NULL DEFAULT 0.0
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prospective_ecg_hash
ON prospective_records (ecg_hash);
"""

_CREATE_SITE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prospective_site_id
ON prospective_records (site_id);
"""


# ---------------------------------------------------------------------------
# ProspectiveCollector
# ---------------------------------------------------------------------------


class ProspectiveCollector:
    """SQLite-backed prospective data collection pipeline.

    Manages the lifecycle of ECG records in a prospective validation
    study: ingestion → AI prediction → ground-truth linkage → outcome
    tracking.

    Parameters
    ----------
    db_dir:
        Directory for the SQLite database and encryption key.
    encryption_key:
        Optional Fernet-compatible key.  If ``None``, a new key is
        generated and persisted to ``<db_dir>/prospective.key``.
    db_filename:
        Name of the SQLite file.  Defaults to ``prospective.db``.
    """

    def __init__(
        self,
        db_dir: str | Path,
        encryption_key: Optional[str | bytes] = None,
        db_filename: str = "prospective.db",
    ) -> None:
        _check_cryptography()

        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._db_dir / db_filename
        self._key_path = self._db_dir / "prospective.key"

        # Resolve encryption key
        if encryption_key is not None:
            key_bytes = (
                encryption_key
                if isinstance(encryption_key, bytes)
                else encryption_key.encode("utf-8")
            )
        elif self._key_path.exists():
            key_bytes = self._key_path.read_bytes().strip()
        else:
            key_bytes = Fernet.generate_key()  # type: ignore[union-attr]
            self._key_path.write_bytes(key_bytes)
            os.chmod(self._key_path, 0o600)

        self._fernet = Fernet(key_bytes)  # type: ignore[arg-type]

        # Open / create database
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.execute(_CREATE_SITE_INDEX_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode("utf-8"))  # type: ignore[union-attr]

    def _decrypt(self, ciphertext: bytes) -> str:
        return self._fernet.decrypt(ciphertext).decode("utf-8")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_ecg(
        self,
        ecg_hash: str,
        site_id: str,
        predictions: dict[str, Any],
        quality: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Ingest an ECG with AI predictions.

        Parameters
        ----------
        ecg_hash:
            Unique identifier for the ECG recording.
        site_id:
            Identifier for the contributing site.
        predictions:
            AI multi-task prediction results.
        quality:
            Signal quality report.
        metadata:
            Optional demographics, device info, etc.
        timestamp:
            Unix timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        int
            The database row ID of the new record.
        """
        ts = timestamp if timestamp is not None else time.time()
        encrypted_predictions = self._encrypt(json.dumps(predictions))

        cursor = self._conn.execute(
            """
            INSERT INTO prospective_records
                (ecg_hash, site_id, predictions_json, quality_json,
                 metadata_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ecg_hash,
                site_id,
                encrypted_predictions,
                json.dumps(quality),
                json.dumps(metadata or {}),
                ts,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Ground truth / outcome linkage
    # ------------------------------------------------------------------

    def add_outcome(
        self,
        record_id: int,
        ground_truth: dict[str, Any],
        clinician_id: str = "",
        outcome: Optional[dict[str, Any]] = None,
        outcome_timestamp: Optional[float] = None,
    ) -> bool:
        """Link ground-truth labels and/or outcome data to a record.

        Parameters
        ----------
        record_id:
            Database row ID from :meth:`ingest_ecg`.
        ground_truth:
            Clinician-verified diagnosis labels.
        clinician_id:
            ID of the clinician providing the ground truth.
        outcome:
            Follow-up outcome data (e.g. 30-day MACE, 12-month echo).
        outcome_timestamp:
            Unix timestamp of outcome entry.

        Returns
        -------
        bool
            ``True`` if the record was updated.
        """
        ots = outcome_timestamp if outcome_timestamp is not None else time.time()
        cursor = self._conn.execute(
            """
            UPDATE prospective_records
            SET ground_truth_json = ?,
                clinician_id = ?,
                outcome_json = ?,
                outcome_timestamp = ?
            WHERE id = ?
            """,
            (
                json.dumps(ground_truth),
                clinician_id,
                json.dumps(outcome or {}),
                ots,
                record_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_record(self, record_id: int) -> Optional[StudyRecord]:
        """Retrieve a single record by ID."""
        row = self._conn.execute(
            """
            SELECT id, ecg_hash, site_id, predictions_json, quality_json,
                   ground_truth_json, clinician_id, outcome_json,
                   metadata_json, timestamp, outcome_timestamp
            FROM prospective_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_record(row)

    def get_record_by_hash(self, ecg_hash: str) -> Optional[StudyRecord]:
        """Retrieve the most recent record for an ECG hash."""
        row = self._conn.execute(
            """
            SELECT id, ecg_hash, site_id, predictions_json, quality_json,
                   ground_truth_json, clinician_id, outcome_json,
                   metadata_json, timestamp, outcome_timestamp
            FROM prospective_records
            WHERE ecg_hash = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (ecg_hash,),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_record(row)

    def list_records(
        self,
        site_id: Optional[str] = None,
        linked_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StudyRecord]:
        """List records with optional filters.

        Parameters
        ----------
        site_id:
            Filter by contributing site.
        linked_only:
            If ``True``, only return records with ground-truth labels.
        limit:
            Maximum records to return.
        offset:
            Pagination offset.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if site_id is not None:
            conditions.append("site_id = ?")
            params.append(site_id)
        if linked_only:
            conditions.append("ground_truth_json != '{}'")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"""
            SELECT id, ecg_hash, site_id, predictions_json, quality_json,
                   ground_truth_json, clinician_id, outcome_json,
                   metadata_json, timestamp, outcome_timestamp
            FROM prospective_records
            {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def count(
        self,
        site_id: Optional[str] = None,
        linked_only: bool = False,
    ) -> int:
        """Count records with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if site_id is not None:
            conditions.append("site_id = ?")
            params.append(site_id)
        if linked_only:
            conditions.append("ground_truth_json != '{}'")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        row = self._conn.execute(
            f"SELECT COUNT(*) FROM prospective_records {where}",
            params,
        ).fetchone()
        return row[0] if row else 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _row_to_record(self, row: tuple[Any, ...]) -> StudyRecord:
        (
            row_id, ecg_hash, site_id, predictions_blob, quality_json,
            ground_truth_json, clinician_id, outcome_json,
            metadata_json, ts, ots,
        ) = row

        predictions_str = self._decrypt(predictions_blob)

        return StudyRecord(
            id=row_id,
            ecg_hash=ecg_hash,
            site_id=site_id,
            predictions=json.loads(predictions_str),
            quality=json.loads(quality_json),
            ground_truth=json.loads(ground_truth_json),
            clinician_id=clinician_id,
            outcome=json.loads(outcome_json),
            metadata=json.loads(metadata_json),
            timestamp=ts,
            outcome_timestamp=ots,
        )

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "ProspectiveCollector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_study_data(
    collector: ProspectiveCollector,
    output_path: str,
    *,
    site_id: Optional[str] = None,
    linked_only: bool = True,
) -> int:
    """Export de-identified study data to CSV.

    Generates a CSV file suitable for statistical analysis with one
    row per ECG record.  Patient-identifying metadata (names, MRNs)
    is **not** exported — only age, sex, site_id, and clinical data.

    Parameters
    ----------
    collector:
        A :class:`ProspectiveCollector` with study data.
    output_path:
        Path to write the CSV file.
    site_id:
        Optional filter by site.
    linked_only:
        If ``True`` (default), only export records with ground-truth
        labels linked.

    Returns
    -------
    int
        Number of records exported.
    """
    records = collector.list_records(
        site_id=site_id,
        linked_only=linked_only,
        limit=100_000,  # high limit for export
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "record_id",
            "ecg_hash",
            "site_id",
            "age",
            "sex",
            "quality_overall",
            "predictions_json",
            "ground_truth_json",
            "clinician_id",
            "outcome_json",
            "timestamp",
            "outcome_timestamp",
        ])

        for rec in records:
            age = rec.metadata.get("age", "")
            sex = rec.metadata.get("sex", "")
            quality_overall = rec.quality.get("overall", "")

            writer.writerow([
                rec.id,
                rec.ecg_hash,
                rec.site_id,
                age,
                sex,
                quality_overall,
                json.dumps(rec.predictions),
                json.dumps(rec.ground_truth),
                rec.clinician_id,
                json.dumps(rec.outcome),
                rec.timestamp,
                rec.outcome_timestamp,
            ])

    return len(records)
