"""SQLite local result storage with AES-256 encryption.

Provides persistent, encrypted storage for ECG analysis results on edge
devices.  Patient data is protected at rest using Fernet (AES-128-CBC via
the ``cryptography`` library's Fernet implementation, which wraps AES-256
key material into a URL-safe base64 token).

Usage::

    from aortica.sync.result_store import ResultStore

    store = ResultStore("/path/to/data")
    store.store_result("abc123", {"rhythm": [0.1, 0.9]}, {"overall": 85})
    result = store.get_result("abc123")
    store.close()
"""

from __future__ import annotations

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
    """Raise if cryptography is not installed."""
    if not HAS_CRYPTOGRAPHY:
        raise ImportError(
            "The 'cryptography' package is required for encrypted result storage. "
            "Install it with: pip install aortica[sync]"
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StoredResult:
    """A single result record retrieved from the store."""

    id: int
    ecg_hash: str
    predictions: dict[str, Any]
    quality: dict[str, Any]
    timestamp: float
    synced: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_hash TEXT NOT NULL,
    predictions_json BLOB NOT NULL,
    quality_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ecg_hash ON results (ecg_hash);
"""


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------


class ResultStore:
    """SQLite-backed local result store with optional AES-256 encryption.

    Parameters
    ----------
    db_dir:
        Directory where the SQLite database file will be created / opened.
        The directory is created automatically if it does not exist.
    encryption_key:
        A Fernet-compatible key (URL-safe base64-encoded 32-byte key).
        If ``None``, a new key is generated automatically and persisted
        to ``<db_dir>/result_store.key``.
    db_filename:
        Name of the SQLite database file.  Defaults to ``results.db``.
    """

    def __init__(
        self,
        db_dir: str | Path,
        encryption_key: Optional[str | bytes] = None,
        db_filename: str = "results.db",
    ) -> None:
        _check_cryptography()

        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._db_dir / db_filename
        self._key_path = self._db_dir / "result_store.key"

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
            # Restrict key file permissions (owner-read/write only)
            os.chmod(self._key_path, 0o600)

        self._fernet = Fernet(key_bytes)  # type: ignore[arg-type]

        # Open / create SQLite database
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, plaintext: str) -> bytes:
        """Encrypt a JSON string and return ciphertext bytes."""
        return self._fernet.encrypt(plaintext.encode("utf-8"))  # type: ignore[union-attr]

    def _decrypt(self, ciphertext: bytes) -> str:
        """Decrypt ciphertext bytes back to a JSON string."""
        return self._fernet.decrypt(ciphertext).decode("utf-8")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store_result(
        self,
        ecg_hash: str,
        predictions: dict[str, Any],
        quality: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Store an ECG analysis result.

        Parameters
        ----------
        ecg_hash:
            A hash or unique identifier for the source ECG recording.
        predictions:
            Multi-task prediction results as a JSON-serialisable dict.
        quality:
            Signal quality report as a JSON-serialisable dict.
        metadata:
            Optional additional metadata (patient info, device, etc.).
        timestamp:
            Unix timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        int
            The database row ID of the newly inserted result.
        """
        ts = timestamp if timestamp is not None else time.time()
        predictions_json = json.dumps(predictions)
        quality_json = json.dumps(quality)
        metadata_json = json.dumps(metadata or {})

        # Encrypt the predictions column (contains clinical data)
        encrypted_predictions = self._encrypt(predictions_json)

        cursor = self._conn.execute(
            """
            INSERT INTO results (ecg_hash, predictions_json, quality_json, metadata_json, timestamp, synced)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (ecg_hash, encrypted_predictions, quality_json, metadata_json, ts),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_result(self, ecg_hash: str) -> Optional[StoredResult]:
        """Retrieve the most recent result for a given ECG hash.

        Returns ``None`` if no result is found.
        """
        row = self._conn.execute(
            """
            SELECT id, ecg_hash, predictions_json, quality_json, metadata_json, timestamp, synced
            FROM results
            WHERE ecg_hash = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (ecg_hash,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_result(row)

    def get_result_by_id(self, result_id: int) -> Optional[StoredResult]:
        """Retrieve a result by its database row ID."""
        row = self._conn.execute(
            """
            SELECT id, ecg_hash, predictions_json, quality_json, metadata_json, timestamp, synced
            FROM results
            WHERE id = ?
            """,
            (result_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_result(row)

    def list_results(
        self,
        limit: int = 100,
        offset: int = 0,
        synced: Optional[bool] = None,
    ) -> list[StoredResult]:
        """List stored results with optional pagination and sync filter.

        Parameters
        ----------
        limit:
            Maximum number of results to return.
        offset:
            Number of results to skip (for pagination).
        synced:
            If not ``None``, filter by sync status.

        Returns
        -------
        list[StoredResult]
            Results ordered by timestamp (newest first).
        """
        if synced is not None:
            rows = self._conn.execute(
                """
                SELECT id, ecg_hash, predictions_json, quality_json, metadata_json, timestamp, synced
                FROM results
                WHERE synced = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (1 if synced else 0, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, ecg_hash, predictions_json, quality_json, metadata_json, timestamp, synced
                FROM results
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [self._row_to_result(row) for row in rows]

    def delete_result(self, ecg_hash: str) -> int:
        """Delete all results for a given ECG hash.

        Returns the number of rows deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM results WHERE ecg_hash = ?",
            (ecg_hash,),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_result_by_id(self, result_id: int) -> bool:
        """Delete a single result by its database row ID.

        Returns ``True`` if a row was deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM results WHERE id = ?",
            (result_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def mark_synced(self, result_id: int) -> bool:
        """Mark a result as synced.

        Returns ``True`` if the row was updated.
        """
        cursor = self._conn.execute(
            "UPDATE results SET synced = 1 WHERE id = ?",
            (result_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, synced: Optional[bool] = None) -> int:
        """Return total result count, optionally filtered by sync status."""
        if synced is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM results WHERE synced = ?",
                (1 if synced else 0,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM results").fetchone()
        return row[0] if row else 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_result(self, row: tuple[Any, ...]) -> StoredResult:
        """Convert a database row tuple to a StoredResult."""
        row_id, ecg_hash, predictions_blob, quality_json, metadata_json, ts, synced = row

        # Decrypt predictions
        predictions_str = self._decrypt(predictions_blob)

        return StoredResult(
            id=row_id,
            ecg_hash=ecg_hash,
            predictions=json.loads(predictions_str),
            quality=json.loads(quality_json),
            timestamp=ts,
            synced=bool(synced),
            metadata=json.loads(metadata_json),
        )

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
