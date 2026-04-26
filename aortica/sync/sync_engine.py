"""Offline-first sync engine with vector clock conflict resolution.

Provides automatic synchronisation of locally stored ECG results to a
remote server when connectivity is available.  Each device maintains a
vector clock to detect and resolve concurrent modifications.

Conflict resolution strategy: **last-writer-wins** by comparing vector
clock timestamps.  No data is ever discarded — conflicting versions are
preserved in a ``conflict_archive`` table for manual review.

Usage::

    from aortica.sync.result_store import ResultStore
    from aortica.sync.sync_engine import SyncEngine

    store = ResultStore("/path/to/data")
    engine = SyncEngine(store, device_id="device-001")
    engine.queue_for_sync(result_id=1)
    report = engine.sync_to_remote("https://central.example.com/api/v1/sync")

"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from aortica.sync.result_store import ResultStore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VectorClock:
    """Lamport-style vector clock tracking per-device logical timestamps.

    Each device increments its own counter on every write.  Comparison
    uses the partial order defined by vector clocks.
    """

    clock: dict[str, int] = field(default_factory=dict)

    def increment(self, device_id: str) -> None:
        """Increment the counter for *device_id*."""
        self.clock[device_id] = self.clock.get(device_id, 0) + 1

    def merge(self, other: VectorClock) -> VectorClock:
        """Return a new VectorClock that is the element-wise max of self and other."""
        all_devices = set(self.clock) | set(other.clock)
        merged = {
            d: max(self.clock.get(d, 0), other.clock.get(d, 0))
            for d in all_devices
        }
        return VectorClock(clock=merged)

    def dominates(self, other: VectorClock) -> bool:
        """Return ``True`` if *self* ≥ *other* on all components and > on at least one."""
        all_devices = set(self.clock) | set(other.clock)
        at_least_one_greater = False
        for d in all_devices:
            s = self.clock.get(d, 0)
            o = other.clock.get(d, 0)
            if s < o:
                return False
            if s > o:
                at_least_one_greater = True
        return at_least_one_greater

    def concurrent_with(self, other: VectorClock) -> bool:
        """Return ``True`` if neither clock dominates the other (concurrent)."""
        return not self.dominates(other) and not other.dominates(self) and self.clock != other.clock

    def to_dict(self) -> dict[str, int]:
        """Serialise to a plain dict."""
        return dict(self.clock)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        """Deserialise from a plain dict."""
        return cls(clock=dict(d))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.clock == other.clock


@dataclass
class SyncQueueEntry:
    """An item queued for synchronisation."""

    id: int
    result_id: int
    ecg_hash: str
    predictions_json: str
    quality_json: str
    metadata_json: str
    timestamp: float
    vector_clock_json: str
    status: str  # "pending", "synced", "conflict"
    created_at: float


@dataclass
class SyncReport:
    """Report returned after a sync operation."""

    uploaded: int = 0
    downloaded: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return ``True`` if no errors occurred."""
        return len(self.errors) == 0


@dataclass
class ConflictRecord:
    """A record preserved from a conflict resolution."""

    id: int
    ecg_hash: str
    local_predictions: str
    remote_predictions: str
    local_clock_json: str
    remote_clock_json: str
    resolution: str  # "local_wins", "remote_wins"
    resolved_at: float


# ---------------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------------

_CREATE_SYNC_QUEUE_SQL = """
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    ecg_hash TEXT NOT NULL,
    predictions_json TEXT NOT NULL,
    quality_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    vector_clock_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
);
"""

_CREATE_CONFLICT_ARCHIVE_SQL = """
CREATE TABLE IF NOT EXISTS conflict_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_hash TEXT NOT NULL,
    local_predictions TEXT NOT NULL,
    remote_predictions TEXT NOT NULL,
    local_clock_json TEXT NOT NULL,
    remote_clock_json TEXT NOT NULL,
    resolution TEXT NOT NULL,
    resolved_at REAL NOT NULL
);
"""

_CREATE_DEVICE_STATE_SQL = """
CREATE TABLE IF NOT EXISTS device_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# SyncEngine
# ---------------------------------------------------------------------------


class SyncEngine:
    """Offline-first sync engine with vector clock conflict resolution.

    Parameters
    ----------
    result_store:
        The :class:`ResultStore` instance to synchronise.
    device_id:
        A unique identifier for this device / node.
    """

    def __init__(
        self,
        result_store: ResultStore,
        device_id: str,
    ) -> None:
        self._store = result_store
        self._device_id = device_id

        # Reuse the ResultStore's database connection for sync tables.
        self._conn: sqlite3.Connection = result_store._conn  # noqa: SLF001

        # Create sync-specific tables
        self._conn.execute(_CREATE_SYNC_QUEUE_SQL)
        self._conn.execute(_CREATE_CONFLICT_ARCHIVE_SQL)
        self._conn.execute(_CREATE_DEVICE_STATE_SQL)
        self._conn.commit()

        # Load or initialise the device vector clock
        self._vector_clock = self._load_vector_clock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device_id(self) -> str:
        """Return the device identifier."""
        return self._device_id

    @property
    def vector_clock(self) -> VectorClock:
        """Return the current device vector clock."""
        return self._vector_clock

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def queue_for_sync(self, result_id: int) -> int:
        """Mark a result as pending upload by adding it to the sync queue.

        Parameters
        ----------
        result_id:
            The database row ID of the result to sync.

        Returns
        -------
        int
            The sync queue entry ID.

        Raises
        ------
        ValueError
            If the result_id does not exist in the store.
        """
        result = self._store.get_result_by_id(result_id)
        if result is None:
            msg = f"Result with id={result_id} not found in store"
            raise ValueError(msg)

        # Increment vector clock for this write event
        self._vector_clock.increment(self._device_id)
        self._save_vector_clock()

        # Decrypt predictions from the store and re-serialise for sync
        predictions_json = json.dumps(result.predictions)
        quality_json = json.dumps(result.quality)
        metadata_json = json.dumps(result.metadata)
        clock_json = json.dumps(self._vector_clock.to_dict())

        cursor = self._conn.execute(
            """
            INSERT INTO sync_queue
                (result_id, ecg_hash, predictions_json, quality_json,
                 metadata_json, timestamp, vector_clock_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                result_id,
                result.ecg_hash,
                predictions_json,
                quality_json,
                metadata_json,
                result.timestamp,
                clock_json,
                time.time(),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def list_pending(self) -> list[SyncQueueEntry]:
        """Return all pending sync queue entries."""
        rows = self._conn.execute(
            """
            SELECT id, result_id, ecg_hash, predictions_json, quality_json,
                   metadata_json, timestamp, vector_clock_json, status, created_at
            FROM sync_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """,
        ).fetchall()
        return [self._row_to_queue_entry(row) for row in rows]

    def pending_count(self) -> int:
        """Return the number of pending sync items."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sync_queue WHERE status = 'pending'",
        ).fetchone()
        return row[0] if row else 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_queue_entry(row: tuple[Any, ...]) -> SyncQueueEntry:
        """Convert a database row tuple to a SyncQueueEntry."""
        return SyncQueueEntry(
            id=row[0],
            result_id=row[1],
            ecg_hash=row[2],
            predictions_json=row[3],
            quality_json=row[4],
            metadata_json=row[5],
            timestamp=row[6],
            vector_clock_json=row[7],
            status=row[8],
            created_at=row[9],
        )

    # ------------------------------------------------------------------
    # Sync operations
    # ------------------------------------------------------------------

    def sync_to_remote(self, remote_url: str) -> SyncReport:
        """Upload pending results to the remote server via HTTPS POST.

        Sends each pending item as a JSON payload to
        ``{remote_url}/push``.  Marks items as ``synced`` on success.

        Parameters
        ----------
        remote_url:
            Base URL of the remote sync server (no trailing slash).

        Returns
        -------
        SyncReport
            Summary of the sync operation.
        """
        report = SyncReport()
        pending = self.list_pending()

        for entry in pending:
            payload = {
                "device_id": self._device_id,
                "ecg_hash": entry.ecg_hash,
                "predictions": json.loads(entry.predictions_json),
                "quality": json.loads(entry.quality_json),
                "metadata": json.loads(entry.metadata_json),
                "timestamp": entry.timestamp,
                "vector_clock": json.loads(entry.vector_clock_json),
            }

            try:
                self._http_post(f"{remote_url}/push", payload)
                # Mark as synced in the queue
                self._conn.execute(
                    "UPDATE sync_queue SET status = 'synced' WHERE id = ?",
                    (entry.id,),
                )
                # Also mark the result itself as synced
                self._store.mark_synced(entry.result_id)
                report.uploaded += 1
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"Failed to push result {entry.result_id}: {exc}")

        self._conn.commit()
        return report

    def pull_from_remote(self, remote_url: str) -> SyncReport:
        """Download new results from the remote server.

        Issues a POST to ``{remote_url}/pull`` with the device's current
        vector clock.  The server returns results that are newer than the
        device's clock.  Conflicts are resolved via last-writer-wins.

        Parameters
        ----------
        remote_url:
            Base URL of the remote sync server (no trailing slash).

        Returns
        -------
        SyncReport
            Summary of the pull operation.
        """
        report = SyncReport()

        pull_payload = {
            "device_id": self._device_id,
            "vector_clock": self._vector_clock.to_dict(),
        }

        try:
            response_data = self._http_post(f"{remote_url}/pull", pull_payload)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"Failed to pull from remote: {exc}")
            return report

        remote_results: list[dict[str, Any]] = response_data.get("results", [])

        for item in remote_results:
            ecg_hash: str = item["ecg_hash"]
            remote_clock = VectorClock.from_dict(item.get("vector_clock", {}))
            remote_timestamp: float = item.get("timestamp", time.time())
            remote_predictions = json.dumps(item.get("predictions", {}))
            remote_quality = json.dumps(item.get("quality", {}))
            remote_metadata = json.dumps(item.get("metadata", {}))

            # Check for existing local result
            local_result = self._store.get_result(ecg_hash)

            if local_result is None:
                # No local copy — accept the remote result
                self._store.store_result(
                    ecg_hash=ecg_hash,
                    predictions=item.get("predictions", {}),
                    quality=item.get("quality", {}),
                    metadata=item.get("metadata", {}),
                    timestamp=remote_timestamp,
                )
                report.downloaded += 1
            else:
                # Conflict resolution: compare vector clocks
                local_clock_entry = self._get_clock_for_result(local_result.id)
                local_clock = VectorClock.from_dict(local_clock_entry) if local_clock_entry else VectorClock()

                if remote_clock.dominates(local_clock):
                    # Remote is strictly newer — overwrite local
                    self._store.delete_result_by_id(local_result.id)
                    self._store.store_result(
                        ecg_hash=ecg_hash,
                        predictions=item.get("predictions", {}),
                        quality=item.get("quality", {}),
                        metadata=item.get("metadata", {}),
                        timestamp=remote_timestamp,
                    )
                    report.downloaded += 1
                elif local_clock.dominates(remote_clock):
                    # Local is strictly newer — keep local
                    pass
                else:
                    # Concurrent writes — last-writer-wins by timestamp,
                    # archive the loser
                    local_predictions = json.dumps(local_result.predictions)

                    if remote_timestamp >= local_result.timestamp:
                        resolution = "remote_wins"
                        self._store.delete_result_by_id(local_result.id)
                        self._store.store_result(
                            ecg_hash=ecg_hash,
                            predictions=item.get("predictions", {}),
                            quality=item.get("quality", {}),
                            metadata=item.get("metadata", {}),
                            timestamp=remote_timestamp,
                        )
                    else:
                        resolution = "local_wins"

                    # Archive the conflict for audit
                    self._conn.execute(
                        """
                        INSERT INTO conflict_archive
                            (ecg_hash, local_predictions, remote_predictions,
                             local_clock_json, remote_clock_json, resolution, resolved_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ecg_hash,
                            local_predictions,
                            remote_predictions,
                            json.dumps(local_clock.to_dict()),
                            json.dumps(remote_clock.to_dict()),
                            resolution,
                            time.time(),
                        ),
                    )
                    report.conflicts += 1
                    report.downloaded += 1 if resolution == "remote_wins" else 0

            # Merge the remote clock into our local clock
            self._vector_clock = self._vector_clock.merge(remote_clock)

        self._save_vector_clock()
        self._conn.commit()

        return report

    # ------------------------------------------------------------------
    # Conflict archive
    # ------------------------------------------------------------------

    def list_conflicts(self) -> list[ConflictRecord]:
        """Return all archived conflict records."""
        rows = self._conn.execute(
            """
            SELECT id, ecg_hash, local_predictions, remote_predictions,
                   local_clock_json, remote_clock_json, resolution, resolved_at
            FROM conflict_archive
            ORDER BY resolved_at DESC
            """,
        ).fetchall()
        return [
            ConflictRecord(
                id=r[0],
                ecg_hash=r[1],
                local_predictions=r[2],
                remote_predictions=r[3],
                local_clock_json=r[4],
                remote_clock_json=r[5],
                resolution=r[6],
                resolved_at=r[7],
            )
            for r in rows
        ]

    def conflict_count(self) -> int:
        """Return total number of archived conflicts."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM conflict_archive",
        ).fetchone()
        return row[0] if row else 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Vector clock persistence
    # ------------------------------------------------------------------

    def _load_vector_clock(self) -> VectorClock:
        """Load the vector clock from the device_state table."""
        row = self._conn.execute(
            "SELECT value FROM device_state WHERE key = 'vector_clock'",
        ).fetchone()
        if row is None:
            return VectorClock()
        return VectorClock.from_dict(json.loads(row[0]))

    def _save_vector_clock(self) -> None:
        """Persist the vector clock to the device_state table."""
        clock_json = json.dumps(self._vector_clock.to_dict())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO device_state (key, value)
            VALUES ('vector_clock', ?)
            """,
            (clock_json,),
        )
        self._conn.commit()

    def _get_clock_for_result(self, result_id: int) -> Optional[dict[str, int]]:
        """Get the vector clock associated with a result from the sync queue."""
        row = self._conn.execute(
            """
            SELECT vector_clock_json FROM sync_queue
            WHERE result_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and return the parsed response.

        Uses stdlib ``urllib`` to avoid external HTTP dependencies.

        Raises
        ------
        ConnectionError
            On network failures or non-2xx responses.
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code} from {url}"
            raise ConnectionError(msg) from exc
        except urllib.error.URLError as exc:
            msg = f"Connection failed to {url}: {exc.reason}"
            raise ConnectionError(msg) from exc
