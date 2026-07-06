"""Edge site monitoring for LMIC pilot deployments (US-061b).

Provides :class:`SiteMonitor`, a lightweight, dependency-free monitor for a
deployed edge site. It records inference events (successes and errors) in a
small SQLite log, reads storage utilisation from the filesystem, and — when
wired to a sync engine — reports sync status and the last successful sync time.

The monitor backs both the ``GET /edge/status`` API endpoint and the
``aortica edge site-report`` CLI command used for remote pilot monitoring.

Design notes:
    * Uses the standard-library ``sqlite3`` only (no cryptography / onnxruntime
      dependency) so it runs on the leanest edge image and in unit tests.
    * Sync integration is duck-typed: any object exposing ``pending_count()``
      can be passed as ``sync_engine``; the last-sync timestamp is recorded via
      :meth:`record_sync`.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

#: Seconds in a rolling 24-hour window.
_DAY_SECONDS: float = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------


@dataclass
class SiteStatus:
    """Point-in-time status of a deployed edge site.

    Attributes:
        site_id: Human-readable site identifier.
        timestamp: Unix time the status was generated.
        daily_inference_count: Successful + failed inferences in the last 24 h.
        daily_error_count: Failed inferences in the last 24 h.
        total_inference_count: All inferences ever recorded.
        error_rate: ``daily_error_count / daily_inference_count`` (0 if none).
        sync_status: ``"synced"``, ``"pending"``, or ``"unknown"``.
        pending_sync_count: Number of results awaiting upload.
        last_sync_timestamp: Unix time of the last successful sync, if any.
        storage_total_bytes: Total bytes on the data volume.
        storage_used_bytes: Used bytes on the data volume.
        storage_free_bytes: Free bytes on the data volume.
        storage_utilization_pct: Percentage of the data volume in use.
    """

    site_id: str
    timestamp: float
    daily_inference_count: int
    daily_error_count: int
    total_inference_count: int
    error_rate: float
    sync_status: str
    pending_sync_count: int
    last_sync_timestamp: Optional[float]
    storage_total_bytes: int
    storage_used_bytes: int
    storage_free_bytes: int
    storage_utilization_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the status to a JSON-friendly dictionary."""
        return {
            "site_id": self.site_id,
            "timestamp": self.timestamp,
            "daily_inference_count": self.daily_inference_count,
            "daily_error_count": self.daily_error_count,
            "total_inference_count": self.total_inference_count,
            "error_rate": self.error_rate,
            "sync_status": self.sync_status,
            "pending_sync_count": self.pending_sync_count,
            "last_sync_timestamp": self.last_sync_timestamp,
            "storage_total_bytes": self.storage_total_bytes,
            "storage_used_bytes": self.storage_used_bytes,
            "storage_free_bytes": self.storage_free_bytes,
            "storage_utilization_pct": self.storage_utilization_pct,
        }

    def summary(self) -> str:
        """Return a human-readable one-block summary."""
        last_sync = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.last_sync_timestamp))
            if self.last_sync_timestamp is not None
            else "never"
        )
        return "\n".join(
            [
                f"Site Status — {self.site_id}",
                "=" * 40,
                f"  Inferences (24h):  {self.daily_inference_count}",
                f"  Errors (24h):      {self.daily_error_count} "
                f"({self.error_rate * 100:.1f}%)",
                f"  Inferences (total):{self.total_inference_count}",
                f"  Sync status:       {self.sync_status} "
                f"({self.pending_sync_count} pending)",
                f"  Last sync:         {last_sync}",
                f"  Storage used:      {self.storage_utilization_pct:.1f}% "
                f"({self.storage_used_bytes / 1e9:.2f}/"
                f"{self.storage_total_bytes / 1e9:.2f} GB)",
            ]
        )


# ---------------------------------------------------------------------------
# SiteMonitor
# ---------------------------------------------------------------------------

_CREATE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS site_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    success INTEGER NOT NULL,
    error_type TEXT
);
"""

_CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS site_meta (
    key TEXT PRIMARY KEY,
    value REAL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_events_ts ON site_events (timestamp);
"""


class SiteMonitor:
    """Records edge-site activity and reports operational status.

    Args:
        data_dir: Directory holding the site's data (also the volume whose
            storage utilisation is reported). Created if missing.
        site_id: Identifier for this site. Default ``"aortica-edge"``.
        sync_engine: Optional object exposing ``pending_count() -> int`` (e.g.
            :class:`aortica.sync.sync_engine.SyncEngine`) for sync status.
        db_filename: Name of the monitor's SQLite log. Default
            ``"site_monitor.db"``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        site_id: str = "aortica-edge",
        sync_engine: Optional[Any] = None,
        db_filename: str = "site_monitor.db",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.site_id = site_id
        self._sync_engine = sync_engine
        self._db_path = self._data_dir / db_filename
        # check_same_thread=False so the monitor can back a threaded web server;
        # all access is serialised through self._lock.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_CREATE_EVENTS_SQL)
            self._conn.execute(_CREATE_META_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)
            self._conn.commit()

    # ---- Recording ------------------------------------------------------

    def record_inference(
        self,
        success: bool = True,
        error_type: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a single inference event.

        Args:
            success: Whether the inference completed successfully.
            error_type: Optional short error category when ``success`` is False.
            timestamp: Event time (Unix). Defaults to now.
        """
        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            self._conn.execute(
                "INSERT INTO site_events (timestamp, success, error_type) "
                "VALUES (?, ?, ?)",
                (ts, 1 if success else 0, error_type),
            )
            self._conn.commit()

    def record_sync(
        self,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record the time of a successful sync to the central server."""
        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            self._conn.execute(
                "INSERT INTO site_meta (key, value) VALUES ('last_sync', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (ts,),
            )
            self._conn.commit()

    # ---- Queries --------------------------------------------------------

    def _count_since(self, since: float, success: Optional[bool] = None) -> int:
        sql = "SELECT COUNT(*) FROM site_events WHERE timestamp >= ?"
        params: list[Any] = [since]
        if success is not None:
            sql += " AND success = ?"
            params.append(1 if success else 0)
        with self._lock:
            cur = self._conn.execute(sql, params)
            return int(cur.fetchone()[0])

    def daily_inference_count(self, now: Optional[float] = None) -> int:
        """Total inferences (success + error) in the last 24 hours."""
        now = time.time() if now is None else now
        return self._count_since(now - _DAY_SECONDS)

    def daily_error_count(self, now: Optional[float] = None) -> int:
        """Failed inferences in the last 24 hours."""
        now = time.time() if now is None else now
        return self._count_since(now - _DAY_SECONDS, success=False)

    def total_inference_count(self) -> int:
        """All inferences ever recorded."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM site_events")
            return int(cur.fetchone()[0])

    def error_rate(self, now: Optional[float] = None) -> float:
        """Daily error rate in ``[0, 1]`` (0 when there are no inferences)."""
        total = self.daily_inference_count(now)
        if total == 0:
            return 0.0
        return self.daily_error_count(now) / total

    def last_sync_timestamp(self) -> Optional[float]:
        """Unix time of the last recorded successful sync, or ``None``."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT value FROM site_meta WHERE key = 'last_sync'"
            )
            row = cur.fetchone()
        return float(row[0]) if row is not None else None

    def sync_status(self) -> tuple[str, int]:
        """Return ``(status, pending_count)``.

        Status is ``"unknown"`` when no sync engine is wired, ``"synced"`` when
        there are no pending items, and ``"pending"`` otherwise.
        """
        if self._sync_engine is None:
            return "unknown", 0
        pending = int(self._sync_engine.pending_count())
        return ("synced" if pending == 0 else "pending"), pending

    def storage_utilization(self) -> tuple[int, int, int, float]:
        """Return ``(total, used, free, used_pct)`` bytes for the data volume."""
        usage = shutil.disk_usage(str(self._data_dir))
        used_pct = (usage.used / usage.total * 100.0) if usage.total > 0 else 0.0
        return usage.total, usage.used, usage.free, used_pct

    def status(self, now: Optional[float] = None) -> SiteStatus:
        """Assemble the full :class:`SiteStatus` snapshot."""
        now = time.time() if now is None else now
        sync_state, pending = self.sync_status()
        total_b, used_b, free_b, used_pct = self.storage_utilization()
        return SiteStatus(
            site_id=self.site_id,
            timestamp=now,
            daily_inference_count=self.daily_inference_count(now),
            daily_error_count=self.daily_error_count(now),
            total_inference_count=self.total_inference_count(),
            error_rate=self.error_rate(now),
            sync_status=sync_state,
            pending_sync_count=pending,
            last_sync_timestamp=self.last_sync_timestamp(),
            storage_total_bytes=total_b,
            storage_used_bytes=used_b,
            storage_free_bytes=free_b,
            storage_utilization_pct=used_pct,
        )

    def daily_report(self, now: Optional[float] = None) -> dict[str, Any]:
        """Build a daily site-activity summary for remote monitoring.

        Returns a JSON-friendly dict combining the current status with a
        generation timestamp, suitable for the ``aortica edge site-report``
        command.
        """
        now = time.time() if now is None else now
        status = self.status(now)
        report = status.to_dict()
        report["report_generated_at"] = now
        report["report_date"] = time.strftime("%Y-%m-%d", time.gmtime(now))
        return report

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SiteMonitor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
