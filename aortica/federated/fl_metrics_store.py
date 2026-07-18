"""SQLite-backed persistence for federated learning campaign metrics.

Provides :class:`FLMetricsStore` which stores per-round aggregated
metrics, per-site participation data, privacy budget tracking, and
convergence indicators — all consumed by the FL monitoring dashboard
(US-113) via the ``/api/v1/federated/`` endpoints.

The store is designed for single-writer (FL server) + multi-reader
(dashboard API) access patterns.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CampaignStatus:
    """Snapshot of the current FL campaign state.

    Attributes:
        campaign_name: Human-readable campaign identifier.
        current_round: Latest completed round number (0 if not started).
        total_rounds: Total rounds configured for this campaign.
        strategy: Aggregation strategy name (fedavg/fedprox/scaffold).
        start_timestamp: Unix epoch when the campaign started.
        elapsed_seconds: Wall-clock seconds since campaign start.
        status: Campaign lifecycle status (running/completed/failed).
    """

    campaign_name: str = "default"
    current_round: int = 0
    total_rounds: int = 0
    strategy: str = "fedavg"
    start_timestamp: float = 0.0
    elapsed_seconds: float = 0.0
    status: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundRecord:
    """Persisted record for a single FL round.

    Attributes:
        round_number: 1-indexed round number.
        loss: Aggregated loss for the round.
        metrics: Per-task metrics dict (e.g. ``{"rhythm_f1": 0.87}``).
        num_clients: Number of participating clients.
        gradient_norm: Average gradient L2 norm across clients.
        timestamp: Unix epoch when the round completed.
    """

    round_number: int
    loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    num_clients: int = 0
    gradient_norm: Optional[float] = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiteRecord:
    """Per-site participation record.

    Attributes:
        site_id: Anonymised site identifier.
        status: Connection status (online/offline).
        samples_contributed: Total training samples contributed.
        last_communication: Unix epoch of last communication.
        local_training_time_ms: Latest local training time in ms.
        epsilon_spent: Cumulative privacy budget consumed.
    """

    site_id: str
    status: str = "offline"
    samples_contributed: int = 0
    last_communication: float = 0.0
    local_training_time_ms: float = 0.0
    epsilon_spent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConvergenceIndicators:
    """Convergence health indicators computed from round history.

    Attributes:
        gradient_norms: Per-round average gradient norms.
        loss_trend: Per-round loss values for plateau detection.
        plateau_detected: Whether loss has plateaued.
        early_stop_recommended: Whether early stopping is recommended.
        plateau_window: Number of rounds used for plateau detection.
    """

    gradient_norms: List[float] = field(default_factory=list)
    loss_trend: List[Optional[float]] = field(default_factory=list)
    plateau_detected: bool = False
    early_stop_recommended: bool = False
    plateau_window: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# FLMetricsStore
# ---------------------------------------------------------------------------


class FLMetricsStore:
    """SQLite-backed store for FL campaign metrics.

    Thread-safe for concurrent reads from the dashboard API while the
    FL server writes.  Uses ``check_same_thread=False`` with a
    threading lock around all writes.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS campaign (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        name        TEXT    NOT NULL DEFAULT 'default',
        total_rounds INTEGER NOT NULL DEFAULT 0,
        strategy    TEXT    NOT NULL DEFAULT 'fedavg',
        start_ts    REAL    NOT NULL DEFAULT 0,
        status      TEXT    NOT NULL DEFAULT 'idle',
        epsilon_budget REAL NOT NULL DEFAULT 1.0
    );

    CREATE TABLE IF NOT EXISTS rounds (
        round_number   INTEGER PRIMARY KEY,
        loss           REAL,
        metrics_json   TEXT    NOT NULL DEFAULT '{}',
        num_clients    INTEGER NOT NULL DEFAULT 0,
        gradient_norm  REAL,
        timestamp      REAL    NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS sites (
        site_id            TEXT PRIMARY KEY,
        status             TEXT    NOT NULL DEFAULT 'offline',
        samples_contributed INTEGER NOT NULL DEFAULT 0,
        last_communication REAL    NOT NULL DEFAULT 0,
        local_training_time_ms REAL NOT NULL DEFAULT 0,
        epsilon_spent      REAL    NOT NULL DEFAULT 0
    );
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    # -- Campaign ----------------------------------------------------------

    def start_campaign(
        self,
        name: str = "default",
        total_rounds: int = 10,
        strategy: str = "fedavg",
        epsilon_budget: float = 1.0,
    ) -> None:
        """Initialise or reset the campaign record."""
        with self._lock:
            self._conn.execute("DELETE FROM campaign")
            self._conn.execute(
                "INSERT INTO campaign (id, name, total_rounds, strategy, "
                "start_ts, status, epsilon_budget) VALUES (1, ?, ?, ?, ?, 'running', ?)",
                (name, total_rounds, strategy, time.time(), epsilon_budget),
            )
            self._conn.commit()

    def complete_campaign(self) -> None:
        """Mark the campaign as completed."""
        with self._lock:
            self._conn.execute(
                "UPDATE campaign SET status = 'completed' WHERE id = 1"
            )
            self._conn.commit()

    def fail_campaign(self) -> None:
        """Mark the campaign as failed."""
        with self._lock:
            self._conn.execute(
                "UPDATE campaign SET status = 'failed' WHERE id = 1"
            )
            self._conn.commit()

    def get_campaign_status(self) -> CampaignStatus:
        """Return the current campaign status snapshot."""
        row = self._conn.execute(
            "SELECT * FROM campaign WHERE id = 1"
        ).fetchone()
        if row is None:
            return CampaignStatus()

        start_ts = row["start_ts"]
        elapsed = time.time() - start_ts if start_ts > 0 else 0.0

        # Current round = max round_number in rounds table
        max_row = self._conn.execute(
            "SELECT MAX(round_number) AS mr FROM rounds"
        ).fetchone()
        current_round = max_row["mr"] or 0

        return CampaignStatus(
            campaign_name=row["name"],
            current_round=current_round,
            total_rounds=row["total_rounds"],
            strategy=row["strategy"],
            start_timestamp=start_ts,
            elapsed_seconds=round(elapsed, 1),
            status=row["status"],
        )

    # -- Rounds ------------------------------------------------------------

    def record_round(
        self,
        round_number: int,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        num_clients: int = 0,
        gradient_norm: Optional[float] = None,
    ) -> None:
        """Persist a completed round's metrics."""
        metrics = metrics or {}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO rounds "
                "(round_number, loss, metrics_json, num_clients, "
                "gradient_norm, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    round_number,
                    loss,
                    json.dumps(metrics),
                    num_clients,
                    gradient_norm,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_rounds(self) -> List[RoundRecord]:
        """Return all persisted rounds ordered by round number."""
        rows = self._conn.execute(
            "SELECT * FROM rounds ORDER BY round_number"
        ).fetchall()
        result: List[RoundRecord] = []
        for row in rows:
            result.append(
                RoundRecord(
                    round_number=row["round_number"],
                    loss=row["loss"],
                    metrics=json.loads(row["metrics_json"]),
                    num_clients=row["num_clients"],
                    gradient_norm=row["gradient_norm"],
                    timestamp=row["timestamp"],
                )
            )
        return result

    # -- Sites -------------------------------------------------------------

    def update_site(
        self,
        site_id: str,
        *,
        status: Optional[str] = None,
        samples_contributed: Optional[int] = None,
        local_training_time_ms: Optional[float] = None,
        epsilon_spent: Optional[float] = None,
    ) -> None:
        """Upsert a site participation record."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM sites WHERE site_id = ?", (site_id,)
            ).fetchone()

            if existing is None:
                self._conn.execute(
                    "INSERT INTO sites (site_id, status, samples_contributed, "
                    "last_communication, local_training_time_ms, epsilon_spent) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        site_id,
                        status or "online",
                        samples_contributed or 0,
                        time.time(),
                        local_training_time_ms or 0.0,
                        epsilon_spent or 0.0,
                    ),
                )
            else:
                updates: list[str] = []
                params: list[Any] = []
                if status is not None:
                    updates.append("status = ?")
                    params.append(status)
                if samples_contributed is not None:
                    updates.append("samples_contributed = ?")
                    params.append(samples_contributed)
                if local_training_time_ms is not None:
                    updates.append("local_training_time_ms = ?")
                    params.append(local_training_time_ms)
                if epsilon_spent is not None:
                    updates.append("epsilon_spent = ?")
                    params.append(epsilon_spent)
                updates.append("last_communication = ?")
                params.append(time.time())
                params.append(site_id)
                self._conn.execute(
                    f"UPDATE sites SET {', '.join(updates)} WHERE site_id = ?",
                    params,
                )
            self._conn.commit()

    def get_sites(self) -> List[SiteRecord]:
        """Return all site participation records."""
        rows = self._conn.execute(
            "SELECT * FROM sites ORDER BY site_id"
        ).fetchall()
        return [
            SiteRecord(
                site_id=row["site_id"],
                status=row["status"],
                samples_contributed=row["samples_contributed"],
                last_communication=row["last_communication"],
                local_training_time_ms=row["local_training_time_ms"],
                epsilon_spent=row["epsilon_spent"],
            )
            for row in rows
        ]

    # -- Privacy budget ----------------------------------------------------

    def get_epsilon_budget(self) -> float:
        """Return the configured total epsilon budget."""
        row = self._conn.execute(
            "SELECT epsilon_budget FROM campaign WHERE id = 1"
        ).fetchone()
        return row["epsilon_budget"] if row else 1.0

    # -- Convergence -------------------------------------------------------

    def get_convergence_indicators(
        self, plateau_window: int = 5
    ) -> ConvergenceIndicators:
        """Compute convergence indicators from persisted round data."""
        rounds = self.get_rounds()
        gradient_norms = [
            r.gradient_norm for r in rounds if r.gradient_norm is not None
        ]
        loss_trend: List[Optional[float]] = [r.loss for r in rounds]

        # Plateau detection: if the last `plateau_window` losses have
        # relative change < 1%, declare plateau
        plateau = False
        early_stop = False
        numeric_losses = [v for v in loss_trend if v is not None]

        if len(numeric_losses) >= plateau_window:
            recent = numeric_losses[-plateau_window:]
            if recent[0] != 0:
                max_change = max(
                    abs(recent[i] - recent[i - 1]) / abs(recent[0])
                    for i in range(1, len(recent))
                )
                plateau = max_change < 0.01
                early_stop = plateau and len(numeric_losses) >= plateau_window + 2

        return ConvergenceIndicators(
            gradient_norms=gradient_norms,
            loss_trend=loss_trend,
            plateau_detected=plateau,
            early_stop_recommended=early_stop,
            plateau_window=plateau_window,
        )

    # -- Cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"FLMetricsStore(db_path={self._db_path!r})"
