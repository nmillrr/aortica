"""Central aggregation of synced edge results (US-128).

Edge devices store results locally (US-054) and sync them to a central
server (US-055/056).  :class:`CentralAggregator` receives those batches,
tags each result with its ``device_id`` and ``site_id``, feeds labeled
results into the central :class:`~aortica.validation.PerformanceMonitor`
(US-100), and exposes per-site analytics, cross-site anomaly detection
(z-score), and sync-gap reconciliation.

The store is SQLite-backed and thread-safe so it can run behind the
central FastAPI server.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# A finding at/above this confidence counts toward a site's finding profile.
_FINDING_THRESHOLD = 0.5
# Critical conditions used for the cross-site anomaly statistic.
_CRITICAL = {"STEMI", "VT", "VF", "occlusive_NSTEMI", "hyperkalaemia"}


@dataclass
class SiteMetrics:
    """Aggregate metrics for one edge site."""

    site_id: str
    device_ids: List[str] = field(default_factory=list)
    total_ecgs: int = 0
    mean_quality: float = 0.0
    critical_rate: float = 0.0
    finding_distribution: Dict[str, int] = field(default_factory=dict)
    last_sync: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "device_ids": self.device_ids,
            "total_ecgs": self.total_ecgs,
            "mean_quality": self.mean_quality,
            "critical_rate": self.critical_rate,
            "finding_distribution": self.finding_distribution,
            "last_sync": self.last_sync,
        }


@dataclass
class Anomaly:
    """A site flagged as an outlier vs the fleet."""

    site_id: str
    metric: str
    value: float
    fleet_mean: float
    fleet_std: float
    z_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "metric": self.metric,
            "value": self.value,
            "fleet_mean": self.fleet_mean,
            "fleet_std": self.fleet_std,
            "z_score": self.z_score,
        }


class CentralAggregator:
    """Receives and aggregates synced edge results for central analytics."""

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        monitor: Optional[Any] = None,
        anomaly_z_threshold: float = 2.0,
    ) -> None:
        self._monitor = monitor
        self._z_threshold = anomaly_z_threshold
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synced_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                site_id TEXT NOT NULL,
                ecg_id TEXT NOT NULL,
                predictions TEXT NOT NULL,
                quality_score REAL,
                inference_ts REAL,
                received_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- Ingestion ----------------------------------------------------------

    def receive_batch(
        self,
        device_id: str,
        site_id: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ingest a batch of synced results from one edge device.

        Each result is a dict with ``ecg_id`` and ``predictions``
        (``{task: {class: confidence}}``), optionally ``quality_score``,
        ``inference_ts``, and ``ground_truth`` (``{task: {class: 0/1}}``).

        Returns a summary ``{received, labeled_forwarded}``.
        """
        now = time.time()
        labeled = 0
        with self._lock:
            for result in results:
                ecg_id = result.get("ecg_id", "")
                predictions = result.get("predictions", {})
                quality = result.get("quality_score")
                inference_ts = result.get("inference_ts")
                self._conn.execute(
                    "INSERT INTO synced_results (device_id, site_id, ecg_id, "
                    "predictions, quality_score, inference_ts, received_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        device_id, site_id, ecg_id, json.dumps(predictions),
                        quality, inference_ts, now,
                    ),
                )
                # Feed labeled results into the central performance monitor.
                ground_truth = result.get("ground_truth")
                if self._monitor is not None and ground_truth:
                    labeled += self._forward_to_monitor(
                        ecg_id, predictions, ground_truth
                    )
            self._conn.commit()
        return {"received": len(results), "labeled_forwarded": labeled}

    def _forward_to_monitor(
        self,
        ecg_id: str,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> int:
        monitor: Any = self._monitor
        forwarded = 0
        for task, class_preds in predictions.items():
            if not isinstance(class_preds, dict):
                continue
            gt = ground_truth.get(task)
            if not isinstance(gt, dict):
                continue
            try:
                monitor.record_prediction(
                    ecg_id=ecg_id, task=task,
                    predictions={k: float(v) for k, v in class_preds.items()},
                    ground_truth={k: int(v) for k, v in gt.items()},
                )
                forwarded += 1
            except Exception:  # noqa: BLE001 - monitoring is best-effort
                pass
        return forwarded

    # -- Site analytics -----------------------------------------------------

    def site_metrics(self) -> List[SiteMetrics]:
        """Return per-site aggregate metrics."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT site_id, device_id, ecg_id, predictions, quality_score, "
                "received_at FROM synced_results"
            ).fetchall()

        by_site: Dict[str, Dict[str, Any]] = {}
        for site_id, device_id, _ecg, preds_json, quality, received_at in rows:
            site = by_site.setdefault(
                site_id,
                {
                    "devices": set(),
                    "count": 0,
                    "quality_sum": 0.0,
                    "quality_n": 0,
                    "critical": 0,
                    "findings": {},
                    "last": None,
                },
            )
            site["devices"].add(device_id)
            site["count"] += 1
            if quality is not None:
                site["quality_sum"] += float(quality)
                site["quality_n"] += 1
            if site["last"] is None or received_at > site["last"]:
                site["last"] = received_at
            preds = json.loads(preds_json)
            is_critical = False
            for class_preds in preds.values():
                if not isinstance(class_preds, dict):
                    continue
                for name, conf in class_preds.items():
                    if float(conf) >= _FINDING_THRESHOLD:
                        site["findings"][name] = site["findings"].get(name, 0) + 1
                        if name in _CRITICAL:
                            is_critical = True
            if is_critical:
                site["critical"] += 1

        metrics: List[SiteMetrics] = []
        for site_id, data in sorted(by_site.items()):
            count = data["count"]
            metrics.append(
                SiteMetrics(
                    site_id=site_id,
                    device_ids=sorted(data["devices"]),
                    total_ecgs=count,
                    mean_quality=(
                        data["quality_sum"] / data["quality_n"]
                        if data["quality_n"]
                        else 0.0
                    ),
                    critical_rate=(data["critical"] / count) if count else 0.0,
                    finding_distribution=data["findings"],
                    last_sync=data["last"],
                )
            )
        return metrics

    # -- Anomaly detection --------------------------------------------------

    def detect_anomalies(self) -> List[Anomaly]:
        """Flag sites whose quality or critical rate deviates (|z| > threshold)."""
        metrics = self.site_metrics()
        if len(metrics) < 3:
            # Need enough sites for a meaningful fleet distribution.
            return []
        anomalies: List[Anomaly] = []
        for metric_name, getter in (
            ("mean_quality", lambda m: m.mean_quality),
            ("critical_rate", lambda m: m.critical_rate),
        ):
            values = [getter(m) for m in metrics]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = var ** 0.5
            if std == 0:
                continue
            for m in metrics:
                z = (getter(m) - mean) / std
                if abs(z) > self._z_threshold:
                    anomalies.append(
                        Anomaly(
                            site_id=m.site_id,
                            metric=metric_name,
                            value=getter(m),
                            fleet_mean=mean,
                            fleet_std=std,
                            z_score=z,
                        )
                    )
        return anomalies

    # -- Reconciliation -----------------------------------------------------

    def reconcile(self, device_id: str, expected_count: int) -> Dict[str, Any]:
        """Report the sync gap between expected and received counts for a device."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM synced_results WHERE device_id = ?",
                (device_id,),
            ).fetchone()
        received = int(row[0]) if row else 0
        gap = expected_count - received
        return {
            "device_id": device_id,
            "expected": expected_count,
            "received": received,
            "gap": gap,
            "complete": gap <= 0,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
