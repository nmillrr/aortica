"""Automated performance monitoring for production Aortica deployments.

Tracks rolling AUC, F1, and calibration metrics against labeled
production data, with drift detection and alerting.

Usage::

    from aortica.validation import PerformanceMonitor

    monitor = PerformanceMonitor("/path/to/monitor_data")
    monitor.record_prediction(
        ecg_id="ecg_001",
        task="rhythm",
        predictions={"AF": 0.85, "STEMI": 0.02},
        ground_truth={"AF": 1, "STEMI": 0},
    )
    status = monitor.get_status()
    print(status.summary())

"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskMetricSnapshot:
    """Metrics snapshot for a single task over a monitoring window.

    Attributes:
        task_name: Task identifier (rhythm, structural, ischaemia, risk).
        auc: Area under the ROC curve.
        f1: Macro-F1 score.
        ece: Expected Calibration Error.
        n_samples: Number of labeled samples in the window.
    """

    task_name: str = ""
    auc: float = 0.0
    f1: float = 0.0
    ece: float = 0.0
    n_samples: int = 0


@dataclass
class DriftAlert:
    """A drift detection alert.

    Attributes:
        task_name: Which task triggered the alert.
        metric_name: Which metric drifted (auc, f1, ece).
        current_value: Current rolling metric value.
        baseline_value: Baseline metric value.
        threshold: The configured threshold or deviation limit.
        alert_type: "below_threshold" or "deviation_from_baseline".
        timestamp: Unix timestamp when the alert was generated.
        message: Human-readable alert message.
    """

    task_name: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    threshold: float = 0.0
    alert_type: str = ""
    timestamp: float = 0.0
    message: str = ""


@dataclass
class SubgroupMetric:
    """Rolling metrics for one demographic subgroup on one task (US-130)."""

    subgroup: str = ""  # e.g. "sex_M", "age_50-59"
    task_name: str = ""
    auc: float = 0.0
    f1: float = 0.0
    ece: float = 0.0
    n_samples: int = 0


@dataclass
class SubgroupStatus:
    """Demographic-stratified monitoring status (US-130)."""

    has_demographics: bool = False
    min_samples: int = 30
    subgroups: list[SubgroupMetric] = field(default_factory=list)
    drift_alerts: list[DriftAlert] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorStatus:
    """Overall performance monitoring status.

    Attributes:
        task_metrics: Current rolling metrics per task.
        drift_alerts: Active drift alerts.
        window_days: Monitoring window in days.
        total_predictions: Total predictions recorded.
        total_labeled: Total predictions with ground-truth labels.
        last_updated: Timestamp of last recorded prediction.
    """

    task_metrics: dict[str, TaskMetricSnapshot] = field(default_factory=dict)
    drift_alerts: list[DriftAlert] = field(default_factory=list)
    window_days: int = 30
    total_predictions: int = 0
    total_labeled: int = 0
    last_updated: float = 0.0

    def has_drift(self) -> bool:
        """Return True if any drift alerts are active."""
        return len(self.drift_alerts) > 0

    def summary(self) -> str:
        """Return a human-readable status summary."""
        lines: list[str] = []
        lines.append(f"Performance Monitor Status (window={self.window_days}d)")
        lines.append("=" * 60)
        lines.append(
            f"Total predictions: {self.total_predictions}  "
            f"Labeled: {self.total_labeled}"
        )

        if self.task_metrics:
            lines.append("\n--- Metrics ---")
            for task_name, snap in sorted(self.task_metrics.items()):
                lines.append(
                    f"  {task_name}: AUC={snap.auc:.4f}  "
                    f"F1={snap.f1:.4f}  ECE={snap.ece:.4f}  "
                    f"(n={snap.n_samples})"
                )

        if self.drift_alerts:
            lines.append(f"\n--- DRIFT ALERTS ({len(self.drift_alerts)}) ---")
            for alert in self.drift_alerts:
                lines.append(f"  ⚠ {alert.message}")
        else:
            lines.append("\n✓ No drift detected")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the status as a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric helpers (lightweight versions for rolling computation)
# ---------------------------------------------------------------------------


def _rolling_auc(predictions: list[float], labels: list[int]) -> float:
    """Compute AUC from lists of predictions and binary labels."""
    if len(predictions) < 2:
        return 0.5

    preds = np.array(predictions, dtype=np.float64)
    labs = np.array(labels, dtype=np.int64)

    n_pos = int(labs.sum())
    n_neg = len(labs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-preds)
    sorted_labels = labs[order]

    tp = 0.0
    fp = 0.0
    tpr_list: list[float] = [0.0]
    fpr_list: list[float] = [0.0]

    for label in sorted_labels:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (
            tpr_list[i] + tpr_list[i - 1]
        ) / 2.0

    return float(auc)


def _rolling_f1(predictions: list[float], labels: list[int], threshold: float = 0.5) -> float:
    """Compute F1 from lists of predictions and binary labels."""
    if not predictions:
        return 0.0

    preds = np.array(predictions) >= threshold
    labs = np.array(labels)

    tp = float(np.sum(preds & (labs == 1)))
    fp = float(np.sum(preds & (labs == 0)))
    fn = float(np.sum(~preds & (labs == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1


def _rolling_ece(
    predictions: list[float], labels: list[int], num_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    if not predictions:
        return 0.0

    preds = np.array(predictions, dtype=np.float64)
    labs = np.array(labels, dtype=np.float64)
    n_total = len(preds)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        avg_conf = float(preds[mask].mean())
        avg_acc = float(labs[mask].mean())
        ece += (n_bin / n_total) * abs(avg_acc - avg_conf)

    return ece


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS monitor_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_id TEXT NOT NULL,
    task TEXT NOT NULL,
    class_name TEXT NOT NULL,
    prediction REAL NOT NULL,
    ground_truth INTEGER DEFAULT NULL,
    timestamp REAL NOT NULL,
    age INTEGER DEFAULT NULL,
    sex TEXT DEFAULT NULL
);
"""

_CREATE_BASELINES_TABLE = """
CREATE TABLE IF NOT EXISTS monitor_baselines (
    task TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    baseline_value REAL NOT NULL,
    set_at REAL NOT NULL,
    PRIMARY KEY (task, metric_name)
);
"""

_CREATE_IDX_TASK_TS = """
CREATE INDEX IF NOT EXISTS idx_monitor_task_ts
ON monitor_predictions (task, timestamp);
"""

_CREATE_IDX_ECG_ID = """
CREATE INDEX IF NOT EXISTS idx_monitor_ecg_id
ON monitor_predictions (ecg_id);
"""


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """SQLite-backed automated performance monitoring.

    Tracks rolling metrics against labeled production data and detects
    drift when metrics drop below thresholds or deviate from baselines.

    Parameters
    ----------
    db_dir:
        Directory for the SQLite database.
    window_days:
        Rolling monitoring window in days.  Default: 30.
    drift_deviation:
        Maximum allowed deviation from baseline (fraction).
        Default: 0.05 (5%).
    min_threshold:
        Per-task minimum metric thresholds.  Keys are ``"task.metric"``
        (e.g. ``"rhythm.auc"``).  Metrics below these trigger alerts.
    webhook_url:
        Optional webhook URL for drift alerts.
    db_filename:
        Name of the SQLite file.  Default: ``monitor.db``.
    """

    def __init__(
        self,
        db_dir: str | Path,
        window_days: int = 30,
        drift_deviation: float = 0.05,
        min_thresholds: Optional[dict[str, float]] = None,
        webhook_url: Optional[str] = None,
        db_filename: str = "monitor.db",
        subgroup_min_samples: int = 30,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / db_filename

        self.window_days = window_days
        self.drift_deviation = drift_deviation
        self.min_thresholds = min_thresholds or {}
        self.webhook_url = webhook_url
        self.subgroup_min_samples = subgroup_min_samples

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_PREDICTIONS_TABLE)
        self._conn.execute(_CREATE_BASELINES_TABLE)
        self._conn.execute(_CREATE_IDX_TASK_TS)
        self._conn.execute(_CREATE_IDX_ECG_ID)
        self._migrate_demographics()
        self._conn.commit()

    def _migrate_demographics(self) -> None:
        """Add age/sex columns to a pre-existing table (US-130)."""
        existing = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(monitor_predictions)"
            ).fetchall()
        }
        if "age" not in existing:
            self._conn.execute(
                "ALTER TABLE monitor_predictions ADD COLUMN age INTEGER DEFAULT NULL"
            )
        if "sex" not in existing:
            self._conn.execute(
                "ALTER TABLE monitor_predictions ADD COLUMN sex TEXT DEFAULT NULL"
            )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        ecg_id: str,
        task: str,
        predictions: dict[str, float],
        ground_truth: Optional[dict[str, int]] = None,
        timestamp: Optional[float] = None,
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> int:
        """Record predictions and optional ground-truth for one ECG/task.

        Parameters
        ----------
        ecg_id:
            Unique ECG identifier.
        task:
            Task name (rhythm, structural, ischaemia, risk).
        predictions:
            Class name → predicted probability mapping.
        ground_truth:
            Class name → binary label mapping (if available).
        timestamp:
            Unix timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        int
            Number of rows inserted.
        """
        ts = timestamp if timestamp is not None else time.time()
        gt = ground_truth or {}
        rows_inserted = 0

        for class_name, pred_value in predictions.items():
            gt_value = gt.get(class_name)
            self._conn.execute(
                """
                INSERT INTO monitor_predictions
                    (ecg_id, task, class_name, prediction, ground_truth,
                     timestamp, age, sex)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ecg_id, task, class_name, pred_value, gt_value, ts, age, sex),
            )
            rows_inserted += 1

        self._conn.commit()
        return rows_inserted

    def add_ground_truth(
        self,
        ecg_id: str,
        task: str,
        ground_truth: dict[str, int],
    ) -> int:
        """Add ground-truth labels to existing predictions.

        Parameters
        ----------
        ecg_id:
            ECG identifier matching a previous ``record_prediction`` call.
        task:
            Task name.
        ground_truth:
            Class name → binary label mapping.

        Returns
        -------
        int
            Number of rows updated.
        """
        updated = 0
        for class_name, gt_value in ground_truth.items():
            cursor = self._conn.execute(
                """
                UPDATE monitor_predictions
                SET ground_truth = ?
                WHERE ecg_id = ? AND task = ? AND class_name = ?
                """,
                (gt_value, ecg_id, task, class_name),
            )
            updated += cursor.rowcount
        self._conn.commit()
        return updated

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        task: str,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Set a baseline metric value for drift comparison.

        Parameters
        ----------
        task:
            Task name.
        metric_name:
            Metric name (auc, f1, ece).
        value:
            Baseline value.
        timestamp:
            When the baseline was set.  Defaults to now.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO monitor_baselines
                (task, metric_name, baseline_value, set_at)
            VALUES (?, ?, ?, ?)
            """,
            (task, metric_name, value, ts),
        )
        self._conn.commit()

    def get_baseline(self, task: str, metric_name: str) -> Optional[float]:
        """Get the baseline value for a task/metric pair."""
        row = self._conn.execute(
            """
            SELECT baseline_value FROM monitor_baselines
            WHERE task = ? AND metric_name = ?
            """,
            (task, metric_name),
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_task_metrics(
        self,
        task: str,
        since_ts: float,
        extra_where: str = "",
        extra_params: tuple[Any, ...] = (),
    ) -> Optional[TaskMetricSnapshot]:
        """Compute rolling metrics for a single task (optionally a subgroup)."""
        rows = self._conn.execute(
            f"""
            SELECT class_name, prediction, ground_truth
            FROM monitor_predictions
            WHERE task = ? AND timestamp >= ? AND ground_truth IS NOT NULL
            {extra_where}
            ORDER BY class_name
            """,
            (task, since_ts, *extra_params),
        ).fetchall()

        if not rows:
            return None

        # Group by class
        class_data: dict[str, tuple[list[float], list[int]]] = {}
        for class_name, pred, gt in rows:
            if class_name not in class_data:
                class_data[class_name] = ([], [])
            class_data[class_name][0].append(pred)
            class_data[class_name][1].append(int(gt))

        # Per-class AUC, F1, then macro-average
        auc_values: list[float] = []
        f1_values: list[float] = []
        all_preds: list[float] = []
        all_labels: list[int] = []

        for _cls, (preds, labels) in class_data.items():
            auc_values.append(_rolling_auc(preds, labels))
            f1_values.append(_rolling_f1(preds, labels))
            all_preds.extend(preds)
            all_labels.extend(labels)

        macro_auc = float(np.mean(auc_values)) if auc_values else 0.0
        macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        ece = _rolling_ece(all_preds, all_labels)

        n_samples = len(set(
            r[0] for r in self._conn.execute(
                f"""
                SELECT DISTINCT ecg_id FROM monitor_predictions
                WHERE task = ? AND timestamp >= ? AND ground_truth IS NOT NULL
                {extra_where}
                """,
                (task, since_ts, *extra_params),
            ).fetchall()
        ))

        return TaskMetricSnapshot(
            task_name=task,
            auc=macro_auc,
            f1=macro_f1,
            ece=ece,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def _detect_drift(
        self, task_metrics: dict[str, TaskMetricSnapshot]
    ) -> list[DriftAlert]:
        """Check for drift in current metrics vs baselines/thresholds."""
        alerts: list[DriftAlert] = []
        now = time.time()

        for task_name, snap in task_metrics.items():
            for metric_name in ("auc", "f1"):
                current = getattr(snap, metric_name)

                # Check absolute threshold
                threshold_key = f"{task_name}.{metric_name}"
                if threshold_key in self.min_thresholds:
                    threshold = self.min_thresholds[threshold_key]
                    if current < threshold:
                        alerts.append(DriftAlert(
                            task_name=task_name,
                            metric_name=metric_name,
                            current_value=current,
                            baseline_value=0.0,
                            threshold=threshold,
                            alert_type="below_threshold",
                            timestamp=now,
                            message=(
                                f"{task_name}.{metric_name}={current:.4f} "
                                f"below threshold {threshold:.4f}"
                            ),
                        ))

                # Check deviation from baseline
                baseline = self.get_baseline(task_name, metric_name)
                if baseline is not None and baseline > 0:
                    deviation = (baseline - current) / baseline
                    if deviation > self.drift_deviation:
                        alerts.append(DriftAlert(
                            task_name=task_name,
                            metric_name=metric_name,
                            current_value=current,
                            baseline_value=baseline,
                            threshold=self.drift_deviation,
                            alert_type="deviation_from_baseline",
                            timestamp=now,
                            message=(
                                f"{task_name}.{metric_name}={current:.4f} "
                                f"deviated {deviation:.1%} from baseline "
                                f"{baseline:.4f} (limit {self.drift_deviation:.0%})"
                            ),
                        ))

        return alerts

    def _send_webhook_alert(self, alerts: list[DriftAlert]) -> bool:
        """Send drift alerts to configured webhook (best-effort).

        Returns True if webhook was attempted, False if no URL configured.
        """
        if not self.webhook_url or not alerts:
            return False

        try:
            import urllib.request

            payload = json.dumps({
                "text": "Aortica Performance Drift Detected",
                "alerts": [asdict(a) for a in alerts],
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            logger.info("Drift alert sent to webhook: %s", self.webhook_url)
            return True
        except Exception:
            logger.warning(
                "Failed to send drift alert to webhook: %s",
                self.webhook_url,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> MonitorStatus:
        """Compute current monitoring status with drift detection.

        Returns
        -------
        MonitorStatus
            Current rolling metrics, drift alerts, and counts.
        """
        window_seconds = self.window_days * 86400
        since_ts = time.time() - window_seconds

        # Discover tasks with data
        task_rows = self._conn.execute(
            "SELECT DISTINCT task FROM monitor_predictions"
        ).fetchall()
        tasks = [r[0] for r in task_rows]

        # Compute rolling metrics per task
        task_metrics: dict[str, TaskMetricSnapshot] = {}
        for task in tasks:
            snap = self._compute_task_metrics(task, since_ts)
            if snap is not None:
                task_metrics[task] = snap

        # Detect drift
        drift_alerts = self._detect_drift(task_metrics)

        # Send webhook if drift detected
        if drift_alerts:
            self._send_webhook_alert(drift_alerts)
            for alert in drift_alerts:
                logger.warning("Drift detected: %s", alert.message)

        # Counts
        total_preds_row = self._conn.execute(
            "SELECT COUNT(DISTINCT ecg_id || '|' || task) FROM monitor_predictions"
        ).fetchone()
        total_preds = total_preds_row[0] if total_preds_row else 0

        total_labeled_row = self._conn.execute(
            "SELECT COUNT(DISTINCT ecg_id || '|' || task) "
            "FROM monitor_predictions WHERE ground_truth IS NOT NULL"
        ).fetchone()
        total_labeled = total_labeled_row[0] if total_labeled_row else 0

        last_row = self._conn.execute(
            "SELECT MAX(timestamp) FROM monitor_predictions"
        ).fetchone()
        last_updated = last_row[0] if last_row and last_row[0] else 0.0

        return MonitorStatus(
            task_metrics=task_metrics,
            drift_alerts=drift_alerts,
            window_days=self.window_days,
            total_predictions=total_preds,
            total_labeled=total_labeled,
            last_updated=last_updated,
        )

    # ------------------------------------------------------------------
    # Demographic-stratified monitoring (US-130)
    # ------------------------------------------------------------------

    def _has_demographics(self) -> bool:
        """Return True if any labeled prediction carries age or sex."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM monitor_predictions "
            "WHERE (age IS NOT NULL OR sex IS NOT NULL) "
            "AND ground_truth IS NOT NULL"
        ).fetchone()
        return bool(row and row[0] > 0)

    @staticmethod
    def _age_decile(age: int) -> str:
        decade = (int(age) // 10) * 10
        return "age_90+" if decade >= 90 else f"age_{decade}-{decade + 9}"

    def get_subgroup_status(
        self, min_samples: Optional[int] = None
    ) -> SubgroupStatus:
        """Compute rolling metrics stratified by sex and age decile.

        Only subgroups with at least ``min_samples`` labeled ECGs are
        reported.  When no demographics have been recorded, returns a status
        with ``has_demographics=False`` and an explanatory note rather than
        implying a breakdown exists.

        Args:
            min_samples: Minimum labeled ECGs per subgroup (default from the
                monitor's ``subgroup_min_samples``, i.e. 30).
        """
        threshold = min_samples if min_samples is not None else self.subgroup_min_samples

        if not self._has_demographics():
            return SubgroupStatus(
                has_demographics=False,
                min_samples=threshold,
                subgroups=[],
                note="No demographic attributes recorded; subgroup "
                "stratification is unavailable.",
            )

        since_ts = time.time() - self.window_days * 86400
        tasks = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT task FROM monitor_predictions"
            ).fetchall()
        ]

        # Discover the distinct sex values and age deciles present.
        sex_values = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT sex FROM monitor_predictions WHERE sex IS NOT NULL"
            ).fetchall()
        ]
        ages = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT age FROM monitor_predictions WHERE age IS NOT NULL"
            ).fetchall()
        ]
        age_deciles = sorted({self._age_decile(a) for a in ages})

        results: list[SubgroupMetric] = []
        for task in tasks:
            for sex in sex_values:
                snap = self._compute_task_metrics(
                    task, since_ts, "AND sex = ?", (sex,)
                )
                if snap and snap.n_samples >= threshold:
                    results.append(self._as_subgroup(f"sex_{sex}", snap))
            for decile in age_deciles:
                low = int(decile.split("_")[1].split("-")[0].rstrip("+"))
                high = low + 9 if not decile.endswith("+") else 200
                snap = self._compute_task_metrics(
                    task, since_ts, "AND age >= ? AND age <= ?", (low, high)
                )
                if snap and snap.n_samples >= threshold:
                    results.append(self._as_subgroup(decile, snap))

        # Optional per-subgroup drift: flag subgroups whose F1 falls below the
        # aggregate task F1 by more than the configured deviation.
        aggregate = {
            t: s for t, s in (
                (task, self._compute_task_metrics(task, since_ts))
                for task in tasks
            ) if s is not None
        }
        subgroup_alerts: list[DriftAlert] = []
        for sm in results:
            agg = aggregate.get(sm.task_name)
            if agg and agg.f1 - sm.f1 > self.drift_deviation:
                subgroup_alerts.append(
                    DriftAlert(
                        task_name=sm.task_name,
                        metric_name="f1",
                        current_value=sm.f1,
                        baseline_value=agg.f1,
                        threshold=self.drift_deviation,
                        alert_type="subgroup_equity_deviation",
                        timestamp=time.time(),
                        message=(
                            f"Subgroup {sm.subgroup} F1 {sm.f1:.3f} is "
                            f"{agg.f1 - sm.f1:.3f} below the aggregate "
                            f"{sm.task_name} F1 ({agg.f1:.3f})"
                        ),
                    )
                )

        return SubgroupStatus(
            has_demographics=True,
            min_samples=threshold,
            subgroups=results,
            drift_alerts=subgroup_alerts,
            note="",
        )

    @staticmethod
    def _as_subgroup(name: str, snap: TaskMetricSnapshot) -> SubgroupMetric:
        return SubgroupMetric(
            subgroup=name,
            task_name=snap.task_name,
            auc=snap.auc,
            f1=snap.f1,
            ece=snap.ece,
            n_samples=snap.n_samples,
        )

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "PerformanceMonitor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
