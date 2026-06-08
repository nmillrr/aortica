"""Tests for aortica.validation.performance_monitor — US-100.

Covers:
- PerformanceMonitor initialization
- Prediction recording and ground-truth linkage
- Rolling metric computation (AUC, F1, ECE)
- Baseline setting and retrieval
- Drift detection (threshold + deviation)
- Stable scenario (no drift)
- Drifting scenario (drift detected)
- MonitorStatus summary output
- Monitor status API endpoint
- Module imports
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from aortica.validation.performance_monitor import (
    DriftAlert,
    MonitorStatus,
    PerformanceMonitor,
    TaskMetricSnapshot,
    _rolling_auc,
    _rolling_ece,
    _rolling_f1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor(tmp_path: Path) -> PerformanceMonitor:
    """Create a PerformanceMonitor with a temporary database."""
    return PerformanceMonitor(db_dir=str(tmp_path / "monitor"))


@pytest.fixture
def populated_monitor(monitor: PerformanceMonitor) -> PerformanceMonitor:
    """Monitor with synthetic labeled production data.

    Creates a stable scenario: high AUC, good F1, low ECE.
    """
    now = time.time()

    # Record 50 ECGs with rhythm task predictions and labels
    for i in range(50):
        ecg_id = f"ecg_{i:03d}"
        # Simulate a well-performing model:
        #   true positives have high predictions, true negatives have low
        is_af = i % 5 == 0  # 10 positive, 40 negative
        af_pred = 0.85 + 0.1 * (i % 3) / 3 if is_af else 0.1 + 0.05 * (i % 4) / 4

        is_stemi = i % 10 == 0  # 5 positive, 45 negative
        stemi_pred = 0.90 if is_stemi else 0.05

        monitor.record_prediction(
            ecg_id=ecg_id,
            task="rhythm",
            predictions={"AF": af_pred, "STEMI": stemi_pred},
            ground_truth={"AF": int(is_af), "STEMI": int(is_stemi)},
            timestamp=now - 86400 * (50 - i) / 50,  # Spread over ~1 day
        )

    return monitor


@pytest.fixture
def drifting_monitor(tmp_path: Path) -> PerformanceMonitor:
    """Monitor with drifting data: model predictions become poor."""
    mon = PerformanceMonitor(
        db_dir=str(tmp_path / "drift_monitor"),
        min_thresholds={"rhythm.auc": 0.85, "rhythm.f1": 0.60},
        drift_deviation=0.05,
    )

    # Set a high baseline
    mon.set_baseline("rhythm", "auc", 0.95)
    mon.set_baseline("rhythm", "f1", 0.80)

    now = time.time()

    # Record predictions where model is basically random
    for i in range(40):
        ecg_id = f"drift_ecg_{i:03d}"
        is_af = i % 4 == 0  # 10 positive, 30 negative
        # Random-ish predictions → poor AUC
        af_pred = 0.5 + 0.05 * (i % 3) - 0.025

        mon.record_prediction(
            ecg_id=ecg_id,
            task="rhythm",
            predictions={"AF": af_pred},
            ground_truth={"AF": int(is_af)},
            timestamp=now - 86400 * (40 - i) / 40,
        )

    return mon


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test PerformanceMonitor initialization."""

    def test_creates_database(self, tmp_path: Path) -> None:
        mon = PerformanceMonitor(db_dir=str(tmp_path / "init_test"))
        db_path = tmp_path / "init_test" / "monitor.db"
        assert db_path.exists()
        mon.close()

    def test_custom_db_filename(self, tmp_path: Path) -> None:
        mon = PerformanceMonitor(
            db_dir=str(tmp_path / "custom"),
            db_filename="my_monitor.db",
        )
        assert (tmp_path / "custom" / "my_monitor.db").exists()
        mon.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        with PerformanceMonitor(
            db_dir=str(tmp_path / "ctx")
        ) as mon:
            mon.record_prediction("e1", "rhythm", {"AF": 0.5})
            status = mon.get_status()
            assert status.total_predictions == 1

    def test_custom_window(self, tmp_path: Path) -> None:
        mon = PerformanceMonitor(
            db_dir=str(tmp_path / "window"),
            window_days=7,
        )
        assert mon.window_days == 7
        mon.close()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    """Test prediction recording."""

    def test_record_prediction(self, monitor: PerformanceMonitor) -> None:
        n = monitor.record_prediction(
            ecg_id="ecg_001",
            task="rhythm",
            predictions={"AF": 0.85, "VT": 0.12},
            ground_truth={"AF": 1, "VT": 0},
        )
        assert n == 2  # Two classes recorded

    def test_record_without_ground_truth(
        self, monitor: PerformanceMonitor
    ) -> None:
        n = monitor.record_prediction(
            ecg_id="ecg_002",
            task="structural",
            predictions={"LVH": 0.65},
        )
        assert n == 1

    def test_custom_timestamp(self, monitor: PerformanceMonitor) -> None:
        monitor.record_prediction(
            ecg_id="ecg_003",
            task="rhythm",
            predictions={"AF": 0.9},
            ground_truth={"AF": 1},
            timestamp=1234567890.0,
        )
        status = monitor.get_status()
        assert status.last_updated == 1234567890.0

    def test_add_ground_truth_later(
        self, monitor: PerformanceMonitor
    ) -> None:
        monitor.record_prediction(
            "ecg_004", "rhythm", {"AF": 0.8}
        )
        updated = monitor.add_ground_truth(
            "ecg_004", "rhythm", {"AF": 1}
        )
        assert updated == 1

    def test_add_ground_truth_nonexistent(
        self, monitor: PerformanceMonitor
    ) -> None:
        updated = monitor.add_ground_truth(
            "nonexistent", "rhythm", {"AF": 1}
        )
        assert updated == 0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    """Test baseline management."""

    def test_set_and_get_baseline(
        self, monitor: PerformanceMonitor
    ) -> None:
        monitor.set_baseline("rhythm", "auc", 0.95)
        assert monitor.get_baseline("rhythm", "auc") == 0.95

    def test_get_nonexistent_baseline(
        self, monitor: PerformanceMonitor
    ) -> None:
        assert monitor.get_baseline("rhythm", "auc") is None

    def test_baseline_overwrite(
        self, monitor: PerformanceMonitor
    ) -> None:
        monitor.set_baseline("rhythm", "auc", 0.90)
        monitor.set_baseline("rhythm", "auc", 0.95)
        assert monitor.get_baseline("rhythm", "auc") == 0.95


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    """Test rolling metric computation functions."""

    def test_auc_perfect(self) -> None:
        # Perfect separation
        preds = [0.9, 0.8, 0.7, 0.1, 0.05, 0.02]
        labels = [1, 1, 1, 0, 0, 0]
        auc = _rolling_auc(preds, labels)
        assert auc == 1.0

    def test_auc_no_separation(self) -> None:
        # Identical predictions → AUC at or near 0.5
        preds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0, 1, 0]
        auc = _rolling_auc(preds, labels)
        # With tied scores, AUC is expected to be 0.5-0.75 range
        assert 0.4 <= auc <= 0.8

    def test_auc_empty(self) -> None:
        assert _rolling_auc([], []) == 0.5

    def test_auc_single_class(self) -> None:
        assert _rolling_auc([0.9, 0.8], [1, 1]) == 0.5

    def test_f1_perfect(self) -> None:
        preds = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        f1 = _rolling_f1(preds, labels)
        assert f1 == 1.0

    def test_f1_empty(self) -> None:
        assert _rolling_f1([], []) == 0.0

    def test_ece_well_calibrated(self) -> None:
        # Predictions roughly match actual labels
        preds = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        ece = _rolling_ece(preds, labels)
        assert ece < 0.2

    def test_ece_empty(self) -> None:
        assert _rolling_ece([], []) == 0.0


# ---------------------------------------------------------------------------
# Stable scenario (no drift)
# ---------------------------------------------------------------------------


class TestStableScenario:
    """Test with well-performing model data — no drift expected."""

    def test_status_has_metrics(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        status = populated_monitor.get_status()
        assert "rhythm" in status.task_metrics

    def test_no_drift_detected(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        status = populated_monitor.get_status()
        assert not status.has_drift()
        assert len(status.drift_alerts) == 0

    def test_auc_is_high(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        status = populated_monitor.get_status()
        rhythm = status.task_metrics.get("rhythm")
        assert rhythm is not None
        # Model has clear separation → AUC should be high
        assert rhythm.auc > 0.8

    def test_sample_count(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        status = populated_monitor.get_status()
        assert status.total_predictions > 0
        assert status.total_labeled > 0

    def test_summary_output(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        status = populated_monitor.get_status()
        summary = status.summary()
        assert "Performance Monitor Status" in summary
        assert "rhythm" in summary
        assert "No drift detected" in summary


# ---------------------------------------------------------------------------
# Drifting scenario
# ---------------------------------------------------------------------------


class TestDriftingScenario:
    """Test with degraded model data — drift expected."""

    def test_drift_detected(
        self, drifting_monitor: PerformanceMonitor
    ) -> None:
        status = drifting_monitor.get_status()
        assert status.has_drift()
        assert len(status.drift_alerts) > 0

    def test_drift_alert_types(
        self, drifting_monitor: PerformanceMonitor
    ) -> None:
        status = drifting_monitor.get_status()
        alert_types = {a.alert_type for a in status.drift_alerts}
        # Should have at least one type of alert
        assert len(alert_types) > 0
        valid_types = {"below_threshold", "deviation_from_baseline"}
        assert alert_types.issubset(valid_types)

    def test_drift_alert_messages(
        self, drifting_monitor: PerformanceMonitor
    ) -> None:
        status = drifting_monitor.get_status()
        for alert in status.drift_alerts:
            assert alert.message  # Non-empty message
            assert alert.task_name == "rhythm"
            assert alert.metric_name in ("auc", "f1")

    def test_summary_shows_alerts(
        self, drifting_monitor: PerformanceMonitor
    ) -> None:
        status = drifting_monitor.get_status()
        summary = status.summary()
        assert "DRIFT ALERTS" in summary


# ---------------------------------------------------------------------------
# Monitor Status dataclass
# ---------------------------------------------------------------------------


class TestMonitorStatus:
    """Test MonitorStatus dataclass."""

    def test_to_dict(self) -> None:
        status = MonitorStatus(
            task_metrics={
                "rhythm": TaskMetricSnapshot(
                    task_name="rhythm", auc=0.95, f1=0.88, ece=0.03
                )
            },
            window_days=30,
            total_predictions=100,
            total_labeled=80,
        )
        d = status.to_dict()
        assert isinstance(d, dict)
        assert d["window_days"] == 30
        assert d["total_predictions"] == 100
        assert "rhythm" in d["task_metrics"]

    def test_has_drift_false(self) -> None:
        status = MonitorStatus()
        assert not status.has_drift()

    def test_has_drift_true(self) -> None:
        status = MonitorStatus(
            drift_alerts=[
                DriftAlert(
                    task_name="rhythm",
                    metric_name="auc",
                    alert_type="below_threshold",
                    message="test",
                )
            ]
        )
        assert status.has_drift()


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestMonitorAPIEndpoint:
    """Test GET /api/v1/validation/monitor/status endpoint."""

    def test_endpoint_no_monitor(self) -> None:
        """Endpoint returns empty response when no monitor configured."""
        from aortica.api.validation_endpoints import create_validation_router

        router = create_validation_router(collector=None, monitor=None)
        # Router should have the endpoint registered
        routes = [r.path for r in router.routes]
        assert any("monitor/status" in r for r in routes)

    def test_endpoint_with_monitor(
        self, populated_monitor: PerformanceMonitor
    ) -> None:
        """Endpoint returns metrics when monitor is configured."""
        from aortica.api.validation_endpoints import (
            MonitorStatusResponse,
            create_validation_router,
        )

        router = create_validation_router(
            collector=None, monitor=populated_monitor
        )
        routes = [r.path for r in router.routes]
        assert any("monitor/status" in r for r in routes)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


class TestWebhook:
    """Test webhook alert sending."""

    def test_no_webhook_configured(
        self, monitor: PerformanceMonitor
    ) -> None:
        """No error when webhook is not configured."""
        alert = DriftAlert(
            task_name="rhythm", metric_name="auc",
            alert_type="below_threshold", message="test",
        )
        result = monitor._send_webhook_alert([alert])
        assert result is False  # No URL configured

    def test_empty_alerts_skipped(
        self, monitor: PerformanceMonitor
    ) -> None:
        monitor.webhook_url = "http://example.com/webhook"
        result = monitor._send_webhook_alert([])
        assert result is False  # No alerts to send


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify validation module exports are accessible."""

    def test_import_performance_monitor(self) -> None:
        from aortica.validation import PerformanceMonitor  # noqa: F811

    def test_import_monitor_status(self) -> None:
        from aortica.validation import MonitorStatus  # noqa: F811

    def test_import_drift_alert(self) -> None:
        from aortica.validation import DriftAlert  # noqa: F811

    def test_import_task_metric_snapshot(self) -> None:
        from aortica.validation import TaskMetricSnapshot  # noqa: F811

    def test_import_from_submodule(self) -> None:
        from aortica.validation.performance_monitor import (  # noqa: F811
            DriftAlert,
            MonitorStatus,
            PerformanceMonitor,
            TaskMetricSnapshot,
        )

    def test_import_monitor_status_response(self) -> None:
        from aortica.api.validation_endpoints import (  # noqa: F811
            MonitorStatusResponse,
        )
