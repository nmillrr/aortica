"""Tests for the central edge-sync aggregator (US-128)."""

from __future__ import annotations

from typing import Any, Dict, List

from aortica.sync.central_aggregator import CentralAggregator


def _result(ecg_id: str, stemi: float = 0.1, quality: float = 80.0, **extra) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "ecg_id": ecg_id,
        "predictions": {"ischaemia": {"STEMI": stemi}, "rhythm": {"AF": 0.1}},
        "quality_score": quality,
    }
    r.update(extra)
    return r


class _RecordingMonitor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record_prediction(self, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return 1


# ─────────────────────────────────────────────────────────────────
# Ingestion + tagging
# ─────────────────────────────────────────────────────────────────


class TestIngestion:
    def test_receive_batch_counts(self) -> None:
        agg = CentralAggregator()
        summary = agg.receive_batch(
            "dev1", "site_a", [_result("e1"), _result("e2")]
        )
        assert summary["received"] == 2

    def test_per_site_tagging(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [_result("e1")])
        agg.receive_batch("dev2", "site_b", [_result("e2")])
        sites = {s.site_id for s in agg.site_metrics()}
        assert sites == {"site_a", "site_b"}

    def test_labeled_forwarded_to_monitor(self) -> None:
        mon = _RecordingMonitor()
        agg = CentralAggregator(monitor=mon)
        agg.receive_batch(
            "dev1", "site_a",
            [_result("e1", ground_truth={"ischaemia": {"STEMI": 1}, "rhythm": {"AF": 0}})],
        )
        # Two tasks with ground-truth → two monitor calls.
        assert len(mon.calls) == 2

    def test_unlabeled_not_forwarded(self) -> None:
        mon = _RecordingMonitor()
        agg = CentralAggregator(monitor=mon)
        agg.receive_batch("dev1", "site_a", [_result("e1")])
        assert mon.calls == []


# ─────────────────────────────────────────────────────────────────
# Site metrics
# ─────────────────────────────────────────────────────────────────


class TestSiteMetrics:
    def test_metrics_aggregate(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [
            _result("e1", stemi=0.9, quality=90),
            _result("e2", stemi=0.1, quality=70),
        ])
        metrics = {m.site_id: m for m in agg.site_metrics()}
        a = metrics["site_a"]
        assert a.total_ecgs == 2
        assert abs(a.mean_quality - 80.0) < 1e-6
        assert a.critical_rate == 0.5  # one STEMI-positive ECG of two
        assert a.finding_distribution.get("STEMI") == 1

    def test_multiple_devices_per_site(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [_result("e1")])
        agg.receive_batch("dev2", "site_a", [_result("e2")])
        m = agg.site_metrics()[0]
        assert set(m.device_ids) == {"dev1", "dev2"}


# ─────────────────────────────────────────────────────────────────
# Anomaly detection
# ─────────────────────────────────────────────────────────────────


class TestAnomalies:
    def test_no_anomalies_when_uniform(self) -> None:
        agg = CentralAggregator()
        for site in ("a", "b", "c", "d"):
            agg.receive_batch(f"dev_{site}", f"site_{site}",
                              [_result(f"{site}1", quality=80)])
        assert agg.detect_anomalies() == []

    def test_quality_outlier_flagged(self) -> None:
        agg = CentralAggregator(anomaly_z_threshold=1.5)
        # Four normal sites + one very low-quality outlier.
        for site in ("a", "b", "c", "d"):
            agg.receive_batch(f"dev_{site}", f"site_{site}",
                              [_result(f"{site}1", quality=85)])
        agg.receive_batch("dev_x", "site_x", [_result("x1", quality=10)])
        anomalies = agg.detect_anomalies()
        assert any(
            a.site_id == "site_x" and a.metric == "mean_quality"
            for a in anomalies
        )

    def test_too_few_sites_no_anomaly(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [_result("e1", quality=10)])
        agg.receive_batch("dev2", "site_b", [_result("e2", quality=90)])
        assert agg.detect_anomalies() == []  # < 3 sites


# ─────────────────────────────────────────────────────────────────
# Reconciliation
# ─────────────────────────────────────────────────────────────────


class TestReconciliation:
    def test_gap_detected(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [_result("e1"), _result("e2")])
        recon = agg.reconcile("dev1", expected_count=5)
        assert recon["received"] == 2
        assert recon["gap"] == 3
        assert recon["complete"] is False

    def test_complete_when_all_received(self) -> None:
        agg = CentralAggregator()
        agg.receive_batch("dev1", "site_a", [_result("e1"), _result("e2")])
        recon = agg.reconcile("dev1", expected_count=2)
        assert recon["gap"] == 0
        assert recon["complete"] is True
