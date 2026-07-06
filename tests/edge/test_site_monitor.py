"""Tests for aortica.edge.site_monitor — edge site monitoring (US-061b)."""

from __future__ import annotations

import tempfile
import threading

import pytest

from aortica.edge.site_monitor import SiteMonitor, SiteStatus


class _FakeSync:
    def __init__(self, pending: int) -> None:
        self._pending = pending

    def pending_count(self) -> int:
        return self._pending


@pytest.fixture()
def data_dir() -> str:
    return tempfile.mkdtemp(prefix="aortica-site-")


def test_records_and_counts_daily(data_dir: str) -> None:
    now = 1_000_000.0
    with SiteMonitor(data_dir) as mon:
        for _ in range(5):
            mon.record_inference(success=True, timestamp=now - 100)
        mon.record_inference(success=False, error_type="read", timestamp=now - 50)
        # Older than 24h — excluded from daily counts.
        mon.record_inference(success=True, timestamp=now - 90_000)

        assert mon.daily_inference_count(now=now) == 6
        assert mon.daily_error_count(now=now) == 1
        assert mon.total_inference_count() == 7


def test_error_rate(data_dir: str) -> None:
    now = 2_000_000.0
    with SiteMonitor(data_dir) as mon:
        assert mon.error_rate(now=now) == 0.0  # no inferences yet
        for _ in range(3):
            mon.record_inference(success=True, timestamp=now)
        mon.record_inference(success=False, timestamp=now)
        assert mon.error_rate(now=now) == pytest.approx(0.25)


def test_sync_status_unknown_without_engine(data_dir: str) -> None:
    with SiteMonitor(data_dir) as mon:
        status, pending = mon.sync_status()
        assert status == "unknown"
        assert pending == 0


def test_sync_status_synced_and_pending(data_dir: str) -> None:
    with SiteMonitor(data_dir, sync_engine=_FakeSync(0)) as mon:
        assert mon.sync_status() == ("synced", 0)
    with SiteMonitor(data_dir, sync_engine=_FakeSync(4)) as mon:
        assert mon.sync_status() == ("pending", 4)


def test_last_sync_timestamp(data_dir: str) -> None:
    with SiteMonitor(data_dir) as mon:
        assert mon.last_sync_timestamp() is None
        mon.record_sync(timestamp=123456.0)
        assert mon.last_sync_timestamp() == 123456.0
        mon.record_sync(timestamp=222222.0)  # upsert overwrites
        assert mon.last_sync_timestamp() == 222222.0


def test_storage_utilization(data_dir: str) -> None:
    with SiteMonitor(data_dir) as mon:
        total, used, free, pct = mon.storage_utilization()
        assert total > 0
        assert 0.0 <= pct <= 100.0
        assert used + free <= total + 1  # rounding tolerance


def test_status_snapshot(data_dir: str) -> None:
    now = 3_000_000.0
    with SiteMonitor(data_dir, site_id="clinic-a", sync_engine=_FakeSync(2)) as mon:
        mon.record_inference(success=True, timestamp=now)
        mon.record_sync(timestamp=now - 5)
        status = mon.status(now=now)
        assert isinstance(status, SiteStatus)
        assert status.site_id == "clinic-a"
        assert status.daily_inference_count == 1
        assert status.sync_status == "pending"
        assert status.pending_sync_count == 2
        assert status.last_sync_timestamp == now - 5
        d = status.to_dict()
        assert d["site_id"] == "clinic-a"
        assert "storage_utilization_pct" in d
        assert "clinic-a" in status.summary()


def test_daily_report(data_dir: str) -> None:
    now = 4_000_000.0
    with SiteMonitor(data_dir, site_id="clinic-b") as mon:
        mon.record_inference(success=True, timestamp=now)
        report = mon.daily_report(now=now)
        assert report["site_id"] == "clinic-b"
        assert report["report_generated_at"] == now
        assert "report_date" in report
        assert report["daily_inference_count"] == 1


def test_persists_across_reopen(data_dir: str) -> None:
    with SiteMonitor(data_dir) as mon:
        mon.record_inference(success=True)
    # Re-open the same directory — event survives.
    with SiteMonitor(data_dir) as mon2:
        assert mon2.total_inference_count() == 1


def test_thread_safe_recording(data_dir: str) -> None:
    mon = SiteMonitor(data_dir)

    def worker() -> None:
        for _ in range(50):
            mon.record_inference(success=True)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mon.total_inference_count() == 200
    mon.close()
