"""Tests for aortica.sync.sync_engine — SyncEngine with vector clocks."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any
from unittest.mock import patch

import pytest

cryptography = pytest.importorskip("cryptography")

from aortica.sync.result_store import ResultStore  # noqa: E402
from aortica.sync.sync_engine import (  # noqa: E402
    ConflictRecord,
    SyncEngine,
    SyncQueueEntry,
    SyncReport,
    VectorClock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path: Any) -> ResultStore:
    return ResultStore(str(tmp_path / "store"))


@pytest.fixture()
def engine(tmp_store: ResultStore) -> SyncEngine:
    return SyncEngine(tmp_store, device_id="dev-A")


@pytest.fixture()
def stored_id(tmp_store: ResultStore) -> int:
    return tmp_store.store_result("hash1", {"rhythm": [0.9]}, {"overall": 85})


# ---------------------------------------------------------------------------
# VectorClock
# ---------------------------------------------------------------------------

class TestVectorClock:
    def test_empty(self) -> None:
        vc = VectorClock()
        assert vc.clock == {}

    def test_increment(self) -> None:
        vc = VectorClock()
        vc.increment("A")
        assert vc.clock == {"A": 1}
        vc.increment("A")
        assert vc.clock == {"A": 2}

    def test_increment_multiple_devices(self) -> None:
        vc = VectorClock()
        vc.increment("A")
        vc.increment("B")
        assert vc.clock == {"A": 1, "B": 1}

    def test_merge(self) -> None:
        a = VectorClock(clock={"A": 3, "B": 1})
        b = VectorClock(clock={"A": 1, "B": 5, "C": 2})
        merged = a.merge(b)
        assert merged.clock == {"A": 3, "B": 5, "C": 2}

    def test_dominates_true(self) -> None:
        a = VectorClock(clock={"A": 3, "B": 2})
        b = VectorClock(clock={"A": 1, "B": 2})
        assert a.dominates(b) is True
        assert b.dominates(a) is False

    def test_dominates_equal(self) -> None:
        a = VectorClock(clock={"A": 1})
        b = VectorClock(clock={"A": 1})
        assert a.dominates(b) is False

    def test_concurrent(self) -> None:
        a = VectorClock(clock={"A": 2, "B": 1})
        b = VectorClock(clock={"A": 1, "B": 2})
        assert a.concurrent_with(b) is True

    def test_not_concurrent_when_dominated(self) -> None:
        a = VectorClock(clock={"A": 2, "B": 2})
        b = VectorClock(clock={"A": 1, "B": 1})
        assert a.concurrent_with(b) is False

    def test_serialisation_roundtrip(self) -> None:
        vc = VectorClock(clock={"A": 5, "B": 3})
        d = vc.to_dict()
        restored = VectorClock.from_dict(d)
        assert restored == vc

    def test_equality(self) -> None:
        a = VectorClock(clock={"A": 1})
        b = VectorClock(clock={"A": 1})
        assert a == b

    def test_equality_not_vc(self) -> None:
        assert VectorClock() != "not a vc"


# ---------------------------------------------------------------------------
# SyncReport
# ---------------------------------------------------------------------------

class TestSyncReport:
    def test_defaults(self) -> None:
        r = SyncReport()
        assert r.uploaded == 0 and r.downloaded == 0 and r.conflicts == 0
        assert r.success is True

    def test_errors_flag(self) -> None:
        r = SyncReport(errors=["oops"])
        assert r.success is False


# ---------------------------------------------------------------------------
# SyncQueueEntry / ConflictRecord
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_sync_queue_entry(self) -> None:
        e = SyncQueueEntry(1, 2, "h", "{}", "{}", "{}", 1.0, "{}", "pending", 1.0)
        assert e.status == "pending"

    def test_conflict_record(self) -> None:
        c = ConflictRecord(1, "h", "{}", "{}", "{}", "{}", "local_wins", 1.0)
        assert c.resolution == "local_wins"


# ---------------------------------------------------------------------------
# SyncEngine — initialisation
# ---------------------------------------------------------------------------

class TestSyncEngineInit:
    def test_device_id(self, engine: SyncEngine) -> None:
        assert engine.device_id == "dev-A"

    def test_initial_clock_empty(self, engine: SyncEngine) -> None:
        assert engine.vector_clock.clock == {}

    def test_tables_created(self, engine: SyncEngine) -> None:
        tables = engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "sync_queue" in names
        assert "conflict_archive" in names
        assert "device_state" in names


# ---------------------------------------------------------------------------
# queue_for_sync
# ---------------------------------------------------------------------------

class TestQueueForSync:
    def test_returns_queue_id(self, engine: SyncEngine, stored_id: int) -> None:
        qid = engine.queue_for_sync(stored_id)
        assert isinstance(qid, int) and qid > 0

    def test_increments_clock(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        assert engine.vector_clock.clock["dev-A"] == 1

    def test_pending_count(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        assert engine.pending_count() == 1

    def test_list_pending(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        items = engine.list_pending()
        assert len(items) == 1
        assert items[0].ecg_hash == "hash1"
        assert items[0].status == "pending"

    def test_invalid_result_id(self, engine: SyncEngine) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.queue_for_sync(9999)

    def test_multiple_queues(self, engine: SyncEngine, tmp_store: ResultStore) -> None:
        id1 = tmp_store.store_result("h1", {"a": 1}, {"q": 1})
        id2 = tmp_store.store_result("h2", {"b": 2}, {"q": 2})
        engine.queue_for_sync(id1)
        engine.queue_for_sync(id2)
        assert engine.pending_count() == 2
        assert engine.vector_clock.clock["dev-A"] == 2

    def test_clock_persisted(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        # Reload clock from DB
        reloaded = engine._load_vector_clock()
        assert reloaded.clock["dev-A"] == 1

    def test_queue_entry_has_predictions(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        items = engine.list_pending()
        preds = json.loads(items[0].predictions_json)
        assert preds == {"rhythm": [0.9]}


# ---------------------------------------------------------------------------
# Mock HTTP server for sync tests
# ---------------------------------------------------------------------------

class _MockSyncHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for sync push/pull testing."""

    push_received: list[dict[str, Any]] = []
    pull_response: dict[str, Any] = {"results": []}
    fail_next: bool = False

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode()) if length else {}

        if _MockSyncHandler.fail_next:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"fail"}')
            return

        if self.path == "/push":
            _MockSyncHandler.push_received.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/pull":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(_MockSyncHandler.pull_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass  # Suppress output


@pytest.fixture()
def mock_server():
    _MockSyncHandler.push_received = []
    _MockSyncHandler.pull_response = {"results": []}
    _MockSyncHandler.fail_next = False
    server = HTTPServer(("127.0.0.1", 0), _MockSyncHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# sync_to_remote
# ---------------------------------------------------------------------------

class TestSyncToRemote:
    def test_uploads_pending(self, engine: SyncEngine, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        report = engine.sync_to_remote(mock_server)
        assert report.uploaded == 1
        assert report.success is True

    def test_marks_synced(self, engine: SyncEngine, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        engine.sync_to_remote(mock_server)
        assert engine.pending_count() == 0

    def test_marks_result_synced(self, engine: SyncEngine, stored_id: int, tmp_store: ResultStore, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        engine.sync_to_remote(mock_server)
        result = tmp_store.get_result_by_id(stored_id)
        assert result is not None and result.synced is True

    def test_push_payload(self, engine: SyncEngine, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        engine.sync_to_remote(mock_server)
        assert len(_MockSyncHandler.push_received) == 1
        p = _MockSyncHandler.push_received[0]
        assert p["device_id"] == "dev-A"
        assert p["ecg_hash"] == "hash1"
        assert p["predictions"] == {"rhythm": [0.9]}

    def test_no_pending(self, engine: SyncEngine, mock_server: str) -> None:
        report = engine.sync_to_remote(mock_server)
        assert report.uploaded == 0 and report.success is True

    def test_server_error(self, engine: SyncEngine, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        _MockSyncHandler.fail_next = True
        report = engine.sync_to_remote(mock_server)
        assert report.uploaded == 0
        assert len(report.errors) == 1
        assert engine.pending_count() == 1  # still pending

    def test_unreachable_server(self, engine: SyncEngine, stored_id: int) -> None:
        engine.queue_for_sync(stored_id)
        report = engine.sync_to_remote("http://127.0.0.1:1")
        assert report.uploaded == 0
        assert len(report.errors) == 1


# ---------------------------------------------------------------------------
# pull_from_remote
# ---------------------------------------------------------------------------

class TestPullFromRemote:
    def test_downloads_new(self, engine: SyncEngine, mock_server: str) -> None:
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "new1", "predictions": {"r": [0.5]}, "quality": {"o": 90},
             "metadata": {}, "timestamp": 100.0, "vector_clock": {"remote": 1}},
        ]}
        report = engine.pull_from_remote(mock_server)
        assert report.downloaded == 1 and report.success

    def test_empty_pull(self, engine: SyncEngine, mock_server: str) -> None:
        report = engine.pull_from_remote(mock_server)
        assert report.downloaded == 0 and report.success

    def test_remote_dominates_overwrites(self, engine: SyncEngine, tmp_store: ResultStore, stored_id: int, mock_server: str) -> None:
        # Queue local so it has a clock
        engine.queue_for_sync(stored_id)
        # Remote has higher clock
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "hash1", "predictions": {"updated": True}, "quality": {"o": 95},
             "metadata": {}, "timestamp": 200.0, "vector_clock": {"dev-A": 5, "remote": 2}},
        ]}
        report = engine.pull_from_remote(mock_server)
        assert report.downloaded == 1
        result = tmp_store.get_result("hash1")
        assert result is not None and result.predictions == {"updated": True}

    def test_local_dominates_keeps_local(self, engine: SyncEngine, tmp_store: ResultStore, stored_id: int, mock_server: str) -> None:
        # Queue multiple times so local clock is high
        engine.queue_for_sync(stored_id)
        engine.queue_for_sync(stored_id)
        engine.queue_for_sync(stored_id)
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "hash1", "predictions": {"old": True}, "quality": {},
             "metadata": {}, "timestamp": 50.0, "vector_clock": {"dev-A": 1}},
        ]}
        report = engine.pull_from_remote(mock_server)
        assert report.downloaded == 0
        result = tmp_store.get_result("hash1")
        assert result is not None and result.predictions == {"rhythm": [0.9]}

    def test_concurrent_conflict_remote_wins(self, engine: SyncEngine, tmp_store: ResultStore, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        # Concurrent: local has {A:1}, remote has {B:1} — remote has later timestamp
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "hash1", "predictions": {"remote": True}, "quality": {},
             "metadata": {}, "timestamp": time.time() + 100, "vector_clock": {"dev-B": 1}},
        ]}
        report = engine.pull_from_remote(mock_server)
        assert report.conflicts == 1
        assert report.downloaded == 1
        assert engine.conflict_count() == 1

    def test_concurrent_conflict_local_wins(self, engine: SyncEngine, tmp_store: ResultStore, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "hash1", "predictions": {"remote": True}, "quality": {},
             "metadata": {}, "timestamp": 0.001, "vector_clock": {"dev-B": 1}},
        ]}
        report = engine.pull_from_remote(mock_server)
        assert report.conflicts == 1
        assert report.downloaded == 0
        result = tmp_store.get_result("hash1")
        assert result is not None and result.predictions == {"rhythm": [0.9]}

    def test_merges_clock(self, engine: SyncEngine, mock_server: str) -> None:
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "x", "predictions": {}, "quality": {},
             "metadata": {}, "timestamp": 1.0, "vector_clock": {"remote": 5}},
        ]}
        engine.pull_from_remote(mock_server)
        assert engine.vector_clock.clock.get("remote") == 5

    def test_network_error(self, engine: SyncEngine) -> None:
        report = engine.pull_from_remote("http://127.0.0.1:1")
        assert len(report.errors) == 1 and report.downloaded == 0


# ---------------------------------------------------------------------------
# Conflict archive
# ---------------------------------------------------------------------------

class TestConflictArchive:
    def test_list_conflicts_empty(self, engine: SyncEngine) -> None:
        assert engine.list_conflicts() == []
        assert engine.conflict_count() == 0

    def test_conflict_record_fields(self, engine: SyncEngine, tmp_store: ResultStore, stored_id: int, mock_server: str) -> None:
        engine.queue_for_sync(stored_id)
        _MockSyncHandler.pull_response = {"results": [
            {"ecg_hash": "hash1", "predictions": {"r": True}, "quality": {},
             "metadata": {}, "timestamp": time.time() + 100, "vector_clock": {"dev-B": 1}},
        ]}
        engine.pull_from_remote(mock_server)
        conflicts = engine.list_conflicts()
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.ecg_hash == "hash1"
        assert c.resolution == "remote_wins"
        assert isinstance(c.resolved_at, float)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_package_exports(self) -> None:
        from aortica.sync import SyncEngine, VectorClock, SyncReport, ConflictRecord, SyncQueueEntry
        assert all(c is not None for c in [SyncEngine, VectorClock, SyncReport, ConflictRecord, SyncQueueEntry])

    def test_module_import(self) -> None:
        from aortica.sync import sync_engine
        assert hasattr(sync_engine, "SyncEngine")
