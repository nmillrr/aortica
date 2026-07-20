"""Tests for aortica.audit.logger (US-121)."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from pathlib import Path

import pytest

from aortica.audit import (
    EVENT_TYPES,
    AuditEvent,
    AuditLogger,
    verify_integrity,
)
from aortica.audit.logger import IntegrityReport


def _logger(tmp_path: Path) -> AuditLogger:
    return AuditLogger(str(tmp_path / "audit.db"), hmac_key="test-key")


# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────


class TestLogging:
    def test_log_returns_event(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        ev = a.log("ecg_ingested", ecg_reference_id="e1", user_id="u1")
        assert isinstance(ev, AuditEvent)
        assert ev.event_type == "ecg_ingested"
        assert ev.hmac

    def test_unknown_event_type_raises(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        with pytest.raises(ValueError):
            a.log("not_a_real_event")

    def test_all_event_types_accepted(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for et in EVENT_TYPES:
            a.log(et, ecg_reference_id="e1")
        assert a.count() == len(EVENT_TYPES)

    def test_first_row_uses_genesis_prev(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        ev = a.log("model_loaded")
        assert ev.prev_hash == "0" * 64

    def test_chain_links_to_previous(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        e1 = a.log("ecg_ingested", ecg_reference_id="e1")
        e2 = a.log("prediction_generated", ecg_reference_id="e1")
        assert e2.prev_hash == e1.hmac


# ─────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────


class TestQuery:
    def test_filter_by_event_type(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", ecg_reference_id="e1")
        a.log("prediction_generated", ecg_reference_id="e1")
        preds = a.query(event_type="prediction_generated")
        assert len(preds) == 1
        assert preds[0].event_type == "prediction_generated"

    def test_filter_by_user_and_ecg(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", ecg_reference_id="e1", user_id="u1")
        a.log("ecg_ingested", ecg_reference_id="e2", user_id="u2")
        assert len(a.query(user_id="u1")) == 1
        assert len(a.query(ecg_reference_id="e2")) == 1

    def test_filter_by_date_range(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", timestamp="2024-01-01T00:00:00+00:00")
        a.log("ecg_ingested", timestamp="2024-06-01T00:00:00+00:00")
        got = a.query(date_from="2024-03-01", date_to="2024-12-31")
        assert len(got) == 1

    def test_newest_first(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", ecg_reference_id="first")
        a.log("ecg_ingested", ecg_reference_id="second")
        rows = a.query()
        assert rows[0].ecg_reference_id == "second"


# ─────────────────────────────────────────────────────────────────
# Integrity
# ─────────────────────────────────────────────────────────────────


class TestIntegrity:
    def test_intact_chain_valid(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for i in range(5):
            a.log("ecg_ingested", ecg_reference_id=f"e{i}")
        report = a.verify()
        assert isinstance(report, IntegrityReport)
        assert report.valid
        assert report.total_rows == 5
        assert report.broken_links == []

    def test_empty_chain_valid(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        assert a.verify().valid

    def test_tamper_detected_and_cascades(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for i in range(5):
            a.log("ecg_ingested", ecg_reference_id=f"e{i}")
        a.close()
        db = str(tmp_path / "audit.db")
        conn = sqlite3.connect(db)
        conn.execute("UPDATE audit_events SET ecg_reference_id='HACK' WHERE id=2")
        conn.commit()
        conn.close()
        report = verify_integrity(db, hmac_key="test-key")
        assert not report.valid
        # Tampering row 2 cascades through all later rows.
        assert report.broken_links == [2, 3, 4, 5]

    def test_wrong_key_fails_verification(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", ecg_reference_id="e1")
        a.close()
        report = verify_integrity(str(tmp_path / "audit.db"), hmac_key="wrong")
        assert not report.valid

    def test_deleted_row_detected(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for i in range(4):
            a.log("ecg_ingested", ecg_reference_id=f"e{i}")
        a.close()
        db = str(tmp_path / "audit.db")
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM audit_events WHERE id=2")
        conn.commit()
        conn.close()
        # Removing a row breaks the chain for subsequent rows.
        report = verify_integrity(db, hmac_key="test-key")
        assert not report.valid


# ─────────────────────────────────────────────────────────────────
# Rotation
# ─────────────────────────────────────────────────────────────────


class TestRotation:
    def test_no_rotation_below_threshold(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        a.log("ecg_ingested", ecg_reference_id="e1")
        assert a.rotate_if_needed(max_bytes=10_000_000) is None

    def test_rotation_archives_and_truncates(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for i in range(20):
            a.log("ecg_ingested", ecg_reference_id=f"e{i}")
        archive = a.rotate_if_needed(max_bytes=1)  # force rotation
        assert archive is not None
        assert os.path.exists(archive)
        assert a.count() == 0
        # Archive holds the rotated events.
        with gzip.open(archive, "rt", encoding="utf-8") as fh:
            events = json.load(fh)
        assert len(events) == 20

    def test_chain_restarts_after_rotation(self, tmp_path: Path) -> None:
        a = _logger(tmp_path)
        for i in range(5):
            a.log("ecg_ingested", ecg_reference_id=f"e{i}")
        a.rotate_if_needed(max_bytes=1)
        new = a.log("prediction_generated", ecg_reference_id="fresh")
        assert new.prev_hash == "0" * 64
        assert a.verify().valid
