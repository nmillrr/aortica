"""Tests for `aortica audit` CLI (US-121)."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from aortica.audit import AuditLogger
from aortica.cli.audit_cmd import audit_group


def _seed(tmp_path: Path) -> str:
    db = str(tmp_path / "audit.db")
    logger = AuditLogger(db, hmac_key="k")
    logger.log("ecg_ingested", ecg_reference_id="e1", user_id="u1")
    logger.log("prediction_generated", ecg_reference_id="e1", model_version="0.2.0")
    logger.close()
    return db


def test_export_csv(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    result = CliRunner().invoke(
        audit_group, ["export", "--db", db, "--format", "csv", "--hmac-key", "k"]
    )
    assert result.exit_code == 0, result.output
    rows = [r for r in csv.reader(io.StringIO(result.output)) if r]
    assert rows[0][:3] == ["id", "timestamp", "event_type"]
    assert len(rows) == 3  # header + 2 events


def test_export_json(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    result = CliRunner().invoke(
        audit_group, ["export", "--db", db, "--format", "json", "--hmac-key", "k"]
    )
    assert result.exit_code == 0
    events = json.loads(result.output)
    assert len(events) == 2
    assert events[0]["event_type"] == "ecg_ingested"  # oldest first


def test_export_to_file(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    out = tmp_path / "export.json"
    result = CliRunner().invoke(
        audit_group,
        ["export", "--db", db, "--format", "json", "--output", str(out), "--hmac-key", "k"],
    )
    assert result.exit_code == 0
    assert "Exported 2 event" in result.output
    assert out.exists()


def test_verify_valid(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    result = CliRunner().invoke(audit_group, ["verify", "--db", db, "--hmac-key", "k"])
    assert result.exit_code == 0
    assert json.loads(result.output)["valid"] is True


def test_verify_tampered_exits_nonzero(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit_events SET user_id='HACK' WHERE id=1")
    conn.commit()
    conn.close()
    result = CliRunner().invoke(audit_group, ["verify", "--db", db, "--hmac-key", "k"])
    assert result.exit_code == 1
    assert json.loads(result.output)["valid"] is False
