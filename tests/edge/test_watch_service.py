"""Tests for the edge inbox service behind ``aortica-edge.service``."""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from aortica.edge.watch_service import EdgeScorer, run_watch_loop, sd_notify  # noqa: E402
from aortica.models.output_gating import default_trained_outputs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "releases" / "v0.3.0" / "aortica_edge_int8_v0.3.0.onnx"
FIXTURES = REPO / "tests" / "fixtures" / "ptbxl" / "records500"


def _drop(inbox: Path, record_id: str) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    for ext in (".hea", ".dat"):
        shutil.copy(FIXTURES / f"{record_id}_hr{ext}", inbox / f"{record_id}_hr{ext}")


@pytest.fixture()
def scorer() -> EdgeScorer:
    return EdgeScorer(MODEL, unload_after_seconds=0)


def test_scores_af_record_end_to_end(tmp_path: Path, scorer: EdgeScorer) -> None:
    inbox, outbox = tmp_path / "inbox", tmp_path / "results"
    _drop(inbox, "00318")  # PTB-XL test fold, labelled AFIB

    n = run_watch_loop(scorer, inbox, outbox, max_cycles=1, sleep=lambda _: None)

    assert n == 1
    result = json.loads((outbox / "00318_hr.json").read_text())
    findings = {f["finding"] for f in result["positive_findings"]}
    assert "AF" in findings
    assert result["urgency_tier"] in ("high", "critical")
    # Only trained outputs are reported; the risk head is never among them.
    reported = {n for task in result["predictions"].values() for n in task}
    assert reported <= default_trained_outputs()
    assert "risk" not in result["predictions"]
    assert "mortality_1y" in result["suppressed_classes"]
    assert result["quality"]["classification"] in ("good", "marginal", "poor")


def test_unreadable_file_does_not_stop_service(tmp_path: Path, scorer: EdgeScorer) -> None:
    inbox, outbox = tmp_path / "inbox", tmp_path / "results"
    inbox.mkdir()
    (inbox / "broken.hea").write_text("not a header\n")
    _drop(inbox, "00009")

    run_watch_loop(scorer, inbox, outbox, max_cycles=1, sleep=lambda _: None)

    assert (outbox / "00009_hr.json").exists()


def test_scored_files_move_to_processed(tmp_path: Path, scorer: EdgeScorer) -> None:
    inbox, outbox, done = tmp_path / "inbox", tmp_path / "results", tmp_path / "done"
    _drop(inbox, "00180")

    run_watch_loop(scorer, inbox, outbox, processed=done, max_cycles=2,
                   sleep=lambda _: None)

    assert not list(inbox.iterdir())
    assert sorted(p.name for p in done.iterdir()) == ["00180_hr.dat", "00180_hr.hea"]
    assert len(list(outbox.iterdir())) == 1  # scored once, not re-scored on cycle 2


def test_model_released_when_idle(tmp_path: Path, scorer: EdgeScorer) -> None:
    _drop(tmp_path / "inbox", "00009")
    run_watch_loop(scorer, tmp_path / "inbox", tmp_path / "out", max_cycles=1,
                   sleep=lambda _: None)
    assert not scorer.is_loaded


def test_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EdgeScorer(tmp_path / "nope.onnx")


def test_sd_notify_without_systemd_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sd_notify("READY=1") is False


def test_sd_notify_speaks_the_systemd_protocol(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch,
                                               scorer: EdgeScorer) -> None:
    path = Path("/tmp") / f"aortica-notify-{tmp_path.name[-12:]}.sock"  # AF_UNIX path limit
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind(str(path))
        sock.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", str(path))
        run_watch_loop(scorer, tmp_path / "inbox", tmp_path / "out", max_cycles=1,
                       sleep=lambda _: None)
        assert [sock.recv(64), sock.recv(64)] == [b"READY=1", b"WATCHDOG=1"]
    finally:
        sock.close()
        path.unlink(missing_ok=True)


def test_cli_watch_command(tmp_path: Path) -> None:
    click = pytest.importorskip("click")  # noqa: F841
    from click.testing import CliRunner

    from aortica.cli.edge_cmd import edge_group

    _drop(tmp_path / "inbox", "00172")
    result = CliRunner().invoke(edge_group, [
        "watch", "--model", str(MODEL), "--inbox", str(tmp_path / "inbox"),
        "--outbox", str(tmp_path / "out"), "--max-cycles", "1", "--interval", "0",
    ])
    assert result.exit_code == 0, result.output
    assert "Processed 1 ECG(s)." in result.output
    data = json.loads((tmp_path / "out" / "00172_hr.json").read_text())
    assert "RBBB" in {f["finding"] for f in data["positive_findings"]}
