"""Tests for ``aortica edge site-report`` CLI command (US-061b)."""

from __future__ import annotations

import json

import pytest

click = pytest.importorskip("click")

from click.testing import CliRunner  # noqa: E402

from aortica.cli import _build_cli  # noqa: E402
from aortica.cli.edge_cmd import edge_group  # noqa: E402


def test_edge_group_registered_in_cli() -> None:
    cli = _build_cli()
    result = CliRunner().invoke(cli, ["edge", "--help"])
    assert result.exit_code == 0
    assert "site-report" in result.output


def test_site_report_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    result = runner.invoke(
        edge_group,
        ["site-report", "--data-dir", str(tmp_path), "--site-id", "clinic-x"],
    )
    assert result.exit_code == 0
    assert "clinic-x" in result.output
    assert "Sync status" in result.output


def test_site_report_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    result = runner.invoke(
        edge_group,
        [
            "site-report",
            "--data-dir",
            str(tmp_path),
            "--site-id",
            "clinic-y",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["site_id"] == "clinic-y"
    assert data["daily_inference_count"] == 0
    assert "report_date" in data
    assert "storage_utilization_pct" in data


def test_site_report_reflects_recorded_events(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from aortica.edge.site_monitor import SiteMonitor

    # Seed the site DB in this directory, then read it back via the CLI.
    with SiteMonitor(str(tmp_path), site_id="seeded") as mon:
        mon.record_inference(success=True)
        mon.record_inference(success=True)

    result = CliRunner().invoke(
        edge_group,
        ["site-report", "--data-dir", str(tmp_path), "--format", "json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["daily_inference_count"] == 2
