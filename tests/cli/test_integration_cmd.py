"""Tests for `aortica integration` CLI (US-125)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from aortica.cli.integration_cmd import integration_group


def _cfg(tmp_path: Path) -> str:
    p = tmp_path / "integration.yaml"
    p.write_text(
        "enabled_channels:\n  - storage\n  - ehr\n  - worklist\n"
        "max_retries: 3\n"
    )
    return str(p)


def test_run_dry_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        integration_group, ["run", "--config", _cfg(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Dry run OK" in result.output
    assert "storage" in result.output


def test_status_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        integration_group, ["status", "--config", _cfg(tmp_path)]
    )
    assert result.exit_code == 0
    assert "enabled_channels" in result.output


def test_run_requires_dirs_without_dry_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        integration_group, ["run", "--config", _cfg(tmp_path)]
    )
    assert result.exit_code != 0
    assert "watch-dir" in result.output.lower()
