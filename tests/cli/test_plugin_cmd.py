"""Tests for `aortica plugin` CLI (US-118)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from aortica.cli.plugin_cmd import plugin_group


def test_plugin_list() -> None:
    result = CliRunner().invoke(plugin_group, ["list"])
    assert result.exit_code == 0
    assert "file_watcher" in result.output
    assert "muse" in result.output
    assert "fhir" in result.output


def test_plugin_run_dry_run(tmp_path: Path) -> None:
    cfg = tmp_path / "plugins.yaml"
    cfg.write_text(
        "plugins:\n"
        "  - name: watcher\n"
        "    type: file_watcher\n"
        "    config:\n"
        "      watch_dir: /data/in\n"
        "      output_dir: /data/out\n"
    )
    result = CliRunner().invoke(
        plugin_group, ["run", "--config", str(cfg), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Dry run OK" in result.output


def test_plugin_run_unknown_type_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "plugins.yaml"
    cfg.write_text(
        "plugins:\n  - name: x\n    type: not_a_real_plugin\n    config: {}\n"
    )
    result = CliRunner().invoke(
        plugin_group, ["run", "--config", str(cfg), "--dry-run"]
    )
    assert result.exit_code == 1
