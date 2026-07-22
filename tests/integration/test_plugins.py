"""Tests for the ECG management system plugin architecture (US-118)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from aortica.integration.plugins import (  # noqa: E402
    ECGSystemPlugin,
    FileWatcherPlugin,
    PluginConfig,
    PluginDaemon,
    default_critical_detector,
    get_plugin,
    list_plugins,
    load_plugins_config,
    register_plugin,
    unregister_plugin,
)
from aortica.integration.plugins.base import (  # noqa: E402
    ON_CRITICAL_FINDING,
    ON_ECG_RECEIVED,
    PluginHealth,
)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _write_wfdb(directory: Path, name: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    sig = rng.normal(0, 1, (500, 2)).astype(np.float64)
    wfdb.wrsamp(
        name, fs=250, units=["mV", "mV"], sig_name=["I", "II"],
        p_signal=sig, write_dir=str(directory),
    )


def _stub_processor(critical: bool = False) -> Any:
    def _proc(payload: Any) -> Dict[str, Any]:
        if critical:
            return {"ischaemia": {"STEMI": 0.95}}
        return {"rhythm": {"normal_sinus_rhythm": 0.99}}
    return _proc


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_builtin_plugins_registered(self) -> None:
        names = list_plugins()
        assert {"file_watcher", "muse", "fhir"} <= set(names)

    def test_get_plugin(self) -> None:
        assert get_plugin("file_watcher") is FileWatcherPlugin

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_plugin("nonexistent")

    def test_register_and_unregister(self) -> None:
        class MyPlugin(FileWatcherPlugin):
            name = "custom_test"

        register_plugin("custom_test", MyPlugin)
        assert get_plugin("custom_test") is MyPlugin
        unregister_plugin("custom_test")
        with pytest.raises(KeyError):
            get_plugin("custom_test")

    def test_register_rejects_non_plugin(self) -> None:
        with pytest.raises(TypeError):
            register_plugin("bad", dict)  # type: ignore[arg-type]

    def test_register_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            register_plugin("", FileWatcherPlugin)


# ─────────────────────────────────────────────────────────────────
# Base contract + hooks
# ─────────────────────────────────────────────────────────────────


class TestBaseContract:
    def test_file_watcher_is_plugin(self) -> None:
        assert issubclass(FileWatcherPlugin, ECGSystemPlugin)

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            ECGSystemPlugin()  # type: ignore[abstract]

    def test_hooks_fire(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "rec0")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})

        received: List[str] = []
        criticals: List[str] = []
        plugin.register_hook(ON_ECG_RECEIVED, lambda eid, p: received.append(eid))
        plugin.register_hook(ON_CRITICAL_FINDING, lambda eid, r: criticals.append(eid))

        report = plugin.process_once(_stub_processor(critical=True))
        assert report.polled == 1
        assert received == ["rec0"]
        assert criticals == ["rec0"]
        assert report.critical_count == 1

    def test_unknown_hook_raises(self) -> None:
        plugin = FileWatcherPlugin()
        with pytest.raises(ValueError):
            plugin.register_hook("on_bogus", lambda: None)

    def test_hook_error_does_not_break_cycle(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "rec0")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})

        def _boom(*_a: Any) -> None:
            raise RuntimeError("hook failure")

        plugin.register_hook(ON_ECG_RECEIVED, _boom)
        report = plugin.process_once(_stub_processor())
        assert len(report.processed) == 1  # still processed despite hook error


# ─────────────────────────────────────────────────────────────────
# Critical detector
# ─────────────────────────────────────────────────────────────────


class TestCriticalDetector:
    def test_detects_stemi(self) -> None:
        assert default_critical_detector({"ischaemia": {"STEMI": 0.9}})

    def test_ignores_low_confidence(self) -> None:
        assert not default_critical_detector({"ischaemia": {"STEMI": 0.2}})

    def test_ignores_non_critical(self) -> None:
        assert not default_critical_detector({"rhythm": {"PAC": 0.99}})


# ─────────────────────────────────────────────────────────────────
# FileWatcherPlugin
# ─────────────────────────────────────────────────────────────────


class TestFileWatcher:
    def test_requires_dirs(self) -> None:
        with pytest.raises(ValueError):
            FileWatcherPlugin().connect({"watch_dir": "/x"})

    def test_poll_reads_new_files(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "a")
        _write_wfdb(watch, "b", seed=1)
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})
        polled = plugin.poll_for_ecgs()
        assert {eid for eid, _ in polled} == {"a", "b"}

    def test_poll_does_not_repeat(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "a")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})
        assert len(plugin.poll_for_ecgs()) == 1
        assert plugin.poll_for_ecgs() == []  # already seen

    def test_submit_writes_result_json(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        out = tmp_path / "out"
        _write_wfdb(watch, "a")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(out)})
        plugin.process_once(_stub_processor())
        result_file = out / "a.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert "rhythm" in data

    def test_processed_dir_moves_source(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        processed = tmp_path / "done"
        _write_wfdb(watch, "a")
        plugin = FileWatcherPlugin()
        plugin.connect({
            "watch_dir": str(watch),
            "output_dir": str(tmp_path / "out"),
            "processed_dir": str(processed),
        })
        plugin.process_once(_stub_processor())
        assert (processed / "a.hea").exists()
        assert not (watch / "a.hea").exists()

    def test_get_worklist(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "a")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})
        wl = plugin.get_worklist()
        assert len(wl) == 1
        assert wl[0]["ecg_id"] == "a"

    def test_health_check(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        plugin = FileWatcherPlugin()
        assert plugin.health_check().healthy is False  # not connected
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})
        health = plugin.health_check()
        assert isinstance(health, PluginHealth)
        assert health.healthy is True


# ─────────────────────────────────────────────────────────────────
# Config loading + daemon
# ─────────────────────────────────────────────────────────────────


class TestConfigAndDaemon:
    def test_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "plugins.yaml"
        cfg_file.write_text(
            "plugins:\n"
            "  - name: watcher\n"
            "    type: file_watcher\n"
            "    poll_interval: 5\n"
            "    config:\n"
            "      watch_dir: /data/in\n"
            "      output_dir: /data/out\n"
        )
        configs = load_plugins_config(cfg_file)
        assert len(configs) == 1
        assert configs[0].plugin_type == "file_watcher"
        assert configs[0].poll_interval == 5.0
        assert configs[0].config["watch_dir"] == "/data/in"

    def test_load_config_missing_type_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "plugins.yaml"
        cfg_file.write_text("plugins:\n  - name: x\n    config: {}\n")
        with pytest.raises(ValueError):
            load_plugins_config(cfg_file)

    def test_daemon_runs_cycles(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        _write_wfdb(watch, "a")
        plugin = FileWatcherPlugin()
        plugin.connect({"watch_dir": str(watch), "output_dir": str(tmp_path / "out")})

        daemon = PluginDaemon(_stub_processor(), sleep=lambda _s: None)
        daemon.add_plugin(
            PluginConfig(name="w", plugin_type="file_watcher", poll_interval=1),
            plugin,
        )
        daemon.run(max_cycles=2)
        assert (tmp_path / "out" / "a.json").exists()

    def test_daemon_add_from_configs(self, tmp_path: Path) -> None:
        watch = tmp_path / "in"
        watch.mkdir()
        out = tmp_path / "out"
        cfg = PluginConfig(
            name="w",
            plugin_type="file_watcher",
            config={"watch_dir": str(watch), "output_dir": str(out)},
        )
        daemon = PluginDaemon(_stub_processor(), sleep=lambda _s: None)
        daemon.add_from_configs([cfg])
        assert len(daemon.plugins) == 1
        summary = daemon.run_cycle()
        assert "w" in summary
