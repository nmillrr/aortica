"""Plugin daemon and YAML configuration (US-118).

Loads ``plugins.yaml``, instantiates the configured plugins from the
registry, and runs a long-lived poll → infer → submit loop.  The daemon
takes an injectable *processor* so the inference backend (or a test stub)
is decoupled from the polling machinery.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aortica.integration.plugins.base import ECGSystemPlugin, Processor
from aortica.integration.plugins.registry import get_plugin

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """One plugin entry from ``plugins.yaml``.

    Attributes:
        name: Instance name (for logging/identification).
        plugin_type: Registry key (``file_watcher``, ``muse``, ``fhir``).
        config: Per-plugin connection parameters.
        poll_interval: Seconds between poll cycles.
    """

    name: str
    plugin_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    poll_interval: float = 30.0


def load_plugins_config(path: str | Path) -> List[PluginConfig]:
    """Load a ``plugins.yaml`` file into a list of :class:`PluginConfig`.

    Expected structure::

        plugins:
          - name: muse-prod
            type: muse
            poll_interval: 60
            config:
              remote_ae: MUSE
              remote_host: 10.0.0.5
              remote_port: 104
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    text = path_obj.read_text()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    except ImportError:
        from aortica.federated.fl_server import _simple_yaml_load

        data = _simple_yaml_load(text)

    if not isinstance(data, dict) or "plugins" not in data:
        raise ValueError("plugins.yaml must contain a top-level 'plugins' list")

    configs: List[PluginConfig] = []
    for entry in data["plugins"]:
        if not isinstance(entry, dict):
            continue
        plugin_type = entry.get("type") or entry.get("plugin_type")
        if not plugin_type:
            raise ValueError(f"plugin entry missing 'type': {entry}")
        configs.append(
            PluginConfig(
                name=str(entry.get("name", plugin_type)),
                plugin_type=str(plugin_type),
                config=entry.get("config", {}) or {},
                poll_interval=float(entry.get("poll_interval", 30.0)),
            )
        )
    return configs


class PluginDaemon:
    """Runs one or more plugins in a poll loop.

    Args:
        processor: Callable mapping a polled ECG payload to a result dict.
        sleep: Sleep function (injectable for tests).
    """

    def __init__(
        self,
        processor: Processor,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._processor = processor
        self._sleep = sleep
        self._plugins: List[tuple[PluginConfig, ECGSystemPlugin]] = []
        self._running = False

    def add_from_configs(self, configs: List[PluginConfig]) -> None:
        """Instantiate and connect plugins from *configs*."""
        for cfg in configs:
            plugin_cls = get_plugin(cfg.plugin_type)
            plugin = plugin_cls()
            plugin.connect(cfg.config)
            self._plugins.append((cfg, plugin))

    def add_plugin(self, cfg: PluginConfig, plugin: ECGSystemPlugin) -> None:
        """Add an already-connected plugin instance."""
        self._plugins.append((cfg, plugin))

    @property
    def plugins(self) -> List[ECGSystemPlugin]:
        return [p for _, p in self._plugins]

    def run_cycle(self) -> Dict[str, Any]:
        """Run one poll cycle across every plugin; return a summary."""
        summary: Dict[str, Any] = {}
        for cfg, plugin in self._plugins:
            report = plugin.process_once(self._processor)
            summary[cfg.name] = {
                "polled": report.polled,
                "processed": len(report.processed),
                "critical": report.critical_count,
                "errors": len(report.errors),
            }
        return summary

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Run the poll loop.

        Args:
            max_cycles: Stop after this many cycles (``None`` = run until
                :meth:`stop` is called).  Between cycles the daemon sleeps
                for the smallest configured poll interval.
        """
        self._running = True
        cycles = 0
        interval = min(
            (cfg.poll_interval for cfg, _ in self._plugins), default=30.0
        )
        while self._running:
            self.run_cycle()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self._sleep(interval)
        self._running = False

    def stop(self) -> None:
        """Signal the run loop to stop after the current cycle."""
        self._running = False
