"""``aortica plugin`` — ECG management system plugin daemon (US-118).

\b
Subcommands:
  aortica plugin list                     — list registered plugins
  aortica plugin run --config plugins.yaml  — run the polling daemon
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import click


def _build_default_processor() -> Any:
    """Build a processor that runs Aortica inference on an ECGRecord.

    Lazily loads a model so importing the CLI does not require torch. The
    returned callable maps an ``ECGRecord`` to a ``{task: {class: conf}}``
    dict, mirroring the model-forward path used by the predict endpoint.
    """
    import numpy as np
    import torch

    from aortica.api.predict import _get_task_class_names
    from aortica.models.registry import load_pretrained

    model = load_pretrained("latest")
    model.eval()
    class_names = _get_task_class_names()

    def _processor(payload: Any) -> Dict[str, Any]:
        x = torch.from_numpy(payload.signals.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        result: Dict[str, Any] = {}
        for task_name, probs in output.as_dict().items():
            if probs is None:
                continue
            names = class_names.get(task_name, [])
            values = probs[0].tolist()
            result[task_name] = {
                (names[i] if i < len(names) else f"{task_name}_{i}"): float(v)
                for i, v in enumerate(values)
            }
        return result

    return _processor


@click.group(name="plugin")
def plugin_group() -> None:
    """ECG management system plugin commands."""


@plugin_group.command(name="list")
def list_cmd() -> None:
    """List all registered plugins."""
    from aortica.integration.plugins import list_plugins

    for name in list_plugins():
        click.echo(name)


@plugin_group.command(name="run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to plugins.yaml.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after N poll cycles (default: run forever).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load config and connect plugins, then exit without polling.",
)
def run_cmd(
    config_path: str, max_cycles: Optional[int], dry_run: bool
) -> None:
    """Start the plugin daemon from a YAML config.

    Polls each configured ECG management system, runs inference on new
    ECGs, and submits results back.
    """
    from aortica.integration.plugins import PluginDaemon, load_plugins_config

    configs = load_plugins_config(config_path)
    click.echo(f"Loaded {len(configs)} plugin(s): {[c.name for c in configs]}")

    if dry_run:
        # Validate that each plugin type resolves without connecting.
        from aortica.integration.plugins import get_plugin

        problems: List[str] = []
        for cfg in configs:
            try:
                get_plugin(cfg.plugin_type)
            except KeyError as exc:
                problems.append(str(exc))
        if problems:
            for p in problems:
                click.echo(f"ERROR: {p}", err=True)
            sys.exit(1)
        click.echo("Dry run OK — all plugin types resolve.")
        return

    daemon = PluginDaemon(_build_default_processor())
    daemon.add_from_configs(configs)
    click.echo("Daemon started. Press Ctrl-C to stop.")
    try:
        daemon.run(max_cycles=max_cycles)
    except KeyboardInterrupt:
        daemon.stop()
        click.echo("\nDaemon stopped.")
