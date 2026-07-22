"""``aortica integration`` — end-to-end EHR integration daemon (US-125).

\b
  aortica integration run --config integration.yaml [--dry-run] [--max-cycles N]

Ingests ECGs from a watched directory, runs the full inference + write-back
pipeline through the :class:`IntegrationOrchestrator`, and reports status.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import click


@click.group(name="integration")
def integration_group() -> None:
    """End-to-end EHR integration commands."""


def _build_processor() -> Any:
    """Build the AI-processing callable (lazy torch import)."""
    import numpy as np
    import torch

    from aortica.api.predict import _get_task_class_names
    from aortica.models.registry import load_pretrained

    model = load_pretrained("latest")
    model.eval()
    class_names = _get_task_class_names()

    def _processor(payload: Any) -> dict:
        x = torch.from_numpy(payload.signals.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        result: dict = {}
        for task, probs in output.as_dict().items():
            if probs is None:
                continue
            names = class_names.get(task, [])
            values = probs[0].tolist()
            result[task] = {
                (names[i] if i < len(names) else f"{task}_{i}"): float(v)
                for i, v in enumerate(values)
            }
        return result

    return _processor


@integration_group.command(name="run")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Path to integration.yaml.")
@click.option("--watch-dir", "watch_dir", default=None,
              type=click.Path(), help="Directory to ingest ECGs from.")
@click.option("--output-dir", "output_dir", default=None,
              type=click.Path(), help="Directory for result JSON.")
@click.option("--max-cycles", type=int, default=None,
              help="Stop after N poll cycles (default: run forever).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Load config and report enabled channels, then exit.")
def run_cmd(
    config_path: str,
    watch_dir: Optional[str],
    output_dir: Optional[str],
    max_cycles: Optional[int],
    dry_run: bool,
) -> None:
    """Start the integration orchestrator daemon."""
    from aortica.integration.orchestrator import (
        IntegrationOrchestrator,
        load_integration_config,
    )

    config = load_integration_config(config_path)
    click.echo(f"Enabled channels: {config.enabled_channels}")

    if dry_run:
        click.echo("Dry run OK — configuration is valid.")
        return

    if not watch_dir or not output_dir:
        raise click.UsageError(
            "--watch-dir and --output-dir are required to run the daemon."
        )

    from aortica.integration.fhir_subscription import SubscriptionManager
    from aortica.integration.plugins.file_watcher import FileWatcherPlugin
    from aortica.integration.worklist_store import WorklistStore

    plugin = FileWatcherPlugin()
    plugin.connect({"watch_dir": watch_dir, "output_dir": output_dir})

    orchestrator = IntegrationOrchestrator(
        config,
        processor=_build_processor(),
        result_store=None,
        worklist_store=WorklistStore(),
        subscription_manager=SubscriptionManager(),
    )

    click.echo("Integration daemon started. Press Ctrl-C to stop.")
    cycles = 0
    try:
        while True:
            for ecg_id, payload in plugin.poll_for_ecgs():
                result = orchestrator.process_ecg(ecg_id, payload)
                plugin.submit_result(ecg_id, result.result)
            orchestrator.retry_failed()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
    except KeyboardInterrupt:
        pass

    click.echo(json.dumps(orchestrator.status(), indent=2))


@integration_group.command(name="status")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Path to integration.yaml.")
def status_cmd(config_path: str) -> None:
    """Print the enabled channels from an integration config."""
    from aortica.integration.orchestrator import load_integration_config

    config = load_integration_config(config_path)
    click.echo(json.dumps({
        "enabled_channels": config.enabled_channels,
        "max_retries": config.max_retries,
        "error_rate_threshold": config.error_rate_threshold,
    }, indent=2))
    if not config.enabled_channels:
        sys.exit(1)
