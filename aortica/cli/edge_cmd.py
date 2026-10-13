"""CLI commands for Aortica edge site operations (US-061b).

Provides:
- ``aortica edge site-report`` — generate a daily site activity summary
  (inferences, errors, sync status, storage) for remote pilot monitoring.
- ``aortica edge watch`` — score ECGs dropped into an inbox with the INT8
  ONNX edge model; the process ``aortica-edge.service`` runs.
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Click is required for the Aortica CLI. "
        "Install it with: pip install aortica[cli]"
    )


def _default_data_dir() -> str:
    """Resolve the edge data directory from the environment."""
    return os.environ.get("AORTICA_DATA_DIR", "/var/lib/aortica/data")


@click.group("edge")
def edge_group() -> None:
    """Edge deployment and site-monitoring commands."""


@edge_group.command("site-report")
@click.option(
    "--data-dir",
    type=click.Path(),
    default=None,
    help="Edge data directory (defaults to $AORTICA_DATA_DIR or "
    "/var/lib/aortica/data).",
)
@click.option(
    "--site-id",
    type=str,
    default="aortica-edge",
    show_default=True,
    help="Site identifier to label the report.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format: human-readable table or raw JSON.",
)
def site_report_cmd(
    data_dir: Optional[str],
    site_id: str,
    output_format: str,
) -> None:
    """Generate a daily site activity summary for remote monitoring.

    Reports the last-24-hour inference count and error rate, sync status,
    storage utilisation, and last-sync timestamp for the deployed edge site.

    Examples:

    \b
        aortica edge site-report
        aortica edge site-report --site-id kigali-clinic --format json
    """
    from aortica.edge.site_monitor import SiteMonitor

    resolved_dir = data_dir or _default_data_dir()

    monitor = SiteMonitor(resolved_dir, site_id=site_id)
    try:
        report = monitor.daily_report()
        status = monitor.status()
    finally:
        monitor.close()

    if output_format == "json":
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(status.summary())
        click.echo(f"\nReport date: {report['report_date']}")


@edge_group.command("watch")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True,
              dir_okay=False), help="ONNX edge model (e.g. aortica_edge_int8.onnx).")
@click.option("--inbox", required=True, type=click.Path(file_okay=False),
              help="Directory to poll for ECG files.")
@click.option("--outbox", required=True, type=click.Path(file_okay=False),
              help="Directory for <ecg_id>.json results.")
@click.option("--processed", default=None, type=click.Path(file_okay=False),
              help="Move scored source files here (default: leave in place).")
@click.option("--thresholds", "thresholds_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="class_thresholds.json for this model (default: packaged).")
@click.option("--interval", type=float, default=5.0, show_default=True,
              help="Seconds between inbox polls.")
@click.option("--unload-after", type=float, default=30.0, show_default=True,
              help="Release the model after this many idle seconds.")
@click.option("--max-cycles", type=int, default=None,
              help="Stop after N polls (default: run until stopped).")
def watch_cmd(
    model_path: str,
    inbox: str,
    outbox: str,
    processed: Optional[str],
    thresholds_path: Optional[str],
    interval: float,
    unload_after: float,
    max_cycles: Optional[int],
) -> None:
    """Score ECGs dropped into INBOX with the edge model, writing to OUTBOX.

    Research use only: results are not a diagnosis.
    """
    import logging
    from pathlib import Path

    from aortica.edge.watch_service import EdgeScorer, run_watch_loop

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    thresholds = None
    if thresholds_path:
        with open(thresholds_path, encoding="utf-8") as fh:
            thresholds = {k: float(v) for k, v in json.load(fh).items()}
    scorer = EdgeScorer(Path(model_path), thresholds=thresholds,
                        unload_after_seconds=unload_after)
    click.echo(f"Watching {inbox} with {Path(model_path).name}; results -> {outbox}")
    try:
        n = run_watch_loop(
            scorer, Path(inbox), Path(outbox),
            processed=Path(processed) if processed else None,
            interval_seconds=interval, max_cycles=max_cycles,
        )
    except KeyboardInterrupt:
        return
    click.echo(f"Processed {n} ECG(s).")
