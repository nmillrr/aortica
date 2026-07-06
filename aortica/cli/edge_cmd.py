"""CLI commands for Aortica edge site operations (US-061b).

Provides:
- ``aortica edge site-report`` — generate a daily site activity summary
  (inferences, errors, sync status, storage) for remote pilot monitoring.
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
