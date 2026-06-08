"""CLI command for quarterly performance report generation.

Provides ``aortica validation quarterly-report`` to produce quarterly
performance reports from the monitoring database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Click is required for the Aortica CLI. "
        "Install it with: pip install aortica[cli]"
    )

try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Click command group
# ---------------------------------------------------------------------------


@click.group(name="validation")
def validation_group() -> None:
    """Prospective validation and performance monitoring commands."""


# ---------------------------------------------------------------------------
# quarterly-report
# ---------------------------------------------------------------------------


@validation_group.command(name="quarterly-report")
@click.option(
    "--quarter",
    type=click.IntRange(1, 4),
    required=True,
    help="Quarter number (1–4).",
)
@click.option(
    "--year",
    type=int,
    required=True,
    help="Year for the report (e.g. 2026).",
)
@click.option(
    "--monitor-db",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to the monitor database directory. Defaults to current directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=".",
    show_default=True,
    help="Directory to write report files.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the summary.",
)
def quarterly_report_cmd(
    quarter: int,
    year: int,
    monitor_db: Optional[str],
    output_dir: str,
    output_format: str,
) -> None:
    """Generate a quarterly public performance report.

    Reads production monitoring data from the monitor database and
    produces both a Markdown report (QUARTERLY_REPORT_YYYY_QN.md) and
    a CSV export (quarterly_report_YYYY_QN.csv).

    Example:

        aortica validation quarterly-report --quarter 1 --year 2026

        aortica validation quarterly-report --quarter 2 --year 2026 --monitor-db /data/monitor
    """
    from aortica.validation.performance_monitor import PerformanceMonitor
    from aortica.validation.quarterly_report import generate_quarterly_report

    # Open monitor
    db_dir = monitor_db or "."
    try:
        monitor = PerformanceMonitor(db_dir=db_dir)
    except Exception as exc:
        raise click.ClickException(f"Failed to open monitor database: {exc}")

    # Generate report
    try:
        result = generate_quarterly_report(
            monitor=monitor,
            output_dir=output_dir,
            quarter=quarter,
            year=year,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:
        raise click.ClickException(f"Report generation failed: {exc}")
    finally:
        monitor.close()

    # Output summary
    if output_format == "json":
        click.echo(json.dumps({
            "markdown_path": result.markdown_path,
            "csv_path": result.csv_path,
            "quarter": result.quarter,
            "year": result.year,
            "total_ecgs": result.total_ecgs,
            "tasks_reported": result.tasks_reported,
            "has_drift": result.has_drift,
        }, indent=2))
    else:
        if HAS_RICH:
            console = Console()
            console.print(
                f"[bold green]✓ Quarterly report generated for "
                f"Q{quarter} {year}[/bold green]"
            )
            console.print(f"  Markdown: {result.markdown_path}")
            console.print(f"  CSV:      {result.csv_path}")
            console.print(f"  ECGs:     {result.total_ecgs}")
            console.print(f"  Tasks:    {', '.join(result.tasks_reported) or 'None'}")
            if result.has_drift:
                console.print("  [bold red]⚠ Drift detected[/bold red]")
            else:
                console.print("  [green]✓ No drift[/green]")
        else:
            click.echo(f"✓ Quarterly report generated for Q{quarter} {year}")
            click.echo(f"  Markdown: {result.markdown_path}")
            click.echo(f"  CSV:      {result.csv_path}")
            click.echo(f"  ECGs:     {result.total_ecgs}")
