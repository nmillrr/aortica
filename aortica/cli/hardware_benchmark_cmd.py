"""CLI command: ``aortica benchmark-hardware`` — cross-hardware benchmark suite.

Wraps :func:`aortica.edge.hardware_benchmark.hardware_benchmark` and
:func:`aortica.edge.hardware_benchmark.benchmark_all_platforms` with
Click arguments and Rich-formatted output.

Example::

    aortica benchmark-hardware --platform server_cpu --model model.onnx
    aortica benchmark-hardware --all --model model.onnx --format csv

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click


@click.command("benchmark-hardware")
@click.option(
    "--platform",
    type=str,
    default=None,
    help="Platform profile name (e.g. server_cpu, rpi4). Use --all for all platforms.",
)
@click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to ONNX model file.",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Run benchmarks for all platform profiles.",
)
@click.option(
    "--n-runs",
    type=int,
    default=50,
    help="Number of inference runs per benchmark (default: 50).",
)
@click.option(
    "--targets-yaml",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Custom platform targets YAML file.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv", "markdown"]),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write output to file instead of stdout.",
)
def benchmark_hardware_cmd(
    platform: Optional[str],
    model: str,
    run_all: bool,
    n_runs: int,
    targets_yaml: Optional[str],
    output_format: str,
    output: Optional[str],
) -> None:
    """Run cross-hardware benchmark against platform targets.

    Benchmarks the specified ONNX model against per-platform latency,
    memory, and throughput targets.  Exits with code 1 if any target
    is missed.
    """
    from aortica.edge.hardware_benchmark import (
        HardwareBenchmarkReport,
        benchmark_all_platforms,
        consolidated_csv,
        consolidated_markdown_table,
        hardware_benchmark,
        load_platform_profiles,
    )

    if not platform and not run_all:
        click.echo(
            "Error: specify --platform <name> or --all. "
            "Use --help for available options.",
            err=True,
        )
        sys.exit(1)

    reports: list[HardwareBenchmarkReport] = []

    if run_all:
        reports = benchmark_all_platforms(
            model_path=model,
            n_runs=n_runs,
            targets_yaml=targets_yaml,
        )
    else:
        assert platform is not None
        report = hardware_benchmark(
            model_path=model,
            platform_profile=platform,
            n_runs=n_runs,
            targets_yaml=targets_yaml,
        )
        reports = [report]

    # Format output
    result_text = ""

    if output_format == "json":
        result_text = json.dumps(
            [r.to_dict() for r in reports], indent=2
        )
    elif output_format == "csv":
        result_text = consolidated_csv(reports)
    elif output_format == "markdown":
        result_text = consolidated_markdown_table(reports)
    else:  # table
        try:
            _render_rich(reports)
        except ImportError:
            for r in reports:
                click.echo(r.summary_table())
                click.echo()
        if output:
            # Also write plain text to file
            result_text = "\n\n".join(r.summary_table() for r in reports)

    if output and result_text:
        Path(output).write_text(result_text)
        click.echo(f"Written to {output}")
    elif result_text and output_format != "table":
        click.echo(result_text)

    # Exit with error if any target missed
    all_pass = all(r.overall_pass for r in reports)
    if not all_pass:
        click.echo("\n❌ One or more platform targets were not met.", err=True)
        sys.exit(1)


def _render_rich(reports: list[Any]) -> None:
    """Render benchmark results as Rich tables."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    for report in reports:
        # Summary table
        table = Table(
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Metric", style="dim")
        table.add_column("Measured", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        for mr in report.metric_results:
            status = "[green]✅ PASS[/green]" if mr.passed else "[red]❌ FAIL[/red]"
            table.add_row(
                mr.metric_name,
                f"{mr.measured:.2f}{mr.unit}",
                f"{mr.comparison} {mr.target:.2f}{mr.unit}",
                status,
            )

        border = "green" if report.overall_pass else "red"
        overall = "✅ PASS" if report.overall_pass else "❌ FAIL"
        title = f"🔧  {report.platform_name} ({report.model_variant}) — {overall}"

        console.print(Panel(table, title=title, border_style=border))

        # Stats panel
        stats = Table(show_header=False, expand=True)
        stats.add_column("", style="dim")
        stats.add_column("", justify="right")
        stats.add_row("Mean latency", f"{report.mean_latency_ms:.2f} ms")
        stats.add_row("p50 latency", f"{report.p50_latency_ms:.2f} ms")
        stats.add_row("p95 latency", f"{report.p95_latency_ms:.2f} ms")
        stats.add_row("p99 latency", f"{report.p99_latency_ms:.2f} ms")
        stats.add_row("Throughput", f"{report.throughput_ips:.2f} ips")
        stats.add_row("Peak memory", f"{report.peak_memory_mb:.2f} MB")
        stats.add_row("Model size", f"{report.model_size_mb:.2f} MB")
        stats.add_row("Runs", str(report.n_runs))

        console.print(Panel(stats, title="📊  Details", border_style="dim"))
        console.print()
