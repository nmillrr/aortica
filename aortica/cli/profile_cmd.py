"""CLI command: ``aortica profile`` — inference profiling for edge models.

Wraps :func:`aortica.edge.profiling.profile_inference` with Click
arguments and Rich-formatted output.

Example::

    aortica profile model.onnx
    aortica profile model.onnx --format json --n-runs 200 --hardware rpi5

"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import numpy as np

from aortica.edge.profiling import (
    DEFAULT_HARDWARE,
    DEFAULT_N_RUNS,
    HARDWARE_TDP,
    InferenceProfile,
    profile_inference,
)


def _render_profile_table(profile: InferenceProfile) -> None:
    """Render the profiling results as a Rich table."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Latency table
    latency_table = Table(
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    latency_table.add_column("Metric", style="dim")
    latency_table.add_column("Value", justify="right")

    latency_table.add_row("Mean", f"{profile.mean_latency_ms:.2f} ms")
    latency_table.add_row("Median (p50)", f"{profile.p50_latency_ms:.2f} ms")
    latency_table.add_row("p95", f"{profile.p95_latency_ms:.2f} ms")
    latency_table.add_row("Min", f"{profile.min_latency_ms:.2f} ms")
    latency_table.add_row("Max", f"{profile.max_latency_ms:.2f} ms")
    latency_table.add_row("Std dev", f"{profile.std_latency_ms:.2f} ms")

    console.print(
        Panel(latency_table, title="⏱  Latency", border_style="cyan"),
    )

    # Resource table
    resource_table = Table(
        show_header=True,
        header_style="bold green",
        expand=True,
    )
    resource_table.add_column("Metric", style="dim")
    resource_table.add_column("Value", justify="right")

    resource_table.add_row(
        "Model size", f"{profile.model_size_mb:.2f} MB",
    )
    resource_table.add_row(
        "Peak memory (est)", f"{profile.peak_memory_mb:.2f} MB",
    )
    resource_table.add_row("Input shape", str(profile.input_shape))
    resource_table.add_row("Runs", str(profile.n_runs))

    console.print(
        Panel(resource_table, title="💾  Resources", border_style="green"),
    )

    # Power table
    power_table = Table(
        show_header=True,
        header_style="bold yellow",
        expand=True,
    )
    power_table.add_column("Metric", style="dim")
    power_table.add_column("Value", justify="right")

    power_table.add_row("Hardware", profile.hardware_profile)
    power_table.add_row("TDP", f"{profile.tdp_watts:.1f} W")
    power_table.add_row(
        "Energy/inference", f"{profile.energy_per_inference_mj:.3f} mJ",
    )
    power_table.add_row(
        "Power draw", f"{profile.power_draw_watts:.3f} W",
    )

    console.print(
        Panel(power_table, title="⚡  Power Estimation", border_style="yellow"),
    )


def _render_profile_json(profile: InferenceProfile) -> None:
    """Render the profiling results as JSON to stdout."""
    data = profile.to_dict()
    # Remove raw latencies from JSON output (too verbose)
    click.echo(json.dumps(data, indent=2))


@click.command("profile")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--n-runs",
    type=int,
    default=DEFAULT_N_RUNS,
    show_default=True,
    help="Number of inference runs for profiling.",
)
@click.option(
    "--hardware",
    type=click.Choice(sorted(HARDWARE_TDP.keys())),
    default=DEFAULT_HARDWARE,
    show_default=True,
    help="Hardware profile for power estimation.",
)
@click.option(
    "--warmup",
    type=int,
    default=5,
    show_default=True,
    help="Number of warmup runs before profiling.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for input data (generates synthetic input).",
)
@click.option(
    "--leads",
    type=int,
    default=12,
    show_default=True,
    help="Number of ECG leads in synthetic input.",
)
@click.option(
    "--duration-samples",
    type=int,
    default=5000,
    show_default=True,
    help="Number of samples (at 500 Hz = 10s) in synthetic input.",
)
def profile_cmd(
    model_path: str,
    output_format: str,
    n_runs: int,
    hardware: str,
    warmup: int,
    batch_size: int,
    leads: int,
    duration_samples: int,
) -> None:
    """Profile inference latency, memory, and power for an ONNX edge model.

    MODEL_PATH is the path to the ONNX model file to profile.

    Generates synthetic ECG input data with the specified shape and runs
    the model repeatedly to collect timing statistics.
    """
    try:
        # Generate synthetic input
        input_data: Any = np.random.randn(
            batch_size, leads, duration_samples,
        ).astype(np.float32)

        result = profile_inference(
            model_path=model_path,
            input_data=input_data,
            n_runs=n_runs,
            hardware_profile=hardware,
            warmup_runs=warmup,
        )

        if output_format == "json":
            _render_profile_json(result)
        else:
            _render_profile_table(result)

    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
