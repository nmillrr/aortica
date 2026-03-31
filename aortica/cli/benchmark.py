"""``aortica benchmark <dataset_path>`` — evaluate model on a dataset.

Wraps :func:`aortica.evaluation.benchmark` with a Click CLI, supporting
task selection, output format, and CSV export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click


_VALID_TASKS = ("rhythm", "structural", "ischaemia", "risk")
_VALID_FORMATS = ("table", "json", "csv")


def _load_model_and_dataset(
    dataset_path: str,
    *,
    model_path: Optional[str],
    tasks: List[str],
    sampling_rate: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Load model and dataset, returning both along with metadata.

    Returns a dict with ``model``, ``dataset``, and ``metadata`` keys.
    """
    try:
        import torch

        from aortica.data.dataset import ECGDataset
        from aortica.data.ptbxl import load_ptbxl
        from aortica.models.aortica_model import AorticaModel
    except ImportError as exc:
        raise ImportError(
            "PyTorch and aortica[torch] are required for benchmarking. "
            f"Install with: pip install aortica[torch]  ({exc})"
        ) from exc

    # Load dataset
    _, _, (test_records, test_labels) = load_ptbxl(
        dataset_path, sampling_rate=sampling_rate,
    )

    dataset = ECGDataset(
        test_records,
        test_labels,
        target_hz=float(sampling_rate),
        augment=False,
    )

    # Extract metadata for subgroup analysis
    metadata: List[Optional[Dict[str, Any]]] = []
    for rec in test_records:
        metadata.append(rec.patient_metadata)

    # Load model
    if model_path is not None:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model = AorticaModel(enabled_tasks=list(tasks))
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, AorticaModel):
            model = checkpoint
        else:
            model = AorticaModel(enabled_tasks=list(tasks))
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        model = AorticaModel(enabled_tasks=list(tasks))
        model.eval()

    return {"model": model, "dataset": dataset, "metadata": metadata}


def _run_benchmark(
    dataset_path: str,
    *,
    model_path: Optional[str],
    tasks: List[str],
    sampling_rate: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Run the benchmark and return the report as a dict."""
    from aortica.evaluation.benchmark import benchmark

    loaded = _load_model_and_dataset(
        dataset_path,
        model_path=model_path,
        tasks=tasks,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
    )

    report = benchmark(
        model=loaded["model"],
        dataset=loaded["dataset"],
        tasks=tasks,
        batch_size=batch_size,
        seed=seed,
        metadata=loaded["metadata"],
    )

    return {
        "report": report,
        "as_dict": report.as_dict(),
        "summary": report.summary_table(),
        "csv": report.to_csv(),
    }


def _render_benchmark_table(result: Dict[str, Any]) -> None:
    """Print benchmark results as Rich-formatted summary."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    summary = result.get("summary", "No results available.")

    console.print()
    console.print(Panel(
        Text(summary),
        title="[bold white]Benchmark Report[/bold white]",
        border_style="cyan",
    ))
    console.print()


def _render_benchmark_json(result: Dict[str, Any]) -> None:
    """Print benchmark results as JSON to stdout."""
    click.echo(json.dumps(result["as_dict"], indent=2, default=str))


def _render_benchmark_csv(
    result: Dict[str, Any],
    csv_path: Optional[str],
) -> None:
    """Write or print benchmark results as CSV."""
    csv_text = result.get("csv", "")
    if csv_path:
        Path(csv_path).write_text(csv_text, encoding="utf-8")
        click.echo(f"CSV results written to: {csv_path}")
    else:
        click.echo(csv_text)


@click.command("benchmark")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output display format.",
)
@click.option(
    "--tasks",
    "tasks_str",
    type=str,
    default=None,
    help="Comma-separated task heads to evaluate (default: all). "
    "Choices: rhythm, structural, ischaemia, risk.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a saved model checkpoint.",
)
@click.option(
    "--csv-export",
    "csv_path",
    type=click.Path(),
    default=None,
    help="Path to write CSV results (only used with --format csv).",
)
@click.option(
    "--sampling-rate",
    type=click.Choice(["100", "500"]),
    default="500",
    show_default=True,
    help="ECG sampling rate in Hz.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size for evaluation.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
def benchmark_cmd(
    dataset_path: str,
    output_format: str,
    tasks_str: Optional[str],
    model_path: Optional[str],
    csv_path: Optional[str],
    sampling_rate: str,
    batch_size: int,
    seed: int,
) -> None:
    """Evaluate model performance on a dataset.

    DATASET_PATH is the path to the PTB-XL (or compatible) dataset directory.

    Examples:

    \b
        aortica benchmark /data/ptbxl
        aortica benchmark /data/ptbxl --format json
        aortica benchmark /data/ptbxl --tasks rhythm,ischaemia
        aortica benchmark /data/ptbxl --format csv --csv-export results.csv
        aortica benchmark /data/ptbxl --model checkpoint.pt
    """
    # Parse tasks
    if tasks_str is not None:
        tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
        invalid = [t for t in tasks if t not in _VALID_TASKS]
        if invalid:
            click.echo(
                f"Error: invalid task(s): {', '.join(invalid)}. "
                f"Valid tasks: {', '.join(_VALID_TASKS)}",
                err=True,
            )
            sys.exit(1)
    else:
        tasks = list(_VALID_TASKS)

    # Run benchmark
    try:
        result = _run_benchmark(
            dataset_path,
            model_path=model_path,
            tasks=tasks,
            sampling_rate=int(sampling_rate),
            batch_size=batch_size,
            seed=seed,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Render output
    if output_format == "json":
        _render_benchmark_json(result)
    elif output_format == "csv":
        _render_benchmark_csv(result, csv_path)
    else:
        _render_benchmark_table(result)
