"""``aortica predict <file>`` — run full inference pipeline from the CLI.

Reads an ECG file, runs denoise → quality scoring → model inference,
and prints results as a Rich table (default) or JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import click

# ---------------------------------------------------------------------------
# Lazy-imported helpers
# ---------------------------------------------------------------------------

_VALID_TASKS = ("rhythm", "structural", "ischaemia", "risk")


def _load_model(
    model_path: Optional[str],
    tasks: Sequence[str],
) -> Any:
    """Attempt to load an AorticaModel checkpoint.

    When *model_path* is ``None``, attempts to load the latest pretrained
    model from HuggingFace Hub via :func:`load_pretrained`.  Falls back
    to ``None`` if the download fails or torch is unavailable.
    """
    if model_path is None:
        # Try loading pretrained from Hub
        try:
            from aortica.models.registry import load_pretrained

            return load_pretrained("latest")
        except Exception:
            return None

    try:
        import torch

        from aortica.models.aortica_model import AorticaModel
    except ImportError:
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = AorticaModel(enabled_tasks=list(tasks))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, AorticaModel):
        model = checkpoint
    else:
        # Assume it's a raw state_dict
        model = AorticaModel(enabled_tasks=list(tasks))
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def _run_pipeline(
    file_path: str,
    *,
    format_override: Optional[str],
    model: Any,
    tasks: Sequence[str],
) -> Dict[str, Any]:
    """Run the inference pipeline and return a plain dict of results."""
    from aortica.api.predict import run_inference_pipeline

    path = Path(file_path)
    file_bytes = path.read_bytes()

    response = run_inference_pipeline(
        file_bytes,
        path.name,
        format_override=format_override,
        model=model,
        enabled_tasks=list(tasks),
    )

    return response.model_dump()


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: Dict[str, str] = {
    "critical": "bold red",
    "high": "red",
    "moderate": "yellow",
    "low": "green",
}


def _classify_severity(prob: float) -> str:
    """Map a probability to a severity label."""
    if prob >= 0.9:
        return "critical"
    if prob >= 0.7:
        return "high"
    if prob >= 0.5:
        return "moderate"
    return "low"


def _render_table(result: Dict[str, Any]) -> None:
    """Print results as Rich tables with severity coloring."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ── Quality report ────────────────────────────────────────────────
    qr = result.get("quality_report", {})
    overall_score = qr.get("overall_score", 0)
    overall_class = qr.get("overall_classification", "unknown")
    recommendation = qr.get("recommendation", "unknown")

    quality_color = (
        "green" if overall_class == "good"
        else "yellow" if overall_class == "marginal"
        else "red"
    )

    quality_table = Table(
        title="Signal Quality",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    quality_table.add_column("Lead", style="bold")
    quality_table.add_column("Score", justify="right")
    quality_table.add_column("Classification")
    quality_table.add_column("Flags")

    for lq in qr.get("per_lead", []):
        lq_color = (
            "green" if lq["classification"] == "good"
            else "yellow" if lq["classification"] == "marginal"
            else "red"
        )
        flags_str = ", ".join(lq.get("flags", [])) or "—"
        quality_table.add_row(
            lq["lead_name"],
            f"[{lq_color}]{lq['score']:.1f}[/{lq_color}]",
            f"[{lq_color}]{lq['classification']}[/{lq_color}]",
            flags_str,
        )

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"Overall: [{quality_color}]{overall_score:.1f} "
            f"({overall_class})[/{quality_color}]  •  "
            f"Recommendation: [{quality_color}]{recommendation}[/{quality_color}]"
        ),
        title="[bold white]Signal Quality Summary[/bold white]",
        border_style="cyan",
    ))
    console.print(quality_table)

    # ── Predictions per task ──────────────────────────────────────────
    predictions: List[Dict[str, Any]] = result.get("predictions", [])
    if not predictions:
        console.print(
            "\n[dim]No model loaded — predictions unavailable.[/dim]\n"
        )
    else:
        for task_pred in predictions:
            task_name = task_pred.get("task", "unknown")
            class_names: List[str] = task_pred.get("class_names", [])
            probabilities: List[float] = task_pred.get("probabilities", [])

            table = Table(
                title=f"{task_name.capitalize()} Predictions",
                show_header=True,
                header_style="bold cyan",
                title_style="bold white",
            )
            table.add_column("Class", style="bold")
            table.add_column("Probability", justify="right")
            table.add_column("Severity")

            # Sort by probability descending for display
            sorted_indices = sorted(
                range(len(probabilities)),
                key=lambda i: probabilities[i],
                reverse=True,
            )

            for idx in sorted_indices:
                prob = probabilities[idx]
                name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                severity = _classify_severity(prob)
                color = _SEVERITY_COLORS[severity]
                table.add_row(
                    name,
                    f"[{color}]{prob:.1%}[/{color}]",
                    f"[{color}]{severity}[/{color}]",
                )

            console.print()
            console.print(table)

    # ── Uncertainty ───────────────────────────────────────────────────
    uncertainty = result.get("uncertainty")
    if uncertainty:
        console.print()
        ood = uncertainty.get("ood_flag", False)
        entropy = uncertainty.get("entropy_score")
        ood_text = "[bold red]YES[/bold red]" if ood else "[green]no[/green]"
        entropy_text = f"{entropy:.4f}" if entropy is not None else "—"
        console.print(Panel(
            Text.from_markup(
                f"OOD flag: {ood_text}  •  Entropy: {entropy_text}"
            ),
            title="[bold white]Uncertainty[/bold white]",
            border_style="cyan",
        ))

    console.print()


def _render_json(result: Dict[str, Any]) -> None:
    """Print results as formatted JSON to stdout."""
    click.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output display format.",
)
@click.option(
    "--tasks",
    "tasks_str",
    type=str,
    default=None,
    help="Comma-separated task heads to run (default: all). "
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
    "--ecg-format",
    "ecg_format",
    type=str,
    default=None,
    help="Explicit ECG file format override (e.g. wfdb, dicom, csv).",
)
def predict(
    file: str,
    output_format: str,
    tasks_str: Optional[str],
    model_path: Optional[str],
    ecg_format: Optional[str],
) -> None:
    """Run AI inference on an ECG file.

    FILE is the path to the ECG recording to analyse.

    Examples:

    \b
        aortica predict recording.hea
        aortica predict recording.hea --format json
        aortica predict recording.hea --tasks rhythm,ischaemia
        aortica predict recording.hea --model checkpoint.pt
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

    # Load model (if provided)
    model = _load_model(model_path, tasks)

    # Run pipeline
    try:
        result = _run_pipeline(
            file,
            format_override=ecg_format,
            model=model,
            tasks=tasks,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Render output
    if output_format == "json":
        _render_json(result)
    else:
        _render_table(result)
