"""Aortica CLI — ``aortica predict <file>`` and related commands.

Uses Click for argument parsing and Rich for coloured terminal output
with severity-coded prediction indicators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Click is required for the Aortica CLI. "
        "Install it with: pip install aortica[cli]"
    )

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

from aortica.api.predict import run_inference_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TASKS: List[str] = ["rhythm", "structural", "ischaemia", "risk"]

# Severity thresholds for colour coding
_HIGH_THRESHOLD = 0.80
_MED_THRESHOLD = 0.50

# Quality classification colours
_QUALITY_COLOURS: Dict[str, str] = {
    "good": "green",
    "marginal": "yellow",
    "poor": "red",
}

# ---------------------------------------------------------------------------
# Top-level CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="aortica")
def main() -> None:
    """Aortica — AI-powered ECG analysis from the command line."""


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


def _severity_style(prob: float) -> str:
    """Return a Rich colour name based on probability thresholds."""
    if prob >= _HIGH_THRESHOLD:
        return "bold red"
    if prob >= _MED_THRESHOLD:
        return "yellow"
    return "green"


def _severity_badge(prob: float) -> str:
    """Return a short severity badge string."""
    if prob >= _HIGH_THRESHOLD:
        return "●"  # high
    if prob >= _MED_THRESHOLD:
        return "◐"  # medium
    return "○"  # low


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _print_results_json(result_dict: Dict[str, Any]) -> None:
    """Print results as indented JSON to stdout."""
    click.echo(json.dumps(result_dict, indent=2))


def _print_results_table(result_dict: Dict[str, Any]) -> None:
    """Print results as richly-formatted table output."""
    if not HAS_RICH:
        # Fallback to plain JSON if rich is unavailable
        _print_results_json(result_dict)
        return

    console = Console()

    # ── Quality report ────────────────────────────────────────────────
    qr = result_dict.get("quality_report", {})
    overall_class = qr.get("overall_classification", "unknown")
    quality_colour = _QUALITY_COLOURS.get(overall_class, "white")
    quality_score = qr.get("overall_score", 0)

    quality_table = Table(title="Signal Quality", show_header=True)
    quality_table.add_column("Lead", style="cyan")
    quality_table.add_column("Score", justify="right")
    quality_table.add_column("Class")
    quality_table.add_column("Flags")

    for lead_info in qr.get("per_lead", []):
        lead_class = lead_info.get("classification", "")
        lc = _QUALITY_COLOURS.get(lead_class, "white")
        flags_str = ", ".join(lead_info.get("flags", [])) or "—"
        quality_table.add_row(
            lead_info.get("lead_name", "?"),
            f"{lead_info.get('score', 0):.0f}",
            Text(lead_class, style=lc),
            flags_str,
        )

    overall_text = Text(
        f"Overall: {quality_score:.0f} ({overall_class}) — {qr.get('recommendation', '?')}",
        style=f"bold {quality_colour}",
    )
    console.print(quality_table)
    console.print(overall_text)
    console.print()

    # ── Predictions ───────────────────────────────────────────────────
    predictions = result_dict.get("predictions", [])
    if not predictions:
        console.print(
            Panel("[dim]No model loaded — predictions unavailable[/dim]",
                  title="Predictions", border_style="dim")
        )
        return

    for task_pred in predictions:
        task_name: str = task_pred.get("task", "?")
        class_names: List[str] = task_pred.get("class_names", [])
        probs: List[float] = task_pred.get("probabilities", [])

        table = Table(title=f"{task_name.capitalize()} Predictions", show_header=True)
        table.add_column("", width=2)
        table.add_column("Finding", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Bar")

        for name, prob in zip(class_names, probs):
            style = _severity_style(prob)
            badge = _severity_badge(prob)
            pct = prob * 100
            bar_width = int(prob * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            table.add_row(
                Text(badge, style=style),
                name,
                Text(f"{pct:.1f}%", style=style),
                Text(bar, style=style),
            )

        console.print(table)
        console.print()

    # ── Uncertainty ───────────────────────────────────────────────────
    uncertainty = result_dict.get("uncertainty")
    if uncertainty:
        ood = uncertainty.get("ood_flag", False)
        entropy = uncertainty.get("entropy_score")
        ood_style = "bold red" if ood else "green"
        ood_text = "⚠ OUT-OF-DISTRIBUTION" if ood else "✓ In-distribution"
        parts = [f"[{ood_style}]{ood_text}[/{ood_style}]"]
        if entropy is not None:
            parts.append(f"  Entropy: {entropy:.4f}")
        console.print(Panel("\n".join(parts), title="Uncertainty", border_style="dim"))


# ---------------------------------------------------------------------------
# `aortica predict` command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format: coloured table or raw JSON.",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Comma-separated task heads to run (e.g. 'rhythm,risk'). Default: all.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to model checkpoint (.pt file).",
)
def predict(
    file: str,
    output_format: str,
    tasks: Optional[str],
    model_path: Optional[str],
) -> None:
    """Run AI inference on a single ECG file.

    Executes the full pipeline: read → denoise → quality scoring → model
    inference.  Results are printed as a coloured table (default) or JSON.
    """
    file_path = Path(file)
    file_bytes = file_path.read_bytes()
    filename = file_path.name

    # ── Parse tasks ───────────────────────────────────────────────────
    enabled_tasks: Optional[List[str]] = None
    if tasks:
        enabled_tasks = [t.strip() for t in tasks.split(",")]
        invalid = [t for t in enabled_tasks if t not in ALL_TASKS]
        if invalid:
            raise click.BadParameter(
                f"Unknown task(s): {', '.join(invalid)}. "
                f"Valid tasks: {', '.join(ALL_TASKS)}",
                param_hint="'--tasks'",
            )

    # ── Load model (optional) ─────────────────────────────────────────
    model: Any = None
    if model_path:
        try:
            import torch

            from aortica.models import AorticaModel

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            # Support both raw state_dict and wrapped checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                config = checkpoint.get("config", {})
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                config = checkpoint.get("config", {})
            else:
                state_dict = checkpoint
                config = {}

            model = AorticaModel(
                enabled_tasks=config.get("enabled_tasks", list(ALL_TASKS)),
                feature_dim=config.get("feature_dim", 252),
                num_leads=config.get("num_leads", 12),
            )
            model.load_state_dict(state_dict)
            model.eval()
        except ImportError:
            raise click.ClickException(
                "PyTorch is required for model inference. "
                "Install it with: pip install aortica[torch]"
            )
        except Exception as exc:
            raise click.ClickException(f"Failed to load model: {exc}")

    # ── Run inference pipeline ────────────────────────────────────────
    try:
        result = run_inference_pipeline(
            file_bytes,
            filename,
            model=model,
            enabled_tasks=enabled_tasks,
        )
    except Exception as exc:
        raise click.ClickException(f"Inference failed: {exc}")

    # ── Output ────────────────────────────────────────────────────────
    result_dict: Dict[str, Any] = result.model_dump()

    if output_format == "json":
        _print_results_json(result_dict)
    else:
        _print_results_table(result_dict)
