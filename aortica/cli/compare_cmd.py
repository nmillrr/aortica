"""``aortica compare`` — model version comparison and A/B analysis (US-116).

Compares two model versions on the same dataset and writes a
``MODEL_COMPARISON.md`` report with per-task/per-class delta metrics,
paired-bootstrap significance, demographic subgroup deltas, regression
warnings, and an upgrade/hold/investigate recommendation.

Two input modes:

* ``--model-a / --model-b / --dataset`` — load two checkpoints and run
  inference over a dataset (requires ``aortica[torch]``).
* ``--predictions-a / --predictions-b / --targets`` — compare from
  pre-computed per-task prediction ``.npz`` bundles (no torch required).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click


def _load_npz_preds(path: str) -> Dict[str, Any]:
    """Load a per-task prediction/target ``.npz`` bundle into a dict."""
    import numpy as np

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


@click.command(name="compare")
@click.option("--model-a", "model_a", type=click.Path(), default=None,
              help="Path to model A checkpoint.")
@click.option("--model-b", "model_b", type=click.Path(), default=None,
              help="Path to model B checkpoint.")
@click.option("--dataset", "dataset_path", type=click.Path(), default=None,
              help="Path to the evaluation dataset (PTB-XL format).")
@click.option("--predictions-a", "predictions_a", type=click.Path(exists=True),
              default=None, help="Pre-computed model-A predictions (.npz).")
@click.option("--predictions-b", "predictions_b", type=click.Path(exists=True),
              default=None, help="Pre-computed model-B predictions (.npz).")
@click.option("--targets", "targets_path", type=click.Path(exists=True),
              default=None, help="Ground-truth targets (.npz), for --predictions-* mode.")
@click.option("--tasks", default="all", show_default=True,
              help="Comma-separated task list or 'all'.")
@click.option("--n-bootstrap", type=int, default=1000, show_default=True,
              help="Bootstrap resamples for significance testing.")
@click.option("--alpha", type=float, default=0.05, show_default=True,
              help="Significance level for regression detection.")
@click.option("--output", "output_path", type=click.Path(), default="MODEL_COMPARISON.md",
              show_default=True, help="Path to write the Markdown report.")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"], case_sensitive=False),
              default="text", show_default=True, help="Console output format.")
def compare_cmd(
    model_a: Optional[str],
    model_b: Optional[str],
    dataset_path: Optional[str],
    predictions_a: Optional[str],
    predictions_b: Optional[str],
    targets_path: Optional[str],
    tasks: str,
    n_bootstrap: int,
    alpha: float,
    output_path: str,
    output_format: str,
) -> None:
    """Compare two model versions and generate a comparison report.

    \b
    Examples:
        aortica compare --model-a v1.pt --model-b v2.pt --dataset ./ptbxl
        aortica compare --predictions-a a.npz --predictions-b b.npz --targets t.npz
    """
    from aortica.evaluation.model_comparison import (
        compare_models,
        compare_predictions,
    )

    task_arg: Any = (
        "all"
        if tasks.strip().lower() == "all"
        else [t.strip() for t in tasks.split(",") if t.strip()]
    )

    use_npz = predictions_a and predictions_b and targets_path
    use_models = model_a and model_b and dataset_path

    if use_npz:
        report = compare_predictions(
            _load_npz_preds(predictions_a),  # type: ignore[arg-type]
            _load_npz_preds(predictions_b),  # type: ignore[arg-type]
            _load_npz_preds(targets_path),  # type: ignore[arg-type]
            version_a=str(predictions_a),
            version_b=str(predictions_b),
            tasks=task_arg,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
    elif use_models:
        from aortica.cli.benchmark import _load_model_and_dataset

        # Reuse the benchmark loader to build the dataset + metadata.
        task_names = (
            ["rhythm", "structural", "ischaemia", "risk"]
            if task_arg == "all"
            else task_arg
        )
        loaded = _load_model_and_dataset(
            dataset_path,  # type: ignore[arg-type]
            model_path=model_a,
            tasks=task_names,
            sampling_rate=100,
            batch_size=64,
        )
        report = compare_models(
            model_a,  # type: ignore[arg-type]
            model_b,  # type: ignore[arg-type]
            loaded["dataset"],
            tasks=task_arg,
            metadata=loaded["metadata"],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
    else:
        raise click.UsageError(
            "Provide either --model-a/--model-b/--dataset or "
            "--predictions-a/--predictions-b/--targets."
        )

    # Write the Markdown report.
    Path(output_path).write_text(report.to_markdown())

    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(f"Wrote comparison report to {output_path}")
        click.echo(f"Recommendation: {report.recommendation.upper()}")
        if report.regressions:
            click.echo(f"Regressions ({len(report.regressions)}):")
            for reg in report.regressions:
                click.echo(f"  - {reg}")

    # Non-zero exit when a regression is detected, for CI gating.
    if report.regressions:
        sys.exit(1)
