"""``aortica performance-card`` — generate a public performance card.

Wraps :func:`aortica.evaluation.generate_performance_card` with a Click CLI,
accepting a benchmark report JSON file and producing PERFORMANCE_CARD.md
and performance_card.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    SubgroupReport,
    TaskReport,
)


def _load_benchmark_report(report_path: str) -> BenchmarkReport:
    """Load a BenchmarkReport from a JSON file.

    Reconstructs the dataclass hierarchy from a plain dict produced by
    :meth:`BenchmarkReport.as_dict()`.
    """
    raw: dict[str, Any] = json.loads(
        Path(report_path).read_text(encoding="utf-8")
    )

    # Reconstruct overall task reports
    overall: dict[str, TaskReport] = {}
    for task_name, tr_data in raw.get("overall", {}).items():
        per_class = [
            ClassMetrics(**cm_data)
            for cm_data in tr_data.get("per_class", [])
        ]
        overall[task_name] = TaskReport(
            task_name=tr_data.get("task_name", task_name),
            macro_f1=tr_data.get("macro_f1", 0.0),
            ece=tr_data.get("ece", 0.0),
            c_index=tr_data.get("c_index", 0.0),
            brier_score=tr_data.get("brier_score", 0.0),
            per_class=per_class,
        )

    # Reconstruct subgroups
    subgroups: list[SubgroupReport] = []
    for sg_data in raw.get("subgroups", []):
        sg_task_reports: dict[str, TaskReport] = {}
        for task_name, tr_data in sg_data.get("task_reports", {}).items():
            per_class = [
                ClassMetrics(**cm_data)
                for cm_data in tr_data.get("per_class", [])
            ]
            sg_task_reports[task_name] = TaskReport(
                task_name=tr_data.get("task_name", task_name),
                macro_f1=tr_data.get("macro_f1", 0.0),
                ece=tr_data.get("ece", 0.0),
                c_index=tr_data.get("c_index", 0.0),
                brier_score=tr_data.get("brier_score", 0.0),
                per_class=per_class,
            )
        subgroups.append(
            SubgroupReport(
                subgroup_name=sg_data.get("subgroup_name", ""),
                n_samples=sg_data.get("n_samples", 0),
                task_reports=sg_task_reports,
            )
        )

    return BenchmarkReport(
        overall=overall,
        subgroups=subgroups,
        n_samples=raw.get("n_samples", 0),
        tasks_evaluated=raw.get("tasks_evaluated", []),
    )


@click.command("performance-card")
@click.argument("report_path", type=click.Path(exists=True))
@click.option(
    "--version",
    "model_version",
    required=True,
    help="Model version string (e.g. 0.3.0).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".",
    show_default=True,
    help="Directory to write PERFORMANCE_CARD.md and performance_card.csv.",
)
@click.option(
    "--model-weights",
    "model_weights_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to model weights file for SHA-256 reproducibility hash.",
)
@click.option(
    "--dataset-info",
    type=str,
    default=None,
    help='Dataset description string (e.g. "PTB-XL test fold 10, 500 Hz").',
)
def performance_card_cmd(
    report_path: str,
    model_version: str,
    output_dir: str,
    model_weights_path: Optional[str],
    dataset_info: Optional[str],
) -> None:
    """Generate a public performance card from a benchmark report JSON.

    REPORT_PATH is the path to a benchmark report JSON file produced by
    ``BenchmarkReport.as_dict()`` saved to disk.

    Examples:

    \\b
        aortica performance-card report.json --version 0.3.0
        aortica performance-card report.json --version 0.3.0 --output-dir ./cards
        aortica performance-card report.json --version 0.3.0 --model-weights model.pt
    """
    from aortica.evaluation.performance_card import generate_performance_card

    try:
        report = _load_benchmark_report(report_path)
    except Exception as exc:
        click.echo(f"Error loading report: {exc}", err=True)
        sys.exit(1)

    try:
        result = generate_performance_card(
            benchmark_report=report,
            model_version=model_version,
            output_dir=output_dir,
            model_weights_path=model_weights_path,
            dataset_info=dataset_info,
        )
    except Exception as exc:
        click.echo(f"Error generating performance card: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Performance card generated:")
    click.echo(f"  Markdown: {result.markdown_path}")
    click.echo(f"  CSV:      {result.csv_path}")
