"""Public Performance Card Generator (US-070).

Generates per-release performance documentation including per-task metrics,
demographic subgroup breakdowns, equity gate results, and reproducibility
metadata.

Usage::

    from aortica.evaluation import generate_performance_card

    generate_performance_card(
        benchmark_report=report,
        model_version="0.3.0",
        output_dir="./cards",
    )

This produces two files in *output_dir*:

- ``PERFORMANCE_CARD.md`` — human-readable markdown card
- ``performance_card.csv`` — same data in tabular format
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    CLASSIFICATION_TASKS,
    SubgroupReport,
    TaskReport,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerformanceCardResult:
    """Result of performance card generation.

    Attributes:
        markdown_path: Absolute path to the generated markdown file.
        csv_path: Absolute path to the generated CSV file.
        markdown_content: Full text of the markdown card.
        csv_content: Full text of the CSV file.
        model_version: Version string used.
        timestamp: ISO-8601 timestamp of generation.
        model_weights_sha256: SHA-256 hash of the model weights (if provided).
    """

    markdown_path: str = ""
    csv_path: str = ""
    markdown_content: str = ""
    csv_content: str = ""
    model_version: str = ""
    timestamp: str = ""
    model_weights_sha256: str = ""


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


def _compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest, or empty string if file not found.
    """
    if not os.path.isfile(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _generate_markdown(
    report: BenchmarkReport,
    model_version: str,
    timestamp: str,
    model_weights_sha256: str,
    dataset_info: Optional[str],
    equity_gate_result: Optional[Any],
) -> str:
    """Generate the markdown content for the performance card."""
    lines: list[str] = []

    # Header
    lines.append(f"# Performance Card — Aortica v{model_version}")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Model Version:** {model_version}")
    if model_weights_sha256:
        lines.append(f"**Weights SHA-256:** `{model_weights_sha256}`")
    lines.append(f"**Samples Evaluated:** {report.n_samples}")
    if dataset_info:
        lines.append(f"**Dataset:** {dataset_info}")
    lines.append("")

    # Per-task overall metrics
    lines.append("## Overall Performance")
    lines.append("")

    for task_name in report.tasks_evaluated:
        tr = report.overall.get(task_name)
        if tr is None:
            continue

        lines.append(f"### {task_name.replace('_', ' ').title()}")
        lines.append("")

        if task_name in CLASSIFICATION_TASKS:
            lines.append(f"- **Macro-F1:** {tr.macro_f1:.4f}")
            lines.append(f"- **ECE:** {tr.ece:.4f}")
            lines.append("")

            # Per-class table
            lines.append(
                "| Class | AUC | Sensitivity | Specificity | F1 |"
            )
            lines.append(
                "|-------|-----|-------------|-------------|----|"
            )
            for cm in tr.per_class:
                lines.append(
                    f"| {cm.name} | {cm.auc:.4f} | "
                    f"{cm.sensitivity:.4f} | {cm.specificity:.4f} | "
                    f"{cm.f1:.4f} |"
                )
            lines.append("")
        else:
            # Risk task
            lines.append(f"- **C-index:** {tr.c_index:.4f}")
            lines.append(f"- **Brier Score:** {tr.brier_score:.4f}")
            lines.append("")

    # Demographic subgroup breakdowns
    if report.subgroups:
        lines.append("## Demographic Subgroup Breakdowns")
        lines.append("")

        # Separate sex and age subgroups
        sex_subgroups = [
            sg for sg in report.subgroups
            if sg.subgroup_name.startswith("sex_")
        ]
        age_subgroups = [
            sg for sg in report.subgroups
            if sg.subgroup_name.startswith("age_")
        ]

        if sex_subgroups:
            lines.append("### By Sex")
            lines.append("")
            _render_subgroup_table(lines, sex_subgroups, report.tasks_evaluated)
            lines.append("")

        if age_subgroups:
            lines.append("### By Age Decile")
            lines.append("")
            _render_subgroup_table(lines, age_subgroups, report.tasks_evaluated)
            lines.append("")

    # Equity gate results
    if equity_gate_result is not None:
        lines.append("## Equity Gate Results")
        lines.append("")
        lines.append(
            f"- **Status:** {'PASSED ✓' if equity_gate_result.passed else 'FAILED ✗'}"
        )
        lines.append(f"- **Alpha:** {equity_gate_result.alpha}")
        lines.append(f"- **Correction:** {equity_gate_result.correction}")
        lines.append(
            f"- **Corrected Alpha:** {equity_gate_result.corrected_alpha:.6f}"
        )
        lines.append(
            f"- **Total Comparisons:** {equity_gate_result.num_comparisons}"
        )
        lines.append(
            f"- **Failing Comparisons:** "
            f"{len(equity_gate_result.failing_comparisons)}"
        )
        lines.append("")

        if equity_gate_result.failing_comparisons:
            lines.append("#### Failing Comparisons")
            lines.append("")
            lines.append(
                "| Task | Class | Group A | AUC A | Group B | AUC B | "
                "Diff | p-value |"
            )
            lines.append(
                "|------|-------|---------|-------|---------|-------|"
                "------|---------|"
            )
            for fc in equity_gate_result.failing_comparisons:
                lines.append(
                    f"| {fc.task} | {fc.class_name} | {fc.group_a} | "
                    f"{fc.auc_a:.4f} | {fc.group_b} | {fc.auc_b:.4f} | "
                    f"{fc.auc_diff:.4f} | {fc.p_value:.4f} |"
                )
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "*This performance card was auto-generated by "
        "`aortica.evaluation.generate_performance_card()`.*"
    )
    lines.append("")

    return "\n".join(lines)


def _render_subgroup_table(
    lines: list[str],
    subgroups: list[SubgroupReport],
    tasks_evaluated: list[str],
) -> None:
    """Render a markdown table of subgroup metrics."""
    # Build header columns
    header_cols = ["Subgroup", "N"]
    for task in tasks_evaluated:
        if task in CLASSIFICATION_TASKS:
            header_cols.append(f"{task} Macro-F1")
            header_cols.append(f"{task} ECE")
        else:
            header_cols.append(f"{task} C-index")
            header_cols.append(f"{task} Brier")

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for sg in subgroups:
        row_cols: list[str] = [sg.subgroup_name, str(sg.n_samples)]
        for task in tasks_evaluated:
            tr = sg.task_reports.get(task)
            if tr is None:
                row_cols.append("—")
                row_cols.append("—")
            elif task in CLASSIFICATION_TASKS:
                row_cols.append(f"{tr.macro_f1:.4f}")
                row_cols.append(f"{tr.ece:.4f}")
            else:
                row_cols.append(f"{tr.c_index:.4f}")
                row_cols.append(f"{tr.brier_score:.4f}")
        lines.append("| " + " | ".join(row_cols) + " |")


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------


def _generate_csv(
    report: BenchmarkReport,
    model_version: str,
    timestamp: str,
    model_weights_sha256: str,
) -> str:
    """Generate the CSV content for the performance card."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    writer.writerow([
        "model_version",
        "timestamp",
        "weights_sha256",
        "n_samples",
        "subgroup",
        "task",
        "class",
        "auc",
        "sensitivity",
        "specificity",
        "f1",
        "macro_f1",
        "ece",
        "c_index",
        "brier_score",
    ])

    # Overall metrics
    _write_csv_rows(
        writer,
        report.overall,
        report.tasks_evaluated,
        subgroup_name="overall",
        model_version=model_version,
        timestamp=timestamp,
        sha256=model_weights_sha256,
        n_samples=report.n_samples,
    )

    # Subgroup metrics
    for sg in report.subgroups:
        _write_csv_rows(
            writer,
            sg.task_reports,
            report.tasks_evaluated,
            subgroup_name=sg.subgroup_name,
            model_version=model_version,
            timestamp=timestamp,
            sha256=model_weights_sha256,
            n_samples=sg.n_samples,
        )

    return buf.getvalue()


def _write_csv_rows(
    writer: Any,
    task_reports: dict[str, TaskReport],
    tasks_evaluated: list[str],
    *,
    subgroup_name: str,
    model_version: str,
    timestamp: str,
    sha256: str,
    n_samples: int,
) -> None:
    """Write CSV rows for a single subgroup or overall."""
    for task_name in tasks_evaluated:
        tr = task_reports.get(task_name)
        if tr is None:
            continue

        if task_name in CLASSIFICATION_TASKS:
            for cm in tr.per_class:
                writer.writerow([
                    model_version,
                    timestamp,
                    sha256,
                    n_samples,
                    subgroup_name,
                    task_name,
                    cm.name,
                    f"{cm.auc:.4f}",
                    f"{cm.sensitivity:.4f}",
                    f"{cm.specificity:.4f}",
                    f"{cm.f1:.4f}",
                    f"{tr.macro_f1:.4f}",
                    f"{tr.ece:.4f}",
                    "",
                    "",
                ])
        else:
            writer.writerow([
                model_version,
                timestamp,
                sha256,
                n_samples,
                subgroup_name,
                task_name,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{tr.c_index:.4f}",
                f"{tr.brier_score:.4f}",
            ])


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def generate_performance_card(
    benchmark_report: BenchmarkReport,
    model_version: str,
    output_dir: str,
    *,
    model_weights_path: Optional[str] = None,
    dataset_info: Optional[str] = None,
    equity_gate_result: Optional[Any] = None,
    timestamp: Optional[str] = None,
) -> PerformanceCardResult:
    """Generate a public performance card from a benchmark report.

    Produces two files in *output_dir*:

    - ``PERFORMANCE_CARD.md`` — human-readable markdown card with
      per-task metrics, demographic subgroup breakdowns, and equity
      gate results.
    - ``performance_card.csv`` — tabular data for programmatic use.

    Args:
        benchmark_report: A :class:`BenchmarkReport` from
            :func:`aortica.evaluation.benchmark`.
        model_version: Semantic version string (e.g. ``"0.3.0"``).
        output_dir: Directory to write output files.
        model_weights_path: Optional path to the model weights file for
            SHA-256 reproducibility hash.
        dataset_info: Optional string describing the dataset split
            (e.g. ``"PTB-XL test fold 10, 500 Hz"``).
        equity_gate_result: Optional :class:`EquityGateResult` from
            :func:`aortica.evaluation.equity_gate`.
        timestamp: Optional ISO-8601 timestamp.  Defaults to current
            UTC time.

    Returns:
        :class:`PerformanceCardResult` with paths and content.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Compute SHA-256 hash of model weights
    sha256 = ""
    if model_weights_path:
        sha256 = _compute_sha256(model_weights_path)

    # Generate markdown
    md_content = _generate_markdown(
        report=benchmark_report,
        model_version=model_version,
        timestamp=timestamp,
        model_weights_sha256=sha256,
        dataset_info=dataset_info,
        equity_gate_result=equity_gate_result,
    )

    # Generate CSV
    csv_content = _generate_csv(
        report=benchmark_report,
        model_version=model_version,
        timestamp=timestamp,
        model_weights_sha256=sha256,
    )

    # Write files
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "PERFORMANCE_CARD.md"
    csv_path = out / "performance_card.csv"

    md_path.write_text(md_content, encoding="utf-8")
    csv_path.write_text(csv_content, encoding="utf-8")

    return PerformanceCardResult(
        markdown_path=str(md_path.resolve()),
        csv_path=str(csv_path.resolve()),
        markdown_content=md_content,
        csv_content=csv_content,
        model_version=model_version,
        timestamp=timestamp,
        model_weights_sha256=sha256,
    )
