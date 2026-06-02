"""AI reporting guideline checklist generation utilities.

Provides :func:`generate_reporting_checklist` which reads a TRIPOD-AI,
STARD-AI, or CONSORT-AI markdown template and optionally pre-fills
sections that can be derived from a
:class:`~aortica.evaluation.benchmark.BenchmarkReport`.

Usage::

    from aortica.regulatory.reporting_checklists import generate_reporting_checklist

    result = generate_reporting_checklist(
        template="tripod_ai",
        benchmark_report=report,
        model_version="0.3.0",
    )
    print(result.output_path)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from aortica.evaluation.benchmark import (
    CLASSIFICATION_TASKS,
    BenchmarkReport,
    TaskReport,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEMPLATE_NAMES: dict[str, str] = {
    "tripod_ai": "TRIPOD_AI.md",
    "stard_ai": "STARD_AI.md",
    "consort_ai": "CONSORT_AI.md",
}

_VALID_TEMPLATES = Literal["tripod_ai", "stard_ai", "consort_ai"]

_REGULATORY_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "docs",
        "regulatory",
    )
)

_FILL_MARKER_RE = re.compile(r"\[FILL:[^\]]*\]")
_AUTO_FILL_RE = re.compile(r"\[FILL:\s*auto-populated[^\]]*\]")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChecklistResult:
    """Result of reporting checklist generation.

    Attributes:
        output_path: Path to the generated checklist file.
        content: Full text of the (optionally pre-filled) checklist.
        template_name: Template key used (e.g. ``'tripod_ai'``).
        model_version: Model version string (if provided).
        sections_populated: Number of auto-populated sections filled.
        total_checklist_items: Total number of checklist items found.
        completed_items: Number of items marked ``[x]``.
        remaining_items: Number of items still marked ``[ ]``.
        remaining_fill_markers: Number of ``[FILL:]`` markers remaining.
    """

    output_path: str = ""
    content: str = ""
    template_name: str = ""
    model_version: str = ""
    sections_populated: int = 0
    total_checklist_items: int = 0
    completed_items: int = 0
    remaining_items: int = 0
    remaining_fill_markers: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_checklist_items(content: str) -> tuple[int, int, int]:
    """Count total, completed, and remaining checklist items.

    Returns:
        Tuple of (total, completed, remaining).
    """
    completed = len(re.findall(r"\[x\]", content))
    remaining = len(re.findall(r"\[ \]", content))
    total = completed + remaining
    return total, completed, remaining


def _populate_benchmark_fields(
    content: str,
    report: BenchmarkReport,
) -> tuple[str, int]:
    """Replace auto-populated FILL markers with benchmark data.

    Returns:
        Tuple of (updated content, number of sections populated).
    """
    sections_populated = 0

    # Populate dataset info
    marker = "[FILL: auto-populated from benchmark report — PTB-XL: single German centre, 21,799 10-second 12-lead ECGs, CC BY 4.0]"
    if marker in content:
        replacement = (
            f"PTB-XL dataset (Wagner et al. 2020, PhysioNet). "
            f"Single German centre, ~21,799 10-second 12-lead ECGs, CC BY 4.0 license. "
            f"Benchmark evaluated {report.n_samples} samples across tasks: "
            f"{', '.join(report.tasks_evaluated)}."
        )
        content = content.replace(marker, replacement)
        sections_populated += 1

    # Populate sample size info
    marker_sample = "[FILL: auto-populated — PTB-XL full dataset: ~21,799 records. Split: folds 1-8 train (~17,440), fold 9 validation (~2,180), fold 10 test (~2,180). Rare class N reported per class.]"
    if marker_sample in content:
        replacement = (
            f"PTB-XL full dataset: ~21,799 records. "
            f"Official recommended split: folds 1–8 train (~17,440), "
            f"fold 9 validation (~2,180), fold 10 test (~2,180). "
            f"Benchmark test set: {report.n_samples} samples."
        )
        content = content.replace(marker_sample, replacement)
        sections_populated += 1

    # Populate overall metrics
    overall_markers = [
        "[FILL: auto-populated — macro-F1, mean AUC with 95% CI per task head from benchmark report]",
        "[FILL: auto-populated — macro-F1, mean AUC with 95% CI per task head]",
    ]
    for marker in overall_markers:
        if marker in content:
            lines = []
            for task_name in report.tasks_evaluated:
                tr = report.overall.get(task_name)
                if tr is None:
                    continue
                task_label = task_name.replace("_", " ").title()
                if task_name in CLASSIFICATION_TASKS:
                    lines.append(
                        f"{task_label}: Macro-F1 = {tr.macro_f1:.4f}, "
                        f"ECE = {tr.ece:.4f}"
                    )
                else:
                    lines.append(
                        f"{task_label}: C-index = {tr.c_index:.4f}, "
                        f"Brier = {tr.brier_score:.4f}"
                    )
            replacement = ". ".join(lines) + "." if lines else "No benchmark data available."
            content = content.replace(marker, replacement)
            sections_populated += 1

    # Populate per-class metrics
    per_class_marker = "[FILL: auto-populated — per-class AUC, sensitivity, specificity, F1 for all 72 outputs. See IEC 80601-2-86 ATD § 4.2–4.5]"
    if per_class_marker in content:
        lines = []
        for task_name in report.tasks_evaluated:
            tr = report.overall.get(task_name)
            if tr is None or not tr.per_class:
                continue
            task_label = task_name.replace("_", " ").title()
            lines.append(f"**{task_label}**:")
            for cm in tr.per_class:
                lines.append(
                    f"  - {cm.name}: AUC={cm.auc:.4f}, "
                    f"Sens={cm.sensitivity:.4f}, "
                    f"Spec={cm.specificity:.4f}, "
                    f"F1={cm.f1:.4f}"
                )
        replacement = "\n".join(lines) if lines else "No per-class metrics available."
        content = content.replace(per_class_marker, replacement)
        sections_populated += 1

    # Populate calibration
    cal_marker = "[FILL: auto-populated — per-task ECE. See IEC 80601-2-86 ATD § 4.6]"
    if cal_marker in content:
        lines = []
        for task_name in report.tasks_evaluated:
            tr = report.overall.get(task_name)
            if tr is None:
                continue
            if task_name in CLASSIFICATION_TASKS:
                task_label = task_name.replace("_", " ").title()
                lines.append(f"{task_label}: ECE = {tr.ece:.4f}")
        replacement = ". ".join(lines) + "." if lines else "No calibration data available."
        content = content.replace(cal_marker, replacement)
        sections_populated += 1

    # Populate subgroup metrics
    subgroup_marker = "[FILL: auto-populated — per-task AUC by sex and age decile from equity gate report]"
    if subgroup_marker in content:
        if report.subgroups:
            replacement = f"Subgroup analysis available for {len(report.subgroups)} subgroups. See equity gate report for detailed per-task AUC by sex and age decile."
        else:
            replacement = "Subgroup analysis: see equity gate report for per-task AUC by sex and age decile."
        content = content.replace(subgroup_marker, replacement, 1)
        sections_populated += 1
        # Handle any remaining occurrences
        if subgroup_marker in content:
            content = content.replace(subgroup_marker, replacement)

    # Populate participant demographics
    demo_marker = "[FILL: auto-populated — age distribution (mean, SD, range), sex distribution, prevalence of each condition in the study population]"
    if demo_marker in content:
        replacement = f"Study population: {report.n_samples} ECG recordings. See benchmark report for detailed demographic breakdown."
        content = content.replace(demo_marker, replacement)
        sections_populated += 1

    # Populate total records / exclusion
    records_marker = "[FILL: auto-populated — total records, excluded records with reasons, final analysis set size]"
    if records_marker in content:
        replacement = f"Total records in analysis: {report.n_samples}. See benchmark report for exclusion details."
        content = content.replace(records_marker, replacement)
        sections_populated += 1

    # Populate N per class/split
    n_marker = "[FILL: auto-populated — N total, N per split, N per demographic subgroup, N per class]"
    if n_marker in content:
        replacement = f"Total analysed: {report.n_samples}. Tasks evaluated: {', '.join(report.tasks_evaluated)}. See benchmark report for per-class and per-subgroup sample sizes."
        content = content.replace(n_marker, replacement)
        sections_populated += 1

    return content, sections_populated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_reporting_checklist(
    template: str = "tripod_ai",
    benchmark_report: Optional[BenchmarkReport] = None,
    model_version: Optional[str] = None,
    *,
    output_path: Optional[str] = None,
) -> ChecklistResult:
    """Generate a partially pre-filled AI reporting checklist.

    Reads the specified template (TRIPOD-AI, STARD-AI, or CONSORT-AI)
    and optionally fills auto-populated sections from a benchmark report.

    Args:
        template: One of ``'tripod_ai'``, ``'stard_ai'``, or
            ``'consort_ai'``.
        benchmark_report: Optional :class:`BenchmarkReport` for
            auto-populating performance fields.
        model_version: Optional model version string to inject.
        output_path: Where to write the generated checklist.  Defaults
            to the regulatory docs directory with a versioned filename.

    Returns:
        :class:`ChecklistResult` with the generated content and metadata.

    Raises:
        ValueError: If *template* is not a recognised template name.
        FileNotFoundError: If the template file does not exist.
    """
    template_key = template.lower().strip()
    if template_key not in _TEMPLATE_NAMES:
        raise ValueError(
            f"Unknown template '{template}'. "
            f"Valid templates: {list(_TEMPLATE_NAMES.keys())}"
        )

    template_filename = _TEMPLATE_NAMES[template_key]
    template_path = os.path.join(_REGULATORY_DIR, template_filename)

    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            f"Template file not found: {template_path}"
        )

    content = Path(template_path).read_text(encoding="utf-8")
    sections_populated = 0

    # Inject model version
    if model_version is not None:
        old = "[FILL: aortica model version]"
        if old in content:
            content = content.replace(old, model_version)
            sections_populated += 1

    # Inject date
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    old_date = "[FILL: document date]"
    if old_date in content:
        content = content.replace(old_date, date_str)
        sections_populated += 1

    # Populate benchmark fields
    if benchmark_report is not None:
        content, bench_populated = _populate_benchmark_fields(
            content, benchmark_report
        )
        sections_populated += bench_populated

    # Count items
    total, completed, remaining = _count_checklist_items(content)

    # Count remaining FILL markers
    fill_markers = _FILL_MARKER_RE.findall(content)

    # Resolve output path
    if output_path is None:
        version_suffix = f"_v{model_version}" if model_version else ""
        base_name = template_filename.replace(
            ".md", f"{version_suffix}.md"
        )
        output_path = os.path.join(_REGULATORY_DIR, base_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")

    return ChecklistResult(
        output_path=output_path,
        content=content,
        template_name=template_key,
        model_version=model_version or "",
        sections_populated=sections_populated,
        total_checklist_items=total,
        completed_items=completed,
        remaining_items=remaining,
        remaining_fill_markers=len(fill_markers),
    )
