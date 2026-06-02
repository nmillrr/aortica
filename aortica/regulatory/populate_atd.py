"""Auto-population script for IEC 80601-2-86 Algorithm Testing Documentation.

Fills performance metric sections of the ATD template from a
:class:`~aortica.evaluation.benchmark.BenchmarkReport`, replacing
``[FILL: auto-populated ...]`` placeholders with actual metric tables.

Usage::

    from aortica.evaluation import benchmark, BenchmarkReport
    from aortica.regulatory import populate_atd

    result = populate_atd(benchmark_report, model_version="0.3.0")
    print(result.output_path)

Also provides :func:`validate_atd_completeness` to check whether any
``[FILL:]`` markers remain in a populated ATD — used by CI to block
v-stable releases with incomplete documentation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aortica.evaluation.benchmark import (
    CLASSIFICATION_TASKS,
    BenchmarkReport,
    ClassMetrics,
    TaskReport,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "docs",
    "regulatory",
    "IEC_80601_2_86_ATD.md",
)

# Regex matching ``[FILL: auto-populated ...]`` markers.
_FILL_MARKER_RE = re.compile(r"\[FILL:[^\]]*\]")

# Regex matching *only* auto-populated fill markers.
_AUTO_FILL_RE = re.compile(r"\[FILL:\s*auto-populated[^\]]*\]")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ATDPopulationResult:
    """Result of ATD auto-population.

    Attributes:
        output_path: Path to the generated/populated ATD file.
        content: Full text of the populated ATD.
        model_version: Model version string used.
        sections_populated: Number of auto-populated sections filled.
        remaining_fill_markers: Number of ``[FILL:]`` markers still
            present (site-specific fields the user must fill manually).
    """

    output_path: str = ""
    content: str = ""
    model_version: str = ""
    sections_populated: int = 0
    remaining_fill_markers: int = 0


@dataclass
class ATDValidationResult:
    """Result of ATD completeness validation.

    Attributes:
        is_complete: ``True`` if no ``[FILL:]`` markers remain.
        remaining_markers: List of remaining ``[FILL:]`` strings.
        marker_count: Count of remaining markers.
    """

    is_complete: bool = True
    remaining_markers: list[str] = field(default_factory=list)
    marker_count: int = 0


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------


def _render_overall_summary_table(report: BenchmarkReport) -> str:
    """Render the § 4.1 overall performance summary table."""
    lines: list[str] = []
    lines.append("| Task | Metric | Value |")
    lines.append("|------|--------|-------|")

    for task_name in report.tasks_evaluated:
        tr = report.overall.get(task_name)
        if tr is None:
            continue

        task_label = task_name.replace("_", " ").title()

        if task_name in CLASSIFICATION_TASKS:
            lines.append(
                f"| {task_label} | Macro-F1 | {tr.macro_f1:.4f} |"
            )
            lines.append(
                f"| {task_label} | ECE | {tr.ece:.4f} |"
            )
        else:
            lines.append(
                f"| {task_label} | C-index | {tr.c_index:.4f} |"
            )
            lines.append(
                f"| {task_label} | Brier Score | {tr.brier_score:.4f} |"
            )

    return "\n".join(lines)


def _render_classification_table(tr: TaskReport) -> str:
    """Render a per-class classification metrics table."""
    lines: list[str] = []
    lines.append("| Class | AUC | Sensitivity | Specificity | F1 |")
    lines.append("|-------|-----|-------------|-------------|-----|")

    for cm in tr.per_class:
        lines.append(
            f"| {cm.name} | {cm.auc:.4f} | {cm.sensitivity:.4f} "
            f"| {cm.specificity:.4f} | {cm.f1:.4f} |"
        )

    return "\n".join(lines)


def _render_risk_table(tr: TaskReport) -> str:
    """Render the risk prediction metrics table."""
    # Risk head doesn't have per-class metrics in the same way.
    # Render overall C-index and Brier.
    lines: list[str] = []
    lines.append("| Output | C-index | Brier Score |")
    lines.append("|--------|---------|-------------|")

    if tr.per_class:
        # If per-class entries exist (named risk outputs), list them.
        for cm in tr.per_class:
            lines.append(
                f"| {cm.name} | — | — |"
            )
    # Always include overall row.
    lines.append(
        f"| **Overall** | {tr.c_index:.4f} | {tr.brier_score:.4f} |"
    )

    return "\n".join(lines)


def _render_calibration_table(report: BenchmarkReport) -> str:
    """Render the § 4.6 calibration table."""
    lines: list[str] = []
    lines.append("| Task | Expected Calibration Error (ECE) |")
    lines.append("|------|----------------------------------|")

    for task_name in report.tasks_evaluated:
        tr = report.overall.get(task_name)
        if tr is None:
            continue
        if task_name in CLASSIFICATION_TASKS:
            task_label = task_name.replace("_", " ").title()
            lines.append(f"| {task_label} | {tr.ece:.4f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section replacement map
# ---------------------------------------------------------------------------

# Maps the ``[FILL: auto-populated ...]`` marker text to a callable that
# produces the replacement content from a BenchmarkReport.

_SECTION_MAP: dict[str, str] = {
    "[FILL: auto-populated performance metrics table]": "overall_summary",
    "[FILL: auto-populated rhythm class metrics]": "rhythm",
    "[FILL: auto-populated structural class metrics]": "structural",
    "[FILL: auto-populated ischaemia class metrics]": "ischaemia",
    "[FILL: auto-populated risk metrics]": "risk",
    "[FILL: auto-populated calibration metrics]": "calibration",
}


def _generate_replacement(
    section_key: str,
    report: BenchmarkReport,
) -> str:
    """Generate replacement text for a given section key."""
    if section_key == "overall_summary":
        return _render_overall_summary_table(report)
    elif section_key in CLASSIFICATION_TASKS:
        tr = report.overall.get(section_key)
        if tr is None:
            return f"*No {section_key} metrics available.*"
        return _render_classification_table(tr)
    elif section_key == "risk":
        tr = report.overall.get("risk")
        if tr is None:
            return "*No risk metrics available.*"
        return _render_risk_table(tr)
    elif section_key == "calibration":
        return _render_calibration_table(report)
    else:
        return f"*Section '{section_key}' not found.*"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def populate_atd(
    benchmark_report: BenchmarkReport,
    model_version: str,
    *,
    template_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> ATDPopulationResult:
    """Fill auto-populated sections of the IEC 80601-2-86 ATD template.

    Reads the ATD markdown template, replaces ``[FILL: auto-populated ...]``
    markers with metric tables derived from *benchmark_report*, and writes
    the result to *output_path*.

    Args:
        benchmark_report: A :class:`BenchmarkReport` from
            :func:`aortica.evaluation.benchmark`.
        model_version: Semantic version string (e.g. ``"0.3.0"``).
        template_path: Path to the ATD template.  Defaults to
            ``docs/regulatory/IEC_80601_2_86_ATD.md`` relative to the
            repository root.
        output_path: Where to write the populated ATD.  Defaults to
            the same directory as the template with filename
            ``IEC_80601_2_86_ATD_v{model_version}.md``.

    Returns:
        :class:`ATDPopulationResult` with output path and content.
    """
    # Resolve template path
    if template_path is None:
        template_path = os.path.normpath(_DEFAULT_TEMPLATE_PATH)

    template_text = Path(template_path).read_text(encoding="utf-8")

    # Replace auto-populated markers
    content = template_text
    sections_populated = 0

    for marker, section_key in _SECTION_MAP.items():
        if marker in content:
            replacement = _generate_replacement(section_key, benchmark_report)
            content = content.replace(marker, replacement)
            sections_populated += 1

    # Also fill the model version in document control
    # Replace the first FILL for model version
    content = content.replace(
        "[FILL: aortica model version]",
        model_version,
    )
    if "[FILL: aortica model version]" != model_version:
        # Count this as a populated section if it was replaced
        if "[FILL: aortica model version]" in template_text:
            sections_populated += 1

    # Fill document date
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = content.replace("[FILL: document date]", date_str)
    if "[FILL: document date]" in template_text:
        sections_populated += 1

    # Count remaining FILL markers
    remaining = _FILL_MARKER_RE.findall(content)

    # Resolve output path
    if output_path is None:
        template_dir = os.path.dirname(template_path)
        output_path = os.path.join(
            template_dir,
            f"IEC_80601_2_86_ATD_v{model_version}.md",
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")

    return ATDPopulationResult(
        output_path=output_path,
        content=content,
        model_version=model_version,
        sections_populated=sections_populated,
        remaining_fill_markers=len(remaining),
    )


def validate_atd_completeness(
    atd_path: str,
) -> ATDValidationResult:
    """Validate that an ATD document has no remaining ``[FILL:]`` markers.

    Used by CI to block v-stable releases with incomplete documentation.

    Args:
        atd_path: Path to the ATD markdown file to validate.

    Returns:
        :class:`ATDValidationResult` with completeness status.
    """
    text = Path(atd_path).read_text(encoding="utf-8")
    markers = _FILL_MARKER_RE.findall(text)

    return ATDValidationResult(
        is_complete=len(markers) == 0,
        remaining_markers=markers,
        marker_count=len(markers),
    )
