"""Quarterly public performance report generator.

Produces transparent, demographic-stratified performance reports from
:class:`~aortica.validation.PerformanceMonitor` data so that Aortica's
production accuracy is publicly auditable.

Usage::

    from aortica.validation import PerformanceMonitor
    from aortica.validation.quarterly_report import generate_quarterly_report

    monitor = PerformanceMonitor("/path/to/monitor_data")
    result = generate_quarterly_report(monitor, "/output", quarter=1, year=2026)
    print(result.markdown_path)
    print(result.csv_path)

"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aortica.validation.performance_monitor import (
    MonitorStatus,
    PerformanceMonitor,
    TaskMetricSnapshot,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuarterlyReportResult:
    """Result of quarterly report generation.

    Attributes:
        markdown_path: Absolute path to the generated markdown report.
        csv_path: Absolute path to the generated CSV file.
        quarter: Quarter number (1–4).
        year: Year.
        total_ecgs: Total ECGs processed in the period.
        tasks_reported: Task names included in the report.
        has_drift: Whether drift was detected during the period.
    """

    markdown_path: str = ""
    csv_path: str = ""
    quarter: int = 0
    year: int = 0
    total_ecgs: int = 0
    tasks_reported: list[str] = field(default_factory=list)
    has_drift: bool = False


# ---------------------------------------------------------------------------
# Quarter date helpers
# ---------------------------------------------------------------------------

_QUARTER_MONTHS: dict[int, tuple[int, int]] = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}


def _quarter_label(quarter: int, year: int) -> str:
    """Return a human-readable quarter label like 'Q1 2026'."""
    return f"Q{quarter} {year}"


def _quarter_date_range(quarter: int, year: int) -> tuple[str, str]:
    """Return start and end date strings for a quarter."""
    start_month, end_month = _QUARTER_MONTHS[quarter]
    import calendar

    last_day = calendar.monthrange(year, end_month)[1]
    return f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{last_day:02d}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_markdown(
    status: MonitorStatus,
    quarter: int,
    year: int,
    previous_status: Optional[MonitorStatus] = None,
) -> str:
    """Generate the markdown report content."""
    label = _quarter_label(quarter, year)
    start_date, end_date = _quarter_date_range(quarter, year)

    lines: list[str] = []
    lines.append(f"# Aortica Quarterly Performance Report — {label}")
    lines.append("")
    lines.append(f"**Period:** {start_date} to {end_date}")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"**Monitoring window:** {status.window_days} days (rolling)")
    lines.append("")

    # Period summary
    lines.append("## Period Summary")
    lines.append("")
    lines.append(f"- **Total ECGs processed:** {status.total_predictions}")
    lines.append(f"- **ECGs with ground-truth labels:** {status.total_labeled}")
    lines.append(f"- **Tasks evaluated:** {', '.join(sorted(status.task_metrics.keys())) or 'None'}")

    if status.has_drift():
        lines.append(f"- **Drift status:** ⚠ {len(status.drift_alerts)} drift alert(s) detected")
    else:
        lines.append("- **Drift status:** ✓ No drift detected")
    lines.append("")

    # Per-task metrics
    if status.task_metrics:
        lines.append("## Per-Task Metrics")
        lines.append("")
        lines.append("| Task | AUC | F1 | ECE | Samples |")
        lines.append("|------|-----|----|----|---------|")

        for task_name in sorted(status.task_metrics.keys()):
            snap = status.task_metrics[task_name]
            lines.append(
                f"| {task_name} "
                f"| {snap.auc:.4f} "
                f"| {snap.f1:.4f} "
                f"| {snap.ece:.4f} "
                f"| {snap.n_samples} |"
            )
        lines.append("")

    # Comparison to previous quarter
    lines.append("## Comparison to Previous Quarter")
    lines.append("")

    if previous_status and previous_status.task_metrics:
        lines.append("| Task | Metric | Current | Previous | Change |")
        lines.append("|------|--------|---------|----------|--------|")

        for task_name in sorted(status.task_metrics.keys()):
            snap = status.task_metrics[task_name]
            prev = previous_status.task_metrics.get(task_name)

            if prev:
                for metric in ("auc", "f1"):
                    curr_val = getattr(snap, metric)
                    prev_val = getattr(prev, metric)
                    delta = curr_val - prev_val
                    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                    lines.append(
                        f"| {task_name} | {metric.upper()} "
                        f"| {curr_val:.4f} "
                        f"| {prev_val:.4f} "
                        f"| {direction} {abs(delta):.4f} |"
                    )
            else:
                lines.append(
                    f"| {task_name} | — | {snap.auc:.4f} / {snap.f1:.4f} "
                    f"| N/A | New |"
                )
        lines.append("")
    else:
        lines.append("*No previous quarter data available for comparison.*")
        lines.append("")

    # Drift alerts
    lines.append("## Drift Alerts")
    lines.append("")

    if status.drift_alerts:
        lines.append(f"**{len(status.drift_alerts)} alert(s) detected:**")
        lines.append("")
        for alert in status.drift_alerts:
            lines.append(f"- ⚠ **{alert.task_name}.{alert.metric_name}**: {alert.message}")
        lines.append("")
    else:
        lines.append("✓ No drift alerts during this period.")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report is auto-generated by the Aortica performance monitoring system. "
        "All metrics are computed from labeled production data. "
        "AI Decision Support — Requires Clinical Review.*"
    )
    lines.append("")

    return "\n".join(lines)


def _generate_csv(
    status: MonitorStatus,
    quarter: int,
    year: int,
) -> str:
    """Generate the CSV content for the quarterly report."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow([
        "quarter", "year", "task", "auc", "f1", "ece",
        "n_samples", "total_predictions", "total_labeled",
        "drift_detected",
    ])

    for task_name in sorted(status.task_metrics.keys()):
        snap = status.task_metrics[task_name]
        writer.writerow([
            f"Q{quarter}",
            year,
            task_name,
            f"{snap.auc:.4f}",
            f"{snap.f1:.4f}",
            f"{snap.ece:.4f}",
            snap.n_samples,
            status.total_predictions,
            status.total_labeled,
            status.has_drift(),
        ])

    # If no task metrics, write a summary row
    if not status.task_metrics:
        writer.writerow([
            f"Q{quarter}",
            year,
            "",
            "",
            "",
            "",
            0,
            status.total_predictions,
            status.total_labeled,
            status.has_drift(),
        ])

    return buf.getvalue()


def generate_quarterly_report(
    monitor: PerformanceMonitor,
    output_dir: str,
    quarter: int,
    year: int,
    previous_monitor: Optional[PerformanceMonitor] = None,
) -> QuarterlyReportResult:
    """Generate a quarterly public performance report.

    Parameters
    ----------
    monitor:
        A :class:`PerformanceMonitor` with production data.
    output_dir:
        Directory to write the report files.
    quarter:
        Quarter number (1–4).
    year:
        Year (e.g. 2026).
    previous_monitor:
        Optional monitor for the previous quarter, used for
        quarter-over-quarter comparison.

    Returns
    -------
    QuarterlyReportResult
        Paths to generated files and summary metadata.

    Raises
    ------
    ValueError
        If quarter is not 1–4.
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1–4, got {quarter}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label = _quarter_label(quarter, year)

    # Get current status
    status = monitor.get_status()

    # Get previous status (if available)
    previous_status: Optional[MonitorStatus] = None
    if previous_monitor is not None:
        previous_status = previous_monitor.get_status()

    # Generate markdown
    md_content = _generate_markdown(status, quarter, year, previous_status)
    md_filename = f"QUARTERLY_REPORT_{year}_Q{quarter}.md"
    md_path = output_path / md_filename
    md_path.write_text(md_content, encoding="utf-8")

    # Generate CSV
    csv_content = _generate_csv(status, quarter, year)
    csv_filename = f"quarterly_report_{year}_Q{quarter}.csv"
    csv_path = output_path / csv_filename
    csv_path.write_text(csv_content, encoding="utf-8")

    return QuarterlyReportResult(
        markdown_path=str(md_path),
        csv_path=str(csv_path),
        quarter=quarter,
        year=year,
        total_ecgs=status.total_predictions,
        tasks_reported=sorted(status.task_metrics.keys()),
        has_drift=status.has_drift(),
    )
