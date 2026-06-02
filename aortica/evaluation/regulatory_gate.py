"""Regulatory performance gate for CI release enforcement.

Reads per-class minimum performance targets from a YAML file and
compares them against a :class:`~aortica.evaluation.benchmark.BenchmarkReport`.
Blocks a release if any metric falls below its configured threshold.

Default targets (from ``regulatory_targets.yaml`` at the repository root):
    - STEMI sensitivity ≥ 0.90
    - AF AUC ≥ 0.95
    - LVSD AUC ≥ 0.88
    - Overall rhythm macro-F1 ≥ 0.90

Usage::

    from aortica.evaluation.regulatory_gate import regulatory_gate

    result = regulatory_gate(benchmark_report, "regulatory_targets.yaml")
    if not result.passed:
        print(result.summary())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]

from aortica.evaluation.benchmark import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    BenchmarkReport,
    ClassMetrics,
    TaskReport,
)


# ---------------------------------------------------------------------------
# Default targets path
# ---------------------------------------------------------------------------

_DEFAULT_TARGETS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "regulatory_targets.yaml",
    )
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassGateResult:
    """Result for a single class/metric comparison.

    Attributes:
        class_name: The class or task being checked.
        metric_name: The metric type (``auc``, ``sensitivity``, etc.).
        target: The minimum required value.
        actual: The actual value from the benchmark report.
        passed: ``True`` if *actual* ≥ *target*.
    """

    class_name: str = ""
    metric_name: str = ""
    target: float = 0.0
    actual: float = 0.0
    passed: bool = True


@dataclass
class RegulatoryGateResult:
    """Aggregate result of the regulatory performance gate.

    Attributes:
        passed: ``True`` if **all** class-level checks passed.
        per_class: List of per-class / per-metric results.
        num_passed: Count of passing checks.
        num_failed: Count of failing checks.
        targets_path: Path to the YAML targets file used.
    """

    passed: bool = True
    per_class: list[ClassGateResult] = field(default_factory=list)
    num_passed: int = 0
    num_failed: int = 0
    targets_path: str = ""

    def summary(self) -> str:
        """Return a human-readable summary of the gate result."""
        lines: list[str] = []
        status = "PASS ✅" if self.passed else "FAIL ❌"
        lines.append(f"Regulatory Gate: {status}")
        lines.append(f"  Checks passed: {self.num_passed}")
        lines.append(f"  Checks failed: {self.num_failed}")

        if self.num_failed > 0:
            lines.append("")
            lines.append("  Failed checks:")
            for r in self.per_class:
                if not r.passed:
                    lines.append(
                        f"    - {r.class_name} {r.metric_name}: "
                        f"actual={r.actual:.4f} < target={r.target:.4f}"
                    )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_targets(targets_path: str) -> dict[str, dict[str, float]]:
    """Load regulatory targets from a YAML file.

    Returns a dict like::

        {
            "sensitivity": {"STEMI": 0.90, ...},
            "auc": {"AF": 0.95, ...},
            "macro_f1": {"rhythm": 0.90},
        }
    """
    text = Path(targets_path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(
            f"Regulatory targets YAML must be a mapping, got {type(data).__name__}"
        )

    result: dict[str, dict[str, float]] = {}
    for metric_name, entries in data.items():
        if not isinstance(entries, dict):
            raise ValueError(
                f"Expected a mapping under '{metric_name}', "
                f"got {type(entries).__name__}"
            )
        result[metric_name] = {
            str(k): float(v) for k, v in entries.items()
        }

    return result


def _find_class_metric(
    report: BenchmarkReport,
    class_name: str,
    metric_name: str,
) -> Optional[float]:
    """Look up a per-class metric value from the benchmark report.

    For ``macro_f1``, *class_name* is treated as a task name.
    For ``sensitivity``, ``specificity``, ``auc``, and ``f1``,
    *class_name* is a per-class label and we search across all
    classification tasks.
    """
    if metric_name == "macro_f1":
        # Task-level metric
        tr = report.overall.get(class_name)
        if tr is None:
            return None
        return tr.macro_f1

    # Per-class metric — search all tasks
    for task_name in report.tasks_evaluated:
        if task_name not in CLASSIFICATION_TASKS:
            continue
        tr = report.overall.get(task_name)
        if tr is None:
            continue
        for cm in tr.per_class:
            if cm.name == class_name:
                return getattr(cm, metric_name, None)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def regulatory_gate(
    benchmark_report: BenchmarkReport,
    targets_yaml: Optional[str] = None,
) -> RegulatoryGateResult:
    """Run the regulatory performance gate.

    Compares each per-class metric target from *targets_yaml* against
    the actual metrics in *benchmark_report*.  Returns a
    :class:`RegulatoryGateResult` with per-class pass/fail and an
    overall pass/fail flag.

    Args:
        benchmark_report: A :class:`BenchmarkReport` from
            :func:`aortica.evaluation.benchmark`.
        targets_yaml: Path to a YAML file with per-class targets.
            Defaults to ``regulatory_targets.yaml`` at the repository
            root.

    Returns:
        :class:`RegulatoryGateResult`.

    Raises:
        FileNotFoundError: If *targets_yaml* does not exist.
        ValueError: If the YAML is malformed.
    """
    if targets_yaml is None:
        targets_yaml = _DEFAULT_TARGETS_PATH

    targets = _load_targets(targets_yaml)

    per_class: list[ClassGateResult] = []
    num_passed = 0
    num_failed = 0

    for metric_name, entries in targets.items():
        for class_name, target_value in entries.items():
            actual = _find_class_metric(
                benchmark_report, class_name, metric_name
            )

            if actual is None:
                # Class not found in benchmark — treat as missing (fail)
                result = ClassGateResult(
                    class_name=class_name,
                    metric_name=metric_name,
                    target=target_value,
                    actual=0.0,
                    passed=False,
                )
                num_failed += 1
            else:
                passed = actual >= target_value
                result = ClassGateResult(
                    class_name=class_name,
                    metric_name=metric_name,
                    target=target_value,
                    actual=actual,
                    passed=passed,
                )
                if passed:
                    num_passed += 1
                else:
                    num_failed += 1

            per_class.append(result)

    return RegulatoryGateResult(
        passed=num_failed == 0,
        per_class=per_class,
        num_passed=num_passed,
        num_failed=num_failed,
        targets_path=targets_yaml,
    )
