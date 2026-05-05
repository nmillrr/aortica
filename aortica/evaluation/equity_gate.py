"""Equity gating check for model releases (US-069).

Compares per-task AUC across demographic subgroups (sex and age deciles)
using permutation tests with Bonferroni correction.  Designed to run
in CI and block releases that exhibit statistically significant
demographic performance disparities.

Usage::

    from aortica.evaluation import equity_gate

    result = equity_gate(benchmark_report, alpha=0.05, correction='bonferroni')
    if not result.passed:
        print("Equity gate FAILED — release blocked.")
        for f in result.failing_comparisons:
            print(f)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    CLASSIFICATION_TASKS,
    SubgroupReport,
    _compute_auc,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupMetrics:
    """Metrics for a single demographic group within a comparison.

    Attributes:
        group_name: Subgroup identifier (e.g. "sex_M", "age_30-39").
        task: Task name being evaluated.
        auc: Per-task AUC for this group.
        n_samples: Number of samples in this group.
    """

    group_name: str = ""
    task: str = ""
    auc: float = 0.0
    n_samples: int = 0


@dataclass
class ComparisonResult:
    """Result of a pairwise subgroup comparison.

    Attributes:
        group_a: First subgroup identifier.
        group_b: Second subgroup identifier.
        task: Task being compared.
        class_index: Index of the class being compared.
        class_name: Name of the class being compared.
        auc_a: AUC for group A.
        auc_b: AUC for group B.
        auc_diff: Absolute difference in AUC.
        p_value: P-value from permutation test.
        significant: Whether the difference is statistically significant
            after correction.
        n_a: Number of samples in group A.
        n_b: Number of samples in group B.
    """

    group_a: str = ""
    group_b: str = ""
    task: str = ""
    class_index: int = 0
    class_name: str = ""
    auc_a: float = 0.0
    auc_b: float = 0.0
    auc_diff: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    n_a: int = 0
    n_b: int = 0


@dataclass
class EquityGateResult:
    """Aggregate result of the equity gate check.

    Attributes:
        passed: Overall pass/fail status.
        alpha: Significance level used (before correction).
        correction: Multiple comparison correction method applied.
        corrected_alpha: Effective per-test alpha after correction.
        num_comparisons: Total number of comparisons performed.
        comparisons: All pairwise comparisons with p-values.
        failing_comparisons: Comparisons that failed the equity gate.
        per_group_metrics: Per-group AUC summaries.
    """

    passed: bool = True
    alpha: float = 0.05
    correction: str = "bonferroni"
    corrected_alpha: float = 0.05
    num_comparisons: int = 0
    comparisons: list[ComparisonResult] = field(default_factory=list)
    failing_comparisons: list[ComparisonResult] = field(default_factory=list)
    per_group_metrics: list[GroupMetrics] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the equity gate result."""
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        lines: list[str] = [
            f"Equity Gate: {status}",
            f"  Alpha: {self.alpha} (corrected: {self.corrected_alpha:.6f})",
            f"  Correction: {self.correction}",
            f"  Comparisons: {self.num_comparisons}",
            f"  Failing: {len(self.failing_comparisons)}",
        ]

        if self.failing_comparisons:
            lines.append("\n  Failing comparisons:")
            for fc in self.failing_comparisons:
                lines.append(
                    f"    {fc.task}/{fc.class_name}: "
                    f"{fc.group_a} (AUC={fc.auc_a:.4f}, n={fc.n_a}) vs "
                    f"{fc.group_b} (AUC={fc.auc_b:.4f}, n={fc.n_b}) — "
                    f"diff={fc.auc_diff:.4f}, p={fc.p_value:.4f}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

_DEFAULT_N_PERMUTATIONS = 1000
_MIN_GROUP_SIZE = 100


def _permutation_test_auc(
    preds_a: npt.NDArray[np.floating[Any]],
    targets_a: npt.NDArray[np.floating[Any]],
    preds_b: npt.NDArray[np.floating[Any]],
    targets_b: npt.NDArray[np.floating[Any]],
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    seed: int = 42,
) -> float:
    """Two-sample permutation test for AUC difference.

    Tests the null hypothesis that AUC(group_a) == AUC(group_b) by
    repeatedly shuffling group assignments and computing the AUC
    difference under the null distribution.

    Args:
        preds_a: Predictions for group A, shape [n_a].
        targets_a: Labels for group A, shape [n_a].
        preds_b: Predictions for group B, shape [n_b].
        targets_b: Labels for group B, shape [n_b].
        n_permutations: Number of permutation iterations.
        seed: Random seed for reproducibility.

    Returns:
        P-value (two-sided): proportion of permuted differences as
        extreme as observed difference.
    """
    rng = np.random.RandomState(seed)

    auc_a = _compute_auc(preds_a, targets_a)
    auc_b = _compute_auc(preds_b, targets_b)
    observed_diff = abs(auc_a - auc_b)

    # Pool all data
    all_preds = np.concatenate([preds_a, preds_b])
    all_targets = np.concatenate([targets_a, targets_b])
    n_a = len(preds_a)
    n_total = len(all_preds)

    count_extreme = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_preds_a = all_preds[perm[:n_a]]
        perm_targets_a = all_targets[perm[:n_a]]
        perm_preds_b = all_preds[perm[n_a:]]
        perm_targets_b = all_targets[perm[n_a:]]

        perm_auc_a = _compute_auc(perm_preds_a, perm_targets_a)
        perm_auc_b = _compute_auc(perm_preds_b, perm_targets_b)
        perm_diff = abs(perm_auc_a - perm_auc_b)

        if perm_diff >= observed_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# Subgroup extraction helpers
# ---------------------------------------------------------------------------


def _extract_sex_subgroups(
    report: BenchmarkReport,
) -> dict[str, SubgroupReport]:
    """Extract sex-based subgroup reports from a benchmark report."""
    sex_groups: dict[str, SubgroupReport] = {}
    for sg in report.subgroups:
        if sg.subgroup_name.startswith("sex_"):
            sex_groups[sg.subgroup_name] = sg
    return sex_groups


def _extract_age_subgroups(
    report: BenchmarkReport,
) -> dict[str, SubgroupReport]:
    """Extract age-based subgroup reports from a benchmark report."""
    age_groups: dict[str, SubgroupReport] = {}
    for sg in report.subgroups:
        if sg.subgroup_name.startswith("age_"):
            age_groups[sg.subgroup_name] = sg
    return age_groups


def _get_per_class_auc(
    subgroup: SubgroupReport,
    task: str,
    class_index: int,
) -> Optional[float]:
    """Get AUC for a specific class in a subgroup's task report."""
    task_report = subgroup.task_reports.get(task)
    if task_report is None:
        return None
    if class_index >= len(task_report.per_class):
        return None
    return task_report.per_class[class_index].auc


# ---------------------------------------------------------------------------
# Main equity gate function
# ---------------------------------------------------------------------------


def equity_gate(
    benchmark_report: BenchmarkReport,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    min_group_size: int = _MIN_GROUP_SIZE,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    seed: int = 42,
    predictions: Optional[dict[str, npt.NDArray[np.floating[Any]]]] = None,
    targets: Optional[dict[str, npt.NDArray[np.floating[Any]]]] = None,
    metadata: Optional[list[Optional[dict[str, Any]]]] = None,
) -> EquityGateResult:
    """Run the equity gating check on a benchmark report.

    Compares per-task AUC across sex groups (male/female) and age deciles
    (30–80) using permutation tests.  Fails if any comparison shows
    p < alpha after Bonferroni correction for classes with N > min_group_size
    test examples.

    There are two modes of operation:

    1. **With raw data** (``predictions``, ``targets``, ``metadata``
       provided): runs full permutation tests on the raw prediction arrays.

    2. **Report-only** (only ``benchmark_report`` provided): compares
       pre-computed per-class AUCs across subgroups.  Since raw data is
       not available, p-values are approximated using a bootstrap-style
       heuristic based on AUC differences and group sizes.

    Args:
        benchmark_report: A :class:`BenchmarkReport` with subgroup data.
        alpha: Significance level (before correction).
        correction: Multiple comparison correction ('bonferroni' or 'none').
        min_group_size: Minimum samples in *both* groups for a comparison
            to be included.  Default 100.
        n_permutations: Number of permutation iterations (raw data mode).
        seed: Random seed for reproducibility.
        predictions: Optional dict mapping task name to prediction arrays
            [N, C] for permutation testing.
        targets: Optional dict mapping task name to target arrays [N, C].
        metadata: Optional list of dicts with 'age' and/or 'sex' keys
            (one per sample).

    Returns:
        :class:`EquityGateResult` with pass/fail and comparison details.
    """
    has_raw_data = (
        predictions is not None
        and targets is not None
        and metadata is not None
    )

    if has_raw_data:
        return _equity_gate_raw(
            benchmark_report=benchmark_report,
            predictions=predictions,  # type: ignore[arg-type]
            targets=targets,  # type: ignore[arg-type]
            metadata=metadata,  # type: ignore[arg-type]
            alpha=alpha,
            correction=correction,
            min_group_size=min_group_size,
            n_permutations=n_permutations,
            seed=seed,
        )
    else:
        return _equity_gate_report_only(
            benchmark_report=benchmark_report,
            alpha=alpha,
            correction=correction,
            min_group_size=min_group_size,
        )


# ---------------------------------------------------------------------------
# Raw-data mode: full permutation tests
# ---------------------------------------------------------------------------


def _build_group_masks(
    metadata: list[Optional[dict[str, Any]]],
    n_samples: int,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Build boolean masks for demographic subgroups."""
    groups: dict[str, list[int]] = {}

    for i, meta in enumerate(metadata):
        if meta is None:
            continue

        # Sex
        sex = meta.get("sex")
        if sex is not None:
            key = f"sex_{sex}"
            groups.setdefault(key, []).append(i)

        # Age (binned into decades, only 30-80 as per PRD)
        age = meta.get("age")
        if age is not None:
            try:
                age_val = int(age)
                decade = (age_val // 10) * 10
                if 30 <= decade <= 80:
                    if decade >= 80:
                        key = "age_80-89"
                    else:
                        key = f"age_{decade}-{decade + 9}"
                    groups.setdefault(key, []).append(i)
            except (ValueError, TypeError):
                pass

    masks: dict[str, npt.NDArray[np.bool_]] = {}
    for name, indices in groups.items():
        mask = np.zeros(n_samples, dtype=bool)
        mask[indices] = True
        masks[name] = mask

    return masks


def _equity_gate_raw(
    benchmark_report: BenchmarkReport,
    predictions: dict[str, npt.NDArray[np.floating[Any]]],
    targets: dict[str, npt.NDArray[np.floating[Any]]],
    metadata: list[Optional[dict[str, Any]]],
    alpha: float = 0.05,
    correction: str = "bonferroni",
    min_group_size: int = _MIN_GROUP_SIZE,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    seed: int = 42,
) -> EquityGateResult:
    """Equity gate with full permutation tests on raw data."""
    n_samples = len(metadata)
    masks = _build_group_masks(metadata, n_samples)

    # Determine tasks to check (classification only — AUC comparison)
    tasks_to_check = [
        t for t in benchmark_report.tasks_evaluated
        if t in CLASSIFICATION_TASKS
    ]

    # Collect all valid comparisons first to count for Bonferroni
    planned_comparisons: list[
        tuple[str, str, str, int, str, npt.NDArray[Any], npt.NDArray[Any],
              npt.NDArray[Any], npt.NDArray[Any]]
    ] = []

    # Sex comparisons (pairwise)
    sex_groups = {k: v for k, v in masks.items() if k.startswith("sex_")}
    sex_keys = sorted(sex_groups.keys())

    for task in tasks_to_check:
        if task not in predictions or task not in targets:
            continue
        task_preds = predictions[task]
        task_tgts = targets[task]
        num_classes = task_preds.shape[1]

        # Get class names from benchmark report
        task_report = benchmark_report.overall.get(task)
        class_names = (
            [cm.name for cm in task_report.per_class]
            if task_report
            else [str(i) for i in range(num_classes)]
        )

        for ci in range(num_classes):
            for i in range(len(sex_keys)):
                for j in range(i + 1, len(sex_keys)):
                    ga, gb = sex_keys[i], sex_keys[j]
                    mask_a, mask_b = sex_groups[ga], sex_groups[gb]
                    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
                    if n_a < min_group_size or n_b < min_group_size:
                        continue
                    # Check that both groups have both positive and negative
                    tgts_a = task_tgts[mask_a, ci]
                    tgts_b = task_tgts[mask_b, ci]
                    if len(np.unique(tgts_a)) < 2 or len(np.unique(tgts_b)) < 2:
                        continue
                    cname = class_names[ci] if ci < len(class_names) else str(ci)
                    planned_comparisons.append((
                        ga, gb, task, ci, cname,
                        task_preds[mask_a, ci], tgts_a,
                        task_preds[mask_b, ci], tgts_b,
                    ))

    # Age comparisons (pairwise across all age bins)
    age_groups = {k: v for k, v in masks.items() if k.startswith("age_")}
    age_keys = sorted(age_groups.keys())

    for task in tasks_to_check:
        if task not in predictions or task not in targets:
            continue
        task_preds = predictions[task]
        task_tgts = targets[task]
        num_classes = task_preds.shape[1]

        task_report = benchmark_report.overall.get(task)
        class_names = (
            [cm.name for cm in task_report.per_class]
            if task_report
            else [str(i) for i in range(num_classes)]
        )

        for ci in range(num_classes):
            for i in range(len(age_keys)):
                for j in range(i + 1, len(age_keys)):
                    ga, gb = age_keys[i], age_keys[j]
                    mask_a, mask_b = age_groups[ga], age_groups[gb]
                    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
                    if n_a < min_group_size or n_b < min_group_size:
                        continue
                    tgts_a = task_tgts[mask_a, ci]
                    tgts_b = task_tgts[mask_b, ci]
                    if len(np.unique(tgts_a)) < 2 or len(np.unique(tgts_b)) < 2:
                        continue
                    cname = class_names[ci] if ci < len(class_names) else str(ci)
                    planned_comparisons.append((
                        ga, gb, task, ci, cname,
                        task_preds[mask_a, ci], tgts_a,
                        task_preds[mask_b, ci], tgts_b,
                    ))

    num_comparisons = len(planned_comparisons)

    # Compute corrected alpha
    if correction == "bonferroni" and num_comparisons > 0:
        corrected_alpha = alpha / num_comparisons
    else:
        corrected_alpha = alpha

    # Run permutation tests
    comparisons: list[ComparisonResult] = []
    failing: list[ComparisonResult] = []
    per_group_metrics: list[GroupMetrics] = []

    # Collect per-group summaries
    _seen_groups: set[tuple[str, str]] = set()

    for (ga, gb, task, ci, cname, preds_a, tgts_a, preds_b, tgts_b) in planned_comparisons:
        p_value = _permutation_test_auc(
            preds_a, tgts_a, preds_b, tgts_b,
            n_permutations=n_permutations,
            seed=seed,
        )

        auc_a = _compute_auc(preds_a, tgts_a)
        auc_b = _compute_auc(preds_b, tgts_b)
        auc_diff = abs(auc_a - auc_b)
        significant = p_value < corrected_alpha
        n_a = len(preds_a)
        n_b = len(preds_b)

        comp = ComparisonResult(
            group_a=ga,
            group_b=gb,
            task=task,
            class_index=ci,
            class_name=cname,
            auc_a=auc_a,
            auc_b=auc_b,
            auc_diff=auc_diff,
            p_value=p_value,
            significant=significant,
            n_a=n_a,
            n_b=n_b,
        )
        comparisons.append(comp)
        if significant:
            failing.append(comp)

        # Track group metrics
        for gname, auc_val, n_val in [(ga, auc_a, n_a), (gb, auc_b, n_b)]:
            key = (gname, task)
            if key not in _seen_groups:
                _seen_groups.add(key)
                per_group_metrics.append(
                    GroupMetrics(
                        group_name=gname, task=task,
                        auc=auc_val, n_samples=n_val,
                    )
                )

    return EquityGateResult(
        passed=len(failing) == 0,
        alpha=alpha,
        correction=correction,
        corrected_alpha=corrected_alpha,
        num_comparisons=num_comparisons,
        comparisons=comparisons,
        failing_comparisons=failing,
        per_group_metrics=per_group_metrics,
    )


# ---------------------------------------------------------------------------
# Report-only mode: heuristic from pre-computed subgroup AUCs
# ---------------------------------------------------------------------------


def _auc_diff_p_heuristic(
    auc_a: float,
    auc_b: float,
    n_a: int,
    n_b: int,
) -> float:
    """Approximate p-value from AUC difference and group sizes.

    Uses a normal approximation: under the null, the standard error of
    an AUC difference is approximately::

        SE ≈ sqrt(1/(12*n_a) + 1/(12*n_b))

    This is a conservative (Hanley–McNeil-inspired) estimate.
    """
    se = np.sqrt(1.0 / (12.0 * max(n_a, 1)) + 1.0 / (12.0 * max(n_b, 1)))
    if se == 0:
        return 1.0
    z = abs(auc_a - auc_b) / se
    # Two-sided p-value from standard normal
    p = 2.0 * (1.0 - _standard_normal_cdf(z))
    return float(p)


def _standard_normal_cdf(x: float) -> float:
    """Approximate CDF of the standard normal distribution."""
    # Using the error function approximation
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: float) -> float:
    """Approximation of the error function (Abramowitz & Stegun)."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x * x))
    return sign * float(y)


def _equity_gate_report_only(
    benchmark_report: BenchmarkReport,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    min_group_size: int = _MIN_GROUP_SIZE,
) -> EquityGateResult:
    """Equity gate using only pre-computed subgroup metrics."""
    sex_groups = _extract_sex_subgroups(benchmark_report)
    age_groups = _extract_age_subgroups(benchmark_report)

    tasks_to_check = [
        t for t in benchmark_report.tasks_evaluated
        if t in CLASSIFICATION_TASKS
    ]

    # Plan comparisons
    planned: list[tuple[str, str, str, int, str, float, float, int, int]] = []

    # Sex comparisons
    sex_keys = sorted(sex_groups.keys())
    for task in tasks_to_check:
        task_report = benchmark_report.overall.get(task)
        if task_report is None:
            continue
        num_classes = len(task_report.per_class)
        for ci in range(num_classes):
            for i in range(len(sex_keys)):
                for j in range(i + 1, len(sex_keys)):
                    ga_key, gb_key = sex_keys[i], sex_keys[j]
                    sg_a, sg_b = sex_groups[ga_key], sex_groups[gb_key]
                    if sg_a.n_samples < min_group_size or sg_b.n_samples < min_group_size:
                        continue
                    auc_a = _get_per_class_auc(sg_a, task, ci)
                    auc_b = _get_per_class_auc(sg_b, task, ci)
                    if auc_a is None or auc_b is None:
                        continue
                    cname = task_report.per_class[ci].name
                    planned.append((
                        ga_key, gb_key, task, ci, cname,
                        auc_a, auc_b, sg_a.n_samples, sg_b.n_samples,
                    ))

    # Age comparisons
    age_keys = sorted(age_groups.keys())
    for task in tasks_to_check:
        task_report = benchmark_report.overall.get(task)
        if task_report is None:
            continue
        num_classes = len(task_report.per_class)
        for ci in range(num_classes):
            for i in range(len(age_keys)):
                for j in range(i + 1, len(age_keys)):
                    ga_key, gb_key = age_keys[i], age_keys[j]
                    sg_a, sg_b = age_groups[ga_key], age_groups[gb_key]
                    if sg_a.n_samples < min_group_size or sg_b.n_samples < min_group_size:
                        continue
                    auc_a = _get_per_class_auc(sg_a, task, ci)
                    auc_b = _get_per_class_auc(sg_b, task, ci)
                    if auc_a is None or auc_b is None:
                        continue
                    cname = task_report.per_class[ci].name
                    planned.append((
                        ga_key, gb_key, task, ci, cname,
                        auc_a, auc_b, sg_a.n_samples, sg_b.n_samples,
                    ))

    num_comparisons = len(planned)
    if correction == "bonferroni" and num_comparisons > 0:
        corrected_alpha = alpha / num_comparisons
    else:
        corrected_alpha = alpha

    comparisons: list[ComparisonResult] = []
    failing: list[ComparisonResult] = []
    per_group_metrics: list[GroupMetrics] = []
    _seen: set[tuple[str, str]] = set()

    for (ga, gb, task, ci, cname, auc_a, auc_b, n_a, n_b) in planned:
        auc_diff = abs(auc_a - auc_b)
        p_value = _auc_diff_p_heuristic(auc_a, auc_b, n_a, n_b)
        significant = p_value < corrected_alpha

        comp = ComparisonResult(
            group_a=ga,
            group_b=gb,
            task=task,
            class_index=ci,
            class_name=cname,
            auc_a=auc_a,
            auc_b=auc_b,
            auc_diff=auc_diff,
            p_value=p_value,
            significant=significant,
            n_a=n_a,
            n_b=n_b,
        )
        comparisons.append(comp)
        if significant:
            failing.append(comp)

        for gname, auc_val, n_val in [(ga, auc_a, n_a), (gb, auc_b, n_b)]:
            key = (gname, task)
            if key not in _seen:
                _seen.add(key)
                per_group_metrics.append(
                    GroupMetrics(
                        group_name=gname, task=task,
                        auc=auc_val, n_samples=n_val,
                    )
                )

    return EquityGateResult(
        passed=len(failing) == 0,
        alpha=alpha,
        correction=correction,
        corrected_alpha=corrected_alpha,
        num_comparisons=num_comparisons,
        comparisons=comparisons,
        failing_comparisons=failing,
        per_group_metrics=per_group_metrics,
    )
