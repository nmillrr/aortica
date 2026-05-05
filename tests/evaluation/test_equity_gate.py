"""Tests for equity_gate module (US-069)."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.evaluation.equity_gate import (
    ComparisonResult,
    EquityGateResult,
    GroupMetrics,
    _auc_diff_p_heuristic,
    _build_group_masks,
    _erf,
    _permutation_test_auc,
    _standard_normal_cdf,
    equity_gate,
)
from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    SubgroupReport,
    TaskReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subgroup(name: str, n: int, task: str, auc_values: list[float]) -> SubgroupReport:
    """Build a SubgroupReport with per-class AUCs for one task."""
    per_class = [ClassMetrics(name=f"cls_{i}", auc=a) for i, a in enumerate(auc_values)]
    return SubgroupReport(
        subgroup_name=name,
        n_samples=n,
        task_reports={task: TaskReport(task_name=task, per_class=per_class, macro_f1=0.8)},
    )


def _make_report_with_subgroups(subgroups: list[SubgroupReport], task: str = "rhythm",
                                 num_classes: int = 2) -> BenchmarkReport:
    per_class = [ClassMetrics(name=f"cls_{i}", auc=0.9) for i in range(num_classes)]
    return BenchmarkReport(
        overall={task: TaskReport(task_name=task, per_class=per_class, macro_f1=0.9)},
        subgroups=subgroups,
        n_samples=1000,
        tasks_evaluated=[task],
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestGroupMetrics:
    def test_defaults(self):
        gm = GroupMetrics()
        assert gm.group_name == ""
        assert gm.auc == 0.0
        assert gm.n_samples == 0

    def test_custom(self):
        gm = GroupMetrics(group_name="sex_M", task="rhythm", auc=0.92, n_samples=500)
        assert gm.group_name == "sex_M"
        assert gm.auc == 0.92


class TestComparisonResult:
    def test_defaults(self):
        cr = ComparisonResult()
        assert cr.p_value == 1.0
        assert cr.significant is False

    def test_custom(self):
        cr = ComparisonResult(group_a="sex_M", group_b="sex_F", task="rhythm",
                              class_index=0, class_name="AF", auc_a=0.95, auc_b=0.85,
                              auc_diff=0.10, p_value=0.001, significant=True, n_a=200, n_b=200)
        assert cr.significant is True
        assert cr.auc_diff == 0.10


class TestEquityGateResult:
    def test_defaults(self):
        r = EquityGateResult()
        assert r.passed is True
        assert r.comparisons == []

    def test_summary_passed(self):
        r = EquityGateResult(passed=True, num_comparisons=5)
        s = r.summary()
        assert "PASSED" in s

    def test_summary_failed(self):
        fc = ComparisonResult(group_a="sex_M", group_b="sex_F", task="rhythm",
                              class_name="AF", auc_a=0.95, auc_b=0.70,
                              auc_diff=0.25, p_value=0.001, significant=True,
                              n_a=200, n_b=200)
        r = EquityGateResult(passed=False, failing_comparisons=[fc], num_comparisons=1)
        s = r.summary()
        assert "FAILED" in s
        assert "AF" in s


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestMathHelpers:
    def test_erf_zero(self):
        assert abs(_erf(0.0)) < 1e-6

    def test_erf_large(self):
        assert _erf(5.0) > 0.99

    def test_erf_negative(self):
        assert _erf(-1.0) < 0

    def test_standard_normal_cdf_center(self):
        assert abs(_standard_normal_cdf(0.0) - 0.5) < 1e-6

    def test_standard_normal_cdf_tail(self):
        assert _standard_normal_cdf(3.0) > 0.99

    def test_auc_diff_p_heuristic_no_diff(self):
        p = _auc_diff_p_heuristic(0.9, 0.9, 200, 200)
        assert p > 0.99  # identical AUCs → high p

    def test_auc_diff_p_heuristic_large_diff(self):
        p = _auc_diff_p_heuristic(0.95, 0.55, 500, 500)
        assert p < 0.01  # huge diff → low p


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_identical_groups(self):
        rng = np.random.RandomState(42)
        n = 200
        preds = rng.rand(n)
        targets = (preds > 0.5).astype(float)
        p = _permutation_test_auc(preds[:100], targets[:100], preds[100:], targets[100:],
                                   n_permutations=100, seed=42)
        assert p > 0.05  # same distribution → not significant

    def test_very_different_groups(self):
        rng = np.random.RandomState(42)
        # Group A: perfect predictions; Group B: random
        n = 150
        targets_a = np.array([1.0]*75 + [0.0]*75)
        preds_a = targets_a.copy()  # perfect
        targets_b = np.array([1.0]*75 + [0.0]*75)
        preds_b = rng.rand(n)  # random
        p = _permutation_test_auc(preds_a, targets_a, preds_b, targets_b,
                                   n_permutations=200, seed=42)
        assert p < 0.1  # should detect a difference

    def test_reproducible(self):
        rng = np.random.RandomState(0)
        n = 100
        preds = rng.rand(n)
        targets = (preds > 0.5).astype(float)
        p1 = _permutation_test_auc(preds[:50], targets[:50], preds[50:], targets[50:],
                                    n_permutations=50, seed=99)
        p2 = _permutation_test_auc(preds[:50], targets[:50], preds[50:], targets[50:],
                                    n_permutations=50, seed=99)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Group mask building
# ---------------------------------------------------------------------------

class TestBuildGroupMasks:
    def test_sex_groups(self):
        meta = [{"sex": "M"}, {"sex": "F"}, {"sex": "M"}, None]
        masks = _build_group_masks(meta, 4)
        assert "sex_M" in masks
        assert "sex_F" in masks
        assert masks["sex_M"].sum() == 2
        assert masks["sex_F"].sum() == 1

    def test_age_groups_in_range(self):
        meta = [{"age": 35}, {"age": 45}, {"age": 25}, {"age": 85}]
        masks = _build_group_masks(meta, 4)
        assert "age_30-39" in masks
        assert "age_40-49" in masks
        assert "age_80-89" in masks
        # age 25 → decade 20 → out of 30-80 range
        assert "age_20-29" not in masks

    def test_empty_metadata(self):
        masks = _build_group_masks([None, None], 2)
        assert masks == {}


# ---------------------------------------------------------------------------
# Report-only equity gate (passing scenario)
# ---------------------------------------------------------------------------

class TestEquityGateReportOnlyPassing:
    def test_similar_aucs_pass(self):
        sg_m = _make_subgroup("sex_M", 200, "rhythm", [0.90, 0.91])
        sg_f = _make_subgroup("sex_F", 200, "rhythm", [0.89, 0.90])
        report = _make_report_with_subgroups([sg_m, sg_f])
        result = equity_gate(report, alpha=0.05)
        assert result.passed is True
        assert len(result.failing_comparisons) == 0

    def test_small_groups_skipped(self):
        sg_m = _make_subgroup("sex_M", 50, "rhythm", [0.95, 0.60])
        sg_f = _make_subgroup("sex_F", 50, "rhythm", [0.60, 0.95])
        report = _make_report_with_subgroups([sg_m, sg_f])
        result = equity_gate(report, alpha=0.05, min_group_size=100)
        assert result.passed is True
        assert result.num_comparisons == 0

    def test_no_subgroups_pass(self):
        report = _make_report_with_subgroups([])
        result = equity_gate(report)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Report-only equity gate (failing scenario)
# ---------------------------------------------------------------------------

class TestEquityGateReportOnlyFailing:
    def test_large_auc_gap_fails(self):
        sg_m = _make_subgroup("sex_M", 500, "rhythm", [0.95, 0.95])
        sg_f = _make_subgroup("sex_F", 500, "rhythm", [0.55, 0.55])
        report = _make_report_with_subgroups([sg_m, sg_f])
        result = equity_gate(report, alpha=0.05)
        assert result.passed is False
        assert len(result.failing_comparisons) > 0

    def test_bonferroni_correction_applied(self):
        sg_m = _make_subgroup("sex_M", 500, "rhythm", [0.90])
        sg_f = _make_subgroup("sex_F", 500, "rhythm", [0.85])
        report = _make_report_with_subgroups([sg_m, sg_f], num_classes=1)
        result = equity_gate(report, alpha=0.05, correction="bonferroni")
        assert result.corrected_alpha == 0.05 / result.num_comparisons

    def test_no_correction(self):
        sg_m = _make_subgroup("sex_M", 500, "rhythm", [0.90])
        sg_f = _make_subgroup("sex_F", 500, "rhythm", [0.85])
        report = _make_report_with_subgroups([sg_m, sg_f], num_classes=1)
        result = equity_gate(report, alpha=0.05, correction="none")
        assert result.corrected_alpha == 0.05


# ---------------------------------------------------------------------------
# Raw-data equity gate
# ---------------------------------------------------------------------------

class TestEquityGateRaw:
    def _make_raw_data(self, n_per_group=150, auc_gap=0.0, seed=42):
        """Generate synthetic raw data for two sex groups."""
        rng = np.random.RandomState(seed)
        n = n_per_group * 2
        # 2-class classification task
        targets = np.zeros((n, 2))
        preds = np.zeros((n, 2))
        for c in range(2):
            targets[:, c] = (rng.rand(n) > 0.5).astype(float)
            preds[:, c] = targets[:, c] * (0.8 - auc_gap * (np.arange(n) >= n_per_group).astype(float)) + rng.rand(n) * 0.3

        meta: list[dict[str, str] | None] = []
        for i in range(n):
            meta.append({"sex": "M" if i < n_per_group else "F"})

        per_class = [ClassMetrics(name=f"cls_{i}", auc=0.9) for i in range(2)]
        report = BenchmarkReport(
            overall={"rhythm": TaskReport(task_name="rhythm", per_class=per_class)},
            subgroups=[],
            n_samples=n,
            tasks_evaluated=["rhythm"],
        )
        return report, {"rhythm": preds}, {"rhythm": targets}, meta

    def test_raw_passing(self):
        report, preds, tgts, meta = self._make_raw_data(auc_gap=0.0)
        result = equity_gate(report, predictions=preds, targets=tgts, metadata=meta,
                             min_group_size=100, n_permutations=50)
        assert result.passed is True

    def test_raw_has_comparisons(self):
        report, preds, tgts, meta = self._make_raw_data(auc_gap=0.0)
        result = equity_gate(report, predictions=preds, targets=tgts, metadata=meta,
                             min_group_size=100, n_permutations=50)
        assert result.num_comparisons > 0

    def test_raw_per_group_metrics(self):
        report, preds, tgts, meta = self._make_raw_data()
        result = equity_gate(report, predictions=preds, targets=tgts, metadata=meta,
                             min_group_size=100, n_permutations=50)
        assert len(result.per_group_metrics) > 0


# ---------------------------------------------------------------------------
# Age subgroup comparisons
# ---------------------------------------------------------------------------

class TestAgeComparisons:
    def test_age_subgroups_compared(self):
        sg30 = _make_subgroup("age_30-39", 200, "rhythm", [0.90])
        sg40 = _make_subgroup("age_40-49", 200, "rhythm", [0.89])
        report = _make_report_with_subgroups([sg30, sg40], num_classes=1)
        result = equity_gate(report, alpha=0.05)
        assert result.num_comparisons >= 1

    def test_many_age_bins_bonferroni(self):
        sgs = [_make_subgroup(f"age_{d}-{d+9}", 200, "rhythm", [0.90])
               for d in range(30, 80, 10)]
        report = _make_report_with_subgroups(sgs, num_classes=1)
        result = equity_gate(report, alpha=0.05)
        # C(5,2) = 10 comparisons × 1 class = 10
        assert result.num_comparisons == 10
        assert result.corrected_alpha == 0.05 / 10


# ---------------------------------------------------------------------------
# Import/export tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_top_level_import(self):
        from aortica.evaluation import equity_gate as eg  # noqa: F811
        assert callable(eg)

    def test_result_types_importable(self):
        from aortica.evaluation import EquityGateResult, ComparisonResult, GroupMetrics  # noqa: F811
        assert EquityGateResult is not None
        assert ComparisonResult is not None
        assert GroupMetrics is not None


# ---------------------------------------------------------------------------
# Typecheck placeholder
# ---------------------------------------------------------------------------

class TestTypecheck:
    def test_module_has_equity_gate(self):
        import importlib
        mod = importlib.import_module("aortica.evaluation.equity_gate")
        assert hasattr(mod, "equity_gate")
        assert hasattr(mod, "EquityGateResult")
        assert hasattr(mod, "ComparisonResult")
        assert hasattr(mod, "GroupMetrics")
