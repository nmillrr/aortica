"""Tests for aortica.evaluation.model_comparison (US-116)."""

from __future__ import annotations

import json

import numpy as np

from aortica.evaluation.model_comparison import (
    ClassDelta,
    ModelComparisonReport,
    TaskDelta,
    compare_predictions,
)

_N = 300
_RHYTHM = 28
_RISK = 6


def _binary_targets(n: int, k: int, prevalence: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, k)) < prevalence).astype(np.float64)


def _preds_at_accuracy(
    targets: np.ndarray, correct_prob: float, seed: int
) -> np.ndarray:
    """Build probability predictions with a controllable accuracy.

    With probability ``correct_prob`` a sample gets a confident-correct
    score (0.9 for positives / 0.1 for negatives); otherwise a
    confident-wrong score.  Higher ``correct_prob`` → better F1/AUC.
    """
    rng = np.random.default_rng(seed)
    correct = rng.random(targets.shape) < correct_prob
    preds = np.where(targets == 1, 0.9, 0.1)
    preds = np.where(correct, preds, 1.0 - preds)
    # A little jitter to avoid exact ties.
    preds = np.clip(preds + rng.normal(0, 0.02, targets.shape), 0.0, 1.0)
    return preds


# ─────────────────────────────────────────────────────────────────
# API / structure
# ─────────────────────────────────────────────────────────────────


class TestApi:
    def test_returns_report(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.7, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=100
        )
        assert isinstance(report, ModelComparisonReport)
        assert "rhythm" in report.task_deltas
        td = report.task_deltas["rhythm"]
        assert isinstance(td, TaskDelta)
        assert len(td.per_class) == _RHYTHM
        assert all(isinstance(c, ClassDelta) for c in td.per_class)

    def test_to_dict_json_serialisable(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.8, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=100
        )
        json.dumps(report.to_dict())

    def test_class_delta_has_all_metrics(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.8, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=100
        )
        cd = report.task_deltas["rhythm"].per_class[0]
        assert set(cd.deltas) == {"f1", "auc", "sensitivity", "specificity"}
        assert cd.ci_low <= cd.ci_high

    def test_tasks_filter(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.7, 2)
        struct_tg = _binary_targets(_N, 19, 0.3, 5)
        report = compare_predictions(
            {"rhythm": pa, "structural": _preds_at_accuracy(struct_tg, 0.7, 6)},
            {"rhythm": pb, "structural": _preds_at_accuracy(struct_tg, 0.7, 7)},
            {"rhythm": tg, "structural": struct_tg},
            tasks=["rhythm"],
            n_bootstrap=50,
        )
        assert report.tasks == ["rhythm"]
        assert "structural" not in report.task_deltas


# ─────────────────────────────────────────────────────────────────
# Improvement scenario
# ─────────────────────────────────────────────────────────────────


class TestImprovement:
    def test_b_better_positive_delta(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.6, 1)  # weaker
        pb = _preds_at_accuracy(tg, 0.95, 2)  # stronger
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=200
        )
        assert report.task_deltas["rhythm"].delta_macro_f1 > 0
        assert report.regressions == []
        assert report.recommendation == "upgrade"

    def test_no_false_regression_on_improvement(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.6, 1)
        pb = _preds_at_accuracy(tg, 0.95, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=200
        )
        assert not any(
            c.is_regression for c in report.task_deltas["rhythm"].per_class
        )


# ─────────────────────────────────────────────────────────────────
# Regression scenario
# ─────────────────────────────────────────────────────────────────


class TestRegression:
    def test_b_worse_flags_regression(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.95, 1)  # strong
        pb = _preds_at_accuracy(tg, 0.55, 2)  # much weaker
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=200
        )
        assert report.task_deltas["rhythm"].delta_macro_f1 < 0
        assert len(report.regressions) > 0
        assert report.recommendation == "investigate"

    def test_regression_identifiers_format(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.95, 1)
        pb = _preds_at_accuracy(tg, 0.55, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=200
        )
        for reg in report.regressions:
            assert reg.startswith("rhythm/")


# ─────────────────────────────────────────────────────────────────
# Mixed scenario
# ─────────────────────────────────────────────────────────────────


class TestMixed:
    def test_mixed_improve_and_regress(self) -> None:
        # rhythm improves, structural regresses.
        r_tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        r_pa = _preds_at_accuracy(r_tg, 0.6, 1)
        r_pb = _preds_at_accuracy(r_tg, 0.95, 2)
        s_tg = _binary_targets(_N, 19, 0.3, 3)
        s_pa = _preds_at_accuracy(s_tg, 0.95, 4)
        s_pb = _preds_at_accuracy(s_tg, 0.55, 5)
        report = compare_predictions(
            {"rhythm": r_pa, "structural": s_pa},
            {"rhythm": r_pb, "structural": s_pb},
            {"rhythm": r_tg, "structural": s_tg},
            n_bootstrap=200,
        )
        assert report.task_deltas["rhythm"].delta_macro_f1 > 0
        assert report.task_deltas["structural"].delta_macro_f1 < 0
        assert any(r.startswith("structural/") for r in report.regressions)
        assert report.recommendation == "investigate"


# ─────────────────────────────────────────────────────────────────
# Risk task (C-index)
# ─────────────────────────────────────────────────────────────────


class TestRiskTask:
    def test_risk_c_index_delta(self) -> None:
        rng = np.random.default_rng(0)
        tg = rng.random((_N, _RISK))
        # A: near-random; B: strongly correlated with target → higher C-index.
        pa = rng.random((_N, _RISK))
        pb = tg + rng.normal(0, 0.05, (_N, _RISK))
        report = compare_predictions(
            {"risk": pa}, {"risk": pb}, {"risk": tg}, n_bootstrap=100
        )
        td = report.task_deltas["risk"]
        assert td.delta_c_index > 0
        assert len(td.per_class) == _RISK
        assert all(c.primary_metric == "c_index" for c in td.per_class)


# ─────────────────────────────────────────────────────────────────
# Demographic subgroups
# ─────────────────────────────────────────────────────────────────


class TestSubgroups:
    def test_subgroup_deltas_present(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.8, 2)
        rng = np.random.default_rng(9)
        metadata = [
            {"age": int(rng.integers(20, 80)), "sex": "M" if i % 2 else "F"}
            for i in range(_N)
        ]
        report = compare_predictions(
            {"rhythm": pa},
            {"rhythm": pb},
            {"rhythm": tg},
            metadata=metadata,
            n_bootstrap=50,
        )
        assert report.subgroup_deltas
        sex_groups = [
            s for s in report.subgroup_deltas if s.subgroup_name.startswith("sex_")
        ]
        assert len(sex_groups) == 2
        for sg in sex_groups:
            assert "rhythm" in sg.delta_macro_f1

    def test_no_metadata_no_subgroups(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.8, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=50
        )
        assert report.subgroup_deltas == []


# ─────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────


class TestMarkdown:
    def test_markdown_contains_sections(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.95, 1)
        pb = _preds_at_accuracy(tg, 0.55, 2)
        report = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=200
        )
        md = report.to_markdown()
        assert "# Model Comparison Report" in md
        assert "## Summary" in md
        assert "Recommendation" in md
        assert "Regression Warnings" in md  # this scenario regresses

    def test_markdown_reproducible(self) -> None:
        tg = _binary_targets(_N, _RHYTHM, 0.3, 0)
        pa = _preds_at_accuracy(tg, 0.7, 1)
        pb = _preds_at_accuracy(tg, 0.8, 2)
        r1 = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=100, seed=7
        )
        r2 = compare_predictions(
            {"rhythm": pa}, {"rhythm": pb}, {"rhythm": tg}, n_bootstrap=100, seed=7
        )
        assert r1.to_markdown() == r2.to_markdown()
