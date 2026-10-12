"""Tests for aortica.evaluation.classification."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.evaluation.classification import (
    auroc,
    average_precision,
    bootstrap_macro_auroc_ci,
    confusion_at,
    evaluate_classes,
    macro_auroc,
    tune_threshold,
)


def test_auroc_perfect_and_inverted() -> None:
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0


def test_auroc_ties_count_half() -> None:
    assert auroc(np.full(4, 0.5), np.array([0, 1, 0, 1])) == 0.5


def test_auroc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    s, y = rng.random(200), rng.random(200) < 0.3
    pos, neg = s[y], s[~y]
    expected = ((pos[:, None] > neg[None, :]).sum()
                + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert auroc(s, y.astype(float)) == pytest.approx(expected)


def test_auroc_single_class_is_nan() -> None:
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_average_precision() -> None:
    # Ranked: P N P -> precision at hits 1/1 and 2/3 -> AP = (1 + 2/3) / 2
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]))
    assert ap == pytest.approx((1 + 2 / 3) / 2)


def test_confusion_at_threshold() -> None:
    c = confusion_at(np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 1, 0, 1]), 0.5)
    assert c["sensitivity"] == 0.5 and c["specificity"] == 0.5 and c["ppv"] == 0.5


def test_tune_threshold_finds_separating_point() -> None:
    s = np.array([0.02, 0.03, 0.12, 0.15])
    assert tune_threshold(s, np.array([0, 0, 1, 1])) == pytest.approx(0.05)


def test_evaluate_classes_respects_mask() -> None:
    probs = np.array([[0.9], [0.1], [0.8]])
    y = np.array([[1.0], [0.0], [0.0]])
    mask = np.array([[1.0], [1.0], [0.0]])  # third row unlabelled for this column
    res = evaluate_classes(probs, y, ["X"], mask=mask, n_boot=0)
    assert res["X"].n == 2 and res["X"].auroc == 1.0


def test_macro_auroc_min_pos() -> None:
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.3]])
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=float)
    res = evaluate_classes(probs, y, ["A", "B"], n_boot=0)
    assert macro_auroc(res, min_pos=2) == (1.0, 1)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random((500, 1)) < 0.3).astype(float)
    p = np.clip(y * 0.4 + rng.random((500, 1)) * 0.6, 0, 1)
    lo, hi = bootstrap_macro_auroc_ci(p, y, [0], n_boot=200)
    assert lo <= auroc(p[:, 0], y[:, 0]) <= hi
