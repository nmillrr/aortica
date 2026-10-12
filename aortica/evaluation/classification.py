"""Per-class multi-label classification metrics, with label masks.

One implementation shared by the reference training run, external
validation and baseline comparison, so a number in one table means the same
thing as the same-named number in another.  NumPy only — no scikit-learn
dependency — and every function accepts a validity mask, because a column a
corpus cannot label is *unknown*, not negative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

#: Threshold grid used for per-class operating-point tuning.  Matches the
#: 0.05-step operating points released with v0.3.0.
THRESHOLD_GRID: NDArray[np.float64] = np.round(np.arange(0.05, 0.951, 0.05), 2)


def auroc(scores: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged).

    Returns NaN when only one class is present.
    """
    t = np.asarray(targets) > 0.5
    n_pos = int(t.sum())
    n_neg = int(t.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))  # ties averaged
    rank_sum = ranks[t].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    t = np.asarray(targets) > 0.5
    n_pos = int(t.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    hits = t[order].astype(np.float64)
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def confusion_at(
    scores: NDArray[np.floating], targets: NDArray[np.floating], threshold: float
) -> Dict[str, float]:
    """Sensitivity, specificity, PPV and F1 at *threshold*."""
    t = np.asarray(targets) > 0.5
    p = np.asarray(scores) >= threshold
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    tn = float(np.sum(~p & ~t))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "f1": f1}


def tune_threshold(
    scores: NDArray[np.floating],
    targets: NDArray[np.floating],
    grid: Sequence[float] = tuple(THRESHOLD_GRID),
) -> float:
    """Return the grid threshold maximising F1 (lowest wins ties)."""
    best_t, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = confusion_at(scores, targets, float(thr))["f1"]
        if not np.isnan(f1) and f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(thr), f1
    return best_t


@dataclass
class ClassResult:
    """Metrics for one output on one evaluation set."""

    n: int
    n_pos: int
    auroc: float
    auprc: float
    threshold: Optional[float]
    sensitivity: float
    specificity: float
    ppv: float
    f1: float
    auroc_ci95: Optional[tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def bootstrap_auroc_ci(
    scores: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Optional[tuple[float, float]]:
    """Percentile 95% CI for AUROC by resampling records with replacement."""
    t = np.asarray(targets) > 0.5
    if t.sum() == 0 or t.sum() == t.size:
        return None
    rng = np.random.default_rng(seed)
    n = len(t)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auroc(np.asarray(scores)[idx], t[idx])
        if not np.isnan(a):
            stats.append(a)
    if not stats:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_classes(
    probs: NDArray[np.floating],
    targets: NDArray[np.floating],
    class_names: Sequence[str],
    *,
    mask: Optional[NDArray[np.floating]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    min_pos_for_ci: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, ClassResult]:
    """Score every class that has at least one labelled row.

    Args:
        probs: ``[N, C]`` predicted probabilities.
        targets: ``[N, C]`` binary labels.
        class_names: ``C`` names.
        mask: ``[N, C]`` validity mask; rows with 0 are excluded per column.
        thresholds: Operating point per class; 0.5 where absent.
        min_pos_for_ci: Only bootstrap a CI with at least this many positives.
    """
    out: Dict[str, ClassResult] = {}
    for c, name in enumerate(class_names):
        keep = np.ones(len(probs), dtype=bool) if mask is None else mask[:, c] > 0
        if not keep.any():
            continue
        s, t = probs[keep, c], targets[keep, c]
        n_pos = int((t > 0.5).sum())
        thr = (thresholds or {}).get(name, 0.5)
        conf = confusion_at(s, t, thr)
        ci = (
            bootstrap_auroc_ci(s, t, n_boot=n_boot, seed=seed + c)
            if n_pos >= min_pos_for_ci
            else None
        )
        out[name] = ClassResult(
            n=int(keep.sum()),
            n_pos=n_pos,
            auroc=auroc(s, t),
            auprc=average_precision(s, t),
            threshold=float(thr),
            auroc_ci95=ci,
            **conf,
        )
    return out


def macro_auroc(
    results: Mapping[str, ClassResult], *, min_pos: int = 1
) -> tuple[float, int]:
    """Mean AUROC over classes with at least *min_pos* positives (and a negative)."""
    vals = [r.auroc for r in results.values() if r.n_pos >= min_pos and not np.isnan(r.auroc)]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def bootstrap_macro_auroc_ci(
    probs: NDArray[np.floating],
    targets: NDArray[np.floating],
    columns: Sequence[int],
    *,
    mask: Optional[NDArray[np.floating]] = None,
    min_pos: int = 1,
    n_boot: int = 500,
    seed: int = 0,
) -> Optional[tuple[float, float]]:
    """Percentile 95% CI for macro-AUROC over *columns*, resampling records."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = []
        for c in columns:
            keep = np.ones(n, dtype=bool)[idx] if mask is None else mask[idx, c] > 0
            t = targets[idx, c][keep]
            if (t > 0.5).sum() < min_pos:
                continue
            a = auroc(probs[idx, c][keep], t)
            if not np.isnan(a):
                vals.append(a)
        if vals:
            stats.append(np.mean(vals))
    if not stats:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
