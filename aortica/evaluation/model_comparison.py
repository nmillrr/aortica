"""Model version comparison and A/B analysis (US-116).

Compares two model versions on the same dataset to quantify the impact of
an update (federated training, expanded task heads, or any change) before
release.  Produces per-task and per-class delta metrics with paired
bootstrap significance testing, demographic subgroup deltas, regression
detection, and a ``MODEL_COMPARISON.md`` report.

Two entry points:

* :func:`compare_predictions` — the torch-free statistical core.  Takes
  per-task prediction and target arrays for both models and computes the
  full :class:`ModelComparisonReport`.  This is what the unit tests
  exercise with synthetic predictions.
* :func:`compare_models` — loads two checkpoints, runs inference over a
  dataset, and delegates to :func:`compare_predictions`.  Requires torch.

Example::

    from aortica.evaluation import compare_models

    report = compare_models("v1.pt", "v2.pt", dataset)
    print(report.to_markdown())
    if report.recommendation == "upgrade":
        ...
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from aortica.evaluation.benchmark import (
    _CLASS_NAMES,
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    _build_subgroup_masks,
    _split_labels,
)

FloatArray = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassDelta:
    """Per-class metric delta (model B minus model A).

    Attributes:
        name: Class (or risk-output) name.
        deltas: Metric-name → delta value.  Classification classes carry
            ``f1``/``auc``/``sensitivity``/``specificity``; risk outputs
            carry ``c_index``.
        primary_metric: The metric significance-tested for this class.
        p_value: Two-sided paired-bootstrap p-value for the primary delta.
        ci_low: Lower bound of the 95% CI for the primary delta.
        ci_high: Upper bound of the 95% CI for the primary delta.
        is_regression: ``True`` if B is significantly worse (delta < 0 and
            p < the configured significance level).
    """

    name: str
    deltas: Dict[str, float]
    primary_metric: str
    p_value: float
    ci_low: float
    ci_high: float
    is_regression: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskDelta:
    """Per-task summary deltas (model B minus model A)."""

    task_name: str
    delta_macro_f1: float = 0.0
    delta_c_index: float = 0.0
    per_class: List[ClassDelta] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "delta_macro_f1": self.delta_macro_f1,
            "delta_c_index": self.delta_c_index,
            "per_class": [c.to_dict() for c in self.per_class],
        }


@dataclass
class SubgroupDelta:
    """Per-subgroup delta metrics for equity-regression detection."""

    subgroup_name: str
    n_samples: int
    delta_macro_f1: Dict[str, float] = field(default_factory=dict)
    delta_c_index: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelComparisonReport:
    """Full model-comparison report.

    Attributes:
        version_a: Identifier for model A.
        version_b: Identifier for model B.
        tasks: Tasks compared.
        task_deltas: Per-task delta metrics.
        subgroup_deltas: Per-demographic-subgroup deltas.
        regressions: ``"task/class"`` identifiers that regressed.
        recommendation: ``"upgrade"``, ``"hold"``, or ``"investigate"``.
        n_samples: Number of samples compared.
        n_bootstrap: Bootstrap resamples used for significance testing.
        alpha: Significance level used for regression detection.
    """

    version_a: str
    version_b: str
    tasks: List[str]
    task_deltas: Dict[str, TaskDelta] = field(default_factory=dict)
    subgroup_deltas: List[SubgroupDelta] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    recommendation: str = "investigate"
    n_samples: int = 0
    n_bootstrap: int = 0
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "tasks": self.tasks,
            "task_deltas": {k: v.to_dict() for k, v in self.task_deltas.items()},
            "subgroup_deltas": [s.to_dict() for s in self.subgroup_deltas],
            "regressions": self.regressions,
            "recommendation": self.recommendation,
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
        }

    def to_markdown(self) -> str:
        """Render the report as ``MODEL_COMPARISON.md`` text."""
        return _render_markdown(self)


# ---------------------------------------------------------------------------
# Metric helpers (self-contained, torch-free)
# ---------------------------------------------------------------------------


def _f1(pred_bin: FloatArray, target: FloatArray) -> float:
    tp = float(np.sum((pred_bin == 1) & (target == 1)))
    fp = float(np.sum((pred_bin == 1) & (target == 0)))
    fn = float(np.sum((pred_bin == 0) & (target == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _sensitivity(pred_bin: FloatArray, target: FloatArray) -> float:
    tp = float(np.sum((pred_bin == 1) & (target == 1)))
    fn = float(np.sum((pred_bin == 0) & (target == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _specificity(pred_bin: FloatArray, target: FloatArray) -> float:
    tn = float(np.sum((pred_bin == 0) & (target == 0)))
    fp = float(np.sum((pred_bin == 1) & (target == 0)))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _auc(pred: FloatArray, target: FloatArray) -> float:
    """One-vs-rest AUC via the rank-sum (Mann-Whitney) statistic."""
    pos = target == 1
    neg = target == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(pred, kind="mergesort")
    ranks = np.empty(len(pred), dtype=np.float64)
    ranks[order] = np.arange(1, len(pred) + 1)
    # Average ranks for ties.
    sorted_pred = pred[order]
    i = 0
    while i < len(sorted_pred):
        j = i
        while j + 1 < len(sorted_pred) and sorted_pred[j + 1] == sorted_pred[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _c_index(pred: FloatArray, target: FloatArray) -> float:
    """Concordance index for a single continuous risk output.

    Vectorised over all comparable pairs (those with differing targets):
    a pair is concordant when the higher-target sample has the higher
    prediction, half-credit on prediction ties.
    """
    n = len(pred)
    if n < 2:
        return 0.5
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # Pairwise sign matrices over the upper triangle.
    t_diff = target[:, None] - target[None, :]
    p_diff = pred[:, None] - pred[None, :]
    comparable = np.triu(t_diff != 0, k=1)
    total = float(np.sum(comparable))
    if total == 0:
        return 0.5
    # Orient each pair so the higher target is the reference.
    concordant = np.sign(t_diff) == np.sign(p_diff)
    ties = p_diff == 0
    conc = float(np.sum(comparable & concordant))
    tie = float(np.sum(comparable & ties))
    return (conc + 0.5 * tie) / total


def _macro_f1(pred: FloatArray, target: FloatArray, threshold: float = 0.5) -> float:
    pred_bin = (pred >= threshold).astype(np.float64)
    scores = [
        _f1(pred_bin[:, c], target[:, c]) for c in range(pred.shape[1])
    ]
    return float(np.mean(scores)) if scores else 0.0


def _mean_c_index(pred: FloatArray, target: FloatArray) -> float:
    scores = [_c_index(pred[:, t], target[:, t]) for t in range(pred.shape[1])]
    return float(np.mean(scores)) if scores else 0.5


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_class(
    pred_a: FloatArray,
    pred_b: FloatArray,
    target: FloatArray,
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Paired-bootstrap a single-class delta.

    Returns ``(p_value, ci_low, ci_high)`` for ``metric_b - metric_a`` over
    resampled rows.  ``p_value`` is two-sided: ``2 * min(P(Δ<=0), P(Δ>=0))``.
    """
    n = len(target)
    deltas = np.empty(n_bootstrap, dtype=np.float64)

    def _metric(pb: FloatArray, tg: FloatArray) -> float:
        if metric == "c_index":
            return _c_index(pb, tg)
        bin_ = (pb >= threshold).astype(np.float64)
        if metric == "f1":
            return _f1(bin_, tg)
        if metric == "sensitivity":
            return _sensitivity(bin_, tg)
        if metric == "specificity":
            return _specificity(bin_, tg)
        if metric == "auc":
            return _auc(pb, tg)
        raise ValueError(f"unknown metric {metric}")

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _metric(pred_b[idx], target[idx]) - _metric(
            pred_a[idx], target[idx]
        )

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    # Two-sided p-value from the sign distribution of the bootstrap deltas.
    p_le = float(np.mean(deltas <= 0.0))
    p_ge = float(np.mean(deltas >= 0.0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))
    return p_value, ci_low, ci_high


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare_predictions(
    preds_a: Dict[str, FloatArray],
    preds_b: Dict[str, FloatArray],
    targets: Dict[str, FloatArray],
    *,
    version_a: str = "model_a",
    version_b: str = "model_b",
    tasks: Any = "all",
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    threshold: float = 0.5,
) -> ModelComparisonReport:
    """Compare two models' predictions and return a comparison report.

    Args:
        preds_a: Per-task prediction arrays for model A (probabilities for
            classification tasks, continuous scores for risk).
        preds_b: Per-task prediction arrays for model B.
        targets: Per-task ground-truth label arrays (shared by both models).
        version_a: Identifier for model A.
        version_b: Identifier for model B.
        tasks: ``"all"`` or an explicit list of task names to compare.
        metadata: Optional per-sample dicts with ``age``/``sex`` for
            demographic subgroup comparison.
        n_bootstrap: Bootstrap resamples for significance testing.
        seed: RNG seed for reproducible bootstrapping.
        alpha: Significance level for regression detection.
        threshold: Decision threshold for binarising classification scores.

    Returns:
        A :class:`ModelComparisonReport`.
    """
    if tasks == "all" or tasks is None:
        task_list = [t for t in ALL_TASKS if t in preds_a and t in preds_b]
    else:
        task_list = [t for t in tasks if t in preds_a and t in preds_b]

    rng = np.random.default_rng(seed)
    task_deltas: Dict[str, TaskDelta] = {}
    regressions: List[str] = []
    n_samples = 0

    for task in task_list:
        pa = np.asarray(preds_a[task], dtype=np.float64)
        pb = np.asarray(preds_b[task], dtype=np.float64)
        tg = np.asarray(targets[task], dtype=np.float64)
        n_samples = max(n_samples, tg.shape[0])
        td = TaskDelta(task_name=task)

        if task in CLASSIFICATION_TASKS:
            td.delta_macro_f1 = _macro_f1(pb, tg, threshold) - _macro_f1(
                pa, tg, threshold
            )
            class_names = _CLASS_NAMES.get(task, [])
            for c in range(tg.shape[1]):
                name = (
                    class_names[c] if c < len(class_names) else f"{task}_{c}"
                )
                pa_c, pb_c, tg_c = pa[:, c], pb[:, c], tg[:, c]
                bin_a = (pa_c >= threshold).astype(np.float64)
                bin_b = (pb_c >= threshold).astype(np.float64)
                deltas = {
                    "f1": _f1(bin_b, tg_c) - _f1(bin_a, tg_c),
                    "auc": _auc(pb_c, tg_c) - _auc(pa_c, tg_c),
                    "sensitivity": _sensitivity(bin_b, tg_c)
                    - _sensitivity(bin_a, tg_c),
                    "specificity": _specificity(bin_b, tg_c)
                    - _specificity(bin_a, tg_c),
                }
                p_value, ci_low, ci_high = _bootstrap_class(
                    pa_c, pb_c, tg_c, "f1", n_bootstrap, rng, threshold
                )
                is_reg = deltas["f1"] < 0 and p_value < alpha
                cd = ClassDelta(
                    name=name,
                    deltas=deltas,
                    primary_metric="f1",
                    p_value=p_value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    is_regression=is_reg,
                )
                td.per_class.append(cd)
                if is_reg:
                    regressions.append(f"{task}/{name}")

        elif task == "risk":
            td.delta_c_index = _mean_c_index(pb, tg) - _mean_c_index(pa, tg)
            class_names = _CLASS_NAMES.get("risk", [])
            for t in range(tg.shape[1]):
                name = (
                    class_names[t] if t < len(class_names) else f"risk_{t}"
                )
                pa_t, pb_t, tg_t = pa[:, t], pb[:, t], tg[:, t]
                delta_ci = _c_index(pb_t, tg_t) - _c_index(pa_t, tg_t)
                p_value, ci_low, ci_high = _bootstrap_class(
                    pa_t, pb_t, tg_t, "c_index", n_bootstrap, rng
                )
                is_reg = delta_ci < 0 and p_value < alpha
                cd = ClassDelta(
                    name=name,
                    deltas={"c_index": delta_ci},
                    primary_metric="c_index",
                    p_value=p_value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    is_regression=is_reg,
                )
                td.per_class.append(cd)
                if is_reg:
                    regressions.append(f"risk/{name}")

        task_deltas[task] = td

    subgroup_deltas = _subgroup_deltas(
        preds_a, preds_b, targets, task_list, metadata, threshold
    )

    recommendation = _recommend(task_deltas, regressions)

    return ModelComparisonReport(
        version_a=version_a,
        version_b=version_b,
        tasks=task_list,
        task_deltas=task_deltas,
        subgroup_deltas=subgroup_deltas,
        regressions=regressions,
        recommendation=recommendation,
        n_samples=n_samples,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def _subgroup_deltas(
    preds_a: Dict[str, FloatArray],
    preds_b: Dict[str, FloatArray],
    targets: Dict[str, FloatArray],
    task_list: List[str],
    metadata: Optional[List[Optional[Dict[str, Any]]]],
    threshold: float,
) -> List[SubgroupDelta]:
    if not metadata:
        return []
    n = len(metadata)
    masks = _build_subgroup_masks(metadata, n)
    out: List[SubgroupDelta] = []
    for sg_name, mask in sorted(masks.items()):
        sg_n = int(mask.sum())
        if sg_n == 0:
            continue
        sd = SubgroupDelta(subgroup_name=sg_name, n_samples=sg_n)
        for task in task_list:
            tg = np.asarray(targets[task], dtype=np.float64)
            if mask.shape[0] != tg.shape[0]:
                continue
            pa = np.asarray(preds_a[task], dtype=np.float64)[mask]
            pb = np.asarray(preds_b[task], dtype=np.float64)[mask]
            tgm = tg[mask]
            if task in CLASSIFICATION_TASKS:
                sd.delta_macro_f1[task] = _macro_f1(
                    pb, tgm, threshold
                ) - _macro_f1(pa, tgm, threshold)
            elif task == "risk":
                sd.delta_c_index[task] = _mean_c_index(pb, tgm) - _mean_c_index(
                    pa, tgm
                )
        out.append(sd)
    return out


def _recommend(
    task_deltas: Dict[str, TaskDelta], regressions: List[str]
) -> str:
    """Recommend upgrade / hold / investigate from the deltas."""
    if regressions:
        return "investigate"
    net = 0.0
    for td in task_deltas.values():
        net += td.delta_macro_f1 + td.delta_c_index
    if net > 0:
        return "upgrade"
    return "hold"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(report: ModelComparisonReport) -> str:
    lines: List[str] = []
    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append(f"- **Model A:** `{report.version_a}`")
    lines.append(f"- **Model B:** `{report.version_b}`")
    lines.append(f"- **Samples compared:** {report.n_samples}")
    lines.append(f"- **Bootstrap resamples:** {report.n_bootstrap}")
    lines.append(
        f"- **Recommendation:** **{report.recommendation.upper()}**"
    )
    lines.append("")

    # Summary table of task-level deltas.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Task | Δ Macro-F1 | Δ C-index | Regressions |")
    lines.append("| --- | ---: | ---: | ---: |")
    for task in report.tasks:
        td = report.task_deltas.get(task)
        if td is None:
            continue
        n_reg = sum(1 for c in td.per_class if c.is_regression)
        f1_cell = f"{td.delta_macro_f1:+.4f}" if task in CLASSIFICATION_TASKS else "—"
        ci_cell = f"{td.delta_c_index:+.4f}" if task == "risk" else "—"
        lines.append(f"| {task} | {f1_cell} | {ci_cell} | {n_reg} |")
    lines.append("")

    # Regression warnings.
    if report.regressions:
        lines.append("## ⚠️ Regression Warnings")
        lines.append("")
        lines.append(
            f"{len(report.regressions)} class(es) where model B is "
            f"significantly worse (p < {report.alpha}):"
        )
        lines.append("")
        for reg in report.regressions:
            lines.append(f"- `{reg}`")
        lines.append("")

    # Per-task breakdown.
    for task in report.tasks:
        td = report.task_deltas.get(task)
        if td is None or not td.per_class:
            continue
        lines.append(f"## {task.capitalize()} — Per-Class Deltas")
        lines.append("")
        if task == "risk":
            lines.append("| Output | Δ C-index | p | 95% CI | Regression |")
            lines.append("| --- | ---: | ---: | :---: | :---: |")
            for c in td.per_class:
                flag = "🔴" if c.is_regression else ""
                lines.append(
                    f"| {c.name} | {c.deltas.get('c_index', 0.0):+.4f} | "
                    f"{c.p_value:.3f} | "
                    f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {flag} |"
                )
        else:
            lines.append(
                "| Class | Δ F1 | Δ AUC | Δ Sens | Δ Spec | p | 95% CI | Reg |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |"
            )
            for c in td.per_class:
                flag = "🔴" if c.is_regression else ""
                lines.append(
                    f"| {c.name} | {c.deltas.get('f1', 0.0):+.4f} | "
                    f"{c.deltas.get('auc', 0.0):+.4f} | "
                    f"{c.deltas.get('sensitivity', 0.0):+.4f} | "
                    f"{c.deltas.get('specificity', 0.0):+.4f} | "
                    f"{c.p_value:.3f} | "
                    f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {flag} |"
                )
        lines.append("")

    # Demographic subgroup deltas.
    if report.subgroup_deltas:
        lines.append("## Demographic Subgroup Deltas")
        lines.append("")
        lines.append("| Subgroup | n | Δ Macro-F1 (mean) | Δ C-index |")
        lines.append("| --- | ---: | ---: | ---: |")
        for sd in report.subgroup_deltas:
            f1_vals = list(sd.delta_macro_f1.values())
            ci_vals = list(sd.delta_c_index.values())
            f1_mean = f"{np.mean(f1_vals):+.4f}" if f1_vals else "—"
            ci_mean = f"{np.mean(ci_vals):+.4f}" if ci_vals else "—"
            lines.append(
                f"| {sd.subgroup_name} | {sd.n_samples} | {f1_mean} | {ci_mean} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model-loading entry point (torch)
# ---------------------------------------------------------------------------


def compare_models(
    model_a_path: str,
    model_b_path: str,
    dataset: Any,
    tasks: Any = "all",
    *,
    batch_size: int = 64,
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> ModelComparisonReport:
    """Load two checkpoints, run inference, and compare them.

    Args:
        model_a_path: Path to model A checkpoint.
        model_b_path: Path to model B checkpoint.
        dataset: A PyTorch ``Dataset`` yielding ``(x, labels)`` tuples.
        tasks: ``"all"`` or a list of task names.
        batch_size: Inference batch size.
        metadata: Optional per-sample demographic dicts.
        n_bootstrap: Bootstrap resamples for significance testing.
        seed: RNG seed.
        alpha: Significance level for regression detection.

    Returns:
        A :class:`ModelComparisonReport`.
    """
    preds_a, targets = _gather_predictions(
        model_a_path, dataset, tasks, batch_size
    )
    preds_b, _ = _gather_predictions(model_b_path, dataset, tasks, batch_size)

    return compare_predictions(
        preds_a,
        preds_b,
        targets,
        version_a=str(model_a_path),
        version_b=str(model_b_path),
        tasks=tasks,
        metadata=metadata,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )


def _load_model(path: str) -> Any:
    """Load an ``AorticaModel`` from a checkpoint path."""
    import torch

    from aortica.models.aortica_model import AorticaModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, AorticaModel):
        return checkpoint
    model = AorticaModel()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    return model


def _gather_predictions(
    model_path: str,
    dataset: Any,
    tasks: Any,
    batch_size: int,
) -> Tuple[Dict[str, FloatArray], Dict[str, FloatArray]]:
    """Run a model over a dataset and collect per-task predictions/targets."""
    import torch

    model = _load_model(model_path)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    enabled = list(getattr(model, "enabled_tasks", ALL_TASKS))
    if tasks == "all" or tasks is None:
        task_list = enabled
    else:
        task_list = [t for t in tasks if t in enabled]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_preds: Dict[str, List[FloatArray]] = {t: [] for t in task_list}
    all_targets: Dict[str, List[FloatArray]] = {t: [] for t in task_list}

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            features = model.attention(model.backbone(batch_x))
            task_labels = _split_labels(batch_y.cpu().numpy(), enabled)
            for task in task_list:
                head = getattr(model, f"{task}_head", None)
                if head is None:
                    continue
                all_preds[task].append(head(features).cpu().numpy())
                if task in task_labels:
                    all_targets[task].append(task_labels[task])

    preds: Dict[str, FloatArray] = {}
    targets: Dict[str, FloatArray] = {}
    for task in task_list:
        if all_preds[task]:
            preds[task] = np.concatenate(all_preds[task], axis=0)
        if all_targets[task]:
            targets[task] = np.concatenate(all_targets[task], axis=0)
    return preds, targets
