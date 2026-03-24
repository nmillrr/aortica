"""Multi-task evaluation harness with demographic subgroup reporting.

Provides :func:`benchmark`, which runs a trained :class:`AorticaModel`
over a dataset and returns a :class:`BenchmarkReport` containing per-task
metrics, per-class breakdowns, and optional demographic subgroup
stratification.

Classification metrics (rhythm, structural, ischaemia):
    - Macro-F1 and per-class F1
    - Per-class AUC (one-vs-rest)
    - Per-class sensitivity (recall) and specificity
    - Expected Calibration Error (ECE)

Risk metrics:
    - Mean C-index across the three risk outputs
    - Brier score (mean squared error of predicted probabilities)

All metric computations are deterministic for a given
model + dataset + seed combination.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn  # noqa: F401

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for benchmark(). "
            "Install with: pip install aortica[torch]"
        )


# Task output sizes (same as train_multitask._TASK_NUM_OUTPUTS)
TASK_NUM_OUTPUTS: dict[str, int] = {
    "rhythm": 22,
    "structural": 15,
    "ischaemia": 10,
    "risk": 3,
}

CLASSIFICATION_TASKS: list[str] = ["rhythm", "structural", "ischaemia"]
ALL_TASKS: list[str] = ["rhythm", "structural", "ischaemia", "risk"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassMetrics:
    """Per-class metrics for a single classification label.

    Attributes:
        name: Class label name (or index string).
        auc: Area under the ROC curve (one-vs-rest).
        sensitivity: True positive rate (recall).
        specificity: True negative rate.
        f1: F1 score.
    """

    name: str = ""
    auc: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0


@dataclass
class TaskReport:
    """Metrics for a single task.

    Attributes:
        task_name: Task identifier (rhythm, structural, ischaemia, risk).
        macro_f1: Macro-averaged F1 (classification tasks only).
        ece: Expected Calibration Error (classification tasks only).
        c_index: Concordance index (risk task only).
        brier_score: Mean squared error of predictions (risk task only).
        per_class: Per-class metric breakdowns (classification tasks only).
    """

    task_name: str = ""
    macro_f1: float = 0.0
    ece: float = 0.0
    c_index: float = 0.0
    brier_score: float = 0.0
    per_class: list[ClassMetrics] = field(default_factory=list)


@dataclass
class SubgroupReport:
    """Metrics for a demographic subgroup.

    Attributes:
        subgroup_name: e.g. "age_50-59", "sex_M".
        n_samples: Number of samples in this subgroup.
        task_reports: Per-task metrics for this subgroup.
    """

    subgroup_name: str = ""
    n_samples: int = 0
    task_reports: dict[str, TaskReport] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark results.

    Attributes:
        overall: Per-task metrics over the full dataset.
        subgroups: Per-task metrics stratified by demographic subgroup.
        n_samples: Total number of samples evaluated.
        tasks_evaluated: List of tasks that were evaluated.
    """

    overall: dict[str, TaskReport] = field(default_factory=dict)
    subgroups: list[SubgroupReport] = field(default_factory=list)
    n_samples: int = 0
    tasks_evaluated: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a plain nested dictionary."""
        return asdict(self)

    def summary_table(self) -> str:
        """Return a human-readable summary table."""
        lines: list[str] = []
        lines.append(f"Benchmark Report ({self.n_samples} samples)")
        lines.append("=" * 60)

        for task_name in self.tasks_evaluated:
            tr = self.overall.get(task_name)
            if tr is None:
                continue
            lines.append(f"\n--- {task_name.upper()} ---")
            if task_name in CLASSIFICATION_TASKS:
                lines.append(f"  Macro-F1: {tr.macro_f1:.4f}")
                lines.append(f"  ECE:      {tr.ece:.4f}")
                lines.append(f"  {'Class':<20} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
                for cm in tr.per_class:
                    lines.append(
                        f"  {cm.name:<20} {cm.auc:>6.3f} "
                        f"{cm.sensitivity:>6.3f} {cm.specificity:>6.3f} "
                        f"{cm.f1:>6.3f}"
                    )
            else:
                lines.append(f"  C-index:    {tr.c_index:.4f}")
                lines.append(f"  Brier:      {tr.brier_score:.4f}")

        if self.subgroups:
            lines.append("\n--- SUBGROUPS ---")
            for sg in self.subgroups:
                lines.append(f"\n  {sg.subgroup_name} (n={sg.n_samples})")
                for task_name, tr in sg.task_reports.items():
                    if task_name in CLASSIFICATION_TASKS:
                        lines.append(
                            f"    {task_name}: macro_f1={tr.macro_f1:.4f} ece={tr.ece:.4f}"
                        )
                    else:
                        lines.append(
                            f"    {task_name}: c_index={tr.c_index:.4f} "
                            f"brier={tr.brier_score:.4f}"
                        )

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export overall per-class results as CSV text."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "task", "class", "auc", "sensitivity", "specificity", "f1",
            "macro_f1", "ece", "c_index", "brier_score",
        ])

        for task_name in self.tasks_evaluated:
            tr = self.overall.get(task_name)
            if tr is None:
                continue
            if task_name in CLASSIFICATION_TASKS:
                for cm in tr.per_class:
                    writer.writerow([
                        task_name, cm.name, f"{cm.auc:.4f}",
                        f"{cm.sensitivity:.4f}", f"{cm.specificity:.4f}",
                        f"{cm.f1:.4f}", f"{tr.macro_f1:.4f}",
                        f"{tr.ece:.4f}", "", "",
                    ])
            else:
                writer.writerow([
                    task_name, "", "", "", "", "",
                    "", "", f"{tr.c_index:.4f}", f"{tr.brier_score:.4f}",
                ])

        return buf.getvalue()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_auc(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute AUC for a single binary class using the trapezoidal rule.

    Args:
        predictions: Predicted probabilities [N].
        targets: Binary ground-truth labels [N].

    Returns:
        AUC value.  Returns 0.5 if only one class is present.
    """
    pos = targets == 1
    neg = targets == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending prediction
    order = np.argsort(-predictions)
    sorted_targets = targets[order]

    # Compute TPR/FPR curve
    tp = 0.0
    fp = 0.0
    tpr_list: list[float] = [0.0]
    fpr_list: list[float] = [0.0]

    for label in sorted_targets:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return float(auc)


def _compute_sensitivity_specificity(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute sensitivity (recall) and specificity for binary predictions."""
    pred_bin = (predictions >= threshold).astype(np.float32)
    tp = float(np.sum((pred_bin == 1) & (targets == 1)))
    fn = float(np.sum((pred_bin == 0) & (targets == 1)))
    tn = float(np.sum((pred_bin == 0) & (targets == 0)))
    fp = float(np.sum((pred_bin == 1) & (targets == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def _compute_f1(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """Compute macro-F1 and per-class F1 from sigmoid outputs."""
    pred_bin = (predictions >= threshold).astype(np.float32)
    num_classes = targets.shape[1]
    f1_scores: list[float] = []

    for c in range(num_classes):
        tp = float(np.sum((pred_bin[:, c] == 1) & (targets[:, c] == 1)))
        fp = float(np.sum((pred_bin[:, c] == 1) & (targets[:, c] == 0)))
        fn = float(np.sum((pred_bin[:, c] == 0) & (targets[:, c] == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return macro_f1, f1_scores


def _compute_ece(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) on numpy arrays."""
    probs = predictions.flatten()
    labs = targets.flatten().astype(np.float32)

    n_total = probs.size
    if n_total == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        avg_confidence = float(probs[mask].mean())
        avg_accuracy = float(labs[mask].mean())
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece


def _compute_c_index(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute concordance index averaged across risk outputs.

    Args:
        predictions: Shape [N, K].
        targets: Shape [N, K].

    Returns:
        Mean C-index across the K tasks (0.0–1.0).
    """
    n, k = predictions.shape
    if n < 2:
        return 0.5

    c_indices: list[float] = []
    for t in range(k):
        concordant: float = 0.0
        discordant: float = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if targets[i, t] == targets[j, t]:
                    continue
                if targets[i, t] > targets[j, t]:
                    hi, lo = i, j
                else:
                    hi, lo = j, i
                if predictions[hi, t] > predictions[lo, t]:
                    concordant += 1.0
                elif predictions[hi, t] < predictions[lo, t]:
                    discordant += 1.0
                else:
                    concordant += 0.5
                    discordant += 0.5
            total = concordant + discordant
        c_indices.append(concordant / total if total > 0 else 0.5)

    return float(np.mean(c_indices))


def _compute_brier_score(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute Brier score (mean squared error of predictions)."""
    return float(np.mean((predictions - targets) ** 2))


# ---------------------------------------------------------------------------
# Task-level evaluation
# ---------------------------------------------------------------------------


def _evaluate_classification_task(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    task_name: str,
    class_names: Optional[list[str]] = None,
) -> TaskReport:
    """Evaluate a classification task and produce a TaskReport."""
    num_classes = predictions.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    macro_f1, per_class_f1 = _compute_f1(predictions, targets)
    ece = _compute_ece(predictions, targets)

    per_class_metrics: list[ClassMetrics] = []
    for c in range(num_classes):
        auc = _compute_auc(predictions[:, c], targets[:, c])
        sens, spec = _compute_sensitivity_specificity(
            predictions[:, c], targets[:, c]
        )
        per_class_metrics.append(
            ClassMetrics(
                name=class_names[c] if c < len(class_names) else str(c),
                auc=auc,
                sensitivity=sens,
                specificity=spec,
                f1=per_class_f1[c] if c < len(per_class_f1) else 0.0,
            )
        )

    return TaskReport(
        task_name=task_name,
        macro_f1=macro_f1,
        ece=ece,
        per_class=per_class_metrics,
    )


def _evaluate_risk_task(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> TaskReport:
    """Evaluate the risk prediction task and produce a TaskReport."""
    c_index = _compute_c_index(predictions, targets)
    brier = _compute_brier_score(predictions, targets)

    return TaskReport(
        task_name="risk",
        c_index=c_index,
        brier_score=brier,
    )


# ---------------------------------------------------------------------------
# Label splitting helper
# ---------------------------------------------------------------------------


def _split_labels(
    labels: npt.NDArray[np.floating[Any]],
    enabled_tasks: list[str],
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Split concatenated label arrays into per-task arrays.

    Convention: labels are ordered by ``enabled_tasks`` with the number
    of columns defined by ``TASK_NUM_OUTPUTS``.
    """
    result: dict[str, npt.NDArray[np.floating[Any]]] = {}
    offset = 0
    for task in enabled_tasks:
        width = TASK_NUM_OUTPUTS[task]
        result[task] = labels[:, offset: offset + width]
        offset += width
    return result


# ---------------------------------------------------------------------------
# Subgroup helpers
# ---------------------------------------------------------------------------


def _build_subgroup_masks(
    metadata: list[Optional[dict[str, Any]]],
    n_samples: int,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Build boolean index masks for age decile and sex subgroups.

    Age is binned into decades: 0-9, 10-19, ..., 80-89, 90+.
    Sex uses the ``sex`` key (values ``M``, ``F``, or similar).

    Only subgroups with at least 1 sample are included.
    """
    age_bins: dict[str, list[int]] = {}
    sex_bins: dict[str, list[int]] = {}

    for i, meta in enumerate(metadata):
        if meta is None:
            continue

        # Age subgroup
        age = meta.get("age")
        if age is not None:
            try:
                age_val = int(age)
                decade = (age_val // 10) * 10
                if decade >= 90:
                    key = "age_90+"
                else:
                    key = f"age_{decade}-{decade + 9}"
                age_bins.setdefault(key, []).append(i)
            except (ValueError, TypeError):
                pass

        # Sex subgroup
        sex = meta.get("sex")
        if sex is not None:
            sex_key = f"sex_{sex}"
            sex_bins.setdefault(sex_key, []).append(i)

    masks: dict[str, npt.NDArray[np.bool_]] = {}
    for name, indices in {**age_bins, **sex_bins}.items():
        mask = np.zeros(n_samples, dtype=bool)
        mask[indices] = True
        masks[name] = mask

    return masks


# ---------------------------------------------------------------------------
# Class name lookups
# ---------------------------------------------------------------------------

_CLASS_NAMES: dict[str, list[str]] = {
    "rhythm": [
        "AF", "AFL", "SVT", "AVNRT", "AVRT", "VT", "VF",
        "idioventricular", "sinus_brady", "sinus_tachy", "PAC", "PVC",
        "1st_AV_block", "2nd_AV_block", "3rd_AV_block", "LBBB", "RBBB",
        "LAFB", "LPFB", "WPW", "pacemaker_rhythm", "NSR",
    ],
    "structural": [
        "LVH", "RVH", "LVSD", "HFpEF_risk", "DCM", "HCM", "ARVC",
        "amyloidosis", "aortic_stenosis", "mitral_regurgitation",
        "pulmonary_HTN", "LA_enlargement", "RA_enlargement",
        "pericarditis", "myocarditis",
    ],
    "ischaemia": [
        "STEMI", "posterior_MI", "occlusive_NSTEMI", "old_MI",
        "hyperkalaemia", "hypokalaemia", "hypercalcaemia",
        "hypothyroidism", "digitalis_effect", "QTc_prolongation",
    ],
    "risk": [
        "mortality_1yr", "hf_hosp_12mo", "af_onset_12mo",
    ],
}


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------


def benchmark(
    model: "nn.Module",
    dataset: "torch.utils.data.Dataset[Any]",
    tasks: Optional[list[str]] = None,
    batch_size: int = 64,
    seed: int = 42,
    metadata: Optional[list[Optional[dict[str, Any]]]] = None,
    device: Optional["torch.device"] = None,
) -> BenchmarkReport:
    """Evaluate a trained model across all task heads.

    Runs the model in eval mode over *dataset*, computing per-task metrics
    and (optionally) demographic subgroup breakdowns.

    Args:
        model: A trained ``AorticaModel`` (or compatible module with
            ``backbone``, ``attention``, and task head attributes).
        dataset: A PyTorch ``Dataset`` yielding ``(x, labels)`` tuples
            where ``labels`` has columns for each enabled task.
        tasks: Which task heads to evaluate.  Default: all tasks present
            in ``model.enabled_tasks``.
        batch_size: Batch size for evaluation.  Default ``64``.
        seed: Random seed for reproducibility (controls DataLoader).
        metadata: Optional list of dicts (one per sample) with ``age``
            and/or ``sex`` keys for subgroup stratification.  Must have
            the same length as *dataset* if provided.
        device: Device to run on.  Default: inferred from model parameters.

    Returns:
        :class:`BenchmarkReport` with overall and subgroup metrics.
    """
    _check_torch()

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if tasks is None:
        tasks = list(getattr(model, "enabled_tasks", ALL_TASKS))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect predictions and labels
    all_preds: dict[str, list[npt.NDArray[np.floating[Any]]]] = {
        t: [] for t in tasks
    }
    all_targets: dict[str, list[npt.NDArray[np.floating[Any]]]] = {
        t: [] for t in tasks
    }

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            features = model.backbone(batch_x)
            features = model.attention(features)

            task_labels = _split_labels(
                batch_y.cpu().numpy(),
                [t for t in model.enabled_tasks if t in tasks],
            )

            for task in tasks:
                head = getattr(model, f"{task}_head", None)
                if head is None:
                    continue
                preds_tensor: torch.Tensor = head(features)
                all_preds[task].append(preds_tensor.cpu().numpy())
                if task in task_labels:
                    all_targets[task].append(task_labels[task])

    # Concatenate
    preds_np: dict[str, npt.NDArray[np.floating[Any]]] = {}
    tgts_np: dict[str, npt.NDArray[np.floating[Any]]] = {}
    for task in tasks:
        if all_preds[task]:
            preds_np[task] = np.concatenate(all_preds[task], axis=0)
        if all_targets[task]:
            tgts_np[task] = np.concatenate(all_targets[task], axis=0)

    n_samples: int = len(dataset)  # type: ignore[arg-type]

    # Overall task reports
    overall: dict[str, TaskReport] = {}
    for task in tasks:
        if task not in preds_np or task not in tgts_np:
            continue
        if task in CLASSIFICATION_TASKS:
            class_names = _CLASS_NAMES.get(task)
            overall[task] = _evaluate_classification_task(
                preds_np[task], tgts_np[task], task, class_names
            )
        elif task == "risk":
            overall[task] = _evaluate_risk_task(preds_np[task], tgts_np[task])

    # Subgroup reports
    subgroups: list[SubgroupReport] = []
    if metadata is not None:
        masks = _build_subgroup_masks(metadata, n_samples)
        for sg_name, mask in sorted(masks.items()):
            sg_n = int(mask.sum())
            if sg_n == 0:
                continue
            sg_task_reports: dict[str, TaskReport] = {}
            for task in tasks:
                if task not in preds_np or task not in tgts_np:
                    continue
                sg_preds = preds_np[task][mask]
                sg_tgts = tgts_np[task][mask]
                if sg_preds.shape[0] < 2:
                    continue
                if task in CLASSIFICATION_TASKS:
                    class_names = _CLASS_NAMES.get(task)
                    sg_task_reports[task] = _evaluate_classification_task(
                        sg_preds, sg_tgts, task, class_names
                    )
                elif task == "risk":
                    sg_task_reports[task] = _evaluate_risk_task(
                        sg_preds, sg_tgts
                    )
            subgroups.append(
                SubgroupReport(
                    subgroup_name=sg_name,
                    n_samples=sg_n,
                    task_reports=sg_task_reports,
                )
            )

    return BenchmarkReport(
        overall=overall,
        subgroups=subgroups,
        n_samples=n_samples,
        tasks_evaluated=tasks,
    )
