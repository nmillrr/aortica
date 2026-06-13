"""Edge model validation harness.

Compares a full :class:`~aortica.models.AorticaModel` against an edge model
(ONNX or ONNX INT8) on a shared dataset, reporting per-task performance
metrics and pass/fail status against a configurable degradation threshold.

The primary entry point is :func:`validate_edge`, which returns an
:class:`EdgeValidationReport`.

Example::

    from aortica.edge import validate_edge

    report = validate_edge(
        full_model=full_model,
        edge_model_path="aortica_edge_int8.onnx",
        dataset=test_dataset,
        tasks=["rhythm", "structural", "ischaemia", "risk"],
    )
    print(report.summary_table())

"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

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

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn = types.ModuleType("nn")  # type: ignore[assignment]
    nn.Module = _DummyModule  # type: ignore[attr-defined]


try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for validate_edge(). "
            "Install with: pip install aortica[torch]"
        )


def _check_onnxruntime() -> None:
    if not HAS_ONNXRUNTIME:
        raise ImportError(
            "onnxruntime is required for edge model validation. "
            "Install with: pip install aortica[edge]"
        )


# Task output sizes (same as benchmark.TASK_NUM_OUTPUTS). Must stay in sync with
# the head class constants (rhythm=28, structural=19, ischaemia=19, risk=6).
TASK_NUM_OUTPUTS: dict[str, int] = {
    "rhythm": 28,
    "structural": 19,
    "ischaemia": 19,
    "risk": 6,
}

CLASSIFICATION_TASKS: list[str] = ["rhythm", "structural", "ischaemia"]
ALL_TASKS: list[str] = ["rhythm", "structural", "ischaemia", "risk"]

# Default pass/fail threshold: maximum allowed relative performance
# degradation (as a fraction, e.g. 0.03 = 3%).
DEFAULT_THRESHOLD: float = 0.03


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskValidation:
    """Per-task validation comparing full and edge model metrics.

    Attributes:
        task_name: Task identifier (rhythm, structural, ischaemia, risk).
        full_metric: Primary metric value for the full model.
        edge_metric: Primary metric value for the edge model.
        metric_name: Name of the primary metric used for comparison.
        degradation: Absolute difference (full - edge).
        relative_degradation: Fractional degradation (degradation / full).
        passed: Whether the edge model is within the threshold.
        threshold: The threshold used for pass/fail.
    """

    task_name: str = ""
    full_metric: float = 0.0
    edge_metric: float = 0.0
    metric_name: str = ""
    degradation: float = 0.0
    relative_degradation: float = 0.0
    passed: bool = True
    threshold: float = DEFAULT_THRESHOLD


@dataclass
class EdgeValidationReport:
    """Full edge validation results.

    Attributes:
        task_validations: Per-task validation results.
        all_passed: Whether all tasks passed validation.
        edge_latency_ms: Mean inference latency per sample (milliseconds).
        full_latency_ms: Mean inference latency per sample (milliseconds).
        n_samples: Number of samples evaluated.
        tasks_evaluated: Which tasks were compared.
    """

    task_validations: dict[str, TaskValidation] = field(default_factory=dict)
    all_passed: bool = True
    edge_latency_ms: float = 0.0
    full_latency_ms: float = 0.0
    n_samples: int = 0
    tasks_evaluated: list[str] = field(default_factory=list)

    def summary_table(self) -> str:
        """Return a human-readable summary table."""
        lines: list[str] = []
        lines.append(f"Edge Validation Report ({self.n_samples} samples)")
        lines.append("=" * 70)

        lines.append(
            f"\n{'Task':<14} {'Metric':<10} {'Full':>8} {'Edge':>8} "
            f"{'Degrad':>8} {'Status':>8}"
        )
        lines.append("-" * 70)

        for task_name in self.tasks_evaluated:
            tv = self.task_validations.get(task_name)
            if tv is None:
                continue
            status = "PASS" if tv.passed else "FAIL"
            lines.append(
                f"{tv.task_name:<14} {tv.metric_name:<10} "
                f"{tv.full_metric:>8.4f} {tv.edge_metric:>8.4f} "
                f"{tv.relative_degradation:>7.2%} "
                f"{'  ' + status:>8}"
            )

        overall = "ALL PASSED" if self.all_passed else "FAILED"
        lines.append(f"\nOverall: {overall}")
        lines.append(
            f"Edge latency: {self.edge_latency_ms:.2f} ms/sample  |  "
            f"Full latency: {self.full_latency_ms:.2f} ms/sample"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric helpers (reused from benchmark module, inlined for independence)
# ---------------------------------------------------------------------------


def _compute_auc(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute AUC for a single binary class using the trapezoidal rule."""
    pos = targets == 1
    neg = targets == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-predictions)
    sorted_targets = targets[order]

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

    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return float(auc)


def _compute_f1(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    threshold: float = 0.5,
) -> float:
    """Compute macro-F1 from sigmoid outputs."""
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

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _compute_mean_auc(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute mean AUC across all classes for a classification task."""
    num_classes = predictions.shape[1]
    auc_scores: list[float] = []
    for c in range(num_classes):
        auc_scores.append(_compute_auc(predictions[:, c], targets[:, c]))
    return float(np.mean(auc_scores)) if auc_scores else 0.5


def _compute_c_index(
    predictions: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
) -> float:
    """Compute concordance index averaged across risk outputs."""
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


# ---------------------------------------------------------------------------
# Label splitting helper
# ---------------------------------------------------------------------------


def _split_labels(
    labels: npt.NDArray[np.floating[Any]],
    enabled_tasks: list[str],
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Split concatenated label arrays into per-task arrays."""
    result: dict[str, npt.NDArray[np.floating[Any]]] = {}
    offset = 0
    for task in enabled_tasks:
        width = TASK_NUM_OUTPUTS[task]
        result[task] = labels[:, offset: offset + width]
        offset += width
    return result


# ---------------------------------------------------------------------------
# Full model inference
# ---------------------------------------------------------------------------


def _run_full_model(
    model: nn.Module,
    dataset: "torch.utils.data.Dataset[Any]",
    tasks: list[str],
    batch_size: int,
    device: "torch.device",
) -> tuple[dict[str, npt.NDArray[np.floating[Any]]], float]:
    """Run full PyTorch model and return predictions + latency.

    Returns:
        Tuple of (per-task predictions dict, mean latency in ms per sample).
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_preds: dict[str, list[npt.NDArray[np.floating[Any]]]] = {
        t: [] for t in tasks
    }

    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            bs = batch_x.shape[0]

            start = time.perf_counter()
            output = model(batch_x)
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_samples += bs

            for task in tasks:
                tensor = getattr(output, task)
                if tensor is not None:
                    arr: npt.NDArray[np.floating[Any]] = tensor.cpu().numpy()
                    all_preds[task].append(arr)

    preds_np: dict[str, npt.NDArray[np.floating[Any]]] = {}
    for task in tasks:
        if all_preds[task]:
            preds_np[task] = np.concatenate(all_preds[task], axis=0)

    latency_ms = (total_time / max(total_samples, 1)) * 1000.0
    return preds_np, latency_ms


# ---------------------------------------------------------------------------
# Edge model inference (ONNX Runtime)
# ---------------------------------------------------------------------------


def _run_edge_model(
    edge_model_path: str,
    dataset: "torch.utils.data.Dataset[Any]",
    tasks: list[str],
    batch_size: int,
) -> tuple[dict[str, npt.NDArray[np.floating[Any]]], float]:
    """Run ONNX edge model and return predictions + latency.

    Returns:
        Tuple of (per-task predictions dict, mean latency in ms per sample).
    """
    session = ort.InferenceSession(edge_model_path)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    # Map output names to task names — convention: "{task}_output"
    output_task_map: dict[int, str] = {}
    for i, name in enumerate(output_names):
        for task in tasks:
            if name == f"{task}_output":
                output_task_map[i] = task
                break

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_preds: dict[str, list[npt.NDArray[np.floating[Any]]]] = {
        t: [] for t in tasks
    }

    total_time = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        # Convert to numpy for ONNX Runtime
        input_np: npt.NDArray[np.floating[Any]] = batch_x.numpy().astype(
            np.float32,
        )
        bs = input_np.shape[0]

        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_np})
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_samples += bs

        for idx, task in output_task_map.items():
            arr: npt.NDArray[np.floating[Any]] = np.asarray(
                outputs[idx], dtype=np.float64,
            )
            all_preds[task].append(arr)

    preds_np: dict[str, npt.NDArray[np.floating[Any]]] = {}
    for task in tasks:
        if all_preds[task]:
            preds_np[task] = np.concatenate(all_preds[task], axis=0)

    latency_ms = (total_time / max(total_samples, 1)) * 1000.0
    return preds_np, latency_ms


# ---------------------------------------------------------------------------
# Evaluate predictions
# ---------------------------------------------------------------------------


def _evaluate_task(
    task: str,
    full_preds: npt.NDArray[np.floating[Any]],
    edge_preds: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.floating[Any]],
    threshold: float,
) -> TaskValidation:
    """Evaluate a single task and compute pass/fail status."""
    if task in CLASSIFICATION_TASKS:
        full_metric = _compute_f1(full_preds, targets)
        edge_metric = _compute_f1(edge_preds, targets)
        metric_name = "macro_f1"

        # Also compute AUC for comparison
        full_auc = _compute_mean_auc(full_preds, targets)
        edge_auc = _compute_mean_auc(edge_preds, targets)

        # Use AUC as primary metric if F1 is 0 for both (edge case)
        if full_metric == 0.0 and edge_metric == 0.0:
            full_metric = full_auc
            edge_metric = edge_auc
            metric_name = "mean_auc"
    else:
        # Risk task — use C-index
        full_metric = _compute_c_index(full_preds, targets)
        edge_metric = _compute_c_index(edge_preds, targets)
        metric_name = "c_index"

    degradation = full_metric - edge_metric
    if full_metric > 0:
        relative_degradation = degradation / full_metric
    else:
        # If full model metric is 0, any positive edge metric is fine
        relative_degradation = 0.0 if edge_metric >= full_metric else 1.0

    passed = relative_degradation <= threshold

    return TaskValidation(
        task_name=task,
        full_metric=full_metric,
        edge_metric=edge_metric,
        metric_name=metric_name,
        degradation=degradation,
        relative_degradation=relative_degradation,
        passed=passed,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------


def validate_edge(
    full_model: nn.Module,
    edge_model_path: Union[str, "pathlib.Path"],
    dataset: "torch.utils.data.Dataset[Any]",
    tasks: Optional[Sequence[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 64,
    device: Optional["torch.device"] = None,
) -> EdgeValidationReport:
    """Validate that the edge model performs within a threshold of the full model.

    Runs both the full PyTorch model and the ONNX edge model over the same
    dataset, computing per-task metrics and reporting pass/fail status based
    on the maximum allowed relative degradation.

    Args:
        full_model: The full :class:`~aortica.models.AorticaModel` (PyTorch).
        edge_model_path: Path to the edge ONNX model (``.onnx``).
        dataset: A PyTorch ``Dataset`` yielding ``(x, labels)`` tuples
            where ``labels`` has columns for each enabled task.
        tasks: Which task heads to evaluate.  Default: all tasks present
            in ``full_model.enabled_tasks``.
        threshold: Maximum allowed relative performance degradation as a
            fraction.  Default ``0.03`` (3%).
        batch_size: Batch size for evaluation.  Default ``64``.
        device: Device for full model inference.  Default: inferred.

    Returns:
        :class:`EdgeValidationReport` with per-task comparisons and
        overall pass/fail status.
    """

    _check_torch()
    _check_onnxruntime()

    edge_model_path = str(pathlib.Path(edge_model_path))

    if device is None:
        try:
            device = next(full_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if tasks is None:
        enabled: list[str] = list(
            getattr(full_model, "enabled_tasks", ALL_TASKS),
        )
    else:
        enabled = list(tasks)

    # --- Collect ground truth labels ---
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    all_labels: list[npt.NDArray[np.floating[Any]]] = []
    for _, batch_y in dataloader:
        arr: npt.NDArray[np.floating[Any]] = batch_y.numpy().astype(np.float64)
        all_labels.append(arr)
    labels_np = np.concatenate(all_labels, axis=0)
    task_labels = _split_labels(labels_np, enabled)

    # --- Run full model ---
    full_preds, full_latency = _run_full_model(
        full_model, dataset, enabled, batch_size, device,
    )

    # --- Run edge model ---
    edge_preds, edge_latency = _run_edge_model(
        edge_model_path, dataset, enabled, batch_size,
    )

    # --- Per-task evaluation ---
    task_validations: dict[str, TaskValidation] = {}
    all_passed = True

    for task in enabled:
        if task not in full_preds or task not in edge_preds:
            continue
        if task not in task_labels:
            continue

        tv = _evaluate_task(
            task,
            full_preds[task],
            edge_preds[task],
            task_labels[task],
            threshold,
        )
        task_validations[task] = tv
        if not tv.passed:
            all_passed = False

    n_samples: int = len(dataset)  # type: ignore[arg-type]

    return EdgeValidationReport(
        task_validations=task_validations,
        all_passed=all_passed,
        edge_latency_ms=edge_latency,
        full_latency_ms=full_latency,
        n_samples=n_samples,
        tasks_evaluated=enabled,
    )
