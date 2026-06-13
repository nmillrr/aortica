"""Knowledge distillation training for edge models.

Trains a lightweight student model (e.g. :class:`MobileNetBackbone1D`) to
mimic a full teacher model (e.g. :class:`AorticaModel`) using a combination
of KL-divergence on temperature-scaled soft targets and hard-label
cross-entropy loss.

The distillation loss is:

    L = alpha * KL(softened_teacher || softened_student) * T^2
      + (1 - alpha) * CE(student_logits, hard_labels)

where T is the temperature and alpha controls the balance between soft
and hard targets.

Usage::

    from aortica.edge.distillation import train_distillation, DistillationConfig

    config = DistillationConfig(temperature=4.0, alpha=0.7)
    result = train_distillation(teacher, student, train_loader, val_loader, config)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]
    nn = types.ModuleType("nn")  # type: ignore[assignment]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for knowledge distillation. "
            "Install with: pip install aortica[torch]"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training.

    Attributes:
        temperature: Softmax temperature for soft targets. Default ``4.0``.
        alpha: Weight for distillation (soft) loss vs hard-label loss.
            ``alpha=1.0`` means pure distillation, ``alpha=0.0`` means
            pure hard-label training.  Default ``0.7``.
        epochs: Number of training epochs.  Default ``30``.
        lr: Peak learning rate.  Default ``1e-3``.
        batch_size: Training batch size.  Default ``64``.
        seed: Random seed.  Default ``42``.
        output_dir: Directory for checkpoints and metrics.  Default
            ``./checkpoints_distillation``.
        weight_decay: AdamW weight decay.  Default ``1e-4``.
        max_grad_norm: Gradient clipping norm.  Default ``1.0``.
        warmup_epochs: Linear LR warm-up epochs.  Default ``5``.
        save_metric: Metric to select best checkpoint.  One of
            ``val_loss``, ``rhythm_f1``, ``structural_f1``,
            ``ischaemia_f1``, ``risk_c_index``.  Default ``val_loss``.
        enabled_tasks: Task heads to distill.  Default all four.
        loss_weights: Per-task loss coefficients.  Default equal.
    """

    temperature: float = 4.0
    alpha: float = 0.7
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    output_dir: str = "./checkpoints_distillation"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_epochs: int = 5
    save_metric: str = "val_loss"
    enabled_tasks: list[str] = field(
        default_factory=lambda: ["rhythm", "structural", "ischaemia", "risk"]
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rhythm": 1.0,
            "structural": 1.0,
            "ischaemia": 1.0,
            "risk": 1.0,
        }
    )


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Must stay in sync with the head class constants (rhythm=28, structural=19,
# ischaemia=19, risk=6) and benchmark.TASK_NUM_OUTPUTS.
_TASK_NUM_OUTPUTS: dict[str, int] = {
    "rhythm": 28,
    "structural": 19,
    "ischaemia": 19,
    "risk": 6,
}

_CLASSIFICATION_TASKS: set[str] = {"rhythm", "structural", "ischaemia"}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def distillation_loss_classification(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    hard_labels: "torch.Tensor",
    temperature: float,
    alpha: float,
) -> "torch.Tensor":
    """Compute distillation loss for a classification (multi-label) head.

    For multi-label (sigmoid) outputs the soft-target loss uses
    element-wise KL divergence between temperature-scaled sigmoid
    probabilities.

    Args:
        student_logits: Raw student logits ``[batch, C]``.
        teacher_logits: Raw teacher logits ``[batch, C]``.
        hard_labels: Binary ground-truth labels ``[batch, C]``.
        temperature: Temperature for softening logits.
        alpha: Weight in ``[0, 1]``.  ``alpha=1`` = pure distillation.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()
    temp = temperature

    # Soft targets via temperature-scaled sigmoid
    teacher_soft = torch.sigmoid(teacher_logits / temp)
    student_soft = torch.sigmoid(student_logits / temp)

    # Element-wise binary KL divergence:
    #   D_KL(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
    # Stabilised with clamp.
    eps = 1e-7
    p = teacher_soft.clamp(eps, 1 - eps)
    q = student_soft.clamp(eps, 1 - eps)

    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    soft_loss = kl.mean() * (temp * temp)  # scale by T^2 per Hinton et al.

    # Hard-label BCE loss
    hard_loss = nn.functional.binary_cross_entropy_with_logits(student_logits, hard_labels)

    combined: torch.Tensor = alpha * soft_loss + (1 - alpha) * hard_loss
    return combined


def distillation_loss_regression(
    student_preds: "torch.Tensor",
    teacher_preds: "torch.Tensor",
    hard_labels: "torch.Tensor",
    alpha: float,
) -> "torch.Tensor":
    """Compute distillation loss for a regression (risk) head.

    Uses MSE between student and teacher predictions as the soft loss
    and MSE between student and ground-truth as the hard loss.

    Args:
        student_preds: Student predictions ``[batch, K]``.
        teacher_preds: Teacher predictions ``[batch, K]``.
        hard_labels: Ground-truth targets ``[batch, K]``.
        alpha: Weight in ``[0, 1]``.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()
    soft_loss = nn.functional.mse_loss(student_preds, teacher_preds)
    hard_loss = nn.functional.mse_loss(student_preds, hard_labels)

    combined: torch.Tensor = alpha * soft_loss + (1 - alpha) * hard_loss
    return combined


# ---------------------------------------------------------------------------
# Epoch result
# ---------------------------------------------------------------------------


@dataclass
class DistillationEpochMetrics:
    """Metrics for one epoch of distillation training."""

    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    per_task_train_loss: dict[str, float] = field(default_factory=dict)
    per_task_val_loss: dict[str, float] = field(default_factory=dict)
    distillation_loss: float = 0.0
    hard_loss: float = 0.0
    rhythm_f1: float = 0.0
    structural_f1: float = 0.0
    ischaemia_f1: float = 0.0
    risk_c_index: float = 0.5


@dataclass
class DistillationResult:
    """Container for the full distillation training result.

    Attributes:
        history: Per-epoch metrics.
        best_epoch: Epoch that achieved the best ``save_metric``.
        best_metric_value: The best value of the configured save metric.
        student_checkpoint_path: Path to the saved best student checkpoint.
    """

    history: list[DistillationEpochMetrics]
    best_epoch: int = 0
    best_metric_value: float = 0.0
    student_checkpoint_path: str = ""


# ---------------------------------------------------------------------------
# Label splitting
# ---------------------------------------------------------------------------


def _split_labels(
    labels: "torch.Tensor",
    enabled_tasks: list[str],
) -> dict[str, "torch.Tensor"]:
    """Split concatenated labels into per-task tensors."""
    result: dict[str, "torch.Tensor"] = {}
    offset = 0
    for task in enabled_tasks:
        width = _TASK_NUM_OUTPUTS[task]
        result[task] = labels[:, offset : offset + width]
        offset += width
    return result


# ---------------------------------------------------------------------------
# Metric helpers (reused from train_multitask)
# ---------------------------------------------------------------------------


def _compute_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
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


def _compute_c_index(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean concordance index across risk outputs."""
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
# LR schedule
# ---------------------------------------------------------------------------


def _cosine_annealing_with_warmup(
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    peak_lr: float,
) -> float:
    """Linear warm-up + cosine annealing LR schedule."""
    if epoch < warmup_epochs:
        lr = peak_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Checkpoint save metric helpers
# ---------------------------------------------------------------------------


def _extract_save_metric(
    metrics: DistillationEpochMetrics,
    save_metric: str,
) -> float:
    """Extract the scalar value of the configured save metric."""
    if save_metric == "val_loss":
        return metrics.val_loss
    return float(getattr(metrics, save_metric, 0.0))


def _is_better(current: float, best: Optional[float], save_metric: str) -> bool:
    """Return True if *current* is an improvement over *best*."""
    if best is None:
        return True
    if save_metric == "val_loss":
        return current < best  # lower is better
    return current > best  # higher is better (F1, C-index)


# ---------------------------------------------------------------------------
# Core: one-epoch distillation
# ---------------------------------------------------------------------------


def _distill_one_epoch(
    teacher: "nn.Module",
    student: "nn.Module",
    dataloader: "DataLoader[Any]",
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    config: DistillationConfig,
) -> tuple[float, dict[str, float]]:
    """Train the student for one epoch via knowledge distillation.

    The teacher runs in eval mode with no gradients.  The student trains
    normally with gradient clipping.

    Returns:
        Tuple of (average_total_loss, per_task_average_losses).
    """
    teacher.eval()
    student.train()

    running_loss = 0.0
    running_per_task: dict[str, float] = {}
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()

        optimizer.zero_grad()

        # Teacher forward — no gradients
        with torch.no_grad():
            teacher_features = teacher.backbone(batch_x)
            teacher_features = teacher.attention(teacher_features)

        # Student forward
        student_features = student.backbone(batch_x)
        student_features = student.attention(student_features)

        task_labels = _split_labels(batch_y, config.enabled_tasks)

        total_loss = torch.tensor(0.0, device=device, dtype=batch_x.dtype)
        per_task: dict[str, float] = {}

        for task in config.enabled_tasks:
            teacher_head = getattr(teacher, f"{task}_head", None)
            student_head = getattr(student, f"{task}_head", None)

            if teacher_head is None or student_head is None:
                continue

            weight = config.loss_weights.get(task, 1.0)

            if task in _CLASSIFICATION_TASKS:
                with torch.no_grad():
                    teacher_logits = teacher_head.forward_logits(teacher_features)
                student_logits = student_head.forward_logits(student_features)

                loss = distillation_loss_classification(
                    student_logits,
                    teacher_logits,
                    task_labels[task],
                    temperature=config.temperature,
                    alpha=config.alpha,
                )
            else:
                # Risk head — regression
                with torch.no_grad():
                    teacher_preds = teacher_head(teacher_features)
                student_preds = student_head(student_features)

                loss = distillation_loss_regression(
                    student_preds,
                    teacher_preds,
                    task_labels[task],
                    alpha=config.alpha,
                )

            total_loss = total_loss + weight * loss
            per_task[task] = loss.item()

        total_loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        optimizer.step()

        running_loss += total_loss.item()
        for k, v in per_task.items():
            running_per_task[k] = running_per_task.get(k, 0.0) + v
        num_batches += 1

    n = max(num_batches, 1)
    avg_loss = running_loss / n
    avg_per_task = {k: v / n for k, v in running_per_task.items()}
    return avg_loss, avg_per_task


def _evaluate_student(
    student: "nn.Module",
    dataloader: "DataLoader[Any]",
    device: "torch.device",
    config: DistillationConfig,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    """Evaluate the student model on a validation set.

    Returns:
        Tuple of (avg_loss, per_task_losses, task_metrics).
    """
    student.eval()
    running_loss = 0.0
    running_per_task: dict[str, float] = {}
    num_batches = 0

    collectors: dict[str, dict[str, list[np.ndarray]]] = {
        t: {"preds": [], "targets": []} for t in config.enabled_tasks
    }

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            student_features = student.backbone(batch_x)
            student_features = student.attention(student_features)

            task_labels = _split_labels(batch_y, config.enabled_tasks)

            total_loss = torch.tensor(0.0, device=device, dtype=batch_x.dtype)
            per_task: dict[str, float] = {}

            for task in config.enabled_tasks:
                head = getattr(student, f"{task}_head", None)
                if head is None:
                    continue

                weight = config.loss_weights.get(task, 1.0)

                if task in _CLASSIFICATION_TASKS:
                    logits = head.forward_logits(student_features)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, task_labels[task],
                    )
                    preds = torch.sigmoid(logits)
                else:
                    preds = head(student_features)
                    loss = nn.functional.mse_loss(preds, task_labels[task])

                total_loss = total_loss + weight * loss
                per_task[task] = loss.item()

                collectors[task]["preds"].append(preds.cpu().numpy())
                collectors[task]["targets"].append(
                    task_labels[task].cpu().numpy()
                )

            running_loss += total_loss.item()
            for k, v in per_task.items():
                running_per_task[k] = running_per_task.get(k, 0.0) + v
            num_batches += 1

    n = max(num_batches, 1)
    avg_loss = running_loss / n
    avg_per_task = {k: v / n for k, v in running_per_task.items()}

    task_metrics: dict[str, Any] = {}
    for task in config.enabled_tasks:
        if not collectors[task]["preds"]:
            continue
        preds_np = np.concatenate(collectors[task]["preds"], axis=0)
        tgts_np = np.concatenate(collectors[task]["targets"], axis=0)

        if task in _CLASSIFICATION_TASKS:
            macro_f1 = _compute_f1(preds_np, tgts_np)
            task_metrics[f"{task}_f1"] = macro_f1
        elif task == "risk":
            c_idx = _compute_c_index(preds_np, tgts_np)
            task_metrics["risk_c_index"] = c_idx

    return avg_loss, avg_per_task, task_metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_distillation(
    teacher: "nn.Module",
    student: "nn.Module",
    train_loader: "DataLoader[Any]",
    val_loader: "DataLoader[Any]",
    config: Optional[DistillationConfig] = None,
) -> DistillationResult:
    """Train a student model via knowledge distillation from a teacher.

    The teacher is frozen (eval mode, no gradients).  The student is
    trained with a combined loss of KL-divergence on temperature-scaled
    soft targets and hard-label cross-entropy / MSE.

    Args:
        teacher: Pre-trained teacher model (e.g. :class:`AorticaModel`).
        student: Untrained student model (same task heads but lighter
            backbone, e.g. :class:`AorticaModel` with
            :class:`MobileNetBackbone1D`).
        train_loader: Training data loader yielding ``(x, y)`` batches.
        val_loader: Validation data loader.
        config: Distillation configuration.  Uses defaults if ``None``.

    Returns:
        A :class:`DistillationResult` with per-epoch metrics and
        best-checkpoint path.
    """
    _check_torch()

    if config is None:
        config = DistillationConfig()

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.to(device)
    student = student.to(device)

    # Freeze teacher entirely
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[DistillationEpochMetrics] = []
    best_metric_val: Optional[float] = None
    best_epoch = 0
    ckpt_path = output_path / "best_student_model.pt"

    for epoch in range(config.epochs):
        lr = _cosine_annealing_with_warmup(
            optimizer, epoch, config.epochs, config.warmup_epochs, config.lr,
        )

        train_loss, train_per_task = _distill_one_epoch(
            teacher, student, train_loader, optimizer, device, config,
        )
        val_loss, val_per_task, task_metrics = _evaluate_student(
            student, val_loader, device, config,
        )

        metrics = DistillationEpochMetrics(
            epoch=epoch + 1,
            lr=lr,
            train_loss=train_loss,
            val_loss=val_loss,
            per_task_train_loss=train_per_task,
            per_task_val_loss=val_per_task,
            rhythm_f1=task_metrics.get("rhythm_f1", 0.0),
            structural_f1=task_metrics.get("structural_f1", 0.0),
            ischaemia_f1=task_metrics.get("ischaemia_f1", 0.0),
            risk_c_index=task_metrics.get("risk_c_index", 0.5),
        )
        history.append(metrics)

        # Log
        task_str = "  ".join(f"{k}={v:.4f}" for k, v in val_per_task.items())
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in task_metrics.items())
        print(
            f"Epoch {epoch + 1}/{config.epochs}  lr={lr:.6f}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"{task_str}  {metric_str}"
        )

        # Save best checkpoint
        current_metric = _extract_save_metric(metrics, config.save_metric)
        if _is_better(current_metric, best_metric_val, config.save_metric):
            best_metric_val = current_metric
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": asdict(metrics),
                    "config": asdict(config),
                },
                ckpt_path,
            )

    # Save training history
    metrics_path = output_path / "distillation_metrics.json"
    metrics_data: list[dict[str, Any]] = [asdict(m) for m in history]
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nDistillation complete.  Best {config.save_metric}: {best_metric_val}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics:    {metrics_path}")

    return DistillationResult(
        history=history,
        best_epoch=best_epoch,
        best_metric_value=best_metric_val if best_metric_val is not None else 0.0,
        student_checkpoint_path=str(ckpt_path),
    )
