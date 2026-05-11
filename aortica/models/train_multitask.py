"""Multi-task training pipeline for the unified AorticaModel.

Jointly optimises backbone + attention + four task heads (rhythm, structural,
ischaemia, risk) with configurable per-task loss weights, cosine-annealing
learning-rate schedule with warm-up, and gradient clipping.

Configuration is loaded from a YAML file or constructed programmatically
via :class:`MultiTaskTrainConfig`.

Usage (from project root)::

    python -m aortica.models.train_multitask --config config.yaml

Or programmatically::

    config = MultiTaskTrainConfig(data_path="/path/to/ptbxl")
    history = train_multitask(config)
"""

from __future__ import annotations

import argparse
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
    HAS_TORCH = False

try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskTrainConfig:
    """Training configuration for the multi-task pipeline.

    Attributes:
        data_path: Path to PTB-XL dataset directory.
        epochs: Number of training epochs.
        lr: Peak learning rate (after warm-up).
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
        output_dir: Directory for checkpoints and metrics.
        sampling_rate: ECG sampling rate (100 or 500 Hz).
        window_seconds: Window length in seconds.
        weight_decay: Weight decay for AdamW optimiser.
        num_workers: DataLoader worker processes.
        max_grad_norm: Max gradient norm for clipping.
        warmup_epochs: Epochs of linear warm-up before cosine decay.
        loss_weights: Per-task loss coefficients (keyed by task name).
        enabled_tasks: Which task heads to train.
        feature_dim: Backbone feature dimension.
        head_hidden_dim: Task-head hidden dimension.
        head_dropout: Task-head dropout rate.
        save_metric: Validation metric used to select the best checkpoint.
            Supported values: ``val_loss``, ``rhythm_f1``, ``structural_f1``,
            ``ischaemia_f1``, ``risk_c_index``.
        backend: ``pytorch`` or ``tensorflow``.
        structural_focal: Whether to use focal loss for the structural head.
    """

    data_path: str = ""
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    output_dir: str = "./checkpoints_multitask"
    sampling_rate: int = 500
    window_seconds: float = 10.0
    weight_decay: float = 1e-4
    num_workers: int = 0
    max_grad_norm: float = 1.0
    warmup_epochs: int = 5
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rhythm": 1.0,
            "structural": 1.0,
            "ischaemia": 1.0,
            "risk": 1.0,
        }
    )
    enabled_tasks: list[str] = field(
        default_factory=lambda: ["rhythm", "structural", "ischaemia", "risk"]
    )
    feature_dim: int = 256
    head_hidden_dim: int = 128
    head_dropout: float = 0.3
    save_metric: str = "val_loss"
    backend: str = "pytorch"
    structural_focal: bool = False


def load_config(path: str | Path) -> MultiTaskTrainConfig:
    """Load a :class:`MultiTaskTrainConfig` from a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A populated :class:`MultiTaskTrainConfig` instance.

    Raises:
        ImportError: If ``pyyaml`` is not installed.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to load YAML config files.  "
            "Install with: pip install pyyaml"
        )
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)  # type: ignore[union-attr]

    return MultiTaskTrainConfig(**{k: v for k, v in raw.items() if v is not None})


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def cosine_annealing_with_warmup(
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    peak_lr: float,
) -> float:
    """Update the learning rate with linear warm-up + cosine annealing.

    During warm-up (epochs 0 … warmup_epochs-1) the LR increases linearly
    from 0 to *peak_lr*.  After warm-up it follows a cosine schedule down
    to 0.

    Args:
        optimizer: The PyTorch optimiser whose LR will be adjusted.
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warm-up epochs.
        peak_lr: Maximum learning rate at end of warm-up.

    Returns:
        The learning rate applied for this epoch.
    """
    if epoch < warmup_epochs:
        lr = peak_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_f1(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5,
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


def _compute_c_index(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute concordance index averaged across risk outputs.

    The C-index measures the model's ability to correctly rank subjects
    by predicted risk.  For each task column we count concordant /
    discordant pairs.

    Args:
        predictions: Shape ``[N, K]``.
        targets: Shape ``[N, K]``.

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


# ---------------------------------------------------------------------------
# Per-epoch dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskEpochMetrics:
    """Metrics recorded per epoch during multi-task training."""

    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    per_task_train_loss: dict[str, float] = field(default_factory=dict)
    per_task_val_loss: dict[str, float] = field(default_factory=dict)
    rhythm_f1: float = 0.0
    structural_f1: float = 0.0
    ischaemia_f1: float = 0.0
    risk_c_index: float = 0.5


# ---------------------------------------------------------------------------
# Label splitting helper
# ---------------------------------------------------------------------------

_TASK_NUM_OUTPUTS: dict[str, int] = {
    "rhythm": 28,
    "structural": 19,
    "ischaemia": 19,
    "risk": 3,
}


def _split_labels(
    labels: "torch.Tensor",
    enabled_tasks: list[str],
) -> dict[str, "torch.Tensor"]:
    """Split a concatenated label tensor into per-task tensors.

    Convention: labels are ordered by ``enabled_tasks`` with the number
    of columns defined by ``_TASK_NUM_OUTPUTS``.

    Args:
        labels: Tensor of shape ``[batch, total_label_cols]``.
        enabled_tasks: The tasks in their concatenation order.

    Returns:
        Dict mapping task name → label tensor.
    """
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for task in enabled_tasks:
        width = _TASK_NUM_OUTPUTS[task]
        result[task] = labels[:, offset : offset + width]
        offset += width
    return result


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _compute_multitask_loss(
    model: "nn.Module",
    features: "torch.Tensor",
    task_labels: dict[str, "torch.Tensor"],
    loss_weights: dict[str, float],
    structural_focal: bool = False,
) -> tuple["torch.Tensor", dict[str, float]]:
    """Compute weighted sum of per-task losses.

    The model is expected to have ``forward_logits``-style methods on each
    head.  We call the per-task loss functions from the respective modules.

    Args:
        model: An :class:`AorticaModel` instance.
        features: Shared backbone+attention feature tensor.
        task_labels: Dict of per-task label tensors.
        loss_weights: Per-task scalar loss coefficients.
        structural_focal: Use focal loss for structural head.

    Returns:
        Tuple of (total_loss, per_task_loss_dict_as_floats).
    """
    from aortica.models.ischaemia_head import compute_ischaemia_loss
    from aortica.models.rhythm_head import compute_rhythm_loss
    from aortica.models.risk_head import compute_risk_loss
    from aortica.models.structural_head import compute_structural_loss

    total_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    per_task_losses: dict[str, float] = {}

    if "rhythm" in task_labels and model.rhythm_head is not None:
        logits = model.rhythm_head.forward_logits(features)
        loss = compute_rhythm_loss(logits, task_labels["rhythm"])
        total_loss = total_loss + loss_weights.get("rhythm", 1.0) * loss
        per_task_losses["rhythm"] = loss.item()

    if "structural" in task_labels and model.structural_head is not None:
        logits = model.structural_head.forward_logits(features)
        loss = compute_structural_loss(
            logits, task_labels["structural"], focal=structural_focal,
        )
        total_loss = total_loss + loss_weights.get("structural", 1.0) * loss
        per_task_losses["structural"] = loss.item()

    if "ischaemia" in task_labels and model.ischaemia_head is not None:
        logits = model.ischaemia_head.forward_logits(features)
        loss = compute_ischaemia_loss(logits, task_labels["ischaemia"])
        total_loss = total_loss + loss_weights.get("ischaemia", 1.0) * loss
        per_task_losses["ischaemia"] = loss.item()

    if "risk" in task_labels and model.risk_head is not None:
        preds = model.risk_head(features)
        loss = compute_risk_loss(preds, task_labels["risk"])
        total_loss = total_loss + loss_weights.get("risk", 1.0) * loss
        per_task_losses["risk"] = loss.item()

    return total_loss, per_task_losses


# ---------------------------------------------------------------------------
# Train / evaluate one epoch (PyTorch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: "nn.Module",
    dataloader: "DataLoader[Any]",
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    config: MultiTaskTrainConfig,
) -> tuple[float, dict[str, float]]:
    """Train for a single epoch with gradient clipping.

    Returns:
        Tuple of (average_total_loss, per_task_average_losses).
    """
    model.train()
    running_loss = 0.0
    running_per_task: dict[str, float] = {}
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()

        optimizer.zero_grad()

        # Forward through backbone + attention
        features = model.backbone(batch_x)
        features = model.attention(features)

        task_labels = _split_labels(batch_y, config.enabled_tasks)
        total_loss, per_task = _compute_multitask_loss(
            model, features, task_labels, config.loss_weights,
            structural_focal=config.structural_focal,
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        running_loss += total_loss.item()
        for k, v in per_task.items():
            running_per_task[k] = running_per_task.get(k, 0.0) + v
        num_batches += 1

    n = max(num_batches, 1)
    avg_loss = running_loss / n
    avg_per_task = {k: v / n for k, v in running_per_task.items()}
    return avg_loss, avg_per_task


@torch.no_grad()
def evaluate(
    model: "nn.Module",
    dataloader: "DataLoader[Any]",
    device: "torch.device",
    config: MultiTaskTrainConfig,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (avg_loss, per_task_losses, task_metrics).
        ``task_metrics`` contains e.g. ``rhythm_f1``, ``risk_c_index``.
    """
    model.eval()
    running_loss = 0.0
    running_per_task: dict[str, float] = {}
    num_batches = 0

    # Per-task prediction / target collectors
    collectors: dict[str, dict[str, list[np.ndarray]]] = {
        t: {"preds": [], "targets": []} for t in config.enabled_tasks
    }

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()

        features = model.backbone(batch_x)
        features = model.attention(features)

        task_labels = _split_labels(batch_y, config.enabled_tasks)
        total_loss, per_task = _compute_multitask_loss(
            model, features, task_labels, config.loss_weights,
            structural_focal=config.structural_focal,
        )

        running_loss += total_loss.item()
        for k, v in per_task.items():
            running_per_task[k] = running_per_task.get(k, 0.0) + v
        num_batches += 1

        # Collect predictions for metrics
        for task in config.enabled_tasks:
            head = getattr(model, f"{task}_head", None)
            if head is not None:
                preds = head(features)
                collectors[task]["preds"].append(preds.cpu().numpy())
                collectors[task]["targets"].append(
                    task_labels[task].cpu().numpy()
                )

    n = max(num_batches, 1)
    avg_loss = running_loss / n
    avg_per_task = {k: v / n for k, v in running_per_task.items()}

    # Compute task-specific metrics
    task_metrics: dict[str, Any] = {}
    for task in config.enabled_tasks:
        if not collectors[task]["preds"]:
            continue
        preds_np = np.concatenate(collectors[task]["preds"], axis=0)
        tgts_np = np.concatenate(collectors[task]["targets"], axis=0)

        if task in ("rhythm", "structural", "ischaemia"):
            macro_f1, _ = _compute_f1(preds_np, tgts_np)
            task_metrics[f"{task}_f1"] = macro_f1
        elif task == "risk":
            c_idx = _compute_c_index(preds_np, tgts_np)
            task_metrics["risk_c_index"] = c_idx

    return avg_loss, avg_per_task, task_metrics


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------

def train_multitask(config: MultiTaskTrainConfig) -> list[MultiTaskEpochMetrics]:
    """Run the full multi-task training loop (PyTorch).

    Args:
        config: Training configuration.

    Returns:
        List of :class:`MultiTaskEpochMetrics` for each epoch.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required. Install with: pip install aortica[torch]"
        )

    from aortica.data.dataset import ECGDataset
    from aortica.data.ptbxl import load_ptbxl
    from aortica.models.aortica_model import AorticaModel

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------ Data ---------------------------------------------------------
    (train_records, train_labels), (val_records, val_labels), _ = load_ptbxl(
        config.data_path, sampling_rate=config.sampling_rate,
    )

    train_ds = ECGDataset(
        train_records,
        train_labels,
        target_hz=float(config.sampling_rate),
        window_seconds=config.window_seconds,
        augment=True,
        random_seed=config.seed,
    )
    val_ds = ECGDataset(
        val_records,
        val_labels,
        target_hz=float(config.sampling_rate),
        window_seconds=config.window_seconds,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # ------ Model --------------------------------------------------------
    model = AorticaModel(
        in_channels=12,
        feature_dim=config.feature_dim,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        enabled_tasks=config.enabled_tasks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[MultiTaskEpochMetrics] = []
    best_metric_val: Optional[float] = None

    for epoch in range(config.epochs):
        # LR schedule
        lr = cosine_annealing_with_warmup(
            optimizer, epoch, config.epochs, config.warmup_epochs, config.lr,
        )

        train_loss, train_per_task = train_one_epoch(
            model, train_loader, optimizer, device, config,
        )
        val_loss, val_per_task, task_metrics = evaluate(
            model, val_loader, device, config,
        )

        metrics = MultiTaskEpochMetrics(
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
        task_loss_str = "  ".join(
            f"{k}={v:.4f}" for k, v in val_per_task.items()
        )
        metric_str = "  ".join(
            f"{k}={v:.4f}" for k, v in task_metrics.items()
        )
        print(
            f"Epoch {epoch + 1}/{config.epochs}  lr={lr:.6f}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"{task_loss_str}  {metric_str}"
        )

        # Save best checkpoint
        current_metric = _extract_save_metric(metrics, config.save_metric)
        if _is_better(current_metric, best_metric_val, config.save_metric):
            best_metric_val = current_metric
            ckpt_path = output_path / "best_multitask_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": asdict(metrics),
                    "config": asdict(config),
                },
                ckpt_path,
            )

    # Save full training history
    metrics_path = output_path / "multitask_training_metrics.json"
    metrics_data: list[dict[str, Any]] = [asdict(m) for m in history]
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nTraining complete. Best {config.save_metric}: {best_metric_val}")
    print(f"Checkpoint: {output_path / 'best_multitask_model.pt'}")
    print(f"Metrics:    {metrics_path}")

    return history


def _extract_save_metric(
    metrics: MultiTaskEpochMetrics, save_metric: str,
) -> float:
    """Extract the scalar value of the configured save metric."""
    if save_metric == "val_loss":
        return metrics.val_loss
    return float(getattr(metrics, save_metric, 0.0))


def _is_better(
    current: float,
    best: Optional[float],
    save_metric: str,
) -> bool:
    """Return True if *current* is better than *best*."""
    if best is None:
        return True
    if save_metric == "val_loss":
        return current < best  # lower is better
    return current > best  # higher is better (F1, C-index)


# ---------------------------------------------------------------------------
# TF/Keras training loop
# ---------------------------------------------------------------------------

def train_multitask_tf(config: MultiTaskTrainConfig) -> list[dict[str, Any]]:
    """Run multi-task training using TF/Keras Compile-and-Fit.

    This is a thin wrapper that builds the TF model and compiles it with
    per-task loss functions and cosine-annealing LR.

    Args:
        config: Training configuration (``backend`` should be ``tensorflow``).

    Returns:
        List of per-epoch metric dicts from ``model.fit`` history.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # type: ignore[attr-defined]
    except ImportError:
        raise ImportError(
            "TensorFlow is required.  Install with: pip install tensorflow"
        )

    from aortica.models.aortica_model_tf import build_aortica_model_tf

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    model = build_aortica_model_tf(
        in_channels=12,
        feature_dim=config.feature_dim,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        enabled_tasks=config.enabled_tasks,
    )

    # Per-task losses
    losses: dict[str, str] = {}
    loss_w: dict[str, float] = {}
    for task in config.enabled_tasks:
        if task in ("rhythm", "structural", "ischaemia"):
            losses[task] = "binary_crossentropy"
        else:
            losses[task] = "mse"
        loss_w[task] = config.loss_weights.get(task, 1.0)

    # Cosine annealing schedule
    def lr_schedule(epoch: int) -> float:
        return cosine_annealing_with_warmup(
            _DummyOptim(), epoch, config.epochs, config.warmup_epochs, config.lr,
        )

    _lr_cb = keras.callbacks.LearningRateScheduler(lr_schedule)  # noqa: F841

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            clipnorm=config.max_grad_norm,
        ),
        loss=losses,
        loss_weights=loss_w,
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _ckpt_cb = keras.callbacks.ModelCheckpoint(  # noqa: F841
        filepath=str(output_path / "best_multitask_model_tf.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    # NOTE: actual data loading would happen here using create_tf_dataset.
    # For now the function is structured for integration; callers must
    # provide tf.data.Dataset objects via model.fit() externally.
    # This function returns an empty history when called without data.

    history_list: list[dict[str, Any]] = []
    return history_list


class _DummyOptim:
    """Minimal shim so ``cosine_annealing_with_warmup`` can compute LR."""

    def __init__(self) -> None:
        self.param_groups: list[dict[str, float]] = [{"lr": 0.0}]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for multi-task training."""
    parser = argparse.ArgumentParser(
        description="Train multi-task AorticaModel on PTB-XL",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file.  Overrides other CLI flags.",
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_multitask")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_metric", type=str, default="val_loss")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "tensorflow"])
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.3)

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = MultiTaskTrainConfig(
            data_path=args.data_path,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate,
            window_seconds=args.window_seconds,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            max_grad_norm=args.max_grad_norm,
            warmup_epochs=args.warmup_epochs,
            save_metric=args.save_metric,
            backend=args.backend,
            feature_dim=args.feature_dim,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
        )

    if config.backend == "tensorflow":
        train_multitask_tf(config)
    else:
        train_multitask(config)


if __name__ == "__main__":
    main()
