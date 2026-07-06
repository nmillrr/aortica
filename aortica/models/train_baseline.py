"""Baseline rhythm classification training script for PTB-XL.

Trains a 1D ResNet-18 model on the PTB-XL dataset for rhythm superclass
multi-label classification.  Supports configurable hyperparameters and
fully reproducible runs with fixed random seeds.

Usage (from project root)::

    python -m aortica.models.train_baseline \\
        --data_path /path/to/ptbxl \\
        --epochs 30 \\
        --lr 1e-3 \\
        --batch_size 64 \\
        --seed 42 \\
        --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aortica.models.resnet1d import ResNet1D


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""

    data_path: str = ""
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    output_dir: str = "./checkpoints"
    sampling_rate: int = 500
    window_seconds: float = 10.0
    num_classes: int = 3
    weight_decay: float = 1e-4
    num_workers: int = 0


@dataclass
class EpochMetrics:
    """Metrics recorded per epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    val_per_class_f1: list[float] = field(default_factory=list)


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


def _compute_f1(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> tuple[float, list[float]]:
    """Compute macro-F1 and per-class F1 from sigmoid outputs.

    Args:
        predictions: Model predictions (after sigmoid), shape ``[N, C]``.
        targets: Binary targets, shape ``[N, C]``.
        threshold: Decision threshold for binarising predictions.

    Returns:
        Tuple of (macro_f1, per_class_f1_list).
    """
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for a single epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss: float = running_loss / max(num_batches, 1)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (avg_loss, macro_f1, per_class_f1).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item()
            num_batches += 1
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    avg_loss = running_loss / max(num_batches, 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    macro_f1, per_class = _compute_f1(preds, targets)

    return avg_loss, macro_f1, per_class


def train(config: TrainConfig) -> list[EpochMetrics]:
    """Run the full training loop.

    Args:
        config: Training configuration.

    Returns:
        List of ``EpochMetrics`` for each epoch.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required. Install with: pip install aortica[torch]"
        )

    from aortica.data.dataset import ECGDataset
    from aortica.data.ptbxl import load_ptbxl

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    (train_records, train_labels), (val_records, val_labels), _ = load_ptbxl(
        config.data_path, sampling_rate=config.sampling_rate
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

    # Model
    model = ResNet1D(
        in_channels=12,
        num_classes=config.num_classes,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[EpochMetrics] = []
    best_f1 = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_per_class = evaluate(
            model, val_loader, criterion, device
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_macro_f1=val_f1,
            val_per_class_f1=val_per_class,
        )
        history.append(metrics)

        print(
            f"Epoch {epoch}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}"
        )

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "config": asdict(config),
                },
                ckpt_path,
            )

    # Save training metrics
    metrics_path = output_path / "training_metrics.json"
    metrics_data: list[dict[str, Any]] = [asdict(m) for m in history]
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {output_path / 'best_model.pt'}")
    print(f"Metrics saved to: {metrics_path}")

    return history


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train baseline rhythm classifier on PTB-XL"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to PTB-XL dataset directory"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--window_seconds", type=float, default=10.0)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    train(config)


if __name__ == "__main__":
    main()
