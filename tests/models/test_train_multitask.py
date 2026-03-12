"""Tests for the multi-task training pipeline (US-022).

Covers:
- MultiTaskTrainConfig defaults and YAML loading
- Cosine annealing with warm-up LR schedule
- Label splitting helper
- F1 and C-index metric helpers
- Multi-task loss computation
- Single-epoch training loop (forward, loss, gradient clipping)
- Evaluation loop with per-task metrics
- Best checkpoint saving logic
- TF/Keras backend structure
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMultiTaskTrainConfig:
    """Tests for ``MultiTaskTrainConfig``."""

    def test_default_values(self) -> None:
        from aortica.models.train_multitask import MultiTaskTrainConfig

        cfg = MultiTaskTrainConfig()
        assert cfg.epochs == 30
        assert cfg.lr == 1e-3
        assert cfg.max_grad_norm == 1.0
        assert cfg.warmup_epochs == 5
        assert cfg.save_metric == "val_loss"
        assert cfg.backend == "pytorch"
        assert len(cfg.enabled_tasks) == 4
        assert len(cfg.loss_weights) == 4
        for v in cfg.loss_weights.values():
            assert v == 1.0

    def test_custom_loss_weights(self) -> None:
        from aortica.models.train_multitask import MultiTaskTrainConfig

        cfg = MultiTaskTrainConfig(
            loss_weights={"rhythm": 2.0, "structural": 1.0, "ischaemia": 0.5, "risk": 0.5},
        )
        assert cfg.loss_weights["rhythm"] == 2.0
        assert cfg.loss_weights["risk"] == 0.5

    def test_load_config_yaml(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        from aortica.models.train_multitask import load_config

        yaml_content = {
            "epochs": 10,
            "lr": 0.0005,
            "batch_size": 32,
            "max_grad_norm": 2.0,
            "warmup_epochs": 3,
            "save_metric": "rhythm_f1",
            "enabled_tasks": ["rhythm", "structural"],
            "loss_weights": {"rhythm": 2.0, "structural": 1.0},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = load_config(cfg_path)
        assert cfg.epochs == 10
        assert cfg.lr == 0.0005
        assert cfg.max_grad_norm == 2.0
        assert cfg.save_metric == "rhythm_f1"
        assert cfg.enabled_tasks == ["rhythm", "structural"]
        assert cfg.loss_weights["rhythm"] == 2.0

    def test_asdict_roundtrip(self) -> None:
        from dataclasses import asdict

        from aortica.models.train_multitask import MultiTaskTrainConfig

        cfg = MultiTaskTrainConfig(epochs=5)
        d = asdict(cfg)
        assert d["epochs"] == 5
        assert isinstance(d["loss_weights"], dict)


# ---------------------------------------------------------------------------
# LR schedule tests
# ---------------------------------------------------------------------------

class TestCosineAnnealingWithWarmup:
    """Tests for ``cosine_annealing_with_warmup``."""

    def _make_optimizer(self, lr: float = 0.1) -> "torch.optim.Optimizer":
        model = nn.Linear(2, 2)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_warmup_linear(self) -> None:
        from aortica.models.train_multitask import cosine_annealing_with_warmup

        opt = self._make_optimizer()
        peak = 0.01

        lr0 = cosine_annealing_with_warmup(opt, 0, 20, 5, peak)
        assert lr0 == pytest.approx(peak * 1 / 5)

        lr4 = cosine_annealing_with_warmup(opt, 4, 20, 5, peak)
        assert lr4 == pytest.approx(peak * 5 / 5)

    def test_cosine_at_end(self) -> None:
        from aortica.models.train_multitask import cosine_annealing_with_warmup

        opt = self._make_optimizer()
        peak = 0.01
        total = 20
        warmup = 5

        lr_last = cosine_annealing_with_warmup(opt, total - 1, total, warmup, peak)
        # Should be close to 0
        assert lr_last < peak * 0.05

    def test_peak_at_warmup_end(self) -> None:
        from aortica.models.train_multitask import cosine_annealing_with_warmup

        opt = self._make_optimizer()
        peak = 0.01

        lr = cosine_annealing_with_warmup(opt, 4, 20, 5, peak)
        assert lr == pytest.approx(peak, abs=1e-8)

    def test_cosine_midpoint(self) -> None:
        from aortica.models.train_multitask import cosine_annealing_with_warmup

        opt = self._make_optimizer()
        peak = 0.01
        total = 20
        warmup = 0  # no warmup for simplicity

        mid = total // 2
        lr = cosine_annealing_with_warmup(opt, mid, total, warmup, peak)
        expected = peak * 0.5 * (1 + math.cos(math.pi * mid / total))
        assert lr == pytest.approx(expected, abs=1e-8)

    def test_sets_optimizer_lr(self) -> None:
        from aortica.models.train_multitask import cosine_annealing_with_warmup

        opt = self._make_optimizer(lr=0.5)
        lr = cosine_annealing_with_warmup(opt, 0, 10, 5, 0.01)
        assert opt.param_groups[0]["lr"] == lr


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for ``_compute_f1`` and ``_compute_c_index``."""

    def test_f1_perfect(self) -> None:
        from aortica.models.train_multitask import _compute_f1

        preds = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        tgts = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        f1, per = _compute_f1(preds, tgts)
        assert f1 == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in per)

    def test_f1_random(self) -> None:
        from aortica.models.train_multitask import _compute_f1

        preds = np.array([[0.9, 0.1], [0.1, 0.9]])
        tgts = np.array([[0.0, 1.0], [1.0, 0.0]])
        f1, _ = _compute_f1(preds, tgts)
        assert f1 == pytest.approx(0.0)

    def test_c_index_perfect_ordering(self) -> None:
        from aortica.models.train_multitask import _compute_c_index

        preds = np.array([[0.1], [0.5], [0.9]])
        tgts = np.array([[0.0], [0.5], [1.0]])
        c = _compute_c_index(preds, tgts)
        assert c == pytest.approx(1.0)

    def test_c_index_reversed_ordering(self) -> None:
        from aortica.models.train_multitask import _compute_c_index

        preds = np.array([[0.9], [0.5], [0.1]])
        tgts = np.array([[0.0], [0.5], [1.0]])
        c = _compute_c_index(preds, tgts)
        assert c == pytest.approx(0.0)

    def test_c_index_single_sample(self) -> None:
        from aortica.models.train_multitask import _compute_c_index

        preds = np.array([[0.5]])
        tgts = np.array([[0.5]])
        c = _compute_c_index(preds, tgts)
        assert c == pytest.approx(0.5)

    def test_c_index_multi_task(self) -> None:
        from aortica.models.train_multitask import _compute_c_index

        preds = np.array([[0.1, 0.9], [0.9, 0.1]])
        tgts = np.array([[0.0, 1.0], [1.0, 0.0]])
        c = _compute_c_index(preds, tgts)
        assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Label splitting
# ---------------------------------------------------------------------------

class TestSplitLabels:
    """Tests for ``_split_labels``."""

    def test_all_tasks(self) -> None:
        from aortica.models.train_multitask import _split_labels

        total_cols = 22 + 15 + 10 + 3  # 50
        labels = torch.randn(4, total_cols)
        tasks = ["rhythm", "structural", "ischaemia", "risk"]
        split = _split_labels(labels, tasks)

        assert split["rhythm"].shape == (4, 22)
        assert split["structural"].shape == (4, 15)
        assert split["ischaemia"].shape == (4, 10)
        assert split["risk"].shape == (4, 3)

    def test_subset_tasks(self) -> None:
        from aortica.models.train_multitask import _split_labels

        total_cols = 22 + 3  # rhythm + risk only
        labels = torch.randn(4, total_cols)
        tasks = ["rhythm", "risk"]
        split = _split_labels(labels, tasks)

        assert split["rhythm"].shape == (4, 22)
        assert split["risk"].shape == (4, 3)
        assert "structural" not in split


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class TestMultitaskLoss:
    """Tests for ``_compute_multitask_loss``."""

    def _make_model(self, enabled_tasks: list[str] | None = None) -> nn.Module:
        from aortica.models.aortica_model import AorticaModel

        return AorticaModel(
            in_channels=12,
            feature_dim=252,  # divisible by 12
            enabled_tasks=enabled_tasks,
        )

    def test_all_tasks_loss(self) -> None:
        from aortica.models.train_multitask import _compute_multitask_loss

        model = self._make_model()
        features = torch.randn(4, 252)

        task_labels = {
            "rhythm": torch.zeros(4, 22),
            "structural": torch.zeros(4, 15),
            "ischaemia": torch.zeros(4, 10),
            "risk": torch.rand(4, 3),
        }
        loss_weights = {"rhythm": 1.0, "structural": 1.0, "ischaemia": 1.0, "risk": 1.0}

        total_loss, per_task = _compute_multitask_loss(
            model, features, task_labels, loss_weights,
        )
        assert total_loss.ndim == 0  # scalar
        assert total_loss.item() > 0
        assert len(per_task) == 4
        for v in per_task.values():
            assert v > 0

    def test_weighted_loss(self) -> None:
        from aortica.models.train_multitask import _compute_multitask_loss

        model = self._make_model()
        features = torch.randn(4, 252)
        task_labels = {
            "rhythm": torch.zeros(4, 22),
            "structural": torch.zeros(4, 15),
            "ischaemia": torch.zeros(4, 10),
            "risk": torch.rand(4, 3),
        }

        # Equal weights
        loss_eq, _ = _compute_multitask_loss(
            model, features, task_labels,
            {"rhythm": 1.0, "structural": 1.0, "ischaemia": 1.0, "risk": 1.0},
        )

        # Double rhythm weight
        loss_2x, _ = _compute_multitask_loss(
            model, features, task_labels,
            {"rhythm": 2.0, "structural": 1.0, "ischaemia": 1.0, "risk": 1.0},
        )

        # With higher weight on one head, the total should be different
        assert loss_eq.item() != loss_2x.item()

    def test_subset_tasks_loss(self) -> None:
        from aortica.models.train_multitask import _compute_multitask_loss

        model = self._make_model(enabled_tasks=["rhythm"])
        features = torch.randn(4, 252)
        task_labels = {"rhythm": torch.zeros(4, 22)}

        total_loss, per_task = _compute_multitask_loss(
            model, features, task_labels, {"rhythm": 1.0},
        )
        assert "rhythm" in per_task
        assert "structural" not in per_task

    def test_gradient_flows(self) -> None:
        from aortica.models.train_multitask import _compute_multitask_loss

        model = self._make_model()
        features = torch.randn(4, 252, requires_grad=True)
        task_labels = {
            "rhythm": torch.zeros(4, 22),
            "structural": torch.zeros(4, 15),
            "ischaemia": torch.zeros(4, 10),
            "risk": torch.rand(4, 3),
        }

        total_loss, _ = _compute_multitask_loss(
            model, features, task_labels,
            {"rhythm": 1.0, "structural": 1.0, "ischaemia": 1.0, "risk": 1.0},
        )
        total_loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Train / evaluate one epoch
# ---------------------------------------------------------------------------

class TestTrainEvalOneEpoch:
    """Tests for ``train_one_epoch`` and ``evaluate``."""

    def _make_tiny_loader(
        self, batch_size: int = 4, n_samples: int = 8,
    ) -> "torch.utils.data.DataLoader[Any]":
        """Create a tiny DataLoader with synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset

        total_labels = 22 + 15 + 10 + 3  # 50
        x = torch.randn(n_samples, 12, 500)
        y = torch.rand(n_samples, total_labels)
        # Binarize classification columns
        y[:, :47] = (y[:, :47] > 0.5).float()
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def _make_config(self) -> Any:
        from aortica.models.train_multitask import MultiTaskTrainConfig

        return MultiTaskTrainConfig(
            epochs=1,
            lr=1e-3,
            batch_size=4,
            seed=42,
            max_grad_norm=1.0,
        )

    def test_train_one_epoch(self) -> None:
        from aortica.models.aortica_model import AorticaModel
        from aortica.models.train_multitask import train_one_epoch

        model = AorticaModel(in_channels=12, feature_dim=252)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")
        loader = self._make_tiny_loader()
        config = self._make_config()

        avg_loss, per_task = train_one_epoch(model, loader, optimizer, device, config)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert len(per_task) == 4

    def test_evaluate(self) -> None:
        from aortica.models.aortica_model import AorticaModel
        from aortica.models.train_multitask import evaluate

        model = AorticaModel(in_channels=12, feature_dim=252)
        device = torch.device("cpu")
        loader = self._make_tiny_loader()
        config = self._make_config()

        avg_loss, per_task, metrics = evaluate(model, loader, device, config)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert "rhythm_f1" in metrics
        assert "risk_c_index" in metrics
        assert 0.0 <= metrics["rhythm_f1"] <= 1.0
        assert 0.0 <= metrics["risk_c_index"] <= 1.0

    def test_gradient_clipping(self) -> None:
        from aortica.models.aortica_model import AorticaModel
        from aortica.models.train_multitask import MultiTaskTrainConfig, train_one_epoch

        model = AorticaModel(in_channels=12, feature_dim=252)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        device = torch.device("cpu")
        loader = self._make_tiny_loader()
        config = MultiTaskTrainConfig(max_grad_norm=0.01, lr=1.0)

        # Train and then check gradients were clipped
        train_one_epoch(model, loader, optimizer, device, config)
        # If clipping works, the model should still produce finite outputs
        x = torch.randn(2, 12, 500)
        out = model(x)
        assert torch.isfinite(out.rhythm).all()


# ---------------------------------------------------------------------------
# Checkpoint save logic
# ---------------------------------------------------------------------------

class TestCheckpointLogic:
    """Tests for ``_extract_save_metric`` and ``_is_better``."""

    def test_extract_val_loss(self) -> None:
        from aortica.models.train_multitask import (
            MultiTaskEpochMetrics,
            _extract_save_metric,
        )

        m = MultiTaskEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.3)
        assert _extract_save_metric(m, "val_loss") == 0.3

    def test_extract_rhythm_f1(self) -> None:
        from aortica.models.train_multitask import (
            MultiTaskEpochMetrics,
            _extract_save_metric,
        )

        m = MultiTaskEpochMetrics(
            epoch=1, lr=0.001, train_loss=0.5, val_loss=0.3, rhythm_f1=0.85,
        )
        assert _extract_save_metric(m, "rhythm_f1") == 0.85

    def test_is_better_lower_val_loss(self) -> None:
        from aortica.models.train_multitask import _is_better

        assert _is_better(0.2, 0.3, "val_loss") is True
        assert _is_better(0.4, 0.3, "val_loss") is False

    def test_is_better_higher_f1(self) -> None:
        from aortica.models.train_multitask import _is_better

        assert _is_better(0.9, 0.8, "rhythm_f1") is True
        assert _is_better(0.7, 0.8, "rhythm_f1") is False

    def test_is_better_none_best(self) -> None:
        from aortica.models.train_multitask import _is_better

        assert _is_better(0.5, None, "val_loss") is True
        assert _is_better(0.5, None, "rhythm_f1") is True


# ---------------------------------------------------------------------------
# MultiTaskEpochMetrics
# ---------------------------------------------------------------------------

class TestMultiTaskEpochMetrics:
    """Tests for the epoch metrics dataclass."""

    def test_defaults(self) -> None:
        from aortica.models.train_multitask import MultiTaskEpochMetrics

        m = MultiTaskEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.3)
        assert m.rhythm_f1 == 0.0
        assert m.risk_c_index == 0.5
        assert m.per_task_train_loss == {}

    def test_serialisation(self) -> None:
        from dataclasses import asdict

        from aortica.models.train_multitask import MultiTaskEpochMetrics

        m = MultiTaskEpochMetrics(
            epoch=1, lr=0.001, train_loss=0.5, val_loss=0.3,
            per_task_train_loss={"rhythm": 0.4},
            rhythm_f1=0.8,
        )
        d = asdict(m)
        j = json.dumps(d)
        assert "rhythm_f1" in j
        assert "0.8" in j


# ---------------------------------------------------------------------------
# TF/Keras backend (structural only)
# ---------------------------------------------------------------------------

class TestTFBackendStructure:
    """Tests that the TF training wrapper exists and has correct signature."""

    def test_train_multitask_tf_exists(self) -> None:
        from aortica.models.train_multitask import train_multitask_tf

        assert callable(train_multitask_tf)

    def test_dummy_optim(self) -> None:
        from aortica.models.train_multitask import _DummyOptim

        opt = _DummyOptim()
        assert len(opt.param_groups) == 1
        assert "lr" in opt.param_groups[0]
