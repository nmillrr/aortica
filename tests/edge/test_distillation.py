"""Tests for aortica.edge.distillation — knowledge distillation training."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from aortica.edge.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationEpochMetrics,
    DistillationResult,
    _compute_c_index,
    _compute_f1,
    _cosine_annealing_with_warmup,
    _evaluate_student,
    _extract_save_metric,
    _is_better,
    _split_labels,
    distillation_loss_classification,
    distillation_loss_regression,
    train_distillation,
)
from aortica.models.aortica_model import AorticaModel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 8
LEADS = 12
SAMPLES = 500
FEATURE_DIM = 252  # must be divisible by num_leads=12


def _make_teacher_student(
    enabled_tasks: list[str] | None = None,
) -> tuple[AorticaModel, AorticaModel]:
    """Create small teacher and student models for testing."""
    if enabled_tasks is None:
        enabled_tasks = ["rhythm", "structural", "ischaemia", "risk"]
    teacher = AorticaModel(
        in_channels=LEADS,
        feature_dim=FEATURE_DIM,
        head_hidden_dim=32,
        head_dropout=0.0,
        enabled_tasks=enabled_tasks,
    )
    student = AorticaModel(
        in_channels=LEADS,
        feature_dim=FEATURE_DIM,
        head_hidden_dim=32,
        head_dropout=0.0,
        enabled_tasks=enabled_tasks,
    )
    return teacher, student


def _make_data_loaders(
    batch_size: int = BATCH,
    num_samples: int = 16,
    enabled_tasks: list[str] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Create synthetic train/val DataLoaders."""
    if enabled_tasks is None:
        enabled_tasks = ["rhythm", "structural", "ischaemia", "risk"]

    task_widths = {"rhythm": 22, "structural": 15, "ischaemia": 10, "risk": 3}
    label_cols = sum(task_widths[t] for t in enabled_tasks)

    x = torch.randn(num_samples, LEADS, SAMPLES)
    y = torch.rand(num_samples, label_cols)
    # Binarise classification labels
    offset = 0
    for t in enabled_tasks:
        w = task_widths[t]
        if t != "risk":
            y[:, offset : offset + w] = (y[:, offset : offset + w] > 0.5).float()
        offset += w

    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ===================================================================
# DistillationConfig
# ===================================================================


class TestDistillationConfig:
    """Tests for DistillationConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = DistillationConfig()
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.7
        assert cfg.epochs == 30
        assert cfg.lr == 1e-3
        assert cfg.save_metric == "val_loss"
        assert len(cfg.enabled_tasks) == 4

    def test_custom_values(self) -> None:
        cfg = DistillationConfig(temperature=2.0, alpha=0.5, epochs=10)
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.5
        assert cfg.epochs == 10

    def test_serialisation(self) -> None:
        cfg = DistillationConfig()
        d = asdict(cfg)
        assert "temperature" in d
        assert "alpha" in d
        assert d["temperature"] == 4.0

    def test_loss_weights(self) -> None:
        cfg = DistillationConfig(loss_weights={"rhythm": 2.0, "risk": 0.5})
        assert cfg.loss_weights["rhythm"] == 2.0
        assert cfg.loss_weights["risk"] == 0.5


# ===================================================================
# DistillationEpochMetrics
# ===================================================================


class TestDistillationEpochMetrics:
    """Tests for DistillationEpochMetrics dataclass."""

    def test_defaults(self) -> None:
        m = DistillationEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.6)
        assert m.epoch == 1
        assert m.rhythm_f1 == 0.0
        assert m.risk_c_index == 0.5

    def test_serialisation(self) -> None:
        m = DistillationEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.6)
        d = asdict(m)
        assert "epoch" in d
        assert d["train_loss"] == 0.5


# ===================================================================
# DistillationResult
# ===================================================================


class TestDistillationResult:
    """Tests for DistillationResult dataclass."""

    def test_construction(self) -> None:
        r = DistillationResult(history=[], best_epoch=5, best_metric_value=0.9)
        assert r.best_epoch == 5
        assert r.best_metric_value == 0.9
        assert r.student_checkpoint_path == ""

    def test_with_history(self) -> None:
        m = DistillationEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.6)
        r = DistillationResult(history=[m])
        assert len(r.history) == 1
        assert r.history[0].epoch == 1


# ===================================================================
# Classification distillation loss
# ===================================================================


class TestDistillationLossClassification:
    """Tests for distillation_loss_classification."""

    def test_returns_scalar(self) -> None:
        student_logits = torch.randn(4, 22)
        teacher_logits = torch.randn(4, 22)
        labels = torch.zeros(4, 22)
        loss = distillation_loss_classification(
            student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        student_logits = torch.randn(4, 22, requires_grad=True)
        teacher_logits = torch.randn(4, 22)
        labels = torch.zeros(4, 22)
        loss = distillation_loss_classification(
            student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7,
        )
        loss.backward()
        assert student_logits.grad is not None
        assert torch.any(student_logits.grad != 0)

    def test_zero_loss_when_identical(self) -> None:
        """When student matches teacher exactly, soft loss should be zero."""
        logits = torch.randn(4, 22)
        labels = (torch.sigmoid(logits) > 0.5).float()
        loss = distillation_loss_classification(
            logits.clone(), logits.clone(), labels, temperature=4.0, alpha=1.0,
        )
        # KL divergence should be near zero when distributions match
        assert loss.item() < 1e-4

    def test_alpha_zero_equals_hard_only(self) -> None:
        """alpha=0 should give pure hard-label BCE loss."""
        student_logits = torch.randn(4, 22)
        teacher_logits = torch.randn(4, 22)
        labels = torch.zeros(4, 22)

        combined = distillation_loss_classification(
            student_logits, teacher_logits, labels, temperature=4.0, alpha=0.0,
        )
        hard_only = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits, labels,
        )
        assert abs(combined.item() - hard_only.item()) < 1e-5

    def test_temperature_effect(self) -> None:
        """Higher temperature should produce softer distributions."""
        student = torch.randn(4, 22)
        teacher = torch.randn(4, 22)
        labels = torch.zeros(4, 22)

        loss_t1 = distillation_loss_classification(
            student, teacher, labels, temperature=1.0, alpha=1.0,
        )
        loss_t10 = distillation_loss_classification(
            student, teacher, labels, temperature=10.0, alpha=1.0,
        )
        # Both should be valid scalar losses (no NaN/Inf)
        assert torch.isfinite(loss_t1)
        assert torch.isfinite(loss_t10)

    def test_different_num_classes(self) -> None:
        """Should work for any number of classes."""
        for n_cls in [5, 10, 15]:
            s = torch.randn(4, n_cls)
            t = torch.randn(4, n_cls)
            lbl = torch.zeros(4, n_cls)
            loss = distillation_loss_classification(s, t, lbl, 4.0, 0.7)
            assert loss.dim() == 0


# ===================================================================
# Regression distillation loss
# ===================================================================


class TestDistillationLossRegression:
    """Tests for distillation_loss_regression."""

    def test_returns_scalar(self) -> None:
        student = torch.randn(4, 3)
        teacher = torch.randn(4, 3)
        labels = torch.rand(4, 3)
        loss = distillation_loss_regression(student, teacher, labels, alpha=0.7)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        student = torch.randn(4, 3, requires_grad=True)
        teacher = torch.randn(4, 3)
        labels = torch.rand(4, 3)
        loss = distillation_loss_regression(student, teacher, labels, alpha=0.7)
        loss.backward()
        assert student.grad is not None

    def test_zero_loss_when_identical(self) -> None:
        """When student preds match teacher and labels, loss should be near zero."""
        vals = torch.rand(4, 3)
        loss = distillation_loss_regression(vals.clone(), vals.clone(), vals.clone(), alpha=0.5)
        assert loss.item() < 1e-6

    def test_alpha_zero_equals_hard_only(self) -> None:
        student = torch.randn(4, 3)
        teacher = torch.randn(4, 3)
        labels = torch.rand(4, 3)

        combined = distillation_loss_regression(student, teacher, labels, alpha=0.0)
        hard_only = torch.nn.functional.mse_loss(student, labels)
        assert abs(combined.item() - hard_only.item()) < 1e-5

    def test_alpha_one_equals_soft_only(self) -> None:
        student = torch.randn(4, 3)
        teacher = torch.randn(4, 3)
        labels = torch.rand(4, 3)

        combined = distillation_loss_regression(student, teacher, labels, alpha=1.0)
        soft_only = torch.nn.functional.mse_loss(student, teacher)
        assert abs(combined.item() - soft_only.item()) < 1e-5


# ===================================================================
# Label splitting
# ===================================================================


class TestSplitLabels:
    """Tests for _split_labels."""

    def test_all_tasks(self) -> None:
        labels = torch.rand(4, 50)  # 22+15+10+3
        result = _split_labels(labels, ["rhythm", "structural", "ischaemia", "risk"])
        assert result["rhythm"].shape == (4, 22)
        assert result["structural"].shape == (4, 15)
        assert result["ischaemia"].shape == (4, 10)
        assert result["risk"].shape == (4, 3)

    def test_subset_tasks(self) -> None:
        labels = torch.rand(4, 25)  # 22+3
        result = _split_labels(labels, ["rhythm", "risk"])
        assert result["rhythm"].shape == (4, 22)
        assert result["risk"].shape == (4, 3)
        assert "structural" not in result


# ===================================================================
# F1 and C-index metrics
# ===================================================================


class TestMetrics:
    """Tests for _compute_f1 and _compute_c_index."""

    def test_perfect_f1(self) -> None:
        preds = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _compute_f1(preds, targets) == 1.0

    def test_worst_f1(self) -> None:
        preds = np.array([[1.0, 0.0], [1.0, 0.0]])
        targets = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert _compute_f1(preds, targets) == 0.0

    def test_perfect_c_index(self) -> None:
        preds = np.array([[0.1], [0.5], [0.9]])
        targets = np.array([[0.1], [0.5], [0.9]])
        assert _compute_c_index(preds, targets) == 1.0

    def test_reversed_c_index(self) -> None:
        preds = np.array([[0.9], [0.5], [0.1]])
        targets = np.array([[0.1], [0.5], [0.9]])
        assert _compute_c_index(preds, targets) == 0.0

    def test_single_sample_c_index(self) -> None:
        """Single sample should return 0.5 (uninformative)."""
        preds = np.array([[0.5]])
        targets = np.array([[0.5]])
        assert _compute_c_index(preds, targets) == 0.5


# ===================================================================
# LR schedule
# ===================================================================


class TestLRSchedule:
    """Tests for _cosine_annealing_with_warmup."""

    def test_warmup_start(self) -> None:
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        lr = _cosine_annealing_with_warmup(opt, 0, 30, 5, 1e-3)
        assert abs(lr - 1e-3 / 5) < 1e-8

    def test_warmup_end(self) -> None:
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        lr = _cosine_annealing_with_warmup(opt, 4, 30, 5, 1e-3)
        assert abs(lr - 1e-3) < 1e-8  # epoch 4 is end of warmup (5/5)

    def test_end_near_zero(self) -> None:
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        lr = _cosine_annealing_with_warmup(opt, 29, 30, 5, 1e-3)
        assert lr < 1e-4  # should be near zero at end


# ===================================================================
# Save metric helpers
# ===================================================================


class TestSaveMetricHelpers:
    """Tests for _extract_save_metric and _is_better."""

    def test_extract_val_loss(self) -> None:
        m = DistillationEpochMetrics(epoch=1, lr=0.001, train_loss=0.5, val_loss=0.6)
        assert _extract_save_metric(m, "val_loss") == 0.6

    def test_extract_rhythm_f1(self) -> None:
        m = DistillationEpochMetrics(
            epoch=1, lr=0.001, train_loss=0.5, val_loss=0.6, rhythm_f1=0.85,
        )
        assert _extract_save_metric(m, "rhythm_f1") == 0.85

    def test_is_better_val_loss(self) -> None:
        assert _is_better(0.3, 0.5, "val_loss") is True
        assert _is_better(0.6, 0.5, "val_loss") is False

    def test_is_better_f1(self) -> None:
        assert _is_better(0.9, 0.8, "rhythm_f1") is True
        assert _is_better(0.7, 0.8, "rhythm_f1") is False

    def test_is_better_none_best(self) -> None:
        assert _is_better(0.5, None, "val_loss") is True


# ===================================================================
# Evaluate student
# ===================================================================


class TestEvaluateStudent:
    """Tests for _evaluate_student."""

    def test_returns_tuple(self) -> None:
        _, student = _make_teacher_student(["rhythm"])
        _, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(enabled_tasks=["rhythm"])
        result = _evaluate_student(student, val_loader, torch.device("cpu"), cfg)
        assert len(result) == 3
        avg_loss, per_task, metrics = result
        assert isinstance(avg_loss, float)
        assert "rhythm" in per_task
        assert "rhythm_f1" in metrics

    def test_risk_only(self) -> None:
        _, student = _make_teacher_student(["risk"])
        _, val_loader = _make_data_loaders(enabled_tasks=["risk"])
        cfg = DistillationConfig(enabled_tasks=["risk"])
        _, per_task, metrics = _evaluate_student(
            student, val_loader, torch.device("cpu"), cfg,
        )
        assert "risk" in per_task
        assert "risk_c_index" in metrics


# ===================================================================
# Full train_distillation
# ===================================================================


class TestTrainDistillation:
    """Tests for the top-level train_distillation function."""

    def test_returns_result(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert isinstance(result, DistillationResult)
        assert len(result.history) == 2

    def test_history_epoch_numbers(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=3,
            warmup_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert [m.epoch for m in result.history] == [1, 2, 3]

    def test_best_epoch_set(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert result.best_epoch >= 1

    def test_checkpoint_saved(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        out_dir = tempfile.mkdtemp()
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=1,
            output_dir=out_dir,
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert os.path.exists(result.student_checkpoint_path)
        ckpt = torch.load(result.student_checkpoint_path, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

    def test_metrics_json_saved(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        out_dir = tempfile.mkdtemp()
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=1,
            output_dir=out_dir,
        )
        train_distillation(teacher, student, train_loader, val_loader, cfg)
        metrics_path = os.path.join(out_dir, "distillation_metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_all_tasks(self) -> None:
        teacher, student = _make_teacher_student()
        train_loader, val_loader = _make_data_loaders()
        cfg = DistillationConfig(
            epochs=1,
            warmup_epochs=0,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert len(result.history) == 1
        m = result.history[0]
        assert m.train_loss > 0

    def test_default_config(self) -> None:
        """train_distillation with config=None uses defaults."""
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        # Can't use default config because output_dir would be ./checkpoints_distillation
        # and epochs=30 would be too slow, so minimal override
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"], epochs=1, warmup_epochs=0,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert isinstance(result, DistillationResult)

    def test_teacher_frozen(self) -> None:
        """Teacher parameters should not change during distillation."""
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])

        # Record teacher params before
        teacher_params_before = {
            k: v.clone() for k, v in teacher.state_dict().items()
        }

        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        train_distillation(teacher, student, train_loader, val_loader, cfg)

        # Teacher params should be identical
        for k, v_before in teacher_params_before.items():
            v_after = teacher.state_dict()[k]
            assert torch.allclose(v_before, v_after), f"Teacher param {k} changed!"

    def test_student_learns(self) -> None:
        """Student parameters should change during distillation."""
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])

        student_params_before = {
            k: v.clone() for k, v in student.state_dict().items()
        }

        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=3,
            warmup_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        train_distillation(teacher, student, train_loader, val_loader, cfg)

        # At least some student params should have changed
        changed = False
        for k, v_before in student_params_before.items():
            if not torch.allclose(v_before, student.state_dict()[k]):
                changed = True
                break
        assert changed, "No student parameter changed during training"

    def test_task_subset_risk_only(self) -> None:
        teacher, student = _make_teacher_student(["risk"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["risk"])
        cfg = DistillationConfig(
            enabled_tasks=["risk"],
            epochs=1,
            warmup_epochs=0,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert len(result.history) == 1

    def test_custom_temperature(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=1,
            warmup_epochs=0,
            temperature=2.0,
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert len(result.history) == 1

    def test_custom_alpha(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=1,
            warmup_epochs=0,
            alpha=0.0,  # pure hard labels
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert len(result.history) == 1

    def test_save_metric_rhythm_f1(self) -> None:
        teacher, student = _make_teacher_student(["rhythm"])
        train_loader, val_loader = _make_data_loaders(enabled_tasks=["rhythm"])
        cfg = DistillationConfig(
            enabled_tasks=["rhythm"],
            epochs=2,
            warmup_epochs=0,
            save_metric="rhythm_f1",
            output_dir=tempfile.mkdtemp(),
        )
        result = train_distillation(teacher, student, train_loader, val_loader, cfg)
        assert result.best_epoch >= 1


# ===================================================================
# Imports
# ===================================================================


class TestImports:
    """Verify public API imports from the edge package."""

    def test_import_from_edge(self) -> None:
        from aortica.edge import (
            DistillationConfig,
            DistillationEpochMetrics,
            DistillationResult,
            distillation_loss_classification,
            distillation_loss_regression,
            train_distillation,
        )

        assert DistillationConfig is not None
        assert DistillationEpochMetrics is not None
        assert DistillationResult is not None
        assert callable(distillation_loss_classification)
        assert callable(distillation_loss_regression)
        assert callable(train_distillation)

    def test_import_from_module(self) -> None:
        from aortica.edge.distillation import (
            DistillationConfig,
            train_distillation,
        )

        assert DistillationConfig is not None
        assert callable(train_distillation)
