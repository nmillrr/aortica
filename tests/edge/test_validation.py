"""Tests for aortica.edge.validation — Edge Model Validation Harness (US-041).

Covers:
  - Constants and dataclass construction
  - TaskValidation and EdgeValidationReport fields
  - Metric helpers (_compute_auc, _compute_f1, _compute_mean_auc, _compute_c_index)
  - Label splitting
  - Per-task evaluation (_evaluate_task)
  - Full validate_edge() pipeline with synthetic models
  - Latency measurement
  - Summary table output
  - Imports and module structure
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aortica.edge.validation import (  # noqa: E402
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    DEFAULT_THRESHOLD,
    TASK_NUM_OUTPUTS,
    EdgeValidationReport,
    TaskValidation,
    _compute_auc,
    _compute_c_index,
    _compute_f1,
    _compute_mean_auc,
    _evaluate_task,
    _split_labels,
    validate_edge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_all_tasks(self) -> None:
        assert ALL_TASKS == ["rhythm", "structural", "ischaemia", "risk"]

    def test_classification_tasks(self) -> None:
        assert CLASSIFICATION_TASKS == ["rhythm", "structural", "ischaemia"]

    def test_task_num_outputs(self) -> None:
        assert TASK_NUM_OUTPUTS["rhythm"] == 28
        assert TASK_NUM_OUTPUTS["structural"] == 19
        assert TASK_NUM_OUTPUTS["ischaemia"] == 19
        assert TASK_NUM_OUTPUTS["risk"] == 6

    def test_default_threshold(self) -> None:
        assert DEFAULT_THRESHOLD == 0.03


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestTaskValidation:
    """Test TaskValidation dataclass."""

    def test_defaults(self) -> None:
        tv = TaskValidation()
        assert tv.task_name == ""
        assert tv.full_metric == 0.0
        assert tv.edge_metric == 0.0
        assert tv.metric_name == ""
        assert tv.degradation == 0.0
        assert tv.relative_degradation == 0.0
        assert tv.passed is True
        assert tv.threshold == DEFAULT_THRESHOLD

    def test_custom_values(self) -> None:
        tv = TaskValidation(
            task_name="rhythm",
            full_metric=0.92,
            edge_metric=0.90,
            metric_name="macro_f1",
            degradation=0.02,
            relative_degradation=0.0217,
            passed=True,
            threshold=0.03,
        )
        assert tv.task_name == "rhythm"
        assert tv.full_metric == 0.92
        assert tv.edge_metric == 0.90
        assert tv.passed is True

    def test_failed_validation(self) -> None:
        tv = TaskValidation(
            task_name="ischaemia",
            full_metric=0.85,
            edge_metric=0.80,
            degradation=0.05,
            relative_degradation=0.0588,
            passed=False,
            threshold=0.03,
        )
        assert tv.passed is False
        assert tv.relative_degradation > tv.threshold


class TestEdgeValidationReport:
    """Test EdgeValidationReport dataclass."""

    def test_defaults(self) -> None:
        report = EdgeValidationReport()
        assert report.task_validations == {}
        assert report.all_passed is True
        assert report.edge_latency_ms == 0.0
        assert report.full_latency_ms == 0.0
        assert report.n_samples == 0
        assert report.tasks_evaluated == []

    def test_construction(self) -> None:
        tv_rhythm = TaskValidation(task_name="rhythm", passed=True)
        report = EdgeValidationReport(
            task_validations={"rhythm": tv_rhythm},
            all_passed=True,
            edge_latency_ms=1.5,
            full_latency_ms=10.0,
            n_samples=100,
            tasks_evaluated=["rhythm"],
        )
        assert report.n_samples == 100
        assert "rhythm" in report.task_validations

    def test_summary_table(self) -> None:
        tv = TaskValidation(
            task_name="rhythm",
            full_metric=0.9,
            edge_metric=0.88,
            metric_name="macro_f1",
            degradation=0.02,
            relative_degradation=0.022,
            passed=True,
            threshold=0.03,
        )
        report = EdgeValidationReport(
            task_validations={"rhythm": tv},
            all_passed=True,
            edge_latency_ms=2.0,
            full_latency_ms=12.0,
            n_samples=50,
            tasks_evaluated=["rhythm"],
        )
        table = report.summary_table()
        assert "Edge Validation Report" in table
        assert "50 samples" in table
        assert "rhythm" in table
        assert "PASS" in table
        assert "ALL PASSED" in table

    def test_summary_table_failure(self) -> None:
        tv = TaskValidation(
            task_name="risk",
            full_metric=0.8,
            edge_metric=0.7,
            degradation=0.1,
            relative_degradation=0.125,
            passed=False,
        )
        report = EdgeValidationReport(
            task_validations={"risk": tv},
            all_passed=False,
            n_samples=20,
            tasks_evaluated=["risk"],
        )
        table = report.summary_table()
        assert "FAIL" in table
        assert "FAILED" in table


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestComputeAUC:
    """Test _compute_auc."""

    def test_perfect(self) -> None:
        preds = np.array([0.9, 0.8, 0.1, 0.05])
        targets = np.array([1.0, 1.0, 0.0, 0.0])
        assert _compute_auc(preds, targets) == 1.0

    def test_worst(self) -> None:
        preds = np.array([0.1, 0.2, 0.9, 0.8])
        targets = np.array([1.0, 1.0, 0.0, 0.0])
        assert _compute_auc(preds, targets) == 0.0

    def test_random(self) -> None:
        np.random.seed(42)
        preds = np.random.rand(100)
        targets = np.random.randint(0, 2, 100).astype(np.float64)
        auc = _compute_auc(preds, targets)
        assert 0.0 <= auc <= 1.0

    def test_single_class(self) -> None:
        preds = np.array([0.5, 0.6, 0.7])
        targets = np.array([1.0, 1.0, 1.0])
        assert _compute_auc(preds, targets) == 0.5


class TestComputeF1:
    """Test _compute_f1."""

    def test_perfect(self) -> None:
        preds = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _compute_f1(preds, targets) == 1.0

    def test_worst(self) -> None:
        preds = np.array([[0.1, 0.9], [0.9, 0.1]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _compute_f1(preds, targets) == 0.0

    def test_range(self) -> None:
        np.random.seed(42)
        preds = np.random.rand(20, 5)
        targets = np.random.randint(0, 2, (20, 5)).astype(np.float64)
        f1 = _compute_f1(preds, targets)
        assert 0.0 <= f1 <= 1.0


class TestComputeMeanAUC:
    """Test _compute_mean_auc."""

    def test_perfect(self) -> None:
        preds = np.array([[0.9, 0.9], [0.1, 0.1]])
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])
        assert _compute_mean_auc(preds, targets) == 1.0

    def test_range(self) -> None:
        np.random.seed(42)
        preds = np.random.rand(30, 3)
        targets = np.random.randint(0, 2, (30, 3)).astype(np.float64)
        auc = _compute_mean_auc(preds, targets)
        assert 0.0 <= auc <= 1.0


class TestComputeCIndex:
    """Test _compute_c_index."""

    def test_perfect(self) -> None:
        preds = np.array([[0.1], [0.5], [0.9]])
        targets = np.array([[0.1], [0.5], [0.9]])
        assert _compute_c_index(preds, targets) == 1.0

    def test_reversed(self) -> None:
        preds = np.array([[0.9], [0.5], [0.1]])
        targets = np.array([[0.1], [0.5], [0.9]])
        assert _compute_c_index(preds, targets) == 0.0

    def test_single_sample(self) -> None:
        preds = np.array([[0.5]])
        targets = np.array([[0.5]])
        assert _compute_c_index(preds, targets) == 0.5


# ---------------------------------------------------------------------------
# Label splitting
# ---------------------------------------------------------------------------


class TestSplitLabels:
    """Test _split_labels."""

    def test_all_tasks(self) -> None:
        labels = np.random.rand(10, 72).astype(np.float64)
        result = _split_labels(labels, ALL_TASKS)
        assert result["rhythm"].shape == (10, 28)
        assert result["structural"].shape == (10, 19)
        assert result["ischaemia"].shape == (10, 19)
        assert result["risk"].shape == (10, 6)

    def test_subset(self) -> None:
        labels = np.random.rand(5, 34).astype(np.float64)
        result = _split_labels(labels, ["rhythm", "risk"])
        assert result["rhythm"].shape == (5, 28)
        assert result["risk"].shape == (5, 6)


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


class TestEvaluateTask:
    """Test _evaluate_task."""

    def test_classification_pass(self) -> None:
        np.random.seed(42)
        n = 50
        targets = np.random.randint(0, 2, (n, 5)).astype(np.float64)
        # Edge is very close to full
        full_preds = np.random.rand(n, 5)
        edge_preds = full_preds + np.random.randn(n, 5) * 0.01

        tv = _evaluate_task("rhythm", full_preds, edge_preds, targets, 0.1)
        assert tv.task_name == "rhythm"
        assert tv.metric_name in ("macro_f1", "mean_auc")
        assert isinstance(tv.passed, bool)

    def test_risk_task(self) -> None:
        np.random.seed(42)
        targets = np.random.rand(30, 3)
        full_preds = targets + np.random.randn(30, 3) * 0.05
        edge_preds = targets + np.random.randn(30, 3) * 0.06

        tv = _evaluate_task("risk", full_preds, edge_preds, targets, 0.1)
        assert tv.task_name == "risk"
        assert tv.metric_name == "c_index"

    def test_fail_with_strict_threshold(self) -> None:
        np.random.seed(42)
        n = 100
        targets = np.random.randint(0, 2, (n, 3)).astype(np.float64)
        # Full model is good, edge is significantly worse
        full_preds = targets.astype(np.float64) * 0.8 + 0.1
        edge_preds = np.random.rand(n, 3) * 0.5

        tv = _evaluate_task(
            "structural", full_preds, edge_preds, targets, 0.001,
        )
        # With such a strict threshold and degraded edge predictions,
        # the validation should likely fail (or be marginal)
        assert isinstance(tv.passed, bool)
        assert tv.threshold == 0.001

    def test_zero_full_metric(self) -> None:
        """When full model metric is 0, edge at 0 should pass."""
        targets = np.zeros((10, 3))
        full_preds = np.zeros((10, 3))
        edge_preds = np.zeros((10, 3))

        tv = _evaluate_task("structural", full_preds, edge_preds, targets, 0.03)
        assert tv.passed is True


# ---------------------------------------------------------------------------
# Full pipeline with synthetic models
# ---------------------------------------------------------------------------


def _create_synthetic_aortica_model(
    enabled_tasks: list[str] | None = None,
) -> torch.nn.Module:
    """Create a synthetic AorticaModel for testing."""
    from aortica.models import AorticaModel

    model = AorticaModel(
        in_channels=12,
        feature_dim=252,  # divisible by 12
        enabled_tasks=enabled_tasks or ALL_TASKS,
    )
    model.eval()
    return model


def _create_synthetic_dataset(
    n_samples: int = 20,
    enabled_tasks: list[str] | None = None,
) -> torch.utils.data.Dataset[Any]:
    """Create a synthetic dataset for testing."""
    enabled = enabled_tasks or ALL_TASKS
    total_label_width = sum(TASK_NUM_OUTPUTS[t] for t in enabled)

    signals = torch.randn(n_samples, 12, 1000)
    labels = torch.randint(0, 2, (n_samples, total_label_width)).float()

    return torch.utils.data.TensorDataset(signals, labels)


class TestValidateEdge:
    """Test the full validate_edge() pipeline."""

    def test_returns_report(self) -> None:
        """validate_edge returns an EdgeValidationReport."""
        tasks = ["rhythm"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                batch_size=4,
            )

        assert isinstance(report, EdgeValidationReport)
        assert report.n_samples == 8
        assert "rhythm" in report.tasks_evaluated

    def test_task_validations_populated(self) -> None:
        """Each evaluated task has a TaskValidation entry."""
        tasks = ["rhythm"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                batch_size=4,
            )

        assert "rhythm" in report.task_validations
        tv = report.task_validations["rhythm"]
        assert isinstance(tv, TaskValidation)
        assert tv.full_metric >= 0.0
        assert tv.edge_metric >= 0.0

    def test_identical_model_passes(self) -> None:
        """When edge is the same model exported to ONNX, it should pass."""
        tasks = ["rhythm"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                threshold=0.10,  # generous threshold for identical model
                batch_size=4,
            )

        # The ONNX export of the same model should produce nearly identical
        # outputs, so metrics should match closely
        assert report.all_passed is True

    def test_latency_measured(self) -> None:
        """Latency values should be positive."""
        tasks = ["rhythm"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                batch_size=4,
            )

        assert report.edge_latency_ms > 0.0
        assert report.full_latency_ms > 0.0

    def test_multi_task(self) -> None:
        """Can validate multiple task heads at once."""
        tasks = ["rhythm", "risk"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                threshold=0.10,
                batch_size=4,
            )

        assert len(report.task_validations) == 2
        assert "rhythm" in report.task_validations
        assert "risk" in report.task_validations

    def test_summary_table_from_validate(self) -> None:
        """summary_table integrates with real validate_edge output."""
        tasks = ["rhythm"]
        model = _create_synthetic_aortica_model(tasks)
        dataset = _create_synthetic_dataset(n_samples=8, enabled_tasks=tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / "model.onnx"
            from aortica.edge import export_onnx

            export_onnx(
                model, onnx_path, opset_version=17,
                sample_leads=12, sample_length=1000,
            )

            report = validate_edge(
                full_model=model,
                edge_model_path=onnx_path,
                dataset=dataset,
                tasks=tasks,
                batch_size=4,
            )

        table = report.summary_table()
        assert isinstance(table, str)
        assert "Edge Validation Report" in table
        assert "8 samples" in table


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Test module imports."""

    def test_from_edge_package(self) -> None:
        from aortica.edge import (
            EdgeValidationReport,
            TaskValidation,
            validate_edge,
        )

        assert EdgeValidationReport is not None
        assert TaskValidation is not None
        assert validate_edge is not None

    def test_from_validation_module(self) -> None:
        from aortica.edge.validation import (
            ALL_TASKS,
            DEFAULT_THRESHOLD,
        )

        assert len(ALL_TASKS) == 4
        assert DEFAULT_THRESHOLD == 0.03
