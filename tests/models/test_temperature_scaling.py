"""Tests for Temperature Scaling Calibration Layer (US-023).

Covers:
- TemperatureScaling module: initialisation, forward, temperature access
- CalibratedModel wrapper: forward pass, output shapes
- calibrate() function: optimisation convergence
- expected_calibration_error: synthetic miscalibrated vs calibrated logits
- reliability_diagram_data: binned output structure
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from aortica.models.temperature_scaling import (  # noqa: E402
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    CalibratedModel,
    ReliabilityDiagramData,
    TemperatureScaling,
    calibrate,
    expected_calibration_error,
    reliability_diagram_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify task lists."""

    def test_classification_tasks(self) -> None:
        assert CLASSIFICATION_TASKS == ["rhythm", "structural", "ischaemia"]

    def test_all_tasks(self) -> None:
        assert ALL_TASKS == ["rhythm", "structural", "ischaemia", "risk"]


# ---------------------------------------------------------------------------
# TemperatureScaling module
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    """TemperatureScaling module tests."""

    def test_default_tasks(self) -> None:
        ts = TemperatureScaling()
        assert ts.tasks == ["rhythm", "structural", "ischaemia"]

    def test_custom_tasks(self) -> None:
        ts = TemperatureScaling(tasks=["rhythm"])
        assert ts.tasks == ["rhythm"]
        assert "rhythm" in ts.log_temperatures
        assert "structural" not in ts.log_temperatures

    def test_initial_temperature_is_one(self) -> None:
        ts = TemperatureScaling(tasks=["rhythm"])
        t = ts.get_temperature("rhythm")
        assert torch.allclose(t, torch.ones(1), atol=1e-6)

    def test_forward_preserves_shape(self) -> None:
        ts = TemperatureScaling(tasks=["rhythm", "structural"])
        logits = {
            "rhythm": torch.randn(4, 22),
            "structural": torch.randn(4, 15),
        }
        scaled = ts(logits)
        assert scaled["rhythm"].shape == (4, 22)
        assert scaled["structural"].shape == (4, 15)

    def test_forward_at_default_temperature_is_identity(self) -> None:
        """T=1 should leave logits unchanged."""
        ts = TemperatureScaling(tasks=["rhythm"])
        logits = {"rhythm": torch.randn(4, 22)}
        scaled = ts(logits)
        assert torch.allclose(scaled["rhythm"], logits["rhythm"], atol=1e-6)

    def test_forward_with_high_temperature_softens(self) -> None:
        """Higher temperature → logits closer to zero."""
        ts = TemperatureScaling(tasks=["rhythm"])
        # Set temperature to 10
        with torch.no_grad():
            ts.log_temperatures["rhythm"].fill_(np.log(10.0))

        logits = {"rhythm": torch.randn(4, 22) * 5}
        scaled = ts(logits)
        # Scaled logits should have smaller magnitude
        assert scaled["rhythm"].abs().mean() < logits["rhythm"].abs().mean()

    def test_forward_passes_through_unknown_tasks(self) -> None:
        """Tasks not in TemperatureScaling are passed through unchanged."""
        ts = TemperatureScaling(tasks=["rhythm"])
        logits = {
            "rhythm": torch.randn(4, 22),
            "risk": torch.randn(4, 3),
        }
        scaled = ts(logits)
        assert torch.equal(scaled["risk"], logits["risk"])

    def test_temperature_is_learnable(self) -> None:
        ts = TemperatureScaling(tasks=["rhythm"])
        params = list(ts.parameters())
        assert len(params) == 1
        assert params[0].requires_grad

    def test_multiple_tasks_independent_temperatures(self) -> None:
        ts = TemperatureScaling(tasks=["rhythm", "structural"])
        with torch.no_grad():
            ts.log_temperatures["rhythm"].fill_(np.log(2.0))
            ts.log_temperatures["structural"].fill_(np.log(5.0))

        t_r = ts.get_temperature("rhythm")
        t_s = ts.get_temperature("structural")
        assert not torch.allclose(t_r, t_s)
        assert torch.allclose(t_r, torch.tensor([2.0]), atol=1e-5)
        assert torch.allclose(t_s, torch.tensor([5.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    """ECE computation tests."""

    def test_perfect_calibration(self) -> None:
        """ECE == 0 when confidence matches accuracy perfectly."""
        # Create perfectly calibrated predictions: bin them so each bin
        # has accuracy equal to its confidence
        probs = torch.tensor([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])
        labels = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])  # ~matches confidence
        ece = expected_calibration_error(probs, labels, num_bins=5)
        # Should be very low (not exactly 0 due to finite samples)
        assert ece < 0.15

    def test_worst_calibration(self) -> None:
        """ECE should be high for maximally miscalibrated predictions."""
        # Very confident predictions, all wrong
        probs = torch.ones(100) * 0.95
        labels = torch.zeros(100)
        ece = expected_calibration_error(probs, labels, num_bins=10)
        assert ece > 0.8

    def test_ece_range(self) -> None:
        """ECE must be in [0, 1]."""
        probs = torch.rand(200)
        labels = (torch.rand(200) > 0.5).float()
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_empty_input(self) -> None:
        probs = torch.tensor([])
        labels = torch.tensor([])
        ece = expected_calibration_error(probs, labels)
        assert ece == 0.0

    def test_2d_input(self) -> None:
        """ECE should work with 2D (multi-class) inputs (flattened)."""
        probs = torch.rand(10, 5)
        labels = (torch.rand(10, 5) > 0.5).float()
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    """Tests for reliability diagram binned output."""

    def test_returns_correct_type(self) -> None:
        probs = torch.rand(50)
        labels = (torch.rand(50) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=10)
        assert isinstance(data, ReliabilityDiagramData)

    def test_num_bins(self) -> None:
        probs = torch.rand(50)
        labels = (torch.rand(50) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=5)
        assert data.num_bins == 5
        assert len(data.bin_confidences) == 5
        assert len(data.bin_accuracies) == 5
        assert len(data.bin_counts) == 5

    def test_bin_counts_sum(self) -> None:
        probs = torch.rand(100)
        labels = (torch.rand(100) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=10)
        assert sum(data.bin_counts) == 100

    def test_ece_matches_standalone(self) -> None:
        probs = torch.rand(80)
        labels = (torch.rand(80) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=10)
        ece_standalone = expected_calibration_error(probs, labels, num_bins=10)
        assert abs(data.ece - ece_standalone) < 1e-6

    def test_confidences_in_range(self) -> None:
        probs = torch.rand(100)
        labels = (torch.rand(100) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=10)
        for conf in data.bin_confidences:
            assert 0.0 <= conf <= 1.0

    def test_accuracies_in_range(self) -> None:
        probs = torch.rand(100)
        labels = (torch.rand(100) > 0.5).float()
        data = reliability_diagram_data(probs, labels, num_bins=10)
        for acc in data.bin_accuracies:
            assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# ECE improvement with temperature scaling
# ---------------------------------------------------------------------------


class TestECEImprovement:
    """Verify ECE improvement on synthetic miscalibrated logits."""

    def test_temperature_scaling_reduces_ece(self) -> None:
        """Temperature scaling should reduce ECE on overconfident logits."""
        torch.manual_seed(42)

        # Generate overconfident (miscalibrated) logits:
        # logits are large magnitude → sigmoid gives probabilities near 0 or 1
        n_samples = 500
        n_classes = 10
        logits = torch.randn(n_samples, n_classes) * 5.0  # overconfident

        # Labels are somewhat random — model is overconfident but not accurate
        labels = (torch.rand(n_samples, n_classes) > 0.5).float()

        # ECE before calibration (overconfident)
        probs_before = torch.sigmoid(logits)
        ece_before = expected_calibration_error(probs_before, labels)

        # Optimise temperature using LBFGS
        ts = TemperatureScaling(tasks=["test_task"])
        optimizer = torch.optim.LBFGS(ts.parameters(), lr=0.01, max_iter=50)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = ts({"test_task": logits})
            nll = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled["test_task"], labels
            )
            nll.backward()
            return nll

        optimizer.step(closure)

        # ECE after calibration
        scaled_logits = ts({"test_task": logits})
        probs_after = torch.sigmoid(scaled_logits["test_task"])
        ece_after = expected_calibration_error(probs_after, labels)

        # Temperature should have increased (T > 1 to soften overconfident logits)
        t_val = ts.get_temperature("test_task").item()
        assert t_val > 1.0, f"Expected T > 1 for overconfident logits, got {t_val}"

        # ECE should have improved
        assert ece_after < ece_before, (
            f"ECE should decrease: before={ece_before:.4f}, after={ece_after:.4f}"
        )


# ---------------------------------------------------------------------------
# CalibratedModel wrapper
# ---------------------------------------------------------------------------


class TestCalibratedModel:
    """CalibratedModel wrapper tests."""

    @pytest.fixture()
    def model_and_ts(self) -> tuple:
        from aortica.models.aortica_model import AorticaModel

        model = AorticaModel(
            in_channels=12,
            feature_dim=252,
            num_leads=12,
            enabled_tasks=["rhythm", "structural"],
            head_dropout=0.0,
        )
        model.eval()

        ts = TemperatureScaling(tasks=["rhythm", "structural"])
        return model, ts

    def test_output_type(self, model_and_ts: tuple) -> None:
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(2, 12, 2500)
        output = calibrated(x)
        assert isinstance(output, dict)

    def test_output_keys(self, model_and_ts: tuple) -> None:
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(2, 12, 2500)
        output = calibrated(x)
        assert "rhythm" in output
        assert "structural" in output

    def test_output_shapes(self, model_and_ts: tuple) -> None:
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(2, 12, 2500)
        output = calibrated(x)
        assert output["rhythm"].shape == (2, 22)
        assert output["structural"].shape == (2, 15)

    def test_output_range(self, model_and_ts: tuple) -> None:
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(4, 12, 2500)
        output = calibrated(x)
        for task in ["rhythm", "structural"]:
            assert (output[task] >= 0.0).all()
            assert (output[task] <= 1.0).all()

    def test_no_gradient_in_output(self, model_and_ts: tuple) -> None:
        """CalibratedModel forward runs in no_grad mode."""
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(2, 12, 2500, requires_grad=True)
        output = calibrated(x)
        assert not output["rhythm"].requires_grad

    def test_task_subset(self, model_and_ts: tuple) -> None:
        model, ts = model_and_ts
        calibrated = CalibratedModel(model, ts)
        x = torch.randn(2, 12, 2500)
        output = calibrated(x, tasks=["rhythm"])
        assert "rhythm" in output
        assert "structural" not in output


# ---------------------------------------------------------------------------
# calibrate() function
# ---------------------------------------------------------------------------


class TestCalibrate:
    """Tests for the calibrate() optimisation function."""

    def _make_val_loader(
        self,
        n_samples: int = 32,
        batch_size: int = 8,
        enabled_tasks: list[str] | None = None,
    ) -> torch.utils.data.DataLoader:
        """Create a synthetic validation DataLoader."""
        if enabled_tasks is None:
            enabled_tasks = ["rhythm", "structural"]

        task_sizes = {
            "rhythm": 22,
            "structural": 15,
            "ischaemia": 10,
            "risk": 3,
        }
        label_dim = sum(task_sizes[t] for t in enabled_tasks)

        x = torch.randn(n_samples, 12, 2500)
        labels = (torch.rand(n_samples, label_dim) > 0.5).float()

        dataset = torch.utils.data.TensorDataset(x, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def test_calibrate_returns_temperature_scaling(self) -> None:
        from aortica.models.aortica_model import AorticaModel

        model = AorticaModel(
            in_channels=12,
            feature_dim=252,
            num_leads=12,
            enabled_tasks=["rhythm", "structural"],
            head_dropout=0.0,
        )
        val_loader = self._make_val_loader(
            enabled_tasks=["rhythm", "structural"]
        )

        ts = calibrate(model, val_loader, device=torch.device("cpu"))
        assert isinstance(ts, TemperatureScaling)

    def test_calibrate_has_correct_tasks(self) -> None:
        from aortica.models.aortica_model import AorticaModel

        model = AorticaModel(
            in_channels=12,
            feature_dim=252,
            num_leads=12,
            enabled_tasks=["rhythm", "structural"],
            head_dropout=0.0,
        )
        val_loader = self._make_val_loader(
            enabled_tasks=["rhythm", "structural"]
        )

        ts = calibrate(model, val_loader, device=torch.device("cpu"))
        assert "rhythm" in ts.tasks
        assert "structural" in ts.tasks

    def test_calibrate_temperature_differs_from_default(self) -> None:
        """After calibration, temperature should differ from 1.0."""
        from aortica.models.aortica_model import AorticaModel

        torch.manual_seed(0)
        model = AorticaModel(
            in_channels=12,
            feature_dim=252,
            num_leads=12,
            enabled_tasks=["rhythm"],
            head_dropout=0.0,
        )
        val_loader = self._make_val_loader(
            n_samples=64,
            enabled_tasks=["rhythm"],
        )

        ts = calibrate(
            model, val_loader,
            tasks=["rhythm"],
            device=torch.device("cpu"),
        )
        t = ts.get_temperature("rhythm").item()
        # The randomly-initialised model will be miscalibrated, so the
        # optimiser should move T away from 1.0
        assert abs(t - 1.0) > 0.01, (
            f"Temperature should move from 1.0, got {t:.4f}"
        )


# ---------------------------------------------------------------------------
# Integration: calibrate → CalibratedModel → ECE
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration test."""

    def test_calibration_pipeline(self) -> None:
        from aortica.models.aortica_model import AorticaModel

        torch.manual_seed(123)

        model = AorticaModel(
            in_channels=12,
            feature_dim=252,
            num_leads=12,
            enabled_tasks=["rhythm"],
            head_dropout=0.0,
        )

        # Synthetic val data
        n = 64
        x = torch.randn(n, 12, 2500)
        labels = (torch.rand(n, 22) > 0.5).float()
        val_ds = torch.utils.data.TensorDataset(x, labels)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)

        # Calibrate
        ts = calibrate(
            model, val_loader,
            tasks=["rhythm"],
            device=torch.device("cpu"),
        )

        # Build calibrated model
        calibrated = CalibratedModel(model, ts)
        output = calibrated(x[:4])

        # Smoke checks
        assert "rhythm" in output
        assert output["rhythm"].shape == (4, 22)
        assert (output["rhythm"] >= 0).all()
        assert (output["rhythm"] <= 1).all()
