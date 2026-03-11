"""Tests for the baseline ResNet1D model and training utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Gracefully skip all tests if torch is not installed
torch = pytest.importorskip("torch")
nn = torch.nn

from aortica.models.resnet1d import ResidualBlock1D, ResNet1D  # noqa: E402
from aortica.models.train_baseline import (  # noqa: E402
    EpochMetrics,
    TrainConfig,
    _compute_f1,
    evaluate,
    set_seed,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def default_model() -> ResNet1D:
    """Standard 12-lead, 3-class ResNet1D."""
    return ResNet1D(in_channels=12, num_classes=3)


@pytest.fixture
def small_model() -> ResNet1D:
    """2-lead, 5-class model for fast tests."""
    return ResNet1D(in_channels=2, num_classes=5)


@pytest.fixture
def random_batch_12lead() -> torch.Tensor:
    """Random batch: [4, 12, 5000] (4 samples, 12 leads, 10s @ 500 Hz)."""
    torch.manual_seed(0)
    return torch.randn(4, 12, 5000)


@pytest.fixture
def random_batch_2lead() -> torch.Tensor:
    """Random batch: [4, 2, 1000]."""
    torch.manual_seed(0)
    return torch.randn(4, 2, 1000)


# ---------------------------------------------------------------------------
# ResidualBlock1D tests
# ---------------------------------------------------------------------------

class TestResidualBlock1D:
    def test_no_downsample(self) -> None:
        block = ResidualBlock1D(64, 64, kernel_size=7)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_with_downsample(self) -> None:
        downsample = nn.Sequential(
            nn.Conv1d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm1d(128),
        )
        block = ResidualBlock1D(64, 128, kernel_size=7, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 128, 50)


# ---------------------------------------------------------------------------
# ResNet1D tests
# ---------------------------------------------------------------------------

class TestResNet1D:
    def test_output_shape_default(
        self, default_model: ResNet1D, random_batch_12lead: torch.Tensor
    ) -> None:
        out = default_model(random_batch_12lead)
        assert out.shape == (4, 3)

    def test_output_shape_custom(
        self, small_model: ResNet1D, random_batch_2lead: torch.Tensor
    ) -> None:
        out = small_model(random_batch_2lead)
        assert out.shape == (4, 5)

    def test_output_range_sigmoid(
        self, default_model: ResNet1D, random_batch_12lead: torch.Tensor
    ) -> None:
        out = default_model(random_batch_12lead)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_different_sequence_lengths(self, default_model: ResNet1D) -> None:
        """Adaptive pooling should handle different signal lengths."""
        for length in [250, 1000, 2500, 5000]:
            x = torch.randn(2, 12, length)
            out = default_model(x)
            assert out.shape == (2, 3), f"Failed for length={length}"

    def test_single_sample(self, default_model: ResNet1D) -> None:
        x = torch.randn(1, 12, 5000)
        out = default_model(x)
        assert out.shape == (1, 3)

    def test_gradient_flow(self, default_model: ResNet1D) -> None:
        x = torch.randn(2, 12, 5000, requires_grad=True)
        out = default_model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_parameter_count(self, default_model: ResNet1D) -> None:
        """Sanity check that the model has a reasonable number of params."""
        num_params = sum(p.numel() for p in default_model.parameters())
        # ResNet-18 1D should have >500k parameters
        assert num_params > 500_000

    def test_reproducibility(self) -> None:
        """Same seed produces same output."""
        torch.manual_seed(42)
        m1 = ResNet1D(in_channels=12, num_classes=3)
        torch.manual_seed(42)
        m2 = ResNet1D(in_channels=12, num_classes=3)

        x = torch.randn(1, 12, 5000)
        out1 = m1(x)
        out2 = m2(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# F1 computation tests
# ---------------------------------------------------------------------------

class TestComputeF1:
    def test_perfect_predictions(self) -> None:
        targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        preds = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype=np.float32)
        macro_f1, per_class = _compute_f1(preds, targets)
        assert macro_f1 == pytest.approx(1.0)
        assert all(f == pytest.approx(1.0) for f in per_class)

    def test_all_wrong(self) -> None:
        targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        preds = np.array([[0.1, 0.9, 0.2], [0.9, 0.1, 0.8]], dtype=np.float32)
        macro_f1, _ = _compute_f1(preds, targets)
        assert macro_f1 == pytest.approx(0.0)

    def test_partial(self) -> None:
        targets = np.array([[1, 0], [1, 1]], dtype=np.float32)
        preds = np.array([[0.8, 0.1], [0.8, 0.1]], dtype=np.float32)
        macro_f1, per_class = _compute_f1(preds, targets)
        # Class 0: TP=2, FP=0, FN=0 → F1=1.0
        assert per_class[0] == pytest.approx(1.0)
        # Class 1: TP=0, FP=0, FN=1 → F1=0.0
        assert per_class[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Training utility tests
# ---------------------------------------------------------------------------

class TestTrainUtilities:
    def test_set_seed_reproducibility(self) -> None:
        set_seed(123)
        t1 = torch.randn(10)
        set_seed(123)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_train_one_epoch(self, default_model: ResNet1D, device: torch.device) -> None:
        """Verify train_one_epoch runs without error and returns a float loss."""
        model = default_model.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 12, 5000),
            torch.randint(0, 2, (8, 3)).float(),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate(self, default_model: ResNet1D, device: torch.device) -> None:
        """Verify evaluate returns loss, macro_f1, and per_class_f1."""
        model = default_model.to(device)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 12, 5000),
            torch.randint(0, 2, (8, 3)).float(),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        val_loss, macro_f1, per_class = evaluate(model, loader, criterion, device)
        assert isinstance(val_loss, float)
        assert 0.0 <= macro_f1 <= 1.0
        assert len(per_class) == 3

    def test_epoch_metrics_dataclass(self) -> None:
        m = EpochMetrics(
            epoch=1, train_loss=0.5, val_loss=0.4, val_macro_f1=0.8,
            val_per_class_f1=[0.7, 0.8, 0.9],
        )
        assert m.epoch == 1
        assert m.val_macro_f1 == 0.8

    def test_train_config_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.epochs == 30
        assert cfg.lr == 1e-3
        assert cfg.seed == 42

    def test_model_checkpoint_save_load(
        self, default_model: ResNet1D, tmp_path: Path, device: torch.device
    ) -> None:
        """Verify checkpoint save / load round-trip produces identical outputs."""
        model = default_model.to(device)
        x = torch.randn(1, 12, 5000).to(device)
        out_before = model(x)

        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        model2 = ResNet1D(in_channels=12, num_classes=3).to(device)
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])
        out_after = model2(x)

        assert torch.allclose(out_before, out_after)

    def test_training_metrics_json(self, tmp_path: Path) -> None:
        """Verify training metrics JSON serialisation."""
        metrics = [
            EpochMetrics(1, 0.7, 0.6, 0.75, [0.7, 0.8, 0.75]),
            EpochMetrics(2, 0.5, 0.45, 0.82, [0.8, 0.85, 0.81]),
        ]
        path = tmp_path / "metrics.json"
        data: list[dict[str, Any]] = [asdict(m) for m in metrics]
        with open(path, "w") as f:
            json.dump(data, f)

        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[1]["val_macro_f1"] == 0.82
