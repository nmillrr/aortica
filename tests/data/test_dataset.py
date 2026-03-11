"""Tests for ECG Dataset and DataLoader utilities."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.data.dataset import HAS_TORCH, ECGDataset, create_tf_dataset
from aortica.io.ecg_record import ECGRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(
    n: int = 5,
    leads: int = 12,
    samples: int = 500,
    sample_rate: float = 100.0,
    seed: int = 0,
) -> list[ECGRecord]:
    """Create deterministic dummy ECGRecord objects."""
    rng = np.random.default_rng(seed)
    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ][:leads]
    records = []
    for _ in range(n):
        signals = rng.standard_normal((leads, samples))
        records.append(
            ECGRecord(
                signals=signals,
                sample_rate=sample_rate,
                lead_names=lead_names,
            )
        )
    return records


# ---------------------------------------------------------------------------
# ECGDataset (PyTorch) tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="requires PyTorch")
class TestECGDataset:
    def test_length(self) -> None:
        records = _make_records(5)
        labels = np.zeros((5, 3), dtype=np.float32)
        ds = ECGDataset(records, labels, window_seconds=2.5)
        assert len(ds) == 5

    def test_output_shapes(self) -> None:
        import torch

        records = _make_records(3)
        labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        ds = ECGDataset(records, labels, window_seconds=2.5)

        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (12, 250)  # 100 Hz * 2.5s = 250 samples
        assert y.shape == (3,)

    def test_output_dtype_float32(self) -> None:
        import torch

        records = _make_records(2)
        labels = np.array([0, 1])  # int labels should be converted to float32
        ds = ECGDataset(records, labels, window_seconds=2.5)

        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_augmentation_changes_output(self) -> None:
        records = _make_records(1)
        labels = np.array([[1.0, 0.0]])
        ds_no_aug = ECGDataset(records, labels, window_seconds=2.5, augment=False)
        ds_aug = ECGDataset(
            records, labels, window_seconds=2.5,
            augment=True, random_seed=42,
        )

        x_clean, _ = ds_no_aug[0]
        x_aug, _ = ds_aug[0]
        # Augmented output should differ from clean
        assert not np.allclose(x_clean.numpy(), x_aug.numpy())

    def test_augmentation_reproducibility(self) -> None:
        records = _make_records(3)
        labels = np.ones((3, 2), dtype=np.float32)

        ds1 = ECGDataset(records, labels, augment=True, random_seed=42, window_seconds=2.5)
        ds2 = ECGDataset(records, labels, augment=True, random_seed=42, window_seconds=2.5)

        x1, _ = ds1[0]
        x2, _ = ds2[0]
        np.testing.assert_array_equal(x1.numpy(), x2.numpy())

    def test_window_seconds_zero_no_padding(self) -> None:
        records = _make_records(1, samples=500)
        labels = np.array([[1.0]])
        ds = ECGDataset(records, labels, window_seconds=0.0)

        x, _ = ds[0]
        # Should keep original length
        assert x.shape == (12, 500)

    def test_target_hz_resampling(self) -> None:
        records = _make_records(2, samples=500, sample_rate=100.0)
        labels = np.array([[1.0], [0.0]])
        ds = ECGDataset(records, labels, target_hz=200.0, window_seconds=2.5)

        x, _ = ds[0]
        # 200 Hz * 2.5s = 500 samples
        assert x.shape == (12, 500)

    def test_mismatched_lengths_raises(self) -> None:
        records = _make_records(3)
        labels = np.zeros((5, 2))
        with pytest.raises(ValueError, match="must match"):
            ECGDataset(records, labels)

    def test_single_record(self) -> None:
        records = _make_records(1)
        labels = np.array([[0.0]])
        ds = ECGDataset(records, labels, window_seconds=2.5)

        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (12, 250)

    def test_scalar_labels(self) -> None:
        import torch

        records = _make_records(3)
        labels = [0.0, 1.0, 0.0]  # plain list
        ds = ECGDataset(records, labels, window_seconds=2.5)

        _, y = ds[1]
        assert y.dtype == torch.float32
        assert y.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# create_tf_dataset tests
# ---------------------------------------------------------------------------

class TestCreateTFDataset:
    @pytest.fixture(autouse=True)
    def _skip_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    def test_basic_batch_shapes(self) -> None:
        import tensorflow as tf

        records = _make_records(5)
        labels = np.array([0, 1, 0, 1, 0], dtype=np.float32)

        ds = create_tf_dataset(
            records, labels, window_seconds=2.5, batch_size=2,
        )

        for batch_x, batch_y in ds.take(1):
            assert isinstance(batch_x, tf.Tensor)
            assert batch_x.shape == (2, 12, 250)
            assert batch_y.shape == (2,)

    def test_multi_label_shapes(self) -> None:
        records = _make_records(4)
        labels = np.zeros((4, 3), dtype=np.float32)

        ds = create_tf_dataset(records, labels, window_seconds=2.5, batch_size=2)

        for batch_x, batch_y in ds.take(1):
            assert batch_x.shape == (2, 12, 250)
            assert batch_y.shape == (2, 3)

    def test_augmentation_applied(self) -> None:
        records = _make_records(2)
        labels = np.array([[1.0, 0.0], [0.0, 1.0]])

        ds_clean = create_tf_dataset(
            records, labels, augment=False, window_seconds=2.5, batch_size=2,
        )
        ds_aug = create_tf_dataset(
            records, labels, augment=True, random_seed=42,
            window_seconds=2.5, batch_size=2,
        )

        for (xc, _), (xa, _) in zip(ds_clean.take(1), ds_aug.take(1)):
            assert not np.allclose(xc.numpy(), xa.numpy())

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            create_tf_dataset([], [], batch_size=2)

    def test_mismatched_lengths_raises(self) -> None:
        records = _make_records(3)
        labels = np.zeros(5)
        with pytest.raises(ValueError, match="must match"):
            create_tf_dataset(records, labels)
