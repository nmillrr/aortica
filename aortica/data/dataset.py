"""PyTorch and TensorFlow datasets for ECG data.

Both :class:`ECGDataset` (PyTorch) and :func:`create_tf_dataset` (TF)
apply the same augmentation pipeline in this order:

1. Lead dropout
2. Time shift
3. Amplitude scaling
4. Gaussian noise injection

See :mod:`aortica.data.augmentations` for details on each transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from aortica.data.augmentations import (
    apply_amplitude_scale,
    apply_gaussian_noise,
    apply_lead_dropout,
    apply_time_shift,
    pad_or_truncate,
)
from aortica.io.ecg_record import ECGRecord

# Optional dependencies support
try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:

    class Dataset:  # type: ignore[no-redef]
        """Dummy class for when torch is missing."""

        pass

    HAS_TORCH = False


def _apply_augmentations(
    signals: NDArray[np.float64],
    rng: np.random.Generator,
    lead_dropout_prob: float,
    time_shift_samples: int,
    amp_scale_range: tuple[float, float],
    noise_std: float,
) -> NDArray[np.float64]:
    """Apply the full augmentation pipeline to *signals*.

    Factored out so both the PyTorch and TF paths share identical logic.
    """
    signals = apply_lead_dropout(signals, lead_dropout_prob, rng)
    signals = apply_time_shift(signals, time_shift_samples, rng)
    signals = apply_amplitude_scale(
        signals, amp_scale_range[0], amp_scale_range[1], rng,
    )
    signals = apply_gaussian_noise(signals, noise_std, rng)
    return signals


class ECGDataset(Dataset):
    """PyTorch Dataset wrapper for a list of ECGRecords.

    Applies configurable windowing and data augmentations.
    Returns tensors of shape ``[leads, samples]`` and float32 labels.
    """

    def __init__(
        self,
        records: list[ECGRecord],
        labels: NDArray[np.float32] | list[Any],
        target_hz: float | None = None,
        window_seconds: float = 10.0,
        augment: bool = False,
        aug_lead_dropout_prob: float = 0.1,
        aug_noise_std: float = 0.05,
        aug_time_shift_samples: int = 50,
        aug_amp_scale_range: tuple[float, float] = (0.8, 1.2),
        random_seed: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            records: List of ECGRecord objects.
            labels: Corresponding labels (numpy array or list).
            target_hz: If provided, resample all records to this rate
                during initialization.
            window_seconds: Target length in seconds for padding/truncation.
                Set to 0.0 to disable windowing.
            augment: Whether to apply data augmentations.
            aug_lead_dropout_prob: Probability of dropping each lead.
            aug_noise_std: Standard deviation of Gaussian noise.
            aug_time_shift_samples: Maximum shift for time-shift augmentation.
            aug_amp_scale_range: Min and max scaling factor for amplitude.
            random_seed: Random seed for reproducible augmentations.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If *records* and *labels* lengths don't match.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is not installed. `ECGDataset` requires `torch`."
            )

        if len(records) != len(labels):
            raise ValueError(
                f"Number of records ({len(records)}) must match "
                f"number of labels ({len(labels)})"
            )

        self.records = records
        self.labels = np.asarray(labels, dtype=np.float32)
        self.window_seconds = window_seconds

        self.target_hz = target_hz
        if self.target_hz is not None:
            self.records = [r.resample(self.target_hz) for r in self.records]

        self.augment = augment
        self.aug_lead_dropout_prob = aug_lead_dropout_prob
        self.aug_noise_std = aug_noise_std
        self.aug_time_shift_samples = aug_time_shift_samples
        self.aug_amp_scale_range = aug_amp_scale_range

        self._rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        signals = record.signals.copy()

        if self.augment:
            signals = _apply_augmentations(
                signals,
                self._rng,
                self.aug_lead_dropout_prob,
                self.aug_time_shift_samples,
                self.aug_amp_scale_range,
                self.aug_noise_std,
            )

        if self.window_seconds > 0:
            target_samples = int(round(record.sample_rate * self.window_seconds))
            signals = pad_or_truncate(signals, target_samples)

        x = torch.from_numpy(signals).float()
        y = torch.as_tensor(self.labels[idx]).float()

        return x, y


def create_tf_dataset(
    records: list[ECGRecord],
    labels: NDArray[np.float32] | list[Any],
    target_hz: float | None = None,
    window_seconds: float = 10.0,
    augment: bool = False,
    aug_lead_dropout_prob: float = 0.1,
    aug_noise_std: float = 0.05,
    aug_time_shift_samples: int = 50,
    aug_amp_scale_range: tuple[float, float] = (0.8, 1.2),
    random_seed: int | None = None,
    batch_size: int = 32,
) -> Any:
    """Create a ``tf.data.Dataset`` from a list of ECGRecords.

    Applies the same windowing and augmentation pipeline as
    :class:`ECGDataset`.

    Args:
        records: List of ECGRecord objects.
        labels: Corresponding labels (numpy array or list).
        target_hz: If provided, resample all records to this rate.
        window_seconds: Target length in seconds for padding/truncation.
        augment: Whether to apply data augmentations.
        aug_lead_dropout_prob: Probability of dropping each lead.
        aug_noise_std: Standard deviation of Gaussian noise.
        aug_time_shift_samples: Maximum shift for time-shift augmentation.
        aug_amp_scale_range: Min and max scaling factor for amplitude.
        random_seed: Random seed for reproducible augmentations.
        batch_size: Batch size for the returned dataset.

    Returns:
        A ``tf.data.Dataset`` instance that yields ``(signals, labels)``
        where signals has shape ``[batch, leads, samples]``.

    Raises:
        ImportError: If tensorflow is not installed.
        ValueError: If *records* and *labels* lengths don't match,
            or if *records* is empty.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "tensorflow is not installed. Required for create_tf_dataset."
        )

    if len(records) != len(labels):
        raise ValueError("Number of records must match number of labels.")
    if len(records) == 0:
        raise ValueError("records must not be empty.")

    if target_hz is not None:
        records = [r.resample(target_hz) for r in records]

    y_arr = np.asarray(labels, dtype=np.float32)
    lead_count = records[0].num_leads
    sample_count: int | None = (
        int(round(records[0].sample_rate * window_seconds))
        if window_seconds > 0
        else None
    )

    def generator() -> Any:
        rng = np.random.default_rng(random_seed)

        for i in range(len(records)):
            record = records[i]
            signals = record.signals.copy()

            if augment:
                signals = _apply_augmentations(
                    signals,
                    rng,
                    aug_lead_dropout_prob,
                    aug_time_shift_samples,
                    aug_amp_scale_range,
                    aug_noise_std,
                )

            if window_seconds > 0:
                target_samples = int(round(record.sample_rate * window_seconds))
                signals = pad_or_truncate(signals, target_samples)

            yield signals.astype(np.float32), y_arr[i]

    label_shape = y_arr[0].shape if y_arr.ndim > 1 else ()

    sig_spec = tf.TensorSpec(
        shape=(lead_count, sample_count), dtype=tf.float32,
    )
    label_spec = tf.TensorSpec(shape=label_shape, dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(sig_spec, label_spec),
    )

    return dataset.batch(batch_size)
