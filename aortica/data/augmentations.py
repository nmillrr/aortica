"""Data augmentations and windowing utilities for ECG signals.

All augmentation functions accept a ``numpy.random.Generator`` instance for
reproducible randomness.  When composing multiple augmentations, the
recommended application order is:

1. Lead dropout (structural change — must precede signal transforms)
2. Time shift (temporal perturbation)
3. Amplitude scaling
4. Gaussian noise injection (should be last so noise magnitude is
   independent of prior scaling)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ensure_rng(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    """Return *rng* if provided, otherwise create a default Generator."""
    if rng is None:
        return np.random.default_rng()
    return rng


def apply_lead_dropout(
    signals: NDArray[np.float64],
    dropout_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Randomly mask out leads with zeros.

    Args:
        signals: Input signals of shape ``[leads, samples]``.
        dropout_prob: Probability of dropping each lead.  Must be in [0, 1].
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Augmented signals array (always a copy).

    Raises:
        ValueError: If *dropout_prob* is outside [0, 1] or signals is empty.
    """
    if signals.size == 0:
        return signals.copy()
    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError(
            f"dropout_prob must be in [0, 1], got {dropout_prob}"
        )

    rng = _ensure_rng(rng)
    aug_signals = signals.copy()
    num_leads = signals.shape[0]
    mask = rng.uniform(size=num_leads) < dropout_prob
    aug_signals[mask, :] = 0.0
    return aug_signals


def apply_gaussian_noise(
    signals: NDArray[np.float64],
    std_dev: float = 0.05,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Inject Gaussian noise into the signal.

    Args:
        signals: Input signals of shape ``[leads, samples]``.
        std_dev: Standard deviation of the Gaussian noise.  Must be ≥ 0.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Augmented signals array.

    Raises:
        ValueError: If *std_dev* is negative.
    """
    if signals.size == 0:
        return signals.copy()
    if std_dev < 0:
        raise ValueError(f"std_dev must be >= 0, got {std_dev}")

    rng = _ensure_rng(rng)
    noise = rng.normal(loc=0.0, scale=std_dev, size=signals.shape)
    return signals + noise


def apply_time_shift(
    signals: NDArray[np.float64],
    max_shift_samples: int = 50,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Randomly shift (roll) the signal along the time axis.

    Zeros are padded at the exposed boundary to avoid wrap-around artifacts.

    Args:
        signals: Input signals of shape ``[leads, samples]``.
        max_shift_samples: Maximum shift in either direction.  Must be ≥ 0.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Augmented signals array (always a copy).

    Raises:
        ValueError: If *max_shift_samples* is negative.
    """
    if signals.size == 0:
        return signals.copy()
    if max_shift_samples < 0:
        raise ValueError(
            f"max_shift_samples must be >= 0, got {max_shift_samples}"
        )
    if max_shift_samples == 0:
        return signals.copy()

    rng = _ensure_rng(rng)
    shift = int(rng.integers(-max_shift_samples, max_shift_samples + 1))
    if shift == 0:
        return signals.copy()

    aug_signals = np.roll(signals, shift, axis=1)

    # Pad exposed boundaries with zeros instead of wrapping
    if shift > 0:
        aug_signals[:, :shift] = 0.0
    else:
        aug_signals[:, shift:] = 0.0

    return aug_signals


def apply_amplitude_scale(
    signals: NDArray[np.float64],
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Randomly scale the amplitude of the signal.

    A single scale factor is drawn uniformly from [min_scale, max_scale]
    and applied to all leads and samples.

    Args:
        signals: Input signals of shape ``[leads, samples]``.
        min_scale: Minimum scaling factor.
        max_scale: Maximum scaling factor.  Must be ≥ min_scale.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Augmented signals array.

    Raises:
        ValueError: If *min_scale* > *max_scale*.
    """
    if signals.size == 0:
        return signals.copy()
    if min_scale > max_scale:
        raise ValueError(
            f"min_scale ({min_scale}) must be <= max_scale ({max_scale})"
        )

    rng = _ensure_rng(rng)
    scale = rng.uniform(min_scale, max_scale)
    return signals * scale


def pad_or_truncate(
    signals: NDArray[np.float64],
    target_samples: int,
    pad_value: float = 0.0,
) -> NDArray[np.float64]:
    """Pad (right) or truncate (right) to exactly *target_samples* length.

    Args:
        signals: Input signals of shape ``[leads, samples]``.
        target_samples: Desired number of samples.  Must be > 0.
        pad_value: Value to pad with if signals are shorter than target.

    Returns:
        Truncated or padded signals array (always a copy).

    Raises:
        ValueError: If *target_samples* ≤ 0.
    """
    if target_samples <= 0:
        raise ValueError(
            f"target_samples must be > 0, got {target_samples}"
        )

    num_leads, num_samples = signals.shape

    if num_samples == target_samples:
        return signals.copy()

    if num_samples > target_samples:
        return signals[:, :target_samples].copy()

    # Pad
    pad_width = target_samples - num_samples
    padded_signals = np.pad(
        signals,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded_signals
