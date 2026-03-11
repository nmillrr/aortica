"""Tests for ECG data augmentation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from aortica.data.augmentations import (
    apply_amplitude_scale,
    apply_gaussian_noise,
    apply_lead_dropout,
    apply_time_shift,
    pad_or_truncate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ones_signal() -> np.ndarray:
    """12-lead, 100-sample signal of ones."""
    return np.ones((12, 100))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# apply_lead_dropout
# ---------------------------------------------------------------------------

class TestApplyLeadDropout:
    def test_shape_preserved(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_lead_dropout(ones_signal, dropout_prob=0.5, rng=rng)
        assert aug.shape == ones_signal.shape

    def test_some_leads_zeroed(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_lead_dropout(ones_signal, dropout_prob=0.5, rng=rng)
        zero_leads = np.where(~aug.any(axis=1))[0]
        assert 0 < len(zero_leads) < 12

    def test_zero_prob_no_dropout(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_lead_dropout(ones_signal, dropout_prob=0.0, rng=rng)
        np.testing.assert_array_equal(aug, ones_signal)

    def test_one_prob_all_dropout(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_lead_dropout(ones_signal, dropout_prob=1.0, rng=rng)
        assert np.all(aug == 0.0)

    def test_returns_copy(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_lead_dropout(ones_signal, dropout_prob=0.0, rng=rng)
        assert aug is not ones_signal

    def test_reproducibility(self, ones_signal: np.ndarray) -> None:
        r1 = apply_lead_dropout(ones_signal, 0.5, rng=np.random.default_rng(99))
        r2 = apply_lead_dropout(ones_signal, 0.5, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_invalid_prob_raises(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="dropout_prob"):
            apply_lead_dropout(ones_signal, dropout_prob=1.5, rng=rng)
        with pytest.raises(ValueError, match="dropout_prob"):
            apply_lead_dropout(ones_signal, dropout_prob=-0.1, rng=rng)

    def test_empty_signal(self, rng: np.random.Generator) -> None:
        empty = np.empty((0, 0))
        result = apply_lead_dropout(empty, rng=rng)
        assert result.size == 0


# ---------------------------------------------------------------------------
# apply_gaussian_noise
# ---------------------------------------------------------------------------

class TestApplyGaussianNoise:
    def test_shape_preserved(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_gaussian_noise(ones_signal, std_dev=0.1, rng=rng)
        assert aug.shape == ones_signal.shape

    def test_noise_injected(self, rng: np.random.Generator) -> None:
        zeros = np.zeros((12, 100))
        aug = apply_gaussian_noise(zeros, std_dev=0.1, rng=rng)
        assert np.any(aug != 0)
        assert np.std(aug) > 0.05

    def test_zero_std_no_noise(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_gaussian_noise(ones_signal, std_dev=0.0, rng=rng)
        np.testing.assert_array_equal(aug, ones_signal)

    def test_reproducibility(self) -> None:
        sig = np.zeros((4, 50))
        r1 = apply_gaussian_noise(sig, 0.1, rng=np.random.default_rng(7))
        r2 = apply_gaussian_noise(sig, 0.1, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1, r2)

    def test_negative_std_raises(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="std_dev"):
            apply_gaussian_noise(ones_signal, std_dev=-1.0, rng=rng)

    def test_empty_signal(self, rng: np.random.Generator) -> None:
        empty = np.empty((0, 0))
        result = apply_gaussian_noise(empty, rng=rng)
        assert result.size == 0


# ---------------------------------------------------------------------------
# apply_time_shift
# ---------------------------------------------------------------------------

class TestApplyTimeShift:
    def test_shape_preserved(self, rng: np.random.Generator) -> None:
        signals = np.zeros((12, 100))
        signals[:, 50] = 1.0
        aug = apply_time_shift(signals, max_shift_samples=10, rng=rng)
        assert aug.shape == signals.shape

    def test_peak_moved(self, rng: np.random.Generator) -> None:
        signals = np.zeros((12, 100))
        signals[:, 50] = 1.0
        aug = apply_time_shift(signals, max_shift_samples=10, rng=rng)
        peak_idx = int(np.argmax(aug[0]))
        assert 40 <= peak_idx <= 60
        assert peak_idx != 50

    def test_boundary_zeroed_on_right_shift(self) -> None:
        """When shifted right, left boundary should be zeros."""
        signals = np.ones((2, 20))
        # Use a seed that produces a positive shift
        for seed in range(100):
            rng = np.random.default_rng(seed)
            aug = apply_time_shift(signals, max_shift_samples=5, rng=rng)
            shift = int(np.argmax(aug[0] != 0)) if not np.all(aug[0] == 0) else 0
            if shift > 0:
                # Left boundary should be zero
                np.testing.assert_array_equal(aug[:, :shift], 0.0)
                break

    def test_boundary_zeroed_on_left_shift(self) -> None:
        """When shifted left, right boundary should be zeros."""
        signals = np.ones((2, 20))
        for seed in range(100):
            rng = np.random.default_rng(seed)
            aug = apply_time_shift(signals, max_shift_samples=5, rng=rng)
            # Check if right boundary is zero (left shift)
            last_nonzero = np.max(np.where(aug[0] != 0)) if np.any(aug[0] != 0) else 19
            if last_nonzero < 19:
                np.testing.assert_array_equal(aug[:, last_nonzero + 1:], 0.0)
                break

    def test_zero_shift_returns_copy(self, ones_signal: np.ndarray) -> None:
        aug = apply_time_shift(ones_signal, max_shift_samples=0)
        np.testing.assert_array_equal(aug, ones_signal)
        assert aug is not ones_signal

    def test_reproducibility(self) -> None:
        sig = np.random.default_rng(0).standard_normal((4, 50))
        r1 = apply_time_shift(sig, 10, rng=np.random.default_rng(7))
        r2 = apply_time_shift(sig, 10, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1, r2)

    def test_negative_shift_raises(self, ones_signal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="max_shift_samples"):
            apply_time_shift(ones_signal, max_shift_samples=-1)

    def test_empty_signal(self, rng: np.random.Generator) -> None:
        empty = np.empty((0, 0))
        result = apply_time_shift(empty, rng=rng)
        assert result.size == 0


# ---------------------------------------------------------------------------
# apply_amplitude_scale
# ---------------------------------------------------------------------------

class TestApplyAmplitudeScale:
    def test_shape_preserved(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_amplitude_scale(ones_signal, 0.5, 2.0, rng=rng)
        assert aug.shape == ones_signal.shape

    def test_scale_in_range(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_amplitude_scale(ones_signal, 0.5, 2.0, rng=rng)
        scale = aug[0, 0]
        assert 0.5 <= scale <= 2.0
        # All values should be scaled uniformly
        np.testing.assert_array_almost_equal(aug, scale)

    def test_identity_scale(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        aug = apply_amplitude_scale(ones_signal, 1.0, 1.0, rng=rng)
        np.testing.assert_array_equal(aug, ones_signal)

    def test_reproducibility(self, ones_signal: np.ndarray) -> None:
        r1 = apply_amplitude_scale(ones_signal, 0.5, 2.0, rng=np.random.default_rng(7))
        r2 = apply_amplitude_scale(ones_signal, 0.5, 2.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1, r2)

    def test_inverted_range_raises(self, ones_signal: np.ndarray, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="min_scale"):
            apply_amplitude_scale(ones_signal, 2.0, 0.5, rng=rng)

    def test_empty_signal(self, rng: np.random.Generator) -> None:
        empty = np.empty((0, 0))
        result = apply_amplitude_scale(empty, rng=rng)
        assert result.size == 0


# ---------------------------------------------------------------------------
# pad_or_truncate
# ---------------------------------------------------------------------------

class TestPadOrTruncate:
    def test_pad(self, ones_signal: np.ndarray) -> None:
        padded = pad_or_truncate(ones_signal, target_samples=150)
        assert padded.shape == (12, 150)
        np.testing.assert_array_equal(padded[:, :100], 1.0)
        np.testing.assert_array_equal(padded[:, 100:], 0.0)

    def test_truncate(self, ones_signal: np.ndarray) -> None:
        truncated = pad_or_truncate(ones_signal, target_samples=50)
        assert truncated.shape == (12, 50)
        np.testing.assert_array_equal(truncated, 1.0)

    def test_exact_length_returns_copy(self, ones_signal: np.ndarray) -> None:
        result = pad_or_truncate(ones_signal, target_samples=100)
        np.testing.assert_array_equal(result, ones_signal)
        assert result is not ones_signal

    def test_custom_pad_value(self) -> None:
        signals = np.zeros((2, 10))
        padded = pad_or_truncate(signals, target_samples=15, pad_value=-1.0)
        np.testing.assert_array_equal(padded[:, 10:], -1.0)

    def test_zero_target_raises(self, ones_signal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="target_samples"):
            pad_or_truncate(ones_signal, target_samples=0)

    def test_negative_target_raises(self, ones_signal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="target_samples"):
            pad_or_truncate(ones_signal, target_samples=-10)
