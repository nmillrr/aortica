"""Tests for the signal denoising module (US-010).

Verifies baseline wander removal, powerline noise removal, highfreq noise
removal, auto-detection of 50/60 Hz powerline, and SNR improvement on
synthetically corrupted signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.signal.denoising import (
    _auto_detect_powerline,
    denoise,
)

# ─────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────

_FS = 500.0  # Hz
_DURATION = 5.0  # seconds
_N_SAMPLES = int(_FS * _DURATION)


def _time_axis() -> np.ndarray:
    """Return a time vector for the default fs / duration."""
    return np.arange(_N_SAMPLES) / _FS


def _clean_ecg_signal(t: np.ndarray | None = None) -> np.ndarray:
    """Synthetic clean ECG-like signal (sine at 1.2 Hz ≈ 72 bpm).

    The signal is a simple sinusoid that lives well within the passband
    of any reasonable clinical filter chain (0.5 – 40 Hz).
    """
    if t is None:
        t = _time_axis()
    return np.sin(2 * np.pi * 1.2 * t) * 500.0  # 500 µV amplitude


def _make_record(
    signal_1d: np.ndarray,
    fs: float = _FS,
    n_leads: int = 1,
) -> ECGRecord:
    """Wrap a 1-D signal into an ECGRecord (optionally duplicated across leads)."""
    signals = np.tile(signal_1d, (n_leads, 1))
    lead_names = [f"Lead{i}" for i in range(n_leads)]
    return ECGRecord(
        signals=signals,
        sample_rate=fs,
        lead_names=lead_names,
        source_format="synthetic",
    )


def _snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    noise = noisy - clean
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0:
        return float("inf")
    return float(10 * np.log10(power_signal / power_noise))


# ─────────────────────────────────────────────────────────────────
# Baseline wander removal
# ─────────────────────────────────────────────────────────────────


class TestBaselineRemoval:
    """Tests for the 'baseline' denoising method."""

    def test_removes_low_freq_drift(self) -> None:
        """Baseline wander (0.1 Hz drift) should be attenuated."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz drift
        corrupted = clean + drift

        record = _make_record(corrupted)
        result = denoise(record, methods=["baseline"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before, (
            f"SNR did not improve: before={snr_before:.1f} dB, "
            f"after={snr_after:.1f} dB"
        )

    def test_preserves_ecg_content(self) -> None:
        """Signal in the passband should be mostly preserved."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        record = _make_record(clean)
        result = denoise(record, methods=["baseline"])

        # Correlation should be very high.
        corr = np.corrcoef(clean, result.signals[0])[0, 1]
        assert corr > 0.95, f"Correlation too low: {corr:.4f}"

    def test_custom_cutoff(self) -> None:
        """Using a higher cutoff still removes baseline wander."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.3 * t)
        record = _make_record(clean + drift)
        result = denoise(record, methods=["baseline"], baseline_cutoff_hz=1.0)

        snr_after = _snr(clean, result.signals[0])
        # With a 1 Hz cutoff the 0.3 Hz drift should still be suppressed.
        assert snr_after > _snr(clean, clean + drift)


# ─────────────────────────────────────────────────────────────────
# Powerline removal
# ─────────────────────────────────────────────────────────────────


class TestPowerlineRemoval:
    """Tests for the 'powerline' denoising method."""

    def test_removes_60hz(self) -> None:
        """60 Hz powerline interference should be attenuated."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 100.0 * np.sin(2 * np.pi * 60.0 * t)
        corrupted = clean + powerline

        record = _make_record(corrupted)
        result = denoise(record, methods=["powerline"], powerline_freq_hz=60.0)

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before, (
            f"SNR did not improve: before={snr_before:.1f} dB, "
            f"after={snr_after:.1f} dB"
        )

    def test_removes_50hz(self) -> None:
        """50 Hz powerline interference should be attenuated."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 100.0 * np.sin(2 * np.pi * 50.0 * t)
        corrupted = clean + powerline

        record = _make_record(corrupted)
        result = denoise(record, methods=["powerline"], powerline_freq_hz=50.0)

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before

    def test_auto_detect_60hz(self) -> None:
        """Auto-detect should select 60 Hz when 60 Hz is dominant."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 100.0 * np.sin(2 * np.pi * 60.0 * t)
        signals = (clean + powerline).reshape(1, -1)

        detected = _auto_detect_powerline(signals, _FS)
        assert detected == 60.0

    def test_auto_detect_50hz(self) -> None:
        """Auto-detect should select 50 Hz when 50 Hz is dominant."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 100.0 * np.sin(2 * np.pi * 50.0 * t)
        signals = (clean + powerline).reshape(1, -1)

        detected = _auto_detect_powerline(signals, _FS)
        assert detected == 50.0

    def test_auto_detect_default_used(self) -> None:
        """denoise() with powerline_freq_hz=None should auto-detect."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 100.0 * np.sin(2 * np.pi * 60.0 * t)
        corrupted = clean + powerline

        record = _make_record(corrupted)
        result = denoise(record, methods=["powerline"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before


# ─────────────────────────────────────────────────────────────────
# High-frequency noise removal
# ─────────────────────────────────────────────────────────────────


class TestHighFreqRemoval:
    """Tests for the 'highfreq' denoising method."""

    def test_removes_high_freq_noise(self) -> None:
        """High-frequency noise (>40 Hz) should be attenuated."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        rng = np.random.default_rng(42)
        hf_noise = 80.0 * np.sin(2 * np.pi * 100.0 * t)
        hf_noise += 30.0 * rng.standard_normal(len(t))
        corrupted = clean + hf_noise

        record = _make_record(corrupted)
        result = denoise(record, methods=["highfreq"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before

    def test_preserves_ecg_content(self) -> None:
        """Signal below 40 Hz should be mostly preserved."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        record = _make_record(clean)
        result = denoise(record, methods=["highfreq"])

        corr = np.corrcoef(clean, result.signals[0])[0, 1]
        assert corr > 0.95

    def test_custom_cutoff(self) -> None:
        """Using a custom lowpass cutoff should work."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        hf = 60.0 * np.sin(2 * np.pi * 80.0 * t)
        corrupted = clean + hf

        record = _make_record(corrupted)
        result = denoise(record, methods=["highfreq"], highfreq_cutoff_hz=30.0)

        snr_after = _snr(clean, result.signals[0])
        assert snr_after > _snr(clean, corrupted)


# ─────────────────────────────────────────────────────────────────
# Combination / pipeline tests
# ─────────────────────────────────────────────────────────────────


class TestCombinedDenoising:
    """Tests for applying multiple filter stages together."""

    def test_all_methods_combined(self) -> None:
        """Applying all three methods on a multiply-corrupted signal."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.15 * t)
        powerline = 80.0 * np.sin(2 * np.pi * 60.0 * t)
        hf = 40.0 * np.sin(2 * np.pi * 120.0 * t)
        corrupted = clean + drift + powerline + hf

        record = _make_record(corrupted)
        result = denoise(record, methods=["baseline", "powerline", "highfreq"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after > snr_before

    def test_subset_methods(self) -> None:
        """Applying only a subset of methods should work."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.1 * t)
        record = _make_record(clean + drift)

        # Only baseline removal — no error should be raised.
        result = denoise(record, methods=["baseline"])
        assert result.signals.shape == record.signals.shape

    def test_empty_methods_returns_copy(self) -> None:
        """An empty methods list should return a copy of the record."""
        clean = _clean_ecg_signal()
        record = _make_record(clean)
        result = denoise(record, methods=[])

        np.testing.assert_array_equal(result.signals, record.signals)
        assert result.signals is not record.signals  # must be a copy

    def test_multi_lead(self) -> None:
        """Denoising works on multi-lead records."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.1 * t)
        record = _make_record(clean + drift, n_leads=12)

        result = denoise(record, methods=["baseline"])
        assert result.signals.shape == (12, _N_SAMPLES)
        assert result.num_leads == 12

    def test_preserves_metadata(self) -> None:
        """Returned ECGRecord should carry forward metadata."""
        clean = _clean_ecg_signal()
        record = ECGRecord(
            signals=clean.reshape(1, -1),
            sample_rate=_FS,
            lead_names=["II"],
            source_format="test",
            units="mV",
            patient_metadata={"id": "patient_1"},
        )
        result = denoise(record, methods=["baseline"])

        assert result.source_format == "test"
        assert result.units == "mV"
        assert result.patient_metadata == {"id": "patient_1"}
        assert result.sample_rate == _FS
        assert result.lead_names == ["II"]


# ─────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────


class TestDenoisingErrors:
    """Tests for error handling and edge cases."""

    def test_invalid_method_raises(self) -> None:
        """An unknown method name should raise ValueError."""
        clean = _clean_ecg_signal()
        record = _make_record(clean)
        with pytest.raises(ValueError, match="Unknown denoise method"):
            denoise(record, methods=["baseline", "invalid_method"])  # type: ignore[list-item]

    def test_low_sample_rate_no_crash(self) -> None:
        """Very low sample rate should not crash (filters degrade gracefully)."""
        fs = 30.0
        n = int(fs * 2)
        t = np.arange(n) / fs
        sig = np.sin(2 * np.pi * 1.0 * t) * 500.0
        record = _make_record(sig, fs=fs)
        # Should not raise even though Nyquist is only 15 Hz.
        result = denoise(record, methods=["baseline", "powerline", "highfreq"])
        assert result.signals.shape == (1, n)

    def test_powerline_above_nyquist_skipped(self) -> None:
        """Powerline freq above Nyquist should be silently skipped."""
        fs = 80.0  # Nyquist = 40 Hz, so 60 Hz cannot be notched.
        n = int(fs * 2)
        t = np.arange(n) / fs
        sig = np.sin(2 * np.pi * 1.0 * t) * 500.0
        record = _make_record(sig, fs=fs)
        # Should not raise, just skip the notch.
        result = denoise(record, methods=["powerline"], powerline_freq_hz=60.0)
        np.testing.assert_array_almost_equal(result.signals[0], sig, decimal=5)


# ─────────────────────────────────────────────────────────────────
# SNR improvement verification (acceptance criterion)
# ─────────────────────────────────────────────────────────────────


class TestSNRImprovement:
    """Verify SNR improvement on synthetically corrupted signals."""

    def test_snr_improves_baseline_wander(self) -> None:
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 300.0 * np.sin(2 * np.pi * 0.2 * t)
        corrupted = clean + drift
        record = _make_record(corrupted)
        result = denoise(record, methods=["baseline"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        # Expect at least 6 dB improvement.
        assert snr_after - snr_before > 6.0, (
            f"Expected >6 dB improvement, got {snr_after - snr_before:.1f} dB"
        )

    def test_snr_improves_powerline(self) -> None:
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        powerline = 150.0 * np.sin(2 * np.pi * 60.0 * t)
        corrupted = clean + powerline
        record = _make_record(corrupted)
        result = denoise(record, methods=["powerline"], powerline_freq_hz=60.0)

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after - snr_before > 6.0, (
            f"Expected >6 dB improvement, got {snr_after - snr_before:.1f} dB"
        )

    def test_snr_improves_highfreq(self) -> None:
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        hf = 100.0 * np.sin(2 * np.pi * 100.0 * t)
        corrupted = clean + hf
        record = _make_record(corrupted)
        result = denoise(record, methods=["highfreq"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after - snr_before > 6.0, (
            f"Expected >6 dB improvement, got {snr_after - snr_before:.1f} dB"
        )

    def test_snr_improves_combined(self) -> None:
        """Full pipeline on a signal with all three noise types."""
        t = _time_axis()
        clean = _clean_ecg_signal(t)
        drift = 200.0 * np.sin(2 * np.pi * 0.15 * t)
        powerline = 100.0 * np.sin(2 * np.pi * 60.0 * t)
        hf = 80.0 * np.sin(2 * np.pi * 100.0 * t)
        corrupted = clean + drift + powerline + hf

        record = _make_record(corrupted)
        result = denoise(record, methods=["baseline", "powerline", "highfreq"])

        snr_before = _snr(clean, corrupted)
        snr_after = _snr(clean, result.signals[0])
        assert snr_after - snr_before > 6.0, (
            f"Expected >6 dB improvement, got {snr_after - snr_before:.1f} dB"
        )
