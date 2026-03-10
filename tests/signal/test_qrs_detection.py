"""Tests for :mod:`aortica.signal.qrs_detection`.

Tests use *synthetic* ECG-like waveforms so they run fast and without
any external data downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord
from aortica.signal.qrs_detection import detect_qrs

# ─────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────

def _make_synthetic_ecg(
    duration_s: float = 5.0,
    sample_rate: float = 500.0,
    heart_rate_bpm: float = 72.0,
    lead_names: list[str] | None = None,
    num_leads: int = 1,
) -> tuple[ECGRecord, NDArray[np.intp]]:
    """Create a synthetic ECG-like signal with known R-peak positions.

    The signal consists of Gaussian-shaped QRS complexes at a fixed
    heart rate, with a small amount of baseline wander.

    Returns
    -------
    ecg_record
        The synthetic ECGRecord.
    expected_peaks
        Array of the true R-peak sample indices.
    """
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # R-R interval
    rr_interval_s = 60.0 / heart_rate_bpm
    rr_samples = int(round(rr_interval_s * sample_rate))

    # Place R-peaks at regular intervals, starting at 0.5 s
    first_peak = int(0.5 * sample_rate)
    peak_indices = np.arange(first_peak, n_samples, rr_samples)

    # Build signal: Gaussian-shaped QRS complexes
    sig = np.zeros(n_samples, dtype=np.float64)
    qrs_width_s = 0.04  # 40 ms half-width
    qrs_width_samples = qrs_width_s * sample_rate

    for p in peak_indices:
        # Gaussian centered at p
        x = np.arange(n_samples) - p
        sig += 1000.0 * np.exp(-(x ** 2) / (2 * qrs_width_samples ** 2))

    # Add small baseline wander
    sig += 50.0 * np.sin(2 * np.pi * 0.3 * t)

    if lead_names is None:
        if num_leads == 1:
            lead_names = ["II"]
        else:
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"][:num_leads]

    # Replicate signal across leads (with slight amplitude variation)
    rng = np.random.default_rng(42)
    signals = np.empty((len(lead_names), n_samples), dtype=np.float64)
    for i in range(len(lead_names)):
        signals[i] = sig * (0.8 + 0.4 * rng.random())

    ecg = ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="synthetic",
        units="µV",
    )
    return ecg, peak_indices


@pytest.fixture()
def synthetic_ecg() -> tuple[ECGRecord, NDArray[np.intp]]:
    """Standard 5-second, single-lead synthetic ECG."""
    return _make_synthetic_ecg()


@pytest.fixture()
def synthetic_12lead() -> tuple[ECGRecord, NDArray[np.intp]]:
    """12-lead synthetic ECG."""
    return _make_synthetic_ecg(num_leads=12)


# ─────────────────────────────────────────────────────────────────
# Tests: Pan-Tompkins backend (always available — scipy only)
# ─────────────────────────────────────────────────────────────────

class TestPanTompkins:
    """Tests for the pan_tompkins QRS detection backend."""

    def test_detects_peaks_single_lead(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, expected = synthetic_ecg
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert isinstance(peaks, np.ndarray)
        assert peaks.ndim == 1
        assert len(peaks) > 0
        # Should detect roughly the right number of peaks (±1)
        assert abs(len(peaks) - len(expected)) <= 2

    def test_peaks_near_expected(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, expected = synthetic_ecg
        peaks = detect_qrs(ecg, method="pan_tompkins")
        # Each detected peak should be within 50 ms of an expected peak
        tolerance_samples = int(0.05 * ecg.sample_rate)
        matched = 0
        for dp in peaks:
            if np.min(np.abs(expected - dp)) <= tolerance_samples:
                matched += 1
        sensitivity = matched / max(len(expected), 1)
        assert sensitivity >= 0.8, f"Sensitivity {sensitivity:.2f} < 0.8"

    def test_12lead_defaults_to_lead_ii(
        self, synthetic_12lead: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_12lead
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert len(peaks) > 0

    def test_explicit_lead(
        self, synthetic_12lead: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_12lead
        peaks = detect_qrs(ecg, method="pan_tompkins", lead="V1")
        assert len(peaks) > 0

    def test_invalid_lead_raises(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_ecg
        with pytest.raises(ValueError, match="Lead 'V5' not found"):
            detect_qrs(ecg, method="pan_tompkins", lead="V5")

    def test_peaks_sorted(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_ecg
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert np.all(np.diff(peaks) > 0), "Peaks must be sorted ascending"

    def test_return_dtype(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_ecg
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert peaks.dtype == np.intp

    def test_short_signal(self) -> None:
        """1 second signal with 1 beat — should detect at least 1 peak."""
        ecg, expected = _make_synthetic_ecg(
            duration_s=1.0, heart_rate_bpm=60.0
        )
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert len(peaks) >= 1

    def test_high_heart_rate(self) -> None:
        """High HR (~180 bpm) should still detect peaks."""
        ecg, expected = _make_synthetic_ecg(
            duration_s=5.0, heart_rate_bpm=180.0
        )
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert len(peaks) > 5  # 5 s at 180 bpm → ~15 beats

    def test_low_heart_rate(self) -> None:
        """Low HR (~40 bpm) should still detect peaks."""
        ecg, expected = _make_synthetic_ecg(
            duration_s=10.0, heart_rate_bpm=40.0
        )
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert len(peaks) >= 3  # 10 s at 40 bpm → ~6–7 beats

    def test_first_lead_fallback(self) -> None:
        """When Lead II is not present, uses first lead."""
        ecg, _ = _make_synthetic_ecg(lead_names=["V1"])
        peaks = detect_qrs(ecg, method="pan_tompkins")
        assert len(peaks) > 0


# ─────────────────────────────────────────────────────────────────
# Tests: NeuroKit2 backend
# ─────────────────────────────────────────────────────────────────

class TestNeuroKit:
    """Tests for the neurokit QRS detection backend."""

    @pytest.fixture(autouse=True)
    def _check_neurokit(self) -> None:
        pytest.importorskip("neurokit2")

    def test_detects_peaks(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, expected = synthetic_ecg
        peaks = detect_qrs(ecg, method="neurokit")
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) > 0
        assert abs(len(peaks) - len(expected)) <= 2

    def test_peaks_near_expected(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, expected = synthetic_ecg
        peaks = detect_qrs(ecg, method="neurokit")
        tolerance_samples = int(0.05 * ecg.sample_rate)
        matched = 0
        for dp in peaks:
            if np.min(np.abs(expected - dp)) <= tolerance_samples:
                matched += 1
        sensitivity = matched / max(len(expected), 1)
        assert sensitivity >= 0.8, f"Sensitivity {sensitivity:.2f} < 0.8"

    def test_12lead(
        self, synthetic_12lead: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_12lead
        peaks = detect_qrs(ecg, method="neurokit")
        assert len(peaks) > 0

    def test_explicit_lead(
        self, synthetic_12lead: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_12lead
        peaks = detect_qrs(ecg, method="neurokit", lead="III")
        assert len(peaks) > 0

    def test_sorted_and_dtype(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_ecg
        peaks = detect_qrs(ecg, method="neurokit")
        assert peaks.dtype == np.intp
        assert np.all(np.diff(peaks) > 0)


# ─────────────────────────────────────────────────────────────────
# Tests: General / dispatch
# ─────────────────────────────────────────────────────────────────

class TestDispatch:
    """Tests for error handling and method dispatch."""

    def test_invalid_method(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        ecg, _ = synthetic_ecg
        with pytest.raises(ValueError, match="Unknown QRS detection method"):
            detect_qrs(ecg, method="bogus")  # type: ignore[arg-type]

    def test_default_method_is_neurokit(
        self, synthetic_ecg: tuple[ECGRecord, NDArray[np.intp]]
    ) -> None:
        """The default method parameter should be 'neurokit'."""
        try:
            import neurokit2  # noqa: F401
        except ImportError:
            pytest.skip("neurokit2 not installed")
        ecg, _ = synthetic_ecg
        peaks = detect_qrs(ecg)
        assert len(peaks) > 0

    def test_import_from_package(self) -> None:
        """detect_qrs should be importable from aortica.signal."""
        from aortica.signal import detect_qrs as dq
        assert callable(dq)
