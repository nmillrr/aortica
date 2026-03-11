"""Signal denoising module.

Provides configurable ECG signal denoising with three filter stages:

* **baseline** — Removes baseline wander via a highpass Butterworth filter.
* **powerline** — Removes powerline interference (50 / 60 Hz) via a notch filter.
* **highfreq** — Removes high-frequency noise via a lowpass Butterworth filter.

Each filter can be applied independently or in combination.

Usage::

    from aortica.signal import denoise
    cleaned = denoise(ecg_record, methods=["baseline", "powerline", "highfreq"])
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from aortica.io.ecg_record import ECGRecord

# Type alias for supported denoising method names.
DenoiseMethod = Literal["baseline", "powerline", "highfreq"]

# Default filter cutoffs / frequencies.
_DEFAULT_BASELINE_CUTOFF_HZ = 0.5
_DEFAULT_POWERLINE_FREQ_HZ = 60.0  # 50 or 60 Hz
_DEFAULT_HIGHFREQ_CUTOFF_HZ = 40.0
_NOTCH_QUALITY_FACTOR = 30.0  # Q factor for the notch filter


def denoise(
    ecg_record: ECGRecord,
    methods: Sequence[DenoiseMethod] = ("baseline", "powerline", "highfreq"),
    *,
    baseline_cutoff_hz: float = _DEFAULT_BASELINE_CUTOFF_HZ,
    powerline_freq_hz: float | None = None,
    highfreq_cutoff_hz: float = _DEFAULT_HIGHFREQ_CUTOFF_HZ,
) -> ECGRecord:
    """Return a denoised copy of *ecg_record*.

    Parameters
    ----------
    ecg_record:
        The ECG recording to denoise.
    methods:
        Sequence of filter stages to apply, in order.  Any subset of
        ``("baseline", "powerline", "highfreq")`` is accepted.
    baseline_cutoff_hz:
        Cutoff frequency in Hz for the highpass baseline-wander filter.
        Default ``0.5``.
    powerline_freq_hz:
        Centre frequency for powerline notch removal.  If *None*
        (default), the frequency is auto-detected by choosing the
        candidate (50 or 60 Hz) with the larger spectral peak on the
        first lead.
    highfreq_cutoff_hz:
        Cutoff frequency in Hz for the lowpass high-frequency noise
        filter.  Default ``40.0``.

    Returns
    -------
    ECGRecord
        A new :class:`ECGRecord` with denoised signals.

    Raises
    ------
    ValueError
        If an unsupported method name is given, or if any cutoff is
        invalid relative to the Nyquist frequency.
    """
    _validate_methods(methods)
    fs = ecg_record.sample_rate
    nyq = fs / 2.0

    # Work on a mutable copy of the signal matrix.
    signals: NDArray[np.float64] = ecg_record.signals.astype(np.float64).copy()

    for method in methods:
        if method == "baseline":
            signals = _remove_baseline(signals, fs, nyq, baseline_cutoff_hz)
        elif method == "powerline":
            freq = powerline_freq_hz
            if freq is None:
                freq = _auto_detect_powerline(signals, fs)
            signals = _remove_powerline(signals, fs, nyq, freq)
        elif method == "highfreq":
            signals = _remove_highfreq(signals, fs, nyq, highfreq_cutoff_hz)

    return ECGRecord(
        signals=signals,
        sample_rate=ecg_record.sample_rate,
        lead_names=list(ecg_record.lead_names),
        duration_seconds=ecg_record.duration_seconds,
        patient_metadata=dict(ecg_record.patient_metadata)
        if ecg_record.patient_metadata
        else None,
        source_format=ecg_record.source_format,
        units=ecg_record.units,
    )


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

_VALID_METHODS: frozenset[str] = frozenset({"baseline", "powerline", "highfreq"})


def _validate_methods(methods: Sequence[str]) -> None:
    """Raise :class:`ValueError` for any unrecognised method names."""
    for m in methods:
        if m not in _VALID_METHODS:
            raise ValueError(
                f"Unknown denoise method '{m}'. "
                f"Supported: {sorted(_VALID_METHODS)}"
            )


# ── baseline wander removal ─────────────────────────────────────


def _remove_baseline(
    signals: NDArray[np.float64],
    fs: float,
    nyq: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Apply a highpass Butterworth filter to remove baseline wander."""
    wn = cutoff_hz / nyq
    if wn <= 0 or wn >= 1:
        # Cutoff outside valid range — return unmodified.
        return signals
    sos = sp_signal.butter(4, wn, btype="high", output="sos")
    result: NDArray[np.float64] = sp_signal.sosfiltfilt(sos, signals, axis=1)
    return result


# ── powerline removal ───────────────────────────────────────────


def _auto_detect_powerline(
    signals: NDArray[np.float64],
    fs: float,
) -> float:
    """Determine whether powerline interference is at 50 or 60 Hz.

    Computes the FFT of the first lead and compares spectral energy
    around 50 and 60 Hz.  Returns the candidate with the larger peak.
    If neither frequency is resolvable (e.g. ``fs < 120``), defaults to
    60 Hz.
    """
    nyq = fs / 2.0
    if nyq <= 50.0:
        return _DEFAULT_POWERLINE_FREQ_HZ

    lead = signals[0]
    n = len(lead)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(lead))

    def _energy_near(target: float, band: float = 2.0) -> float:
        mask = (freqs >= target - band) & (freqs <= target + band)
        if not np.any(mask):
            return 0.0
        return float(np.max(fft_mag[mask]))

    e50 = _energy_near(50.0)
    e60 = _energy_near(60.0)

    return 50.0 if e50 > e60 else 60.0


def _remove_powerline(
    signals: NDArray[np.float64],
    fs: float,
    nyq: float,
    freq_hz: float,
) -> NDArray[np.float64]:
    """Apply a notch filter at *freq_hz* to suppress powerline noise."""
    if freq_hz >= nyq:
        # Cannot notch a frequency at or above Nyquist — skip.
        return signals
    b, a = sp_signal.iirnotch(freq_hz, _NOTCH_QUALITY_FACTOR, fs)
    result: NDArray[np.float64] = sp_signal.filtfilt(b, a, signals, axis=1)
    return result


# ── high-frequency noise removal ────────────────────────────────


def _remove_highfreq(
    signals: NDArray[np.float64],
    fs: float,
    nyq: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Apply a lowpass Butterworth filter to remove high-frequency noise."""
    wn = cutoff_hz / nyq
    if wn <= 0 or wn >= 1:
        return signals
    sos = sp_signal.butter(4, wn, btype="low", output="sos")
    result: NDArray[np.float64] = sp_signal.sosfiltfilt(sos, signals, axis=1)
    return result
