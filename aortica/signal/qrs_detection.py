"""QRS detection module.

Provides R-peak detection for ECG signals with two backends:

* **neurokit** — Uses :mod:`neurokit2` (requires ``pip install aortica[signal]``).
* **pan_tompkins** — A pure-SciPy Pan–Tompkins implementation (no extra deps).

Usage::

    from aortica.signal import detect_qrs
    peaks = detect_qrs(ecg_record, method="neurokit")
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from aortica.io.ecg_record import ECGRecord

# The default lead used for R-peak detection when working on multi-lead
# recordings.
_DEFAULT_LEAD = "II"

# Type alias for the supported method names.
QRSMethod = Literal["neurokit", "pan_tompkins"]


def detect_qrs(
    ecg_record: ECGRecord,
    method: QRSMethod = "neurokit",
    lead: str | None = None,
) -> NDArray[np.intp]:
    """Detect R-peak locations in *ecg_record*.

    Parameters
    ----------
    ecg_record:
        An :class:`~aortica.io.ecg_record.ECGRecord` instance.
    method:
        Detection backend to use.  ``"neurokit"`` (default) uses
        NeuroKit2's ``ecg_findpeaks``.  ``"pan_tompkins"`` uses a
        pure-SciPy implementation of the Pan–Tompkins algorithm.
    lead:
        Name of the lead to run detection on.  If *None* (default),
        Lead **II** is used when available; otherwise the first lead is
        used.

    Returns
    -------
    numpy.ndarray
        1-D integer array of R-peak sample indices, sorted in ascending
        order.

    Raises
    ------
    ValueError
        If *method* is not one of the supported backends, or if the
        requested *lead* is not present in *ecg_record*.
    ImportError
        If ``method="neurokit"`` is chosen but ``neurokit2`` is not
        installed.
    """
    # ── resolve the target lead ──────────────────────────────────
    sig = _select_lead(ecg_record, lead)
    fs = ecg_record.sample_rate

    # ── dispatch to the requested backend ────────────────────────
    if method == "neurokit":
        return _detect_neurokit(sig, fs)
    if method == "pan_tompkins":
        return _detect_pan_tompkins(sig, fs)

    raise ValueError(
        f"Unknown QRS detection method '{method}'. "
        f"Supported: 'neurokit', 'pan_tompkins'."
    )


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────


def _select_lead(ecg_record: ECGRecord, lead: str | None) -> NDArray[np.float64]:
    """Return the 1-D signal for the chosen lead."""
    if lead is not None:
        if lead not in ecg_record.lead_names:
            raise ValueError(
                f"Lead '{lead}' not found. "
                f"Available leads: {ecg_record.lead_names}"
            )
        idx = ecg_record.lead_names.index(lead)
        result: NDArray[np.float64] = ecg_record.signals[idx].astype(np.float64)
        return result

    # Default: Lead II if available, else first lead.
    if _DEFAULT_LEAD in ecg_record.lead_names:
        idx = ecg_record.lead_names.index(_DEFAULT_LEAD)
    else:
        idx = 0
    default_result: NDArray[np.float64] = ecg_record.signals[idx].astype(np.float64)
    return default_result


# ─────────────────────────────────────────────────────────────────
# Backend: NeuroKit2
# ─────────────────────────────────────────────────────────────────


def _detect_neurokit(
    sig: NDArray[np.float64],
    fs: float,
) -> NDArray[np.intp]:
    """R-peak detection via NeuroKit2's ``ecg_findpeaks``."""
    try:
        import neurokit2 as nk  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "neurokit2 is required for method='neurokit'. "
            "Install it with: pip install aortica[signal]"
        ) from exc

    # nk.ecg_findpeaks expects a cleaned signal.  We first apply nk's
    # cleaning (bandpass + powerline removal), then find peaks.
    cleaned = nk.ecg_clean(sig, sampling_rate=int(fs))
    peak_info = nk.ecg_findpeaks(cleaned, sampling_rate=int(fs))
    peaks: list[int] = peak_info.get("ECG_R_Peaks", [])
    return np.asarray(sorted(peaks), dtype=np.intp)


# ─────────────────────────────────────────────────────────────────
# Backend: Pan–Tompkins (pure SciPy)
# ─────────────────────────────────────────────────────────────────


def _detect_pan_tompkins(
    sig: NDArray[np.float64],
    fs: float,
) -> NDArray[np.intp]:
    """Pan–Tompkins R-peak detection using SciPy filters.

    Implements the classic Pan & Tompkins (1985) algorithm:

    1. Bandpass filter (5–15 Hz)
    2. Derivative
    3. Squaring
    4. Moving-window integration
    5. Adaptive thresholding via ``scipy.signal.find_peaks``
    """
    # 1) Bandpass filter (5–15 Hz) ────────────────────────────────
    nyq = fs / 2.0
    low = 5.0 / nyq
    high = min(15.0 / nyq, 0.99)  # guard against fs <= 30 Hz edge case
    if low >= high:
        # Sampling rate too low for a meaningful bandpass;
        # fall back to simple peak detection on the raw signal.
        return _simple_peak_detection(sig, fs)

    sos = sp_signal.butter(2, [low, high], btype="band", output="sos")
    filtered = sp_signal.sosfiltfilt(sos, sig)

    # 2) Differentiation ──────────────────────────────────────────
    diff = np.diff(filtered, prepend=filtered[0])

    # 3) Squaring ─────────────────────────────────────────────────
    squared = diff ** 2

    # 4) Moving-window integration ────────────────────────────────
    win_size = int(round(0.15 * fs))  # ~150 ms window
    win_size = max(win_size, 1)
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5) Peak detection with adaptive threshold ───────────────────
    # Minimum distance between R-peaks: assume max HR ~220 bpm → ~0.27 s
    min_distance = int(round(0.27 * fs))
    min_distance = max(min_distance, 1)

    # Height threshold: fraction of the maximum of the integrated signal.
    height_threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 0.0

    peaks, _ = sp_signal.find_peaks(
        integrated,
        distance=min_distance,
        height=height_threshold,
    )

    return np.asarray(peaks, dtype=np.intp)


def _simple_peak_detection(
    sig: NDArray[np.float64],
    fs: float,
) -> NDArray[np.intp]:
    """Minimal fallback peak detection for very low sample rates."""
    min_distance = max(int(round(0.27 * fs)), 1)
    peaks, _ = sp_signal.find_peaks(sig, distance=min_distance)
    return np.asarray(peaks, dtype=np.intp)
