"""WFDB format reader.

Loads ECG recordings in WFDB format (``.hea`` / ``.dat``) and returns
:class:`~aortica.io.ecg_record.ECGRecord` instances.  Supports both
single-segment and multi-segment WFDB records.

Requires the ``wfdb`` package (``pip install wfdb``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from aortica.io.ecg_record import ECGRecord


def read_wfdb(path: str | Path) -> ECGRecord:
    """Read a WFDB record and return an :class:`ECGRecord`.

    Parameters
    ----------
    path:
        Path to the WFDB record **without** file extension.  For example,
        ``"data/mitbih/100"`` when the directory contains ``100.hea``
        and ``100.dat``.

    Returns
    -------
    ECGRecord
        A standardised ECG record with signals in physical units (µV).

    Raises
    ------
    FileNotFoundError
        If the header file cannot be found at *path*.
    ValueError
        If the record contains no signal data.
    """
    path = Path(path)

    # Strip .hea / .dat extension if the caller included one.
    if path.suffix in (".hea", ".dat"):
        path = path.with_suffix("")

    header_path = path.with_suffix(".hea")
    if not header_path.exists():
        raise FileNotFoundError(f"WFDB header not found: {header_path}")

    # Read the record.  ``m2s=True`` converts multi-segment records
    # into a single contiguous ``Record`` object automatically.
    record = wfdb.rdrecord(str(path), physical=True, m2s=True)

    # Extract the physical signal matrix ──────────────────────────
    p_signal: np.ndarray | None = getattr(record, "p_signal", None)
    if p_signal is None or p_signal.size == 0:
        raise ValueError(f"Record at '{path}' contains no physical signal data.")

    # wfdb returns shape [samples, channels] — we need [leads, samples]
    signals = p_signal.T.astype(np.float64)

    # Lead/channel names ──────────────────────────────────────────
    sig_names: list[str] = list(record.sig_name)

    # Sample rate ─────────────────────────────────────────────────
    sample_rate: float = float(record.fs)

    # Physical units ──────────────────────────────────────────────
    # WFDB units are per-channel; ECGRecord has a single ``units`` field.
    # We normalise everything to µV.
    raw_units: list[str] = list(record.units)
    signals = _normalise_to_microvolts(signals, raw_units)

    # Patient / record metadata ───────────────────────────────────
    metadata: dict[str, object] = {}
    if record.comments:
        metadata["comments"] = record.comments
    # Store the base-name of the record for traceability.
    metadata["record_name"] = record.record_name

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=sig_names,
        source_format="wfdb",
        units="µV",
        patient_metadata=metadata if metadata else None,
    )


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_UNIT_TO_UV_FACTOR: dict[str, float] = {
    "uv":  1.0,
    "µv":  1.0,
    "mv":  1e3,
    "v":   1e6,
}


def _normalise_to_microvolts(
    signals: np.ndarray,
    units: list[str],
) -> np.ndarray:
    """Scale each lead so that the output is in micro-volts (µV).

    If a unit string is not recognised (e.g. ``"bpm"`` for a non-ECG
    channel), the signal is left unchanged and a factor of 1.0 is used.
    """
    result = signals.copy()
    for idx, unit_str in enumerate(units):
        factor = _UNIT_TO_UV_FACTOR.get(unit_str.strip().lower(), 1.0)
        if factor != 1.0:
            result[idx] *= factor
    return result
