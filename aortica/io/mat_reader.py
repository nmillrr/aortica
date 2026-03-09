"""MATLAB ``.mat`` format reader.

Loads ECG recordings stored as MATLAB ``.mat`` files and returns
:class:`~aortica.io.ecg_record.ECGRecord` instances.

Uses :func:`scipy.io.loadmat` — ``scipy`` is already a core dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import io as sio

from aortica.io.ecg_record import ECGRecord


@dataclass
class MATConfig:
    """Configuration for reading ECG data from a ``.mat`` file.

    Attributes:
        sample_rate: Sampling rate in Hz (required).
        signal_variable: Name of the MATLAB variable containing the signal
            matrix.  The matrix should be 2-D; its orientation is controlled
            by *orientation*.
        lead_names_variable: Optional variable name containing lead name
            strings.  If *None*, generic names (``Lead_0``, …) are used.
        sample_rate_variable: Optional variable name containing the sample
            rate.  Overrides *sample_rate* if present in the file.
        units: Amplitude units of the signal data (default ``"µV"``).
        orientation: ``"leads_first"`` means shape ``[leads, samples]``
            (default); ``"samples_first"`` means ``[samples, leads]`` and
            will be transposed.
    """

    sample_rate: float
    signal_variable: str = "signals"
    lead_names_variable: Optional[str] = None
    sample_rate_variable: Optional[str] = None
    units: str = "µV"
    orientation: str = "leads_first"  # or "samples_first"
    patient_metadata: Optional[dict[str, object]] = field(default=None)


def read_mat(path: str | Path, config: MATConfig) -> ECGRecord:
    """Read an ECG recording from a MATLAB ``.mat`` file.

    Parameters
    ----------
    path:
        Path to the ``.mat`` file.
    config:
        A :class:`MATConfig` describing which variables to extract.

    Returns
    -------
    ECGRecord
        A standardised ECG record.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the specified signal variable is not found in the file.
    ValueError
        If the signal data is not a 2-D numeric array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat_data = sio.loadmat(str(path))

    # ── Signal data ──────────────────────────────────────────────
    if config.signal_variable not in mat_data:
        available = [k for k in mat_data if not k.startswith("__")]
        raise KeyError(
            f"Variable '{config.signal_variable}' not found in MAT file. "
            f"Available variables: {available}"
        )

    signals: np.ndarray = np.asarray(mat_data[config.signal_variable], dtype=np.float64)

    if signals.ndim == 1:
        # Single-lead recording — promote to [1, samples].
        signals = signals.reshape(1, -1)
    elif signals.ndim != 2:
        raise ValueError(
            f"Signal variable must be a 1-D or 2-D array, got shape {signals.shape}"
        )

    if config.orientation == "samples_first":
        signals = signals.T  # [samples, leads] → [leads, samples]

    # ── Sample rate ──────────────────────────────────────────────
    sample_rate = config.sample_rate
    if config.sample_rate_variable and config.sample_rate_variable in mat_data:
        sr_val = mat_data[config.sample_rate_variable]
        sample_rate = float(np.squeeze(sr_val))

    # ── Lead names ───────────────────────────────────────────────
    lead_names: list[str]
    if config.lead_names_variable and config.lead_names_variable in mat_data:
        raw_names = mat_data[config.lead_names_variable]
        lead_names = _extract_lead_names(raw_names, signals.shape[0])
    else:
        lead_names = [f"Lead_{i}" for i in range(signals.shape[0])]

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="mat",
        units=config.units,
        patient_metadata=config.patient_metadata,
    )


def _extract_lead_names(raw: object, expected: int) -> list[str]:
    """Best-effort extraction of lead name strings from a MAT variable.

    MATLAB cell arrays of strings are returned by ``loadmat`` in various
    nested ndarray forms.  This helper tries the common patterns:

    * 1-D array of ``str``.
    * 1-D/2-D object array whose elements are ``str`` or single-element
      ndarrays containing a ``str``.
    """
    arr = np.asarray(raw).flatten()
    names: list[str] = []
    for item in arr:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, np.ndarray):
            names.append(str(item.flat[0]))
        else:
            names.append(str(item))

    if len(names) != expected:
        return [f"Lead_{i}" for i in range(expected)]
    return names
