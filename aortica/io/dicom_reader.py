"""DICOM ECG waveform reader.

Loads ECG recordings from DICOM waveform objects (Supplement 30/130) and
returns :class:`~aortica.io.ecg_record.ECGRecord` instances.

Requires the ``pydicom`` package (``pip install pydicom``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

from aortica.io.ecg_record import ECGRecord


def read_dicom(path: str | Path) -> ECGRecord:
    """Read a DICOM ECG waveform file and return an :class:`ECGRecord`.

    Parameters
    ----------
    path:
        Path to a DICOM file containing ECG waveform data
        (Supplement 30 / 130).

    Returns
    -------
    ECGRecord
        A standardised ECG record with signals in physical units (µV).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the DICOM file contains no waveform sequence or no channel data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path))

    # ── Waveform sequence ────────────────────────────────────────
    if not hasattr(ds, "WaveformSequence") or len(ds.WaveformSequence) == 0:
        raise ValueError(
            f"DICOM file '{path}' contains no WaveformSequence."
        )

    # Use the first waveform group (typically the only one for 12-lead ECG).
    waveform = ds.WaveformSequence[0]

    # ── Sampling rate ────────────────────────────────────────────
    sample_rate = float(waveform.SamplingFrequency)

    # ── Number of channels and samples ───────────────────────────
    num_channels: int = int(waveform.NumberOfWaveformChannels)
    num_samples_total: int = int(waveform.NumberOfWaveformSamples)

    # ── Channel definitions ──────────────────────────────────────
    channel_defs = waveform.ChannelDefinitionSequence
    lead_names: list[str] = []
    sensitivities: list[float] = []
    baselines: list[float] = []
    units_list: list[str] = []

    for ch_def in channel_defs:
        # Channel label — prefer ChannelLabel, fall back to source sequence
        label = getattr(ch_def, "ChannelLabel", None)
        if label is None:
            src_seq = getattr(ch_def, "ChannelSourceSequence", None)
            if src_seq and len(src_seq) > 0:
                label = getattr(src_seq[0], "CodeMeaning", f"Ch{len(lead_names)}")
            else:
                label = f"Ch{len(lead_names)}"
        lead_names.append(str(label))

        # Sensitivity (digital-to-physical conversion factor)
        sensitivity = float(getattr(ch_def, "ChannelSensitivity", 1.0))
        correction = float(
            getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0)
        )
        sensitivities.append(sensitivity * correction)

        # Baseline
        baselines.append(float(getattr(ch_def, "ChannelBaseline", 0.0)))

        # Units
        unit_seq = getattr(ch_def, "ChannelSensitivityUnitsSequence", None)
        if unit_seq and len(unit_seq) > 0:
            unit_meaning = getattr(unit_seq[0], "CodeMeaning", "uV")
        else:
            unit_meaning = "uV"
        units_list.append(str(unit_meaning))

    # ── Decode waveform data ─────────────────────────────────────
    raw_data = np.frombuffer(
        waveform.WaveformData,
        dtype=_waveform_dtype(waveform),
    )

    # Reshape to [samples, channels]
    raw_data = raw_data.reshape(num_samples_total, num_channels)

    # Convert to physical units: physical = (digital + baseline) * sensitivity
    # Note: DICOM stores baseline differently depending on implementation.
    # The common formula is: physical = (digital - baseline) * sensitivity
    # but many implementations use: physical = digital * sensitivity.
    # We follow the standard: physical = digital * sensitivity + baseline_offset
    signals = np.empty((num_channels, num_samples_total), dtype=np.float64)
    for ch_idx in range(num_channels):
        signals[ch_idx] = (
            raw_data[:, ch_idx].astype(np.float64) * sensitivities[ch_idx]
            + baselines[ch_idx]
        )

    # Normalise units to µV
    signals = _normalise_to_microvolts(signals, units_list)

    # ── Patient metadata ─────────────────────────────────────────
    metadata: dict[str, object] = {}
    if hasattr(ds, "PatientName") and ds.PatientName:
        metadata["patient_name"] = str(ds.PatientName)
    if hasattr(ds, "PatientID") and ds.PatientID:
        metadata["patient_id"] = str(ds.PatientID)
    if hasattr(ds, "PatientSex") and ds.PatientSex:
        metadata["patient_sex"] = str(ds.PatientSex)
    if hasattr(ds, "PatientBirthDate") and ds.PatientBirthDate:
        metadata["patient_birth_date"] = str(ds.PatientBirthDate)
    if hasattr(ds, "StudyDate") and ds.StudyDate:
        metadata["study_date"] = str(ds.StudyDate)
    if hasattr(ds, "StudyDescription") and ds.StudyDescription:
        metadata["study_description"] = str(ds.StudyDescription)

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="dicom",
        units="µV",
        patient_metadata=metadata if metadata else None,
    )


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_UNIT_TO_UV_FACTOR: dict[str, float] = {
    "uv": 1.0,
    "µv": 1.0,
    "microvolt": 1.0,
    "microvolts": 1.0,
    "mv": 1e3,
    "millivolt": 1e3,
    "millivolts": 1e3,
    "v": 1e6,
    "volt": 1e6,
    "volts": 1e6,
}


def _normalise_to_microvolts(
    signals: np.ndarray,
    units: list[str],
) -> np.ndarray:
    """Scale each channel so that the output is in micro-volts (µV)."""
    result = signals.copy()
    for idx, unit_str in enumerate(units):
        factor = _UNIT_TO_UV_FACTOR.get(unit_str.strip().lower(), 1.0)
        if factor != 1.0:
            result[idx] *= factor
    return result


def _waveform_dtype(waveform: pydicom.Dataset) -> np.dtype:  # type: ignore[type-arg]
    """Determine the numpy dtype for the waveform data."""
    bits = int(waveform.WaveformBitsAllocated)
    sample_interp = getattr(waveform, "WaveformSampleInterpretation", "SS")
    sample_interp = str(sample_interp)

    dtype_map: dict[tuple[int, str], np.dtype] = {  # type: ignore[type-arg]
        (8, "SB"): np.dtype("int8"),
        (8, "UB"): np.dtype("uint8"),
        (8, "SS"): np.dtype("int8"),
        (16, "SS"): np.dtype("<i2"),
        (16, "US"): np.dtype("<u2"),
        (32, "SS"): np.dtype("<i4"),
        (32, "SL"): np.dtype("<i4"),
        (32, "US"): np.dtype("<u4"),
        (32, "UL"): np.dtype("<u4"),
    }
    key = (bits, sample_interp)
    if key not in dtype_map:
        raise ValueError(
            f"Unsupported waveform format: {bits} bits, interpretation '{sample_interp}'"
        )
    return dtype_map[key]
