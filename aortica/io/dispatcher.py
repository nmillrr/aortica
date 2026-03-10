"""Universal ECG format dispatcher.

Provides :func:`read_ecg`, a single entry‑point that auto‑detects the file
format (by extension and/or magic bytes) and dispatches to the correct
reader.  The result is a normalised :class:`ECGRecord` with consistent lead
ordering and an optional resample to a target sample rate.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aortica.io.csv_reader import CSVConfig, read_csv
from aortica.io.dicom_reader import read_dicom
from aortica.io.ecg_record import ECGRecord
from aortica.io.hl7_aecg_reader import read_hl7_aecg
from aortica.io.mat_reader import MATConfig, read_mat
from aortica.io.scp_reader import read_scp
from aortica.io.wfdb_reader import read_wfdb


# ── Standard 12‑lead ordering ────────────────────────────────────────

STANDARD_12_LEAD_ORDER: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# ── Supported format identifiers ─────────────────────────────────────

SUPPORTED_FORMATS: set[str] = {
    "wfdb", "csv", "mat", "dicom", "scp", "hl7_aecg", "xml",
}


# ── Exception ─────────────────────────────────────────────────────────

class UnsupportedFormatError(Exception):
    """Raised when the ECG file format cannot be detected or is not supported."""


# ── Extension → format mapping ────────────────────────────────────────

_EXTENSION_MAP: dict[str, str] = {
    ".hea": "wfdb",
    ".dat": "wfdb",
    ".csv": "csv",
    ".tsv": "csv",
    ".mat": "mat",
    ".dcm": "dicom",
    ".dicom": "dicom",
    ".scp": "scp",
    ".xml": "xml",   # further refined by magic‑byte sniffing
}


# ── Magic‑byte sniffing ──────────────────────────────────────────────

def _sniff_format(path: Path) -> Optional[str]:
    """Try to determine format from the first bytes of a file.

    Returns a format string or ``None`` if no match is found.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(512)
    except OSError:
        return None

    if not header:
        return None

    # DICOM: bytes 128–131 should read "DICM"
    if len(header) >= 132 and header[128:132] == b"DICM":
        return "dicom"

    # SCP-ECG: starts with a 2-byte CRC followed by a 4-byte record length.
    # The record length (bytes 2–5, little-endian uint32) should be > 0 and
    # there is no simple "magic" header, but Section 0 starts at offset 6
    # with section_id == 0 and section_length > 0.  We check that the first
    # section_id byte (offset 6) is 0x00 and the next byte (section length
    # MSB) is small.  This is a heuristic – SCP files that don't match will
    # fall through to extension-based detection.
    if len(header) >= 10:
        try:
            _crc = struct.unpack_from("<H", header, 0)[0]  # noqa: F841
            rec_len = struct.unpack_from("<I", header, 2)[0]
            if 100 < rec_len < 100_000_000 and header[6] == 0:
                return "scp"
        except struct.error:
            pass

    # XML-based formats ─ look for HL7 aECG vs generic XML
    if header.lstrip()[:5] in (b"<?xml", b"<Anno", b"<hl7:", b"<urn:"):
        # Quick check for HL7 namespace
        header_str = header.decode("utf-8", errors="replace")
        if "urn:hl7-org:v3" in header_str or "AnnotatedECG" in header_str:
            return "hl7_aecg"
        return "xml"

    # WFDB: text header files typically start with a record line like
    # "record_name num_signals ..."
    # First line is ASCII text with spaces; not a reliable magic but helps.
    # We skip this in sniffing and rely on extension for WFDB.

    return None


# ── Lead normalisation ───────────────────────────────────────────────

def _normalise_leads(record: ECGRecord) -> ECGRecord:
    """Reorder leads to the standard 12-lead order when applicable.

    If the record has exactly 12 leads and all standard lead names are
    present (case-insensitive comparison), the leads are reordered to
    ``STANDARD_12_LEAD_ORDER``.  Otherwise the record is returned
    unchanged.
    """
    if record.num_leads != 12:
        return record

    # Build a case-insensitive lookup: upper(name) → index
    upper_to_idx: dict[str, int] = {
        name.upper(): idx for idx, name in enumerate(record.lead_names)
    }

    # Check that all 12 standard leads are present
    try:
        indices = [upper_to_idx[name.upper()] for name in STANDARD_12_LEAD_ORDER]
    except KeyError:
        return record  # not all standard leads present

    return ECGRecord(
        signals=record.signals[indices, :].copy(),
        sample_rate=record.sample_rate,
        lead_names=list(STANDARD_12_LEAD_ORDER),
        duration_seconds=record.duration_seconds,
        patient_metadata=dict(record.patient_metadata)
        if record.patient_metadata
        else None,
        source_format=record.source_format,
        units=record.units,
    )


# ── XML sub-dispatch ─────────────────────────────────────────────────

def _read_xml(path: Path) -> ECGRecord:
    """Attempt to read a generic ``.xml`` file.

    Currently the only supported XML dialect is HL7 aECG.  If the file
    cannot be parsed as HL7 aECG, ``UnsupportedFormatError`` is raised.
    """
    try:
        return read_hl7_aecg(str(path))
    except Exception as exc:
        raise UnsupportedFormatError(
            f"XML file '{path}' could not be parsed as HL7 aECG: {exc}"
        ) from exc


# ── Public API ────────────────────────────────────────────────────────

def read_ecg(
    path: str | Path,
    *,
    format: Optional[str] = None,  # noqa: A002 – shadows builtin intentionally
    target_rate: float = 500.0,
    resample: bool = True,
    csv_config: Optional[CSVConfig] = None,
    mat_config: Optional[MATConfig] = None,
    **kwargs: Any,
) -> ECGRecord:
    """Read an ECG file with automatic format detection.

    Parameters
    ----------
    path:
        Path to the ECG file.  For WFDB records supply either the ``.hea``
        file or the base path (without extension).
    format:
        Explicit format identifier.  One of ``"wfdb"``, ``"csv"``,
        ``"mat"``, ``"dicom"``, ``"scp"``, ``"hl7_aecg"``, ``"xml"``.
        When ``None`` (default), the format is auto-detected by
        extension and file magic bytes.
    target_rate:
        Target sample rate in Hz.  Default ``500.0``.
    resample:
        If ``True`` (default), the returned record is resampled to
        *target_rate*.  Set to ``False`` to keep the original rate.
    csv_config:
        Configuration for CSV reader (required when format is ``"csv"``
        and auto-detection is used for ``.csv`` files).
    mat_config:
        Configuration for MAT reader (optional).
    **kwargs:
        Additional keyword arguments forwarded to the format reader.

    Returns
    -------
    ECGRecord
        A normalised ECG record with consistent lead ordering
        (12-lead standard when applicable) and optionally resampled.

    Raises
    ------
    UnsupportedFormatError
        If the file format cannot be detected or is not supported.
    FileNotFoundError
        If *path* does not point to an existing file (for non-WFDB
        formats).
    """
    filepath = Path(path)

    # ── Resolve format ────────────────────────────────────────────
    detected_format = format

    if detected_format is None:
        # First try magic bytes
        if filepath.exists():
            detected_format = _sniff_format(filepath)

        # Fall back to extension
        if detected_format is None:
            ext = filepath.suffix.lower()
            detected_format = _EXTENSION_MAP.get(ext)

        # WFDB can be referenced without extension — check for .hea sibling
        if detected_format is None:
            hea_path = filepath.with_suffix(".hea")
            if hea_path.exists():
                detected_format = "wfdb"

    if detected_format is None:
        raise UnsupportedFormatError(
            f"Cannot detect format for '{path}'. "
            f"Provide an explicit format= parameter. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    # Validate explicit format
    if detected_format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format '{detected_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    # ── Dispatch ──────────────────────────────────────────────────
    record: ECGRecord

    if detected_format == "wfdb":
        # WFDB reader wants the base path (no extension)
        wfdb_path = str(filepath.with_suffix("")) if filepath.suffix else str(filepath)
        record = read_wfdb(wfdb_path, **kwargs)

    elif detected_format == "csv":
        if csv_config is None:
            raise ValueError(
                "csv_config is required for CSV files (sample_rate must be specified). "
                "Pass csv_config=CSVConfig(sample_rate=...) to read_ecg()."
            )
        record = read_csv(str(filepath), config=csv_config, **kwargs)

    elif detected_format == "mat":
        if mat_config is None:
            raise ValueError(
                "mat_config is required for MAT files (sample_rate must be specified). "
                "Pass mat_config=MATConfig(sample_rate=...) to read_ecg()."
            )
        record = read_mat(str(filepath), config=mat_config, **kwargs)

    elif detected_format == "dicom":
        record = read_dicom(str(filepath), **kwargs)

    elif detected_format == "scp":
        record = read_scp(str(filepath), **kwargs)

    elif detected_format in ("hl7_aecg",):
        record = read_hl7_aecg(str(filepath), **kwargs)

    elif detected_format == "xml":
        record = _read_xml(filepath)

    else:  # pragma: no cover – unreachable after validation
        raise UnsupportedFormatError(f"No reader for format '{detected_format}'")

    # ── Normalise leads ───────────────────────────────────────────
    record = _normalise_leads(record)

    # ── Resample ──────────────────────────────────────────────────
    if resample and record.sample_rate != target_rate:
        record = record.resample(target_rate)

    return record
