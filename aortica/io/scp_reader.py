"""SCP-ECG format reader.

Loads ECG recordings in SCP-ECG (Standard Communications Protocol for
computer-assisted Electrocardiography) format and returns
:class:`~aortica.io.ecg_record.ECGRecord` instances.

SCP-ECG is a binary format defined by EN 1064 / ISO 11073-91064.  This
reader implements a minimal parser that extracts waveform data and basic
metadata from the file structure.

The format is organised into numbered sections:
- Section 0: Pointers to other sections
- Section 1: Patient/header data
- Section 3: Lead definitions
- Section 6: Rhythm (long-term) or short-term ECG data

This implementation focuses on extracting uncompressed waveform data
(Section 6) along with lead info (Section 3) and patient data (Section 1).
Huffman-compressed data (Section 2 codebooks) is not yet supported.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from aortica.io.ecg_record import ECGRecord

# Standard 12-lead ECG lead codes per SCP-ECG / AHA conventions
_LEAD_CODE_MAP: dict[int, str] = {
    0: "Unknown",
    1: "I",
    2: "II",
    3: "V1",
    4: "V2",
    5: "V3",
    6: "V4",
    7: "V5",
    8: "V6",
    9: "V7",
    10: "V8",
    11: "V9",
    12: "V3R",
    13: "V4R",
    14: "V5R",
    15: "V6R",
    16: "V7R",
    17: "III",
    18: "aVR",
    19: "aVL",
    20: "aVF",
}


def read_scp(path: str | Path) -> ECGRecord:
    """Read an SCP-ECG file and return an :class:`ECGRecord`.

    Parameters
    ----------
    path:
        Path to an SCP-ECG file.

    Returns
    -------
    ECGRecord
        A standardised ECG record with signals in physical units (µV).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as SCP-ECG or contains no waveform data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SCP-ECG file not found: {path}")

    data = path.read_bytes()
    if len(data) < 6:
        raise ValueError(f"File too small to be SCP-ECG: {path}")

    # ── Parse global header ──────────────────────────────────────
    # Bytes 0-1: CRC (uint16 LE)
    # Bytes 2-5: Record size (uint32 LE)
    _crc = struct.unpack_from("<H", data, 0)[0]
    record_size = struct.unpack_from("<I", data, 2)[0]

    if record_size > len(data):
        # Some files have inaccurate sizes — use actual file length
        record_size = len(data)

    # ── Parse section pointers (Section 0) ───────────────────────
    sections = _parse_section_pointers(data)

    # ── Section 1: Patient/header data ───────────────────────────
    metadata: dict[str, object] = {}
    if 1 in sections:
        sec_offset, sec_len = sections[1]
        metadata = _parse_section1(data, sec_offset, sec_len)

    # ── Section 3: Lead definitions ──────────────────────────────
    num_leads = 0
    lead_names: list[str] = []
    sample_rate = 500.0  # default
    if 3 in sections:
        sec_offset, sec_len = sections[3]
        lead_info = _parse_section3(data, sec_offset, sec_len)
        num_leads = lead_info["num_leads"]
        lead_names = lead_info["lead_names"]

    # ── Section 6: Rhythm / waveform data ────────────────────────
    if 6 not in sections:
        raise ValueError(f"SCP-ECG file '{path}' has no waveform data (Section 6 missing).")

    sec6_offset, sec6_len = sections[6]
    waveform_info = _parse_section6(data, sec6_offset, sec6_len, num_leads)
    signals = waveform_info["signals"]
    if waveform_info.get("sample_rate"):
        sample_rate = waveform_info["sample_rate"]

    # If section 3 didn't provide lead names, generate generic ones
    actual_num_leads = signals.shape[0]
    if not lead_names or len(lead_names) != actual_num_leads:
        lead_names = [f"Lead_{i}" for i in range(actual_num_leads)]

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="scp-ecg",
        units="µV",
        patient_metadata=metadata if metadata else None,
    )


# ──────────────────────────────────────────────────────────────────
# Internal parsers
# ──────────────────────────────────────────────────────────────────


def _parse_section_pointers(data: bytes) -> dict[int, tuple[int, int]]:
    """Parse Section 0 to find section offsets and lengths.

    Section 0 starts right after the 6-byte global header.
    Each section 0 entry is 10 bytes:
      - section header (2 bytes section ID + 4 bytes length)
      followed by pointer entries.

    But Section 0 itself has a section header. The section header is
    16 bytes for every section (CRC 2 + ID 2 + Length 4 + Version 1 +
    Protocol 1 + reserved 6).

    Each pointer record in section 0 body is 10 bytes:
      - Section ID (uint16 LE)
      - Record length (uint32 LE)
      - Index/offset (uint32 LE)
    """
    sections: dict[int, tuple[int, int]] = {}

    # Section 0 starts at byte 6 (after global header)
    sec0_start = 6
    if len(data) < sec0_start + 16:
        return sections

    # Read section 0 header
    sec0_len = struct.unpack_from("<I", data, sec0_start + 4)[0]
    body_start = sec0_start + 16  # skip 16-byte section header
    body_end = sec0_start + sec0_len

    if body_end > len(data):
        body_end = len(data)

    offset = body_start
    while offset + 10 <= body_end:
        sec_id = struct.unpack_from("<H", data, offset)[0]
        sec_len = struct.unpack_from("<I", data, offset + 2)[0]
        sec_index = struct.unpack_from("<I", data, offset + 6)[0]

        if sec_id > 0 and sec_len > 0 and sec_index > 0:
            # sec_index is 1-based byte offset from start of record
            actual_offset = sec_index - 1
            sections[sec_id] = (actual_offset, sec_len)

        offset += 10

    return sections


def _parse_section1(
    data: bytes, offset: int, length: int
) -> dict[str, object]:
    """Parse Section 1 (patient/header data) for demographic info."""
    metadata: dict[str, object] = {}

    # Skip 16-byte section header
    body_start = offset + 16
    body_end = offset + length
    if body_end > len(data):
        body_end = len(data)

    pos = body_start
    while pos + 3 <= body_end:
        tag = struct.unpack_from("<B", data, pos)[0]
        tag_len = struct.unpack_from("<H", data, pos + 1)[0]
        value_start = pos + 3
        value_end = value_start + tag_len

        if value_end > body_end:
            break

        if tag_len > 0:
            raw = data[value_start:value_end]
            # Tag 0: Last name
            if tag == 0:
                metadata["last_name"] = raw.rstrip(b"\x00").decode("latin-1", errors="replace")
            # Tag 1: First name
            elif tag == 1:
                metadata["first_name"] = raw.rstrip(b"\x00").decode("latin-1", errors="replace")
            # Tag 2: Patient ID
            elif tag == 2:
                metadata["patient_id"] = raw.rstrip(b"\x00").decode("latin-1", errors="replace")
            # Tag 8: Sex (1=M, 2=F, 0=unknown)
            elif tag == 8 and tag_len >= 1:
                sex_map = {0: "unknown", 1: "M", 2: "F"}
                metadata["sex"] = sex_map.get(raw[0], "unknown")

        pos = value_end

    return metadata


def _parse_section3(
    data: bytes, offset: int, length: int
) -> dict[str, Any]:
    """Parse Section 3 (lead definitions)."""
    info: dict[str, Any] = {"num_leads": 0, "lead_names": []}

    # Skip 16-byte section header
    body_start = offset + 16
    body_end = offset + length
    if body_end > len(data):
        body_end = len(data)

    if body_start + 1 > body_end:
        return info

    num_leads = struct.unpack_from("<B", data, body_start)[0]
    info["num_leads"] = num_leads

    # Flags byte at body_start + 1
    # Each lead entry is 9 bytes starting at body_start + 2
    lead_names: list[str] = []
    entry_start = body_start + 2
    for i in range(num_leads):
        entry_offset = entry_start + i * 9
        if entry_offset + 9 > body_end:
            break
        # Bytes 0-3: start sample (uint32)
        # Bytes 4-7: end sample (uint32)
        # Byte 8: lead ID
        lead_id = struct.unpack_from("<B", data, entry_offset + 8)[0]
        lead_name = _LEAD_CODE_MAP.get(lead_id, f"Lead_{lead_id}")
        lead_names.append(lead_name)

    info["lead_names"] = lead_names
    return info


def _parse_section6(
    data: bytes, offset: int, length: int, expected_leads: int
) -> dict[str, Any]:
    """Parse Section 6 (rhythm / waveform data).

    Section 6 body layout (after 16-byte section header):
    - 2 bytes: AVM (amplitude value multiplier) in nanovolts
    - 2 bytes: sample time interval in microseconds
    - 1 byte: encoding type (0 = real data)
    - 1 byte: compression type (0 = no compression)
    Then per-lead data follows.
    """
    info: dict[str, Any] = {}

    body_start = offset + 16
    body_end = offset + length
    if body_end > len(data):
        body_end = len(data)

    if body_start + 6 > body_end:
        raise ValueError("Section 6 body too short.")

    # AVM: amplitude value multiplier, in nanovolts per LSB
    avm_nv = struct.unpack_from("<H", data, body_start)[0]
    if avm_nv == 0:
        avm_nv = 1000  # default: 1 µV per LSB

    # Convert nV to µV multiplier
    avm_uv = avm_nv / 1000.0

    # Sample time interval in microseconds
    sample_interval_us = struct.unpack_from("<H", data, body_start + 2)[0]
    if sample_interval_us > 0:
        info["sample_rate"] = 1_000_000.0 / sample_interval_us

    # Encoding & compression bytes
    _encoding = struct.unpack_from("<B", data, body_start + 4)[0]
    _compression = struct.unpack_from("<B", data, body_start + 5)[0]

    # Waveform data starts after the 6-byte sub-header
    waveform_start = body_start + 6

    # Data is int16 LE, interleaved by lead
    raw = data[waveform_start:body_end]
    samples_array = np.frombuffer(raw, dtype="<i2")

    # Determine number of leads
    num_leads = expected_leads if expected_leads > 0 else 1

    if len(samples_array) == 0:
        raise ValueError("Section 6 contains no waveform samples.")

    # Try to reshape — data is typically interleaved [sample0_ch0, sample0_ch1, ...]
    total_samples = len(samples_array)
    samples_per_lead = total_samples // num_leads

    if samples_per_lead * num_leads != total_samples:
        # Try single lead interpretation
        num_leads = 1
        samples_per_lead = total_samples

    # Reshape to [samples, leads] then transpose to [leads, samples]
    reshaped = samples_array[: samples_per_lead * num_leads].reshape(
        samples_per_lead, num_leads
    )
    signals = reshaped.T.astype(np.float64) * avm_uv

    info["signals"] = signals
    return info
