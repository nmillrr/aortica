"""HL7 aECG (Annotated ECG) XML format reader.

Loads ECG recordings from HL7 aECG XML files (FDA XML standard) and
returns :class:`~aortica.io.ecg_record.ECGRecord` instances.

The HL7 aECG format is an XML-based standard derived from the HL7 RIM,
mandated by the FDA for electronic ECG data submissions.  It stores waveform
data as space-separated digit strings within ``<digits>`` elements, grouped
by series and sequence (one sequence per lead).

No additional dependencies beyond the Python standard library's
:mod:`xml.etree.ElementTree` are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from aortica.io.ecg_record import ECGRecord

# HL7 v3 namespace — used in all aECG XML files.
_NS = "urn:hl7-org:v3"
_NSMAP = {"hl7": _NS}


def _find(el: ET.Element, xpath: str) -> ET.Element | None:
    """Namespace-aware ``find`` helper."""
    return el.find(xpath, _NSMAP)


def _findall(el: ET.Element, xpath: str) -> list[ET.Element]:
    """Namespace-aware ``findall`` helper."""
    return el.findall(xpath, _NSMAP)


def _findtext(el: ET.Element, xpath: str, default: str = "") -> str:
    """Namespace-aware ``findtext`` helper."""
    txt = el.findtext(xpath, default=default, namespaces=_NSMAP)
    return txt if txt is not None else default


# Standard HL7 aECG lead code → conventional name mapping.
_LEAD_CODE_MAP: dict[str, str] = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL",
    "MDC_ECG_LEAD_AVF": "aVF",
    "MDC_ECG_LEAD_V1": "V1",
    "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3",
    "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5",
    "MDC_ECG_LEAD_V6": "V6",
}


def read_hl7_aecg(path: str | Path) -> ECGRecord:
    """Read an HL7 aECG XML file and return an :class:`ECGRecord`.

    Parameters
    ----------
    path:
        Path to an HL7 aECG XML file.

    Returns
    -------
    ECGRecord
        A standardised ECG record with signals in µV.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the XML file cannot be parsed or contains no waveform data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HL7 aECG file not found: {path}")

    try:
        tree = ET.parse(path)  # noqa: S314
    except ET.ParseError as exc:
        raise ValueError(f"Cannot parse HL7 aECG XML: {exc}") from exc

    root = tree.getroot()

    # ── Patient demographics ─────────────────────────────────────
    metadata = _extract_demographics(root)

    # ── Find the first series containing waveform sequences ──────
    series_list = _findall(root, ".//hl7:series")
    if not series_list:
        # Also try without wrapper — some files embed <sequenceSet> at
        # a different depth.
        series_list = _findall(root, ".//hl7:sequenceSet/..")
        if not series_list:
            raise ValueError(
                "HL7 aECG file contains no <series> or <sequenceSet> elements."
            )

    # Use the first series that has component sequences with waveform data.
    lead_names: list[str] = []
    lead_signals: list[np.ndarray] = []
    sample_rate: float | None = None
    units: str = "µV"

    for series_el in series_list:
        lead_names, lead_signals, sample_rate, units = _parse_series(series_el)
        if lead_signals:
            break

    if not lead_signals:
        raise ValueError("HL7 aECG file contains no waveform data.")
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("Could not determine a valid sample rate from the file.")

    signals = np.array(lead_signals, dtype=np.float64)

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="hl7_aecg",
        units=units,
        patient_metadata=metadata if metadata else None,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _extract_demographics(root: ET.Element) -> dict[str, Any]:
    """Extract patient demographics from the HL7 aECG XML tree."""
    meta: dict[str, Any] = {}

    # Patient name
    _name_xpath = (
        ".//hl7:subjectOf/hl7:annotatedECG/hl7:subject"
        "/hl7:trialSubject/hl7:subjectDemographicPerson/hl7:name"
    )
    name_el = _find(root, _name_xpath)
    if name_el is None:
        # Try alternative paths common in aECG files.
        name_el = _find(root, ".//hl7:subject//hl7:name")
    if name_el is not None and name_el.text:
        meta["patient_name"] = name_el.text.strip()

    # Patient ID — look for trialSubject > id or subject > id
    id_el = _find(root, ".//hl7:trialSubject/hl7:id")
    if id_el is None:
        id_el = _find(root, ".//hl7:subject/hl7:trialSubject/hl7:id")
    if id_el is not None:
        ext = id_el.get("extension")
        if ext:
            meta["patient_id"] = ext

    # Sex
    sex_el = _find(root, ".//hl7:subjectDemographicPerson/hl7:administrativeGenderCode")
    if sex_el is not None:
        code = sex_el.get("code")
        if code:
            meta["patient_sex"] = code

    # Birth date
    birth_el = _find(root, ".//hl7:subjectDemographicPerson/hl7:birthTime")
    if birth_el is not None:
        val = birth_el.get("value")
        if val:
            meta["patient_birth_date"] = val

    return meta


def _parse_series(
    series_el: ET.Element,
) -> tuple[list[str], list[np.ndarray], float | None, str]:
    """Parse a single <series> element into lead data.

    Returns (lead_names, lead_signals, sample_rate, units).
    """
    lead_names: list[str] = []
    lead_signals: list[np.ndarray] = []
    sample_rate: float | None = None
    units: str = "µV"

    # effectiveTime in the series can tell us the timing, but we mainly
    # rely on the <increment> element inside sequenceSet for sample rate.

    # Locate the sequenceSet
    seq_set = _find(series_el, ".//hl7:sequenceSet")
    if seq_set is None:
        return lead_names, lead_signals, sample_rate, units

    # The sequenceSet has component elements. The first component often
    # holds the time base (increment), subsequent ones hold lead data.
    components = _findall(seq_set, "hl7:component")
    if not components:
        return lead_names, lead_signals, sample_rate, units

    for component in components:
        sequence = _find(component, "hl7:sequence")
        if sequence is None:
            continue

        # Check if this is a time-base sequence (has <increment> but no
        # <code> pointing to a lead).
        code_el = _find(sequence, "hl7:code")
        increment_el = _find(sequence, "hl7:value/hl7:increment")
        if increment_el is None:
            increment_el = _find(sequence, "hl7:increment")

        # Extract increment (sample interval) to determine sample rate.
        if increment_el is not None and sample_rate is None:
            inc_val = increment_el.get("value")
            inc_unit = increment_el.get("unit", "s")
            if inc_val:
                interval = float(inc_val)
                if inc_unit == "ms":
                    interval /= 1000.0
                if interval > 0:
                    sample_rate = 1.0 / interval

        # If there's a code, it's a lead sequence.
        if code_el is not None:
            lead_code = code_el.get("code", "")
            lead_name = _LEAD_CODE_MAP.get(lead_code, lead_code)

            # Try to get displayName attribute as fallback
            if not lead_name:
                lead_name = code_el.get("displayName", f"Lead_{len(lead_names)}")

            # Parse digit data
            digits_el = _find(sequence, "hl7:value/hl7:digits")
            if digits_el is None:
                digits_el = _find(sequence, "hl7:digits")

            if digits_el is not None and digits_el.text:
                data = _parse_digits(digits_el.text.strip())
                if data is not None and len(data) > 0:
                    lead_names.append(lead_name)
                    lead_signals.append(data)

                    # Check for scale/origin to determine units
                    origin_el = _find(sequence, "hl7:value/hl7:origin")
                    scale_el = _find(sequence, "hl7:value/hl7:scale")
                    _apply_scale_origin(
                        lead_signals, len(lead_signals) - 1,
                        origin_el, scale_el,
                    )
                    # Check units from scale element
                    if scale_el is not None:
                        u = scale_el.get("unit", "")
                        if u:
                            units = _normalise_unit_string(u)

    return lead_names, lead_signals, sample_rate, units


def _parse_digits(text: str) -> np.ndarray | None:
    """Parse a space-separated string of digits into a float64 array."""
    if not text:
        return None
    try:
        values = [float(v) for v in text.split()]
        return np.array(values, dtype=np.float64)
    except ValueError:
        return None


def _apply_scale_origin(
    signals: list[np.ndarray],
    idx: int,
    origin_el: ET.Element | None,
    scale_el: ET.Element | None,
) -> None:
    """Apply scale and origin to convert raw digits to physical values."""
    scale = 1.0
    origin = 0.0

    if scale_el is not None:
        val = scale_el.get("value")
        if val:
            scale = float(val)

    if origin_el is not None:
        val = origin_el.get("value")
        if val:
            origin = float(val)

    if scale != 1.0 or origin != 0.0:
        signals[idx] = signals[idx] * scale + origin


_UNIT_NORMALISATION: dict[str, str] = {
    "uv": "µV",
    "µv": "µV",
    "microvolt": "µV",
    "microvolts": "µV",
    "mv": "mV",
    "millivolt": "mV",
    "millivolts": "mV",
    "v": "V",
    "volt": "V",
    "volts": "V",
}


def _normalise_unit_string(unit: str) -> str:
    """Normalise a unit string to a standard form."""
    return _UNIT_NORMALISATION.get(unit.strip().lower(), unit)
