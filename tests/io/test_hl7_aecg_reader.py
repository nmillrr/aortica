"""Tests for the HL7 aECG XML format reader.

All tests use synthetically constructed XML files — no external downloads
or real patient data are required.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.io.hl7_aecg_reader import read_hl7_aecg

# HL7 v3 namespace
_NS = "urn:hl7-org:v3"


# =====================================================================
# Helpers — synthetic HL7 aECG XML builder
# =====================================================================


def _make_hl7_aecg(
    path: Path,
    signals: np.ndarray,
    sample_rate: float = 500.0,
    lead_codes: list[str] | None = None,
    patient_id: str | None = None,
    patient_name: str | None = None,
    patient_sex: str | None = None,
    patient_birth_date: str | None = None,
    scale_value: float = 1.0,
    origin_value: float = 0.0,
    scale_unit: str = "uV",
) -> Path:
    """Create a minimal HL7 aECG XML file for testing.

    Parameters
    ----------
    signals:
        Array of shape ``[leads, samples]`` with ECG signal data.
        If scale_value is 1.0 and origin_value is 0.0, values are stored
        directly as digits (assumed to be in the unit given by scale_unit).
    lead_codes:
        HL7 lead codes, e.g. ``["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II"]``.
    """
    num_leads, num_samples = signals.shape

    if lead_codes is None:
        standard = [
            "MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II", "MDC_ECG_LEAD_III",
            "MDC_ECG_LEAD_AVR", "MDC_ECG_LEAD_AVL", "MDC_ECG_LEAD_AVF",
            "MDC_ECG_LEAD_V1", "MDC_ECG_LEAD_V2", "MDC_ECG_LEAD_V3",
            "MDC_ECG_LEAD_V4", "MDC_ECG_LEAD_V5", "MDC_ECG_LEAD_V6",
        ]
        lead_codes = standard[:num_leads]

    # Convert signals to digit values (inverse of scale/origin transform)
    if scale_value != 0:
        digit_signals = (signals - origin_value) / scale_value
    else:
        digit_signals = signals

    # Root element
    root = ET.Element(f"{{{_NS}}}AnnotatedECG")

    # Patient demographics
    if patient_id or patient_name or patient_sex or patient_birth_date:
        clin_trial = ET.SubElement(root, f"{{{_NS}}}componentOf")
        clin_study = ET.SubElement(clin_trial, f"{{{_NS}}}clinicalTrial")
        subject_of = ET.SubElement(clin_study, f"{{{_NS}}}subjectOf")  # noqa: F841
        # Wrap in annotatedECG -> subject -> trialSubject path
        # Simpler: put demographics directly under root
        subject = ET.SubElement(root, f"{{{_NS}}}subject")
        trial_subject = ET.SubElement(subject, f"{{{_NS}}}trialSubject")

        if patient_id:
            id_el = ET.SubElement(trial_subject, f"{{{_NS}}}id")
            id_el.set("extension", patient_id)

        demo_person = ET.SubElement(trial_subject, f"{{{_NS}}}subjectDemographicPerson")

        if patient_name:
            name_el = ET.SubElement(demo_person, f"{{{_NS}}}name")
            name_el.text = patient_name

        if patient_sex:
            sex_el = ET.SubElement(demo_person, f"{{{_NS}}}administrativeGenderCode")
            sex_el.set("code", patient_sex)

        if patient_birth_date:
            birth_el = ET.SubElement(demo_person, f"{{{_NS}}}birthTime")
            birth_el.set("value", patient_birth_date)

    # Component → series → sequenceSet
    component = ET.SubElement(root, f"{{{_NS}}}component")
    series = ET.SubElement(component, f"{{{_NS}}}series")

    # effectiveTime
    eff_time = ET.SubElement(series, f"{{{_NS}}}effectiveTime")
    low = ET.SubElement(eff_time, f"{{{_NS}}}low")
    low.set("value", "20250101120000")

    seq_set = ET.SubElement(series, f"{{{_NS}}}sequenceSet")

    # Time-base component (holds the sample interval)
    time_comp = ET.SubElement(seq_set, f"{{{_NS}}}component")
    time_seq = ET.SubElement(time_comp, f"{{{_NS}}}sequence")
    time_val = ET.SubElement(time_seq, f"{{{_NS}}}value")
    increment = ET.SubElement(time_val, f"{{{_NS}}}increment")
    interval_s = 1.0 / sample_rate
    increment.set("value", str(interval_s))
    increment.set("unit", "s")

    # Lead components
    for i, code in enumerate(lead_codes):
        comp = ET.SubElement(seq_set, f"{{{_NS}}}component")
        seq = ET.SubElement(comp, f"{{{_NS}}}sequence")

        code_el = ET.SubElement(seq, f"{{{_NS}}}code")
        code_el.set("code", code)

        val_el = ET.SubElement(seq, f"{{{_NS}}}value")

        # Scale
        scale_el = ET.SubElement(val_el, f"{{{_NS}}}scale")
        scale_el.set("value", str(scale_value))
        scale_el.set("unit", scale_unit)

        # Origin
        origin_el = ET.SubElement(val_el, f"{{{_NS}}}origin")
        origin_el.set("value", str(origin_value))

        # Digits
        digits_el = ET.SubElement(val_el, f"{{{_NS}}}digits")
        digits_el.text = " ".join(str(int(v)) for v in digit_signals[i])

    # Write XML
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")
    return path


# =====================================================================
# Basic loading tests
# =====================================================================


class TestReadHL7aECGBasic:
    """Basic HL7 aECG loading."""

    def test_basic_3_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 3-lead HL7 aECG file."""
        signals = np.array(
            [[100, 200, 300, 400], [150, 250, 350, 450], [50, 100, 150, 200]],
            dtype=np.float64,
        )
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg.xml",
            signals,
            sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II", "MDC_ECG_LEAD_III"],
        )
        rec = read_hl7_aecg(xml_file)

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 3
        assert rec.num_samples == 4
        assert rec.sample_rate == 500.0
        assert rec.lead_names == ["I", "II", "III"]
        assert rec.source_format == "hl7_aecg"
        np.testing.assert_array_almost_equal(rec.signals, signals, decimal=0)

    def test_12_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 12-lead HL7 aECG file."""
        expected_leads = [
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ]
        signals = np.round(np.random.randn(12, 2500) * 100).astype(np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg12.xml", signals, sample_rate=500.0,
        )
        rec = read_hl7_aecg(xml_file)

        assert rec.num_leads == 12
        assert rec.num_samples == 2500
        assert rec.lead_names == expected_leads

    def test_sample_rate_preserved(self, tmp_path: Path) -> None:
        """Custom sample rate is correctly read."""
        signals = np.ones((2, 100), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_250.xml", signals, sample_rate=250.0,
            lead_codes=["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II"],
        )
        rec = read_hl7_aecg(xml_file)
        assert rec.sample_rate == 250.0

    def test_single_lead(self, tmp_path: Path) -> None:
        """Load a single-lead recording."""
        signals = np.array([[10, 20, 30, 40, 50]], dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_single.xml", signals, sample_rate=1000.0,
            lead_codes=["MDC_ECG_LEAD_II"],
        )
        rec = read_hl7_aecg(xml_file)

        assert rec.num_leads == 1
        assert rec.lead_names == ["II"]
        assert rec.sample_rate == 1000.0
        np.testing.assert_array_almost_equal(rec.signals, signals, decimal=0)

    def test_duration_computed(self, tmp_path: Path) -> None:
        """Duration is correctly computed from signals and sample rate."""
        signals = np.ones((1, 500), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_dur.xml", signals, sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I"],
        )
        rec = read_hl7_aecg(xml_file)
        assert rec.duration_seconds == pytest.approx(1.0)


# =====================================================================
# Scale and origin tests
# =====================================================================


class TestReadHL7aECGScaleOrigin:
    """Scale and origin conversion."""

    def test_scale_applied(self, tmp_path: Path) -> None:
        """Scale factor is applied to convert digits to physical values."""
        # Store raw digits = [10, 20, 30], with scale=2.0 → physical = [20, 40, 60]
        expected = np.array([[20, 40, 60]], dtype=np.float64)
        signals = expected  # _make_hl7_aecg will inverse-transform
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_scale.xml",
            signals,
            sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I"],
            scale_value=2.0,
        )
        rec = read_hl7_aecg(xml_file)
        np.testing.assert_array_almost_equal(rec.signals, expected, decimal=0)

    def test_origin_applied(self, tmp_path: Path) -> None:
        """Origin offset is applied to convert digits to physical values."""
        expected = np.array([[110, 120, 130]], dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_origin.xml",
            expected,
            sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I"],
            scale_value=1.0,
            origin_value=100.0,
        )
        rec = read_hl7_aecg(xml_file)
        np.testing.assert_array_almost_equal(rec.signals, expected, decimal=0)


# =====================================================================
# Metadata tests
# =====================================================================


class TestReadHL7aECGMetadata:
    """Patient demographic extraction."""

    def test_full_demographics(self, tmp_path: Path) -> None:
        """Patient ID, name, sex, and birth date are extracted."""
        signals = np.ones((1, 10), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_demo.xml",
            signals,
            lead_codes=["MDC_ECG_LEAD_I"],
            patient_id="SUBJ-001",
            patient_name="Doe John",
            patient_sex="M",
            patient_birth_date="19800115",
        )
        rec = read_hl7_aecg(xml_file)

        assert rec.patient_metadata is not None
        assert rec.patient_metadata["patient_id"] == "SUBJ-001"
        assert rec.patient_metadata["patient_name"] == "Doe John"
        assert rec.patient_metadata["patient_sex"] == "M"
        assert rec.patient_metadata["patient_birth_date"] == "19800115"

    def test_partial_demographics(self, tmp_path: Path) -> None:
        """Only patient ID present — other fields absent."""
        signals = np.ones((1, 10), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_pid.xml",
            signals,
            lead_codes=["MDC_ECG_LEAD_I"],
            patient_id="PAT-42",
        )
        rec = read_hl7_aecg(xml_file)

        assert rec.patient_metadata is not None
        assert rec.patient_metadata["patient_id"] == "PAT-42"
        assert "patient_name" not in rec.patient_metadata

    def test_no_demographics(self, tmp_path: Path) -> None:
        """File without any demographics → patient_metadata is None."""
        signals = np.ones((1, 10), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "ecg_nodemo.xml",
            signals,
            lead_codes=["MDC_ECG_LEAD_I"],
        )
        rec = read_hl7_aecg(xml_file)
        assert rec.patient_metadata is None


# =====================================================================
# Error handling tests
# =====================================================================


class TestReadHL7aECGErrors:
    """Error handling for the HL7 aECG reader."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_hl7_aecg("/nonexistent/ecg.xml")

    def test_invalid_xml(self, tmp_path: Path) -> None:
        """Non-XML file raises ValueError."""
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("this is not xml <><><>", encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot parse"):
            read_hl7_aecg(bad_file)

    def test_empty_xml(self, tmp_path: Path) -> None:
        """XML with no waveform data raises ValueError."""
        empty_file = tmp_path / "empty.xml"
        root = ET.Element(f"{{{_NS}}}AnnotatedECG")
        tree = ET.ElementTree(root)
        tree.write(str(empty_file), xml_declaration=True, encoding="UTF-8")

        with pytest.raises(ValueError, match="no.*waveform|no.*series|no.*sequenceSet"):
            read_hl7_aecg(empty_file)


# =====================================================================
# Integration: round-trip through ECGRecord
# =====================================================================


class TestHL7aECGProducesValidECGRecord:
    """Reader produces records that work with ECGRecord utility methods."""

    def test_resample(self, tmp_path: Path) -> None:
        signals = np.round(np.random.randn(2, 500) * 100).astype(np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "rt_resample.xml", signals, sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II"],
        )
        rec = read_hl7_aecg(xml_file)
        resampled = rec.resample(250.0)
        assert resampled.sample_rate == 250.0
        assert resampled.num_samples == 250

    def test_select_leads(self, tmp_path: Path) -> None:
        signals = np.ones((3, 100), dtype=np.float64)
        xml_file = _make_hl7_aecg(
            tmp_path / "rt_select.xml", signals, sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II", "MDC_ECG_LEAD_III"],
        )
        rec = read_hl7_aecg(xml_file)
        sub = rec.select_leads(["III", "I"])
        assert sub.lead_names == ["III", "I"]

    def test_to_millivolts(self, tmp_path: Path) -> None:
        signals = np.ones((1, 100), dtype=np.float64) * 1000  # 1000 µV
        xml_file = _make_hl7_aecg(
            tmp_path / "rt_mv.xml", signals, sample_rate=500.0,
            lead_codes=["MDC_ECG_LEAD_I"],
        )
        rec = read_hl7_aecg(xml_file)
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_array_almost_equal(mv.signals, np.ones((1, 100)), decimal=0)
