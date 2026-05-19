"""Tests for aortica.integration.dicom_sr — DICOM SR write-back (US-082)."""

from __future__ import annotations

import io
from typing import Any, Dict

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from aortica.integration.dicom_sr import (
    DICOMSROutput,
    _build_finding_items,
    _build_risk_items,
    _make_code_content_item,
    _make_code_sequence,
    _make_container_content_item,
    _make_num_content_item,
    _make_text_content_item,
    to_structured_report,
    to_structured_report_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_COMPREHENSIVE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"


@pytest.fixture
def sample_multi_task_output() -> Dict[str, Any]:
    """Sample multi-task output with a mix of findings."""
    return {
        "rhythm": {"AF": 0.95, "normal_sinus_rhythm": 0.02, "VT": 0.75},
        "structural": {"LVH": 0.80, "RVH": 0.10},
        "ischaemia": {"STEMI": 0.92, "hyperkalaemia": 0.15},
        "risk": {
            "mortality_1y": 0.35,
            "hf_hosp_12m": 0.12,
            "af_onset_12m": 0.88,
        },
    }


@pytest.fixture
def minimal_output() -> Dict[str, Any]:
    """Minimal output with only one finding."""
    return {
        "rhythm": {"AF": 0.95},
    }


@pytest.fixture
def empty_output() -> Dict[str, Any]:
    """Output with no findings above threshold."""
    return {
        "rhythm": {"AF": 0.01, "normal_sinus_rhythm": 0.02},
        "structural": {"LVH": 0.05},
        "ischaemia": {"STEMI": 0.01},
    }


@pytest.fixture
def original_sop_uid() -> str:
    return "1.2.840.113619.2.5.1762583153.215519.978957063.78"


# ---------------------------------------------------------------------------
# Test: to_structured_report produces valid Dataset
# ---------------------------------------------------------------------------


class TestToStructuredReport:
    """Tests for the main to_structured_report function."""

    def test_returns_dataset(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert isinstance(ds, Dataset)

    def test_sop_class_uid(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.SOPClassUID == _COMPREHENSIVE_SR_SOP_CLASS

    def test_modality_is_sr(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.Modality == "SR"

    def test_patient_module(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(
            sample_multi_task_output,
            patient_name="DOE^JOHN",
            patient_id="PAT123",
        )
        assert str(ds.PatientName) == "DOE^JOHN"
        assert ds.PatientID == "PAT123"

    def test_default_patient(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert str(ds.PatientName) == "ANONYMOUS"
        assert ds.PatientID == "ANONYMOUS"

    def test_study_instance_uid_generated(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.StudyInstanceUID is not None
        assert len(ds.StudyInstanceUID) > 0

    def test_study_instance_uid_custom(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        custom_uid = "1.2.3.4.5.6.7.8.9"
        ds = to_structured_report(
            sample_multi_task_output, study_instance_uid=custom_uid
        )
        assert ds.StudyInstanceUID == custom_uid

    def test_series_instance_uid(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.SeriesInstanceUID is not None

    def test_completion_flag(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.CompletionFlag == "COMPLETE"

    def test_verification_flag(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.VerificationFlag == "UNVERIFIED"

    def test_root_container_value_type(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.ValueType == "CONTAINER"

    def test_root_concept_name(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        cn = ds.ConceptNameCodeSequence[0]
        assert cn.CodeValue == "126000"
        assert cn.CodeMeaning == "Imaging Report"

    def test_content_sequence_exists(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "ContentSequence")
        assert len(ds.ContentSequence) > 0

    def test_content_has_language(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        lang_item = ds.ContentSequence[0]
        assert lang_item.ValueType == "CODE"
        assert lang_item.RelationshipType == "HAS CONCEPT MOD"
        cn = lang_item.ConceptNameCodeSequence[0]
        assert cn.CodeMeaning == "Language of Content Item and Descendants"

    def test_content_has_procedure(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        proc_item = ds.ContentSequence[1]
        assert proc_item.ValueType == "CODE"
        code = proc_item.ConceptCodeSequence[0]
        assert code.CodeValue == "11524-6"
        assert code.CodeMeaning == "EKG study"

    def test_manufacturer(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert ds.Manufacturer == "Aortica"

    def test_file_meta_present(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "file_meta")
        assert ds.file_meta.MediaStorageSOPClassUID == _COMPREHENSIVE_SR_SOP_CLASS


# ---------------------------------------------------------------------------
# Test: Referenced SOP Instance UID
# ---------------------------------------------------------------------------


class TestReferencedSOP:
    """Tests for original DICOM ECG reference."""

    def test_reference_included(
        self,
        sample_multi_task_output: Dict[str, Any],
        original_sop_uid: str,
    ) -> None:
        ds = to_structured_report(
            sample_multi_task_output, original_dicom_ref=original_sop_uid
        )
        # Find the IMAGE content item
        image_items = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "IMAGE"
        ]
        assert len(image_items) == 1
        ref = image_items[0].ReferencedSOPSequence[0]
        assert ref.ReferencedSOPInstanceUID == original_sop_uid

    def test_no_reference_when_none(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        image_items = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "IMAGE"
        ]
        assert len(image_items) == 0


# ---------------------------------------------------------------------------
# Test: Findings content items
# ---------------------------------------------------------------------------


class TestFindings:
    """Tests for classification finding content items."""

    def test_findings_section_present(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        containers = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "CONTAINER"
            and item.ConceptNameCodeSequence[0].CodeValue == "121070"
        ]
        assert len(containers) == 1

    def test_finding_count(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        # AF=0.95, VT=0.75, LVH=0.80, STEMI=0.92 are above 0.30
        items = _build_finding_items(sample_multi_task_output, 0.30)
        assert len(items) == 4

    def test_finding_has_code_and_confidence(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        items = _build_finding_items(sample_multi_task_output, 0.30)
        # First finding should be AF
        af_container = items[0]
        assert af_container.ValueType == "CONTAINER"
        children = af_container.ContentSequence
        # CODE item (finding), NUM item (confidence), TEXT item (comment)
        assert len(children) == 3
        code_item = children[0]
        assert code_item.ValueType == "CODE"
        assert code_item.ConceptCodeSequence[0].CodeValue == "49436004"

        num_item = children[1]
        assert num_item.ValueType == "NUM"

    def test_threshold_filters_findings(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        items_low = _build_finding_items(sample_multi_task_output, 0.10)
        items_high = _build_finding_items(sample_multi_task_output, 0.90)
        assert len(items_low) > len(items_high)

    def test_empty_findings(
        self, empty_output: Dict[str, Any]
    ) -> None:
        items = _build_finding_items(empty_output, 0.30)
        assert len(items) == 0

    def test_no_findings_section_when_empty(
        self, empty_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(empty_output)
        containers = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "CONTAINER"
            and item.ConceptNameCodeSequence[0].CodeValue == "121070"
        ]
        assert len(containers) == 0

    def test_unknown_class_gets_local_code(self) -> None:
        output = {"rhythm": {"some_unknown_class": 0.85}}
        items = _build_finding_items(output, 0.30)
        assert len(items) == 1
        code_item = items[0].ContentSequence[0]
        assert code_item.ConceptCodeSequence[0].CodingSchemeDesignator == "99AORTICA"


# ---------------------------------------------------------------------------
# Test: Risk content items
# ---------------------------------------------------------------------------


class TestRiskItems:
    """Tests for risk score content items."""

    def test_risk_section_present(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        containers = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "CONTAINER"
            and item.ConceptNameCodeSequence[0].CodeValue == "121073"
        ]
        assert len(containers) == 1

    def test_risk_count(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        items = _build_risk_items(sample_multi_task_output)
        assert len(items) == 3  # mortality, hf_hosp, af_onset

    def test_risk_item_structure(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        items = _build_risk_items(sample_multi_task_output)
        risk_container = items[0]
        assert risk_container.ValueType == "CONTAINER"
        children = risk_container.ContentSequence
        assert len(children) == 2  # NUM + TEXT

    def test_no_risk_section_when_absent(self) -> None:
        output: Dict[str, Any] = {"rhythm": {"AF": 0.95}}
        ds = to_structured_report(output)
        containers = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "CONTAINER"
            and item.ConceptNameCodeSequence[0].CodeValue == "121073"
        ]
        assert len(containers) == 0


# ---------------------------------------------------------------------------
# Test: Conclusion
# ---------------------------------------------------------------------------


class TestConclusion:
    """Tests for conclusion content item."""

    def test_conclusion_present(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        text_items = [
            item
            for item in ds.ContentSequence
            if item.ValueType == "TEXT"
            and item.ConceptNameCodeSequence[0].CodeValue == "121076"
        ]
        assert len(text_items) == 1
        assert "clinician review" in text_items[0].TextValue.lower()


# ---------------------------------------------------------------------------
# Test: DICOMSROutput wrapper
# ---------------------------------------------------------------------------


class TestDICOMSROutput:
    """Tests for the to_structured_report_output wrapper."""

    def test_returns_output_object(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        output = to_structured_report_output(sample_multi_task_output)
        assert isinstance(output, DICOMSROutput)
        assert isinstance(output.dataset, Dataset)

    def test_finding_count(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        output = to_structured_report_output(sample_multi_task_output)
        assert output.finding_count == 4

    def test_risk_count(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        output = to_structured_report_output(sample_multi_task_output)
        assert output.risk_count == 3

    def test_sop_instance_uid(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        output = to_structured_report_output(sample_multi_task_output)
        assert len(output.sop_instance_uid) > 0

    def test_with_reference(
        self,
        sample_multi_task_output: Dict[str, Any],
        original_sop_uid: str,
    ) -> None:
        output = to_structured_report_output(
            sample_multi_task_output,
            original_dicom_ref=original_sop_uid,
        )
        assert isinstance(output.dataset, Dataset)


# ---------------------------------------------------------------------------
# Test: DICOM conformance (IOD modules present)
# ---------------------------------------------------------------------------


class TestDICOMConformance:
    """Verify correct DICOM IOD modules are present."""

    def test_patient_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "PatientName")
        assert hasattr(ds, "PatientID")

    def test_study_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "StudyInstanceUID")
        assert hasattr(ds, "StudyDate")
        assert hasattr(ds, "StudyTime")
        assert hasattr(ds, "AccessionNumber")

    def test_series_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "SeriesInstanceUID")
        assert hasattr(ds, "SeriesNumber")
        assert ds.Modality == "SR"

    def test_equipment_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "Manufacturer")
        assert hasattr(ds, "ManufacturerModelName")

    def test_sop_common_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "SOPClassUID")
        assert hasattr(ds, "SOPInstanceUID")
        assert hasattr(ds, "SpecificCharacterSet")

    def test_sr_document_general_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "CompletionFlag")
        assert hasattr(ds, "VerificationFlag")
        assert hasattr(ds, "ContentDate")
        assert hasattr(ds, "ContentTime")

    def test_sr_content_module_tags(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        ds = to_structured_report(sample_multi_task_output)
        assert hasattr(ds, "ValueType")
        assert hasattr(ds, "ConceptNameCodeSequence")
        assert hasattr(ds, "ContentSequence")
        assert hasattr(ds, "ContinuityOfContent")

    def test_dataset_serializable(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        """Verify dataset can be written to bytes (valid DICOM)."""
        ds = to_structured_report(sample_multi_task_output)
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, write_like_original=False)
        buf.seek(0)
        data = buf.read()
        assert len(data) > 0

    def test_dataset_round_trip(
        self, sample_multi_task_output: Dict[str, Any]
    ) -> None:
        """Write and re-read the DICOM SR to verify integrity."""
        ds = to_structured_report(sample_multi_task_output)
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, write_like_original=False)
        buf.seek(0)
        ds2 = pydicom.dcmread(buf, force=True)
        assert ds2.SOPClassUID == _COMPREHENSIVE_SR_SOP_CLASS
        assert ds2.Modality == "SR"
        assert ds2.ValueType == "CONTAINER"
        assert len(ds2.ContentSequence) > 0


# ---------------------------------------------------------------------------
# Test: Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_make_code_sequence(self) -> None:
        seq = _make_code_sequence("12345", "SCT", "Test")
        assert len(seq) == 1
        assert seq[0].CodeValue == "12345"
        assert seq[0].CodingSchemeDesignator == "SCT"
        assert seq[0].CodeMeaning == "Test"

    def test_make_text_content_item(self) -> None:
        item = _make_text_content_item("111", "DCM", "Test", "Hello")
        assert item.ValueType == "TEXT"
        assert item.TextValue == "Hello"

    def test_make_code_content_item(self) -> None:
        item = _make_code_content_item(
            "111", "DCM", "Name", "222", "SCT", "Value"
        )
        assert item.ValueType == "CODE"
        assert item.ConceptCodeSequence[0].CodeValue == "222"

    def test_make_num_content_item(self) -> None:
        item = _make_num_content_item("111", "DCM", "Name", 0.95)
        assert item.ValueType == "NUM"
        mv = item.MeasuredValueSequence[0]
        assert float(mv.NumericValue) == pytest.approx(0.95, abs=0.001)

    def test_make_container_content_item(self) -> None:
        child = _make_text_content_item("111", "DCM", "Child", "text")
        container = _make_container_content_item(
            "222", "DCM", "Parent", [child]
        )
        assert container.ValueType == "CONTAINER"
        assert len(container.ContentSequence) == 1


# ---------------------------------------------------------------------------
# Test: List-based input (not just dict)
# ---------------------------------------------------------------------------


class TestListInput:
    """Test that list-based predictions work (not just dict)."""

    def test_list_rhythm_input(self) -> None:
        # 28 rhythm classes — set first (AF) high
        preds = [0.01] * 28
        preds[0] = 0.92  # AF
        output: Dict[str, Any] = {"rhythm": preds}
        items = _build_finding_items(output, 0.30)
        assert len(items) == 1

    def test_list_risk_input(self) -> None:
        output: Dict[str, Any] = {"risk": [0.35, 0.12, 0.88, 0.5, 0.2, 0.1]}
        items = _build_risk_items(output)
        assert len(items) == 6
