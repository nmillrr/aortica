"""DICOM Structured Report (SR) write-back for ECG AI findings.

Converts Aortica multi-task ECG predictions into a DICOM SR dataset
using the TID 2000 (Basic Diagnostic Imaging Report) template
structure.  Each positive classification finding is encoded as a
CONTENT ITEM with concept name, coded value, and confidence.  Risk
scores are encoded as numeric content items.

The generated SR references the original DICOM ECG instance via
Referenced SOP Instance UID so that AI findings are linked to the
source waveform in the PACS archive.

Requires ``pydicom`` (already a core dependency)::

    pip install pydicom
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import generate_uid

# ---------------------------------------------------------------------------
# DICOM SR UIDs and constants
# ---------------------------------------------------------------------------

# SOP Class UIDs
_COMPREHENSIVE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"
_TRANSFER_SYNTAX_EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
_ECG_WAVEFORM_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.1"

# Coding scheme designators
_DCM = "DCM"  # DICOM Content Mapping Resource
_SCT = "SCT"  # SNOMED CT
_LN = "LN"  # LOINC

# ---------------------------------------------------------------------------
# SNOMED CT code mappings (consistent with fhir.py / hl7v2.py)
# ---------------------------------------------------------------------------

_SNOMED_CODES: Dict[str, Tuple[str, str]] = {
    # Rhythm classes
    "AF": ("49436004", "Atrial fibrillation"),
    "AFL": ("5370000", "Atrial flutter"),
    "SVT": ("6456007", "Supraventricular tachycardia"),
    "AVNRT": ("251166008", "Atrioventricular nodal reentry tachycardia"),
    "AVRT": ("233897008", "Atrioventricular reentrant tachycardia"),
    "VT": ("25569003", "Ventricular tachycardia"),
    "VF": ("71908006", "Ventricular fibrillation"),
    "idioventricular": ("29320008", "Idioventricular rhythm"),
    "sinus_brady": ("49710005", "Sinus bradycardia"),
    "sinus_tachy": ("11092001", "Sinus tachycardia"),
    "PAC": ("284470004", "Premature atrial contraction"),
    "PVC": ("427172004", "Premature ventricular contraction"),
    "av_block_1st": ("270492004", "First degree atrioventricular block"),
    "av_block_2nd": ("195042002", "Second degree atrioventricular block"),
    "av_block_3rd": ("27885002", "Complete atrioventricular block"),
    "LBBB": ("63467002", "Left bundle branch block"),
    "RBBB": ("59118001", "Right bundle branch block"),
    "LAFB": ("445118002", "Left anterior fascicular block"),
    "LPFB": ("445211001", "Left posterior fascicular block"),
    "WPW": ("74390002", "Wolff-Parkinson-White pattern"),
    "pacemaker_rhythm": ("10370003", "Paced rhythm"),
    "normal_sinus_rhythm": ("426783006", "Normal sinus rhythm"),
    "brugada_pattern": ("418818005", "Brugada syndrome"),
    "short_QT_syndrome": ("699255007", "Short QT syndrome"),
    "CPVT": ("419671004", "Catecholaminergic polymorphic VT"),
    "fascicular_VT": ("426749004", "Fascicular ventricular tachycardia"),
    "atypical_atrial_flutter": ("5370000", "Atypical atrial flutter"),
    "inappropriate_sinus_tachy": ("11092001", "Inappropriate sinus tachycardia"),
    # Structural classes
    "LVH": ("55827005", "Left ventricular hypertrophy"),
    "RVH": ("44313006", "Right ventricular hypertrophy"),
    "LVSD": ("48867003", "Left ventricular systolic dysfunction"),
    "HFpEF_risk": ("446221000", "Heart failure with preserved ejection fraction"),
    "DCM": ("195021002", "Dilated cardiomyopathy"),
    "HCM": ("45227007", "Hypertrophic cardiomyopathy"),
    "ARVC": ("253528005", "Arrhythmogenic right ventricular cardiomyopathy"),
    "amyloidosis": ("17602002", "Cardiac amyloidosis"),
    "aortic_stenosis": ("60573004", "Aortic stenosis"),
    "mitral_regurgitation": ("48724000", "Mitral regurgitation"),
    "pulmonary_HTN": ("70995007", "Pulmonary hypertension"),
    "LA_enlargement": ("67751000", "Left atrial enlargement"),
    "RA_enlargement": ("67741000119100", "Right atrial enlargement"),
    "pericarditis": ("3238004", "Pericarditis"),
    "myocarditis": ("50920009", "Myocarditis"),
    "LV_strain_grade": ("55827005", "Left ventricular strain pattern"),
    "RV_strain_PE": ("59282003", "Right ventricular strain - pulmonary embolism"),
    "Takotsubo_pattern": ("423727003", "Takotsubo cardiomyopathy"),
    "infiltrative_cardiomyopathy_strain": (
        "415295002",
        "Infiltrative cardiomyopathy",
    ),
    # Ischaemia classes
    "STEMI": ("401303003", "ST elevation myocardial infarction"),
    "posterior_MI": ("73795002", "Posterior myocardial infarction"),
    "occlusive_NSTEMI": ("401314000", "Non-ST elevation myocardial infarction"),
    "old_MI": ("22298006", "Old myocardial infarction"),
    "hyperkalaemia": ("14140009", "Hyperkalaemia"),
    "hypokalaemia": ("43339004", "Hypokalaemia"),
    "hypercalcaemia": ("66931009", "Hypercalcaemia"),
    "hypothyroidism_pattern": ("40930008", "Hypothyroidism"),
    "digitalis_effect": ("13384006", "Digitalis effect"),
    "QTc_prolongation": ("111975006", "Prolonged QT interval"),
    "early_repol_vs_STEMI": ("428417006", "Early repolarization pattern"),
    "de_Winter_T_wave": ("840544004", "De Winter T-wave pattern"),
    "Wellens_syndrome": ("840546002", "Wellens syndrome"),
    "aVR_ST_elevation": ("401303003", "aVR ST elevation pattern"),
    "Sgarbossa_criteria": ("401303003", "Sgarbossa criteria positive"),
    "hyperkalaemia_severity_grade": ("14140009", "Hyperkalaemia severity grading"),
    "hypothermia_osborn_waves": ("386689009", "Hypothermia - Osborn waves"),
    "TCA_toxicity": ("212601003", "Tricyclic antidepressant toxicity"),
    "digoxin_effect_vs_toxicity": ("81060008", "Digoxin toxicity"),
}

# Risk output display names
_RISK_DISPLAY: Dict[str, str] = {
    "mortality_1y": "1-year all-cause mortality risk",
    "hf_hosp_12m": "12-month HF hospitalization probability",
    "af_onset_12m": "12-month AF onset risk",
    "ecg_predicted_ef": "ECG-predicted ejection fraction",
    "conduction_disease_trajectory": "Progressive conduction disease trajectory",
    "sudden_cardiac_death_risk": "Sudden cardiac death risk",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_code_sequence(
    value: str,
    designator: str,
    meaning: str,
) -> DicomSequence:
    """Create a single-item code sequence for concept names/values."""
    ds = Dataset()
    ds.CodeValue = value
    ds.CodingSchemeDesignator = designator
    ds.CodeMeaning = meaning
    return DicomSequence([ds])


def _make_text_content_item(
    concept_name_value: str,
    concept_name_designator: str,
    concept_name_meaning: str,
    text_value: str,
    relationship_type: str = "CONTAINS",
) -> Dataset:
    """Create a TEXT content item."""
    item = Dataset()
    item.RelationshipType = relationship_type
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = _make_code_sequence(
        concept_name_value, concept_name_designator, concept_name_meaning
    )
    item.TextValue = text_value
    return item


def _make_code_content_item(
    concept_name_value: str,
    concept_name_designator: str,
    concept_name_meaning: str,
    code_value: str,
    code_designator: str,
    code_meaning: str,
    relationship_type: str = "CONTAINS",
) -> Dataset:
    """Create a CODE content item."""
    item = Dataset()
    item.RelationshipType = relationship_type
    item.ValueType = "CODE"
    item.ConceptNameCodeSequence = _make_code_sequence(
        concept_name_value, concept_name_designator, concept_name_meaning
    )
    item.ConceptCodeSequence = _make_code_sequence(
        code_value, code_designator, code_meaning
    )
    return item


def _make_num_content_item(
    concept_name_value: str,
    concept_name_designator: str,
    concept_name_meaning: str,
    numeric_value: float,
    unit_value: str = "{probability}",
    unit_designator: str = "UCUM",
    unit_meaning: str = "probability",
    relationship_type: str = "CONTAINS",
) -> Dataset:
    """Create a NUM (numeric) content item."""
    item = Dataset()
    item.RelationshipType = relationship_type
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = _make_code_sequence(
        concept_name_value, concept_name_designator, concept_name_meaning
    )
    # MeasuredValueSequence
    mv = Dataset()
    mv.NumericValue = pydicom.valuerep.DSfloat(round(numeric_value, 4))
    mv.MeasurementUnitsCodeSequence = _make_code_sequence(
        unit_value, unit_designator, unit_meaning
    )
    item.MeasuredValueSequence = DicomSequence([mv])
    return item


def _make_container_content_item(
    concept_name_value: str,
    concept_name_designator: str,
    concept_name_meaning: str,
    children: Sequence[Dataset],
    relationship_type: str = "CONTAINS",
    continuity_of_content: str = "SEPARATE",
) -> Dataset:
    """Create a CONTAINER content item with child items."""
    item = Dataset()
    item.RelationshipType = relationship_type
    item.ValueType = "CONTAINER"
    item.ConceptNameCodeSequence = _make_code_sequence(
        concept_name_value, concept_name_designator, concept_name_meaning
    )
    item.ContinuityOfContent = continuity_of_content
    if children:
        item.ContentSequence = DicomSequence(list(children))
    return item


def _get_class_names_for_task(task: str) -> List[str]:
    """Get canonical class names for a given task head."""
    if task == "rhythm":
        from aortica.models.rhythm_head import RHYTHM_CLASSES

        return list(RHYTHM_CLASSES)
    elif task == "structural":
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        return list(STRUCTURAL_CLASSES)
    elif task == "ischaemia":
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES

        return list(ISCHAEMIA_CLASSES)
    elif task == "risk":
        from aortica.models.risk_head import RISK_OUTPUTS

        return list(RISK_OUTPUTS)
    return []


def _build_finding_items(
    multi_task_output: Dict[str, Any],
    confidence_threshold: float,
) -> List[Dataset]:
    """Build CONTENT ITEMS for classification findings."""
    items: List[Dataset] = []
    classification_tasks = ["rhythm", "structural", "ischaemia"]

    for task in classification_tasks:
        task_preds = multi_task_output.get(task)
        if task_preds is None:
            continue

        if isinstance(task_preds, dict):
            pred_items: Iterable[Tuple[str, Any]] = task_preds.items()
        elif isinstance(task_preds, (list, tuple)):
            class_names = _get_class_names_for_task(task)
            pred_items = zip(class_names, task_preds)
        else:
            continue

        for class_name, confidence in pred_items:
            conf_val = float(confidence)
            if conf_val < confidence_threshold:
                continue

            # Look up SNOMED code
            if class_name in _SNOMED_CODES:
                code, display = _SNOMED_CODES[class_name]
                designator = _SCT
            else:
                code = class_name
                display = class_name.replace("_", " ").title()
                designator = "99AORTICA"

            # Container for this finding: CODE item + NUM confidence
            finding_code = _make_code_content_item(
                concept_name_value="121071",
                concept_name_designator=_DCM,
                concept_name_meaning="Finding",
                code_value=code,
                code_designator=designator,
                code_meaning=display,
            )

            confidence_num = _make_num_content_item(
                concept_name_value="118000",
                concept_name_designator=_DCM,
                concept_name_meaning="Algorithm Confidence",
                numeric_value=conf_val,
            )

            task_label = _make_text_content_item(
                concept_name_value="121106",
                concept_name_designator=_DCM,
                concept_name_meaning="Comment",
                text_value=f"Task head: {task}",
            )

            finding_container = _make_container_content_item(
                concept_name_value="121071",
                concept_name_designator=_DCM,
                concept_name_meaning="Finding",
                children=[finding_code, confidence_num, task_label],
            )
            items.append(finding_container)

    return items


def _build_risk_items(
    multi_task_output: Dict[str, Any],
) -> List[Dataset]:
    """Build CONTENT ITEMS for risk score predictions."""
    items: List[Dataset] = []
    risk_preds = multi_task_output.get("risk")

    if risk_preds is None:
        return items

    if isinstance(risk_preds, dict):
        risk_iter: Iterable[Tuple[str, Any]] = risk_preds.items()
    elif isinstance(risk_preds, (list, tuple)):
        from aortica.models.risk_head import RISK_OUTPUTS

        risk_iter = zip(RISK_OUTPUTS, risk_preds)
    else:
        return items

    for risk_name, probability in risk_iter:
        prob_val = float(probability)
        display = _RISK_DISPLAY.get(
            risk_name, risk_name.replace("_", " ").title()
        )

        risk_num = _make_num_content_item(
            concept_name_value="121071",
            concept_name_designator=_DCM,
            concept_name_meaning="Finding",
            numeric_value=prob_val,
        )

        risk_label = _make_text_content_item(
            concept_name_value="121106",
            concept_name_designator=_DCM,
            concept_name_meaning="Comment",
            text_value=f"Risk: {display}",
        )

        risk_container = _make_container_content_item(
            concept_name_value="121073",
            concept_name_designator=_DCM,
            concept_name_meaning="Impression",
            children=[risk_num, risk_label],
        )
        items.append(risk_container)

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class DICOMSROutput:
    """Container for the generated DICOM SR dataset.

    Attributes:
        dataset: The ``pydicom.Dataset`` containing the SR IOD.
        finding_count: Number of classification finding content items.
        risk_count: Number of risk score content items.
        sop_instance_uid: SOP Instance UID of the generated SR.
    """

    dataset: Any  # pydicom.Dataset
    finding_count: int = 0
    risk_count: int = 0
    sop_instance_uid: str = ""


def to_structured_report(
    multi_task_output: Dict[str, Any],
    original_dicom_ref: Optional[str] = None,
    *,
    patient_name: str = "ANONYMOUS",
    patient_id: str = "ANONYMOUS",
    confidence_threshold: float = 0.30,
    study_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    study_description: str = "AI-Assisted ECG Analysis",
) -> Dataset:
    """Convert Aortica multi-task predictions to a DICOM SR dataset.

    Generates a Comprehensive SR (SOP Class 1.2.840.10008.5.1.4.1.1.88.33)
    using the TID 2000 (Basic Diagnostic Imaging Report) structure.

    Args:
        multi_task_output: Dictionary with task names as keys and
            prediction values.  Classification tasks map class names
            to float probabilities.  The ``risk`` task maps output
            names to float probabilities.
        original_dicom_ref: Referenced SOP Instance UID of the
            original DICOM ECG waveform instance.  If provided, the
            SR will contain a reference to this instance.
        patient_name: Patient name for the SR header.
        patient_id: Patient ID for the SR header.
        confidence_threshold: Minimum confidence for a classification
            finding to be included.  Default ``0.30``.
        study_instance_uid: Study Instance UID.  Generated if not
            provided.
        series_instance_uid: Series Instance UID.  Generated if not
            provided.
        study_description: Study description string.

    Returns:
        A ``pydicom.Dataset`` containing a valid DICOM SR object
        with TID 2000 content tree structure.
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")[:13]

    sop_instance_uid = generate_uid()
    study_uid = study_instance_uid or generate_uid()
    series_uid = series_instance_uid or generate_uid()

    # ------------------------------------------------------------------
    # Build the DICOM SR dataset with required IOD modules
    # ------------------------------------------------------------------
    ds = Dataset()

    # -- File Meta Information --
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = _COMPREHENSIVE_SR_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = _TRANSFER_SYNTAX_EXPLICIT_VR_LE
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    file_meta.ImplementationVersionName = "AORTICA_SR"
    ds.file_meta = file_meta  # type: ignore[assignment]

    # -- Patient Module --
    ds.PatientName = patient_name
    ds.PatientID = patient_id

    # -- General Study Module --
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.StudyDescription = study_description
    ds.AccessionNumber = ""

    # -- General Series Module --
    ds.Modality = "SR"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "Aortica AI ECG Structured Report"

    # -- SR Document Series Module --
    # (Modality already set above)

    # -- General Equipment Module --
    ds.Manufacturer = "Aortica"
    ds.ManufacturerModelName = "Aortica AI ECG Engine"
    ds.SoftwareVersions = "1.0"

    # -- SOP Common Module --
    ds.SOPClassUID = _COMPREHENSIVE_SR_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str
    ds.SpecificCharacterSet = "ISO_IR 100"

    # -- SR Document General Module --
    ds.InstanceNumber = 1
    ds.ContentDate = date_str
    ds.ContentTime = time_str
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"

    # Referenced Request Sequence (empty but required)
    ds.ReferencedRequestSequence = DicomSequence([])

    # -- SR Document Content Module (TID 2000) --
    ds.ValueType = "CONTAINER"
    ds.ConceptNameCodeSequence = _make_code_sequence(
        "126000", _DCM, "Imaging Report"
    )
    ds.ContinuityOfContent = "SEPARATE"

    # Build content tree
    content_items: List[Dataset] = []

    # -- Language of Content Item (required by TID 2000) --
    language_item = _make_code_content_item(
        concept_name_value="121049",
        concept_name_designator=_DCM,
        concept_name_meaning="Language of Content Item and Descendants",
        code_value="en",
        code_designator="RFC5646",
        code_meaning="English",
        relationship_type="HAS CONCEPT MOD",
    )
    content_items.append(language_item)

    # -- Procedure Reported --
    procedure_item = _make_code_content_item(
        concept_name_value="121058",
        concept_name_designator=_DCM,
        concept_name_meaning="Procedure Reported",
        code_value="11524-6",
        code_designator=_LN,
        code_meaning="EKG study",
        relationship_type="HAS CONCEPT MOD",
    )
    content_items.append(procedure_item)

    # -- Reference to original ECG (if provided) --
    if original_dicom_ref:
        ref_item = Dataset()
        ref_item.RelationshipType = "CONTAINS"
        ref_item.ValueType = "IMAGE"
        ref_item.ConceptNameCodeSequence = _make_code_sequence(
            "121111", _DCM, "Current Procedure Evidence"
        )
        ref_sop = Dataset()
        ref_sop.ReferencedSOPClassUID = _ECG_WAVEFORM_SOP_CLASS
        ref_sop.ReferencedSOPInstanceUID = original_dicom_ref
        ref_item.ReferencedSOPSequence = DicomSequence([ref_sop])
        content_items.append(ref_item)

    # -- Findings Section --
    finding_items = _build_finding_items(multi_task_output, confidence_threshold)
    if finding_items:
        findings_section = _make_container_content_item(
            concept_name_value="121070",
            concept_name_designator=_DCM,
            concept_name_meaning="Findings",
            children=finding_items,
        )
        content_items.append(findings_section)

    # -- Risk / Impression Section --
    risk_items = _build_risk_items(multi_task_output)
    if risk_items:
        impression_section = _make_container_content_item(
            concept_name_value="121073",
            concept_name_designator=_DCM,
            concept_name_meaning="Impression",
            children=risk_items,
        )
        content_items.append(impression_section)

    # -- Conclusion text --
    conclusion_text = (
        "AI decision support only — requires clinician review."
    )
    conclusion_item = _make_text_content_item(
        concept_name_value="121076",
        concept_name_designator=_DCM,
        concept_name_meaning="Conclusion",
        text_value=conclusion_text,
    )
    content_items.append(conclusion_item)

    # Set content sequence on root
    ds.ContentSequence = DicomSequence(content_items)

    return ds


def to_structured_report_output(
    multi_task_output: Dict[str, Any],
    original_dicom_ref: Optional[str] = None,
    **kwargs: Any,
) -> DICOMSROutput:
    """Convert predictions to DICOM SR and return a rich output object.

    Wrapper around :func:`to_structured_report` that also returns
    counts for findings and risk items.

    Args:
        multi_task_output: See :func:`to_structured_report`.
        original_dicom_ref: See :func:`to_structured_report`.
        **kwargs: Forwarded to :func:`to_structured_report`.

    Returns:
        :class:`DICOMSROutput` with the dataset and item counts.
    """
    threshold = kwargs.get("confidence_threshold", 0.30)
    ds = to_structured_report(
        multi_task_output, original_dicom_ref, **kwargs
    )

    finding_count = len(
        _build_finding_items(multi_task_output, threshold)
    )
    risk_count = len(_build_risk_items(multi_task_output))

    return DICOMSROutput(
        dataset=ds,
        finding_count=finding_count,
        risk_count=risk_count,
        sop_instance_uid=str(ds.SOPInstanceUID),
    )
