"""HL7 v2.x ORU^R01 message generation for legacy EHR integration.

Converts Aortica multi-task ECG predictions into valid HL7 v2.x
ORU^R01 messages that can be transmitted via MLLP to legacy EHR
systems.

Each active classification finding is encoded as an OBX segment
with its corresponding SNOMED CT or local code, display name,
and confidence value.  Risk scores are encoded as numeric OBX
segments with units.

Special characters (|, ^, &, ~, \\) are escaped automatically by
the ``hl7apy`` library per the HL7 v2.x specification.

Requires the ``hl7`` optional dependency::

    pip install aortica[hl7]
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from hl7apy.consts import VALIDATION_LEVEL
    from hl7apy.core import Message
    from hl7apy.parser import parse_message

    HAS_HL7 = True
except ImportError:
    HAS_HL7 = False


def _check_hl7() -> None:
    """Raise ImportError if hl7apy is not installed."""
    if not HAS_HL7:
        raise ImportError(
            "hl7apy is required for HL7 v2.x output. "
            "Install with: pip install aortica[hl7]"
        )


# ---------------------------------------------------------------------------
# SNOMED CT and local code mappings for ECG findings
# ---------------------------------------------------------------------------

# Reuse the same SNOMED mappings as fhir.py for consistency.
# Mapping: class_name → (code, display, coding_system)
_FINDING_CODES: Dict[str, Tuple[str, str, str]] = {
    # Rhythm classes (SNOMED CT)
    "AF": ("49436004", "Atrial fibrillation", "SCT"),
    "AFL": ("5370000", "Atrial flutter", "SCT"),
    "SVT": ("6456007", "Supraventricular tachycardia", "SCT"),
    "AVNRT": ("251166008", "AV nodal reentry tachycardia", "SCT"),
    "AVRT": ("233897008", "AV reentrant tachycardia", "SCT"),
    "VT": ("25569003", "Ventricular tachycardia", "SCT"),
    "VF": ("71908006", "Ventricular fibrillation", "SCT"),
    "idioventricular": ("29320008", "Idioventricular rhythm", "SCT"),
    "sinus_brady": ("49710005", "Sinus bradycardia", "SCT"),
    "sinus_tachy": ("11092001", "Sinus tachycardia", "SCT"),
    "PAC": ("284470004", "Premature atrial contraction", "SCT"),
    "PVC": ("427172004", "Premature ventricular contraction", "SCT"),
    "av_block_1st": ("270492004", "First degree AV block", "SCT"),
    "av_block_2nd": ("195042002", "Second degree AV block", "SCT"),
    "av_block_3rd": ("27885002", "Complete AV block", "SCT"),
    "LBBB": ("63467002", "Left bundle branch block", "SCT"),
    "RBBB": ("59118001", "Right bundle branch block", "SCT"),
    "LAFB": ("445118002", "Left anterior fascicular block", "SCT"),
    "LPFB": ("445211001", "Left posterior fascicular block", "SCT"),
    "WPW": ("74390002", "Wolff-Parkinson-White pattern", "SCT"),
    "pacemaker_rhythm": ("10370003", "Paced rhythm", "SCT"),
    "normal_sinus_rhythm": ("426783006", "Normal sinus rhythm", "SCT"),
    "brugada_pattern": ("418818005", "Brugada syndrome", "SCT"),
    "short_QT_syndrome": ("699255007", "Short QT syndrome", "SCT"),
    "CPVT": ("419671004", "CPVT", "SCT"),
    "fascicular_VT": ("426749004", "Fascicular VT", "SCT"),
    "atypical_atrial_flutter": ("5370000", "Atypical atrial flutter", "SCT"),
    "inappropriate_sinus_tachy": ("11092001", "Inappropriate sinus tachycardia", "SCT"),
    # Structural classes
    "LVH": ("55827005", "Left ventricular hypertrophy", "SCT"),
    "RVH": ("44313006", "Right ventricular hypertrophy", "SCT"),
    "LVSD": ("48867003", "LV systolic dysfunction", "SCT"),
    "HFpEF_risk": ("446221000", "HFpEF risk", "SCT"),
    "DCM": ("195021002", "Dilated cardiomyopathy", "SCT"),
    "HCM": ("45227007", "Hypertrophic cardiomyopathy", "SCT"),
    "ARVC": ("253528005", "ARVC", "SCT"),
    "amyloidosis": ("17602002", "Cardiac amyloidosis", "SCT"),
    "aortic_stenosis": ("60573004", "Aortic stenosis", "SCT"),
    "mitral_regurgitation": ("48724000", "Mitral regurgitation", "SCT"),
    "pulmonary_HTN": ("70995007", "Pulmonary hypertension", "SCT"),
    "LA_enlargement": ("67751000", "Left atrial enlargement", "SCT"),
    "RA_enlargement": ("67741000119100", "Right atrial enlargement", "SCT"),
    "pericarditis": ("3238004", "Pericarditis", "SCT"),
    "myocarditis": ("50920009", "Myocarditis", "SCT"),
    "LV_strain_grade": ("55827005", "LV strain pattern", "SCT"),
    "RV_strain_PE": ("59282003", "RV strain - PE", "SCT"),
    "Takotsubo_pattern": ("423727003", "Takotsubo cardiomyopathy", "SCT"),
    "infiltrative_cardiomyopathy_strain": (
        "415295002",
        "Infiltrative cardiomyopathy",
        "SCT",
    ),
    # Ischaemia classes
    "STEMI": ("401303003", "STEMI", "SCT"),
    "posterior_MI": ("73795002", "Posterior MI", "SCT"),
    "occlusive_NSTEMI": ("401314000", "NSTEMI", "SCT"),
    "old_MI": ("22298006", "Old MI", "SCT"),
    "hyperkalaemia": ("14140009", "Hyperkalaemia", "SCT"),
    "hypokalaemia": ("43339004", "Hypokalaemia", "SCT"),
    "hypercalcaemia": ("66931009", "Hypercalcaemia", "SCT"),
    "hypothyroidism_pattern": ("40930008", "Hypothyroidism", "SCT"),
    "digitalis_effect": ("13384006", "Digitalis effect", "SCT"),
    "QTc_prolongation": ("111975006", "Prolonged QT interval", "SCT"),
    "early_repol_vs_STEMI": ("428417006", "Early repolarization pattern", "SCT"),
    "de_Winter_T_wave": ("840544004", "De Winter T-wave pattern", "SCT"),
    "Wellens_syndrome": ("840546002", "Wellens syndrome", "SCT"),
    "aVR_ST_elevation": ("401303003", "aVR ST elevation pattern", "SCT"),
    "Sgarbossa_criteria": ("401303003", "Sgarbossa criteria positive", "SCT"),
    "hyperkalaemia_severity_grade": (
        "14140009",
        "Hyperkalaemia severity grading",
        "SCT",
    ),
    "hypothermia_osborn_waves": ("386689009", "Hypothermia - Osborn waves", "SCT"),
    "TCA_toxicity": ("212601003", "TCA toxicity", "SCT"),
    "digoxin_effect_vs_toxicity": ("81060008", "Digoxin toxicity", "SCT"),
}

# Risk output display names and LOINC codes
_RISK_DISPLAY: Dict[str, Tuple[str, str]] = {
    "mortality_1y": ("75889-6", "1-year all-cause mortality risk"),
    "hf_hosp_12m": ("LOCAL-HF", "12-month HF hospitalization probability"),
    "af_onset_12m": ("83136-4", "12-month AF onset risk"),
    "ecg_predicted_ef": ("10230-1", "ECG-predicted ejection fraction"),
    "conduction_disease_trajectory": (
        "LOCAL-CDT",
        "Progressive conduction disease trajectory",
    ),
    "sudden_cardiac_death_risk": (
        "LOCAL-SCD",
        "Sudden cardiac death risk",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_control_id() -> str:
    """Generate a unique HL7 message control ID."""
    return uuid.uuid4().hex[:12].upper()


def _hl7_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime as HL7 timestamp (YYYYMMDDHHmmss)."""
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class HL7v2Output:
    """Container for the generated HL7 v2.x ORU^R01 message.

    Attributes:
        message_string: The complete ER7-encoded HL7 v2.x message.
        segment_count: Number of segments in the message.
        obx_count: Number of OBX (observation) segments.
    """

    message_string: str
    segment_count: int
    obx_count: int


def to_oru_r01(
    multi_task_output: Dict[str, Any],
    patient_id: Optional[str] = None,
    order_id: Optional[str] = None,
    *,
    confidence_threshold: float = 0.30,
    sending_application: str = "Aortica",
    sending_facility: str = "AorticaSite",
    receiving_application: str = "EHR",
    receiving_facility: str = "Hospital",
) -> str:
    """Convert Aortica multi-task predictions to an HL7 v2.x ORU^R01 message.

    Generates a valid HL7 v2.x ORU^R01 message containing OBX segments
    for each active classification finding and each risk score prediction.

    Classification findings above ``confidence_threshold`` are encoded
    as string-type (ST) OBX segments with the confidence value and
    corresponding SNOMED CT or local diagnostic code.

    Risk scores are encoded as numeric (NM) OBX segments with
    probability units.

    Args:
        multi_task_output: Dictionary with task names as keys and
            prediction values.  Classification tasks map class names
            to float probabilities.  The ``risk`` task maps output
            names to float probabilities.  Example::

                {
                    "rhythm": {"AF": 0.95, "normal_sinus_rhythm": 0.02},
                    "structural": {"LVH": 0.80},
                    "ischaemia": {"STEMI": 0.10},
                    "risk": {"mortality_1y": 0.15, "hf_hosp_12m": 0.08},
                }

        patient_id: Patient identifier for the PID segment.  If
            ``None``, a default ``ANONYMOUS`` ID is used.
        order_id: Order/accession number for the OBR segment.  If
            ``None``, a UUID-based ID is generated.
        confidence_threshold: Minimum confidence for a classification
            finding to be included.  Default ``0.30``.
        sending_application: MSH-3 sending application name.
        sending_facility: MSH-4 sending facility name.
        receiving_application: MSH-5 receiving application name.
        receiving_facility: MSH-6 receiving facility name.

    Returns:
        An ER7-encoded HL7 v2.x ORU^R01 message string.

    Raises:
        ImportError: If ``hl7apy`` is not installed.
    """
    _check_hl7()

    control_id = _generate_control_id()
    timestamp = _hl7_timestamp()
    order_id = order_id or uuid.uuid4().hex[:10].upper()
    patient_id = patient_id or "ANONYMOUS"

    # ------------------------------------------------------------------
    # Build the HL7 v2.x message using hl7apy
    # ------------------------------------------------------------------
    msg = Message("ORU_R01", validation_level=VALIDATION_LEVEL.TOLERANT)

    # MSH — Message Header
    msg.msh.msh_3 = sending_application
    msg.msh.msh_4 = sending_facility
    msg.msh.msh_5 = receiving_application
    msg.msh.msh_6 = receiving_facility
    msg.msh.msh_9.msh_9_1 = "ORU"
    msg.msh.msh_9.msh_9_2 = "R01"
    msg.msh.msh_9.msh_9_3 = "ORU_R01"
    msg.msh.msh_10 = control_id
    msg.msh.msh_11.msh_11_1 = "P"  # Production
    msg.msh.msh_12 = "2.5"

    # Patient Result group
    patient_result = msg.add_group("ORU_R01_PATIENT_RESULT")

    # PID — Patient Identification
    patient_group = patient_result.add_group("ORU_R01_PATIENT")
    pid = patient_group.add_segment("PID")
    pid.pid_3 = patient_id

    # OBR — Order/Observation Request (one per report)
    order_obs = patient_result.add_group("ORU_R01_ORDER_OBSERVATION")
    obr = order_obs.add_segment("OBR")
    obr.obr_2 = order_id
    obr.obr_4.obr_4_1 = "11524-6"
    obr.obr_4.obr_4_2 = "EKG study"
    obr.obr_4.obr_4_3 = "LN"
    obr.obr_7 = timestamp
    obr.obr_25 = "F"  # Final result

    # ------------------------------------------------------------------
    # Build OBX segments for classification findings
    # ------------------------------------------------------------------
    obx_counter = 0
    classification_tasks = ["rhythm", "structural", "ischaemia"]

    for task in classification_tasks:
        task_preds = multi_task_output.get(task)
        if task_preds is None:
            continue

        if isinstance(task_preds, dict):
            items: Iterable[Tuple[str, Any]] = task_preds.items()
        elif isinstance(task_preds, (list, tuple)):
            class_names = _get_class_names_for_task(task)
            items = zip(class_names, task_preds)
        else:
            continue

        for class_name, confidence in items:
            conf_val = float(confidence)
            if conf_val < confidence_threshold:
                continue

            obx_counter += 1

            # Look up code
            if class_name in _FINDING_CODES:
                code, display, system = _FINDING_CODES[class_name]
            else:
                code = class_name
                display = class_name.replace("_", " ").title()
                system = "LOCAL"

            obs_group = order_obs.add_group("ORU_R01_OBSERVATION")
            obx = obs_group.add_segment("OBX")
            obx.obx_1 = str(obx_counter)
            obx.obx_2 = "NM"
            obx.obx_3.obx_3_1 = code
            obx.obx_3.obx_3_2 = display
            obx.obx_3.obx_3_3 = system
            obx.obx_5 = str(round(conf_val, 4))
            obx.obx_6 = "{probability}"
            obx.obx_8 = _interpret_abnormal_flag(conf_val)
            obx.obx_11 = "F"  # Final

    # ------------------------------------------------------------------
    # Build OBX segments for risk scores
    # ------------------------------------------------------------------
    risk_preds = multi_task_output.get("risk")
    if risk_preds is not None:
        if isinstance(risk_preds, dict):
            risk_items: Iterable[Tuple[str, Any]] = risk_preds.items()
        elif isinstance(risk_preds, (list, tuple)):
            from aortica.models.risk_head import RISK_OUTPUTS

            risk_items = zip(RISK_OUTPUTS, risk_preds)
        else:
            risk_items = []

        for risk_name, probability in risk_items:
            obx_counter += 1
            prob_val = float(probability)

            if risk_name in _RISK_DISPLAY:
                code, display = _RISK_DISPLAY[risk_name]
            else:
                code = risk_name
                display = risk_name.replace("_", " ").title()

            # Determine coding system
            coding_system = "LN" if not code.startswith("LOCAL") else "LOCAL"

            obs_group = order_obs.add_group("ORU_R01_OBSERVATION")
            obx = obs_group.add_segment("OBX")
            obx.obx_1 = str(obx_counter)
            obx.obx_2 = "NM"
            obx.obx_3.obx_3_1 = code
            obx.obx_3.obx_3_2 = display
            obx.obx_3.obx_3_3 = coding_system
            obx.obx_5 = str(round(prob_val, 4))
            obx.obx_6 = "{probability}"
            obx.obx_11 = "F"  # Final

    # ------------------------------------------------------------------
    # Serialize and return
    # ------------------------------------------------------------------
    return str(msg.to_er7())


def _interpret_abnormal_flag(confidence: float) -> str:
    """Map confidence to HL7 v2.x abnormal flag.

    Returns:
        ``'AA'`` (critical abnormal) for ≥ 0.90,
        ``'A'`` (abnormal) for ≥ 0.50,
        ``'N'`` (normal) for < 0.50.
    """
    if confidence >= 0.90:
        return "AA"
    elif confidence >= 0.50:
        return "A"
    return "N"


def parse_oru_r01(message_string: str) -> Dict[str, Any]:
    """Parse an HL7 v2.x ORU^R01 message back into structured data.

    This utility function parses an ER7-encoded HL7 message and
    extracts the OBX segment data for validation purposes.

    Args:
        message_string: ER7-encoded HL7 v2.x message string.

    Returns:
        Dictionary with keys ``message_type``, ``control_id``,
        ``patient_id``, ``observations`` (list of dicts), and
        ``segment_count``.

    Raises:
        ImportError: If ``hl7apy`` is not installed.
    """
    _check_hl7()

    parsed = parse_message(
        message_string, validation_level=VALIDATION_LEVEL.TOLERANT
    )

    result: Dict[str, Any] = {
        "message_type": "ORU^R01",
        "control_id": "",
        "patient_id": "",
        "observations": [],
        "segment_count": 0,
    }

    # Count all segments recursively
    segment_count = 0
    observations: List[Dict[str, Any]] = []

    def _walk(element: Any) -> None:
        nonlocal segment_count
        if hasattr(element, "name"):
            if element.name == "MSH":
                segment_count += 1
                try:
                    result["control_id"] = str(element.msh_10.value)
                except Exception:
                    pass
            elif element.name == "PID":
                segment_count += 1
                try:
                    result["patient_id"] = str(element.pid_3.value)
                except Exception:
                    pass
            elif element.name == "OBR":
                segment_count += 1
            elif element.name == "OBX":
                segment_count += 1
                obs: Dict[str, Any] = {}
                try:
                    obs["set_id"] = str(element.obx_1.value)
                except Exception:
                    pass
                try:
                    obs["value_type"] = str(element.obx_2.value)
                except Exception:
                    pass
                try:
                    obs["code"] = str(element.obx_3.obx_3_1.value)
                except Exception:
                    pass
                try:
                    obs["display"] = str(element.obx_3.obx_3_2.value)
                except Exception:
                    pass
                try:
                    obs["value"] = str(element.obx_5.value)
                except Exception:
                    pass
                try:
                    obs["units"] = str(element.obx_6.value)
                except Exception:
                    pass
                try:
                    obs["status"] = str(element.obx_11.value)
                except Exception:
                    pass
                observations.append(obs)
            else:
                # It's a group — recurse into children
                if hasattr(element, "children"):
                    for child in element.children:
                        _walk(child)
                return

        if hasattr(element, "children"):
            for child in element.children:
                _walk(child)

    _walk(parsed)

    result["observations"] = observations
    result["segment_count"] = segment_count

    return result
