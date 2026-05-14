"""FHIR R4 DiagnosticReport output generator.

Converts Aortica multi-task ECG predictions into standards-compliant
FHIR R4 resources:

- **DiagnosticReport** — top-level container referencing all child resources
- **Observation** — one per positive classification finding (confidence ≥ threshold)
- **RiskAssessment** — one per risk prediction output

The output validates against FHIR R4 resource schemas via the
``fhir.resources`` library.

Requires the ``fhir`` optional dependency::

    pip install aortica[fhir]
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from fhir.resources.R4B.bundle import Bundle, BundleEntry, BundleEntryRequest
    from fhir.resources.R4B.codeableconcept import CodeableConcept
    from fhir.resources.R4B.coding import Coding
    from fhir.resources.R4B.diagnosticreport import DiagnosticReport
    from fhir.resources.R4B.observation import Observation
    from fhir.resources.R4B.reference import Reference
    from fhir.resources.R4B.riskassessment import (
        RiskAssessment,
        RiskAssessmentPrediction,
    )

    HAS_FHIR = True
except ImportError:
    HAS_FHIR = False


def _check_fhir() -> None:
    """Raise ImportError if fhir.resources is not installed."""
    if not HAS_FHIR:
        raise ImportError(
            "fhir.resources is required for FHIR output. "
            "Install with: pip install aortica[fhir]"
        )


# ---------------------------------------------------------------------------
# SNOMED CT and LOINC mappings for ECG findings
# ---------------------------------------------------------------------------

# LOINC codes for the overall ECG study
_ECG_STUDY_LOINC = ("11524-6", "EKG study")
_ECG_INTERPRETATION_LOINC = ("18844-9", "ECG impression interpretation")

# Mapping from Aortica class names → (SNOMED CT code, display name)
# Not all classes have standard codes; those without get a local code.
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
    "CPVT": ("419671004", "Catecholaminergic polymorphic ventricular tachycardia"),
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

# Interpretation codes for confidence levels
_INTERPRETATION_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
)

# Risk output display names
_RISK_DISPLAY_NAMES: Dict[str, str] = {
    "mortality_1y": "1-year all-cause mortality risk",
    "hf_hosp_12m": "12-month heart failure hospitalization probability",
    "af_onset_12m": "12-month atrial fibrillation onset risk",
    "ecg_predicted_ef": "ECG-predicted ejection fraction",
    "conduction_disease_trajectory": "Progressive conduction disease trajectory",
    "sudden_cardiac_death_risk": "Sudden cardiac death risk",
}

# Risk output LOINC codes (where available)
_RISK_LOINC_CODES: Dict[str, Tuple[str, str]] = {
    "mortality_1y": ("75889-6", "1-year mortality risk"),
    "af_onset_12m": ("83136-4", "AF onset risk"),
    "ecg_predicted_ef": ("10230-1", "Left ventricular ejection fraction"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    """Generate a unique FHIR resource ID (UUID without dashes)."""
    return uuid.uuid4().hex[:16]


def _interpret_confidence(confidence: float) -> Tuple[str, str]:
    """Map a confidence score to an HL7 interpretation code.

    Returns (code, display) tuple.
    """
    if confidence >= 0.80:
        return ("H", "High")
    elif confidence >= 0.50:
        return ("A", "Abnormal")
    else:
        return ("L", "Low")


def _make_finding_observation(
    class_name: str,
    confidence: float,
    task: str,
    patient_ref: Optional[str] = None,
    issued: Optional[str] = None,
) -> "Observation":
    """Create a FHIR Observation for a single positive finding.

    Args:
        class_name: Aortica class name (e.g. ``'AF'``, ``'STEMI'``).
        confidence: Model confidence probability in ``[0, 1]``.
        task: Task head name (rhythm, structural, ischaemia).
        patient_ref: FHIR Patient reference (e.g. ``'Patient/123'``).
        issued: ISO-8601 datetime string for the observation.

    Returns:
        A validated FHIR R4 Observation resource.
    """
    _check_fhir()

    obs_id = _generate_id()

    # Code — prefer SNOMED CT when available, fall back to local
    if class_name in _SNOMED_CODES:
        snomed_code, display = _SNOMED_CODES[class_name]
        code_concept = CodeableConcept(
            coding=[
                Coding(
                    system="http://snomed.info/sct",
                    code=snomed_code,
                    display=display,
                ),
            ],
            text=display,
        )
    else:
        code_concept = CodeableConcept(
            coding=[
                Coding(
                    system="http://aortica.io/fhir/ecg-finding",
                    code=class_name,
                    display=class_name.replace("_", " ").title(),
                ),
            ],
            text=class_name.replace("_", " ").title(),
        )

    # Interpretation based on confidence
    interp_code, interp_display = _interpret_confidence(confidence)
    interpretation = [
        CodeableConcept(
            coding=[
                Coding(
                    system=_INTERPRETATION_SYSTEM,
                    code=interp_code,
                    display=interp_display,
                ),
            ],
        ),
    ]

    # Build Observation
    obs_data: Dict[str, Any] = {
        "id": obs_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam",
                    }
                ]
            }
        ],
        "code": code_concept.model_dump(exclude_none=True),
        "valueQuantity": {
            "value": round(confidence, 4),
            "unit": "probability",
            "system": "http://unitsofmeasure.org",
            "code": "{probability}",
        },
        "interpretation": [i.model_dump(exclude_none=True) for i in interpretation],
    }

    if patient_ref:
        obs_data["subject"] = {"reference": patient_ref}

    if issued:
        obs_data["issued"] = issued

    # Add component for task head categorisation
    obs_data["component"] = [
        {
            "code": {
                "coding": [
                    {
                        "system": "http://aortica.io/fhir/task-head",
                        "code": task,
                        "display": f"Aortica {task} task head",
                    }
                ]
            },
            "valueString": task,
        }
    ]

    return Observation(**obs_data)


def _make_risk_assessment(
    risk_name: str,
    probability: float,
    patient_ref: Optional[str] = None,
) -> "RiskAssessment":
    """Create a FHIR RiskAssessment for a single risk output.

    Args:
        risk_name: Aortica risk output name (e.g. ``'mortality_1y'``).
        probability: Predicted probability in ``[0, 1]``.
        patient_ref: FHIR Patient reference.

    Returns:
        A validated FHIR R4 RiskAssessment resource.
    """
    _check_fhir()

    ra_id = _generate_id()
    display = _RISK_DISPLAY_NAMES.get(
        risk_name, risk_name.replace("_", " ").title()
    )

    # Outcome coding — use LOINC if available
    outcome_coding: List[Dict[str, str]] = []
    if risk_name in _RISK_LOINC_CODES:
        loinc_code, loinc_display = _RISK_LOINC_CODES[risk_name]
        outcome_coding.append(
            {
                "system": "http://loinc.org",
                "code": loinc_code,
                "display": loinc_display,
            }
        )

    outcome_concept: Dict[str, Any] = {"text": display}
    if outcome_coding:
        outcome_concept["coding"] = outcome_coding

    prediction = RiskAssessmentPrediction(
        outcome=CodeableConcept(**outcome_concept),
        probabilityDecimal=Decimal(str(round(probability, 4))),
    )

    # subject is required by FHIR R4 RiskAssessment
    subject_ref = patient_ref or "Patient/anonymous"

    ra_data: Dict[str, Any] = {
        "id": ra_id,
        "status": "final",
        "subject": {"reference": subject_ref},
        "prediction": [prediction.model_dump(exclude_none=True)],
    }

    return RiskAssessment(**ra_data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class FHIROutput:
    """Container for the generated FHIR resources.

    Attributes:
        diagnostic_report: The top-level DiagnosticReport resource.
        observations: Child Observation resources for classification findings.
        risk_assessments: Child RiskAssessment resources for risk predictions.
        bundle_json: The complete FHIR Bundle as a JSON string.
    """

    diagnostic_report: Any  # DiagnosticReport instance
    observations: List[Any] = field(default_factory=list)
    risk_assessments: List[Any] = field(default_factory=list)
    bundle_json: str = ""


def to_diagnostic_report(
    multi_task_output: Dict[str, Any],
    patient_ref: Optional[str] = None,
    ecg_metadata: Optional[Dict[str, Any]] = None,
    *,
    confidence_threshold: float = 0.30,
) -> FHIROutput:
    """Convert Aortica multi-task predictions to FHIR R4 resources.

    Generates a FHIR R4 ``DiagnosticReport`` with child ``Observation``
    resources for each positive classification finding and
    ``RiskAssessment`` resources for each risk prediction output.

    All resources are bundled into a FHIR ``Bundle`` (type=collection)
    that validates against the FHIR R4 schema.

    Args:
        multi_task_output: Dictionary with task names as keys and
            prediction values.  Each classification task value should
            be a dict mapping class names to float probabilities.
            The ``risk`` task value should be a dict mapping risk
            output names to float probabilities.  Example::

                {
                    "rhythm": {"AF": 0.95, "normal_sinus_rhythm": 0.02, ...},
                    "structural": {"LVH": 0.80, ...},
                    "ischaemia": {"STEMI": 0.10, ...},
                    "risk": {"mortality_1y": 0.15, "hf_hosp_12m": 0.08, ...},
                }

        patient_ref: FHIR Patient reference string
            (e.g. ``'Patient/123'``).  If ``None``, subject is omitted.
        ecg_metadata: Optional metadata dict with keys like
            ``acquisition_datetime``, ``device``, ``operator``.
        confidence_threshold: Minimum confidence for a classification
            finding to be included as an Observation.  Default ``0.30``.

    Returns:
        :class:`FHIROutput` containing the DiagnosticReport, child
        Observations, RiskAssessments, and the complete Bundle as JSON.

    Raises:
        ImportError: If ``fhir.resources`` is not installed.
    """
    _check_fhir()

    ecg_metadata = ecg_metadata or {}
    now_iso = datetime.now(timezone.utc).isoformat()
    issued = ecg_metadata.get("acquisition_datetime", now_iso)

    # ------------------------------------------------------------------
    # 1. Build Observation resources for classification findings
    # ------------------------------------------------------------------
    observations: List[Observation] = []
    classification_tasks = ["rhythm", "structural", "ischaemia"]

    for task in classification_tasks:
        task_preds = multi_task_output.get(task)
        if task_preds is None:
            continue

        if isinstance(task_preds, dict):
            items: Iterable[Tuple[str, Any]] = task_preds.items()
        elif isinstance(task_preds, (list, tuple)):
            # If it's a list of floats, try to pair with class names
            class_names = _get_class_names_for_task(task)
            items = zip(class_names, task_preds)
        else:
            continue

        for class_name, confidence in items:
            conf_val = float(confidence)
            if conf_val >= confidence_threshold:
                obs = _make_finding_observation(
                    class_name=class_name,
                    confidence=conf_val,
                    task=task,
                    patient_ref=patient_ref,
                    issued=issued,
                )
                observations.append(obs)

    # ------------------------------------------------------------------
    # 2. Build RiskAssessment resources
    # ------------------------------------------------------------------
    risk_assessments: List[RiskAssessment] = []
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
            ra = _make_risk_assessment(
                risk_name=risk_name,
                probability=float(probability),
                patient_ref=patient_ref,
            )
            risk_assessments.append(ra)

    # ------------------------------------------------------------------
    # 3. Build the DiagnosticReport
    # ------------------------------------------------------------------
    report_id = _generate_id()

    # References to child resources
    result_refs = [
        Reference(reference=f"Observation/{obs.id}")
        for obs in observations
    ]

    report_data: Dict[str, Any] = {
        "id": report_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": _ECG_STUDY_LOINC[0],
                    "display": _ECG_STUDY_LOINC[1],
                },
            ],
            "text": "AI-assisted ECG interpretation",
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "CUS",
                        "display": "Cardiac Ultrasound",
                    },
                ],
            },
        ],
        "issued": issued,
        "conclusion": _build_conclusion(observations, risk_assessments),
    }

    if patient_ref:
        report_data["subject"] = {"reference": patient_ref}

    if result_refs:
        report_data["result"] = [
            r.model_dump(exclude_none=True) for r in result_refs
        ]

    diagnostic_report = DiagnosticReport(**report_data)

    # ------------------------------------------------------------------
    # 4. Bundle everything together
    # ------------------------------------------------------------------
    bundle_entries: List[BundleEntry] = []

    # DiagnosticReport entry
    bundle_entries.append(
        BundleEntry(
            fullUrl=f"urn:uuid:{report_id}",
            resource=diagnostic_report,
        )
    )

    # Observation entries
    for obs in observations:
        bundle_entries.append(
            BundleEntry(
                fullUrl=f"urn:uuid:{obs.id}",
                resource=obs,
            )
        )

    # RiskAssessment entries
    for ra in risk_assessments:
        bundle_entries.append(
            BundleEntry(
                fullUrl=f"urn:uuid:{ra.id}",
                resource=ra,
            )
        )

    bundle = Bundle(
        type="collection",
        entry=bundle_entries,
    )

    bundle_json = bundle.model_dump_json(indent=2, exclude_none=True)

    return FHIROutput(
        diagnostic_report=diagnostic_report,
        observations=observations,
        risk_assessments=risk_assessments,
        bundle_json=bundle_json,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _build_conclusion(
    observations: List[Any],
    risk_assessments: List[Any],
) -> str:
    """Build a human-readable conclusion string from the findings."""
    parts: List[str] = []

    if observations:
        finding_names = []
        for obs in observations:
            text = obs.code.text or "Unknown finding"
            finding_names.append(text)
        parts.append(
            f"AI-detected findings ({len(observations)}): "
            + ", ".join(finding_names)
            + "."
        )

    if risk_assessments:
        risk_parts = []
        for ra in risk_assessments:
            if ra.prediction:
                pred = ra.prediction[0]
                outcome = pred.outcome.text if pred.outcome else "Risk"
                prob = pred.probabilityDecimal
                risk_parts.append(f"{outcome}: {prob:.1%}")
        if risk_parts:
            parts.append("Risk scores: " + "; ".join(risk_parts) + ".")

    if not parts:
        return "No significant AI findings detected."

    return " ".join(parts) + " AI decision support only — requires clinician review."
