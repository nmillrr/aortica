"""JSON-LD Machine-Readable Report Generator.

Generates a JSON-LD 1.1 document from Aortica multi-task ECG predictions.

Features:

* Uses Schema.org ``MedicalTest`` and ``MedicalObservation`` types
* Findings linked to SNOMED CT and LOINC codes via ``@context``
* Provenance metadata: model version, inference timestamp, input file hash,
  confidence intervals
* Validates against JSON-LD 1.1 spec (compaction/expansion round-trips)

Requires the ``jsonld`` optional dependency::

    pip install aortica[jsonld]

Usage::

    from aortica.reports import generate_jsonld
    doc = generate_jsonld(multi_task_output, ecg_metadata)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES


# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------


def _get_pyld() -> Any:
    """Lazily import pyld."""
    try:
        import pyld  # type: ignore[import-untyped]

        return pyld
    except ImportError:
        raise ImportError(
            "PyLD is required for JSON-LD report generation. "
            "Install with: pip install aortica[jsonld]"
        ) from None


# ---------------------------------------------------------------------------
# SNOMED CT code mappings (reused from fhir.py)
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

# LOINC codes for risk outputs
_RISK_LOINC_CODES: Dict[str, Tuple[str, str]] = {
    "mortality_1y": ("75889-6", "1-year mortality risk"),
    "hf_hosp_12m": ("88023-8", "Heart failure hospitalization risk"),
    "af_onset_12m": ("83136-4", "AF onset risk"),
    "ecg_predicted_ef": ("10230-1", "Left ventricular ejection fraction"),
    "conduction_disease_trajectory": (
        "18844-9",
        "Progressive conduction disease trajectory",
    ),
    "sudden_cardiac_death_risk": ("18844-9", "Sudden cardiac death risk"),
}

# Risk output display names
_RISK_DISPLAY_NAMES: Dict[str, str] = {
    "mortality_1y": "1-year all-cause mortality risk",
    "hf_hosp_12m": "12-month heart failure hospitalization probability",
    "af_onset_12m": "12-month atrial fibrillation onset risk",
    "ecg_predicted_ef": "ECG-predicted ejection fraction",
    "conduction_disease_trajectory": "Progressive conduction disease trajectory",
    "sudden_cardiac_death_risk": "Sudden cardiac death risk",
}


# ---------------------------------------------------------------------------
# JSON-LD context
# ---------------------------------------------------------------------------

_JSONLD_CONTEXT: Dict[str, Any] = {
    "@vocab": "https://schema.org/",
    "snomed": "http://snomed.info/id/",
    "loinc": "http://loinc.org/rdf/",
    "aortica": "http://aortica.io/vocab/",
    "MedicalTest": "https://schema.org/MedicalTest",
    "MedicalObservation": "https://schema.org/MedicalObservation",
    "MedicalRiskEstimator": "https://schema.org/MedicalRiskEstimator",
    "usedToDiagnose": "https://schema.org/usedToDiagnose",
    "normalRange": "https://schema.org/normalRange",
    "identifier": "https://schema.org/identifier",
    "name": "https://schema.org/name",
    "description": "https://schema.org/description",
    "dateCreated": "https://schema.org/dateCreated",
    "result": "https://schema.org/result",
    "snomedCode": {
        "@id": "snomed:",
        "@type": "@id",
    },
    "loincCode": {
        "@id": "loinc:",
        "@type": "@id",
    },
    "confidence": "aortica:confidence",
    "confidenceInterval": "aortica:confidenceInterval",
    "taskHead": "aortica:taskHead",
    "modelVersion": "aortica:modelVersion",
    "inputFileHash": "aortica:inputFileHash",
    "inferenceTimestamp": "aortica:inferenceTimestamp",
    "oasisWarning": "aortica:oasisWarning",
}


# ---------------------------------------------------------------------------
# Prediction extraction helpers
# ---------------------------------------------------------------------------


def _extract_predictions(
    multi_task_output: Any,
) -> Dict[str, List[float]]:
    """Extract per-task prediction lists from various output formats.

    Handles MultiTaskOutput dataclass, dict, and tensor/array types.
    """
    import numpy as np

    tasks = {
        "rhythm": RHYTHM_CLASSES,
        "structural": STRUCTURAL_CLASSES,
        "ischaemia": ISCHAEMIA_CLASSES,
        "risk": RISK_OUTPUTS,
    }

    result: Dict[str, List[float]] = {}

    for task_name, class_names in tasks.items():
        # Get raw values
        if hasattr(multi_task_output, task_name):
            values = getattr(multi_task_output, task_name)
        elif isinstance(multi_task_output, dict):
            values = multi_task_output.get(task_name)
        else:
            values = None

        if values is None:
            continue

        # Convert tensor/array to list
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()  # type: ignore[union-attr]
        if isinstance(values, np.ndarray):
            values = values.tolist()

        # Handle batch dimension
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], list):
                values = values[0]

        # Ensure length matches class count
        n = min(len(values), len(class_names))
        result[task_name] = [float(v) for v in values[:n]]

    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JSONLDReport:
    """Container for a generated JSON-LD report.

    Attributes:
        document: The JSON-LD document as a Python dict.
        json_string: The serialised JSON-LD string.
        is_valid: Whether the document passes compaction/expansion
            round-trip validation.
    """

    document: Dict[str, Any] = field(default_factory=dict)
    json_string: str = ""
    is_valid: bool = False


# ---------------------------------------------------------------------------
# Building JSON-LD observations
# ---------------------------------------------------------------------------


def _build_finding_observation(
    class_name: str,
    confidence: float,
    task: str,
    confidence_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Build a JSON-LD observation node for a single finding.

    Args:
        class_name: Aortica class name.
        confidence: Model confidence in [0, 1].
        task: Task head name.
        confidence_interval: Optional (lower, upper) bounds.

    Returns:
        A JSON-LD compatible dict.
    """
    display = class_name.replace("_", " ").title()

    observation: Dict[str, Any] = {
        "@type": "MedicalObservation",
        "name": display,
        "identifier": class_name,
        "confidence": round(confidence, 4),
        "taskHead": task,
    }

    # SNOMED coding
    if class_name in _SNOMED_CODES:
        code, snomed_display = _SNOMED_CODES[class_name]
        observation["snomedCode"] = f"snomed:{code}"
        observation["description"] = snomed_display

    # Confidence interval
    if confidence_interval is not None:
        observation["confidenceInterval"] = {
            "lower": round(confidence_interval[0], 4),
            "upper": round(confidence_interval[1], 4),
        }

    return observation


def _build_risk_observation(
    risk_name: str,
    probability: float,
    confidence_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Build a JSON-LD observation node for a risk prediction.

    Args:
        risk_name: Aortica risk output name.
        probability: Predicted probability in [0, 1].
        confidence_interval: Optional (lower, upper) bounds.

    Returns:
        A JSON-LD compatible dict.
    """
    display = _RISK_DISPLAY_NAMES.get(
        risk_name, risk_name.replace("_", " ").title()
    )

    observation: Dict[str, Any] = {
        "@type": "MedicalRiskEstimator",
        "name": display,
        "identifier": risk_name,
        "confidence": round(probability, 4),
        "taskHead": "risk",
    }

    # LOINC coding
    if risk_name in _RISK_LOINC_CODES:
        code, loinc_display = _RISK_LOINC_CODES[risk_name]
        observation["loincCode"] = f"loinc:{code}"
        observation["description"] = loinc_display

    # Confidence interval
    if confidence_interval is not None:
        observation["confidenceInterval"] = {
            "lower": round(confidence_interval[0], 4),
            "upper": round(confidence_interval[1], 4),
        }

    return observation


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_jsonld(document: Dict[str, Any]) -> bool:
    """Validate JSON-LD via compaction/expansion round-trip.

    Returns True if the document survives expansion → compaction and
    retains its essential structure.
    """
    pyld = _get_pyld()
    jsonld = pyld.jsonld

    try:
        # Expand the document (removes @context, resolves all IRIs)
        expanded = jsonld.expand(document)

        if not expanded:
            return False

        # Compact back with original context
        context = document.get("@context", {})
        compacted = jsonld.compact(expanded, context)

        # Verify essential fields survived the round-trip
        if "@type" not in compacted:
            return False

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_jsonld(
    multi_task_output: Any,
    ecg_metadata: Optional[Dict[str, Any]] = None,
    *,
    model_version: str = "unknown",
    input_file_hash: Optional[str] = None,
    confidence_threshold: float = 0.30,
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> JSONLDReport:
    """Generate a JSON-LD machine-readable report from multi-task predictions.

    Parameters
    ----------
    multi_task_output:
        Multi-task model predictions (``MultiTaskOutput``, dict, or
        similar structure with ``rhythm``, ``structural``, ``ischaemia``,
        ``risk`` keys/attributes).
    ecg_metadata:
        Optional dict with keys like ``acquisition_datetime``, ``device``,
        ``patient_id``, ``file_path``.
    model_version:
        Model version string for provenance.
    input_file_hash:
        SHA-256 hash of the input ECG file.  If ``None`` and
        ``ecg_metadata['file_path']`` is provided, the hash is computed.
    confidence_threshold:
        Minimum confidence for a classification finding to be included.
    confidence_intervals:
        Optional mapping of class name → (lower, upper) confidence bounds.

    Returns
    -------
    JSONLDReport
        Container with the JSON-LD document, serialised string, and
        validation status.
    """
    ecg_metadata = ecg_metadata or {}
    confidence_intervals = confidence_intervals or {}

    now_iso = datetime.now(timezone.utc).isoformat()
    inference_timestamp = ecg_metadata.get("acquisition_datetime", now_iso)

    # Compute input file hash if not provided
    if input_file_hash is None:
        file_path = ecg_metadata.get("file_path")
        if file_path is not None:
            try:
                with open(file_path, "rb") as f:
                    input_file_hash = hashlib.sha256(f.read()).hexdigest()
            except (OSError, IOError):
                input_file_hash = "unavailable"
        else:
            input_file_hash = "unavailable"

    # Extract predictions
    predictions = _extract_predictions(multi_task_output)

    # Build observations for classification findings
    observations: List[Dict[str, Any]] = []
    classification_tasks = {
        "rhythm": RHYTHM_CLASSES,
        "structural": STRUCTURAL_CLASSES,
        "ischaemia": ISCHAEMIA_CLASSES,
    }

    for task_name, class_names in classification_tasks.items():
        preds = predictions.get(task_name, [])
        for i, conf in enumerate(preds):
            if i < len(class_names) and conf >= confidence_threshold:
                class_name = class_names[i]
                ci = confidence_intervals.get(class_name)
                obs = _build_finding_observation(
                    class_name=class_name,
                    confidence=conf,
                    task=task_name,
                    confidence_interval=ci,
                )
                observations.append(obs)

    # Build observations for risk predictions (always included)
    risk_preds = predictions.get("risk", [])
    for i, prob in enumerate(risk_preds):
        if i < len(RISK_OUTPUTS):
            risk_name = RISK_OUTPUTS[i]
            ci = confidence_intervals.get(risk_name)
            obs = _build_risk_observation(
                risk_name=risk_name,
                probability=prob,
                confidence_interval=ci,
            )
            observations.append(obs)

    # Build the top-level JSON-LD document
    document: Dict[str, Any] = {
        "@context": _JSONLD_CONTEXT,
        "@type": "MedicalTest",
        "name": "Aortica AI ECG Analysis",
        "description": (
            "AI-assisted multi-task ECG interpretation generated by the "
            "Aortica platform. This is decision support only and requires "
            "clinician review."
        ),
        "identifier": ecg_metadata.get("ecg_id", "unknown"),
        "usedToDiagnose": "Electrocardiogram analysis",
        "dateCreated": inference_timestamp,
        "modelVersion": model_version,
        "inferenceTimestamp": inference_timestamp,
        "inputFileHash": input_file_hash,
        "result": observations,
    }

    # Add optional ECG metadata fields
    if "device" in ecg_metadata:
        document["aortica:device"] = ecg_metadata["device"]
    if "sample_rate" in ecg_metadata:
        document["aortica:sampleRate"] = ecg_metadata["sample_rate"]
    if "duration_seconds" in ecg_metadata:
        document["aortica:durationSeconds"] = ecg_metadata["duration_seconds"]
    if "num_leads" in ecg_metadata:
        document["aortica:numLeads"] = ecg_metadata["num_leads"]
    if "source_format" in ecg_metadata:
        document["aortica:sourceFormat"] = ecg_metadata["source_format"]

    # Serialise
    json_string = json.dumps(document, indent=2, ensure_ascii=False)

    # Validate via round-trip
    is_valid = _validate_jsonld(document)

    return JSONLDReport(
        document=document,
        json_string=json_string,
        is_valid=is_valid,
    )
