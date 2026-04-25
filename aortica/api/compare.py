"""Second Reader Comparison — ``POST /api/v1/compare``.

Accepts a cardiologist's interpretation (structured checkboxes + free-text)
alongside an ECG reference (file upload or cached prediction ID) and
produces a visual diff highlighting agreements, AI-only findings, and
clinician-only findings ranked by clinical importance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants: all model output class names by task
# ---------------------------------------------------------------------------

# Lazy-loaded via _get_all_class_names() to avoid hard torch dep at import
_ALL_CLASS_NAMES: Dict[str, List[str]] = {}


def _get_all_class_names() -> Dict[str, List[str]]:
    """Lazily load canonical class names from model head modules."""
    if _ALL_CLASS_NAMES:
        return _ALL_CLASS_NAMES

    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    _ALL_CLASS_NAMES["rhythm"] = list(RHYTHM_CLASSES)
    _ALL_CLASS_NAMES["structural"] = list(STRUCTURAL_CLASSES)
    _ALL_CLASS_NAMES["ischaemia"] = list(ISCHAEMIA_CLASSES)
    return _ALL_CLASS_NAMES


# ---------------------------------------------------------------------------
# Clinical importance ranking (higher = more urgent)
# ---------------------------------------------------------------------------

# Maps canonical class names → integer priority (10 = life-threatening,
# 1 = incidental).  Anything not listed defaults to 3.
_CLINICAL_IMPORTANCE: Dict[str, int] = {
    # Rhythm — high urgency
    "VF": 10,
    "VT": 9,
    "av_block_3rd": 9,
    "WPW": 8,
    "av_block_2nd": 7,
    "AVRT": 7,
    "AVNRT": 7,
    "AF": 6,
    "AFL": 6,
    "SVT": 5,
    "LBBB": 5,
    "RBBB": 4,
    "av_block_1st": 4,
    "LAFB": 3,
    "LPFB": 3,
    "sinus_tachy": 3,
    "sinus_brady": 3,
    "PAC": 2,
    "PVC": 2,
    "pacemaker_rhythm": 2,
    "idioventricular": 6,
    "normal_sinus_rhythm": 1,
    # Structural — high urgency
    "LVSD": 8,
    "HCM": 7,
    "ARVC": 7,
    "DCM": 7,
    "amyloidosis": 7,
    "aortic_stenosis": 6,
    "pulmonary_HTN": 6,
    "LVH": 5,
    "RVH": 5,
    "mitral_regurgitation": 5,
    "HFpEF_risk": 5,
    "LA_enlargement": 4,
    "RA_enlargement": 4,
    "pericarditis": 5,
    "myocarditis": 7,
    # Ischaemia — high urgency
    "STEMI": 10,
    "posterior_MI": 9,
    "occlusive_NSTEMI": 9,
    "old_MI": 5,
    "hyperkalaemia": 8,
    "hypokalaemia": 6,
    "hypercalcaemia": 6,
    "hypothyroidism_pattern": 3,
    "digitalis_effect": 4,
    "QTc_prolongation": 7,
}

DEFAULT_IMPORTANCE = 3


def _importance(class_name: str) -> int:
    """Return clinical importance score for *class_name*."""
    return _CLINICAL_IMPORTANCE.get(class_name, DEFAULT_IMPORTANCE)


# ---------------------------------------------------------------------------
# Alias map: human-friendly labels → canonical class names
# ---------------------------------------------------------------------------

# Allows flexible matching of cardiologist input (e.g. "Atrial Fibrillation"
# matches "AF", "Left Bundle Branch Block" matches "LBBB").
_ALIASES: Dict[str, str] = {
    # Rhythm aliases
    "atrial fibrillation": "AF",
    "atrial flutter": "AFL",
    "supraventricular tachycardia": "SVT",
    "av nodal reentrant tachycardia": "AVNRT",
    "av reentrant tachycardia": "AVRT",
    "ventricular tachycardia": "VT",
    "ventricular fibrillation": "VF",
    "idioventricular rhythm": "idioventricular",
    "sinus bradycardia": "sinus_brady",
    "sinus tachycardia": "sinus_tachy",
    "premature atrial complex": "PAC",
    "premature ventricular complex": "PVC",
    "first degree av block": "av_block_1st",
    "1st degree av block": "av_block_1st",
    "second degree av block": "av_block_2nd",
    "2nd degree av block": "av_block_2nd",
    "third degree av block": "av_block_3rd",
    "3rd degree av block": "av_block_3rd",
    "complete heart block": "av_block_3rd",
    "left bundle branch block": "LBBB",
    "right bundle branch block": "RBBB",
    "left anterior fascicular block": "LAFB",
    "left posterior fascicular block": "LPFB",
    "wolff-parkinson-white": "WPW",
    "wpw": "WPW",
    "pacemaker rhythm": "pacemaker_rhythm",
    "normal sinus rhythm": "normal_sinus_rhythm",
    "nsr": "normal_sinus_rhythm",
    # Structural aliases
    "left ventricular hypertrophy": "LVH",
    "lvh": "LVH",
    "right ventricular hypertrophy": "RVH",
    "rvh": "RVH",
    "lv systolic dysfunction": "LVSD",
    "lvsd": "LVSD",
    "hfpef risk": "HFpEF_risk",
    "dilated cardiomyopathy": "DCM",
    "dcm": "DCM",
    "hypertrophic cardiomyopathy": "HCM",
    "hcm": "HCM",
    "arvc": "ARVC",
    "arrhythmogenic right ventricular cardiomyopathy": "ARVC",
    "cardiac amyloidosis": "amyloidosis",
    "amyloidosis": "amyloidosis",
    "aortic stenosis": "aortic_stenosis",
    "mitral regurgitation": "mitral_regurgitation",
    "pulmonary hypertension": "pulmonary_HTN",
    "la enlargement": "LA_enlargement",
    "left atrial enlargement": "LA_enlargement",
    "ra enlargement": "RA_enlargement",
    "right atrial enlargement": "RA_enlargement",
    "pericarditis": "pericarditis",
    "myocarditis": "myocarditis",
    # Ischaemia aliases
    "stemi": "STEMI",
    "st elevation mi": "STEMI",
    "posterior mi": "posterior_MI",
    "nstemi": "occlusive_NSTEMI",
    "occlusive nstemi": "occlusive_NSTEMI",
    "old mi": "old_MI",
    "old myocardial infarction": "old_MI",
    "hyperkalemia": "hyperkalaemia",
    "hyperkalaemia": "hyperkalaemia",
    "hypokalemia": "hypokalaemia",
    "hypokalaemia": "hypokalaemia",
    "hypercalcemia": "hypercalcaemia",
    "hypercalcaemia": "hypercalcaemia",
    "hypothyroidism": "hypothyroidism_pattern",
    "hypothyroidism pattern": "hypothyroidism_pattern",
    "digitalis effect": "digitalis_effect",
    "digoxin effect": "digitalis_effect",
    "qt prolongation": "QTc_prolongation",
    "qtc prolongation": "QTc_prolongation",
    "long qt": "QTc_prolongation",
}


def _normalise_finding(name: str) -> Optional[str]:
    """Map a clinician-entered finding name to a canonical class name.

    Tries exact match first, then lowercase alias lookup, then case-
    insensitive canonical match.  Returns ``None`` if no match found.
    """
    # 1. Exact match against canonical names
    all_classes = _get_all_class_names()
    all_canonical: set[str] = set()
    for class_list in all_classes.values():
        all_canonical.update(class_list)

    if name in all_canonical:
        return name

    # 2. Alias lookup
    lower = name.lower().strip()
    if lower in _ALIASES:
        return _ALIASES[lower]

    # 3. Case-insensitive canonical match
    for canonical in all_canonical:
        if canonical.lower() == lower:
            return canonical

    return None


def _class_to_task(class_name: str) -> str:
    """Return the task head name for a canonical class name."""
    all_classes = _get_all_class_names()
    for task, classes in all_classes.items():
        if class_name in classes:
            return task
    return "unknown"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ClinicianInterpretation(BaseModel):
    """Structured cardiologist interpretation input."""

    findings: List[str] = Field(
        default_factory=list,
        description="List of finding names selected by the clinician "
        "(can be canonical class names or human-friendly labels)",
    )
    free_text: str = Field(
        default="",
        description="Optional free-text notes from the clinician",
    )


class CompareRequest(BaseModel):
    """Request body for ``POST /api/v1/compare``."""

    interpretation: ClinicianInterpretation = Field(
        ..., description="Clinician's interpretation"
    )
    ai_predictions: Dict[str, List[float]] = Field(
        ...,
        description="AI predictions per task (task → list of probabilities)",
    )
    threshold: float = Field(
        default=0.50,
        description="Probability threshold to consider an AI finding positive",
    )


class DiscrepancyItem(BaseModel):
    """A single agreement or discrepancy between AI and clinician."""

    class_name: str = Field(..., description="Canonical class name")
    task: str = Field(..., description="Task head (rhythm/structural/ischaemia)")
    status: str = Field(
        ...,
        description="'agreement', 'ai_only' (AI found, clinician missed), "
        "or 'clinician_only' (clinician found, AI didn't)",
    )
    ai_probability: Optional[float] = Field(
        default=None,
        description="AI's predicted probability (None for clinician-only with unknown mapping)",
    )
    clinician_selected: bool = Field(
        ..., description="Whether clinician selected this finding"
    )
    clinical_importance: int = Field(
        ..., description="Clinical importance score (1–10)"
    )


class CompareResponse(BaseModel):
    """Response from the comparison endpoint."""

    agreements: List[DiscrepancyItem] = Field(
        ..., description="Findings where AI and clinician agree"
    )
    ai_only: List[DiscrepancyItem] = Field(
        ...,
        description="AI detected but clinician did not select "
        "(red — potential missed findings)",
    )
    clinician_only: List[DiscrepancyItem] = Field(
        ...,
        description="Clinician selected but AI did not detect "
        "(yellow — AI may have missed)",
    )
    summary: Dict[str, int] = Field(
        ...,
        description="Counts: total_agreements, total_ai_only, total_clinician_only",
    )
    unmatched_clinician_inputs: List[str] = Field(
        default_factory=list,
        description="Clinician inputs that could not be mapped to any canonical class",
    )


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_interpretations(
    interpretation: ClinicianInterpretation,
    ai_predictions: Dict[str, List[float]],
    threshold: float = 0.50,
) -> CompareResponse:
    """Compare a clinician's interpretation against AI predictions.

    Parameters
    ----------
    interpretation:
        Structured clinician input with a list of finding names + free text.
    ai_predictions:
        Per-task probability arrays from the AI model, keyed by task name.
    threshold:
        Probability threshold above which an AI prediction is considered
        positive.  Default ``0.50``.

    Returns
    -------
    CompareResponse
        Categorised comparison with agreements, AI-only, and clinician-only
        findings sorted by clinical importance (most important first).
    """
    all_classes = _get_all_class_names()

    # --- Build clinician finding set ---
    clinician_canonical: set[str] = set()
    unmatched: list[str] = []
    for finding in interpretation.findings:
        canonical = _normalise_finding(finding)
        if canonical is not None:
            clinician_canonical.add(canonical)
        else:
            unmatched.append(finding)

    # --- Build AI positive finding set ---
    ai_positive: Dict[str, float] = {}  # canonical → probability
    ai_all: Dict[str, float] = {}  # canonical → probability (for lookup)

    for task, class_list in all_classes.items():
        probs = ai_predictions.get(task, [])
        for i, class_name in enumerate(class_list):
            prob = probs[i] if i < len(probs) else 0.0
            ai_all[class_name] = prob
            if prob >= threshold:
                ai_positive[class_name] = prob

    # --- Categorise ---
    agreements: list[DiscrepancyItem] = []
    ai_only: list[DiscrepancyItem] = []
    clinician_only: list[DiscrepancyItem] = []

    # Classes that both AI and clinician flagged
    for class_name in clinician_canonical & set(ai_positive.keys()):
        agreements.append(
            DiscrepancyItem(
                class_name=class_name,
                task=_class_to_task(class_name),
                status="agreement",
                ai_probability=ai_all.get(class_name),
                clinician_selected=True,
                clinical_importance=_importance(class_name),
            )
        )

    # AI found but clinician did not
    for class_name, prob in ai_positive.items():
        if class_name not in clinician_canonical:
            ai_only.append(
                DiscrepancyItem(
                    class_name=class_name,
                    task=_class_to_task(class_name),
                    status="ai_only",
                    ai_probability=prob,
                    clinician_selected=False,
                    clinical_importance=_importance(class_name),
                )
            )

    # Clinician found but AI did not
    for class_name in clinician_canonical:
        if class_name not in ai_positive:
            clinician_only.append(
                DiscrepancyItem(
                    class_name=class_name,
                    task=_class_to_task(class_name),
                    status="clinician_only",
                    ai_probability=ai_all.get(class_name),
                    clinician_selected=True,
                    clinical_importance=_importance(class_name),
                )
            )

    # Sort each list by clinical importance descending
    agreements.sort(key=lambda x: x.clinical_importance, reverse=True)
    ai_only.sort(key=lambda x: x.clinical_importance, reverse=True)
    clinician_only.sort(key=lambda x: x.clinical_importance, reverse=True)

    return CompareResponse(
        agreements=agreements,
        ai_only=ai_only,
        clinician_only=clinician_only,
        summary={
            "total_agreements": len(agreements),
            "total_ai_only": len(ai_only),
            "total_clinician_only": len(clinician_only),
        },
        unmatched_clinician_inputs=unmatched,
    )
