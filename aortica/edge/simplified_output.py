"""CHW-Facing Simplified Output Interface.

Provides :func:`simplify_output`, which maps multi-task AI predictions into
three plain-language risk tiers suitable for community health workers (CHWs)
without cardiology training.

The three tiers are:

    1. **Low risk** — no immediate action required.
    2. **Refer for assessment** — schedule a follow-up with a clinician.
    3. **Urgent referral** — seek immediate medical care.

Tier assignment is driven by configurable per-condition confidence thresholds
and the clinical importance of each detected finding.  Localization is
supported via JSON locale files.

Example::

    from aortica.edge.simplified_output import simplify_output

    report = simplify_output(multi_task_output)
    print(report.tier)      # 'urgent'
    print(report.summary)   # 'Urgent referral recommended ...'
    print(report.actions)   # ['Seek immediate medical care', ...]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid tier names, ordered from lowest to highest severity.
TIER_LOW: str = "low"
TIER_REFER: str = "refer"
TIER_URGENT: str = "urgent"

VALID_TIERS: list[str] = [TIER_LOW, TIER_REFER, TIER_URGENT]

#: Task head class lists — kept in sync with the model head modules.
_RHYTHM_CLASSES: list[str] = [
    "AF", "AFL", "SVT", "AVNRT", "AVRT", "VT", "VF", "idioventricular",
    "sinus_brady", "sinus_tachy", "PAC", "PVC", "av_block_1st", "av_block_2nd",
    "av_block_3rd", "LBBB", "RBBB", "LAFB", "LPFB", "WPW",
    "pacemaker_rhythm", "normal_sinus_rhythm",
    # Phase 3 rare arrhythmia subtypes (US-072)
    "brugada_pattern", "short_QT_syndrome", "CPVT", "fascicular_VT",
    "atypical_atrial_flutter", "inappropriate_sinus_tachy",
]

_STRUCTURAL_CLASSES: list[str] = [
    "LVH", "RVH", "LVSD", "HFpEF_risk", "DCM", "HCM", "ARVC", "amyloidosis",
    "aortic_stenosis", "mitral_regurgitation", "pulmonary_HTN",
    "LA_enlargement", "RA_enlargement", "pericarditis", "myocarditis",
]

_ISCHAEMIA_CLASSES: list[str] = [
    "STEMI", "posterior_MI", "occlusive_NSTEMI", "old_MI", "hyperkalaemia",
    "hypokalaemia", "hypercalcaemia", "hypothyroidism_pattern",
    "digitalis_effect", "QTc_prolongation",
]

_RISK_OUTPUTS: list[str] = [
    "mortality_1y", "hf_hosp_12m", "af_onset_12m",
]

# ---------------------------------------------------------------------------
# Default confidence thresholds per condition category
# ---------------------------------------------------------------------------

#: Conditions that trigger **urgent** referral when above their threshold.
_URGENT_CONDITIONS: dict[str, float] = {
    # Rhythm — life-threatening arrhythmias
    "VT": 0.40,
    "VF": 0.30,
    "av_block_3rd": 0.50,
    # Phase 3 rare arrhythmias — high urgency
    "CPVT": 0.40,
    # Ischaemia — acute coronary syndromes
    "STEMI": 0.40,
    "posterior_MI": 0.50,
    "occlusive_NSTEMI": 0.50,
    # Metabolic emergencies
    "hyperkalaemia": 0.40,
}

#: Conditions that trigger **refer** when above their threshold.
_REFER_CONDITIONS: dict[str, float] = {
    # Rhythm — significant but not immediately life-threatening
    "AF": 0.50,
    "AFL": 0.50,
    "SVT": 0.50,
    "AVNRT": 0.50,
    "AVRT": 0.50,
    "WPW": 0.40,
    "av_block_2nd": 0.50,
    "sinus_brady": 0.60,
    "sinus_tachy": 0.60,
    "idioventricular": 0.50,
    # Structural — needs specialist evaluation
    "LVSD": 0.40,
    "HCM": 0.50,
    "ARVC": 0.50,
    "DCM": 0.50,
    "amyloidosis": 0.50,
    "aortic_stenosis": 0.50,
    "pulmonary_HTN": 0.50,
    "pericarditis": 0.50,
    "myocarditis": 0.50,
    # Ischaemia — non-acute but noteworthy
    "old_MI": 0.50,
    "QTc_prolongation": 0.50,
    "hypokalaemia": 0.50,
    "hypercalcaemia": 0.50,
    "digitalis_effect": 0.50,
    # Structural — moderate concern
    "LVH": 0.60,
    "RVH": 0.60,
    "HFpEF_risk": 0.60,
    "mitral_regurgitation": 0.60,
    "LA_enlargement": 0.60,
    "RA_enlargement": 0.60,
    "LBBB": 0.60,
    "RBBB": 0.60,
    "LAFB": 0.60,
    "LPFB": 0.60,
    "hypothyroidism_pattern": 0.60,
    # Phase 3 rare arrhythmias — need specialist evaluation
    "brugada_pattern": 0.40,
    "short_QT_syndrome": 0.40,
    "fascicular_VT": 0.40,
    "atypical_atrial_flutter": 0.50,
    "inappropriate_sinus_tachy": 0.60,
}

#: Risk score thresholds for tier escalation.
_RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "mortality_1y": {"urgent": 0.70, "refer": 0.40},
    "hf_hosp_12m": {"urgent": 0.70, "refer": 0.40},
    "af_onset_12m": {"urgent": 0.80, "refer": 0.50},
}


# ---------------------------------------------------------------------------
# Default locale strings (English)
# ---------------------------------------------------------------------------

_DEFAULT_LOCALE: dict[str, Any] = {
    "tier_labels": {
        "low": "Low risk — no immediate action",
        "refer": "Refer for assessment — schedule follow-up",
        "urgent": "Urgent referral recommended — seek immediate care",
    },
    "summaries": {
        "low": "The ECG analysis shows no findings requiring immediate attention. "
               "Continue routine monitoring as scheduled.",
        "refer": "The ECG analysis detected findings that should be reviewed "
                 "by a clinician. Please schedule a follow-up appointment.",
        "urgent": "The ECG analysis detected potentially serious findings. "
                  "Seek immediate medical evaluation.",
    },
    "actions": {
        "low": [
            "Continue routine monitoring",
            "No immediate action required",
        ],
        "refer": [
            "Schedule follow-up with clinician within 1 week",
            "Share this report with the nearest health facility",
            "Monitor patient for symptom changes",
        ],
        "urgent": [
            "Seek immediate medical care",
            "Transport patient to nearest hospital if possible",
            "Contact emergency medical services",
            "Do not delay — time-critical findings detected",
        ],
    },
    "finding_names": {},  # Optional human-readable finding name overrides
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TierThresholds:
    """Configurable confidence thresholds for tier assignment.

    Attributes:
        urgent_conditions: Mapping of class name → confidence threshold
            for urgent tier escalation.
        refer_conditions: Mapping of class name → confidence threshold
            for referral tier escalation.
        risk_thresholds: Mapping of risk output name → dict with
            'urgent' and 'refer' threshold values.
    """

    urgent_conditions: dict[str, float] = field(
        default_factory=lambda: dict(_URGENT_CONDITIONS),
    )
    refer_conditions: dict[str, float] = field(
        default_factory=lambda: dict(_REFER_CONDITIONS),
    )
    risk_thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _RISK_THRESHOLDS.items()
        },
    )


@dataclass
class KeyFinding:
    """A single finding contributing to the tier decision.

    Attributes:
        class_name: Canonical model class name (e.g. ``'STEMI'``).
        confidence: Model confidence for this class (0.0–1.0).
        task: Which task head produced this finding
            (``'rhythm'``, ``'structural'``, ``'ischaemia'``, ``'risk'``).
        tier_contribution: Which tier this finding contributes to
            (``'urgent'`` or ``'refer'``).
    """

    class_name: str
    confidence: float
    task: str
    tier_contribution: str


@dataclass
class SimplifiedReport:
    """Simplified risk-tier report for community health workers.

    Attributes:
        tier: Risk tier — ``'low'``, ``'refer'``, or ``'urgent'``.
        tier_label: Human-readable tier label.
        summary: 1–2 sentence plain-language summary.
        actions: List of recommended actions.
        key_findings: Findings that drove the tier assignment.
        locale: Locale code used (e.g. ``'en'``).
    """

    tier: str
    tier_label: str
    summary: str
    actions: list[str]
    key_findings: list[KeyFinding] = field(default_factory=list)
    locale: str = "en"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report as a plain dictionary."""
        return {
            "tier": self.tier,
            "tier_label": self.tier_label,
            "summary": self.summary,
            "actions": self.actions,
            "key_findings": [
                {
                    "class_name": f.class_name,
                    "confidence": f.confidence,
                    "task": f.task,
                    "tier_contribution": f.tier_contribution,
                }
                for f in self.key_findings
            ],
            "locale": self.locale,
        }


# ---------------------------------------------------------------------------
# Locale loading
# ---------------------------------------------------------------------------

def load_locale(locale_path: str | Path) -> dict[str, Any]:
    """Load a locale JSON file.

    The JSON must contain ``tier_labels``, ``summaries``, and ``actions``
    keys matching the structure of :data:`_DEFAULT_LOCALE`.

    Args:
        locale_path: Path to a JSON locale file.

    Returns:
        Parsed locale dictionary.

    Raises:
        FileNotFoundError: If the locale file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(locale_path)
    if not path.is_file():
        raise FileNotFoundError(f"Locale file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _get_locale(
    locale: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Resolve locale strings.

    If *locale* is a path to a JSON file, load it.  If ``None``, return
    the built-in English default.

    Args:
        locale: ``None`` for default English, or path to a JSON locale file.

    Returns:
        Locale dictionary.
    """
    if locale is None:
        return dict(_DEFAULT_LOCALE)

    locale_str = str(locale)
    if os.path.isfile(locale_str):
        return load_locale(locale_str)

    # Not a file — treat as unsupported locale code, fall back to default.
    return dict(_DEFAULT_LOCALE)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _extract_predictions(
    multi_task_output: Any,
) -> dict[str, dict[str, float]]:
    """Extract per-class confidence values from a MultiTaskOutput-like object.

    Supports:
    - :class:`MultiTaskOutput` dataclass (with ``rhythm``, ``structural``,
      ``ischaemia``, ``risk`` tensor fields).
    - Plain ``dict[str, list[float] | None]`` for dict-based outputs.
    - ``dict`` with nested ``dict[str, float]`` per-task mappings
      (e.g. from JSON API response).

    Returns:
        ``{task_name: {class_name: confidence, ...}, ...}``
    """
    result: dict[str, dict[str, float]] = {}

    task_class_lists: dict[str, list[str]] = {
        "rhythm": _RHYTHM_CLASSES,
        "structural": _STRUCTURAL_CLASSES,
        "ischaemia": _ISCHAEMIA_CLASSES,
        "risk": _RISK_OUTPUTS,
    }

    for task_name, class_list in task_class_lists.items():
        preds = _get_task_predictions(multi_task_output, task_name)
        if preds is None:
            continue

        # If preds is already a dict, use it directly.
        if isinstance(preds, dict):
            result[task_name] = {
                k: float(v) for k, v in preds.items()
            }
            continue

        # Otherwise treat as a list/array of floats.
        values: list[float]
        try:
            # Handle torch tensors, numpy arrays, and plain lists.
            if hasattr(preds, "detach"):
                # Torch tensor
                values = preds.detach().cpu().squeeze().tolist()
            elif hasattr(preds, "tolist"):
                # Numpy array
                values = preds.squeeze().tolist()  # type: ignore[union-attr]
            else:
                values = list(preds)
        except (TypeError, AttributeError):
            continue

        # Ensure values is a list of floats (handle scalar case).
        if isinstance(values, (int, float)):
            values = [float(values)]

        if len(values) != len(class_list):
            continue

        result[task_name] = {
            cls: float(val) for cls, val in zip(class_list, values)
        }

    return result


def _get_task_predictions(
    output: Any,
    task_name: str,
) -> Any:
    """Extract predictions for a single task from a multi-task output.

    Handles both dataclass-style (attribute access) and dict-style outputs.
    """
    # Dict-style
    if isinstance(output, dict):
        return output.get(task_name)

    # Dataclass / object-style
    return getattr(output, task_name, None)


def simplify_output(
    multi_task_output: Any,
    thresholds: Optional[TierThresholds] = None,
    locale: Optional[str | Path] = None,
) -> SimplifiedReport:
    """Map multi-task AI predictions to a simplified risk tier.

    This function is designed for community health worker (CHW) interfaces.
    The output uses plain language that does not require cardiology training
    to interpret.

    Args:
        multi_task_output: A :class:`MultiTaskOutput` dataclass, a dict
            ``{task: tensor | list | dict}``, or any object with task-named
            attributes.
        thresholds: Optional custom :class:`TierThresholds`.  If ``None``,
            built-in clinical defaults are used.
        locale: ``None`` for English (default), or a path to a JSON locale
            file containing translated output strings.

    Returns:
        A :class:`SimplifiedReport` with the assigned tier, human-readable
        summary, recommended actions, and the key findings that drove the
        tier assignment.

    Example::

        report = simplify_output(model_output)
        print(report.tier)        # 'low', 'refer', or 'urgent'
        print(report.summary)     # Plain-language summary
        print(report.actions)     # Recommended next steps
    """
    if thresholds is None:
        thresholds = TierThresholds()

    locale_data = _get_locale(locale)
    locale_code = "en"
    if locale is not None and os.path.isfile(str(locale)):
        locale_code = Path(locale).stem

    # Extract per-class predictions.
    predictions = _extract_predictions(multi_task_output)

    # Determine tier from findings.
    key_findings: list[KeyFinding] = []
    tier = TIER_LOW

    # Check classification outputs (rhythm, structural, ischaemia).
    for task_name in ("rhythm", "structural", "ischaemia"):
        task_preds = predictions.get(task_name, {})
        for class_name, confidence in task_preds.items():
            # Check urgent conditions first.
            if class_name in thresholds.urgent_conditions:
                threshold = thresholds.urgent_conditions[class_name]
                if confidence >= threshold:
                    tier = TIER_URGENT
                    key_findings.append(
                        KeyFinding(
                            class_name=class_name,
                            confidence=confidence,
                            task=task_name,
                            tier_contribution=TIER_URGENT,
                        )
                    )
                    continue

            # Check refer conditions.
            if class_name in thresholds.refer_conditions:
                threshold = thresholds.refer_conditions[class_name]
                if confidence >= threshold:
                    if tier != TIER_URGENT:
                        tier = TIER_REFER
                    key_findings.append(
                        KeyFinding(
                            class_name=class_name,
                            confidence=confidence,
                            task=task_name,
                            tier_contribution=TIER_REFER,
                        )
                    )

    # Check risk scores.
    risk_preds = predictions.get("risk", {})
    for risk_name, score in risk_preds.items():
        risk_thresholds = thresholds.risk_thresholds.get(risk_name, {})
        urgent_threshold = risk_thresholds.get("urgent", 1.0)
        refer_threshold = risk_thresholds.get("refer", 1.0)

        if score >= urgent_threshold:
            tier = TIER_URGENT
            key_findings.append(
                KeyFinding(
                    class_name=risk_name,
                    confidence=score,
                    task="risk",
                    tier_contribution=TIER_URGENT,
                )
            )
        elif score >= refer_threshold:
            if tier != TIER_URGENT:
                tier = TIER_REFER
            key_findings.append(
                KeyFinding(
                    class_name=risk_name,
                    confidence=score,
                    task="risk",
                    tier_contribution=TIER_REFER,
                )
            )

    # Sort key findings: urgent first, then by confidence descending.
    tier_order = {TIER_URGENT: 0, TIER_REFER: 1, TIER_LOW: 2}
    key_findings.sort(
        key=lambda f: (tier_order.get(f.tier_contribution, 2), -f.confidence),
    )

    # Build report strings from locale.
    tier_labels = locale_data.get("tier_labels", _DEFAULT_LOCALE["tier_labels"])
    summaries = locale_data.get("summaries", _DEFAULT_LOCALE["summaries"])
    actions_map = locale_data.get("actions", _DEFAULT_LOCALE["actions"])

    tier_label = tier_labels.get(tier, tier_labels.get("low", tier))
    summary = summaries.get(tier, summaries.get("low", ""))
    actions = actions_map.get(tier, actions_map.get("low", []))

    return SimplifiedReport(
        tier=tier,
        tier_label=tier_label,
        summary=summary,
        actions=list(actions),
        key_findings=key_findings,
        locale=locale_code,
    )
