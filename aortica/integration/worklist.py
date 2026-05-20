"""AI-driven Worklist Prioritization Module.

Provides :class:`WorklistPrioritizer`, which sorts a batch of multi-task
ECG inference results by clinical urgency so that the most critical ECGs
are reviewed first.

Each ECG is assigned an **urgency score** (0–100) based on configurable
rules mapping findings to priority tiers:

    * **Critical** (80–100): life-threatening findings requiring immediate
      attention (e.g. STEMI, VT, VF, severe hyperkalaemia).
    * **High** (60–79): dangerous conditions needing prompt review
      (e.g. new AF, Brugada, Wellens, complete AV block).
    * **Moderate** (40–59): significant findings warranting timely follow-up.
    * **Low** (0–39): routine findings with no immediate concern.

Urgency rules are configurable via a YAML file for site-specific
customization.

Example::

    from aortica.integration.worklist import WorklistPrioritizer

    prioritizer = WorklistPrioritizer()
    worklist = prioritizer.prioritize(results)
    for item in worklist.items:
        print(f"{item.ecg_id}: urgency={item.urgency_score}, "
              f"top={item.top_finding}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:  # pragma: no cover
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Class lists — kept in sync with the model head modules.
# ---------------------------------------------------------------------------

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
    # Phase 3 strain pattern sub-classifiers (US-074)
    "LV_strain_grade", "RV_strain_PE", "Takotsubo_pattern",
    "infiltrative_cardiomyopathy_strain",
]

_ISCHAEMIA_CLASSES: list[str] = [
    "STEMI", "posterior_MI", "occlusive_NSTEMI", "old_MI", "hyperkalaemia",
    "hypokalaemia", "hypercalcaemia", "hypothyroidism_pattern",
    "digitalis_effect", "QTc_prolongation",
    # Phase 3 STEMI mimics & subtle ischaemia subtypes (US-073)
    "early_repol_vs_STEMI", "de_Winter_T_wave", "Wellens_syndrome",
    "aVR_ST_elevation", "Sgarbossa_criteria",
    # Phase 3 metabolic & drug effect detectors (US-075)
    "hyperkalaemia_severity_grade", "hypothermia_osborn_waves",
    "TCA_toxicity", "digoxin_effect_vs_toxicity",
]

_RISK_OUTPUTS: list[str] = [
    "mortality_1y", "hf_hosp_12m", "af_onset_12m",
    # Phase 3 risk refinement (US-076)
    "ecg_predicted_ef", "conduction_disease_trajectory",
    "sudden_cardiac_death_risk",
]

# ---------------------------------------------------------------------------
# Default urgency rules
# ---------------------------------------------------------------------------

#: Conditions triggering **critical** urgency (score 80–100).
_DEFAULT_CRITICAL_CONDITIONS: dict[str, float] = {
    # Life-threatening arrhythmias
    "VT": 0.40,
    "VF": 0.30,
    "av_block_3rd": 0.50,
    "CPVT": 0.40,
    # Acute coronary syndromes
    "STEMI": 0.40,
    "posterior_MI": 0.50,
    "occlusive_NSTEMI": 0.50,
    # Metabolic emergencies
    "hyperkalaemia": 0.40,
    "hyperkalaemia_severity_grade": 0.40,
    # STEMI mimics — high urgency
    "de_Winter_T_wave": 0.40,
    "aVR_ST_elevation": 0.40,
    # Drug toxicity
    "TCA_toxicity": 0.40,
}

#: Conditions triggering **high** urgency (score 60–79).
_DEFAULT_HIGH_CONDITIONS: dict[str, float] = {
    # Dangerous arrhythmias / conduction
    "AF": 0.50,
    "AFL": 0.50,
    "WPW": 0.40,
    "brugada_pattern": 0.40,
    "short_QT_syndrome": 0.40,
    "fascicular_VT": 0.40,
    "av_block_2nd": 0.50,
    # STEMI mimics — high concern
    "Wellens_syndrome": 0.40,
    "Sgarbossa_criteria": 0.40,
    # Structural — acute
    "Takotsubo_pattern": 0.40,
    "RV_strain_PE": 0.40,
    "LVSD": 0.40,
    "myocarditis": 0.50,
    "pericarditis": 0.50,
}

#: Conditions triggering **moderate** urgency (score 40–59).
_DEFAULT_MODERATE_CONDITIONS: dict[str, float] = {
    # Arrhythmias — moderate concern
    "SVT": 0.50,
    "AVNRT": 0.50,
    "AVRT": 0.50,
    "idioventricular": 0.50,
    "sinus_brady": 0.60,
    "sinus_tachy": 0.60,
    "atypical_atrial_flutter": 0.50,
    "inappropriate_sinus_tachy": 0.60,
    # Structural — moderate concern
    "LVH": 0.60,
    "RVH": 0.60,
    "HCM": 0.50,
    "ARVC": 0.50,
    "DCM": 0.50,
    "amyloidosis": 0.50,
    "aortic_stenosis": 0.50,
    "pulmonary_HTN": 0.50,
    "HFpEF_risk": 0.60,
    "mitral_regurgitation": 0.60,
    "LA_enlargement": 0.60,
    "RA_enlargement": 0.60,
    "LV_strain_grade": 0.50,
    "infiltrative_cardiomyopathy_strain": 0.50,
    # Ischaemia — non-acute
    "old_MI": 0.50,
    "early_repol_vs_STEMI": 0.50,
    "QTc_prolongation": 0.50,
    "hypokalaemia": 0.50,
    "hypercalcaemia": 0.50,
    "digitalis_effect": 0.50,
    "hypothyroidism_pattern": 0.60,
    "hypothermia_osborn_waves": 0.50,
    "digoxin_effect_vs_toxicity": 0.50,
    # Conduction
    "av_block_1st": 0.60,
    "LBBB": 0.60,
    "RBBB": 0.60,
    "LAFB": 0.60,
    "LPFB": 0.60,
}

#: Risk score thresholds for urgency escalation.
_DEFAULT_RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "mortality_1y": {"critical": 0.70, "high": 0.50, "moderate": 0.30},
    "hf_hosp_12m": {"critical": 0.70, "high": 0.50, "moderate": 0.30},
    "af_onset_12m": {"critical": 0.80, "high": 0.60, "moderate": 0.40},
    "ecg_predicted_ef": {"critical": 0.70, "high": 0.50, "moderate": 0.30},
    "conduction_disease_trajectory": {
        "critical": 0.70, "high": 0.50, "moderate": 0.30,
    },
    "sudden_cardiac_death_risk": {
        "critical": 0.60, "high": 0.40, "moderate": 0.20,
    },
}

#: Recommended actions per urgency tier.
_DEFAULT_ACTIONS: dict[str, str] = {
    "critical": "Immediate review required — potential life-threatening finding",
    "high": "Priority review recommended — urgent clinical attention needed",
    "moderate": "Timely review recommended — schedule follow-up",
    "low": "Routine review — no urgent findings detected",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UrgencyRules:
    """Configurable urgency rules for worklist prioritization.

    Attributes:
        critical_conditions: Condition name → confidence threshold for
            critical urgency (score 80–100).
        high_conditions: Condition name → confidence threshold for
            high urgency (score 60–79).
        moderate_conditions: Condition name → confidence threshold for
            moderate urgency (score 40–59).
        risk_thresholds: Risk output name → dict with 'critical', 'high',
            and 'moderate' threshold values.
        actions: Mapping of urgency tier → recommended action string.
    """

    critical_conditions: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CRITICAL_CONDITIONS),
    )
    high_conditions: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_HIGH_CONDITIONS),
    )
    moderate_conditions: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MODERATE_CONDITIONS),
    )
    risk_thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_RISK_THRESHOLDS.items()
        },
    )
    actions: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_ACTIONS),
    )


@dataclass
class WorklistItem:
    """A single ECG entry in the prioritized worklist.

    Attributes:
        ecg_id: Identifier for this ECG record.
        urgency_score: Urgency score 0–100 (higher = more urgent).
        urgency_tier: Tier label: 'critical', 'high', 'moderate', or 'low'.
        top_finding: The highest-confidence clinically significant finding.
        top_finding_confidence: Confidence of the top finding (0.0–1.0).
        recommended_action: Plain-language recommended action string.
        active_findings: List of (class_name, confidence, task, tier) tuples
            for all findings that contributed to urgency.
    """

    ecg_id: str
    urgency_score: int
    urgency_tier: str
    top_finding: str
    top_finding_confidence: float
    recommended_action: str
    active_findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a plain dictionary."""
        return {
            "ecg_id": self.ecg_id,
            "urgency_score": self.urgency_score,
            "urgency_tier": self.urgency_tier,
            "top_finding": self.top_finding,
            "top_finding_confidence": self.top_finding_confidence,
            "recommended_action": self.recommended_action,
            "active_findings": list(self.active_findings),
        }


@dataclass
class PrioritizedWorklist:
    """A prioritized list of ECGs sorted by clinical urgency.

    Attributes:
        items: List of :class:`WorklistItem` sorted by urgency (descending).
        total_count: Total number of ECGs in the worklist.
        critical_count: Number of ECGs with critical urgency.
        high_count: Number of ECGs with high urgency.
        moderate_count: Number of ECGs with moderate urgency.
        low_count: Number of ECGs with low urgency.
    """

    items: list[WorklistItem] = field(default_factory=list)
    total_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    moderate_count: int = 0
    low_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a plain dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "moderate_count": self.moderate_count,
            "low_count": self.low_count,
        }


# ---------------------------------------------------------------------------
# Prediction extraction helpers (shared with simplified_output.py)
# ---------------------------------------------------------------------------

def _get_task_predictions(output: Any, task_name: str) -> Any:
    """Extract predictions for a single task from a multi-task output.

    Handles both dataclass-style (attribute access) and dict-style outputs.
    """
    if isinstance(output, dict):
        return output.get(task_name)
    return getattr(output, task_name, None)


def _extract_predictions(
    multi_task_output: Any,
) -> dict[str, dict[str, float]]:
    """Extract per-class confidence values from a MultiTaskOutput-like object.

    Supports:
    - ``MultiTaskOutput`` dataclass (tensor fields).
    - Plain ``dict[str, list[float] | None]``.
    - ``dict`` with nested ``dict[str, float]`` per-task mappings.

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
            if hasattr(preds, "detach"):
                values = preds.detach().cpu().squeeze().tolist()
            elif hasattr(preds, "tolist"):
                values = preds.squeeze().tolist()  # type: ignore[union-attr]
            else:
                values = list(preds)
        except (TypeError, AttributeError):
            continue

        if isinstance(values, (int, float)):
            values = [float(values)]

        if len(values) != len(class_list):
            continue

        result[task_name] = {
            cls: float(val) for cls, val in zip(class_list, values)
        }

    return result


# ---------------------------------------------------------------------------
# YAML rule loading
# ---------------------------------------------------------------------------

def load_urgency_rules(yaml_path: str | Path) -> UrgencyRules:
    """Load urgency rules from a YAML file.

    The YAML file should have keys matching :class:`UrgencyRules` fields:
    ``critical_conditions``, ``high_conditions``, ``moderate_conditions``,
    ``risk_thresholds``, and ``actions``.  Any missing key falls back to
    the built-in defaults.

    Args:
        yaml_path: Path to the YAML rules file.

    Returns:
        An :class:`UrgencyRules` instance with loaded values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ImportError: If ``pyyaml`` is not installed.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to load urgency rules from YAML. "
            "Install with: pip install pyyaml"
        )

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Urgency rules file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    rules = UrgencyRules()
    if "critical_conditions" in data:
        rules.critical_conditions = {
            str(k): float(v) for k, v in data["critical_conditions"].items()
        }
    if "high_conditions" in data:
        rules.high_conditions = {
            str(k): float(v) for k, v in data["high_conditions"].items()
        }
    if "moderate_conditions" in data:
        rules.moderate_conditions = {
            str(k): float(v) for k, v in data["moderate_conditions"].items()
        }
    if "risk_thresholds" in data:
        rules.risk_thresholds = {}
        for k, v in data["risk_thresholds"].items():
            rules.risk_thresholds[str(k)] = {
                str(tk): float(tv) for tk, tv in v.items()
            }
    if "actions" in data:
        rules.actions = {str(k): str(v) for k, v in data["actions"].items()}

    return rules


# ---------------------------------------------------------------------------
# Core prioritization logic
# ---------------------------------------------------------------------------


class WorklistPrioritizer:
    """AI-driven worklist prioritizer sorting ECGs by clinical urgency.

    The prioritizer scores each ECG result (0–100) based on the detected
    findings and risk scores, then sorts by urgency descending.

    Args:
        rules: Optional :class:`UrgencyRules`.  If ``None``, built-in
            clinical defaults are used.
        rules_yaml: Optional path to a YAML urgency rules file.
            If both ``rules`` and ``rules_yaml`` are provided, ``rules``
            takes precedence.

    Example::

        prioritizer = WorklistPrioritizer()
        worklist = prioritizer.prioritize(results)
        for item in worklist.items:
            print(item.ecg_id, item.urgency_score, item.top_finding)
    """

    def __init__(
        self,
        rules: Optional[UrgencyRules] = None,
        rules_yaml: Optional[str | Path] = None,
    ) -> None:
        if rules is not None:
            self.rules = rules
        elif rules_yaml is not None:
            self.rules = load_urgency_rules(rules_yaml)
        else:
            self.rules = UrgencyRules()

    def _score_single(
        self,
        predictions: dict[str, dict[str, float]],
    ) -> tuple[int, str, list[dict[str, Any]]]:
        """Score a single ECG's urgency.

        Returns:
            (urgency_score, urgency_tier, active_findings)
        """
        max_tier_level = 0  # 0=low, 1=moderate, 2=high, 3=critical
        active_findings: list[dict[str, Any]] = []

        # Check classification tasks (rhythm, structural, ischaemia)
        for task_name in ("rhythm", "structural", "ischaemia"):
            task_preds = predictions.get(task_name, {})
            for class_name, confidence in task_preds.items():
                # Check critical conditions first
                if class_name in self.rules.critical_conditions:
                    threshold = self.rules.critical_conditions[class_name]
                    if confidence >= threshold:
                        max_tier_level = max(max_tier_level, 3)
                        active_findings.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "task": task_name,
                            "tier": "critical",
                        })
                        continue

                # Check high conditions
                if class_name in self.rules.high_conditions:
                    threshold = self.rules.high_conditions[class_name]
                    if confidence >= threshold:
                        max_tier_level = max(max_tier_level, 2)
                        active_findings.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "task": task_name,
                            "tier": "high",
                        })
                        continue

                # Check moderate conditions
                if class_name in self.rules.moderate_conditions:
                    threshold = self.rules.moderate_conditions[class_name]
                    if confidence >= threshold:
                        max_tier_level = max(max_tier_level, 1)
                        active_findings.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "task": task_name,
                            "tier": "moderate",
                        })

        # Check risk scores
        risk_preds = predictions.get("risk", {})
        for risk_name, score in risk_preds.items():
            thresholds = self.rules.risk_thresholds.get(risk_name, {})
            critical_thresh = thresholds.get("critical", 1.0)
            high_thresh = thresholds.get("high", 1.0)
            moderate_thresh = thresholds.get("moderate", 1.0)

            if score >= critical_thresh:
                max_tier_level = max(max_tier_level, 3)
                active_findings.append({
                    "class_name": risk_name,
                    "confidence": score,
                    "task": "risk",
                    "tier": "critical",
                })
            elif score >= high_thresh:
                max_tier_level = max(max_tier_level, 2)
                active_findings.append({
                    "class_name": risk_name,
                    "confidence": score,
                    "task": "risk",
                    "tier": "high",
                })
            elif score >= moderate_thresh:
                max_tier_level = max(max_tier_level, 1)
                active_findings.append({
                    "class_name": risk_name,
                    "confidence": score,
                    "task": "risk",
                    "tier": "moderate",
                })

        # Map tier level to tier name
        tier_map = {0: "low", 1: "moderate", 2: "high", 3: "critical"}
        urgency_tier = tier_map[max_tier_level]

        # Compute urgency score (0–100)
        urgency_score = self._compute_score(
            max_tier_level, active_findings,
        )

        # Sort findings: highest tier first, then by confidence
        tier_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
        active_findings.sort(
            key=lambda f: (
                tier_order.get(f["tier"], 3),
                -f["confidence"],
            ),
        )

        return urgency_score, urgency_tier, active_findings

    def _compute_score(
        self,
        tier_level: int,
        findings: list[dict[str, Any]],
    ) -> int:
        """Compute the 0–100 urgency score.

        Score bands:
            critical: 80–100
            high:     60–79
            moderate: 40–59
            low:       0–39

        Within each band, the score is boosted by the number and confidence
        of active findings.
        """
        if tier_level == 0:
            # Low — baseline 0, but can go up to 39 based on findings
            if not findings:
                return 0
            max_conf = max(f["confidence"] for f in findings)
            return min(39, int(max_conf * 39))

        band_min = {1: 40, 2: 60, 3: 80}[tier_level]
        band_max = {1: 59, 2: 79, 3: 100}[tier_level]
        band_range = band_max - band_min

        # Base score is band_min; boost by highest confidence in this tier
        tier_name = {1: "moderate", 2: "high", 3: "critical"}[tier_level]
        tier_findings = [
            f for f in findings if f["tier"] == tier_name
        ]

        if not tier_findings:
            return band_min

        max_conf = max(f["confidence"] for f in tier_findings)
        n_findings = len(tier_findings)

        # Scale within band: confidence contributes 70%, count 30%
        conf_boost = max_conf * 0.7
        count_boost = min(n_findings / 5.0, 1.0) * 0.3
        total_boost = conf_boost + count_boost

        score = band_min + int(total_boost * band_range)
        return min(band_max, score)

    def prioritize(
        self,
        results: list[Any],
        ecg_ids: Optional[list[str]] = None,
    ) -> PrioritizedWorklist:
        """Prioritize a list of multi-task ECG results by urgency.

        Args:
            results: List of ``MultiTaskOutput`` dataclasses, dicts, or
                any objects with task-named attributes. Each element
                represents one ECG's inference output.
            ecg_ids: Optional list of ECG identifiers matching ``results``.
                If ``None``, sequential IDs (``ecg_001``, ``ecg_002``, …)
                are generated.

        Returns:
            A :class:`PrioritizedWorklist` sorted by urgency (descending).
        """
        if ecg_ids is None:
            ecg_ids = [
                f"ecg_{i + 1:03d}" for i in range(len(results))
            ]

        if len(ecg_ids) != len(results):
            raise ValueError(
                f"ecg_ids length ({len(ecg_ids)}) must match "
                f"results length ({len(results)})"
            )

        items: list[WorklistItem] = []

        for ecg_id, result in zip(ecg_ids, results):
            predictions = _extract_predictions(result)
            urgency_score, urgency_tier, active_findings = self._score_single(
                predictions,
            )

            # Determine top finding
            if active_findings:
                top = active_findings[0]
                top_finding = top["class_name"]
                top_confidence = top["confidence"]
            else:
                top_finding = "normal_sinus_rhythm"
                top_confidence = 0.0

            # Get recommended action
            action = self.rules.actions.get(
                urgency_tier,
                _DEFAULT_ACTIONS.get(urgency_tier, ""),
            )

            items.append(WorklistItem(
                ecg_id=ecg_id,
                urgency_score=urgency_score,
                urgency_tier=urgency_tier,
                top_finding=top_finding,
                top_finding_confidence=top_confidence,
                recommended_action=action,
                active_findings=active_findings,
            ))

        # Sort by urgency score descending, then by ECG ID for stability
        items.sort(key=lambda x: (-x.urgency_score, x.ecg_id))

        # Count tiers
        critical_count = sum(1 for i in items if i.urgency_tier == "critical")
        high_count = sum(1 for i in items if i.urgency_tier == "high")
        moderate_count = sum(1 for i in items if i.urgency_tier == "moderate")
        low_count = sum(1 for i in items if i.urgency_tier == "low")

        return PrioritizedWorklist(
            items=items,
            total_count=len(items),
            critical_count=critical_count,
            high_count=high_count,
            moderate_count=moderate_count,
            low_count=low_count,
        )
