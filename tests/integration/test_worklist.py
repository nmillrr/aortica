"""Tests for aortica.integration.worklist — Worklist Prioritization Module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aortica.integration.worklist import (
    PrioritizedWorklist,
    UrgencyRules,
    WorklistItem,
    WorklistPrioritizer,
    _extract_predictions,
    load_urgency_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**kwargs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build a minimal multi-task result dict."""
    return dict(kwargs)


def _make_normal_result() -> dict[str, dict[str, float]]:
    """Build a result with all low-confidence findings."""
    return {
        "rhythm": {"normal_sinus_rhythm": 0.95, "AF": 0.02, "VT": 0.01},
        "structural": {"LVH": 0.05},
        "ischaemia": {"STEMI": 0.01},
        "risk": {"mortality_1y": 0.05, "sudden_cardiac_death_risk": 0.03},
    }


def _make_critical_result() -> dict[str, dict[str, float]]:
    """Build a result with critical findings (STEMI + VF)."""
    return {
        "rhythm": {"VF": 0.85, "VT": 0.60, "normal_sinus_rhythm": 0.01},
        "structural": {"LVSD": 0.30},
        "ischaemia": {"STEMI": 0.92},
        "risk": {"mortality_1y": 0.80, "sudden_cardiac_death_risk": 0.75},
    }


def _make_high_result() -> dict[str, dict[str, float]]:
    """Build a result with high-priority findings (new AF, Wellens)."""
    return {
        "rhythm": {"AF": 0.75, "normal_sinus_rhythm": 0.10},
        "structural": {"LVSD": 0.55},
        "ischaemia": {"Wellens_syndrome": 0.60},
        "risk": {"mortality_1y": 0.25},
    }


def _make_moderate_result() -> dict[str, dict[str, float]]:
    """Build a result with moderate findings (LVH, old MI)."""
    return {
        "rhythm": {"sinus_brady": 0.70, "normal_sinus_rhythm": 0.30},
        "structural": {"LVH": 0.75, "RVH": 0.65},
        "ischaemia": {"old_MI": 0.60},
        "risk": {"mortality_1y": 0.15},
    }


# ---------------------------------------------------------------------------
# Tests: UrgencyRules dataclass
# ---------------------------------------------------------------------------

class TestUrgencyRules:
    def test_default_rules_have_critical_conditions(self) -> None:
        rules = UrgencyRules()
        assert "VT" in rules.critical_conditions
        assert "STEMI" in rules.critical_conditions
        assert "VF" in rules.critical_conditions

    def test_default_rules_have_high_conditions(self) -> None:
        rules = UrgencyRules()
        assert "AF" in rules.high_conditions
        assert "brugada_pattern" in rules.high_conditions
        assert "Wellens_syndrome" in rules.high_conditions

    def test_default_rules_have_moderate_conditions(self) -> None:
        rules = UrgencyRules()
        assert "LVH" in rules.moderate_conditions
        assert "old_MI" in rules.moderate_conditions

    def test_default_rules_have_risk_thresholds(self) -> None:
        rules = UrgencyRules()
        assert "mortality_1y" in rules.risk_thresholds
        assert "critical" in rules.risk_thresholds["mortality_1y"]

    def test_default_actions(self) -> None:
        rules = UrgencyRules()
        assert "critical" in rules.actions
        assert "low" in rules.actions

    def test_custom_rules(self) -> None:
        rules = UrgencyRules(
            critical_conditions={"custom_condition": 0.5},
            high_conditions={},
            moderate_conditions={},
        )
        assert "custom_condition" in rules.critical_conditions
        assert len(rules.high_conditions) == 0


# ---------------------------------------------------------------------------
# Tests: WorklistItem and PrioritizedWorklist dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_worklist_item_to_dict(self) -> None:
        item = WorklistItem(
            ecg_id="ecg_001",
            urgency_score=85,
            urgency_tier="critical",
            top_finding="VF",
            top_finding_confidence=0.90,
            recommended_action="Immediate review",
            active_findings=[{"class_name": "VF", "confidence": 0.9, "task": "rhythm", "tier": "critical"}],
        )
        d = item.to_dict()
        assert d["ecg_id"] == "ecg_001"
        assert d["urgency_score"] == 85
        assert d["urgency_tier"] == "critical"
        assert len(d["active_findings"]) == 1

    def test_prioritized_worklist_to_dict(self) -> None:
        item = WorklistItem(
            ecg_id="ecg_001", urgency_score=50, urgency_tier="moderate",
            top_finding="LVH", top_finding_confidence=0.7,
            recommended_action="Follow up",
        )
        wl = PrioritizedWorklist(
            items=[item], total_count=1, moderate_count=1,
        )
        d = wl.to_dict()
        assert d["total_count"] == 1
        assert d["moderate_count"] == 1
        assert len(d["items"]) == 1


# ---------------------------------------------------------------------------
# Tests: Prediction extraction
# ---------------------------------------------------------------------------

class TestExtractPredictions:
    def test_dict_input(self) -> None:
        result = {"rhythm": {"AF": 0.8, "VT": 0.1}, "risk": {"mortality_1y": 0.3}}
        preds = _extract_predictions(result)
        assert preds["rhythm"]["AF"] == pytest.approx(0.8)
        assert preds["risk"]["mortality_1y"] == pytest.approx(0.3)

    def test_list_input(self) -> None:
        result = {"rhythm": [0.1] * 28}
        preds = _extract_predictions(result)
        assert "rhythm" in preds
        assert len(preds["rhythm"]) == 28

    def test_none_task_skipped(self) -> None:
        result = {"rhythm": None, "risk": {"mortality_1y": 0.5}}
        preds = _extract_predictions(result)
        assert "rhythm" not in preds
        assert "risk" in preds

    def test_wrong_length_skipped(self) -> None:
        result = {"rhythm": [0.1, 0.2]}  # wrong length
        preds = _extract_predictions(result)
        assert "rhythm" not in preds


# ---------------------------------------------------------------------------
# Tests: WorklistPrioritizer
# ---------------------------------------------------------------------------

class TestWorklistPrioritizer:
    def test_default_construction(self) -> None:
        p = WorklistPrioritizer()
        assert p.rules is not None

    def test_custom_rules(self) -> None:
        rules = UrgencyRules(critical_conditions={"test": 0.5})
        p = WorklistPrioritizer(rules=rules)
        assert "test" in p.rules.critical_conditions

    def test_prioritize_empty_list(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([])
        assert wl.total_count == 0
        assert len(wl.items) == 0

    def test_prioritize_single_normal(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_normal_result()], ecg_ids=["n1"])
        assert wl.total_count == 1
        assert wl.items[0].urgency_tier == "low"
        assert wl.items[0].urgency_score < 40

    def test_prioritize_single_critical(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_critical_result()], ecg_ids=["c1"])
        assert wl.total_count == 1
        assert wl.items[0].urgency_tier == "critical"
        assert wl.items[0].urgency_score >= 80

    def test_prioritize_single_high(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_high_result()], ecg_ids=["h1"])
        assert wl.total_count == 1
        assert wl.items[0].urgency_tier == "high"
        assert 60 <= wl.items[0].urgency_score <= 79

    def test_prioritize_single_moderate(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_moderate_result()], ecg_ids=["m1"])
        assert wl.total_count == 1
        assert wl.items[0].urgency_tier == "moderate"
        assert 40 <= wl.items[0].urgency_score <= 59

    def test_sorted_by_urgency_descending(self) -> None:
        p = WorklistPrioritizer()
        results = [
            _make_normal_result(),
            _make_critical_result(),
            _make_moderate_result(),
            _make_high_result(),
        ]
        ids = ["normal", "critical", "moderate", "high"]
        wl = p.prioritize(results, ecg_ids=ids)

        scores = [item.urgency_score for item in wl.items]
        assert scores == sorted(scores, reverse=True)
        assert wl.items[0].ecg_id == "critical"

    def test_tier_counts(self) -> None:
        p = WorklistPrioritizer()
        results = [
            _make_normal_result(),
            _make_critical_result(),
            _make_moderate_result(),
            _make_high_result(),
        ]
        wl = p.prioritize(results)
        assert wl.total_count == 4
        assert wl.critical_count == 1
        assert wl.high_count == 1
        assert wl.moderate_count == 1
        assert wl.low_count == 1

    def test_auto_generated_ecg_ids(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_normal_result(), _make_normal_result()])
        ids = {item.ecg_id for item in wl.items}
        assert "ecg_001" in ids
        assert "ecg_002" in ids

    def test_ecg_ids_length_mismatch_raises(self) -> None:
        p = WorklistPrioritizer()
        with pytest.raises(ValueError, match="must match"):
            p.prioritize([_make_normal_result()], ecg_ids=["a", "b"])

    def test_top_finding_is_most_urgent(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_critical_result()], ecg_ids=["c1"])
        item = wl.items[0]
        assert item.top_finding in ("VF", "STEMI", "VT", "mortality_1y", "sudden_cardiac_death_risk")

    def test_recommended_action_set(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_critical_result()], ecg_ids=["c1"])
        assert "Immediate" in wl.items[0].recommended_action

    def test_active_findings_populated(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_critical_result()], ecg_ids=["c1"])
        assert len(wl.items[0].active_findings) > 0

    def test_risk_score_escalation_critical(self) -> None:
        """Risk scores above critical threshold escalate tier."""
        result = {
            "rhythm": {"normal_sinus_rhythm": 0.90},
            "risk": {"mortality_1y": 0.85},
        }
        p = WorklistPrioritizer()
        wl = p.prioritize([result], ecg_ids=["r1"])
        assert wl.items[0].urgency_tier == "critical"

    def test_risk_score_escalation_high(self) -> None:
        result = {
            "rhythm": {"normal_sinus_rhythm": 0.90},
            "risk": {"mortality_1y": 0.55},
        }
        p = WorklistPrioritizer()
        wl = p.prioritize([result], ecg_ids=["r1"])
        assert wl.items[0].urgency_tier == "high"

    def test_risk_score_escalation_moderate(self) -> None:
        result = {
            "rhythm": {"normal_sinus_rhythm": 0.90},
            "risk": {"mortality_1y": 0.35},
        }
        p = WorklistPrioritizer()
        wl = p.prioritize([result], ecg_ids=["r1"])
        assert wl.items[0].urgency_tier == "moderate"

    def test_worklist_to_dict_serializable(self) -> None:
        p = WorklistPrioritizer()
        wl = p.prioritize([_make_critical_result(), _make_normal_result()])
        d = wl.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["total_count"] == 2

    def test_many_ecgs_performance(self) -> None:
        """Prioritize 100 ECGs without error."""
        p = WorklistPrioritizer()
        results = [_make_normal_result() for _ in range(100)]
        wl = p.prioritize(results)
        assert wl.total_count == 100


# ---------------------------------------------------------------------------
# Tests: YAML rule loading
# ---------------------------------------------------------------------------

class TestYAMLLoading:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
critical_conditions:
  custom_critical: 0.35
high_conditions:
  custom_high: 0.45
moderate_conditions:
  custom_moderate: 0.55
risk_thresholds:
  mortality_1y:
    critical: 0.80
    high: 0.60
    moderate: 0.40
actions:
  critical: "Custom critical action"
  high: "Custom high action"
  moderate: "Custom moderate action"
  low: "Custom low action"
"""
        yaml_file = tmp_path / "rules.yaml"
        yaml_file.write_text(yaml_content)

        rules = load_urgency_rules(yaml_file)
        assert "custom_critical" in rules.critical_conditions
        assert rules.critical_conditions["custom_critical"] == pytest.approx(0.35)
        assert "custom_high" in rules.high_conditions
        assert rules.actions["critical"] == "Custom critical action"

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_urgency_rules("/nonexistent/path.yaml")

    def test_partial_yaml_uses_defaults(self, tmp_path: Path) -> None:
        yaml_content = """
critical_conditions:
  only_one: 0.5
"""
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text(yaml_content)

        rules = load_urgency_rules(yaml_file)
        assert "only_one" in rules.critical_conditions
        # high/moderate should use defaults since not overridden
        assert len(rules.risk_thresholds) > 0

    def test_prioritizer_with_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
critical_conditions:
  normal_sinus_rhythm: 0.50
"""
        yaml_file = tmp_path / "rules.yaml"
        yaml_file.write_text(yaml_content)

        p = WorklistPrioritizer(rules_yaml=yaml_file)
        result = {"rhythm": {"normal_sinus_rhythm": 0.95}}
        wl = p.prioritize([result])
        assert wl.items[0].urgency_tier == "critical"


# ---------------------------------------------------------------------------
# Tests: API endpoint
# ---------------------------------------------------------------------------

class TestWorklistAPI:
    @pytest.fixture()
    def client(self) -> Any:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from aortica.api.app import create_app
        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_prioritize_endpoint_success(self, client: Any) -> None:
        body = {
            "results": [
                _make_critical_result(),
                _make_normal_result(),
            ],
            "ecg_ids": ["crit_1", "norm_1"],
        }
        resp = client.post("/api/v1/worklist/prioritize", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 2
        assert data["items"][0]["ecg_id"] == "crit_1"
        assert data["items"][0]["urgency_tier"] == "critical"

    def test_prioritize_endpoint_empty_results(self, client: Any) -> None:
        resp = client.post("/api/v1/worklist/prioritize", json={"results": []})
        assert resp.status_code == 422

    def test_prioritize_endpoint_no_results_key(self, client: Any) -> None:
        resp = client.post("/api/v1/worklist/prioritize", json={})
        assert resp.status_code == 422

    def test_prioritize_endpoint_id_mismatch(self, client: Any) -> None:
        body = {
            "results": [_make_normal_result()],
            "ecg_ids": ["a", "b"],
        }
        resp = client.post("/api/v1/worklist/prioritize", json=body)
        assert resp.status_code == 422

    def test_prioritize_endpoint_all_tiers(self, client: Any) -> None:
        body = {
            "results": [
                _make_critical_result(),
                _make_high_result(),
                _make_moderate_result(),
                _make_normal_result(),
            ],
        }
        resp = client.post("/api/v1/worklist/prioritize", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["critical_count"] == 1
        assert data["high_count"] == 1
        assert data["moderate_count"] == 1
        assert data["low_count"] == 1
        # Verify sorted descending
        scores = [i["urgency_score"] for i in data["items"]]
        assert scores == sorted(scores, reverse=True)
