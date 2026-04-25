"""Tests for the Second Reader Comparison module (US-050).

Covers: Pydantic models, alias normalisation, clinical importance ranking,
comparison logic, and the /api/v1/compare endpoint.
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------

try:
    from starlette.testclient import TestClient

    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

from aortica.api.compare import (
    ClinicianInterpretation,
    CompareRequest,
    CompareResponse,
    DiscrepancyItem,
    _ALIASES,
    _CLINICAL_IMPORTANCE,
    _class_to_task,
    _importance,
    _normalise_finding,
    compare_interpretations,
)


# ===================================================================
# Pydantic model construction
# ===================================================================


class TestClinicianInterpretation:
    """ClinicianInterpretation model tests."""

    def test_default_empty(self) -> None:
        interp = ClinicianInterpretation()
        assert interp.findings == []
        assert interp.free_text == ""

    def test_with_findings(self) -> None:
        interp = ClinicianInterpretation(
            findings=["AF", "LBBB"],
            free_text="Irregular rhythm noted",
        )
        assert len(interp.findings) == 2
        assert interp.free_text == "Irregular rhythm noted"

    def test_roundtrip(self) -> None:
        interp = ClinicianInterpretation(findings=["VT"], free_text="Urgent")
        d = interp.model_dump()
        restored = ClinicianInterpretation(**d)
        assert restored.findings == interp.findings
        assert restored.free_text == interp.free_text


class TestCompareRequest:
    """CompareRequest model tests."""

    def test_construction(self) -> None:
        req = CompareRequest(
            interpretation=ClinicianInterpretation(findings=["AF"]),
            ai_predictions={"rhythm": [0.9] + [0.0] * 21},
            threshold=0.5,
        )
        assert req.threshold == 0.5
        assert req.interpretation.findings == ["AF"]

    def test_default_threshold(self) -> None:
        req = CompareRequest(
            interpretation=ClinicianInterpretation(),
            ai_predictions={},
        )
        assert req.threshold == 0.50


class TestDiscrepancyItem:
    """DiscrepancyItem model tests."""

    def test_agreement(self) -> None:
        item = DiscrepancyItem(
            class_name="AF",
            task="rhythm",
            status="agreement",
            ai_probability=0.92,
            clinician_selected=True,
            clinical_importance=6,
        )
        assert item.status == "agreement"
        assert item.clinician_selected is True

    def test_ai_only(self) -> None:
        item = DiscrepancyItem(
            class_name="STEMI",
            task="ischaemia",
            status="ai_only",
            ai_probability=0.85,
            clinician_selected=False,
            clinical_importance=10,
        )
        assert item.status == "ai_only"
        assert item.clinical_importance == 10

    def test_clinician_only(self) -> None:
        item = DiscrepancyItem(
            class_name="LVH",
            task="structural",
            status="clinician_only",
            ai_probability=0.20,
            clinician_selected=True,
            clinical_importance=5,
        )
        assert item.status == "clinician_only"

    def test_roundtrip(self) -> None:
        item = DiscrepancyItem(
            class_name="AF",
            task="rhythm",
            status="agreement",
            ai_probability=0.9,
            clinician_selected=True,
            clinical_importance=6,
        )
        d = item.model_dump()
        restored = DiscrepancyItem(**d)
        assert restored.class_name == item.class_name
        assert restored.ai_probability == item.ai_probability


class TestCompareResponse:
    """CompareResponse model tests."""

    def test_construction(self) -> None:
        resp = CompareResponse(
            agreements=[],
            ai_only=[],
            clinician_only=[],
            summary={
                "total_agreements": 0,
                "total_ai_only": 0,
                "total_clinician_only": 0,
            },
        )
        assert resp.summary["total_agreements"] == 0

    def test_with_items(self) -> None:
        item = DiscrepancyItem(
            class_name="AF",
            task="rhythm",
            status="agreement",
            ai_probability=0.85,
            clinician_selected=True,
            clinical_importance=6,
        )
        resp = CompareResponse(
            agreements=[item],
            ai_only=[],
            clinician_only=[],
            summary={
                "total_agreements": 1,
                "total_ai_only": 0,
                "total_clinician_only": 0,
            },
        )
        assert len(resp.agreements) == 1
        assert resp.agreements[0].class_name == "AF"


# ===================================================================
# Alias normalisation
# ===================================================================


class TestNormaliseFinding:
    """_normalise_finding() alias resolution tests."""

    def test_exact_canonical_match(self) -> None:
        assert _normalise_finding("AF") == "AF"
        assert _normalise_finding("STEMI") == "STEMI"
        assert _normalise_finding("normal_sinus_rhythm") == "normal_sinus_rhythm"

    def test_human_friendly_alias(self) -> None:
        assert _normalise_finding("Atrial Fibrillation") == "AF"
        assert _normalise_finding("left bundle branch block") == "LBBB"
        assert _normalise_finding("Ventricular Tachycardia") == "VT"

    def test_case_insensitive_canonical(self) -> None:
        assert _normalise_finding("af") == "AF"
        assert _normalise_finding("stemi") == "STEMI"

    def test_unmatched_returns_none(self) -> None:
        assert _normalise_finding("Completely Made Up Finding") is None
        assert _normalise_finding("") is None

    def test_whitespace_handling(self) -> None:
        assert _normalise_finding("  atrial fibrillation  ") == "AF"

    def test_all_aliases_resolve(self) -> None:
        """Every entry in the alias dict should resolve to a valid canonical name."""
        for alias, canonical in _ALIASES.items():
            result = _normalise_finding(alias)
            assert result == canonical, f"Alias '{alias}' failed to resolve"


# ===================================================================
# Clinical importance
# ===================================================================


class TestClinicalImportance:
    """Clinical importance scoring tests."""

    def test_vf_is_highest(self) -> None:
        assert _importance("VF") == 10

    def test_stemi_is_highest(self) -> None:
        assert _importance("STEMI") == 10

    def test_normal_sinus_is_lowest(self) -> None:
        assert _importance("normal_sinus_rhythm") == 1

    def test_unknown_gets_default(self) -> None:
        assert _importance("unknown_class_xyz") == 3

    def test_all_mapped_classes_have_importance(self) -> None:
        """All classes in the importance dict should be valid canonical names."""
        for class_name in _CLINICAL_IMPORTANCE:
            task = _class_to_task(class_name)
            assert task != "unknown", f"{class_name} not in any task"


class TestClassToTask:
    """_class_to_task() mapping tests."""

    def test_rhythm_class(self) -> None:
        assert _class_to_task("AF") == "rhythm"

    def test_structural_class(self) -> None:
        assert _class_to_task("LVH") == "structural"

    def test_ischaemia_class(self) -> None:
        assert _class_to_task("STEMI") == "ischaemia"

    def test_unknown_class(self) -> None:
        assert _class_to_task("not_a_real_class") == "unknown"


# ===================================================================
# Comparison logic
# ===================================================================


class TestCompareInterpretations:
    """compare_interpretations() tests."""

    def _make_ai_preds(
        self,
        positive_classes: dict[str, float] | None = None,
    ) -> dict[str, list[float]]:
        """Helper: build AI prediction dict with specified positives."""
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
        from aortica.models.rhythm_head import RHYTHM_CLASSES
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        preds: dict[str, list[float]] = {
            "rhythm": [0.0] * len(RHYTHM_CLASSES),
            "structural": [0.0] * len(STRUCTURAL_CLASSES),
            "ischaemia": [0.0] * len(ISCHAEMIA_CLASSES),
        }

        if positive_classes:
            for class_name, prob in positive_classes.items():
                for task, classes in [
                    ("rhythm", RHYTHM_CLASSES),
                    ("structural", STRUCTURAL_CLASSES),
                    ("ischaemia", ISCHAEMIA_CLASSES),
                ]:
                    if class_name in classes:
                        idx = classes.index(class_name)
                        preds[task][idx] = prob
                        break

        return preds

    def test_perfect_agreement(self) -> None:
        """Clinician and AI agree on the same findings."""
        ai = self._make_ai_preds({"AF": 0.90, "LBBB": 0.70})
        interp = ClinicianInterpretation(findings=["AF", "LBBB"])

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 2
        assert len(result.ai_only) == 0
        assert len(result.clinician_only) == 0
        assert result.summary["total_agreements"] == 2

    def test_ai_only_finding(self) -> None:
        """AI detects something clinician missed."""
        ai = self._make_ai_preds({"STEMI": 0.85, "AF": 0.70})
        interp = ClinicianInterpretation(findings=["AF"])

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 1
        assert result.agreements[0].class_name == "AF"
        assert len(result.ai_only) == 1
        assert result.ai_only[0].class_name == "STEMI"

    def test_clinician_only_finding(self) -> None:
        """Clinician finds something AI missed."""
        ai = self._make_ai_preds({"AF": 0.70})
        interp = ClinicianInterpretation(findings=["AF", "LVH"])

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 1
        assert len(result.clinician_only) == 1
        assert result.clinician_only[0].class_name == "LVH"

    def test_mixed_results(self) -> None:
        """Mix of agreements, AI-only, and clinician-only."""
        ai = self._make_ai_preds({"AF": 0.85, "STEMI": 0.70, "VT": 0.60})
        interp = ClinicianInterpretation(findings=["AF", "LVH", "LBBB"])

        result = compare_interpretations(interp, ai)

        assert result.summary["total_agreements"] == 1  # AF
        assert result.summary["total_ai_only"] == 2  # STEMI, VT
        assert result.summary["total_clinician_only"] == 2  # LVH, LBBB

    def test_alias_matching(self) -> None:
        """Clinician uses human-friendly names."""
        ai = self._make_ai_preds({"AF": 0.90})
        interp = ClinicianInterpretation(
            findings=["Atrial Fibrillation"]
        )

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 1
        assert result.agreements[0].class_name == "AF"

    def test_unmatched_clinician_input(self) -> None:
        """Clinician enters a finding that can't be mapped."""
        ai = self._make_ai_preds({"AF": 0.90})
        interp = ClinicianInterpretation(
            findings=["AF", "Mystery Finding XYZ"]
        )

        result = compare_interpretations(interp, ai)

        assert len(result.unmatched_clinician_inputs) == 1
        assert "Mystery Finding XYZ" in result.unmatched_clinician_inputs

    def test_threshold_affects_ai_positives(self) -> None:
        """AI findings below threshold are not counted as positive."""
        ai = self._make_ai_preds({"AF": 0.45, "VT": 0.80})
        interp = ClinicianInterpretation(findings=["AF"])

        # With default threshold 0.50, AF is below threshold → clinician_only
        result = compare_interpretations(interp, ai, threshold=0.50)
        assert len(result.clinician_only) == 1
        assert result.clinician_only[0].class_name == "AF"
        assert len(result.ai_only) == 1  # VT above threshold

    def test_low_threshold(self) -> None:
        """Lower threshold captures more AI findings."""
        ai = self._make_ai_preds({"AF": 0.30})
        interp = ClinicianInterpretation(findings=["AF"])

        result = compare_interpretations(interp, ai, threshold=0.25)
        assert len(result.agreements) == 1

    def test_empty_clinician(self) -> None:
        """Clinician enters no findings."""
        ai = self._make_ai_preds({"AF": 0.90, "STEMI": 0.80})
        interp = ClinicianInterpretation(findings=[])

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 0
        assert len(result.ai_only) == 2
        assert len(result.clinician_only) == 0

    def test_empty_ai(self) -> None:
        """AI predicts nothing above threshold."""
        ai = self._make_ai_preds()  # all zeros
        interp = ClinicianInterpretation(findings=["AF", "VT"])

        result = compare_interpretations(interp, ai)

        assert len(result.agreements) == 0
        assert len(result.ai_only) == 0
        assert len(result.clinician_only) == 2

    def test_both_empty(self) -> None:
        """No findings from either side."""
        ai = self._make_ai_preds()
        interp = ClinicianInterpretation(findings=[])

        result = compare_interpretations(interp, ai)

        assert result.summary == {
            "total_agreements": 0,
            "total_ai_only": 0,
            "total_clinician_only": 0,
        }

    def test_sorted_by_importance(self) -> None:
        """AI-only findings sorted by clinical importance (most urgent first)."""
        ai = self._make_ai_preds({
            "normal_sinus_rhythm": 0.90,  # importance 1
            "AF": 0.80,                    # importance 6
            "VF": 0.60,                    # importance 10
        })
        interp = ClinicianInterpretation(findings=[])

        result = compare_interpretations(interp, ai)

        assert len(result.ai_only) == 3
        importances = [item.clinical_importance for item in result.ai_only]
        assert importances == sorted(importances, reverse=True)
        assert result.ai_only[0].class_name == "VF"

    def test_ai_probability_included(self) -> None:
        """AI probability is included in discrepancy items."""
        ai = self._make_ai_preds({"AF": 0.92})
        interp = ClinicianInterpretation(findings=["AF"])

        result = compare_interpretations(interp, ai)

        assert result.agreements[0].ai_probability == pytest.approx(0.92)

    def test_clinician_only_has_ai_prob(self) -> None:
        """Clinician-only items include the AI's (low) probability."""
        ai = self._make_ai_preds({"AF": 0.20})
        interp = ClinicianInterpretation(findings=["AF"])

        result = compare_interpretations(interp, ai, threshold=0.50)

        assert len(result.clinician_only) == 1
        assert result.clinician_only[0].ai_probability == pytest.approx(0.20)

    def test_task_assignment(self) -> None:
        """Each discrepancy item has correct task assignment."""
        ai = self._make_ai_preds({"AF": 0.90, "LVH": 0.80, "STEMI": 0.70})
        interp = ClinicianInterpretation(findings=[])

        result = compare_interpretations(interp, ai)

        tasks = {item.class_name: item.task for item in result.ai_only}
        assert tasks["AF"] == "rhythm"
        assert tasks["LVH"] == "structural"
        assert tasks["STEMI"] == "ischaemia"

    def test_duplicate_clinician_findings_deduplicated(self) -> None:
        """Duplicate clinician findings are deduplicated."""
        ai = self._make_ai_preds({"AF": 0.90})
        interp = ClinicianInterpretation(findings=["AF", "AF", "Atrial Fibrillation"])

        result = compare_interpretations(interp, ai)

        # AF should only appear once in agreements
        assert len(result.agreements) == 1

    def test_response_serialisable(self) -> None:
        """CompareResponse can be serialised to JSON."""
        ai = self._make_ai_preds({"AF": 0.90, "STEMI": 0.70})
        interp = ClinicianInterpretation(findings=["AF", "LVH"])

        result = compare_interpretations(interp, ai)

        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert "agreements" in parsed
        assert "ai_only" in parsed
        assert "clinician_only" in parsed


# ===================================================================
# API endpoint tests
# ===================================================================


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="starlette not installed")
class TestCompareEndpoint:
    """POST /api/v1/compare endpoint tests."""

    def _get_client(self) -> TestClient:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_compare_200(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {
                "findings": ["AF"],
                "free_text": "Irregular rhythm",
            },
            "ai_predictions": {
                "rhythm": [0.90] + [0.0] * 21,
                "structural": [0.0] * 15,
                "ischaemia": [0.0] * 10,
            },
            "threshold": 0.50,
        }

        resp = client.post("/api/v1/compare", json=body)

        assert resp.status_code == 200
        data = resp.json()
        assert "agreements" in data
        assert "ai_only" in data
        assert "clinician_only" in data
        assert "summary" in data

    def test_compare_agreement_in_response(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": ["AF"]},
            "ai_predictions": {
                "rhythm": [0.90] + [0.0] * 21,
            },
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        assert len(data["agreements"]) == 1
        assert data["agreements"][0]["class_name"] == "AF"

    def test_compare_ai_only_in_response(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": []},
            "ai_predictions": {
                "rhythm": [0.90] + [0.0] * 21,
            },
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        assert len(data["ai_only"]) == 1
        assert data["ai_only"][0]["class_name"] == "AF"

    def test_compare_clinician_only_in_response(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": ["LVH"]},
            "ai_predictions": {
                "rhythm": [0.0] * 22,
                "structural": [0.0] * 15,
                "ischaemia": [0.0] * 10,
            },
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        assert len(data["clinician_only"]) == 1
        assert data["clinician_only"][0]["class_name"] == "LVH"

    def test_compare_custom_threshold(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": ["AF"]},
            "ai_predictions": {
                "rhythm": [0.30] + [0.0] * 21,
            },
            "threshold": 0.25,
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        # AF at 0.30 > threshold 0.25 → agreement
        assert len(data["agreements"]) == 1

    def test_compare_unmatched_inputs(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": ["Not A Real Finding"]},
            "ai_predictions": {},
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        assert "Not A Real Finding" in data["unmatched_clinician_inputs"]

    def test_compare_content_type(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": []},
            "ai_predictions": {},
        }

        resp = client.post("/api/v1/compare", json=body)

        assert resp.headers["content-type"] == "application/json"

    def test_compare_summary_counts(self) -> None:
        client = self._get_client()
        body = {
            "interpretation": {"findings": ["AF", "LVH"]},
            "ai_predictions": {
                "rhythm": [0.90] + [0.0] * 21,
                "structural": [0.0] * 15,
                "ischaemia": [0.80] + [0.0] * 9,
            },
        }

        resp = client.post("/api/v1/compare", json=body)
        data = resp.json()

        assert data["summary"]["total_agreements"] == 1  # AF
        assert data["summary"]["total_ai_only"] == 1  # STEMI
        assert data["summary"]["total_clinician_only"] == 1  # LVH


# ===================================================================
# Import verification
# ===================================================================


class TestImports:
    """Verify module imports work correctly."""

    def test_compare_module_imports(self) -> None:
        from aortica.api import compare  # noqa: F401

    def test_all_public_names(self) -> None:
        from aortica.api.compare import (  # noqa: F401
            ClinicianInterpretation,
            CompareRequest,
            CompareResponse,
            DiscrepancyItem,
            compare_interpretations,
        )
