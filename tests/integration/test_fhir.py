"""Tests for FHIR R4 DiagnosticReport output (US-080).

Validates:
- DiagnosticReport creation with synthetic multi-task output
- Child Observation resources for positive classification findings
- RiskAssessment resources for risk predictions
- SNOMED CT / LOINC code mappings
- Confidence thresholds filter low-confidence findings
- Output validates against FHIR R4 resource schemas
- Bundle structure is valid JSON
- Patient reference propagation
- Edge cases: empty predictions, all-below-threshold, risk-only
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

try:
    from fhir.resources.R4B.bundle import Bundle
    from fhir.resources.R4B.diagnosticreport import DiagnosticReport
    from fhir.resources.R4B.observation import Observation
    from fhir.resources.R4B.riskassessment import RiskAssessment

    HAS_FHIR = True
except ImportError:
    HAS_FHIR = False

pytestmark = pytest.mark.skipif(not HAS_FHIR, reason="fhir.resources not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_multi_task_output() -> Dict[str, Any]:
    """Create a synthetic multi-task output dict with mixed findings."""
    return {
        "rhythm": {
            "AF": 0.95,
            "normal_sinus_rhythm": 0.02,
            "VT": 0.85,
            "sinus_brady": 0.10,
            "WPW": 0.55,
        },
        "structural": {
            "LVH": 0.80,
            "LVSD": 0.15,
            "DCM": 0.45,
        },
        "ischaemia": {
            "STEMI": 0.92,
            "old_MI": 0.05,
            "QTc_prolongation": 0.35,
        },
        "risk": {
            "mortality_1y": 0.15,
            "hf_hosp_12m": 0.08,
            "af_onset_12m": 0.72,
            "ecg_predicted_ef": 0.45,
            "conduction_disease_trajectory": 0.20,
            "sudden_cardiac_death_risk": 0.05,
        },
    }


@pytest.fixture
def minimal_output() -> Dict[str, Any]:
    """Minimal output with a single finding."""
    return {
        "rhythm": {"AF": 0.95},
    }


@pytest.fixture
def risk_only_output() -> Dict[str, Any]:
    """Output with only risk predictions (no classification findings)."""
    return {
        "risk": {
            "mortality_1y": 0.30,
            "hf_hosp_12m": 0.10,
            "af_onset_12m": 0.05,
            "ecg_predicted_ef": 0.60,
            "conduction_disease_trajectory": 0.12,
            "sudden_cardiac_death_risk": 0.03,
        },
    }


# ---------------------------------------------------------------------------
# Tests — Basic functionality
# ---------------------------------------------------------------------------


class TestTodiagnosticReport:
    """Tests for the main to_diagnostic_report() function."""

    def test_produces_fhir_output(self, synthetic_multi_task_output: Dict[str, Any]) -> None:
        """to_diagnostic_report returns a FHIROutput with all components."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        assert result.diagnostic_report is not None
        assert isinstance(result.diagnostic_report, DiagnosticReport)
        assert len(result.observations) > 0
        assert len(result.risk_assessments) > 0
        assert result.bundle_json != ""

    def test_diagnostic_report_status_final(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """DiagnosticReport status is 'final'."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        assert result.diagnostic_report.status == "final"

    def test_diagnostic_report_has_ecg_loinc_code(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """DiagnosticReport code contains ECG study LOINC."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        dr = result.diagnostic_report
        assert dr.code is not None
        assert dr.code.coding is not None
        assert len(dr.code.coding) > 0

        loinc_coding = dr.code.coding[0]
        assert loinc_coding.system == "http://loinc.org"
        assert loinc_coding.code == "11524-6"

    def test_diagnostic_report_has_conclusion(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """DiagnosticReport includes a human-readable conclusion."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        assert result.diagnostic_report.conclusion is not None
        assert len(result.diagnostic_report.conclusion) > 0
        assert "AI" in result.diagnostic_report.conclusion

    def test_patient_reference_propagated(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Patient reference propagates to DiagnosticReport and child resources."""
        from aortica.integration.fhir import to_diagnostic_report

        patient_ref = "Patient/abc-123"
        result = to_diagnostic_report(
            synthetic_multi_task_output, patient_ref=patient_ref
        )

        # DiagnosticReport subject
        assert result.diagnostic_report.subject is not None
        assert result.diagnostic_report.subject.reference == patient_ref

        # Observations
        for obs in result.observations:
            assert obs.subject is not None
            assert obs.subject.reference == patient_ref

        # RiskAssessments
        for ra in result.risk_assessments:
            assert ra.subject is not None
            assert ra.subject.reference == patient_ref

    def test_no_patient_reference(
        self, minimal_output: Dict[str, Any]
    ) -> None:
        """Resources are valid even without a patient reference."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(minimal_output)
        assert result.diagnostic_report.subject is None


# ---------------------------------------------------------------------------
# Tests — Observations
# ---------------------------------------------------------------------------


class TestObservations:
    """Tests for Observation resource generation."""

    def test_observation_count_matches_positive_findings(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Number of observations matches findings above threshold."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(
            synthetic_multi_task_output, confidence_threshold=0.30
        )

        # Expected positive findings (>= 0.30):
        # rhythm: AF(0.95), VT(0.85), WPW(0.55) = 3
        # structural: LVH(0.80), DCM(0.45) = 2
        # ischaemia: STEMI(0.92), QTc_prolongation(0.35) = 2
        # Total = 7
        assert len(result.observations) == 7

    def test_observation_has_snomed_code(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Observations for known classes have SNOMED CT codes."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        # Find the AF observation
        af_obs = [
            obs
            for obs in result.observations
            if obs.code.text and "Atrial fibrillation" in obs.code.text
        ]
        assert len(af_obs) == 1
        af = af_obs[0]
        assert af.code.coding[0].system == "http://snomed.info/sct"
        assert af.code.coding[0].code == "49436004"

    def test_observation_has_confidence_value(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Observations include confidence as valueQuantity."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        for obs in result.observations:
            assert obs.valueQuantity is not None
            assert obs.valueQuantity.value is not None
            assert 0.0 <= obs.valueQuantity.value <= 1.0
            assert obs.valueQuantity.unit == "probability"

    def test_observation_interpretation_high(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """High-confidence findings get 'H' (High) interpretation."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        # AF has 0.95 confidence → should be High
        af_obs = [
            obs
            for obs in result.observations
            if obs.valueQuantity and obs.valueQuantity.value is not None
            and obs.valueQuantity.value > 0.90
        ]
        assert len(af_obs) > 0
        for obs in af_obs:
            assert obs.interpretation is not None
            assert obs.interpretation[0].coding[0].code == "H"

    def test_observation_has_task_component(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Observations include a component identifying the task head."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        for obs in result.observations:
            assert obs.component is not None
            assert len(obs.component) > 0
            task_component = obs.component[0]
            assert task_component.code is not None
            assert task_component.valueString in [
                "rhythm",
                "structural",
                "ischaemia",
            ]

    def test_confidence_threshold_filters(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Higher threshold reduces observation count."""
        from aortica.integration.fhir import to_diagnostic_report

        low_thresh = to_diagnostic_report(
            synthetic_multi_task_output, confidence_threshold=0.30
        )
        high_thresh = to_diagnostic_report(
            synthetic_multi_task_output, confidence_threshold=0.80
        )

        assert len(high_thresh.observations) < len(low_thresh.observations)

    def test_all_below_threshold_no_observations(self) -> None:
        """When all findings are below threshold, no observations created."""
        from aortica.integration.fhir import to_diagnostic_report

        low_confidence = {
            "rhythm": {"AF": 0.10, "normal_sinus_rhythm": 0.05},
            "structural": {"LVH": 0.01},
        }
        result = to_diagnostic_report(low_confidence, confidence_threshold=0.30)
        assert len(result.observations) == 0

    def test_observation_status_final(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """All observations have status 'final'."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        for obs in result.observations:
            assert obs.status == "final"


# ---------------------------------------------------------------------------
# Tests — RiskAssessments
# ---------------------------------------------------------------------------


class TestRiskAssessments:
    """Tests for RiskAssessment resource generation."""

    def test_risk_assessment_count(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """One RiskAssessment per risk output."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        assert len(result.risk_assessments) == 6

    def test_risk_assessment_has_prediction(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Each RiskAssessment has a prediction with probability."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        for ra in result.risk_assessments:
            assert ra.prediction is not None
            assert len(ra.prediction) > 0
            pred = ra.prediction[0]
            assert pred.probabilityDecimal is not None
            assert 0.0 <= float(pred.probabilityDecimal) <= 1.0

    def test_risk_assessment_has_outcome(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Each RiskAssessment prediction has an outcome description."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)

        for ra in result.risk_assessments:
            pred = ra.prediction[0]
            assert pred.outcome is not None
            assert pred.outcome.text is not None
            assert len(pred.outcome.text) > 0

    def test_risk_assessment_status_final(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """All RiskAssessments have status 'final'."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        for ra in result.risk_assessments:
            assert ra.status == "final"

    def test_risk_only_output(self, risk_only_output: Dict[str, Any]) -> None:
        """Works with risk-only output (no classification findings)."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(risk_only_output)
        assert len(result.observations) == 0
        assert len(result.risk_assessments) == 6
        assert result.diagnostic_report is not None


# ---------------------------------------------------------------------------
# Tests — Bundle and JSON validation
# ---------------------------------------------------------------------------


class TestBundle:
    """Tests for FHIR Bundle structure and JSON output."""

    def test_bundle_is_valid_json(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """bundle_json is valid JSON."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)
        assert isinstance(parsed, dict)
        assert parsed["resourceType"] == "Bundle"

    def test_bundle_type_is_collection(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Bundle type is 'collection'."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)
        assert parsed["type"] == "collection"

    def test_bundle_entry_count(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Bundle entries = 1 DiagnosticReport + N Observations + M RiskAssessments."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)

        expected_entries = (
            1  # DiagnosticReport
            + len(result.observations)
            + len(result.risk_assessments)
        )
        assert len(parsed["entry"]) == expected_entries

    def test_bundle_first_entry_is_diagnostic_report(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """First bundle entry is the DiagnosticReport."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)
        first_resource = parsed["entry"][0]["resource"]
        assert first_resource["resourceType"] == "DiagnosticReport"

    def test_bundle_contains_observations(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Bundle contains Observation entries."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)

        obs_entries = [
            e
            for e in parsed["entry"]
            if e["resource"]["resourceType"] == "Observation"
        ]
        assert len(obs_entries) == len(result.observations)

    def test_bundle_contains_risk_assessments(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Bundle contains RiskAssessment entries."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)

        ra_entries = [
            e
            for e in parsed["entry"]
            if e["resource"]["resourceType"] == "RiskAssessment"
        ]
        assert len(ra_entries) == len(result.risk_assessments)

    def test_bundle_validates_as_fhir_bundle(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Bundle JSON validates against FHIR R4 Bundle schema."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        parsed = json.loads(result.bundle_json)

        # Re-parse through fhir.resources to validate
        validated_bundle = Bundle(**parsed)
        assert validated_bundle.type == "collection"
        assert len(validated_bundle.entry) > 0


# ---------------------------------------------------------------------------
# Tests — DiagnosticReport result references
# ---------------------------------------------------------------------------


class TestDiagnosticReportReferences:
    """Tests for result references from DiagnosticReport to child resources."""

    def test_report_references_observations(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """DiagnosticReport.result references all Observation resources."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(synthetic_multi_task_output)
        dr = result.diagnostic_report

        if result.observations:
            assert dr.result is not None
            assert len(dr.result) == len(result.observations)
            for ref in dr.result:
                assert ref.reference.startswith("Observation/")

    def test_no_observations_no_result_references(self) -> None:
        """When there are no observations, result references are absent."""
        from aortica.integration.fhir import to_diagnostic_report

        empty = {"rhythm": {"AF": 0.01}}
        result = to_diagnostic_report(empty, confidence_threshold=0.50)
        assert result.diagnostic_report.result is None or len(result.diagnostic_report.result) == 0


# ---------------------------------------------------------------------------
# Tests — ECG metadata
# ---------------------------------------------------------------------------


class TestECGMetadata:
    """Tests for ECG metadata handling."""

    def test_acquisition_datetime_used_as_issued(
        self, minimal_output: Dict[str, Any]
    ) -> None:
        """acquisition_datetime from metadata is used as the issued datetime."""
        from aortica.integration.fhir import to_diagnostic_report

        meta = {"acquisition_datetime": "2024-01-15T10:30:00+00:00"}
        result = to_diagnostic_report(
            minimal_output, ecg_metadata=meta
        )
        assert result.diagnostic_report.issued is not None

    def test_empty_metadata_still_works(
        self, minimal_output: Dict[str, Any]
    ) -> None:
        """Works with empty metadata dict."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(minimal_output, ecg_metadata={})
        assert result.diagnostic_report is not None


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_multi_task_output(self) -> None:
        """Empty dict produces a valid (minimal) DiagnosticReport."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report({})
        assert result.diagnostic_report is not None
        assert len(result.observations) == 0
        assert len(result.risk_assessments) == 0
        assert "No significant AI findings" in result.diagnostic_report.conclusion

    def test_single_task_output(self) -> None:
        """Works with only one task head present."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report({"rhythm": {"AF": 0.99}})
        assert len(result.observations) == 1
        assert len(result.risk_assessments) == 0

    def test_high_threshold_filters_all(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Very high threshold filters out all classification findings."""
        from aortica.integration.fhir import to_diagnostic_report

        result = to_diagnostic_report(
            synthetic_multi_task_output, confidence_threshold=0.99
        )
        # Only AF (0.95), VT (0.85), STEMI (0.92) — all below 0.99
        assert len(result.observations) == 0

    def test_list_format_predictions(self) -> None:
        """Works when predictions are lists of floats (paired with class names)."""
        from aortica.integration.fhir import to_diagnostic_report

        # This tests the list-of-floats input format
        output = {
            "risk": [0.15, 0.08, 0.72, 0.45, 0.20, 0.05],
        }
        result = to_diagnostic_report(output)
        assert len(result.risk_assessments) == 6


# ---------------------------------------------------------------------------
# Tests — SNOMED/LOINC code coverage
# ---------------------------------------------------------------------------


class TestCodeMappings:
    """Tests for SNOMED CT and LOINC code mappings."""

    def test_all_snomed_codes_are_strings(self) -> None:
        """All SNOMED codes in the mapping are non-empty strings."""
        from aortica.integration.fhir import _SNOMED_CODES

        for class_name, (code, display) in _SNOMED_CODES.items():
            assert isinstance(code, str) and len(code) > 0, (
                f"Missing SNOMED code for {class_name}"
            )
            assert isinstance(display, str) and len(display) > 0, (
                f"Missing display for {class_name}"
            )

    def test_risk_display_names_complete(self) -> None:
        """All 6 risk outputs have display names."""
        from aortica.integration.fhir import _RISK_DISPLAY_NAMES
        from aortica.models.risk_head import RISK_OUTPUTS

        for risk_name in RISK_OUTPUTS:
            assert risk_name in _RISK_DISPLAY_NAMES, (
                f"Missing display name for risk output {risk_name}"
            )

    def test_snomed_covers_all_head_classes(self) -> None:
        """SNOMED mapping covers all classes from all task heads."""
        from aortica.integration.fhir import _SNOMED_CODES
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
        from aortica.models.rhythm_head import RHYTHM_CLASSES
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        all_classes = list(RHYTHM_CLASSES) + list(STRUCTURAL_CLASSES) + list(ISCHAEMIA_CLASSES)
        for cls in all_classes:
            assert cls in _SNOMED_CODES, f"Missing SNOMED mapping for {cls}"
