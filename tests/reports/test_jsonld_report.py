"""Tests for JSON-LD machine-readable report generator."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from aortica.reports.jsonld_report import (
    JSONLDReport,
    _JSONLD_CONTEXT,
    _SNOMED_CODES,
    _RISK_LOINC_CODES,
    _build_finding_observation,
    _build_risk_observation,
    _extract_predictions,
    generate_jsonld,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_rhythm_output() -> Dict[str, Any]:
    """Multi-task output dict with rhythm findings above threshold."""
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.structural_head import STRUCTURAL_CLASSES
    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.risk_head import RISK_OUTPUTS

    rhythm = [0.0] * len(RHYTHM_CLASSES)
    rhythm[0] = 0.95  # AF
    rhythm[5] = 0.85  # VT

    structural = [0.0] * len(STRUCTURAL_CLASSES)
    structural[0] = 0.70  # LVH

    ischaemia = [0.0] * len(ISCHAEMIA_CLASSES)
    ischaemia[0] = 0.60  # STEMI

    risk = [0.15, 0.08, 0.22, 0.55, 0.10, 0.05]

    return {
        "rhythm": rhythm,
        "structural": structural,
        "ischaemia": ischaemia,
        "risk": risk,
    }


@pytest.fixture()
def sample_metadata() -> Dict[str, Any]:
    return {
        "ecg_id": "ECG-12345",
        "device": "Aortica Edge v2",
        "sample_rate": 500,
        "duration_seconds": 10.0,
        "num_leads": 12,
        "source_format": "wfdb",
    }


# ---------------------------------------------------------------------------
# Unit tests: _build_finding_observation
# ---------------------------------------------------------------------------


class TestBuildFindingObservation:
    def test_known_snomed_code(self) -> None:
        obs = _build_finding_observation("AF", 0.95, "rhythm")
        assert obs["@type"] == "MedicalObservation"
        assert obs["name"] == "Af"
        assert obs["identifier"] == "AF"
        assert obs["confidence"] == 0.95
        assert obs["taskHead"] == "rhythm"
        assert "snomedCode" in obs
        assert obs["snomedCode"] == "snomed:49436004"

    def test_unknown_class_no_snomed(self) -> None:
        obs = _build_finding_observation("unknown_class", 0.5, "rhythm")
        assert obs["@type"] == "MedicalObservation"
        assert "snomedCode" not in obs

    def test_confidence_interval(self) -> None:
        obs = _build_finding_observation("AF", 0.9, "rhythm", (0.85, 0.95))
        assert obs["confidenceInterval"]["lower"] == 0.85
        assert obs["confidenceInterval"]["upper"] == 0.95

    def test_no_confidence_interval(self) -> None:
        obs = _build_finding_observation("AF", 0.9, "rhythm")
        assert "confidenceInterval" not in obs


# ---------------------------------------------------------------------------
# Unit tests: _build_risk_observation
# ---------------------------------------------------------------------------


class TestBuildRiskObservation:
    def test_known_loinc_code(self) -> None:
        obs = _build_risk_observation("mortality_1y", 0.15)
        assert obs["@type"] == "MedicalRiskEstimator"
        assert obs["confidence"] == 0.15
        assert obs["taskHead"] == "risk"
        assert "loincCode" in obs

    def test_unknown_risk_no_loinc(self) -> None:
        obs = _build_risk_observation("unknown_risk", 0.5)
        assert "loincCode" not in obs

    def test_risk_confidence_interval(self) -> None:
        obs = _build_risk_observation("mortality_1y", 0.15, (0.10, 0.20))
        assert obs["confidenceInterval"]["lower"] == 0.10
        assert obs["confidenceInterval"]["upper"] == 0.20


# ---------------------------------------------------------------------------
# Unit tests: _extract_predictions
# ---------------------------------------------------------------------------


class TestExtractPredictions:
    def test_dict_input(self, sample_rhythm_output: Dict[str, Any]) -> None:
        preds = _extract_predictions(sample_rhythm_output)
        assert "rhythm" in preds
        assert "risk" in preds
        assert preds["rhythm"][0] == 0.95

    def test_partial_output(self) -> None:
        preds = _extract_predictions({"rhythm": [0.5] * 28})
        assert "rhythm" in preds
        assert "structural" not in preds

    def test_empty_output(self) -> None:
        preds = _extract_predictions({})
        assert preds == {}


# ---------------------------------------------------------------------------
# Unit tests: generate_jsonld
# ---------------------------------------------------------------------------


class TestGenerateJsonld:
    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_basic_structure(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
        sample_metadata: Dict[str, Any],
    ) -> None:
        # Mock pyld for validation
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(
            sample_rhythm_output,
            sample_metadata,
            model_version="0.2.0",
        )

        assert isinstance(report, JSONLDReport)
        assert report.document["@type"] == "MedicalTest"
        assert "@context" in report.document
        assert report.document["modelVersion"] == "0.2.0"
        assert report.document["identifier"] == "ECG-12345"

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_findings_included(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output, confidence_threshold=0.30)
        results = report.document["result"]

        # Should have AF(0.95), VT(0.85), LVH(0.70), STEMI(0.60), + 6 risk
        finding_names = [r["identifier"] for r in results if r["@type"] == "MedicalObservation"]
        assert "AF" in finding_names
        assert "VT" in finding_names
        assert "LVH" in finding_names
        assert "STEMI" in finding_names

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_risk_always_included(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        results = report.document["result"]
        risk_obs = [r for r in results if r["@type"] == "MedicalRiskEstimator"]
        assert len(risk_obs) == 6

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_threshold_filtering(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output, confidence_threshold=0.90)
        results = report.document["result"]
        findings = [r for r in results if r["@type"] == "MedicalObservation"]
        # Only AF (0.95) should pass 0.90 threshold
        assert len(findings) == 1
        assert findings[0]["identifier"] == "AF"

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_json_string_valid(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        parsed = json.loads(report.json_string)
        assert parsed["@type"] == "MedicalTest"

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_context_has_standard_ontologies(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        ctx = report.document["@context"]
        assert "snomed" in ctx
        assert "loinc" in ctx
        assert "http://snomed.info/id/" in str(ctx["snomed"])
        assert "http://loinc.org/rdf/" in str(ctx["loinc"])

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_provenance_metadata(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(
            sample_rhythm_output,
            model_version="0.2.0",
            input_file_hash="abc123",
        )
        doc = report.document
        assert doc["modelVersion"] == "0.2.0"
        assert doc["inputFileHash"] == "abc123"
        assert "inferenceTimestamp" in doc

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_snomed_code_mappings(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        results = report.document["result"]
        af_obs = [r for r in results if r.get("identifier") == "AF"]
        assert len(af_obs) == 1
        assert af_obs[0]["snomedCode"] == "snomed:49436004"

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_ecg_metadata_fields(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
        sample_metadata: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output, sample_metadata)
        doc = report.document
        assert doc["aortica:device"] == "Aortica Edge v2"
        assert doc["aortica:sampleRate"] == 500

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_validation_success(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        assert report.is_valid is True

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_validation_failure(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = []  # Empty expansion = invalid
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        assert report.is_valid is False

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_empty_output(self, mock_pyld: MagicMock) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld({})
        assert report.document["@type"] == "MedicalTest"
        assert report.document["result"] == []

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_confidence_intervals_propagated(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(
            sample_rhythm_output,
            confidence_intervals={"AF": (0.90, 0.98)},
        )
        results = report.document["result"]
        af_obs = [r for r in results if r.get("identifier") == "AF"]
        assert af_obs[0]["confidenceInterval"]["lower"] == 0.90

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_no_metadata_defaults(
        self,
        mock_pyld: MagicMock,
        sample_rhythm_output: Dict[str, Any],
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        report = generate_jsonld(sample_rhythm_output)
        assert report.document["identifier"] == "unknown"
        assert report.document["inputFileHash"] == "unavailable"

    @patch("aortica.reports.jsonld_report._get_pyld")
    def test_loinc_code_in_risk(
        self,
        mock_pyld: MagicMock,
    ) -> None:
        mock_jsonld = MagicMock()
        mock_jsonld.expand.return_value = [{"@type": ["https://schema.org/MedicalTest"]}]
        mock_jsonld.compact.return_value = {"@type": "MedicalTest"}
        mock_pyld.return_value.jsonld = mock_jsonld

        output = {"risk": [0.15, 0.08, 0.22, 0.55, 0.10, 0.05]}
        report = generate_jsonld(output)
        results = report.document["result"]
        mort = [r for r in results if r.get("identifier") == "mortality_1y"]
        assert len(mort) == 1
        assert "loincCode" in mort[0]


# ---------------------------------------------------------------------------
# Context structure tests
# ---------------------------------------------------------------------------


class TestJsonldContext:
    def test_context_keys(self) -> None:
        assert "snomed" in _JSONLD_CONTEXT
        assert "loinc" in _JSONLD_CONTEXT
        assert "@vocab" in _JSONLD_CONTEXT
        assert "MedicalTest" in _JSONLD_CONTEXT
        assert "MedicalObservation" in _JSONLD_CONTEXT

    def test_snomed_iri(self) -> None:
        assert _JSONLD_CONTEXT["snomed"] == "http://snomed.info/id/"

    def test_loinc_iri(self) -> None:
        assert _JSONLD_CONTEXT["loinc"] == "http://loinc.org/rdf/"


# ---------------------------------------------------------------------------
# Code mapping coverage tests
# ---------------------------------------------------------------------------


class TestCodeMappings:
    def test_all_rhythm_classes_have_snomed(self) -> None:
        from aortica.models.rhythm_head import RHYTHM_CLASSES
        for cls in RHYTHM_CLASSES:
            assert cls in _SNOMED_CODES, f"Missing SNOMED for rhythm class: {cls}"

    def test_all_structural_classes_have_snomed(self) -> None:
        from aortica.models.structural_head import STRUCTURAL_CLASSES
        for cls in STRUCTURAL_CLASSES:
            assert cls in _SNOMED_CODES, f"Missing SNOMED for structural class: {cls}"

    def test_all_ischaemia_classes_have_snomed(self) -> None:
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
        for cls in ISCHAEMIA_CLASSES:
            assert cls in _SNOMED_CODES, f"Missing SNOMED for ischaemia class: {cls}"

    def test_risk_outputs_have_loinc(self) -> None:
        from aortica.models.risk_head import RISK_OUTPUTS
        for output in RISK_OUTPUTS:
            assert output in _RISK_LOINC_CODES, f"Missing LOINC for risk: {output}"
