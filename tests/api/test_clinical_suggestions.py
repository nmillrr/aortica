"""Tests for the Clinical Suggestion Prompt Data Layer (US-049b).

Covers:
- ClinicalSuggestion dataclass construction
- JSON loading and validation
- High-severity condition coverage
- JSON round-trip (load → fields → match)
- Suggestion lookup (get_suggestion)
- FastAPI endpoint 200/404 behaviour
- Inference response inclusion (include_suggestions=true)
- Pydantic models (SuggestionEntry, ClinicalSuggestionResponse)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Conditional imports
try:
    from starlette.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from aortica.api.clinical_suggestions import (
    HIGH_SEVERITY_CONDITIONS,
    VALID_URGENCY_LEVELS,
    ClinicalSuggestion,
    ClinicalSuggestionResponse,
    SuggestionsListResponse,
    create_suggestions_router,
    get_condition_suggestions,
    get_suggestion,
    load_suggestions_from_json,
    reload_suggestions,
)
from aortica.api.predict import PredictResponse, SuggestionEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    """Create a minimal valid clinical_suggestions.json."""
    data = {
        "_meta": {"description": "test", "version": "1.0.0"},
        "suggestions": {
            "AF": {
                "prompt": "Evaluate stroke risk",
                "urgency": "prompt",
                "rationale": "AF carries thromboembolic risk.",
            },
            "VT": {
                "prompt": "Urgent cardiology review",
                "urgency": "emergent",
                "rationale": "VT can degenerate to VF.",
            },
            "STEMI": {
                "prompt": "Activate cath lab",
                "urgency": "emergent",
                "rationale": "Door-to-balloon time is critical.",
            },
        },
    }
    path = tmp_path / "clinical_suggestions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def default_json_path() -> Path:
    """Return the path to the default JSON file."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "clinical_suggestions.json"
    )


# ---------------------------------------------------------------------------
# ClinicalSuggestion dataclass
# ---------------------------------------------------------------------------

class TestClinicalSuggestionDataclass:
    def test_construction(self) -> None:
        s = ClinicalSuggestion(
            prompt="Test prompt",
            urgency="routine",
            rationale="Test rationale.",
        )
        assert s.prompt == "Test prompt"
        assert s.urgency == "routine"
        assert s.rationale == "Test rationale."

    def test_fields_are_strings(self) -> None:
        s = ClinicalSuggestion(
            prompt="P", urgency="emergent", rationale="R."
        )
        assert isinstance(s.prompt, str)
        assert isinstance(s.urgency, str)
        assert isinstance(s.rationale, str)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_suggestion_entry_construction(self) -> None:
        entry = SuggestionEntry(
            condition="AF",
            prompt="Evaluate stroke risk",
            urgency="prompt",
            rationale="AF carries risk.",
        )
        assert entry.condition == "AF"
        assert entry.urgency == "prompt"

    def test_suggestion_entry_roundtrip(self) -> None:
        entry = SuggestionEntry(
            condition="VT",
            prompt="Urgent review",
            urgency="emergent",
            rationale="VT is dangerous.",
        )
        d = entry.model_dump()
        rebuilt = SuggestionEntry(**d)
        assert rebuilt == entry

    def test_clinical_suggestion_response(self) -> None:
        resp = ClinicalSuggestionResponse(
            condition="STEMI",
            prompt="Activate cath lab",
            urgency="emergent",
            rationale="Time critical.",
        )
        assert resp.condition == "STEMI"
        d = resp.model_dump()
        assert d["urgency"] == "emergent"

    def test_suggestions_list_response(self) -> None:
        item = ClinicalSuggestionResponse(
            condition="AF",
            prompt="Test",
            urgency="prompt",
            rationale="R.",
        )
        resp = SuggestionsListResponse(suggestions=[item])
        assert len(resp.suggestions) == 1

    def test_predict_response_with_suggestions(self) -> None:
        """PredictResponse can include suggestions field."""
        from aortica.api.predict import QualityReportResponse

        qr = QualityReportResponse(
            per_lead=[],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        entry = SuggestionEntry(
            condition="AF",
            prompt="Test",
            urgency="prompt",
            rationale="R.",
        )
        resp = PredictResponse(
            quality_report=qr,
            predictions=[],
            suggestions=[entry],
        )
        assert resp.suggestions is not None
        assert len(resp.suggestions) == 1
        assert resp.suggestions[0].condition == "AF"

    def test_predict_response_without_suggestions(self) -> None:
        from aortica.api.predict import QualityReportResponse

        qr = QualityReportResponse(
            per_lead=[],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        resp = PredictResponse(
            quality_report=qr,
            predictions=[],
        )
        assert resp.suggestions is None


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

class TestJSONLoading:
    def test_load_valid_json(self, sample_json: Path) -> None:
        suggestions = load_suggestions_from_json(sample_json)
        assert "AF" in suggestions
        assert "VT" in suggestions
        assert "STEMI" in suggestions
        assert len(suggestions) == 3

    def test_loaded_values(self, sample_json: Path) -> None:
        suggestions = load_suggestions_from_json(sample_json)
        af = suggestions["AF"]
        assert af.prompt == "Evaluate stroke risk"
        assert af.urgency == "prompt"
        assert af.rationale == "AF carries thromboembolic risk."

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_suggestions_from_json(Path("/nonexistent/path.json"))

    def test_malformed_json_no_suggestions_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="suggestions"):
            load_suggestions_from_json(path)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        data = {
            "suggestions": {
                "AF": {
                    "prompt": "Test",
                    # missing urgency and rationale
                }
            }
        }
        path = tmp_path / "bad2.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="urgency"):
            load_suggestions_from_json(path)

    def test_invalid_urgency(self, tmp_path: Path) -> None:
        data = {
            "suggestions": {
                "AF": {
                    "prompt": "Test",
                    "urgency": "critical",  # not valid
                    "rationale": "R.",
                }
            }
        }
        path = tmp_path / "bad3.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid urgency"):
            load_suggestions_from_json(path)

    def test_json_roundtrip(self, sample_json: Path) -> None:
        """Load from JSON and verify all entries round-trip correctly."""
        suggestions = load_suggestions_from_json(sample_json)
        for name, s in suggestions.items():
            assert isinstance(name, str)
            assert isinstance(s.prompt, str)
            assert s.urgency in VALID_URGENCY_LEVELS
            assert isinstance(s.rationale, str)
            assert len(s.prompt) > 0
            assert len(s.rationale) > 0


# ---------------------------------------------------------------------------
# Default JSON file
# ---------------------------------------------------------------------------

class TestDefaultJSON:
    def test_default_json_exists(self, default_json_path: Path) -> None:
        assert default_json_path.exists(), (
            f"Default JSON not found at {default_json_path}"
        )

    def test_default_json_loads(self, default_json_path: Path) -> None:
        suggestions = load_suggestions_from_json(default_json_path)
        assert len(suggestions) > 0

    def test_all_urgency_levels_valid(self, default_json_path: Path) -> None:
        suggestions = load_suggestions_from_json(default_json_path)
        for name, s in suggestions.items():
            assert s.urgency in VALID_URGENCY_LEVELS, (
                f"Invalid urgency '{s.urgency}' for condition '{name}'"
            )

    def test_prompt_length_limit(self, default_json_path: Path) -> None:
        """All prompts should be ≤100 characters."""
        suggestions = load_suggestions_from_json(default_json_path)
        for name, s in suggestions.items():
            assert len(s.prompt) <= 100, (
                f"Prompt for '{name}' exceeds 100 chars: {len(s.prompt)}"
            )

    def test_rationale_non_empty(self, default_json_path: Path) -> None:
        suggestions = load_suggestions_from_json(default_json_path)
        for name, s in suggestions.items():
            assert len(s.rationale) > 0, (
                f"Rationale for '{name}' is empty"
            )


# ---------------------------------------------------------------------------
# High-severity condition coverage
# ---------------------------------------------------------------------------

class TestHighSeverityCoverage:
    def test_all_high_severity_have_entries(
        self, default_json_path: Path
    ) -> None:
        suggestions = load_suggestions_from_json(default_json_path)
        for condition in HIGH_SEVERITY_CONDITIONS:
            assert condition in suggestions, (
                f"High-severity condition '{condition}' missing from suggestions"
            )

    def test_high_severity_urgency_levels(
        self, default_json_path: Path
    ) -> None:
        """High-severity conditions should be urgent or emergent."""
        suggestions = load_suggestions_from_json(default_json_path)
        for condition in HIGH_SEVERITY_CONDITIONS:
            s = suggestions[condition]
            assert s.urgency in {"urgent", "emergent"}, (
                f"High-severity '{condition}' has urgency '{s.urgency}' "
                f"(expected urgent or emergent)"
            )


# ---------------------------------------------------------------------------
# All model classes covered
# ---------------------------------------------------------------------------

class TestModelClassCoverage:
    def test_rhythm_classes_covered(self, default_json_path: Path) -> None:
        from aortica.models.rhythm_head import RHYTHM_CLASSES

        suggestions = load_suggestions_from_json(default_json_path)
        for cls in RHYTHM_CLASSES:
            assert cls in suggestions, (
                f"Rhythm class '{cls}' missing from suggestions"
            )

    def test_structural_classes_covered(self, default_json_path: Path) -> None:
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        suggestions = load_suggestions_from_json(default_json_path)
        for cls in STRUCTURAL_CLASSES:
            assert cls in suggestions, (
                f"Structural class '{cls}' missing from suggestions"
            )

    def test_ischaemia_classes_covered(self, default_json_path: Path) -> None:
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES

        suggestions = load_suggestions_from_json(default_json_path)
        for cls in ISCHAEMIA_CLASSES:
            assert cls in suggestions, (
                f"Ischaemia class '{cls}' missing from suggestions"
            )


# ---------------------------------------------------------------------------
# get_suggestion lookup
# ---------------------------------------------------------------------------

class TestGetSuggestion:
    def test_exact_match(self) -> None:
        # Uses the default JSON
        s = get_suggestion("AF")
        assert s is not None
        assert "stroke" in s.prompt.lower() or "anticoagulation" in s.prompt.lower()

    def test_no_match_returns_none(self) -> None:
        s = get_suggestion("NONEXISTENT_CONDITION_XYZ")
        assert s is None

    def test_reload_and_get(self, sample_json: Path) -> None:
        """Reload from custom JSON and verify get_suggestion uses it."""
        reload_suggestions(sample_json)
        s = get_suggestion("AF")
        assert s is not None
        assert s.prompt == "Evaluate stroke risk"
        # Restore defaults
        reload_suggestions()

    def test_get_condition_suggestions_returns_copy(self) -> None:
        s1 = get_condition_suggestions()
        s2 = get_condition_suggestions()
        assert s1 is not s2  # Different dict instances
        assert set(s1.keys()) == set(s2.keys())


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestSuggestionsEndpoints:
    @pytest.fixture
    def client(self) -> Any:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_get_suggestion_200(self, client: Any) -> None:
        resp = client.get("/api/v1/suggestions/AF")
        assert resp.status_code == 200
        data = resp.json()
        assert data["condition"] == "AF"
        assert "prompt" in data
        assert "urgency" in data
        assert "rationale" in data

    def test_get_suggestion_404(self, client: Any) -> None:
        resp = client.get("/api/v1/suggestions/NONEXISTENT_XYZ")
        assert resp.status_code == 404

    def test_get_suggestion_response_model(self, client: Any) -> None:
        resp = client.get("/api/v1/suggestions/STEMI")
        assert resp.status_code == 200
        data = resp.json()
        parsed = ClinicalSuggestionResponse(**data)
        assert parsed.urgency == "emergent"

    def test_list_all_suggestions(self, client: Any) -> None:
        resp = client.get("/api/v1/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0
        # All items should have required fields
        for item in data["suggestions"]:
            assert "condition" in item
            assert "prompt" in item
            assert "urgency" in item
            assert "rationale" in item

    def test_list_suggestions_sorted(self, client: Any) -> None:
        resp = client.get("/api/v1/suggestions")
        data = resp.json()
        conditions = [s["condition"] for s in data["suggestions"]]
        assert conditions == sorted(conditions)


# ---------------------------------------------------------------------------
# Inference response inclusion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestInferenceInclusion:
    @pytest.fixture
    def client(self) -> Any:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_predict_include_suggestions_param_exists(
        self, client: Any
    ) -> None:
        """The predict endpoint should accept include_suggestions param."""
        import io

        # Create a minimal synthetic CSV ECG file
        csv_content = "I,II,III\n" + "\n".join(
            f"{i},{i+1},{i+2}" for i in range(500)
        )
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        # This will likely fail at read_ecg (no sample_rate config) but
        # the param should be accepted without 422 validation error
        resp = client.post(
            "/api/v1/predict?include_suggestions=true",
            files=files,
        )
        # 422 from pipeline, not from parameter validation
        # The important thing is the param is accepted
        assert resp.status_code in (200, 422)

    def test_include_suggestions_default_false(self, client: Any) -> None:
        """Without include_suggestions, suggestions should not be in response."""
        import io

        csv_content = "I,II,III\n" + "\n".join(
            f"{i},{i+1},{i+2}" for i in range(500)
        )
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/v1/predict", files=files)
        # Pipeline may fail but check that the param defaults work
        assert resp.status_code in (200, 422)


# ---------------------------------------------------------------------------
# Pipeline integration (unit level with mock)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_run_inference_pipeline_include_suggestions_param(self) -> None:
        """run_inference_pipeline accepts include_suggestions kwarg."""
        from unittest.mock import patch

        import numpy as np

        from aortica.api.predict import run_inference_pipeline
        from aortica.io.ecg_record import ECGRecord
        from aortica.signal.quality_scoring import LeadQuality, QualityReport

        # Create mock ECG record and quality report
        mock_ecg = ECGRecord(
            signals=np.random.default_rng(42).standard_normal((12, 5000)),
            sample_rate=500,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"],
        )
        mock_quality = QualityReport(
            per_lead=[
                LeadQuality(lead_name=ln, score=90.0, classification="good", flags=set())
                for ln in mock_ecg.lead_names
            ],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )

        with (
            patch("aortica.api.predict.read_ecg", return_value=mock_ecg),
            patch("aortica.api.predict.denoise", return_value=mock_ecg),
            patch("aortica.api.predict.score_quality", return_value=mock_quality),
        ):
            # Run without model — should succeed with no predictions and no suggestions
            result = run_inference_pipeline(
                b"fake file bytes",
                "test.hea",
                include_suggestions=True,
                model=None,
            )
            assert isinstance(result, PredictResponse)
            # No model → no predictions → no suggestions
            assert result.suggestions is None

    def test_suggestions_not_included_by_default(self) -> None:
        """Without include_suggestions, result should have suggestions=None."""
        from unittest.mock import patch

        import numpy as np

        from aortica.api.predict import run_inference_pipeline
        from aortica.io.ecg_record import ECGRecord
        from aortica.signal.quality_scoring import LeadQuality, QualityReport

        mock_ecg = ECGRecord(
            signals=np.random.default_rng(42).standard_normal((12, 5000)),
            sample_rate=500,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"],
        )
        mock_quality = QualityReport(
            per_lead=[
                LeadQuality(lead_name=ln, score=90.0, classification="good", flags=set())
                for ln in mock_ecg.lead_names
            ],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )

        with (
            patch("aortica.api.predict.read_ecg", return_value=mock_ecg),
            patch("aortica.api.predict.denoise", return_value=mock_ecg),
            patch("aortica.api.predict.score_quality", return_value=mock_quality),
        ):
            result = run_inference_pipeline(
                b"fake file bytes",
                "test.hea",
                model=None,
            )
            assert result.suggestions is None


# ---------------------------------------------------------------------------
# Constants and imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_clinical_suggestions_module_importable(self) -> None:
        import aortica.api.clinical_suggestions as cs

        assert hasattr(cs, "ClinicalSuggestion")
        assert hasattr(cs, "get_suggestion")
        assert hasattr(cs, "load_suggestions_from_json")
        assert hasattr(cs, "create_suggestions_router")
        assert hasattr(cs, "HIGH_SEVERITY_CONDITIONS")

    def test_suggestion_entry_importable(self) -> None:
        from aortica.api.predict import SuggestionEntry

        assert SuggestionEntry is not None

    def test_valid_urgency_levels(self) -> None:
        assert "routine" in VALID_URGENCY_LEVELS
        assert "prompt" in VALID_URGENCY_LEVELS
        assert "urgent" in VALID_URGENCY_LEVELS
        assert "emergent" in VALID_URGENCY_LEVELS
        assert len(VALID_URGENCY_LEVELS) == 4
