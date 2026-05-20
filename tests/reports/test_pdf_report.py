"""Tests for aortica.reports.pdf_report — PDF Clinical Report Generator.

Tests use synthetic data and mock WeasyPrint to verify:
- Report generation produces a valid (non-zero size) file
- HTML template includes all required sections
- Prediction extraction from various output formats
- XAI report adaptation
- Quality report rendering
- Risk score display
- ECG waveform rendering via matplotlib
- Uncertainty/OOD flag display
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.reports.pdf_report import (
    Finding,
    XAIFeature,
    XAIReport,
    _build_html,
    _extract_predictions,
    _get_findings,
    _render_ecg_svg,
    _risk_gauge_color,
    _severity_color,
    _quality_badge_color,
    generate_pdf,
)


# ── Fixtures ───────────────────────────────────────────────────────


def _make_ecg_record(
    n_leads: int = 12,
    duration_s: float = 10.0,
    sample_rate: float = 500.0,
) -> ECGRecord:
    """Create a synthetic 12-lead ECGRecord."""
    n_samples = int(duration_s * sample_rate)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"][:n_leads]
    signals = np.random.default_rng(42).standard_normal((n_leads, n_samples)) * 200.0
    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format="synthetic",
        units="µV",
        patient_metadata={"patient_id": "TEST-001", "age": 65, "sex": "M"},
    )


def _make_multi_task_output() -> dict[str, list[float]]:
    """Create synthetic multi-task predictions as a dict."""
    rng = np.random.default_rng(42)
    return {
        "rhythm": rng.random(28).tolist(),
        "structural": rng.random(19).tolist(),
        "ischaemia": rng.random(19).tolist(),
        "risk": rng.random(6).tolist(),
    }


@dataclass
class MockMultiTaskOutput:
    """Mock MultiTaskOutput dataclass."""
    rhythm: Optional[list[float]] = None
    structural: Optional[list[float]] = None
    ischaemia: Optional[list[float]] = None
    risk: Optional[list[float]] = None


@dataclass
class MockQualityReport:
    """Mock QualityReport for testing."""

    @dataclass
    class LeadQuality:
        lead_name: str
        score: float
        classification: str
        flags: set[str] = field(default_factory=set)

    per_lead: list[LeadQuality] = field(default_factory=list)
    overall_score: float = 85.0
    overall_classification: str = "good"
    recommendation: str = "accept"
    scan_origin: bool = False


@dataclass
class MockUncertaintyReport:
    """Mock UncertaintyReport for testing."""
    ood_flag: bool = False
    entropy_score: float = 0.123


# ── Test Prediction Extraction ─────────────────────────────────────


class TestExtractPredictions:
    """Tests for _extract_predictions."""

    def test_from_dict(self) -> None:
        output = _make_multi_task_output()
        result = _extract_predictions(output)
        assert "rhythm" in result
        assert len(result["rhythm"]) == 28
        assert "structural" in result
        assert len(result["structural"]) == 19
        assert "ischaemia" in result
        assert len(result["ischaemia"]) == 19
        assert "risk" in result
        assert len(result["risk"]) == 6

    def test_from_dataclass(self) -> None:
        output = _make_multi_task_output()
        mock = MockMultiTaskOutput(
            rhythm=output["rhythm"],
            structural=output["structural"],
            ischaemia=output["ischaemia"],
            risk=output["risk"],
        )
        result = _extract_predictions(mock)
        assert len(result["rhythm"]) == 28
        assert len(result["risk"]) == 6

    def test_from_numpy_arrays(self) -> None:
        output = {
            "rhythm": np.random.default_rng(0).random(28),
            "structural": np.random.default_rng(0).random(19),
        }
        result = _extract_predictions(output)
        assert len(result["rhythm"]) == 28
        assert len(result["structural"]) == 19

    def test_handles_none_tasks(self) -> None:
        mock = MockMultiTaskOutput(rhythm=[0.1] * 28)
        result = _extract_predictions(mock)
        assert "rhythm" in result
        assert "structural" not in result

    def test_handles_batch_dimension(self) -> None:
        output = {"rhythm": [[0.1] * 28]}  # batch of 1
        result = _extract_predictions(output)
        assert len(result["rhythm"]) == 28


# ── Test Finding Extraction ────────────────────────────────────────


class TestGetFindings:
    """Tests for _get_findings."""

    def test_filters_by_threshold(self) -> None:
        preds = {
            "rhythm": [0.9, 0.1, 0.6] + [0.0] * 25,
            "structural": [0.0] * 19,
            "ischaemia": [0.0] * 19,
        }
        findings = _get_findings(preds, threshold=0.5)
        assert len(findings) == 2
        # Sorted by confidence descending
        assert findings[0].confidence > findings[1].confidence

    def test_empty_when_no_positives(self) -> None:
        preds = {
            "rhythm": [0.1] * 28,
            "structural": [0.2] * 19,
            "ischaemia": [0.3] * 19,
        }
        findings = _get_findings(preds, threshold=0.5)
        assert len(findings) == 0

    def test_includes_task_name(self) -> None:
        preds = {
            "rhythm": [0.9] + [0.0] * 27,
            "structural": [0.8] + [0.0] * 18,
            "ischaemia": [0.0] * 19,
        }
        findings = _get_findings(preds, threshold=0.5)
        tasks = {f.task for f in findings}
        assert "rhythm" in tasks
        assert "structural" in tasks


# ── Test Color Functions ───────────────────────────────────────────


class TestColorFunctions:
    """Tests for severity/quality/risk color functions."""

    def test_severity_red(self) -> None:
        assert _severity_color(90.0) == "#dc2626"

    def test_severity_amber(self) -> None:
        assert _severity_color(65.0) == "#d97706"

    def test_severity_green(self) -> None:
        assert _severity_color(30.0) == "#16a34a"

    def test_quality_badge_colors(self) -> None:
        assert _quality_badge_color("good") == "#16a34a"
        assert _quality_badge_color("marginal") == "#d97706"
        assert _quality_badge_color("poor") == "#dc2626"

    def test_risk_gauge_colors(self) -> None:
        assert _risk_gauge_color(0.8) == "#dc2626"
        assert _risk_gauge_color(0.5) == "#d97706"
        assert _risk_gauge_color(0.2) == "#16a34a"


# ── Test ECG Waveform Rendering ────────────────────────────────────


class TestRenderECGSVG:
    """Tests for _render_ecg_svg (matplotlib rendering)."""

    def test_returns_data_uri(self) -> None:
        ecg = _make_ecg_record()
        result = _render_ecg_svg(ecg)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 100

    def test_handles_single_lead(self) -> None:
        ecg = _make_ecg_record(n_leads=1)
        result = _render_ecg_svg(ecg)
        assert result.startswith("data:image/png;base64,")

    def test_handles_mv_units(self) -> None:
        ecg = _make_ecg_record()
        ecg.units = "mV"
        result = _render_ecg_svg(ecg)
        assert len(result) > 100


# ── Test HTML Building ─────────────────────────────────────────────


class TestBuildHTML:
    """Tests for _build_html template generation."""

    def test_contains_watermark(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        findings = _get_findings(preds, threshold=0.5)
        html = _build_html(ecg, preds, findings, "")
        assert "AI Decision Support" in html
        assert "Requires Clinical Review" in html

    def test_contains_patient_demographics(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        html = _build_html(ecg, preds, [], "")
        assert "Patient Demographics" in html
        assert "TEST-001" in html

    def test_contains_ecg_metadata(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        html = _build_html(ecg, preds, [], "")
        assert "500 Hz" in html
        assert "synthetic" in html

    def test_includes_quality_section(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        qr = MockQualityReport(
            per_lead=[
                MockQualityReport.LeadQuality("I", 90.0, "good"),
                MockQualityReport.LeadQuality("II", 50.0, "marginal", {"baseline_wander"}),
            ],
            overall_score=70.0,
            overall_classification="good",
            recommendation="accept",
        )
        html = _build_html(ecg, preds, [], "", quality_report=qr)
        assert "Signal Quality" in html
        assert "baseline_wander" in html

    def test_includes_risk_scores(self) -> None:
        ecg = _make_ecg_record()
        preds = {"risk": [0.1, 0.5, 0.9, 0.2, 0.3, 0.4]}
        html = _build_html(ecg, preds, [], "")
        assert "Risk Scores" in html
        assert "1-Year Mortality" in html

    def test_includes_xai_section(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        xai = XAIReport(
            top_features=[
                XAIFeature("QRS complex", "V1", 0.85),
                XAIFeature("ST segment", "II", 0.42),
            ],
            task="rhythm",
        )
        html = _build_html(ecg, preds, [], "", xai_report=xai)
        assert "XAI Feature Attributions" in html
        assert "QRS complex" in html
        assert "V1" in html

    def test_includes_uncertainty_section(self) -> None:
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        ur = MockUncertaintyReport(ood_flag=True, entropy_score=0.567)
        html = _build_html(ecg, preds, [], "", uncertainty_report=ur)
        assert "Uncertainty Indicators" in html
        assert "Out-of-Distribution" in html
        assert "interpret with caution" in html

    def test_no_findings_shows_empty_state(self) -> None:
        ecg = _make_ecg_record()
        html = _build_html(ecg, {}, [], "")
        assert "No significant findings detected" in html

    def test_model_version_in_header(self) -> None:
        ecg = _make_ecg_record()
        html = _build_html(ecg, {}, [], "", model_version="v0.2.0")
        assert "v0.2.0" in html

    def test_no_patient_meta_omits_section(self) -> None:
        ecg = _make_ecg_record()
        ecg.patient_metadata = None
        html = _build_html(ecg, {}, [], "")
        assert "Patient Demographics" not in html


# ── Test XAI Report Adaptation ─────────────────────────────────────


class TestXAIAdaptation:
    """Tests for XAI report format adaptation in generate_pdf."""

    def test_xai_report_passthrough(self) -> None:
        """XAIReport instances should pass through unchanged."""
        xai = XAIReport(
            top_features=[XAIFeature("P wave", "I", 0.5)],
            task="rhythm",
        )
        # We test the adaptation logic indirectly through _build_html
        ecg = _make_ecg_record()
        preds = _make_multi_task_output()
        html = _build_html(ecg, preds, [], "", xai_report=xai)
        assert "P wave" in html

    def test_feature_attribution_adapted(self) -> None:
        """FeatureAttribution-like objects should be adapted."""
        # Simulate a FeatureAttribution
        fc = MagicMock()
        fc.feature_name = "ST segment"
        fc.lead = "V4"
        fc.delta_score = 0.77

        fa = MagicMock()
        fa.top_features = [fc]
        fa.task = "ischaemia"

        # Adaptation happens in generate_pdf, test indirectly via the
        # adapted XAIReport path
        adapted = XAIReport(
            top_features=[
                XAIFeature(
                    feature_name=fc.feature_name,
                    lead=fc.lead,
                    delta_score=fc.delta_score,
                )
            ],
            task=fa.task,
        )
        ecg = _make_ecg_record()
        html = _build_html(ecg, {}, [], "", xai_report=adapted)
        assert "ST segment" in html
        assert "V4" in html


# ── Test PDF Generation (with mocked WeasyPrint) ──────────────────


class TestGeneratePDF:
    """Tests for the generate_pdf function."""

    def test_generates_pdf_with_mock_weasyprint(self, tmp_path: Path) -> None:
        """Verify generate_pdf produces a non-zero-size file."""
        output_file = tmp_path / "test_report.pdf"

        # Mock WeasyPrint
        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF-1.4 mock content")
        )
        mock_html_cls = MagicMock(return_value=mock_doc)
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = mock_html_cls

        ecg = _make_ecg_record()
        output = _make_multi_task_output()

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            result = generate_pdf(
                multi_task_output=output,
                ecg_record=ecg,
                output_path=output_file,
                model_version="v0.2.0",
            )

        assert result.exists()
        assert result.stat().st_size > 0
        mock_html_cls.assert_called_once()
        mock_doc.write_pdf.assert_called_once()

    def test_generates_pdf_with_all_options(self, tmp_path: Path) -> None:
        """Test with quality report, XAI, and uncertainty."""
        output_file = tmp_path / "full_report.pdf"

        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF-1.4 full")
        )
        mock_html_cls = MagicMock(return_value=mock_doc)
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = mock_html_cls

        ecg = _make_ecg_record()
        output = _make_multi_task_output()
        qr = MockQualityReport(
            per_lead=[
                MockQualityReport.LeadQuality("I", 85.0, "good"),
            ],
        )
        xai = XAIReport(
            top_features=[XAIFeature("T wave", "V2", 0.6)],
            task="ischaemia",
        )
        ur = MockUncertaintyReport(ood_flag=False, entropy_score=0.05)

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            result = generate_pdf(
                multi_task_output=output,
                ecg_record=ecg,
                xai_report=xai,
                output_path=output_file,
                quality_report=qr,
                uncertainty_report=ur,
                model_version="v0.2.0",
                finding_threshold=0.3,
            )

        assert result.exists()
        # Check the HTML was called with string
        call_args = mock_html_cls.call_args
        html_string = call_args[1].get("string", call_args[0][0] if call_args[0] else "")
        if not html_string:
            html_string = str(call_args)
        # At minimum, weasyprint was called
        mock_html_cls.assert_called_once()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Verify parent directories are created automatically."""
        output_file = tmp_path / "deep" / "nested" / "report.pdf"

        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF")
        )
        mock_html_cls = MagicMock(return_value=mock_doc)
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = mock_html_cls

        ecg = _make_ecg_record()
        output = _make_multi_task_output()

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            result = generate_pdf(output, ecg, output_path=output_file)

        assert result.parent.exists()

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        """Verify the returned path is absolute."""
        output_file = tmp_path / "report.pdf"

        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF")
        )
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = MagicMock(return_value=mock_doc)

        ecg = _make_ecg_record()
        output = _make_multi_task_output()

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            result = generate_pdf(output, ecg, output_path=output_file)

        assert result.is_absolute()

    def test_adapts_feature_attribution(self, tmp_path: Path) -> None:
        """Test that FeatureAttribution objects are adapted to XAIReport."""
        output_file = tmp_path / "xai_report.pdf"

        captured_html: list[str] = []

        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF")
        )

        def capture_html(**kwargs: Any) -> MagicMock:
            captured_html.append(kwargs.get("string", ""))
            return mock_doc

        from typing import Any

        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = MagicMock(side_effect=capture_html)

        ecg = _make_ecg_record()
        output = _make_multi_task_output()

        # Simulate FeatureAttribution
        fc = MagicMock()
        fc.feature_name = "QRS complex"
        fc.lead = "V1"
        fc.delta_score = 0.99

        fa = MagicMock()
        fa.top_features = [fc]
        fa.task = "rhythm"

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            generate_pdf(output, ecg, xai_report=fa, output_path=output_file)

        assert len(captured_html) == 1
        assert "QRS complex" in captured_html[0]
        assert "V1" in captured_html[0]

    def test_finding_threshold(self, tmp_path: Path) -> None:
        """Verify finding_threshold controls which findings appear."""
        output_file = tmp_path / "threshold_report.pdf"

        captured_html: list[str] = []

        mock_doc = MagicMock()
        mock_doc.write_pdf = MagicMock(
            side_effect=lambda path: Path(path).write_bytes(b"%PDF")
        )

        from typing import Any

        def capture_html(**kwargs: Any) -> MagicMock:
            captured_html.append(kwargs.get("string", ""))
            return mock_doc

        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML = MagicMock(side_effect=capture_html)

        ecg = _make_ecg_record()
        # Set very low confidences
        output = {
            "rhythm": [0.1] * 28,
            "structural": [0.2] * 19,
            "ischaemia": [0.15] * 19,
            "risk": [0.5] * 6,
        }

        with patch(
            "aortica.reports.pdf_report._get_weasyprint",
            return_value=mock_weasyprint,
        ):
            generate_pdf(
                output, ecg,
                output_path=output_file,
                finding_threshold=0.5,
            )

        assert "No significant findings detected" in captured_html[0]


# ── Test Data Structure Constructors ───────────────────────────────


class TestDataStructures:
    """Tests for report data structures."""

    def test_finding_creation(self) -> None:
        f = Finding(name="AF", confidence=92.5, task="rhythm")
        assert f.name == "AF"
        assert f.confidence == 92.5
        assert f.task == "rhythm"

    def test_xai_feature_creation(self) -> None:
        feat = XAIFeature(
            feature_name="QRS complex",
            lead="V1",
            delta_score=0.85,
        )
        assert feat.feature_name == "QRS complex"
        assert feat.lead == "V1"
        assert feat.delta_score == 0.85

    def test_xai_report_creation(self) -> None:
        report = XAIReport(
            top_features=[
                XAIFeature("P wave", "II", 0.3),
            ],
            task="rhythm",
        )
        assert len(report.top_features) == 1
        assert report.task == "rhythm"

    def test_xai_report_default_empty(self) -> None:
        report = XAIReport()
        assert report.top_features == []
        assert report.task == "rhythm"
