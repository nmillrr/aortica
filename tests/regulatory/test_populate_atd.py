"""Tests for aortica.regulatory.populate_atd — US-094.

Covers:
- ATD template auto-population with synthetic benchmark data
- Model version and date injection
- Per-task metric table rendering (rhythm, structural, ischaemia, risk)
- Calibration table rendering
- Overall summary table rendering
- Remaining FILL marker count (site-specific fields stay unfilled)
- validate_atd_completeness() for CI enforcement
- Edge cases: missing tasks, empty benchmark report
- Output file creation
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    TaskReport,
)
from aortica.regulatory.populate_atd import (
    ATDPopulationResult,
    ATDValidationResult,
    populate_atd,
    validate_atd_completeness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_class_metrics(name: str) -> ClassMetrics:
    """Create a synthetic ClassMetrics."""
    return ClassMetrics(
        name=name,
        auc=0.92,
        sensitivity=0.85,
        specificity=0.95,
        f1=0.88,
    )


def _make_classification_report(task_name: str, classes: list[str]) -> TaskReport:
    """Create a synthetic classification TaskReport."""
    return TaskReport(
        task_name=task_name,
        macro_f1=0.87,
        ece=0.04,
        per_class=[_make_class_metrics(c) for c in classes],
    )


def _make_risk_report() -> TaskReport:
    """Create a synthetic risk TaskReport."""
    return TaskReport(
        task_name="risk",
        c_index=0.78,
        brier_score=0.12,
    )


@pytest.fixture
def synthetic_benchmark() -> BenchmarkReport:
    """Full synthetic BenchmarkReport with all tasks."""
    rhythm_classes = ["AF", "AFL", "SVT", "VT"]
    structural_classes = ["LVH", "RVH", "LVSD"]
    ischaemia_classes = ["STEMI", "posterior_MI", "old_MI"]

    return BenchmarkReport(
        overall={
            "rhythm": _make_classification_report("rhythm", rhythm_classes),
            "structural": _make_classification_report("structural", structural_classes),
            "ischaemia": _make_classification_report("ischaemia", ischaemia_classes),
            "risk": _make_risk_report(),
        },
        subgroups=[],
        n_samples=1000,
        tasks_evaluated=["rhythm", "structural", "ischaemia", "risk"],
    )


@pytest.fixture
def template_path() -> str:
    """Path to the real ATD template."""
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "docs",
            "regulatory",
            "IEC_80601_2_86_ATD.md",
        )
    )


# ---------------------------------------------------------------------------
# Template existence
# ---------------------------------------------------------------------------


class TestTemplateExists:
    """Verify the ATD template file exists."""

    def test_template_file_exists(self, template_path: str) -> None:
        assert os.path.isfile(template_path), (
            f"ATD template not found at {template_path}"
        )

    def test_template_has_fill_markers(self, template_path: str) -> None:
        text = Path(template_path).read_text(encoding="utf-8")
        assert "[FILL:" in text, "Template should contain [FILL:] markers"

    def test_template_has_auto_populated_markers(self, template_path: str) -> None:
        text = Path(template_path).read_text(encoding="utf-8")
        assert "auto-populated" in text, (
            "Template should contain auto-populated markers"
        )

    def test_template_has_required_sections(self, template_path: str) -> None:
        text = Path(template_path).read_text(encoding="utf-8")
        required_sections = [
            "Algorithm Description",
            "Intended Use",
            "Training Data Description",
            "Performance Metrics",
            "Known Limitations",
            "Test Methodology",
            "Device Compatibility Matrix",
        ]
        for section in required_sections:
            assert section in text, f"Template missing required section: {section}"


# ---------------------------------------------------------------------------
# Auto-population
# ---------------------------------------------------------------------------


class TestPopulateATD:
    """Test populate_atd() auto-population."""

    def test_basic_population(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_populated.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.3.0",
            template_path=template_path,
            output_path=output,
        )
        assert isinstance(result, ATDPopulationResult)
        assert result.output_path == output
        assert result.model_version == "0.3.0"
        assert result.sections_populated > 0
        assert os.path.isfile(output)

    def test_model_version_injected(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_v.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="1.2.3",
            template_path=template_path,
            output_path=output,
        )
        assert "1.2.3" in result.content

    def test_date_injected(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_date.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        # Should contain a date in YYYY-MM-DD format
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", result.content)

    def test_rhythm_metrics_populated(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_rhythm.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "AF" in result.content
        assert "0.9200" in result.content  # AUC
        assert "0.8500" in result.content  # Sensitivity

    def test_structural_metrics_populated(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_struct.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "LVH" in result.content
        assert "RVH" in result.content

    def test_ischaemia_metrics_populated(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_isch.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "STEMI" in result.content
        assert "posterior_MI" in result.content

    def test_risk_metrics_populated(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_risk.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "0.7800" in result.content  # C-index
        assert "0.1200" in result.content  # Brier

    def test_calibration_table_populated(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_cal.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        # ECE should appear in calibration table
        assert "0.0400" in result.content

    def test_overall_summary_table(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_summary.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "Macro-F1" in result.content
        assert "0.8700" in result.content

    def test_auto_populated_markers_removed(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_no_auto.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        # No auto-populated FILL markers should remain
        assert "[FILL: auto-populated" not in result.content

    def test_site_specific_markers_remain(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "atd_fill.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        # Site-specific FILL markers should still be present
        assert result.remaining_fill_markers > 0
        assert "[FILL:" in result.content

    def test_default_output_path(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
    ) -> None:
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.5.0",
            template_path=template_path,
        )
        assert result.output_path.endswith("IEC_80601_2_86_ATD_v0.5.0.md")
        assert os.path.isfile(result.output_path)
        # Clean up
        os.remove(result.output_path)

    def test_output_file_written(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "subdir" / "atd.md")
        result = populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert os.path.isfile(output)
        content = Path(output).read_text(encoding="utf-8")
        assert content == result.content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPopulateATDEdgeCases:
    """Edge case tests for populate_atd()."""

    def test_empty_benchmark(
        self,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        empty_report = BenchmarkReport(
            overall={},
            subgroups=[],
            n_samples=0,
            tasks_evaluated=[],
        )
        output = str(tmp_path / "atd_empty.md")
        result = populate_atd(
            empty_report,
            model_version="0.0.0",
            template_path=template_path,
            output_path=output,
        )
        assert isinstance(result, ATDPopulationResult)
        # Even with empty data, the document should be generated
        assert os.path.isfile(output)

    def test_partial_tasks(
        self,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        # Only rhythm task
        report = BenchmarkReport(
            overall={
                "rhythm": _make_classification_report("rhythm", ["AF", "VT"]),
            },
            subgroups=[],
            n_samples=100,
            tasks_evaluated=["rhythm"],
        )
        output = str(tmp_path / "atd_partial.md")
        result = populate_atd(
            report,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "AF" in result.content
        assert isinstance(result.sections_populated, int)

    def test_risk_only(
        self,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        report = BenchmarkReport(
            overall={
                "risk": _make_risk_report(),
            },
            subgroups=[],
            n_samples=50,
            tasks_evaluated=["risk"],
        )
        output = str(tmp_path / "atd_risk_only.md")
        result = populate_atd(
            report,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        assert "0.7800" in result.content


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateATDCompleteness:
    """Test validate_atd_completeness() for CI enforcement."""

    def test_template_is_incomplete(self, template_path: str) -> None:
        """Raw template should have FILL markers → incomplete."""
        result = validate_atd_completeness(template_path)
        assert isinstance(result, ATDValidationResult)
        assert not result.is_complete
        assert result.marker_count > 0
        assert len(result.remaining_markers) == result.marker_count

    def test_populated_still_incomplete(
        self,
        synthetic_benchmark: BenchmarkReport,
        template_path: str,
        tmp_path: Path,
    ) -> None:
        """Populated ATD still has site-specific FILL markers → incomplete."""
        output = str(tmp_path / "atd_for_validation.md")
        populate_atd(
            synthetic_benchmark,
            model_version="0.1.0",
            template_path=template_path,
            output_path=output,
        )
        result = validate_atd_completeness(output)
        assert not result.is_complete
        assert result.marker_count > 0

    def test_fully_complete_document(self, tmp_path: Path) -> None:
        """A document with no FILL markers should pass."""
        complete_doc = tmp_path / "complete_atd.md"
        complete_doc.write_text(
            "# ATD\n\nAll sections filled. No placeholders.\n",
            encoding="utf-8",
        )
        result = validate_atd_completeness(str(complete_doc))
        assert result.is_complete
        assert result.marker_count == 0
        assert result.remaining_markers == []

    def test_validation_returns_marker_text(self, tmp_path: Path) -> None:
        """Should return the actual marker text found."""
        doc = tmp_path / "partial.md"
        doc.write_text(
            "# ATD\n\n[FILL: intended use statement]\n\nSome content.\n",
            encoding="utf-8",
        )
        result = validate_atd_completeness(str(doc))
        assert not result.is_complete
        assert result.marker_count == 1
        assert "[FILL: intended use statement]" in result.remaining_markers


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify regulatory module is importable."""

    def test_import_regulatory(self) -> None:
        import aortica.regulatory

    def test_import_populate_atd(self) -> None:
        from aortica.regulatory import populate_atd

    def test_import_validate(self) -> None:
        from aortica.regulatory import validate_atd_completeness

    def test_import_result_types(self) -> None:
        from aortica.regulatory.populate_atd import (
            ATDPopulationResult,
            ATDValidationResult,
        )
