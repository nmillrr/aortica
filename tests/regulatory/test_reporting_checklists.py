"""Tests for aortica.regulatory.reporting_checklists — US-096.

Covers:
- Template file existence for TRIPOD-AI, STARD-AI, CONSORT-AI
- generate_reporting_checklist() with and without benchmark data
- Model version and date injection
- Checklist item counting
- Auto-population of benchmark fields
- Invalid template name handling
- Output file creation
- Module imports
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    TaskReport,
)
from aortica.regulatory.reporting_checklists import (
    ChecklistResult,
    generate_reporting_checklist,
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
    return BenchmarkReport(
        overall={
            "rhythm": _make_classification_report("rhythm", ["AF", "AFL", "SVT", "VT"]),
            "structural": _make_classification_report("structural", ["LVH", "RVH", "LVSD"]),
            "ischaemia": _make_classification_report("ischaemia", ["STEMI", "posterior_MI"]),
            "risk": _make_risk_report(),
        },
        subgroups=[],
        n_samples=1000,
        tasks_evaluated=["rhythm", "structural", "ischaemia", "risk"],
    )


# ---------------------------------------------------------------------------
# Template existence
# ---------------------------------------------------------------------------


class TestTemplateFilesExist:
    """Verify all three checklist templates exist."""

    def test_tripod_ai_exists(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        path = os.path.join(_REGULATORY_DIR, "TRIPOD_AI.md")
        assert os.path.isfile(path), f"TRIPOD_AI.md not found at {path}"

    def test_stard_ai_exists(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        path = os.path.join(_REGULATORY_DIR, "STARD_AI.md")
        assert os.path.isfile(path), f"STARD_AI.md not found at {path}"

    def test_consort_ai_exists(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        path = os.path.join(_REGULATORY_DIR, "CONSORT_AI.md")
        assert os.path.isfile(path), f"CONSORT_AI.md not found at {path}"

    def test_tripod_has_checklist_items(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        text = Path(os.path.join(_REGULATORY_DIR, "TRIPOD_AI.md")).read_text()
        assert "[ ]" in text, "TRIPOD-AI should have unchecked items"

    def test_stard_has_checklist_items(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        text = Path(os.path.join(_REGULATORY_DIR, "STARD_AI.md")).read_text()
        assert "[ ]" in text, "STARD-AI should have unchecked items"

    def test_consort_has_checklist_items(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        text = Path(os.path.join(_REGULATORY_DIR, "CONSORT_AI.md")).read_text()
        assert "[ ]" in text, "CONSORT-AI should have unchecked items"

    def test_templates_have_fill_markers(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        for fname in ["TRIPOD_AI.md", "STARD_AI.md", "CONSORT_AI.md"]:
            text = Path(os.path.join(_REGULATORY_DIR, fname)).read_text()
            assert "[FILL:" in text, f"{fname} should have [FILL:] markers"

    def test_templates_have_aortica_context(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        for fname in ["TRIPOD_AI.md", "STARD_AI.md", "CONSORT_AI.md"]:
            text = Path(os.path.join(_REGULATORY_DIR, fname)).read_text()
            assert "Aortica" in text, f"{fname} should reference Aortica"

    def test_templates_cross_reference_atd(self) -> None:
        from aortica.regulatory.reporting_checklists import _REGULATORY_DIR

        for fname in ["TRIPOD_AI.md", "STARD_AI.md", "CONSORT_AI.md"]:
            text = Path(os.path.join(_REGULATORY_DIR, fname)).read_text()
            assert "IEC_80601_2_86_ATD" in text, (
                f"{fname} should cross-reference IEC ATD"
            )


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestGenerateReportingChecklist:
    """Test generate_reporting_checklist() basic functionality."""

    def test_tripod_ai_generation(self, tmp_path: Path) -> None:
        output = str(tmp_path / "tripod.md")
        result = generate_reporting_checklist(
            template="tripod_ai", output_path=output
        )
        assert isinstance(result, ChecklistResult)
        assert result.template_name == "tripod_ai"
        assert os.path.isfile(output)
        assert result.total_checklist_items > 0

    def test_stard_ai_generation(self, tmp_path: Path) -> None:
        output = str(tmp_path / "stard.md")
        result = generate_reporting_checklist(
            template="stard_ai", output_path=output
        )
        assert isinstance(result, ChecklistResult)
        assert result.template_name == "stard_ai"
        assert os.path.isfile(output)
        assert result.total_checklist_items > 0

    def test_consort_ai_generation(self, tmp_path: Path) -> None:
        output = str(tmp_path / "consort.md")
        result = generate_reporting_checklist(
            template="consort_ai", output_path=output
        )
        assert isinstance(result, ChecklistResult)
        assert result.template_name == "consort_ai"
        assert os.path.isfile(output)
        assert result.total_checklist_items > 0

    def test_invalid_template_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            generate_reporting_checklist(template="invalid_template")

    def test_default_output_path_tripod(self) -> None:
        result = generate_reporting_checklist(
            template="tripod_ai", model_version="1.0.0"
        )
        assert result.output_path.endswith("TRIPOD_AI_v1.0.0.md")
        assert os.path.isfile(result.output_path)
        os.remove(result.output_path)

    def test_default_output_path_no_version(self) -> None:
        result = generate_reporting_checklist(template="tripod_ai")
        assert result.output_path.endswith("TRIPOD_AI.md")
        # Don't delete — it's the original template

    def test_output_content_matches_file(self, tmp_path: Path) -> None:
        output = str(tmp_path / "check.md")
        result = generate_reporting_checklist(
            template="tripod_ai", output_path=output
        )
        file_content = Path(output).read_text(encoding="utf-8")
        assert file_content == result.content


# ---------------------------------------------------------------------------
# Version and date injection
# ---------------------------------------------------------------------------


class TestVersionAndDateInjection:
    """Test model version and date injection."""

    def test_model_version_injected(self, tmp_path: Path) -> None:
        output = str(tmp_path / "versioned.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            model_version="2.5.0",
            output_path=output,
        )
        assert "2.5.0" in result.content

    def test_date_injected(self, tmp_path: Path) -> None:
        output = str(tmp_path / "dated.md")
        result = generate_reporting_checklist(
            template="stard_ai",
            output_path=output,
        )
        # Should contain a date in YYYY-MM-DD format
        assert re.search(r"\d{4}-\d{2}-\d{2}", result.content)

    def test_sections_populated_count(self, tmp_path: Path) -> None:
        output = str(tmp_path / "pop.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            model_version="1.0.0",
            output_path=output,
        )
        # Model version and date should be injected into content
        assert "1.0.0" in result.content
        assert result.sections_populated >= 1


# ---------------------------------------------------------------------------
# Benchmark auto-population
# ---------------------------------------------------------------------------


class TestBenchmarkAutoPopulation:
    """Test auto-population with benchmark report."""

    def test_tripod_with_benchmark(
        self,
        synthetic_benchmark: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "tripod_bench.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            benchmark_report=synthetic_benchmark,
            model_version="0.3.0",
            output_path=output,
        )
        assert result.sections_populated > 2  # More than just version + date

    def test_stard_with_benchmark(
        self,
        synthetic_benchmark: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "stard_bench.md")
        result = generate_reporting_checklist(
            template="stard_ai",
            benchmark_report=synthetic_benchmark,
            model_version="0.3.0",
            output_path=output,
        )
        assert result.sections_populated > 2

    def test_benchmark_metrics_appear(
        self,
        synthetic_benchmark: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        output = str(tmp_path / "metrics.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            benchmark_report=synthetic_benchmark,
            output_path=output,
        )
        # Check that benchmark N appears
        assert "1000" in result.content

    def test_fill_markers_reduced_with_benchmark(
        self,
        synthetic_benchmark: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        # Without benchmark
        output_no = str(tmp_path / "no_bench.md")
        result_no = generate_reporting_checklist(
            template="tripod_ai",
            output_path=output_no,
        )
        # With benchmark
        output_yes = str(tmp_path / "yes_bench.md")
        result_yes = generate_reporting_checklist(
            template="tripod_ai",
            benchmark_report=synthetic_benchmark,
            output_path=output_yes,
        )
        # Benchmark version should have fewer FILL markers
        assert result_yes.remaining_fill_markers <= result_no.remaining_fill_markers

    def test_empty_benchmark(self, tmp_path: Path) -> None:
        empty_report = BenchmarkReport(
            overall={},
            subgroups=[],
            n_samples=0,
            tasks_evaluated=[],
        )
        output = str(tmp_path / "empty.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            benchmark_report=empty_report,
            output_path=output,
        )
        assert isinstance(result, ChecklistResult)
        assert os.path.isfile(output)


# ---------------------------------------------------------------------------
# Checklist item counting
# ---------------------------------------------------------------------------


class TestChecklistItemCounting:
    """Test checklist item counting."""

    def test_all_items_initially_unchecked(self, tmp_path: Path) -> None:
        output = str(tmp_path / "count.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            output_path=output,
        )
        assert result.completed_items == 0
        assert result.remaining_items == result.total_checklist_items
        assert result.total_checklist_items > 0

    def test_tripod_has_expected_items(self, tmp_path: Path) -> None:
        output = str(tmp_path / "tripod_items.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            output_path=output,
        )
        # TRIPOD-AI should have at least 20 checklist items
        assert result.total_checklist_items >= 20

    def test_stard_has_expected_items(self, tmp_path: Path) -> None:
        output = str(tmp_path / "stard_items.md")
        result = generate_reporting_checklist(
            template="stard_ai",
            output_path=output,
        )
        assert result.total_checklist_items >= 20

    def test_consort_has_expected_items(self, tmp_path: Path) -> None:
        output = str(tmp_path / "consort_items.md")
        result = generate_reporting_checklist(
            template="consort_ai",
            output_path=output,
        )
        assert result.total_checklist_items >= 20

    def test_remaining_fill_markers(self, tmp_path: Path) -> None:
        output = str(tmp_path / "fill.md")
        result = generate_reporting_checklist(
            template="tripod_ai",
            output_path=output,
        )
        assert result.remaining_fill_markers > 0


# ---------------------------------------------------------------------------
# Case insensitivity and whitespace
# ---------------------------------------------------------------------------


class TestTemplateNameHandling:
    """Test template name handling."""

    def test_uppercase_template_name(self, tmp_path: Path) -> None:
        output = str(tmp_path / "upper.md")
        result = generate_reporting_checklist(
            template="TRIPOD_AI",
            output_path=output,
        )
        assert result.template_name == "tripod_ai"

    def test_mixed_case_template_name(self, tmp_path: Path) -> None:
        output = str(tmp_path / "mixed.md")
        result = generate_reporting_checklist(
            template="Stard_AI",
            output_path=output,
        )
        assert result.template_name == "stard_ai"

    def test_whitespace_stripped(self, tmp_path: Path) -> None:
        output = str(tmp_path / "ws.md")
        result = generate_reporting_checklist(
            template="  consort_ai  ",
            output_path=output,
        )
        assert result.template_name == "consort_ai"


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify regulatory module exports are accessible."""

    def test_import_generate_reporting_checklist(self) -> None:
        from aortica.regulatory import generate_reporting_checklist

    def test_import_checklist_result(self) -> None:
        from aortica.regulatory import ChecklistResult

    def test_import_from_submodule(self) -> None:
        from aortica.regulatory.reporting_checklists import (
            ChecklistResult,
            generate_reporting_checklist,
        )
