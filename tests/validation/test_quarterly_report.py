"""Tests for aortica.validation.quarterly_report — US-101.

Covers:
- Quarterly report generation (markdown + CSV)
- Report content validation
- Quarter-over-quarter comparison
- Empty monitor handling
- Invalid quarter validation
- CLI command registration
- Module imports
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from aortica.validation.performance_monitor import PerformanceMonitor
from aortica.validation.quarterly_report import (
    QuarterlyReportResult,
    _quarter_date_range,
    _quarter_label,
    generate_quarterly_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_monitor(tmp_path: Path) -> PerformanceMonitor:
    """Monitor with synthetic labeled production data."""
    mon = PerformanceMonitor(db_dir=str(tmp_path / "monitor"))
    now = time.time()

    for i in range(30):
        ecg_id = f"ecg_{i:03d}"
        is_af = i % 5 == 0
        af_pred = 0.88 if is_af else 0.08

        mon.record_prediction(
            ecg_id=ecg_id,
            task="rhythm",
            predictions={"AF": af_pred},
            ground_truth={"AF": int(is_af)},
            timestamp=now - 86400 * (30 - i) / 30,
        )

    return mon


@pytest.fixture
def previous_monitor(tmp_path: Path) -> PerformanceMonitor:
    """Monitor representing a previous quarter with slightly different metrics."""
    mon = PerformanceMonitor(db_dir=str(tmp_path / "prev_monitor"))
    now = time.time()

    for i in range(20):
        ecg_id = f"prev_ecg_{i:03d}"
        is_af = i % 4 == 0
        af_pred = 0.80 if is_af else 0.15

        mon.record_prediction(
            ecg_id=ecg_id,
            task="rhythm",
            predictions={"AF": af_pred},
            ground_truth={"AF": int(is_af)},
            timestamp=now - 86400 * (20 - i) / 20,
        )

    return mon


@pytest.fixture
def empty_monitor(tmp_path: Path) -> PerformanceMonitor:
    """Monitor with no data."""
    return PerformanceMonitor(db_dir=str(tmp_path / "empty_monitor"))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Test quarterly report file generation."""

    def test_generates_markdown_file(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "output"), quarter=1, year=2026
        )
        assert Path(result.markdown_path).exists()
        assert result.markdown_path.endswith(".md")

    def test_generates_csv_file(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "output"), quarter=2, year=2026
        )
        assert Path(result.csv_path).exists()
        assert result.csv_path.endswith(".csv")

    def test_file_naming(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "output"), quarter=3, year=2026
        )
        assert "QUARTERLY_REPORT_2026_Q3.md" in result.markdown_path
        assert "quarterly_report_2026_Q3.csv" in result.csv_path

    def test_result_metadata(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "output"), quarter=1, year=2026
        )
        assert result.quarter == 1
        assert result.year == 2026
        assert result.total_ecgs > 0
        assert "rhythm" in result.tasks_reported

    def test_creates_output_dir(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        nested = str(tmp_path / "deep" / "nested" / "output")
        result = generate_quarterly_report(
            populated_monitor, nested, quarter=1, year=2026
        )
        assert Path(result.markdown_path).exists()


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------


class TestMarkdownContent:
    """Test markdown report content."""

    def test_has_title(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "Quarterly Performance Report" in content
        assert "Q1 2026" in content

    def test_has_period_dates(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "2026-01-01" in content
        assert "2026-03-31" in content

    def test_has_metrics_table(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "Per-Task Metrics" in content
        assert "AUC" in content
        assert "F1" in content
        assert "rhythm" in content

    def test_has_drift_section(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "Drift Alerts" in content

    def test_has_disclaimer(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "Requires Clinical Review" in content


# ---------------------------------------------------------------------------
# CSV content
# ---------------------------------------------------------------------------


class TestCSVContent:
    """Test CSV report content."""

    def test_csv_has_headers(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        with open(result.csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        expected = [
            "quarter", "year", "subgroup", "task", "auc", "f1", "ece",
            "n_samples", "total_predictions", "total_labeled",
            "drift_detected",
        ]
        assert headers == expected

    def test_csv_has_data_rows(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=2, year=2026
        )
        with open(result.csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + at least 1 data row
        assert len(rows) >= 2
        assert rows[1][0] == "Q2"
        assert rows[1][1] == "2026"

    def test_csv_valid_format(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        with open(result.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert "auc" in rows[0]
        assert "f1" in rows[0]


# ---------------------------------------------------------------------------
# Quarter comparison
# ---------------------------------------------------------------------------


class TestQuarterComparison:
    """Test quarter-over-quarter comparison."""

    def test_comparison_section_present(
        self,
        populated_monitor: PerformanceMonitor,
        previous_monitor: PerformanceMonitor,
        tmp_path: Path,
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor,
            str(tmp_path / "out"),
            quarter=2,
            year=2026,
            previous_monitor=previous_monitor,
        )
        content = Path(result.markdown_path).read_text()
        assert "Comparison to Previous Quarter" in content
        assert "Current" in content
        assert "Previous" in content

    def test_no_previous_data(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            populated_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        content = Path(result.markdown_path).read_text()
        assert "No previous quarter data" in content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_monitor(
        self, empty_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        result = generate_quarterly_report(
            empty_monitor, str(tmp_path / "out"), quarter=1, year=2026
        )
        assert result.total_ecgs == 0
        assert result.tasks_reported == []
        assert Path(result.markdown_path).exists()
        assert Path(result.csv_path).exists()

    def test_invalid_quarter(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Quarter must be"):
            generate_quarterly_report(
                populated_monitor, str(tmp_path / "out"), quarter=5, year=2026
            )

    def test_all_quarters(
        self, populated_monitor: PerformanceMonitor, tmp_path: Path
    ) -> None:
        for q in (1, 2, 3, 4):
            result = generate_quarterly_report(
                populated_monitor, str(tmp_path / "out"), quarter=q, year=2026
            )
            assert result.quarter == q
            assert Path(result.markdown_path).exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test helper functions."""

    def test_quarter_label(self) -> None:
        assert _quarter_label(1, 2026) == "Q1 2026"
        assert _quarter_label(4, 2025) == "Q4 2025"

    def test_quarter_date_range_q1(self) -> None:
        start, end = _quarter_date_range(1, 2026)
        assert start == "2026-01-01"
        assert end == "2026-03-31"

    def test_quarter_date_range_q2(self) -> None:
        start, end = _quarter_date_range(2, 2026)
        assert start == "2026-04-01"
        assert end == "2026-06-30"

    def test_quarter_date_range_q4(self) -> None:
        start, end = _quarter_date_range(4, 2026)
        assert start == "2026-10-01"
        assert end == "2026-12-31"


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestCLICommand:
    """Test CLI command registration."""

    def test_validation_group_exists(self) -> None:
        from aortica.cli.validation_cmd import validation_group

        assert validation_group.name == "validation"

    def test_quarterly_report_command(self) -> None:
        from aortica.cli.validation_cmd import validation_group

        commands = list(validation_group.commands.keys())
        assert "quarterly-report" in commands

    def test_cli_includes_validation(self) -> None:
        from aortica.cli import _build_cli

        cli = _build_cli()
        commands = list(cli.commands.keys())
        assert "validation" in commands


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify quarterly report module exports are accessible."""

    def test_import_generate_quarterly_report(self) -> None:
        from aortica.validation import generate_quarterly_report  # noqa: F811

    def test_import_quarterly_report_result(self) -> None:
        from aortica.validation import QuarterlyReportResult  # noqa: F811

    def test_import_from_submodule(self) -> None:
        from aortica.validation.quarterly_report import (  # noqa: F811
            QuarterlyReportResult,
            generate_quarterly_report,
        )
