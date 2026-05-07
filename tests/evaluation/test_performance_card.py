"""Tests for US-070: Public Performance Card Generator.

Tests cover:
- PerformanceCardResult dataclass
- SHA-256 computation
- Markdown generation (overall, subgroups, equity gate)
- CSV generation (overall, subgroups)
- generate_performance_card() file output
- CLI command (help, invocation, missing args, JSON round-trip)
- Imports from evaluation package
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    SubgroupReport,
    TaskReport,
)
from aortica.evaluation.performance_card import (
    PerformanceCardResult,
    _compute_sha256,
    _generate_csv,
    _generate_markdown,
    generate_performance_card,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_classification_task_report(
    task_name: str,
    num_classes: int = 3,
) -> TaskReport:
    """Build a synthetic classification TaskReport."""
    per_class = []
    for i in range(num_classes):
        per_class.append(
            ClassMetrics(
                name=f"class_{i}",
                auc=0.90 + i * 0.01,
                sensitivity=0.85 + i * 0.02,
                specificity=0.92 + i * 0.01,
                f1=0.88 + i * 0.01,
            )
        )
    return TaskReport(
        task_name=task_name,
        macro_f1=0.8900,
        ece=0.0350,
        per_class=per_class,
    )


def _make_risk_task_report() -> TaskReport:
    """Build a synthetic risk TaskReport."""
    return TaskReport(
        task_name="risk",
        c_index=0.7800,
        brier_score=0.1200,
    )


def _make_benchmark_report(
    *,
    with_subgroups: bool = False,
) -> BenchmarkReport:
    """Build a synthetic BenchmarkReport."""
    overall = {
        "rhythm": _make_classification_task_report("rhythm", num_classes=3),
        "structural": _make_classification_task_report("structural", num_classes=2),
        "risk": _make_risk_task_report(),
    }
    tasks_evaluated = ["rhythm", "structural", "risk"]
    subgroups: list[SubgroupReport] = []

    if with_subgroups:
        subgroups = [
            SubgroupReport(
                subgroup_name="sex_M",
                n_samples=500,
                task_reports={
                    "rhythm": TaskReport(
                        task_name="rhythm",
                        macro_f1=0.8850,
                        ece=0.0380,
                        per_class=[
                            ClassMetrics(name="class_0", auc=0.89, sensitivity=0.84, specificity=0.91, f1=0.87),
                        ],
                    ),
                    "risk": TaskReport(
                        task_name="risk",
                        c_index=0.7700,
                        brier_score=0.1250,
                    ),
                },
            ),
            SubgroupReport(
                subgroup_name="sex_F",
                n_samples=450,
                task_reports={
                    "rhythm": TaskReport(
                        task_name="rhythm",
                        macro_f1=0.8950,
                        ece=0.0320,
                        per_class=[
                            ClassMetrics(name="class_0", auc=0.91, sensitivity=0.86, specificity=0.93, f1=0.89),
                        ],
                    ),
                    "risk": TaskReport(
                        task_name="risk",
                        c_index=0.7900,
                        brier_score=0.1150,
                    ),
                },
            ),
            SubgroupReport(
                subgroup_name="age_30-39",
                n_samples=200,
                task_reports={
                    "rhythm": TaskReport(
                        task_name="rhythm",
                        macro_f1=0.8800,
                        ece=0.0400,
                    ),
                },
            ),
            SubgroupReport(
                subgroup_name="age_60-69",
                n_samples=300,
                task_reports={
                    "rhythm": TaskReport(
                        task_name="rhythm",
                        macro_f1=0.8700,
                        ece=0.0450,
                    ),
                },
            ),
        ]

    return BenchmarkReport(
        overall=overall,
        subgroups=subgroups,
        n_samples=1000,
        tasks_evaluated=tasks_evaluated,
    )


# ---------------------------------------------------------------------------
# PerformanceCardResult dataclass tests
# ---------------------------------------------------------------------------


class TestPerformanceCardResult:
    """Tests for the PerformanceCardResult dataclass."""

    def test_defaults(self) -> None:
        result = PerformanceCardResult()
        assert result.markdown_path == ""
        assert result.csv_path == ""
        assert result.markdown_content == ""
        assert result.csv_content == ""
        assert result.model_version == ""
        assert result.timestamp == ""
        assert result.model_weights_sha256 == ""

    def test_custom_values(self) -> None:
        result = PerformanceCardResult(
            markdown_path="/tmp/PERFORMANCE_CARD.md",
            csv_path="/tmp/performance_card.csv",
            markdown_content="# Card",
            csv_content="a,b\n1,2\n",
            model_version="0.3.0",
            timestamp="2026-05-07T00:00:00Z",
            model_weights_sha256="abc123",
        )
        assert result.model_version == "0.3.0"
        assert result.model_weights_sha256 == "abc123"


# ---------------------------------------------------------------------------
# SHA-256 tests
# ---------------------------------------------------------------------------


class TestComputeSha256:
    """Tests for the _compute_sha256 helper."""

    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        content = b"fake model weights"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _compute_sha256(str(f)) == expected

    def test_nonexistent_file(self) -> None:
        assert _compute_sha256("/nonexistent/path.pt") == ""

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.pt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _compute_sha256(str(f)) == expected


# ---------------------------------------------------------------------------
# Markdown generation tests
# ---------------------------------------------------------------------------


class TestGenerateMarkdown:
    """Tests for _generate_markdown()."""

    def test_header_contains_version(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "# Performance Card — Aortica v0.3.0" in md
        assert "2026-05-07T00:00:00Z" in md

    def test_sha256_included_when_present(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "abc123def", None, None,
        )
        assert "`abc123def`" in md

    def test_sha256_omitted_when_empty(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "SHA-256" not in md

    def test_dataset_info_included(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "",
            "PTB-XL test fold 10", None,
        )
        assert "PTB-XL test fold 10" in md

    def test_classification_task_table(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        # Check table header
        assert "| Class | AUC |" in md
        # Check per-class rows
        assert "class_0" in md
        assert "0.9000" in md
        # Check macro-F1
        assert "Macro-F1" in md

    def test_risk_task_metrics(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "C-index" in md
        assert "0.7800" in md
        assert "Brier Score" in md
        assert "0.1200" in md

    def test_subgroup_section_present(self) -> None:
        report = _make_benchmark_report(with_subgroups=True)
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "## Demographic Subgroup Breakdowns" in md
        assert "### By Sex" in md
        assert "### By Age Decile" in md
        assert "sex_M" in md
        assert "sex_F" in md
        assert "age_30-39" in md

    def test_no_subgroup_section_without_subgroups(self) -> None:
        report = _make_benchmark_report(with_subgroups=False)
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "## Demographic Subgroup Breakdowns" not in md

    def test_equity_gate_passed(self) -> None:
        from aortica.evaluation.equity_gate import EquityGateResult

        eq = EquityGateResult(
            passed=True,
            alpha=0.05,
            correction="bonferroni",
            corrected_alpha=0.005,
            num_comparisons=10,
        )
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, eq,
        )
        assert "## Equity Gate Results" in md
        assert "PASSED ✓" in md

    def test_equity_gate_failed_with_details(self) -> None:
        from aortica.evaluation.equity_gate import (
            ComparisonResult,
            EquityGateResult,
        )

        fc = ComparisonResult(
            group_a="sex_M",
            group_b="sex_F",
            task="rhythm",
            class_index=0,
            class_name="AF",
            auc_a=0.95,
            auc_b=0.80,
            auc_diff=0.15,
            p_value=0.001,
            significant=True,
            n_a=500,
            n_b=450,
        )
        eq = EquityGateResult(
            passed=False,
            alpha=0.05,
            correction="bonferroni",
            corrected_alpha=0.005,
            num_comparisons=10,
            failing_comparisons=[fc],
        )
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, eq,
        )
        assert "FAILED ✗" in md
        assert "#### Failing Comparisons" in md
        assert "AF" in md
        assert "sex_M" in md

    def test_footer_present(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "auto-generated" in md

    def test_n_samples_displayed(self) -> None:
        report = _make_benchmark_report()
        md = _generate_markdown(
            report, "0.3.0", "2026-05-07T00:00:00Z", "", None, None,
        )
        assert "1000" in md


# ---------------------------------------------------------------------------
# CSV generation tests
# ---------------------------------------------------------------------------


class TestGenerateCsv:
    """Tests for _generate_csv()."""

    def _parse_csv(self, csv_text: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(csv_text))
        return list(reader)

    def test_header_row(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "sha_abc")
        rows = self._parse_csv(csv_text)
        assert len(rows) > 0
        expected_cols = {
            "model_version", "timestamp", "weights_sha256", "n_samples",
            "subgroup", "task", "class", "auc", "sensitivity", "specificity",
            "f1", "macro_f1", "ece", "c_index", "brier_score",
        }
        assert set(rows[0].keys()) == expected_cols

    def test_overall_rows_count(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "")
        rows = self._parse_csv(csv_text)
        # rhythm has 3 classes, structural has 2 classes, risk has 1 row = 6
        overall_rows = [r for r in rows if r["subgroup"] == "overall"]
        assert len(overall_rows) == 6

    def test_version_in_all_rows(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "1.2.3", "2026-05-07", "")
        rows = self._parse_csv(csv_text)
        for row in rows:
            assert row["model_version"] == "1.2.3"

    def test_subgroup_rows_present(self) -> None:
        report = _make_benchmark_report(with_subgroups=True)
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "")
        rows = self._parse_csv(csv_text)
        subgroup_rows = [r for r in rows if r["subgroup"] != "overall"]
        assert len(subgroup_rows) > 0

    def test_risk_row_has_c_index(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "")
        rows = self._parse_csv(csv_text)
        risk_rows = [r for r in rows if r["task"] == "risk"]
        assert len(risk_rows) == 1
        assert risk_rows[0]["c_index"] == "0.7800"
        assert risk_rows[0]["brier_score"] == "0.1200"

    def test_classification_row_has_auc(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "")
        rows = self._parse_csv(csv_text)
        rhythm_rows = [r for r in rows if r["task"] == "rhythm"]
        assert len(rhythm_rows) == 3
        assert rhythm_rows[0]["auc"] == "0.9000"

    def test_sha256_in_csv(self) -> None:
        report = _make_benchmark_report()
        csv_text = _generate_csv(report, "0.3.0", "2026-05-07", "deadbeef")
        rows = self._parse_csv(csv_text)
        for row in rows:
            assert row["weights_sha256"] == "deadbeef"


# ---------------------------------------------------------------------------
# Integration: generate_performance_card()
# ---------------------------------------------------------------------------


class TestGeneratePerformanceCard:
    """Tests for the main generate_performance_card() function."""

    def test_creates_files(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "PERFORMANCE_CARD.md").exists()
        assert (tmp_path / "performance_card.csv").exists()

    def test_result_paths_populated(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        assert result.markdown_path.endswith("PERFORMANCE_CARD.md")
        assert result.csv_path.endswith("performance_card.csv")

    def test_result_content_matches_files(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        md_on_disk = (tmp_path / "PERFORMANCE_CARD.md").read_text(encoding="utf-8")
        csv_on_disk = (tmp_path / "performance_card.csv").read_text(encoding="utf-8")
        assert result.markdown_content == md_on_disk
        # Normalize line endings — csv module may produce \r\n on some platforms
        assert result.csv_content.replace('\r\n', '\n') == csv_on_disk.replace('\r\n', '\n')

    def test_version_in_result(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="2.1.0",
            output_dir=str(tmp_path),
        )
        assert result.model_version == "2.1.0"

    def test_timestamp_populated(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        assert result.timestamp != ""
        assert "T" in result.timestamp  # ISO format

    def test_custom_timestamp(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
            timestamp="2026-01-01T00:00:00Z",
        )
        assert result.timestamp == "2026-01-01T00:00:00Z"
        assert "2026-01-01T00:00:00Z" in result.markdown_content

    def test_sha256_with_model_weights(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"model weights content")
        expected_sha = hashlib.sha256(b"model weights content").hexdigest()

        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
            model_weights_path=str(weights),
        )
        assert result.model_weights_sha256 == expected_sha
        assert expected_sha in result.markdown_content

    def test_creates_nested_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(nested),
        )
        assert (nested / "PERFORMANCE_CARD.md").exists()

    def test_with_subgroups(self, tmp_path: Path) -> None:
        report = _make_benchmark_report(with_subgroups=True)
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        assert "sex_M" in result.markdown_content
        assert "sex_F" in result.markdown_content
        assert "age_30-39" in result.markdown_content

    def test_with_equity_gate(self, tmp_path: Path) -> None:
        from aortica.evaluation.equity_gate import EquityGateResult

        eq = EquityGateResult(passed=True, num_comparisons=5)
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
            equity_gate_result=eq,
        )
        assert "Equity Gate Results" in result.markdown_content

    def test_with_dataset_info(self, tmp_path: Path) -> None:
        report = _make_benchmark_report()
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
            dataset_info="PTB-XL test fold 10, 500 Hz",
        )
        assert "PTB-XL test fold 10, 500 Hz" in result.markdown_content

    def test_valid_markdown_tables(self, tmp_path: Path) -> None:
        """Verify that markdown tables have matching column counts."""
        report = _make_benchmark_report(with_subgroups=True)
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        lines = result.markdown_content.split("\n")
        in_table = False
        expected_cols: int | None = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cols = stripped.count("|") - 1  # border pipes don't count
                if not in_table:
                    in_table = True
                    expected_cols = cols
                else:
                    assert cols == expected_cols, (
                        f"Table column mismatch: expected {expected_cols}, "
                        f"got {cols} in line: {line}"
                    )
            else:
                in_table = False
                expected_cols = None

    def test_valid_csv(self, tmp_path: Path) -> None:
        """Verify that the CSV is parseable and has consistent row lengths."""
        report = _make_benchmark_report(with_subgroups=True)
        result = generate_performance_card(
            benchmark_report=report,
            model_version="0.3.0",
            output_dir=str(tmp_path),
        )
        reader = csv.reader(io.StringIO(result.csv_content))
        rows = list(reader)
        assert len(rows) > 1  # header + data
        header_len = len(rows[0])
        for i, row in enumerate(rows[1:], start=2):
            assert len(row) == header_len, (
                f"Row {i} has {len(row)} cols, expected {header_len}"
            )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestPerformanceCardCli:
    """Tests for the aortica performance-card CLI command."""

    def test_help(self) -> None:
        from click.testing import CliRunner

        from aortica.cli.performance_card_cmd import performance_card_cmd

        runner = CliRunner()
        result = runner.invoke(performance_card_cmd, ["--help"])
        assert result.exit_code == 0
        assert "performance-card" in result.output or "REPORT_PATH" in result.output

    def test_missing_version(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from aortica.cli.performance_card_cmd import performance_card_cmd

        # Create a dummy report file
        report = _make_benchmark_report()
        report_file = tmp_path / "report.json"
        report_file.write_text(
            json.dumps(report.as_dict(), default=str),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(performance_card_cmd, [str(report_file)])
        assert result.exit_code != 0  # --version is required

    def test_generate_from_json(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from aortica.cli.performance_card_cmd import performance_card_cmd

        report = _make_benchmark_report(with_subgroups=True)
        report_file = tmp_path / "report.json"
        report_file.write_text(
            json.dumps(report.as_dict(), default=str),
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(
            performance_card_cmd,
            [
                str(report_file),
                "--version", "0.3.0",
                "--output-dir", str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output_dir / "PERFORMANCE_CARD.md").exists()
        assert (output_dir / "performance_card.csv").exists()

    def test_invalid_report_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from aortica.cli.performance_card_cmd import performance_card_cmd

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            performance_card_cmd,
            [str(bad_file), "--version", "0.3.0"],
        )
        assert result.exit_code != 0

    def test_cli_registered_in_group(self) -> None:
        """Verify that performance-card command is registered in the main CLI."""
        try:
            from aortica.cli import _build_cli

            cli = _build_cli()
            cmd_names = [cmd for cmd in cli.commands]
            assert "performance-card" in cmd_names
        except (NameError, ImportError):
            # Pre-existing issue: train_multitask.py requires torch at
            # module level — skip gracefully in envs without torch
            pytest.skip("CLI group requires torch (train_multitask.py)")

    def test_json_roundtrip_reconstruction(self, tmp_path: Path) -> None:
        """Verify that loading a report from JSON reconstructs correctly."""
        from aortica.cli.performance_card_cmd import _load_benchmark_report

        report = _make_benchmark_report(with_subgroups=True)
        report_file = tmp_path / "report.json"
        report_file.write_text(
            json.dumps(report.as_dict(), default=str),
            encoding="utf-8",
        )

        loaded = _load_benchmark_report(str(report_file))
        assert loaded.n_samples == report.n_samples
        assert loaded.tasks_evaluated == report.tasks_evaluated
        assert len(loaded.overall) == len(report.overall)
        assert len(loaded.subgroups) == len(report.subgroups)

        # Verify per-class metrics survived round-trip
        for task in report.tasks_evaluated:
            orig = report.overall.get(task)
            recon = loaded.overall.get(task)
            if orig is not None:
                assert recon is not None
                assert abs(orig.macro_f1 - recon.macro_f1) < 1e-6
                assert len(orig.per_class) == len(recon.per_class)


# ---------------------------------------------------------------------------
# Package-level import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Test that performance card is importable from the package."""

    def test_import_from_evaluation(self) -> None:
        from aortica.evaluation import (
            PerformanceCardResult,
            generate_performance_card,
        )
        assert callable(generate_performance_card)
        assert PerformanceCardResult is not None

    def test_import_direct(self) -> None:
        from aortica.evaluation.performance_card import (
            PerformanceCardResult,
            generate_performance_card,
        )
        assert callable(generate_performance_card)


# ---------------------------------------------------------------------------
# Typecheck (marker test)
# ---------------------------------------------------------------------------


class TestTypecheck:
    """Marker test to verify the module is importable without errors."""

    def test_module_imports(self) -> None:
        import aortica.evaluation.performance_card
        import aortica.cli.performance_card_cmd
        assert True
