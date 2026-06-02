"""Tests for aortica.evaluation.regulatory_gate — US-097.

Covers:
- Loading targets from YAML
- regulatory_gate() with passing and failing scenarios
- Per-class and task-level metric checks
- Missing class handling (class not in benchmark → fail)
- Invalid YAML handling
- Default targets path resolution
- Summary output
- Module imports
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    TaskReport,
)
from aortica.evaluation.regulatory_gate import (
    ClassGateResult,
    RegulatoryGateResult,
    _find_class_metric,
    _load_targets,
    regulatory_gate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cm(name: str, *, auc: float = 0.95, sens: float = 0.92,
        spec: float = 0.96, f1: float = 0.90) -> ClassMetrics:
    """Create a ClassMetrics with customisable values."""
    return ClassMetrics(
        name=name, auc=auc, sensitivity=sens, specificity=spec, f1=f1
    )


def _make_passing_report() -> BenchmarkReport:
    """Benchmark report where all default targets pass."""
    return BenchmarkReport(
        overall={
            "rhythm": TaskReport(
                task_name="rhythm",
                macro_f1=0.92,
                ece=0.03,
                per_class=[
                    _cm("AF", auc=0.97),
                    _cm("VT", sens=0.90),
                    _cm("VF", sens=0.88),
                    _cm("LBBB", auc=0.95),
                    _cm("RBBB", auc=0.94),
                    _cm("WPW", auc=0.91),
                    _cm("av_block_3rd", sens=0.87),
                ],
            ),
            "structural": TaskReport(
                task_name="structural",
                macro_f1=0.85,
                ece=0.04,
                per_class=[
                    _cm("LVSD", auc=0.91),
                ],
            ),
            "ischaemia": TaskReport(
                task_name="ischaemia",
                macro_f1=0.84,
                ece=0.05,
                per_class=[
                    _cm("STEMI", auc=0.96, sens=0.93, spec=0.90),
                    _cm("hyperkalaemia", sens=0.82),
                ],
            ),
            "risk": TaskReport(
                task_name="risk",
                c_index=0.80,
                brier_score=0.10,
            ),
        },
        subgroups=[],
        n_samples=2000,
        tasks_evaluated=["rhythm", "structural", "ischaemia", "risk"],
    )


def _make_failing_report() -> BenchmarkReport:
    """Benchmark report where several targets fail."""
    return BenchmarkReport(
        overall={
            "rhythm": TaskReport(
                task_name="rhythm",
                macro_f1=0.85,  # < 0.90 target
                ece=0.06,
                per_class=[
                    _cm("AF", auc=0.93),  # < 0.95 target
                    _cm("VT", sens=0.80),  # < 0.85 target
                    _cm("VF", sens=0.88),
                    _cm("LBBB", auc=0.95),
                    _cm("RBBB", auc=0.94),
                    _cm("WPW", auc=0.91),
                    _cm("av_block_3rd", sens=0.87),
                ],
            ),
            "structural": TaskReport(
                task_name="structural",
                macro_f1=0.82,
                ece=0.05,
                per_class=[
                    _cm("LVSD", auc=0.86),  # < 0.88 target
                ],
            ),
            "ischaemia": TaskReport(
                task_name="ischaemia",
                macro_f1=0.80,
                ece=0.06,
                per_class=[
                    _cm("STEMI", auc=0.96, sens=0.88, spec=0.90),  # sens < 0.90
                    _cm("hyperkalaemia", sens=0.82),
                ],
            ),
            "risk": TaskReport(
                task_name="risk",
                c_index=0.75,
                brier_score=0.15,
            ),
        },
        subgroups=[],
        n_samples=2000,
        tasks_evaluated=["rhythm", "structural", "ischaemia", "risk"],
    )


@pytest.fixture
def passing_report() -> BenchmarkReport:
    return _make_passing_report()


@pytest.fixture
def failing_report() -> BenchmarkReport:
    return _make_failing_report()


@pytest.fixture
def targets_yaml(tmp_path: Path) -> str:
    """Write a simple targets YAML and return its path."""
    content = """\
sensitivity:
  STEMI: 0.90
  VT: 0.85

auc:
  AF: 0.95
  LVSD: 0.88

macro_f1:
  rhythm: 0.90
"""
    path = tmp_path / "targets.yaml"
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------


class TestLoadTargets:
    """Test YAML target file loading."""

    def test_load_valid_yaml(self, targets_yaml: str) -> None:
        targets = _load_targets(targets_yaml)
        assert "sensitivity" in targets
        assert "auc" in targets
        assert "macro_f1" in targets
        assert targets["sensitivity"]["STEMI"] == 0.90
        assert targets["auc"]["AF"] == 0.95

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_targets("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("just a string")
        with pytest.raises(ValueError, match="mapping"):
            _load_targets(str(path))

    def test_load_nested_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_nested.yaml"
        path.write_text("sensitivity: not_a_dict")
        with pytest.raises(ValueError, match="mapping"):
            _load_targets(str(path))

    def test_default_targets_exist(self) -> None:
        from aortica.evaluation.regulatory_gate import _DEFAULT_TARGETS_PATH

        assert os.path.isfile(_DEFAULT_TARGETS_PATH), (
            f"Default targets file not found at {_DEFAULT_TARGETS_PATH}"
        )


# ---------------------------------------------------------------------------
# Metric lookup
# ---------------------------------------------------------------------------


class TestFindClassMetric:
    """Test _find_class_metric helper."""

    def test_find_sensitivity(self, passing_report: BenchmarkReport) -> None:
        val = _find_class_metric(passing_report, "STEMI", "sensitivity")
        assert val is not None
        assert val == pytest.approx(0.93)

    def test_find_auc(self, passing_report: BenchmarkReport) -> None:
        val = _find_class_metric(passing_report, "AF", "auc")
        assert val is not None
        assert val == pytest.approx(0.97)

    def test_find_macro_f1(self, passing_report: BenchmarkReport) -> None:
        val = _find_class_metric(passing_report, "rhythm", "macro_f1")
        assert val is not None
        assert val == pytest.approx(0.92)

    def test_missing_class(self, passing_report: BenchmarkReport) -> None:
        val = _find_class_metric(passing_report, "NONEXISTENT", "auc")
        assert val is None

    def test_missing_task(self, passing_report: BenchmarkReport) -> None:
        val = _find_class_metric(
            passing_report, "nonexistent_task", "macro_f1"
        )
        assert val is None


# ---------------------------------------------------------------------------
# Regulatory gate — passing
# ---------------------------------------------------------------------------


class TestRegulatoryGatePassing:
    """Test regulatory_gate with metrics that meet all targets."""

    def test_overall_pass(
        self, passing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(passing_report, targets_yaml)
        assert isinstance(result, RegulatoryGateResult)
        assert result.passed is True
        assert result.num_failed == 0

    def test_all_checks_pass(
        self, passing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(passing_report, targets_yaml)
        assert all(r.passed for r in result.per_class)
        # 5 checks: STEMI sens, VT sens, AF auc, LVSD auc, rhythm macro_f1
        assert result.num_passed == 5

    def test_per_class_results(
        self, passing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(passing_report, targets_yaml)
        stemi_result = next(
            r for r in result.per_class
            if r.class_name == "STEMI" and r.metric_name == "sensitivity"
        )
        assert stemi_result.actual >= stemi_result.target

    def test_targets_path_recorded(
        self, passing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(passing_report, targets_yaml)
        assert result.targets_path == targets_yaml


# ---------------------------------------------------------------------------
# Regulatory gate — failing
# ---------------------------------------------------------------------------


class TestRegulatoryGateFailing:
    """Test regulatory_gate with metrics below targets."""

    def test_overall_fail(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        assert result.passed is False
        assert result.num_failed > 0

    def test_specific_failures(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        failed = {r.class_name for r in result.per_class if not r.passed}
        # STEMI sens=0.88 < 0.90, AF auc=0.93 < 0.95,
        # LVSD auc=0.86 < 0.88, VT sens=0.80 < 0.85,
        # rhythm macro_f1=0.85 < 0.90
        assert "STEMI" in failed
        assert "AF" in failed
        assert "LVSD" in failed
        assert "VT" in failed
        assert "rhythm" in failed

    def test_failure_count(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        assert result.num_failed == 5
        assert result.num_passed == 0

    def test_actual_vs_target_recorded(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        stemi_result = next(
            r for r in result.per_class
            if r.class_name == "STEMI" and r.metric_name == "sensitivity"
        )
        assert stemi_result.actual == pytest.approx(0.88)
        assert stemi_result.target == pytest.approx(0.90)
        assert stemi_result.passed is False


# ---------------------------------------------------------------------------
# Missing class handling
# ---------------------------------------------------------------------------


class TestMissingClassHandling:
    """Test behavior when benchmark report lacks required classes."""

    def test_missing_class_fails(self, tmp_path: Path) -> None:
        """A target for a class not in the benchmark should fail."""
        targets_path = tmp_path / "missing.yaml"
        targets_path.write_text("auc:\n  NONEXISTENT_CLASS: 0.90\n")

        report = BenchmarkReport(
            overall={},
            subgroups=[],
            n_samples=100,
            tasks_evaluated=[],
        )
        result = regulatory_gate(report, str(targets_path))
        assert result.passed is False
        assert result.num_failed == 1
        missing = result.per_class[0]
        assert missing.class_name == "NONEXISTENT_CLASS"
        assert missing.actual == 0.0

    def test_empty_report_all_fail(self, targets_yaml: str) -> None:
        """Empty benchmark report should fail all targets."""
        empty_report = BenchmarkReport(
            overall={},
            subgroups=[],
            n_samples=0,
            tasks_evaluated=[],
        )
        result = regulatory_gate(empty_report, targets_yaml)
        assert result.passed is False
        assert result.num_passed == 0


# ---------------------------------------------------------------------------
# Default targets
# ---------------------------------------------------------------------------


class TestDefaultTargets:
    """Test with the default regulatory_targets.yaml file."""

    def test_default_targets_used(self) -> None:
        """regulatory_gate() should use default targets when none specified."""
        report = _make_passing_report()
        result = regulatory_gate(report)
        assert isinstance(result, RegulatoryGateResult)
        # Should have checked multiple targets
        assert len(result.per_class) > 0

    def test_default_targets_stemi_sensitivity(self) -> None:
        """Default targets include STEMI sensitivity ≥ 0.90."""
        report = _make_passing_report()
        result = regulatory_gate(report)
        stemi_sens = [
            r for r in result.per_class
            if r.class_name == "STEMI" and r.metric_name == "sensitivity"
        ]
        assert len(stemi_sens) == 1
        assert stemi_sens[0].target == pytest.approx(0.90)

    def test_default_targets_af_auc(self) -> None:
        """Default targets include AF AUC ≥ 0.95."""
        report = _make_passing_report()
        result = regulatory_gate(report)
        af_auc = [
            r for r in result.per_class
            if r.class_name == "AF" and r.metric_name == "auc"
        ]
        assert len(af_auc) == 1
        assert af_auc[0].target == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    """Test the summary() method."""

    def test_passing_summary_contains_pass(
        self, passing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(passing_report, targets_yaml)
        summary = result.summary()
        assert "PASS" in summary
        assert "failed: 0" in summary

    def test_failing_summary_contains_fail(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        summary = result.summary()
        assert "FAIL" in summary
        assert "STEMI" in summary

    def test_summary_includes_actual_vs_target(
        self, failing_report: BenchmarkReport, targets_yaml: str
    ) -> None:
        result = regulatory_gate(failing_report, targets_yaml)
        summary = result.summary()
        assert "actual=" in summary
        assert "target=" in summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_threshold_passes(self, tmp_path: Path) -> None:
        """A metric exactly equal to the target should pass (>=)."""
        targets_path = tmp_path / "exact.yaml"
        targets_path.write_text("sensitivity:\n  STEMI: 0.93\n")

        report = _make_passing_report()  # STEMI sens = 0.93
        result = regulatory_gate(report, str(targets_path))
        assert result.passed is True

    def test_just_below_threshold_fails(self, tmp_path: Path) -> None:
        """A metric just below the target should fail."""
        targets_path = tmp_path / "below.yaml"
        targets_path.write_text("sensitivity:\n  STEMI: 0.94\n")

        report = _make_passing_report()  # STEMI sens = 0.93
        result = regulatory_gate(report, str(targets_path))
        assert result.passed is False

    def test_specificity_target(self, tmp_path: Path) -> None:
        """Specificity targets should work."""
        targets_path = tmp_path / "spec.yaml"
        targets_path.write_text("specificity:\n  STEMI: 0.85\n")

        report = _make_passing_report()  # STEMI spec = 0.90
        result = regulatory_gate(report, str(targets_path))
        assert result.passed is True

    def test_empty_targets(self, tmp_path: Path) -> None:
        """Empty targets YAML should pass (no checks to fail)."""
        targets_path = tmp_path / "empty.yaml"
        targets_path.write_text("{}\n")

        report = _make_passing_report()
        result = regulatory_gate(report, str(targets_path))
        assert result.passed is True
        assert result.num_passed == 0
        assert result.num_failed == 0


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify evaluation module exports are accessible."""

    def test_import_regulatory_gate(self) -> None:
        from aortica.evaluation import regulatory_gate  # noqa: F811

    def test_import_regulatory_gate_result(self) -> None:
        from aortica.evaluation import RegulatoryGateResult  # noqa: F811

    def test_import_class_gate_result(self) -> None:
        from aortica.evaluation import ClassGateResult  # noqa: F811

    def test_import_from_submodule(self) -> None:
        from aortica.evaluation.regulatory_gate import (  # noqa: F811
            ClassGateResult,
            RegulatoryGateResult,
            regulatory_gate,
        )
