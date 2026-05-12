"""Tests for the multi-task evaluation harness (US-028 + US-079).

Updated for expanded task heads: rhythm=28, structural=19, ischaemia=19, risk=6.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from aortica.evaluation.benchmark import (  # noqa: E402
    CLASSIFICATION_TASKS,
    TASK_NUM_OUTPUTS,
    BenchmarkReport,
    ClassMetrics,
    SubgroupReport,
    TaskReport,
    _build_subgroup_masks,
    _CLASS_NAMES,
    _compute_auc,
    _compute_brier_score,
    _compute_c_index,
    _compute_ece,
    _compute_f1,
    _compute_sensitivity_specificity,
    _evaluate_classification_task,
    _evaluate_risk_task,
    _split_labels,
    benchmark,
)


# Total outputs: 28 + 19 + 19 + 6 = 72
TOTAL_OUTPUTS = sum(TASK_NUM_OUTPUTS.values())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aortica_model() -> torch.nn.Module:
    """Create a small AorticaModel for testing."""
    from aortica.models.aortica_model import AorticaModel

    return AorticaModel(
        in_channels=12,
        feature_dim=252,
        num_leads=12,
        enabled_tasks=["rhythm", "structural", "ischaemia", "risk"],
    )


@pytest.fixture()
def synthetic_dataset() -> torch.utils.data.Dataset:  # type: ignore[type-arg]
    """Create a small synthetic dataset with expanded labels (72 cols)."""
    n = 32
    signals = torch.randn(n, 12, 500)
    labels = torch.zeros(n, TOTAL_OUTPUTS)
    rng = np.random.RandomState(42)
    r, s, i, k = 28, 19, 19, 6
    labels[:, :r] = torch.tensor(
        rng.binomial(1, 0.2, size=(n, r)), dtype=torch.float32
    )
    labels[:, r:r+s] = torch.tensor(
        rng.binomial(1, 0.15, size=(n, s)), dtype=torch.float32
    )
    labels[:, r+s:r+s+i] = torch.tensor(
        rng.binomial(1, 0.1, size=(n, i)), dtype=torch.float32
    )
    labels[:, r+s+i:] = torch.tensor(
        rng.uniform(0, 1, size=(n, k)), dtype=torch.float32
    )
    return torch.utils.data.TensorDataset(signals, labels)


@pytest.fixture()
def sample_metadata() -> list[dict[str, object] | None]:
    """Metadata for 32 samples with age and sex."""
    rng = np.random.RandomState(42)
    return [
        {"age": int(rng.randint(20, 85)), "sex": rng.choice(["M", "F"])}
        for _ in range(32)
    ]


# ===================================================================
# Task output size tests (US-079)
# ===================================================================


class TestExpandedDimensions:
    """Verify TASK_NUM_OUTPUTS matches expanded heads."""

    def test_rhythm_28(self) -> None:
        assert TASK_NUM_OUTPUTS["rhythm"] == 28

    def test_structural_19(self) -> None:
        assert TASK_NUM_OUTPUTS["structural"] == 19

    def test_ischaemia_19(self) -> None:
        assert TASK_NUM_OUTPUTS["ischaemia"] == 19

    def test_risk_6(self) -> None:
        assert TASK_NUM_OUTPUTS["risk"] == 6

    def test_total_72(self) -> None:
        assert TOTAL_OUTPUTS == 72

    def test_class_names_match_output_sizes(self) -> None:
        for task, names in _CLASS_NAMES.items():
            assert len(names) == TASK_NUM_OUTPUTS[task], (
                f"{task}: {len(names)} names != {TASK_NUM_OUTPUTS[task]} outputs"
            )

    def test_rhythm_includes_rare_arrhythmias(self) -> None:
        rare = {"brugada_pattern", "short_QT_syndrome", "CPVT",
                "fascicular_VT", "atypical_atrial_flutter",
                "inappropriate_sinus_tachy"}
        assert rare.issubset(set(_CLASS_NAMES["rhythm"]))

    def test_structural_includes_strain_patterns(self) -> None:
        strain = {"LV_strain_grade", "RV_strain_PE", "Takotsubo_pattern",
                  "infiltrative_cardiomyopathy_strain"}
        assert strain.issubset(set(_CLASS_NAMES["structural"]))

    def test_ischaemia_includes_stemi_mimics(self) -> None:
        mimics = {"early_repol_vs_STEMI", "de_Winter_T_wave",
                  "Wellens_syndrome", "aVR_ST_elevation", "Sgarbossa_criteria"}
        assert mimics.issubset(set(_CLASS_NAMES["ischaemia"]))

    def test_ischaemia_includes_metabolic_drug(self) -> None:
        metabolic = {"hyperkalaemia_severity_grade", "hypothermia_osborn_waves",
                     "TCA_toxicity", "digoxin_effect_vs_toxicity"}
        assert metabolic.issubset(set(_CLASS_NAMES["ischaemia"]))

    def test_risk_includes_refined_outputs(self) -> None:
        refined = {"ecg_predicted_ef", "conduction_disease_trajectory",
                   "sudden_cardiac_death_risk"}
        assert refined.issubset(set(_CLASS_NAMES["risk"]))


# ===================================================================
# Dataclass tests
# ===================================================================


class TestClassMetrics:
    def test_defaults(self) -> None:
        cm = ClassMetrics()
        assert cm.name == ""
        assert cm.auc == 0.0

    def test_construction(self) -> None:
        cm = ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88)
        assert cm.name == "AF"
        assert cm.auc == 0.95


class TestTaskReport:
    def test_defaults(self) -> None:
        tr = TaskReport()
        assert tr.task_name == ""
        assert tr.per_class == []

    def test_classification_report(self) -> None:
        tr = TaskReport(
            task_name="rhythm", macro_f1=0.85, ece=0.05,
            per_class=[ClassMetrics(name="AF", auc=0.92)],
        )
        assert len(tr.per_class) == 1


class TestSubgroupReport:
    def test_defaults(self) -> None:
        sg = SubgroupReport()
        assert sg.n_samples == 0


class TestBenchmarkReport:
    def test_defaults(self) -> None:
        br = BenchmarkReport()
        assert br.overall == {}
        assert br.equity_gate_result is None

    def test_as_dict(self) -> None:
        report = BenchmarkReport(
            n_samples=100, tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.88)},
        )
        d = report.as_dict()
        assert d["n_samples"] == 100

    def test_summary_table_returns_string(self) -> None:
        report = BenchmarkReport(
            n_samples=100, tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(
                task_name="rhythm", macro_f1=0.88, ece=0.03,
                per_class=[ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88)],
            )},
        )
        table = report.summary_table()
        assert "Macro-F1" in table
        assert "AF" in table

    def test_summary_table_risk(self) -> None:
        report = BenchmarkReport(
            n_samples=50, tasks_evaluated=["risk"],
            overall={"risk": TaskReport(task_name="risk", c_index=0.78, brier_score=0.12)},
        )
        assert "C-index" in report.summary_table()

    def test_summary_table_subgroups(self) -> None:
        report = BenchmarkReport(
            n_samples=50, tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.88, ece=0.03)},
            subgroups=[SubgroupReport(
                subgroup_name="sex_M", n_samples=25,
                task_reports={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.85)},
            )],
        )
        assert "sex_M" in report.summary_table()

    def test_to_csv_classification(self) -> None:
        report = BenchmarkReport(
            n_samples=100, tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(
                task_name="rhythm", macro_f1=0.88, ece=0.03,
                per_class=[ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88)],
            )},
        )
        csv_str = report.to_csv()
        assert "AF" in csv_str

    def test_to_csv_risk(self) -> None:
        report = BenchmarkReport(
            n_samples=50, tasks_evaluated=["risk"],
            overall={"risk": TaskReport(task_name="risk", c_index=0.78, brier_score=0.12)},
        )
        assert "0.7800" in report.to_csv()

    def test_equity_gate_result_field(self) -> None:
        """BenchmarkReport can carry equity gate results."""
        report = BenchmarkReport(
            n_samples=100, tasks_evaluated=["rhythm"],
            equity_gate_result={"passed": True},
        )
        assert report.equity_gate_result is not None
        assert report.equity_gate_result["passed"] is True


# ===================================================================
# Metric helper tests
# ===================================================================


class TestComputeAUC:
    def test_perfect(self) -> None:
        preds = np.array([0.9, 0.8, 0.1, 0.05])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        assert _compute_auc(preds, tgts) == pytest.approx(1.0)

    def test_worst(self) -> None:
        preds = np.array([0.1, 0.2, 0.9, 0.8])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        assert _compute_auc(preds, tgts) == pytest.approx(0.0)

    def test_random_around_half(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, size=200)
        tgts = rng.binomial(1, 0.5, size=200).astype(np.float64)
        assert 0.3 <= _compute_auc(preds, tgts) <= 0.7

    def test_single_class(self) -> None:
        assert _compute_auc(np.array([0.5, 0.6]), np.array([1.0, 1.0])) == 0.5


class TestSensitivitySpecificity:
    def test_perfect(self) -> None:
        preds = np.array([0.9, 0.8, 0.1, 0.05])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        sens, spec = _compute_sensitivity_specificity(preds, tgts)
        assert sens == 1.0
        assert spec == 1.0


class TestComputeF1:
    def test_perfect(self) -> None:
        preds = np.array([[0.9, 0.1], [0.1, 0.9]])
        tgts = np.array([[1.0, 0.0], [0.0, 1.0]])
        macro_f1, per_class = _compute_f1(preds, tgts)
        assert macro_f1 == pytest.approx(1.0)

    def test_expanded_classes(self) -> None:
        """F1 computation works with 28 classes (expanded rhythm)."""
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 28))
        tgts = rng.binomial(1, 0.2, (50, 28)).astype(np.float64)
        macro_f1, per_class = _compute_f1(preds, tgts)
        assert 0.0 <= macro_f1 <= 1.0
        assert len(per_class) == 28


class TestComputeECE:
    def test_perfect_calibration(self) -> None:
        preds = np.full(100, 0.5)
        tgts = np.array([1.0] * 50 + [0.0] * 50)
        assert _compute_ece(preds, tgts) == pytest.approx(0.0, abs=0.01)

    def test_empty(self) -> None:
        assert _compute_ece(np.array([]), np.array([])) == 0.0


class TestComputeCIndex:
    def test_perfect(self) -> None:
        preds = np.array([[0.9], [0.7], [0.5], [0.3]])
        tgts = np.array([[0.9], [0.7], [0.5], [0.3]])
        assert _compute_c_index(preds, tgts) == pytest.approx(1.0)

    def test_expanded_risk(self) -> None:
        """C-index works with 6 risk outputs."""
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (20, 6))
        tgts = rng.uniform(0, 1, (20, 6))
        c_idx = _compute_c_index(preds, tgts)
        assert 0.0 <= c_idx <= 1.0


class TestBrierScore:
    def test_perfect(self) -> None:
        preds = np.array([[0.0, 1.0], [1.0, 0.0]])
        tgts = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert _compute_brier_score(preds, tgts) == pytest.approx(0.0)


# ===================================================================
# Task-level evaluation tests
# ===================================================================


class TestEvaluateClassificationTask:
    def test_returns_task_report(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 5))
        tgts = rng.binomial(1, 0.3, (50, 5)).astype(np.float64)
        tr = _evaluate_classification_task(preds, tgts, "rhythm")
        assert isinstance(tr, TaskReport)
        assert len(tr.per_class) == 5

    def test_expanded_rhythm_28(self) -> None:
        """Evaluate classification with 28-class rhythm head."""
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 28))
        tgts = rng.binomial(1, 0.2, (50, 28)).astype(np.float64)
        names = _CLASS_NAMES["rhythm"]
        tr = _evaluate_classification_task(preds, tgts, "rhythm", names)
        assert len(tr.per_class) == 28
        assert tr.per_class[0].name == "AF"
        assert tr.per_class[22].name == "brugada_pattern"

    def test_expanded_ischaemia_19(self) -> None:
        """Evaluate classification with 19-class ischaemia head."""
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 19))
        tgts = rng.binomial(1, 0.15, (50, 19)).astype(np.float64)
        names = _CLASS_NAMES["ischaemia"]
        tr = _evaluate_classification_task(preds, tgts, "ischaemia", names)
        assert len(tr.per_class) == 19
        assert tr.per_class[10].name == "early_repol_vs_STEMI"

    def test_expanded_structural_19(self) -> None:
        """Evaluate classification with 19-class structural head."""
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 19))
        tgts = rng.binomial(1, 0.15, (50, 19)).astype(np.float64)
        names = _CLASS_NAMES["structural"]
        tr = _evaluate_classification_task(preds, tgts, "structural", names)
        assert len(tr.per_class) == 19
        assert tr.per_class[15].name == "LV_strain_grade"


class TestEvaluateRiskTask:
    def test_returns_task_report(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 6))
        tgts = rng.uniform(0, 1, (50, 6))
        tr = _evaluate_risk_task(preds, tgts)
        assert tr.task_name == "risk"
        assert 0.0 <= tr.c_index <= 1.0

    def test_perfect_predictions(self) -> None:
        tgts = np.array([[0.1, 0.5, 0.9, 0.3, 0.7, 0.2],
                         [0.2, 0.6, 0.8, 0.4, 0.6, 0.3]])
        tr = _evaluate_risk_task(tgts, tgts)
        assert tr.c_index == pytest.approx(1.0)
        assert tr.brier_score == pytest.approx(0.0)


# ===================================================================
# Label splitting tests (expanded)
# ===================================================================


class TestSplitLabels:
    def test_all_tasks_expanded(self) -> None:
        labels = np.random.rand(10, 72)
        result = _split_labels(labels, ["rhythm", "structural", "ischaemia", "risk"])
        assert result["rhythm"].shape == (10, 28)
        assert result["structural"].shape == (10, 19)
        assert result["ischaemia"].shape == (10, 19)
        assert result["risk"].shape == (10, 6)

    def test_subset(self) -> None:
        labels = np.random.rand(10, 34)  # rhythm(28) + risk(6)
        result = _split_labels(labels, ["rhythm", "risk"])
        assert result["rhythm"].shape == (10, 28)
        assert result["risk"].shape == (10, 6)

    def test_split_content_correct(self) -> None:
        """Verify label values are correctly partitioned."""
        rng = np.random.RandomState(42)
        labels = rng.rand(5, 72)
        result = _split_labels(labels, ["rhythm", "structural", "ischaemia", "risk"])
        np.testing.assert_array_equal(result["rhythm"], labels[:, :28])
        np.testing.assert_array_equal(result["structural"], labels[:, 28:47])
        np.testing.assert_array_equal(result["ischaemia"], labels[:, 47:66])
        np.testing.assert_array_equal(result["risk"], labels[:, 66:72])


# ===================================================================
# Subgroup tests
# ===================================================================


class TestBuildSubgroupMasks:
    def test_age_and_sex(self) -> None:
        metadata: list[dict[str, object] | None] = [
            {"age": 25, "sex": "M"}, {"age": 55, "sex": "F"},
            {"age": 52, "sex": "M"}, None,
        ]
        masks = _build_subgroup_masks(metadata, 4)
        assert masks["age_20-29"].sum() == 1
        assert masks["age_50-59"].sum() == 2
        assert masks["sex_M"].sum() == 2

    def test_empty_metadata(self) -> None:
        assert _build_subgroup_masks([None, None], 2) == {}


# ===================================================================
# Full benchmark tests (expanded heads)
# ===================================================================


class TestBenchmark:
    def test_returns_benchmark_report(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16, seed=42)
        assert isinstance(report, BenchmarkReport)
        assert report.n_samples == 32

    def test_tasks_evaluated(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        assert set(report.tasks_evaluated) == {"rhythm", "structural", "ischaemia", "risk"}

    def test_classification_per_class_expanded(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        assert len(report.overall["rhythm"].per_class) == 28
        assert report.overall["rhythm"].per_class[0].name == "AF"
        assert len(report.overall["structural"].per_class) == 19
        assert len(report.overall["ischaemia"].per_class) == 19

    def test_risk_metrics_expanded(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        risk_report = report.overall["risk"]
        assert risk_report.c_index >= 0.0
        assert risk_report.brier_score >= 0.0

    def test_metric_ranges(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        for task in CLASSIFICATION_TASKS:
            if task in report.overall:
                tr = report.overall[task]
                assert 0.0 <= tr.macro_f1 <= 1.0
                assert 0.0 <= tr.ece <= 1.0

    def test_task_subset(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, tasks=["rhythm"], batch_size=16)
        assert report.tasks_evaluated == ["rhythm"]
        assert "risk" not in report.overall

    def test_with_metadata(
        self, aortica_model: torch.nn.Module,
        synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
        sample_metadata: list[dict[str, object] | None],
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, metadata=sample_metadata, batch_size=16)
        assert len(report.subgroups) > 0
        sg_names = [sg.subgroup_name for sg in report.subgroups]
        assert any("sex_" in name for name in sg_names)

    def test_reproducible(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        r1 = benchmark(aortica_model, synthetic_dataset, batch_size=16, seed=42)
        r2 = benchmark(aortica_model, synthetic_dataset, batch_size=16, seed=42)
        assert r1.overall["rhythm"].macro_f1 == r2.overall["rhythm"].macro_f1

    def test_csv_export(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        # header + 28 rhythm + 19 structural + 19 ischaemia + 1 risk = 68 lines
        assert len(lines) > 1

    def test_summary_table_includes_new_classes(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        table = report.summary_table()
        assert "RHYTHM" in table
        assert "ISCHAEMIA" in table
        assert "STRUCTURAL" in table


# ===================================================================
# Imports test
# ===================================================================


class TestImports:
    def test_evaluation_package_exports(self) -> None:
        from aortica.evaluation import (
            BenchmarkReport, ClassMetrics, SubgroupReport, TaskReport, benchmark,
        )
        assert all(x is not None for x in [BenchmarkReport, ClassMetrics, SubgroupReport, TaskReport, benchmark])
