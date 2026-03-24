"""Tests for the multi-task evaluation harness (US-028)."""

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
    """Create a small synthetic dataset with known labels."""
    n = 32
    signals = torch.randn(n, 12, 500)
    # Labels: rhythm(22) + structural(15) + ischaemia(10) + risk(3) = 50
    labels = torch.zeros(n, 50)
    # Sprinkle some positive labels for classification tasks
    rng = np.random.RandomState(42)
    labels[:, :22] = torch.tensor(
        rng.binomial(1, 0.2, size=(n, 22)), dtype=torch.float32
    )
    labels[:, 22:37] = torch.tensor(
        rng.binomial(1, 0.15, size=(n, 15)), dtype=torch.float32
    )
    labels[:, 37:47] = torch.tensor(
        rng.binomial(1, 0.1, size=(n, 10)), dtype=torch.float32
    )
    # Risk labels: continuous [0, 1]
    labels[:, 47:50] = torch.tensor(
        rng.uniform(0, 1, size=(n, 3)), dtype=torch.float32
    )

    return torch.utils.data.TensorDataset(signals, labels)


@pytest.fixture()
def sample_metadata() -> list[dict[str, object] | None]:
    """Metadata for 32 samples with age and sex."""
    rng = np.random.RandomState(42)
    meta: list[dict[str, object] | None] = []
    for _ in range(32):
        meta.append({
            "age": int(rng.randint(20, 85)),
            "sex": rng.choice(["M", "F"]),
        })
    return meta


# ===================================================================
# Dataclass tests
# ===================================================================


class TestClassMetrics:
    """Tests for ClassMetrics dataclass."""

    def test_defaults(self) -> None:
        cm = ClassMetrics()
        assert cm.name == ""
        assert cm.auc == 0.0
        assert cm.sensitivity == 0.0
        assert cm.specificity == 0.0
        assert cm.f1 == 0.0

    def test_construction(self) -> None:
        cm = ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88)
        assert cm.name == "AF"
        assert cm.auc == 0.95


class TestTaskReport:
    """Tests for TaskReport dataclass."""

    def test_defaults(self) -> None:
        tr = TaskReport()
        assert tr.task_name == ""
        assert tr.macro_f1 == 0.0
        assert tr.ece == 0.0
        assert tr.c_index == 0.0
        assert tr.brier_score == 0.0
        assert tr.per_class == []

    def test_classification_report(self) -> None:
        tr = TaskReport(
            task_name="rhythm",
            macro_f1=0.85,
            ece=0.05,
            per_class=[ClassMetrics(name="AF", auc=0.92)],
        )
        assert tr.task_name == "rhythm"
        assert len(tr.per_class) == 1
        assert tr.per_class[0].name == "AF"


class TestSubgroupReport:
    """Tests for SubgroupReport dataclass."""

    def test_defaults(self) -> None:
        sg = SubgroupReport()
        assert sg.subgroup_name == ""
        assert sg.n_samples == 0
        assert sg.task_reports == {}


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""

    def test_defaults(self) -> None:
        br = BenchmarkReport()
        assert br.overall == {}
        assert br.subgroups == []
        assert br.n_samples == 0

    def test_as_dict(self) -> None:
        report = BenchmarkReport(
            n_samples=100,
            tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.88)},
        )
        d = report.as_dict()
        assert isinstance(d, dict)
        assert d["n_samples"] == 100
        assert "rhythm" in d["overall"]

    def test_summary_table_returns_string(self) -> None:
        report = BenchmarkReport(
            n_samples=100,
            tasks_evaluated=["rhythm"],
            overall={
                "rhythm": TaskReport(
                    task_name="rhythm",
                    macro_f1=0.88,
                    ece=0.03,
                    per_class=[ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88)],
                ),
            },
        )
        table = report.summary_table()
        assert isinstance(table, str)
        assert "Macro-F1" in table
        assert "AF" in table
        assert "100" in table

    def test_summary_table_risk(self) -> None:
        report = BenchmarkReport(
            n_samples=50,
            tasks_evaluated=["risk"],
            overall={"risk": TaskReport(task_name="risk", c_index=0.78, brier_score=0.12)},
        )
        table = report.summary_table()
        assert "C-index" in table
        assert "Brier" in table

    def test_summary_table_subgroups(self) -> None:
        report = BenchmarkReport(
            n_samples=50,
            tasks_evaluated=["rhythm"],
            overall={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.88, ece=0.03)},
            subgroups=[
                SubgroupReport(
                    subgroup_name="sex_M",
                    n_samples=25,
                    task_reports={"rhythm": TaskReport(task_name="rhythm", macro_f1=0.85)},
                ),
            ],
        )
        table = report.summary_table()
        assert "SUBGROUPS" in table
        assert "sex_M" in table

    def test_to_csv_classification(self) -> None:
        report = BenchmarkReport(
            n_samples=100,
            tasks_evaluated=["rhythm"],
            overall={
                "rhythm": TaskReport(
                    task_name="rhythm",
                    macro_f1=0.88,
                    ece=0.03,
                    per_class=[
                        ClassMetrics(name="AF", auc=0.95, sensitivity=0.9, specificity=0.85, f1=0.88),
                    ],
                ),
            },
        )
        csv_str = report.to_csv()
        assert "task,class,auc" in csv_str
        assert "AF" in csv_str
        assert "rhythm" in csv_str

    def test_to_csv_risk(self) -> None:
        report = BenchmarkReport(
            n_samples=50,
            tasks_evaluated=["risk"],
            overall={"risk": TaskReport(task_name="risk", c_index=0.78, brier_score=0.12)},
        )
        csv_str = report.to_csv()
        assert "risk" in csv_str
        assert "0.7800" in csv_str


# ===================================================================
# Metric helper tests
# ===================================================================


class TestComputeAUC:
    """Tests for _compute_auc."""

    def test_perfect(self) -> None:
        preds = np.array([0.9, 0.8, 0.1, 0.05])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        auc = _compute_auc(preds, tgts)
        assert auc == pytest.approx(1.0)

    def test_worst(self) -> None:
        preds = np.array([0.1, 0.2, 0.9, 0.8])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        auc = _compute_auc(preds, tgts)
        assert auc == pytest.approx(0.0)

    def test_random_around_half(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, size=200)
        tgts = rng.binomial(1, 0.5, size=200).astype(np.float64)
        auc = _compute_auc(preds, tgts)
        assert 0.3 <= auc <= 0.7

    def test_single_class(self) -> None:
        preds = np.array([0.5, 0.6, 0.7])
        tgts = np.array([1.0, 1.0, 1.0])
        assert _compute_auc(preds, tgts) == 0.5

    def test_range(self) -> None:
        rng = np.random.RandomState(123)
        preds = rng.uniform(0, 1, 100)
        tgts = rng.binomial(1, 0.5, 100).astype(np.float64)
        auc = _compute_auc(preds, tgts)
        assert 0.0 <= auc <= 1.0


class TestSensitivitySpecificity:
    """Tests for _compute_sensitivity_specificity."""

    def test_perfect(self) -> None:
        preds = np.array([0.9, 0.8, 0.1, 0.05])
        tgts = np.array([1.0, 1.0, 0.0, 0.0])
        sens, spec = _compute_sensitivity_specificity(preds, tgts)
        assert sens == 1.0
        assert spec == 1.0

    def test_no_positives(self) -> None:
        preds = np.array([0.1, 0.2])
        tgts = np.array([0.0, 0.0])
        sens, spec = _compute_sensitivity_specificity(preds, tgts)
        assert sens == 0.0  # no positives → sens = 0
        assert spec == 1.0

    def test_all_positive_preds(self) -> None:
        preds = np.array([0.9, 0.8])
        tgts = np.array([1.0, 0.0])
        sens, spec = _compute_sensitivity_specificity(preds, tgts)
        assert sens == 1.0
        assert spec == 0.0


class TestComputeF1:
    """Tests for _compute_f1."""

    def test_perfect(self) -> None:
        preds = np.array([[0.9, 0.1], [0.1, 0.9]])
        tgts = np.array([[1.0, 0.0], [0.0, 1.0]])
        macro_f1, per_class = _compute_f1(preds, tgts)
        assert macro_f1 == pytest.approx(1.0)
        assert len(per_class) == 2

    def test_worst(self) -> None:
        preds = np.array([[0.9, 0.1], [0.9, 0.1]])
        tgts = np.array([[0.0, 1.0], [0.0, 1.0]])
        macro_f1, _ = _compute_f1(preds, tgts)
        assert macro_f1 == 0.0

    def test_range(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 5))
        tgts = rng.binomial(1, 0.3, (50, 5)).astype(np.float64)
        macro_f1, per_class = _compute_f1(preds, tgts)
        assert 0.0 <= macro_f1 <= 1.0
        assert len(per_class) == 5


class TestComputeECE:
    """Tests for _compute_ece."""

    def test_perfect_calibration(self) -> None:
        # All predictions 0.5, half positive → ECE ≈ 0
        preds = np.full(100, 0.5)
        tgts = np.array([1.0] * 50 + [0.0] * 50)
        ece = _compute_ece(preds, tgts)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration(self) -> None:
        # All predictions 1.0, all targets 0 → ECE ≈ 1
        preds = np.ones(100)
        tgts = np.zeros(100)
        ece = _compute_ece(preds, tgts)
        assert ece == pytest.approx(1.0, abs=0.01)

    def test_range(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, 200)
        tgts = rng.binomial(1, 0.5, 200).astype(np.float64)
        ece = _compute_ece(preds, tgts)
        assert 0.0 <= ece <= 1.0

    def test_empty(self) -> None:
        preds = np.array([])
        tgts = np.array([])
        assert _compute_ece(preds, tgts) == 0.0


class TestComputeCIndex:
    """Tests for _compute_c_index."""

    def test_perfect(self) -> None:
        preds = np.array([[0.9], [0.7], [0.5], [0.3]])
        tgts = np.array([[0.9], [0.7], [0.5], [0.3]])
        c_idx = _compute_c_index(preds, tgts)
        assert c_idx == pytest.approx(1.0)

    def test_reversed(self) -> None:
        preds = np.array([[0.3], [0.5], [0.7], [0.9]])
        tgts = np.array([[0.9], [0.7], [0.5], [0.3]])
        c_idx = _compute_c_index(preds, tgts)
        assert c_idx == pytest.approx(0.0)

    def test_single_sample(self) -> None:
        preds = np.array([[0.5]])
        tgts = np.array([[0.5]])
        assert _compute_c_index(preds, tgts) == 0.5

    def test_multi_output(self) -> None:
        preds = np.array([[0.9, 0.1], [0.1, 0.9]])
        tgts = np.array([[0.9, 0.1], [0.1, 0.9]])
        c_idx = _compute_c_index(preds, tgts)
        assert c_idx == pytest.approx(1.0)


class TestBrierScore:
    """Tests for _compute_brier_score."""

    def test_perfect(self) -> None:
        preds = np.array([[0.0, 1.0], [1.0, 0.0]])
        tgts = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert _compute_brier_score(preds, tgts) == pytest.approx(0.0)

    def test_worst(self) -> None:
        preds = np.array([[1.0, 0.0]])
        tgts = np.array([[0.0, 1.0]])
        assert _compute_brier_score(preds, tgts) == pytest.approx(1.0)


# ===================================================================
# Task-level evaluation tests
# ===================================================================


class TestEvaluateClassificationTask:
    """Tests for _evaluate_classification_task."""

    def test_returns_task_report(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 5))
        tgts = rng.binomial(1, 0.3, (50, 5)).astype(np.float64)
        tr = _evaluate_classification_task(preds, tgts, "rhythm")
        assert isinstance(tr, TaskReport)
        assert tr.task_name == "rhythm"
        assert 0.0 <= tr.macro_f1 <= 1.0
        assert 0.0 <= tr.ece <= 1.0
        assert len(tr.per_class) == 5

    def test_per_class_names(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (30, 3))
        tgts = rng.binomial(1, 0.5, (30, 3)).astype(np.float64)
        names = ["A", "B", "C"]
        tr = _evaluate_classification_task(preds, tgts, "test", names)
        assert [cm.name for cm in tr.per_class] == ["A", "B", "C"]

    def test_per_class_auc_range(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 5))
        tgts = rng.binomial(1, 0.3, (50, 5)).astype(np.float64)
        tr = _evaluate_classification_task(preds, tgts, "rhythm")
        for cm in tr.per_class:
            assert 0.0 <= cm.auc <= 1.0
            assert 0.0 <= cm.sensitivity <= 1.0
            assert 0.0 <= cm.specificity <= 1.0
            assert 0.0 <= cm.f1 <= 1.0


class TestEvaluateRiskTask:
    """Tests for _evaluate_risk_task."""

    def test_returns_task_report(self) -> None:
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, (50, 3))
        tgts = rng.uniform(0, 1, (50, 3))
        tr = _evaluate_risk_task(preds, tgts)
        assert isinstance(tr, TaskReport)
        assert tr.task_name == "risk"
        assert 0.0 <= tr.c_index <= 1.0
        assert tr.brier_score >= 0.0

    def test_perfect_predictions(self) -> None:
        tgts = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]])
        tr = _evaluate_risk_task(tgts, tgts)
        assert tr.c_index == pytest.approx(1.0)
        assert tr.brier_score == pytest.approx(0.0)


# ===================================================================
# Label splitting tests
# ===================================================================


class TestSplitLabels:
    """Tests for _split_labels."""

    def test_all_tasks(self) -> None:
        labels = np.random.rand(10, 50)
        result = _split_labels(labels, ["rhythm", "structural", "ischaemia", "risk"])
        assert result["rhythm"].shape == (10, 22)
        assert result["structural"].shape == (10, 15)
        assert result["ischaemia"].shape == (10, 10)
        assert result["risk"].shape == (10, 3)

    def test_subset(self) -> None:
        labels = np.random.rand(10, 25)  # rhythm(22) + risk(3)
        result = _split_labels(labels, ["rhythm", "risk"])
        assert result["rhythm"].shape == (10, 22)
        assert result["risk"].shape == (10, 3)


# ===================================================================
# Subgroup tests
# ===================================================================


class TestBuildSubgroupMasks:
    """Tests for _build_subgroup_masks."""

    def test_age_and_sex(self) -> None:
        metadata: list[dict[str, object] | None] = [
            {"age": 25, "sex": "M"},
            {"age": 55, "sex": "F"},
            {"age": 52, "sex": "M"},
            None,
        ]
        masks = _build_subgroup_masks(metadata, 4)
        assert "age_20-29" in masks
        assert "age_50-59" in masks
        assert "sex_M" in masks
        assert "sex_F" in masks
        assert masks["age_20-29"].sum() == 1
        assert masks["age_50-59"].sum() == 2
        assert masks["sex_M"].sum() == 2
        assert masks["sex_F"].sum() == 1

    def test_empty_metadata(self) -> None:
        metadata: list[None] = [None, None, None]
        masks = _build_subgroup_masks(metadata, 3)
        assert masks == {}

    def test_age_90_plus(self) -> None:
        metadata: list[dict[str, object] | None] = [
            {"age": 92},
            {"age": 95},
        ]
        masks = _build_subgroup_masks(metadata, 2)
        assert "age_90+" in masks
        assert masks["age_90+"].sum() == 2


# ===================================================================
# Full benchmark tests
# ===================================================================


class TestBenchmark:
    """Tests for the benchmark() function."""

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

    def test_overall_keys(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        assert "rhythm" in report.overall
        assert "risk" in report.overall

    def test_classification_per_class(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        rhythm_report = report.overall["rhythm"]
        assert len(rhythm_report.per_class) == 22
        assert rhythm_report.per_class[0].name == "AF"

    def test_risk_metrics(
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
        report = benchmark(
            aortica_model, synthetic_dataset,
            tasks=["rhythm"], batch_size=16,
        )
        assert report.tasks_evaluated == ["rhythm"]
        assert "rhythm" in report.overall
        assert "risk" not in report.overall

    def test_with_metadata(
        self,
        aortica_model: torch.nn.Module,
        synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
        sample_metadata: list[dict[str, object] | None],
    ) -> None:
        report = benchmark(
            aortica_model, synthetic_dataset,
            metadata=sample_metadata, batch_size=16,
        )
        assert len(report.subgroups) > 0
        sg_names = [sg.subgroup_name for sg in report.subgroups]
        assert any("sex_" in name for name in sg_names)
        assert any("age_" in name for name in sg_names)

    def test_subgroup_has_metrics(
        self,
        aortica_model: torch.nn.Module,
        synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
        sample_metadata: list[dict[str, object] | None],
    ) -> None:
        report = benchmark(
            aortica_model, synthetic_dataset,
            metadata=sample_metadata, batch_size=16,
        )
        for sg in report.subgroups:
            assert sg.n_samples > 0
            assert len(sg.task_reports) > 0

    def test_reproducible(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        r1 = benchmark(aortica_model, synthetic_dataset, batch_size=16, seed=42)
        r2 = benchmark(aortica_model, synthetic_dataset, batch_size=16, seed=42)
        assert r1.overall["rhythm"].macro_f1 == r2.overall["rhythm"].macro_f1
        assert r1.overall["risk"].c_index == r2.overall["risk"].c_index

    def test_as_dict_output(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        d = report.as_dict()
        assert isinstance(d, dict)
        assert "overall" in d
        assert "subgroups" in d
        assert "n_samples" in d

    def test_summary_table_output(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        table = report.summary_table()
        assert isinstance(table, str)
        assert "RHYTHM" in table

    def test_csv_export(
        self, aortica_model: torch.nn.Module, synthetic_dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    ) -> None:
        report = benchmark(aortica_model, synthetic_dataset, batch_size=16)
        csv_str = report.to_csv()
        assert isinstance(csv_str, str)
        lines = csv_str.strip().split("\n")
        assert len(lines) > 1  # header + data rows


# ===================================================================
# Imports test
# ===================================================================


class TestImports:
    """Test that the public API is importable."""

    def test_evaluation_package_exports(self) -> None:
        from aortica.evaluation import (
            BenchmarkReport,
            ClassMetrics,
            SubgroupReport,
            TaskReport,
            benchmark,
        )
        assert BenchmarkReport is not None
        assert ClassMetrics is not None
        assert SubgroupReport is not None
        assert TaskReport is not None
        assert benchmark is not None
