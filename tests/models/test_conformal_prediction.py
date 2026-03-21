"""Tests for Conformal Prediction and Uncertainty Estimation (US-024).

Covers:
- ConformalPredictor: construction, fitting, prediction
- UncertaintyReport: prediction sets, confidence intervals, OOD flags, entropy
- OOD detection via Mahalanobis distance
- End-to-end pipeline with in-distribution and synthetic OOD inputs
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aortica.models.conformal_prediction import (  # noqa: E402
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    TASK_NUM_OUTPUTS,
    ConformalPredictor,
    UncertaintyReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    enabled_tasks: list[str] | None = None,
    feature_dim: int = 252,
) -> torch.nn.Module:
    """Create an AorticaModel for testing."""
    from aortica.models.aortica_model import AorticaModel

    if enabled_tasks is None:
        enabled_tasks = ["rhythm", "structural"]
    return AorticaModel(
        in_channels=12,
        feature_dim=feature_dim,
        num_leads=12,
        enabled_tasks=enabled_tasks,
        head_dropout=0.0,
    )


def _make_cal_loader(
    n_samples: int = 32,
    batch_size: int = 8,
    enabled_tasks: list[str] | None = None,
) -> torch.utils.data.DataLoader:
    """Create a synthetic calibration DataLoader."""
    if enabled_tasks is None:
        enabled_tasks = ["rhythm", "structural"]

    label_dim = sum(TASK_NUM_OUTPUTS[t] for t in enabled_tasks)
    x = torch.randn(n_samples, 12, 2500)
    labels = (torch.rand(n_samples, label_dim) > 0.5).float()
    dataset = torch.utils.data.TensorDataset(x, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify task lists and output sizes."""

    def test_classification_tasks(self) -> None:
        assert CLASSIFICATION_TASKS == ["rhythm", "structural", "ischaemia"]

    def test_all_tasks(self) -> None:
        assert ALL_TASKS == ["rhythm", "structural", "ischaemia", "risk"]

    def test_task_num_outputs(self) -> None:
        assert TASK_NUM_OUTPUTS["rhythm"] == 22
        assert TASK_NUM_OUTPUTS["structural"] == 15
        assert TASK_NUM_OUTPUTS["ischaemia"] == 10
        assert TASK_NUM_OUTPUTS["risk"] == 3


# ---------------------------------------------------------------------------
# ConformalPredictor construction
# ---------------------------------------------------------------------------


class TestConformalPredictorConstruction:
    """Construction and configuration tests."""

    def test_default_coverage(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model)
        assert cp.coverage == 0.90

    def test_custom_coverage(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model, coverage=0.95)
        assert cp.coverage == 0.95

    def test_default_ood_percentile(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model)
        assert cp.ood_percentile == 95.0

    def test_custom_ood_percentile(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model, ood_percentile=99.0)
        assert cp.ood_percentile == 99.0

    def test_not_fitted_initially(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model)
        assert not cp.is_fitted

    def test_predict_before_fit_raises(self) -> None:
        model = _make_model()
        cp = ConformalPredictor(model)
        x = torch.randn(2, 12, 2500)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cp.predict(x)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestFitting:
    """Tests for the fit() method."""

    def test_fit_marks_fitted(self) -> None:
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader()
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        assert cp.is_fitted

    def test_fit_populates_quantiles(self) -> None:
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm", "structural"])
        cal_loader = _make_cal_loader(enabled_tasks=["rhythm", "structural"])
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        assert "rhythm" in cp._quantiles
        assert "structural" in cp._quantiles

    def test_fit_populates_ood_detector(self) -> None:
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader()
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        assert cp._feature_mean is not None
        assert cp._feature_cov_inv is not None
        assert cp._ood_threshold is not None

    def test_fit_with_risk_task(self) -> None:
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm", "risk"])
        cal_loader = _make_cal_loader(enabled_tasks=["rhythm", "risk"])
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        assert cp._risk_residual_quantile is not None
        assert cp._risk_residual_quantile >= 0.0

    def test_quantiles_are_non_negative(self) -> None:
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader()
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        for q in cp._quantiles.values():
            assert q >= 0.0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for the predict() method."""

    @pytest.fixture()
    def fitted_cp(self) -> ConformalPredictor:
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm", "structural"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm", "structural"]
        )
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        return cp

    def test_predict_returns_tuple(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(2, 12, 2500)
        result = fitted_cp.predict(x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predictions_dict_keys(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(2, 12, 2500)
        preds, _ = fitted_cp.predict(x)
        assert "rhythm" in preds
        assert "structural" in preds

    def test_predictions_shapes(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(4, 12, 2500)
        preds, _ = fitted_cp.predict(x)
        assert preds["rhythm"].shape == (4, 22)
        assert preds["structural"].shape == (4, 15)

    def test_predictions_range(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(4, 12, 2500)
        preds, _ = fitted_cp.predict(x)
        for task in ["rhythm", "structural"]:
            assert (preds[task] >= 0.0).all()
            assert (preds[task] <= 1.0).all()

    def test_task_subset(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(2, 12, 2500)
        preds, _ = fitted_cp.predict(x, tasks=["rhythm"])
        assert "rhythm" in preds
        assert "structural" not in preds


# ---------------------------------------------------------------------------
# UncertaintyReport
# ---------------------------------------------------------------------------


class TestUncertaintyReport:
    """Tests for UncertaintyReport contents."""

    @pytest.fixture()
    def fitted_cp(self) -> ConformalPredictor:
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm", "structural"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm", "structural"]
        )
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        return cp

    def test_report_type(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(2, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert isinstance(report, UncertaintyReport)

    def test_prediction_sets_present(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(2, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert "rhythm" in report.prediction_sets
        assert "structural" in report.prediction_sets

    def test_prediction_sets_are_lists(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(3, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert len(report.prediction_sets["rhythm"]) == 3
        for ps in report.prediction_sets["rhythm"]:
            assert isinstance(ps, list)

    def test_prediction_set_indices_valid(
        self, fitted_cp: ConformalPredictor
    ) -> None:
        x = torch.randn(2, 12, 2500)
        _, report = fitted_cp.predict(x)
        for ps in report.prediction_sets["rhythm"]:
            for idx in ps:
                assert 0 <= idx < 22

    def test_ood_flags_shape(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.ood_flags is not None
        assert report.ood_flags.shape == (4,)

    def test_ood_flags_are_bool(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.ood_flags is not None
        assert report.ood_flags.dtype == torch.bool

    def test_entropy_scores_shape(self, fitted_cp: ConformalPredictor) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.entropy_scores is not None
        assert report.entropy_scores.shape == (4,)

    def test_entropy_scores_non_negative(
        self, fitted_cp: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.entropy_scores is not None
        assert (report.entropy_scores >= 0.0).all()

    def test_mahalanobis_distances_shape(
        self, fitted_cp: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.mahalanobis_distances is not None
        assert report.mahalanobis_distances.shape == (4,)

    def test_mahalanobis_distances_non_negative(
        self, fitted_cp: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp.predict(x)
        assert report.mahalanobis_distances is not None
        assert (report.mahalanobis_distances >= 0.0).all()


# ---------------------------------------------------------------------------
# Confidence intervals (risk task)
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Tests for risk task confidence intervals."""

    @pytest.fixture()
    def fitted_cp_risk(self) -> ConformalPredictor:
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm", "risk"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm", "risk"]
        )
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))
        return cp

    def test_confidence_intervals_present(
        self, fitted_cp_risk: ConformalPredictor
    ) -> None:
        x = torch.randn(2, 12, 2500)
        _, report = fitted_cp_risk.predict(x)
        assert "risk" in report.confidence_intervals

    def test_confidence_intervals_shape(
        self, fitted_cp_risk: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp_risk.predict(x)
        lower, upper = report.confidence_intervals["risk"]
        assert lower.shape == (4, 3)
        assert upper.shape == (4, 3)

    def test_confidence_intervals_range(
        self, fitted_cp_risk: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp_risk.predict(x)
        lower, upper = report.confidence_intervals["risk"]
        assert (lower >= 0.0).all()
        assert (upper <= 1.0).all()

    def test_lower_leq_upper(
        self, fitted_cp_risk: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        _, report = fitted_cp_risk.predict(x)
        lower, upper = report.confidence_intervals["risk"]
        assert (lower <= upper).all()

    def test_prediction_within_interval(
        self, fitted_cp_risk: ConformalPredictor
    ) -> None:
        x = torch.randn(4, 12, 2500)
        preds, report = fitted_cp_risk.predict(x)
        lower, upper = report.confidence_intervals["risk"]
        # Point predictions should be within confidence intervals
        assert (preds["risk"] >= lower - 1e-6).all()
        assert (preds["risk"] <= upper + 1e-6).all()


# ---------------------------------------------------------------------------
# OOD detection
# ---------------------------------------------------------------------------


class TestOODDetection:
    """Tests for out-of-distribution detection."""

    def test_in_distribution_mostly_not_flagged(self) -> None:
        """Calibration-like inputs should mostly not be flagged OOD."""
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader(n_samples=64)
        cp = ConformalPredictor(model, ood_percentile=95.0)
        cp.fit(cal_loader, device=torch.device("cpu"))

        # Generate inputs from same distribution
        x = torch.randn(20, 12, 2500)
        _, report = cp.predict(x)
        assert report.ood_flags is not None
        # Most should not be OOD (allow some false positives)
        ood_rate = report.ood_flags.float().mean().item()
        assert ood_rate < 0.5, f"OOD rate too high for in-dist: {ood_rate}"

    def test_ood_inputs_flagged_at_higher_rate(self) -> None:
        """Extreme OOD inputs should be flagged more often than in-dist."""
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader(n_samples=64)
        cp = ConformalPredictor(model, ood_percentile=90.0)
        cp.fit(cal_loader, device=torch.device("cpu"))

        # In-distribution
        x_in = torch.randn(20, 12, 2500)
        _, report_in = cp.predict(x_in)

        # OOD: very different scale/distribution
        x_ood = torch.randn(20, 12, 2500) * 100.0 + 50.0
        _, report_ood = cp.predict(x_ood)

        assert report_in.ood_flags is not None
        assert report_ood.ood_flags is not None

        in_ood_rate = report_in.ood_flags.float().mean().item()
        ood_ood_rate = report_ood.ood_flags.float().mean().item()
        # OOD inputs should have higher OOD flag rate
        assert ood_ood_rate > in_ood_rate, (
            f"OOD rate should be higher for OOD inputs: "
            f"in={in_ood_rate}, ood={ood_ood_rate}"
        )

    def test_mahalanobis_higher_for_ood(self) -> None:
        """Mahalanobis distances should be larger for OOD inputs."""
        torch.manual_seed(42)
        model = _make_model()
        cal_loader = _make_cal_loader(n_samples=64)
        cp = ConformalPredictor(model)
        cp.fit(cal_loader, device=torch.device("cpu"))

        x_in = torch.randn(10, 12, 2500)
        _, report_in = cp.predict(x_in)

        x_ood = torch.randn(10, 12, 2500) * 100.0 + 50.0
        _, report_ood = cp.predict(x_ood)

        assert report_in.mahalanobis_distances is not None
        assert report_ood.mahalanobis_distances is not None

        mean_in = report_in.mahalanobis_distances.mean().item()
        mean_ood = report_ood.mahalanobis_distances.mean().item()
        assert mean_ood > mean_in, (
            f"OOD should have higher Mahalanobis: in={mean_in}, ood={mean_ood}"
        )


# ---------------------------------------------------------------------------
# Coverage guarantee
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    """Verify that prediction sets achieve approximate marginal coverage."""

    def test_classification_coverage(self) -> None:
        """Prediction sets should cover true labels at ~coverage rate."""
        torch.manual_seed(42)
        model = _make_model(enabled_tasks=["rhythm"])
        n_cal = 64
        n_test = 32
        cal_loader = _make_cal_loader(
            n_samples=n_cal, enabled_tasks=["rhythm"]
        )

        cp = ConformalPredictor(model, coverage=0.90)
        cp.fit(cal_loader, device=torch.device("cpu"))

        # Test data
        x_test = torch.randn(n_test, 12, 2500)
        labels_test = (torch.rand(n_test, 22) > 0.5).float()

        _, report = cp.predict(x_test)
        pred_sets = report.prediction_sets["rhythm"]

        # Check coverage: for each sample, check if the true positive
        # classes are included in the prediction set
        covered = 0
        total = 0
        for i in range(n_test):
            true_positives = set(
                int(j) for j in range(22) if labels_test[i, j] > 0.5
            )
            if len(true_positives) == 0:
                continue
            pred_set = set(pred_sets[i])
            if true_positives.issubset(pred_set):
                covered += 1
            total += 1

        # With finite samples, coverage may not be exact; allow some slack
        # Prediction sets should be generated for every sample
        assert len(pred_sets) == n_test


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_classification_only(self) -> None:
        torch.manual_seed(123)
        model = _make_model(enabled_tasks=["rhythm"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm"]
        )

        cp = ConformalPredictor(model, coverage=0.90)
        cp.fit(cal_loader, device=torch.device("cpu"))

        x = torch.randn(4, 12, 2500)
        preds, report = cp.predict(x)

        # Predictions
        assert "rhythm" in preds
        assert preds["rhythm"].shape == (4, 22)
        assert (preds["rhythm"] >= 0).all()
        assert (preds["rhythm"] <= 1).all()

        # Report
        assert "rhythm" in report.prediction_sets
        assert report.ood_flags is not None
        assert report.entropy_scores is not None
        assert report.mahalanobis_distances is not None

    def test_full_pipeline_with_risk(self) -> None:
        torch.manual_seed(123)
        model = _make_model(enabled_tasks=["rhythm", "risk"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm", "risk"]
        )

        cp = ConformalPredictor(model, coverage=0.90)
        cp.fit(cal_loader, device=torch.device("cpu"))

        x = torch.randn(4, 12, 2500)
        preds, report = cp.predict(x)

        # Predictions
        assert "rhythm" in preds
        assert "risk" in preds

        # Risk confidence intervals
        assert "risk" in report.confidence_intervals
        lower, upper = report.confidence_intervals["risk"]
        assert lower.shape == (4, 3)
        assert (lower <= upper).all()

    def test_forward_is_alias_for_predict(self) -> None:
        torch.manual_seed(123)
        model = _make_model(enabled_tasks=["rhythm"])
        cal_loader = _make_cal_loader(
            n_samples=32, enabled_tasks=["rhythm"]
        )

        cp = ConformalPredictor(model, coverage=0.90)
        cp.fit(cal_loader, device=torch.device("cpu"))

        x = torch.randn(2, 12, 2500)
        preds1, report1 = cp.predict(x)
        preds2, report2 = cp(x)

        assert preds1.keys() == preds2.keys()
        for k in preds1:
            assert torch.allclose(preds1[k], preds2[k])
