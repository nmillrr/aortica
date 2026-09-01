"""Untrained model outputs must never reach a caller by default.

44 of the model's 72 outputs — the entire risk head among them — have never
been trained by any released checkpoint.  They still emit numbers, and those
numbers drift with the backbone.  The model card said callers had to gate
them; nothing did, so ``aortica predict`` and ``POST /api/v1/predict`` could
serve ``mortality_1y`` and ``sudden_cardiac_death_risk`` as if they were
predictions, straight into a three-tier triage UI.

Gating now lives in ``run_inference_pipeline``, the one path every caller
goes through, and is on unless explicitly disabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aortica.api.predict import run_inference_pipeline
from aortica.models.output_gating import default_trained_outputs

ALL_TASKS = ["rhythm", "structural", "ischaemia", "risk"]
LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def _mock_record() -> object:
    from aortica.io.ecg_record import ECGRecord

    return ECGRecord(
        signals=np.random.default_rng(0)
        .normal(0, 100, (12, 5000))
        .astype(np.float64),
        sample_rate=500.0,
        lead_names=list(LEAD_NAMES),
        duration_seconds=10.0,
        source_format="wfdb",
    )


def _mock_model(enabled_tasks: list[str] | None = None) -> MagicMock:
    torch = pytest.importorskip("torch")
    from aortica.models.aortica_model import MultiTaskOutput

    tasks = ALL_TASKS if enabled_tasks is None else enabled_tasks

    # Every head saturated, so an ungated response is unmistakable.
    output = MultiTaskOutput(
        rhythm=torch.ones(1, 28) if "rhythm" in tasks else None,
        structural=torch.ones(1, 19) if "structural" in tasks else None,
        ischaemia=torch.ones(1, 19) if "ischaemia" in tasks else None,
        risk=torch.ones(1, 6) if "risk" in tasks else None,
    )

    model = MagicMock()
    model.enabled_tasks = list(tasks)
    model.eval = MagicMock()
    model.return_value = output
    return model


def _run(model: MagicMock, **kwargs: object) -> object:
    with patch("aortica.api.predict.read_ecg", return_value=_mock_record()):
        return run_inference_pipeline(b"dummy", "test.hea", model=model, **kwargs)


class TestDefaultSuppression:
    """Suppression is the default, not an opt-in."""

    def test_risk_head_reports_nothing(self) -> None:
        pytest.importorskip("torch")
        result = _run(_mock_model())

        risk = [p for p in result.predictions if p.task == "risk"][0]
        assert risk.class_names == []
        assert risk.probabilities == []
        assert "mortality_1y" in risk.suppressed_classes
        assert "sudden_cardiac_death_risk" in risk.suppressed_classes

    def test_no_untrained_output_is_ever_reported(self) -> None:
        pytest.importorskip("torch")
        result = _run(_mock_model())

        allowlist = default_trained_outputs()
        for pred in result.predictions:
            for name in pred.class_names:
                assert name in allowlist, f"{name} was never trained"

    def test_trained_outputs_survive(self) -> None:
        """Gating removes the untrained heads, not the useful ones."""
        pytest.importorskip("torch")
        result = _run(_mock_model())

        reported = {n for p in result.predictions for n in p.class_names}
        assert {"AF", "RBBB", "LVH", "STEMI"} <= reported

    def test_every_output_is_accounted_for(self) -> None:
        """Nothing is dropped silently: kept + suppressed covers the head."""
        pytest.importorskip("torch")
        result = _run(_mock_model())

        widths = {"rhythm": 28, "structural": 19, "ischaemia": 19, "risk": 6}
        for pred in result.predictions:
            total = len(pred.class_names) + len(pred.suppressed_classes)
            assert total == widths[pred.task]

    def test_lists_stay_aligned(self) -> None:
        pytest.importorskip("torch")
        result = _run(_mock_model())

        for pred in result.predictions:
            assert len(pred.class_names) == len(pred.probabilities)


class TestExplicitOptOut:
    """Turning gating off has to be deliberate."""

    def test_opt_out_reports_every_output(self) -> None:
        pytest.importorskip("torch")
        result = _run(_mock_model(), suppress_untrained=False)

        risk = [p for p in result.predictions if p.task == "risk"][0]
        assert len(risk.class_names) == 6
        assert risk.suppressed_classes == []

    def test_opt_out_is_not_the_default(self) -> None:
        """The signature default must stay True."""
        import inspect

        signature = inspect.signature(run_inference_pipeline)
        assert signature.parameters["suppress_untrained"].default is True


class TestDownstreamConsumers:
    """Suppression must not corrupt data keyed by output position."""

    def test_clinical_suggestions_skip_untrained_findings(self) -> None:
        """A saturated untrained head must not generate a suggestion."""
        pytest.importorskip("torch")
        result = _run(_mock_model(), include_suggestions=True)

        allowlist = default_trained_outputs()
        for entry in result.suggestions or []:
            assert entry.condition in allowlist
