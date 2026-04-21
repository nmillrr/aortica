"""Tests for ``POST /api/v1/predict`` — single ECG inference endpoint."""

from __future__ import annotations

import io
import struct
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402
from aortica.api.predict import (  # noqa: E402
    LeadQualityResponse,
    PredictResponse,
    QualityReportResponse,
    TaskPrediction,
    UncertaintyResponse,
    run_inference_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_csv_bytes(
    *,
    num_leads: int = 12,
    num_samples: int = 500,
    sample_rate: float = 500.0,
) -> bytes:
    """Create a synthetic CSV file with sine-wave ECG data.

    The first row is a header with lead names (I, II, III, ... V6).
    Subsequent rows contain sample values.
    """
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"][:num_leads]
    header = ",".join(lead_names)

    rng = np.random.default_rng(42)
    t = np.arange(num_samples) / sample_rate
    lines = [header]
    for i in range(num_samples):
        values = []
        for ch in range(num_leads):
            # Synthetic QRS-like signal with slight variation per lead
            val = float(
                200.0 * np.sin(2 * np.pi * 1.0 * t[i] + ch * 0.3)
                + rng.normal(0, 5)
            )
            values.append(f"{val:.4f}")
        lines.append(",".join(values))

    return ("\n".join(lines)).encode("utf-8")


def _make_synthetic_wfdb_files(
    tmp_path: Any,
    *,
    record_name: str = "test_record",
    num_leads: int = 2,
    num_samples: int = 500,
    sample_rate: int = 500,
) -> str:
    """Create a synthetic WFDB record (.hea + .dat) and return the base path."""
    wfdb = pytest.importorskip("wfdb")

    rng = np.random.default_rng(42)
    signals = rng.normal(0, 0.5, (num_samples, num_leads)).astype(np.float64)
    lead_names = ["II", "V5"][:num_leads]

    base_path = str(tmp_path / record_name)
    wfdb.wrsamp(
        record_name=record_name,
        fs=sample_rate,
        units=["mV"] * num_leads,
        sig_name=lead_names,
        p_signal=signals,
        write_dir=str(tmp_path),
    )
    return base_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> Any:
    """Return a default-configured FastAPI application (no model loaded)."""
    return create_app(enable_auth=False)


@pytest.fixture()
def client(app: Any) -> TestClient:
    """Synchronous test client for the default app."""
    return TestClient(app)


@pytest.fixture()
def csv_bytes() -> bytes:
    """Synthetic CSV ECG file bytes."""
    return _make_synthetic_csv_bytes()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TestLeadQualityResponse:
    """Tests for LeadQualityResponse model."""

    def test_construction(self) -> None:
        resp = LeadQualityResponse(
            lead_name="I",
            score=85.0,
            classification="good",
            flags=["baseline_wander"],
        )
        assert resp.lead_name == "I"
        assert resp.score == 85.0
        assert resp.classification == "good"
        assert resp.flags == ["baseline_wander"]

    def test_empty_flags(self) -> None:
        resp = LeadQualityResponse(
            lead_name="II", score=100.0, classification="good"
        )
        assert resp.flags == []


class TestQualityReportResponse:
    """Tests for QualityReportResponse model."""

    def test_construction(self) -> None:
        lead = LeadQualityResponse(
            lead_name="I", score=90.0, classification="good"
        )
        resp = QualityReportResponse(
            per_lead=[lead],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        assert len(resp.per_lead) == 1
        assert resp.recommendation == "accept"


class TestTaskPrediction:
    """Tests for TaskPrediction model."""

    def test_construction(self) -> None:
        pred = TaskPrediction(
            task="rhythm",
            class_names=["AF", "NSR"],
            probabilities=[0.95, 0.05],
        )
        assert pred.task == "rhythm"
        assert len(pred.class_names) == 2
        assert len(pred.probabilities) == 2


class TestUncertaintyResponse:
    """Tests for UncertaintyResponse model."""

    def test_defaults(self) -> None:
        resp = UncertaintyResponse()
        assert resp.prediction_sets == {}
        assert resp.confidence_intervals == {}
        assert resp.ood_flag is False
        assert resp.entropy_score is None

    def test_with_values(self) -> None:
        resp = UncertaintyResponse(
            prediction_sets={"rhythm": [[0, 1, 2]]},
            ood_flag=True,
            entropy_score=0.42,
        )
        assert resp.ood_flag is True
        assert resp.entropy_score == 0.42


class TestPredictResponse:
    """Tests for PredictResponse model."""

    def test_construction(self) -> None:
        lead = LeadQualityResponse(
            lead_name="I", score=90.0, classification="good"
        )
        qr = QualityReportResponse(
            per_lead=[lead],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        resp = PredictResponse(
            quality_report=qr,
            predictions=[],
            uncertainty=None,
        )
        assert resp.quality_report.overall_score == 90.0
        assert resp.predictions == []

    def test_json_roundtrip(self) -> None:
        lead = LeadQualityResponse(
            lead_name="II", score=75.0, classification="good"
        )
        qr = QualityReportResponse(
            per_lead=[lead],
            overall_score=75.0,
            overall_classification="good",
            recommendation="accept",
        )
        pred = TaskPrediction(
            task="rhythm",
            class_names=["AF"],
            probabilities=[0.9],
        )
        resp = PredictResponse(
            quality_report=qr,
            predictions=[pred],
        )
        data = resp.model_dump()
        assert data["quality_report"]["overall_score"] == 75.0
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["task"] == "rhythm"


# ---------------------------------------------------------------------------
# POST /api/v1/predict — endpoint tests (no model loaded)
# ---------------------------------------------------------------------------


class TestPredictEndpointNoModel:
    """Tests for POST /api/v1/predict without a loaded model."""

    def test_returns_200_with_wfdb(self, tmp_path: Any) -> None:
        """Upload a WFDB file and get quality report + empty predictions."""
        base_path = _make_synthetic_wfdb_files(tmp_path)

        app = create_app(enable_auth=False)
        client = TestClient(app)

        hea_path = base_path + ".hea"
        with open(hea_path, "rb") as f:
            hea_bytes = f.read()
        dat_path = base_path + ".dat"
        with open(dat_path, "rb") as f:
            dat_bytes = f.read()

        # For WFDB we need both files; use the format override approach
        # with a single .hea file, but WFDB reader needs both.
        # Instead we'll mock read_ecg to return a record directly.
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", hea_bytes, "application/octet-stream")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "quality_report" in body
        assert "predictions" in body
        assert body["predictions"] == []  # no model loaded

    def test_returns_200_with_quality_report(self) -> None:
        """Quality report fields are present in the response."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        qr = resp.json()["quality_report"]
        assert "per_lead" in qr
        assert "overall_score" in qr
        assert "overall_classification" in qr
        assert "recommendation" in qr
        assert isinstance(qr["per_lead"], list)
        assert len(qr["per_lead"]) == 2

    def test_quality_report_per_lead_fields(self) -> None:
        """Each per-lead quality entry has the expected fields."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.zeros((2, 500), dtype=np.float64),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="csv",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.csv", b"dummy", "text/csv")},
            )

        assert resp.status_code == 200
        lead = resp.json()["quality_report"]["per_lead"][0]
        assert "lead_name" in lead
        assert "score" in lead
        assert "classification" in lead
        assert "flags" in lead

    def test_format_query_parameter(self) -> None:
        """The format query parameter is forwarded to the pipeline."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="dicom",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record) as mock_read:
            resp = client.post(
                "/api/v1/predict?format=dicom",
                files={"file": ("ecg.bin", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        # Verify the format parameter was passed through
        call_kwargs = mock_read.call_args
        assert call_kwargs is not None
        # read_ecg is called with format= keyword
        assert "format" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 1
        )

    def test_returns_422_for_unsupported_format(self) -> None:
        """Unsupported format returns 422 with error detail."""
        from aortica.io.dispatcher import UnsupportedFormatError

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch(
            "aortica.api.predict.read_ecg",
            side_effect=UnsupportedFormatError("Unknown format 'xyz'"),
        ):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.xyz", b"bogus", "application/octet-stream")},
            )

        assert resp.status_code == 422
        assert "detail" in resp.json()

    def test_returns_422_for_parse_error(self) -> None:
        """Files that fail to parse return 422."""
        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch(
            "aortica.api.predict.read_ecg",
            side_effect=ValueError("Failed to parse ECG data"),
        ):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("bad.csv", b"not,real,data", "text/csv")},
            )

        assert resp.status_code == 422
        assert "detail" in resp.json()

    def test_content_type_json(self) -> None:
        """Response content type is JSON."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert "application/json" in resp.headers["content-type"]

    def test_no_predictions_without_model(self) -> None:
        """When no model is loaded, predictions list is empty."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        assert resp.json()["predictions"] == []
        assert resp.json()["uncertainty"] is None

    def test_response_matches_pydantic_model(self) -> None:
        """Response conforms to PredictResponse schema."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        parsed = PredictResponse(**resp.json())
        assert parsed.quality_report is not None
        assert isinstance(parsed.predictions, list)


# ---------------------------------------------------------------------------
# POST /api/v1/predict — endpoint tests with mock model
# ---------------------------------------------------------------------------


class TestPredictEndpointWithModel:
    """Tests for POST /api/v1/predict with a mock model."""

    @staticmethod
    def _make_mock_model(
        enabled_tasks: list[str] | None = None,
    ) -> MagicMock:
        """Create a mock AorticaModel."""
        if enabled_tasks is None:
            enabled_tasks = ["rhythm", "structural", "ischaemia", "risk"]

        torch = pytest.importorskip("torch")

        model = MagicMock()
        model.enabled_tasks = enabled_tasks
        model.eval = MagicMock()

        # Mock forward output
        from aortica.models.aortica_model import MultiTaskOutput

        output = MultiTaskOutput(
            rhythm=torch.rand(1, 22) if "rhythm" in enabled_tasks else None,
            structural=torch.rand(1, 15) if "structural" in enabled_tasks else None,
            ischaemia=torch.rand(1, 10) if "ischaemia" in enabled_tasks else None,
            risk=torch.rand(1, 3) if "risk" in enabled_tasks else None,
        )
        model.__call__ = MagicMock(return_value=output)
        model.return_value = output
        return model

    def test_returns_predictions_with_model(self) -> None:
        """When a model is loaded, predictions are populated."""
        torch = pytest.importorskip("torch")
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (12, 5000)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"],
            duration_seconds=10.0,
            source_format="wfdb",
        )

        mock_model = self._make_mock_model()
        app = create_app(model=mock_model, model_loaded=True, enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == 4
        task_names = {p["task"] for p in preds}
        assert task_names == {"rhythm", "structural", "ischaemia", "risk"}

    def test_predictions_have_class_names(self) -> None:
        """Predictions include human-readable class names."""
        torch = pytest.importorskip("torch")
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (12, 5000)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"],
            duration_seconds=10.0,
            source_format="wfdb",
        )

        mock_model = self._make_mock_model(enabled_tasks=["rhythm"])
        app = create_app(
            model=mock_model,
            model_loaded=True,
            enabled_tasks=["rhythm"],
            enable_auth=False,
        )
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) >= 1
        rhythm_pred = [p for p in preds if p["task"] == "rhythm"][0]
        assert len(rhythm_pred["class_names"]) == 22
        assert len(rhythm_pred["probabilities"]) == 22
        # Check probabilities are all between 0 and 1
        for p in rhythm_pred["probabilities"]:
            assert 0.0 <= p <= 1.0

    def test_prediction_probability_ranges(self) -> None:
        """All prediction probabilities are in [0, 1]."""
        torch = pytest.importorskip("torch")
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (12, 5000)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"],
            duration_seconds=10.0,
            source_format="wfdb",
        )

        mock_model = self._make_mock_model()
        app = create_app(model=mock_model, model_loaded=True, enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        for pred in resp.json()["predictions"]:
            for prob in pred["probabilities"]:
                assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# run_inference_pipeline — unit tests
# ---------------------------------------------------------------------------


class TestRunInferencePipeline:
    """Unit tests for the run_inference_pipeline function."""

    def test_returns_predict_response(self) -> None:
        """Pipeline returns a PredictResponse object."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_inference_pipeline(
                b"dummy-bytes",
                "test.hea",
            )

        assert isinstance(result, PredictResponse)

    def test_quality_report_populated(self) -> None:
        """Quality report is always populated."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_inference_pipeline(
                b"dummy-bytes",
                "test.hea",
            )

        assert result.quality_report.overall_score >= 0
        assert result.quality_report.overall_score <= 100
        assert result.quality_report.recommendation in ("accept", "review", "reject")

    def test_no_model_empty_predictions(self) -> None:
        """Without a model, predictions list is empty."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_inference_pipeline(
                b"dummy-bytes",
                "test.hea",
                model=None,
            )

        assert result.predictions == []
        assert result.uncertainty is None

    def test_format_override_passed(self) -> None:
        """Format override is forwarded to read_ecg."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="dicom",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record) as mock_fn:
            run_inference_pipeline(
                b"dummy-bytes",
                "ecg.bin",
                format_override="dicom",
            )

        # Check format was passed
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get("format") == "dicom"

    def test_unsupported_format_raises(self) -> None:
        """UnsupportedFormatError propagates from read_ecg."""
        from aortica.io.dispatcher import UnsupportedFormatError

        with patch(
            "aortica.api.predict.read_ecg",
            side_effect=UnsupportedFormatError("nope"),
        ):
            with pytest.raises(UnsupportedFormatError):
                run_inference_pipeline(
                    b"dummy-bytes",
                    "test.xyz",
                )

    def test_denoise_is_called(self) -> None:
        """The denoise step is invoked during the pipeline."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record), \
             patch("aortica.api.predict.denoise", return_value=mock_record) as mock_dn:
            run_inference_pipeline(b"dummy-bytes", "test.hea")

        mock_dn.assert_called_once()

    def test_score_quality_is_called(self) -> None:
        """The score_quality step is invoked during the pipeline."""
        from aortica.io.ecg_record import ECGRecord
        from aortica.signal.quality_scoring import LeadQuality, QualityReport

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        mock_quality = QualityReport(
            per_lead=[
                LeadQuality(lead_name="II", score=90.0, classification="good"),
                LeadQuality(lead_name="V5", score=85.0, classification="good"),
            ],
            overall_score=87.5,
            overall_classification="good",
            recommendation="accept",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record), \
             patch("aortica.api.predict.denoise", return_value=mock_record), \
             patch(
                 "aortica.api.predict.score_quality",
                 return_value=mock_quality,
             ) as mock_sq:
            result = run_inference_pipeline(b"dummy-bytes", "test.hea")

        mock_sq.assert_called_once()
        assert result.quality_report.overall_score == 87.5

    def test_temp_file_cleanup(self) -> None:
        """Temp files are cleaned up after pipeline execution."""
        import os
        import tempfile as _tempfile

        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        # Track which temp files are created
        created_paths: list[str] = []
        original_ntf = _tempfile.NamedTemporaryFile

        def tracking_ntf(*args: Any, **kwargs: Any) -> Any:
            f = original_ntf(*args, **kwargs)
            created_paths.append(f.name)
            return f

        with patch("aortica.api.predict.read_ecg", return_value=mock_record), \
             patch("aortica.api.predict.tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            result = run_inference_pipeline(b"dummy-bytes", "test.hea")

        # The pipeline should clean up its temp file
        assert isinstance(result, PredictResponse)
        for p in created_paths:
            assert not os.path.exists(p), f"Temp file was not cleaned up: {p}"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API imports work."""

    def test_import_predict_module(self) -> None:
        from aortica.api import predict

        assert hasattr(predict, "run_inference_pipeline")
        assert hasattr(predict, "PredictResponse")

    def test_import_response_models(self) -> None:
        from aortica.api.predict import (
            LeadQualityResponse,
            PredictResponse,
            QualityReportResponse,
            TaskPrediction,
            UncertaintyResponse,
            XAIAttributionResponse,
            XAIFeatureContribution,
            XAISegmentAttribution,
        )

        assert callable(LeadQualityResponse)
        assert callable(PredictResponse)
        assert callable(QualityReportResponse)
        assert callable(TaskPrediction)
        assert callable(UncertaintyResponse)
        assert callable(XAIAttributionResponse)
        assert callable(XAIFeatureContribution)
        assert callable(XAISegmentAttribution)


# ---------------------------------------------------------------------------
# XAI Response Models — US-046
# ---------------------------------------------------------------------------


class TestXAIFeatureContribution:
    """Tests for XAIFeatureContribution model."""

    def test_construction(self) -> None:
        from aortica.api.predict import XAIFeatureContribution

        fc = XAIFeatureContribution(
            feature_name="QRS complex",
            lead="V1",
            delta_score=0.82,
        )
        assert fc.feature_name == "QRS complex"
        assert fc.lead == "V1"
        assert fc.delta_score == 0.82


class TestXAISegmentAttribution:
    """Tests for XAISegmentAttribution model."""

    def test_construction(self) -> None:
        from aortica.api.predict import XAISegmentAttribution

        sa = XAISegmentAttribution(
            lead="II",
            segments={"P wave": 0.12, "QRS complex": 0.85, "T wave": 0.31},
        )
        assert sa.lead == "II"
        assert len(sa.segments) == 3
        assert sa.segments["QRS complex"] == 0.85


class TestXAIAttributionResponse:
    """Tests for XAIAttributionResponse model."""

    def test_construction(self) -> None:
        from aortica.api.predict import (
            XAIAttributionResponse,
            XAIFeatureContribution,
            XAISegmentAttribution,
        )

        resp = XAIAttributionResponse(
            task="rhythm",
            per_lead_attributions={"I": [0.1, 0.2, 0.3], "II": [0.4, 0.5, 0.6]},
            segment_attributions=[
                XAISegmentAttribution(lead="I", segments={"QRS complex": 0.7}),
            ],
            top_features=[
                XAIFeatureContribution(
                    feature_name="QRS complex", lead="I", delta_score=0.7
                ),
            ],
        )
        assert resp.task == "rhythm"
        assert len(resp.per_lead_attributions) == 2
        assert resp.per_lead_attributions["I"] == [0.1, 0.2, 0.3]
        assert len(resp.segment_attributions) == 1
        assert len(resp.top_features) == 1

    def test_json_serialization(self) -> None:
        from aortica.api.predict import (
            XAIAttributionResponse,
            XAIFeatureContribution,
            XAISegmentAttribution,
        )

        resp = XAIAttributionResponse(
            task="structural",
            per_lead_attributions={"V1": [0.1]},
            segment_attributions=[
                XAISegmentAttribution(lead="V1", segments={"ST segment": 0.5}),
            ],
            top_features=[
                XAIFeatureContribution(
                    feature_name="ST segment", lead="V1", delta_score=0.5
                ),
            ],
            segment_boundaries={
                "V1": [{"p_start": 0, "p_end": 20, "qrs_start": 30,
                         "qrs_end": 50, "t_start": 70, "t_end": 100}],
            },
        )
        data = resp.model_dump()
        assert data["task"] == "structural"
        assert "V1" in data["per_lead_attributions"]
        assert len(data["segment_boundaries"]["V1"]) == 1


class TestPredictResponseWithXAI:
    """Tests for PredictResponse with XAI field."""

    def test_xai_default_none(self) -> None:
        from aortica.api.predict import (
            LeadQualityResponse,
            PredictResponse,
            QualityReportResponse,
        )

        lead = LeadQualityResponse(
            lead_name="I", score=90.0, classification="good"
        )
        qr = QualityReportResponse(
            per_lead=[lead],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        resp = PredictResponse(quality_report=qr, predictions=[])
        assert resp.xai is None

    def test_xai_populated(self) -> None:
        from aortica.api.predict import (
            LeadQualityResponse,
            PredictResponse,
            QualityReportResponse,
            XAIAttributionResponse,
            XAIFeatureContribution,
            XAISegmentAttribution,
        )

        lead = LeadQualityResponse(
            lead_name="I", score=90.0, classification="good"
        )
        qr = QualityReportResponse(
            per_lead=[lead],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        xai_item = XAIAttributionResponse(
            task="rhythm",
            per_lead_attributions={"I": [0.1, 0.2]},
            segment_attributions=[
                XAISegmentAttribution(lead="I", segments={"QRS complex": 0.5}),
            ],
            top_features=[
                XAIFeatureContribution(
                    feature_name="QRS complex", lead="I", delta_score=0.5
                ),
            ],
        )
        resp = PredictResponse(
            quality_report=qr, predictions=[], xai=[xai_item]
        )
        assert resp.xai is not None
        assert len(resp.xai) == 1
        assert resp.xai[0].task == "rhythm"

    def test_json_roundtrip_with_xai(self) -> None:
        from aortica.api.predict import (
            LeadQualityResponse,
            PredictResponse,
            QualityReportResponse,
            XAIAttributionResponse,
            XAIFeatureContribution,
            XAISegmentAttribution,
        )

        lead = LeadQualityResponse(
            lead_name="II", score=80.0, classification="good"
        )
        qr = QualityReportResponse(
            per_lead=[lead],
            overall_score=80.0,
            overall_classification="good",
            recommendation="accept",
        )
        xai_item = XAIAttributionResponse(
            task="ischaemia",
            per_lead_attributions={"II": [0.5, 0.3]},
            segment_attributions=[
                XAISegmentAttribution(lead="II", segments={"T wave": 0.4}),
            ],
            top_features=[],
        )
        resp = PredictResponse(
            quality_report=qr, predictions=[], xai=[xai_item]
        )
        data = resp.model_dump()
        restored = PredictResponse(**data)
        assert restored.xai is not None
        assert restored.xai[0].task == "ischaemia"


class TestPipelineIncludeXAI:
    """Tests for include_xai parameter in run_inference_pipeline."""

    def test_include_xai_false_by_default(self) -> None:
        """XAI is not included by default."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_inference_pipeline(b"dummy-bytes", "test.hea")

        assert result.xai is None

    def test_include_xai_true_no_model(self) -> None:
        """When include_xai=True but no model, XAI is still None."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_inference_pipeline(
                b"dummy-bytes", "test.hea", include_xai=True
            )

        assert result.xai is None

    def test_endpoint_accepts_include_xai_param(self) -> None:
        """The predict endpoint accepts include_xai as query parameter."""
        from aortica.io.ecg_record import ECGRecord

        mock_record = ECGRecord(
            signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
                np.float64
            ),
            sample_rate=500.0,
            lead_names=["II", "V5"],
            duration_seconds=1.0,
            source_format="wfdb",
        )

        app = create_app(enable_auth=False)
        client = TestClient(app)

        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict?include_xai=true",
                files={"file": ("test.hea", b"dummy", "application/octet-stream")},
            )

        assert resp.status_code == 200
        body = resp.json()
        # No model loaded, so xai should be null
        assert body.get("xai") is None

