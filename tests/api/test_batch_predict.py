"""Tests for ``POST /api/v1/predict/batch`` — batch ECG inference endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402
from aortica.api.batch_predict import (  # noqa: E402
    DEFAULT_MAX_BATCH_SIZE,
    BatchFileResult,
    BatchPredictResponse,
    run_batch_inference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ecg_record() -> Any:
    """Return a mock ECGRecord for patching read_ecg."""
    from aortica.io.ecg_record import ECGRecord

    return ECGRecord(
        signals=np.random.default_rng(0).normal(0, 100, (2, 500)).astype(
            np.float64
        ),
        sample_rate=500.0,
        lead_names=["II", "V5"],
        duration_seconds=1.0,
        source_format="wfdb",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Synchronous test client for the default app (no model)."""
    return TestClient(create_app(enable_auth=False))


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TestBatchFileResult:
    """Tests for BatchFileResult model."""

    def test_construction_success(self) -> None:
        from aortica.api.predict import PredictResponse, QualityReportResponse

        qr = QualityReportResponse(
            per_lead=[],
            overall_score=90.0,
            overall_classification="good",
            recommendation="accept",
        )
        pr = PredictResponse(
            quality_report=qr, predictions=[], uncertainty=None
        )
        r = BatchFileResult(
            filename="test.hea", status="success", result=pr
        )
        assert r.filename == "test.hea"
        assert r.status == "success"
        assert r.error is None
        assert r.result is not None

    def test_construction_error(self) -> None:
        r = BatchFileResult(
            filename="bad.csv",
            status="error",
            error="Failed to parse ECG data",
        )
        assert r.status == "error"
        assert r.error == "Failed to parse ECG data"
        assert r.result is None

    def test_json_roundtrip(self) -> None:
        r = BatchFileResult(
            filename="test.csv",
            status="error",
            error="parse error",
        )
        data = r.model_dump()
        assert data["filename"] == "test.csv"
        assert data["status"] == "error"
        assert data["result"] is None


class TestBatchPredictResponse:
    """Tests for BatchPredictResponse model."""

    def test_construction(self) -> None:
        resp = BatchPredictResponse(
            total=3, succeeded=2, failed=1, results=[]
        )
        assert resp.total == 3
        assert resp.succeeded == 2
        assert resp.failed == 1

    def test_counts_match_results(self) -> None:
        r1 = BatchFileResult(filename="a.csv", status="success")
        r2 = BatchFileResult(filename="b.csv", status="error", error="fail")
        resp = BatchPredictResponse(
            total=2, succeeded=1, failed=1, results=[r1, r2]
        )
        assert len(resp.results) == resp.total


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for batch_predict constants."""

    def test_default_max_batch_size(self) -> None:
        assert DEFAULT_MAX_BATCH_SIZE == 50


# ---------------------------------------------------------------------------
# run_batch_inference — unit tests
# ---------------------------------------------------------------------------


class TestRunBatchInference:
    """Unit tests for the run_batch_inference function."""

    def test_returns_batch_predict_response(self) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_batch_inference(
                [(b"data", "test.hea")],
            )
        assert isinstance(result, BatchPredictResponse)

    def test_single_file_success(self) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_batch_inference(
                [(b"data", "test.hea")],
            )
        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0
        assert result.results[0].status == "success"
        assert result.results[0].result is not None

    def test_multiple_files_success(self) -> None:
        mock_record = _mock_ecg_record()
        files = [(b"data", f"file_{i}.hea") for i in range(5)]
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_batch_inference(files)
        assert result.total == 5
        assert result.succeeded == 5
        assert result.failed == 0

    def test_file_failure_captured(self) -> None:
        """Failing files are captured with error messages, not raised."""
        with patch(
            "aortica.api.predict.read_ecg",
            side_effect=ValueError("bad file"),
        ):
            result = run_batch_inference(
                [(b"data", "bad.xyz")],
            )
        assert result.total == 1
        assert result.succeeded == 0
        assert result.failed == 1
        assert result.results[0].status == "error"
        assert "bad file" in (result.results[0].error or "")

    def test_mixed_success_and_failure(self) -> None:
        """Some files succeed, some fail — both are captured."""
        mock_record = _mock_ecg_record()
        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("parse error")
            return mock_record

        files = [
            (b"data1", "good1.hea"),
            (b"data2", "bad.xyz"),
            (b"data3", "good2.hea"),
        ]
        with patch(
            "aortica.api.predict.read_ecg", side_effect=_side_effect
        ):
            result = run_batch_inference(files)

        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1
        assert result.results[0].status == "success"
        assert result.results[1].status == "error"
        assert result.results[2].status == "success"

    def test_filenames_preserved(self) -> None:
        mock_record = _mock_ecg_record()
        files = [(b"d", "alice.hea"), (b"d", "bob.csv")]
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            result = run_batch_inference(files)
        assert [r.filename for r in result.results] == [
            "alice.hea",
            "bob.csv",
        ]

    def test_exceeds_max_batch_size_raises(self) -> None:
        files = [(b"d", f"f{i}.csv") for i in range(10)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            run_batch_inference(files, max_batch_size=5)

    def test_empty_batch(self) -> None:
        result = run_batch_inference([])
        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0
        assert result.results == []


# ---------------------------------------------------------------------------
# POST /api/v1/predict/batch — endpoint tests
# ---------------------------------------------------------------------------


class TestBatchEndpointNoModel:
    """Tests for POST /api/v1/predict/batch without a loaded model."""

    def test_returns_200(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("test1.hea", b"dummy", "application/octet-stream")),
                    ("files", ("test2.hea", b"dummy", "application/octet-stream")),
                ],
            )
        assert resp.status_code == 200

    def test_response_structure(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("a.hea", b"dummy", "application/octet-stream")),
                    ("files", ("b.hea", b"dummy", "application/octet-stream")),
                ],
            )
        body = resp.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2
        assert body["failed"] == 0
        assert len(body["results"]) == 2

    def test_per_file_result_fields(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("test.hea", b"dummy", "application/octet-stream")),
                ],
            )
        result = resp.json()["results"][0]
        assert result["filename"] == "test.hea"
        assert result["status"] == "success"
        assert result["error"] is None
        assert "quality_report" in result["result"]
        assert "predictions" in result["result"]

    def test_no_predictions_without_model(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("test.hea", b"dummy", "application/octet-stream")),
                ],
            )
        result = resp.json()["results"][0]["result"]
        assert result["predictions"] == []

    def test_mixed_success_error(self, client: TestClient) -> None:
        """One valid and one invalid file yield mixed results."""
        mock_record = _mock_ecg_record()
        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("corrupt file")
            return mock_record

        with patch(
            "aortica.api.predict.read_ecg", side_effect=_side_effect
        ):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("good.hea", b"ok", "application/octet-stream")),
                    ("files", ("bad.xyz", b"bad", "application/octet-stream")),
                ],
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["succeeded"] == 1
        assert body["failed"] == 1
        assert body["results"][0]["status"] == "success"
        assert body["results"][1]["status"] == "error"
        assert "corrupt file" in body["results"][1]["error"]

    def test_exceeds_max_batch_size_422(self) -> None:
        """Exceeding max batch size returns 422."""
        app = create_app(enable_auth=False)
        app.state.max_batch_size = 2  # type: ignore[attr-defined]
        c = TestClient(app)

        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = c.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("a.hea", b"d", "application/octet-stream")),
                    ("files", ("b.hea", b"d", "application/octet-stream")),
                    ("files", ("c.hea", b"d", "application/octet-stream")),
                ],
            )
        assert resp.status_code == 422
        assert "exceeds maximum" in resp.json()["detail"]

    def test_default_max_batch_size_allows_50(self, client: TestClient) -> None:
        """Default batch size allows up to 50 files."""
        mock_record = _mock_ecg_record()
        files = [
            ("files", (f"f{i}.hea", b"d", "application/octet-stream"))
            for i in range(3)
        ]
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post("/api/v1/predict/batch", files=files)
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_format_query_parameter(self, client: TestClient) -> None:
        """Format parameter is applied to all files."""
        mock_record = _mock_ecg_record()
        with patch(
            "aortica.api.predict.read_ecg", return_value=mock_record
        ):
            resp = client.post(
                "/api/v1/predict/batch?format=dicom",
                files=[
                    ("files", ("a.bin", b"d", "application/octet-stream")),
                ],
            )
        assert resp.status_code == 200

    def test_content_type_json(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("t.hea", b"d", "application/octet-stream")),
                ],
            )
        assert "application/json" in resp.headers["content-type"]

    def test_response_matches_pydantic_model(self, client: TestClient) -> None:
        mock_record = _mock_ecg_record()
        with patch("aortica.api.predict.read_ecg", return_value=mock_record):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("files", ("t.hea", b"d", "application/octet-stream")),
                ],
            )
        parsed = BatchPredictResponse(**resp.json())
        assert parsed.total == 1
        assert isinstance(parsed.results, list)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API imports work."""

    def test_import_batch_predict(self) -> None:
        from aortica.api import batch_predict

        assert hasattr(batch_predict, "BatchFileResult")
        assert hasattr(batch_predict, "BatchPredictResponse")
        assert hasattr(batch_predict, "run_batch_inference")
        assert hasattr(batch_predict, "DEFAULT_MAX_BATCH_SIZE")
