"""Tests for the gRPC ECG Prediction Service.

Covers:
  - proto message construction and serialisation
  - ECGPredictionServicer (Predict / PredictBatch)
  - serve() startup function
  - Error handling (bad files, batch size exceeded)
  - End-to-end test via gRPC test channel
"""

from __future__ import annotations

import importlib
import struct
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# These tests require grpcio — skip the whole module if missing.
grpc = pytest.importorskip("grpc")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_wfdb_bytes() -> bytes:
    """Return minimal bytes that look like a WFDB .hea + .dat pair.

    Since the inference pipeline writes to a temp file and delegates to
    ``read_ecg()``, we create a minimal valid ECG as raw bytes that the
    pipeline can consume (using CSV format for simplicity).
    """
    # CSV with 12 leads, 500 samples at 500 Hz = 1 second
    lines = ["lead_I,lead_II,lead_III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6"]
    rng = np.random.default_rng(42)
    for _ in range(500):
        vals = rng.normal(0, 100, 12)
        lines.append(",".join(f"{v:.2f}" for v in vals))
    return "\n".join(lines).encode("utf-8")


@pytest.fixture()
def synthetic_csv_bytes() -> bytes:
    """Synthetic CSV ECG file bytes."""
    return _make_synthetic_wfdb_bytes()


@pytest.fixture()
def pb2() -> Any:
    """Import and return the protobuf module."""
    return importlib.import_module("aortica.api.ecg_service_pb2")


@pytest.fixture()
def pb2_grpc() -> Any:
    """Import and return the gRPC stub module."""
    return importlib.import_module("aortica.api.ecg_service_pb2_grpc")


@pytest.fixture()
def mock_predict_response() -> Any:
    """Create a mock PredictResponse for testing."""
    from aortica.api.predict import (
        LeadQualityResponse,
        PredictResponse,
        QualityReportResponse,
        TaskPrediction,
    )

    quality = QualityReportResponse(
        per_lead=[
            LeadQualityResponse(
                lead_name="II",
                score=85.0,
                classification="good",
                flags=[],
            )
        ],
        overall_score=85.0,
        overall_classification="good",
        recommendation="accept",
    )
    predictions = [
        TaskPrediction(
            task="rhythm",
            class_names=["AF", "NSR"],
            probabilities=[0.1, 0.9],
        ),
    ]
    return PredictResponse(
        quality_report=quality,
        predictions=predictions,
        uncertainty=None,
    )


# ---------------------------------------------------------------------------
# Protobuf message tests
# ---------------------------------------------------------------------------


class TestProtobufMessages:
    """Tests for protobuf message construction."""

    def test_predict_request_construction(self, pb2: Any) -> None:
        req = pb2.PredictRequest(
            file_data=b"test_data",
            filename="test.csv",
            format_override="csv",
        )
        assert req.file_data == b"test_data"
        assert req.filename == "test.csv"
        assert req.format_override == "csv"

    def test_predict_request_defaults(self, pb2: Any) -> None:
        req = pb2.PredictRequest()
        assert req.file_data == b""
        assert req.filename == ""
        assert req.format_override == ""

    def test_predict_batch_request(self, pb2: Any) -> None:
        files = [
            pb2.PredictRequest(file_data=b"a", filename="a.csv"),
            pb2.PredictRequest(file_data=b"b", filename="b.csv"),
        ]
        req = pb2.PredictBatchRequest(files=files)
        assert len(req.files) == 2
        assert req.files[0].filename == "a.csv"

    def test_lead_quality_message(self, pb2: Any) -> None:
        lq = pb2.LeadQuality(
            lead_name="II",
            score=90.5,
            classification="good",
            flags=["clean"],
        )
        assert lq.lead_name == "II"
        assert abs(lq.score - 90.5) < 0.01
        assert lq.classification == "good"
        assert list(lq.flags) == ["clean"]

    def test_quality_report_message(self, pb2: Any) -> None:
        qr = pb2.QualityReport(
            overall_score=85.0,
            overall_classification="good",
            recommendation="accept",
        )
        assert abs(qr.overall_score - 85.0) < 0.01
        assert qr.recommendation == "accept"

    def test_task_prediction_message(self, pb2: Any) -> None:
        tp = pb2.TaskPrediction(
            task="rhythm",
            class_names=["AF", "NSR"],
            probabilities=[0.2, 0.8],
        )
        assert tp.task == "rhythm"
        assert list(tp.class_names) == ["AF", "NSR"]
        assert len(tp.probabilities) == 2

    def test_predict_reply_construction(self, pb2: Any) -> None:
        reply = pb2.PredictReply(
            quality_report=pb2.QualityReport(
                overall_score=80.0,
                overall_classification="good",
                recommendation="accept",
            ),
            predictions=[
                pb2.TaskPrediction(task="rhythm", probabilities=[0.5])
            ],
        )
        assert reply.quality_report.overall_score == 80.0
        assert len(reply.predictions) == 1

    def test_batch_file_result(self, pb2: Any) -> None:
        result = pb2.BatchFileResult(
            filename="test.csv",
            status="success",
        )
        assert result.filename == "test.csv"
        assert result.status == "success"
        assert result.error == ""  # default empty

    def test_batch_predict_reply(self, pb2: Any) -> None:
        reply = pb2.PredictBatchReply(
            total=3,
            succeeded=2,
            failed=1,
            results=[
                pb2.BatchFileResult(filename="a.csv", status="success"),
                pb2.BatchFileResult(filename="b.csv", status="success"),
                pb2.BatchFileResult(filename="c.csv", status="error", error="bad"),
            ],
        )
        assert reply.total == 3
        assert reply.succeeded == 2
        assert reply.failed == 1
        assert len(reply.results) == 3

    def test_serialise_roundtrip(self, pb2: Any) -> None:
        req = pb2.PredictRequest(
            file_data=b"payload",
            filename="test.dat",
            format_override="csv",
        )
        serialised = req.SerializeToString()
        restored = pb2.PredictRequest()
        restored.ParseFromString(serialised)
        assert restored.file_data == b"payload"
        assert restored.filename == "test.dat"

    def test_uncertainty_report_message(self, pb2: Any) -> None:
        ur = pb2.UncertaintyReport(
            ood_flag=True,
            entropy_score=0.5,
            prediction_sets={
                "rhythm": pb2.PredictionSet(indices=[0, 1, 2])
            },
            confidence_intervals={
                "risk": pb2.ConfidenceInterval(
                    lower=[0.1, 0.2], upper=[0.8, 0.9]
                )
            },
        )
        assert ur.ood_flag is True
        assert abs(ur.entropy_score - 0.5) < 0.01
        assert list(ur.prediction_sets["rhythm"].indices) == [0, 1, 2]
        ci = ur.confidence_intervals["risk"]
        assert len(ci.lower) == 2


# ---------------------------------------------------------------------------
# Servicer unit tests (mocked pipeline)
# ---------------------------------------------------------------------------


class TestECGPredictionServicer:
    """Test the servicer implementation with mocked inference."""

    def test_construction(self) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        s = ECGPredictionServicer()
        assert s._max_batch_size == 50
        assert s._enabled_tasks == ["rhythm", "structural", "ischaemia", "risk"]

    def test_construction_custom(self) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        s = ECGPredictionServicer(
            enabled_tasks=["rhythm"],
            max_batch_size=10,
        )
        assert s._max_batch_size == 10
        assert s._enabled_tasks == ["rhythm"]

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_success(
        self, mock_inference: Any, pb2: Any, mock_predict_response: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        mock_inference.return_value = mock_predict_response

        servicer = ECGPredictionServicer()
        request = pb2.PredictRequest(
            file_data=b"test",
            filename="test.csv",
        )
        context = MagicMock()

        reply = servicer.Predict(request, context)
        assert reply.quality_report.overall_score == 85.0
        assert reply.quality_report.recommendation == "accept"
        assert len(reply.predictions) == 1
        assert reply.predictions[0].task == "rhythm"
        context.set_code.assert_not_called()

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_error(self, mock_inference: Any, pb2: Any) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        mock_inference.side_effect = ValueError("Bad format")

        servicer = ECGPredictionServicer()
        request = pb2.PredictRequest(file_data=b"x", filename="bad.xyz")
        context = MagicMock()

        reply = servicer.Predict(request, context)
        context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details.assert_called_once_with("Bad format")

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_batch_success(
        self, mock_inference: Any, pb2: Any, mock_predict_response: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        mock_inference.return_value = mock_predict_response

        servicer = ECGPredictionServicer()
        request = pb2.PredictBatchRequest(
            files=[
                pb2.PredictRequest(file_data=b"a", filename="a.csv"),
                pb2.PredictRequest(file_data=b"b", filename="b.csv"),
            ]
        )
        context = MagicMock()

        reply = servicer.PredictBatch(request, context)
        assert reply.total == 2
        assert reply.succeeded == 2
        assert reply.failed == 0
        assert len(reply.results) == 2
        assert reply.results[0].status == "success"
        assert reply.results[0].filename == "a.csv"

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_batch_mixed(
        self, mock_inference: Any, pb2: Any, mock_predict_response: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        mock_inference.side_effect = [
            mock_predict_response,
            ValueError("Failed"),
        ]

        servicer = ECGPredictionServicer()
        request = pb2.PredictBatchRequest(
            files=[
                pb2.PredictRequest(file_data=b"a", filename="good.csv"),
                pb2.PredictRequest(file_data=b"b", filename="bad.xyz"),
            ]
        )
        context = MagicMock()

        reply = servicer.PredictBatch(request, context)
        assert reply.total == 2
        assert reply.succeeded == 1
        assert reply.failed == 1
        assert reply.results[0].status == "success"
        assert reply.results[1].status == "error"
        assert "Failed" in reply.results[1].error

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_batch_exceeds_max_size(
        self, mock_inference: Any, pb2: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        servicer = ECGPredictionServicer(max_batch_size=2)
        files = [
            pb2.PredictRequest(file_data=b"x", filename=f"{i}.csv")
            for i in range(5)
        ]
        request = pb2.PredictBatchRequest(files=files)
        context = MagicMock()

        reply = servicer.PredictBatch(request, context)
        context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        assert "exceeds maximum" in context.set_details.call_args[0][0]

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_default_filename(
        self, mock_inference: Any, pb2: Any, mock_predict_response: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        mock_inference.return_value = mock_predict_response

        servicer = ECGPredictionServicer()
        # No filename — should default to "upload.dat"
        request = pb2.PredictRequest(file_data=b"data")
        context = MagicMock()

        servicer.Predict(request, context)
        # Check the pipeline was called with default filename
        call_args = mock_inference.call_args
        assert call_args[0][1] == "upload.dat"

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_predict_batch_empty(
        self, mock_inference: Any, pb2: Any
    ) -> None:
        from aortica.api.grpc_server import ECGPredictionServicer

        servicer = ECGPredictionServicer()
        request = pb2.PredictBatchRequest(files=[])
        context = MagicMock()

        reply = servicer.PredictBatch(request, context)
        assert reply.total == 0
        assert reply.succeeded == 0
        assert reply.failed == 0


# ---------------------------------------------------------------------------
# Server startup tests
# ---------------------------------------------------------------------------


class TestServeFunction:
    """Test the serve() startup function."""

    def test_serve_returns_server(self) -> None:
        from aortica.api.grpc_server import serve

        server = serve(port=0)  # port=0 lets OS pick a free port
        assert server is not None
        server.stop(grace=0)

    def test_serve_custom_port(self) -> None:
        from aortica.api.grpc_server import serve

        # Use a high port that's likely free
        server = serve(port=50099)
        assert server is not None
        server.stop(grace=0)

    def test_serve_with_custom_config(self) -> None:
        from aortica.api.grpc_server import serve

        server = serve(
            port=0,
            enabled_tasks=["rhythm"],
            max_batch_size=5,
            max_workers=2,
        )
        assert server is not None
        server.stop(grace=0)


# ---------------------------------------------------------------------------
# End-to-end test via gRPC test channel
# ---------------------------------------------------------------------------


class TestGRPCChannel:
    """End-to-end tests using a real gRPC in-process server and channel."""

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_e2e_predict(
        self, mock_inference: Any, pb2: Any, pb2_grpc: Any, mock_predict_response: Any
    ) -> None:
        mock_inference.return_value = mock_predict_response

        from aortica.api.grpc_server import ECGPredictionServicer

        servicer = ECGPredictionServicer()

        server = grpc.server(
            __import__("concurrent").futures.ThreadPoolExecutor(max_workers=2)
        )
        pb2_grpc.add_ECGPredictionServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        server.start()

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            stub = pb2_grpc.ECGPredictionServiceStub(channel)

            request = pb2.PredictRequest(
                file_data=b"test_ecg",
                filename="test.csv",
            )
            reply = stub.Predict(request)

            assert reply.quality_report.overall_score == 85.0
            assert len(reply.predictions) == 1
            assert reply.predictions[0].task == "rhythm"
            channel.close()
        finally:
            server.stop(grace=0)

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_e2e_predict_batch(
        self, mock_inference: Any, pb2: Any, pb2_grpc: Any, mock_predict_response: Any
    ) -> None:
        mock_inference.return_value = mock_predict_response

        from aortica.api.grpc_server import ECGPredictionServicer

        servicer = ECGPredictionServicer()

        server = grpc.server(
            __import__("concurrent").futures.ThreadPoolExecutor(max_workers=2)
        )
        pb2_grpc.add_ECGPredictionServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        server.start()

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            stub = pb2_grpc.ECGPredictionServiceStub(channel)

            request = pb2.PredictBatchRequest(
                files=[
                    pb2.PredictRequest(file_data=b"a", filename="a.csv"),
                    pb2.PredictRequest(file_data=b"b", filename="b.csv"),
                ]
            )
            reply = stub.PredictBatch(request)

            assert reply.total == 2
            assert reply.succeeded == 2
            assert reply.failed == 0
            assert len(reply.results) == 2
            channel.close()
        finally:
            server.stop(grace=0)

    @patch("aortica.api.grpc_server.ECGPredictionServicer._run_single_inference")
    def test_e2e_predict_error_returns_grpc_error(
        self, mock_inference: Any, pb2: Any, pb2_grpc: Any
    ) -> None:
        mock_inference.side_effect = ValueError("Unsupported format")

        from aortica.api.grpc_server import ECGPredictionServicer

        servicer = ECGPredictionServicer()

        server = grpc.server(
            __import__("concurrent").futures.ThreadPoolExecutor(max_workers=2)
        )
        pb2_grpc.add_ECGPredictionServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        server.start()

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            stub = pb2_grpc.ECGPredictionServiceStub(channel)

            request = pb2.PredictRequest(
                file_data=b"bad",
                filename="bad.xyz",
            )
            # gRPC error should be returned as an RpcError
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Predict(request)
            assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "Unsupported format" in exc_info.value.details()
            channel.close()
        finally:
            server.stop(grace=0)


# ---------------------------------------------------------------------------
# Import / module tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API is importable."""

    def test_grpc_server_importable(self) -> None:
        from aortica.api import grpc_server

        assert hasattr(grpc_server, "ECGPredictionServicer")
        assert hasattr(grpc_server, "serve")

    def test_proto_modules_importable(self) -> None:
        from aortica.api import ecg_service_pb2, ecg_service_pb2_grpc

        assert hasattr(ecg_service_pb2, "PredictRequest")
        assert hasattr(ecg_service_pb2, "PredictReply")
        assert hasattr(ecg_service_pb2_grpc, "ECGPredictionServiceStub")
        assert hasattr(ecg_service_pb2_grpc, "ECGPredictionServiceServicer")

    def test_has_grpc_flag(self) -> None:
        from aortica.api.grpc_server import HAS_GRPC

        assert HAS_GRPC is True  # grpc is installed in test env
