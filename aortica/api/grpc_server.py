"""gRPC server implementing the ECG Prediction Service.

Wraps the same inference pipeline as the REST API, providing the
``Predict`` and ``PredictBatch`` RPCs defined in ``ecg_service.proto``.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional gRPC dependency
# ---------------------------------------------------------------------------

try:
    import grpc

    HAS_GRPC = True
except ImportError:  # pragma: no cover
    HAS_GRPC = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_grpc() -> None:
    """Raise *ImportError* if grpcio is not installed."""
    if not HAS_GRPC:
        raise ImportError(
            "grpcio is required for the Aortica gRPC server. "
            "Install it with: pip install aortica[grpc]"
        )


# ---------------------------------------------------------------------------
# Servicer implementation
# ---------------------------------------------------------------------------


class ECGPredictionServicer:
    """gRPC servicer implementing ``ECGPredictionService``.

    Parameters
    ----------
    model:
        A loaded ``AorticaModel`` instance, or ``None`` (quality-only).
    conformal_predictor:
        A fitted ``ConformalPredictor``, or ``None``.
    enabled_tasks:
        Task heads to run.  Defaults to all four heads.
    max_batch_size:
        Maximum number of files in a single ``PredictBatch`` call.
    """

    def __init__(
        self,
        *,
        model: Any = None,
        conformal_predictor: Any = None,
        enabled_tasks: Optional[List[str]] = None,
        max_batch_size: int = 50,
    ) -> None:
        _check_grpc()
        self._model = model
        self._conformal_predictor = conformal_predictor
        self._enabled_tasks = enabled_tasks or [
            "rhythm",
            "structural",
            "ischaemia",
            "risk",
        ]
        self._max_batch_size = max_batch_size

    # -- internal helpers ---------------------------------------------------

    def _run_single_inference(
        self, file_data: bytes, filename: str, format_override: str
    ) -> Any:
        """Run the inference pipeline and return a ``PredictResponse``."""
        from aortica.api.predict import run_inference_pipeline

        return run_inference_pipeline(
            file_data,
            filename,
            format_override=format_override or None,
            model=self._model,
            conformal_predictor=self._conformal_predictor,
            enabled_tasks=self._enabled_tasks,
        )

    def _predict_response_to_proto(self, resp: Any) -> Any:
        """Convert a ``PredictResponse`` to a ``PredictReply`` protobuf."""
        from aortica.api import ecg_service_pb2 as pb2

        # -- quality report --
        per_lead = [
            pb2.LeadQuality(
                lead_name=lq.lead_name,
                score=lq.score,
                classification=lq.classification,
                flags=lq.flags,
            )
            for lq in resp.quality_report.per_lead
        ]
        quality = pb2.QualityReport(
            per_lead=per_lead,
            overall_score=resp.quality_report.overall_score,
            overall_classification=resp.quality_report.overall_classification,
            recommendation=resp.quality_report.recommendation,
        )

        # -- predictions --
        predictions = [
            pb2.TaskPrediction(
                task=tp.task,
                class_names=tp.class_names,
                probabilities=tp.probabilities,
            )
            for tp in resp.predictions
        ]

        # -- uncertainty (optional) --
        uncertainty = None
        if resp.uncertainty is not None:
            u = resp.uncertainty
            pred_sets = {}
            for task_name, pset_list in u.prediction_sets.items():
                # pset_list is a list of lists of ints
                # Flatten to single PredictionSet per task for proto
                flat: List[int] = []
                for s in pset_list:
                    flat.extend(s)
                pred_sets[task_name] = pb2.PredictionSet(indices=flat)

            conf_intervals = {}
            for task_name, ci_dict in u.confidence_intervals.items():
                conf_intervals[task_name] = pb2.ConfidenceInterval(
                    lower=ci_dict["lower"],
                    upper=ci_dict["upper"],
                )

            uncertainty = pb2.UncertaintyReport(
                prediction_sets=pred_sets,
                confidence_intervals=conf_intervals,
                ood_flag=u.ood_flag,
                entropy_score=u.entropy_score or 0.0,
            )

        return pb2.PredictReply(
            quality_report=quality,
            predictions=predictions,
            uncertainty=uncertainty,
        )

    # -- RPC implementations ------------------------------------------------

    def Predict(  # noqa: N802
        self, request: Any, context: Any
    ) -> Any:
        """Handle a single ECG prediction request."""
        from aortica.api import ecg_service_pb2 as pb2

        try:
            resp = self._run_single_inference(
                request.file_data,
                request.filename or "upload.dat",
                request.format_override,
            )
            return self._predict_response_to_proto(resp)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)  # type: ignore[union-attr]
            context.set_details(str(exc))
            return pb2.PredictReply()

    def PredictBatch(  # noqa: N802
        self, request: Any, context: Any
    ) -> Any:
        """Handle a batch ECG prediction request."""
        from aortica.api import ecg_service_pb2 as pb2

        files = list(request.files)
        if len(files) > self._max_batch_size:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)  # type: ignore[union-attr]
            context.set_details(
                f"Batch size {len(files)} exceeds maximum "
                f"allowed batch size of {self._max_batch_size}"
            )
            return pb2.PredictBatchReply()

        results: List[Any] = []
        succeeded = 0
        failed = 0

        for file_req in files:
            filename = file_req.filename or "upload.dat"
            try:
                resp = self._run_single_inference(
                    file_req.file_data,
                    filename,
                    file_req.format_override,
                )
                results.append(
                    pb2.BatchFileResult(
                        filename=filename,
                        status="success",
                        result=self._predict_response_to_proto(resp),
                    )
                )
                succeeded += 1
            except Exception as exc:
                results.append(
                    pb2.BatchFileResult(
                        filename=filename,
                        status="error",
                        error=str(exc),
                    )
                )
                failed += 1

        return pb2.PredictBatchReply(
            total=len(files),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


def serve(
    *,
    port: int = 50051,
    model: Any = None,
    conformal_predictor: Any = None,
    enabled_tasks: Optional[List[str]] = None,
    max_batch_size: int = 50,
    max_workers: int = 10,
) -> Any:
    """Start the gRPC ECG Prediction Service server.

    Parameters
    ----------
    port:
        Port number to bind the server to (default 50051).
    model:
        A loaded ``AorticaModel`` instance, or ``None``.
    conformal_predictor:
        A fitted ``ConformalPredictor``, or ``None``.
    enabled_tasks:
        Task heads to run.
    max_batch_size:
        Maximum files per batch request.
    max_workers:
        Size of the ``ThreadPoolExecutor``.

    Returns
    -------
    grpc.Server
        The started gRPC server instance.
    """
    _check_grpc()

    from aortica.api.ecg_service_pb2_grpc import (
        add_ECGPredictionServiceServicer_to_server,
    )

    servicer = ECGPredictionServicer(
        model=model,
        conformal_predictor=conformal_predictor,
        enabled_tasks=enabled_tasks,
        max_batch_size=max_batch_size,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))  # type: ignore[union-attr]
    add_ECGPredictionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server started on port %d", port)
    return server
