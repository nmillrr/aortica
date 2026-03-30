"""Batch ECG inference endpoint — ``POST /api/v1/predict/batch``.

Accepts multiple ECG file uploads and processes each through the full
inference pipeline (``read_ecg`` → ``denoise`` → ``score_quality`` →
model inference → uncertainty estimation).  Returns per-file results
with individual success/error status.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from aortica.api.predict import PredictResponse, run_inference_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_BATCH_SIZE: int = 50

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BatchFileResult(BaseModel):
    """Result for a single file in a batch prediction request."""

    filename: str = Field(..., description="Original filename")
    status: str = Field(
        ...,
        description="Processing status: 'success' or 'error'",
        examples=["success"],
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (only present when status is 'error')",
    )
    result: Optional[PredictResponse] = Field(
        default=None,
        description="Inference result (only present when status is 'success')",
    )


class BatchPredictResponse(BaseModel):
    """Full response from the batch ECG inference endpoint."""

    total: int = Field(..., description="Total number of files submitted")
    succeeded: int = Field(..., description="Number of files processed successfully")
    failed: int = Field(..., description="Number of files that failed processing")
    results: List[BatchFileResult] = Field(
        ..., description="Per-file inference results"
    )


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------


def run_batch_inference(
    files: List[tuple[bytes, str]],
    *,
    format_override: Optional[str] = None,
    model: Any = None,
    conformal_predictor: Any = None,
    enabled_tasks: Optional[List[str]] = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> BatchPredictResponse:
    """Process multiple ECG files through the inference pipeline.

    Parameters
    ----------
    files:
        List of ``(file_bytes, filename)`` tuples.
    format_override:
        Explicit format string for all files.
    model:
        A loaded ``AorticaModel`` instance, or ``None``.
    conformal_predictor:
        A fitted ``ConformalPredictor``, or ``None``.
    enabled_tasks:
        Task heads to run.
    max_batch_size:
        Maximum number of files to accept (default 50).

    Returns
    -------
    BatchPredictResponse
        Aggregated results with per-file status.

    Raises
    ------
    ValueError
        If the number of files exceeds ``max_batch_size``.
    """
    if len(files) > max_batch_size:
        raise ValueError(
            f"Batch size {len(files)} exceeds maximum allowed "
            f"batch size of {max_batch_size}"
        )

    results: List[BatchFileResult] = []
    succeeded = 0
    failed = 0

    for file_bytes, filename in files:
        try:
            result = run_inference_pipeline(
                file_bytes,
                filename,
                format_override=format_override,
                model=model,
                conformal_predictor=conformal_predictor,
                enabled_tasks=enabled_tasks,
            )
            results.append(
                BatchFileResult(
                    filename=filename,
                    status="success",
                    result=result,
                )
            )
            succeeded += 1
        except Exception as exc:
            results.append(
                BatchFileResult(
                    filename=filename,
                    status="error",
                    error=str(exc),
                )
            )
            failed += 1

    return BatchPredictResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
