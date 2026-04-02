"""INT8 quantization pipeline for ONNX edge models.

Provides :func:`quantize_int8` to apply static INT8 quantization to an
exported ONNX model using representative ECG calibration data.

After quantization, the function validates that the quantized model runs
successfully via ONNX Runtime and reports the maximum absolute difference
between the original and quantized model outputs.

Example::

    import numpy as np
    from aortica.edge import quantize_int8

    # Generate representative calibration data (or use real ECG samples)
    calibration_data = [np.random.randn(1, 12, 5000).astype(np.float32) for _ in range(100)]
    report = quantize_int8("model.onnx", calibration_data, "model_int8.onnx")
    print(f"Max diff: {report.max_abs_diff:.6f}")

"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

    # Dummy base class so ECGCalibrationDataReader can be defined at import
    # time even when onnxruntime.quantization is not installed.
    class CalibrationDataReader:  # type: ignore[no-redef]
        """Placeholder when onnxruntime.quantization is missing."""

        def get_next(self) -> None:
            return None


logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Ensure onnxruntime and quantization tools are available."""
    if not HAS_ONNXRUNTIME:
        raise ImportError(
            "onnxruntime is required for INT8 quantization. "
            "Install with: pip install aortica[edge]"
        )
    if not HAS_QUANTIZATION:
        raise ImportError(
            "onnxruntime.quantization is required for INT8 quantization. "
            "Install with: pip install onnxruntime"
        )


# ---------------------------------------------------------------------------
# Calibration Data Reader
# ---------------------------------------------------------------------------


class ECGCalibrationDataReader(CalibrationDataReader):
    """Feeds representative ECG samples to the ONNX Runtime quantizer.

    Each call to :meth:`get_next` returns a dictionary mapping the model's
    input name to a single calibration sample (numpy array).

    Args:
        calibration_data: Sequence of numpy arrays, each of shape
            ``[1, leads, samples]`` (float32).
        input_name: Name of the ONNX model's input tensor.
            Default ``"ecg_input"`` (matching :func:`export_onnx`).
    """

    def __init__(
        self,
        calibration_data: Sequence[np.ndarray],
        input_name: str = "ecg_input",
    ) -> None:
        super().__init__()
        self.calibration_data = list(calibration_data)
        self.input_name = input_name
        self._iter = iter(self.calibration_data)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next calibration sample or ``None`` when exhausted."""
        sample = next(self._iter, None)
        if sample is None:
            return None
        # Ensure float32 and batch dimension
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # add batch dim
        return {self.input_name: arr}

    def rewind(self) -> None:
        """Reset the iterator to the beginning."""
        self._iter = iter(self.calibration_data)


# ---------------------------------------------------------------------------
# Quantization Report
# ---------------------------------------------------------------------------


@dataclass
class QuantizationReport:
    """Results from :func:`quantize_int8`.

    Attributes:
        output_path: Resolved path to the quantized ONNX model.
        original_path: Path to the original (non-quantized) ONNX model.
        num_calibration_samples: Number of calibration samples used.
        max_abs_diff: Maximum absolute difference between original and
            quantized outputs across all output tensors.
        per_output_max_diff: Per-output-tensor maximum absolute differences.
        success: Whether the quantized model runs successfully.
    """

    output_path: pathlib.Path
    original_path: pathlib.Path
    num_calibration_samples: int
    max_abs_diff: float = 0.0
    per_output_max_diff: dict[str, float] = field(default_factory=dict)
    success: bool = True


# ---------------------------------------------------------------------------
# Main quantization function
# ---------------------------------------------------------------------------


def quantize_int8(
    onnx_model_path: Union[str, pathlib.Path],
    calibration_data: Sequence[np.ndarray],
    output_path: Union[str, pathlib.Path],
    num_calibration_samples: Optional[int] = None,
    input_name: str = "ecg_input",
) -> QuantizationReport:
    """Quantize an ONNX model to INT8 using static quantization.

    Applies static quantization with the QDQ (QuantizeLinear/DequantizeLinear)
    format using the provided calibration data to determine scale and
    zero-point parameters.

    After quantization, the function validates that the quantized model runs
    successfully via ONNX Runtime and compares outputs against the original
    model.

    Args:
        onnx_model_path: Path to the original ONNX model.
        calibration_data: Sequence of numpy arrays, each of shape
            ``[1, leads, samples]`` (or ``[leads, samples]``).  Used as
            representative inputs for calibration.
        output_path: Destination path for the quantized ``.onnx`` model.
        num_calibration_samples: Maximum number of calibration samples to use.
            If ``None``, uses all provided samples.  Default ``None``
            (the caller can pass up to the default 100 or any count).
        input_name: Name of the ONNX model's input tensor.
            Default ``"ecg_input"``.

    Returns:
        A :class:`QuantizationReport` with paths, sample count, and
        max absolute difference between original and quantized outputs.

    Raises:
        ImportError: If onnxruntime or its quantization tools are missing.
        FileNotFoundError: If ``onnx_model_path`` does not exist.
        ValueError: If ``calibration_data`` is empty.
    """
    _check_dependencies()

    onnx_model_path = pathlib.Path(onnx_model_path)
    output_path = pathlib.Path(output_path)

    if not onnx_model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_model_path}"
        )

    if len(calibration_data) == 0:
        raise ValueError("calibration_data must not be empty.")

    # Limit calibration samples if requested
    samples = list(calibration_data)
    if num_calibration_samples is not None and num_calibration_samples > 0:
        samples = samples[:num_calibration_samples]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create calibration data reader
    reader = ECGCalibrationDataReader(samples, input_name=input_name)

    # Run static quantization
    logger.info(
        "Quantizing %s with %d calibration samples → %s",
        onnx_model_path,
        len(samples),
        output_path,
    )

    quantize_static(
        model_input=str(onnx_model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    # Validate the quantized model
    report = QuantizationReport(
        output_path=output_path.resolve(),
        original_path=onnx_model_path.resolve(),
        num_calibration_samples=len(samples),
    )

    try:
        report = _validate_quantized_model(
            onnx_model_path, output_path, samples, input_name, report,
        )
    except Exception as exc:
        logger.error("Quantized model validation failed: %s", exc)
        report.success = False

    return report


def _validate_quantized_model(
    original_path: pathlib.Path,
    quantized_path: pathlib.Path,
    calibration_samples: list[np.ndarray],
    input_name: str,
    report: QuantizationReport,
) -> QuantizationReport:
    """Compare original and quantized model outputs.

    Uses the first calibration sample as a test input.
    """
    # Create sessions for both models
    original_session = ort.InferenceSession(str(original_path))
    quantized_session = ort.InferenceSession(str(quantized_path))

    # Use first calibration sample for comparison
    test_sample = np.asarray(calibration_samples[0], dtype=np.float32)
    if test_sample.ndim == 2:
        test_sample = test_sample[np.newaxis, ...]

    # Run both models
    original_outputs = original_session.run(None, {input_name: test_sample})
    quantized_outputs = quantized_session.run(None, {input_name: test_sample})

    # Get output names
    output_names = [
        out.name for out in original_session.get_outputs()
    ]

    # Compare outputs
    max_diff_overall = 0.0
    per_output_diffs: dict[str, float] = {}

    for i, name in enumerate(output_names):
        orig_arr = original_outputs[i]
        quant_arr = quantized_outputs[i]
        diff = float(np.max(np.abs(orig_arr - quant_arr)))
        per_output_diffs[name] = diff
        max_diff_overall = max(max_diff_overall, diff)
        logger.info(
            "Output '%s': max abs diff = %.6f", name, diff,
        )

    report.max_abs_diff = max_diff_overall
    report.per_output_max_diff = per_output_diffs
    report.success = True

    logger.info(
        "Quantization validation: max absolute diff = %.6f",
        max_diff_overall,
    )

    return report
