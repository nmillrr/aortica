"""Android mobile model export utility.

Packages the INT8 ONNX edge model with Android-specific metadata
(input/output tensor names, shape expectations, version tag) for
consumption by the Aortica Android app (US-061c).

The exported bundle is a directory containing:

- ``model.onnx`` — the INT8 quantized ONNX model
- ``model_metadata.json`` — metadata for the Android inference pipeline

Example::

    from aortica.edge.mobile_export import export_mobile_model

    export_mobile_model(
        model_path="aortica_edge_int8.onnx",
        output_path="mobile_bundle/",
        version="0.3.0",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default task heads expected in the edge model.
DEFAULT_TASK_HEADS: list[dict[str, object]] = [
    {
        "name": "rhythm",
        "output_name": "rhythm_output",
        "num_classes": 28,
        "type": "classification",
    },
    {
        "name": "structural",
        "output_name": "structural_output",
        "num_classes": 19,
        "type": "classification",
    },
    {
        "name": "ischaemia",
        "output_name": "ischaemia_output",
        "num_classes": 19,
        "type": "classification",
    },
    {
        "name": "risk",
        "output_name": "risk_output",
        "num_classes": 6,
        "type": "regression",
    },
]

#: Input tensor specification for the edge model.
DEFAULT_INPUT_SPEC: dict[str, object] = {
    "name": "ecg_input",
    "shape": [1, 12, 5000],
    "dtype": "float32",
    "description": "ECG signal: [batch, leads, samples] at 500 Hz",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MobileModelMetadata:
    """Metadata describing the ONNX model for the Android app."""

    version: str
    model_filename: str = "model.onnx"
    sha256: str = ""
    file_size_bytes: int = 0
    input_spec: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_INPUT_SPEC))
    task_heads: list[dict[str, object]] = field(
        default_factory=lambda: [dict(d) for d in DEFAULT_TASK_HEADS]
    )
    sample_rate_hz: int = 500
    supported_lead_configs: list[str] = field(
        default_factory=lambda: ["12-lead", "6-lead-limb", "single-lead-I", "single-lead-II"]
    )
    min_app_version: str = "1.0.0"
    quantization: str = "INT8"
    opset_version: int = 17

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dictionary suitable for JSON encoding."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MobileModelMetadata:
        """Deserialize from a dictionary."""
        filtered: dict[str, Any] = {
            k: v for k, v in data.items() if k in cls.__dataclass_fields__
        }
        return cls(**filtered)  # type: ignore[arg-type]


@dataclass
class MobileExportResult:
    """Result of a mobile model export operation."""

    output_dir: pathlib.Path
    model_path: pathlib.Path
    metadata_path: pathlib.Path
    metadata: MobileModelMetadata
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------


def _compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _validate_onnx_model(model_path: pathlib.Path) -> dict[str, object]:
    """Validate and inspect an ONNX model, returning input/output specs.

    Returns a dict with ``input_spec`` and ``output_specs`` keys.
    If onnxruntime is not available, returns defaults without validation.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        input_info = inputs[0]
        input_spec = {
            "name": input_info.name,
            "shape": input_info.shape,
            "dtype": input_info.type,
            "description": "ECG signal: [batch, leads, samples] at 500 Hz",
        }

        output_specs = []
        for out in outputs:
            output_specs.append(
                {
                    "name": out.name,
                    "shape": out.shape,
                    "dtype": out.type,
                }
            )

        return {"input_spec": input_spec, "output_specs": output_specs}
    except ImportError:
        logger.warning(
            "onnxruntime not available — skipping model validation, "
            "using default input/output specs"
        )
        return {"input_spec": dict(DEFAULT_INPUT_SPEC), "output_specs": []}
    except Exception as exc:
        logger.warning("ONNX model validation failed: %s", exc)
        return {"input_spec": dict(DEFAULT_INPUT_SPEC), "output_specs": []}


def export_mobile_model(
    model_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    version: str = "0.1.0",
    min_app_version: str = "1.0.0",
    task_heads: Optional[list[dict[str, object]]] = None,
    sample_rate_hz: int = 500,
) -> MobileExportResult:
    """Package an INT8 ONNX model with Android-specific metadata.

    Copies the ONNX model to ``output_path/model.onnx`` and writes
    ``output_path/model_metadata.json`` with tensor names, shapes,
    version, and integrity hash.

    Args:
        model_path: Path to the source INT8 ONNX model file.
        output_path: Destination directory for the mobile bundle.
        version: Semantic version tag for this model release.
        min_app_version: Minimum Android app version that supports this model.
        task_heads: Override task head definitions.  Defaults to the
            standard 4-head configuration.
        sample_rate_hz: Expected input sample rate.  Default ``500``.

    Returns:
        :class:`MobileExportResult` with paths and metadata.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *model_path* is not an ``.onnx`` file.
    """
    model_path = pathlib.Path(model_path)
    output_path = pathlib.Path(output_path)

    # ---- Validate source ----
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected .onnx file, got: {model_path.suffix}")

    # ---- Prepare output directory ----
    output_path.mkdir(parents=True, exist_ok=True)
    dest_model = output_path / "model.onnx"

    # Copy model
    shutil.copy2(str(model_path), str(dest_model))
    logger.info("Copied ONNX model to %s", dest_model)

    # ---- Compute integrity hash ----
    sha256 = _compute_sha256(dest_model)
    file_size = dest_model.stat().st_size

    # ---- Inspect model ----
    model_info = _validate_onnx_model(dest_model)

    # ---- Build metadata ----
    metadata = MobileModelMetadata(
        version=version,
        model_filename="model.onnx",
        sha256=sha256,
        file_size_bytes=file_size,
        input_spec=model_info["input_spec"],  # type: ignore[arg-type]
        task_heads=task_heads if task_heads is not None else [dict(d) for d in DEFAULT_TASK_HEADS],
        sample_rate_hz=sample_rate_hz,
        min_app_version=min_app_version,
    )

    # ---- Write metadata ----
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2, default=str)
    logger.info("Wrote model metadata to %s", metadata_path)

    return MobileExportResult(
        output_dir=output_path.resolve(),
        model_path=dest_model.resolve(),
        metadata_path=metadata_path.resolve(),
        metadata=metadata,
    )
