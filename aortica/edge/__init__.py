"""Edge deployment: ONNX export, quantization, and edge-optimized models."""

from aortica.edge.onnx_export import export_onnx, validate_onnx

__all__ = [
    "export_onnx",
    "validate_onnx",
]
