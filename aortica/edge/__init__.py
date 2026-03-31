"""Edge deployment: ONNX export, quantization, and edge-optimized models."""

from aortica.edge.mobilenet_backbone import MobileNetBackbone1D
from aortica.edge.onnx_export import export_onnx, validate_onnx

__all__ = [
    "MobileNetBackbone1D",
    "export_onnx",
    "validate_onnx",
]
