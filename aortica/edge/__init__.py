"""Edge deployment: ONNX export, quantization, and edge-optimized models."""

from aortica.edge.distillation import (
    DistillationConfig,
    DistillationEpochMetrics,
    DistillationResult,
    distillation_loss_classification,
    distillation_loss_regression,
    train_distillation,
)
from aortica.edge.mobilenet_backbone import MobileNetBackbone1D
from aortica.edge.onnx_export import export_onnx, validate_onnx

__all__ = [
    "DistillationConfig",
    "DistillationEpochMetrics",
    "DistillationResult",
    "MobileNetBackbone1D",
    "distillation_loss_classification",
    "distillation_loss_regression",
    "export_onnx",
    "train_distillation",
    "validate_onnx",
]
