"""Edge deployment: ONNX export, quantization, and edge-optimized models."""

from aortica.edge.deploy_profiles import (
    RaspberryPiProfile,
    generate_pi_image_script,
    generate_systemd_service,
    write_pi_image_script,
    write_systemd_service,
)
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
from aortica.edge.quantization import (
    ECGCalibrationDataReader,
    QuantizationReport,
    quantize_int8,
)
from aortica.edge.validation import (
    EdgeValidationReport,
    TaskValidation,
    validate_edge,
)

__all__ = [
    "DistillationConfig",
    "DistillationEpochMetrics",
    "DistillationResult",
    "ECGCalibrationDataReader",
    "EdgeValidationReport",
    "MobileNetBackbone1D",
    "QuantizationReport",
    "RaspberryPiProfile",
    "TaskValidation",
    "distillation_loss_classification",
    "distillation_loss_regression",
    "export_onnx",
    "generate_pi_image_script",
    "generate_systemd_service",
    "quantize_int8",
    "train_distillation",
    "validate_edge",
    "validate_onnx",
    "write_pi_image_script",
    "write_systemd_service",
]
