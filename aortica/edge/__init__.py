"""Edge deployment: ONNX export, quantization, and edge-optimized models."""

from aortica.edge.deploy_profiles import (
    DutyCycledModelLoader,
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
from aortica.edge.hardware_benchmark import (
    HardwareBenchmarkReport,
    MetricResult,
    PlatformProfile,
    benchmark_all_platforms,
    consolidated_csv,
    consolidated_markdown_table,
    hardware_benchmark,
    load_platform_profiles,
)
from aortica.edge.mobilenet_backbone import MobileNetBackbone1D
from aortica.edge.onnx_export import export_onnx, validate_onnx
from aortica.edge.power_validation import (
    PowerValidationReport,
    compute_sustained_power,
    validate_power_consumption,
)
from aortica.edge.profiling import InferenceProfile, profile_inference
from aortica.edge.quantization import (
    ECGCalibrationDataReader,
    QuantizationReport,
    quantize_int8,
)
from aortica.edge.simplified_output import (
    KeyFinding,
    SimplifiedReport,
    TierThresholds,
    load_locale,
    simplify_output,
)
from aortica.edge.site_monitor import SiteMonitor, SiteStatus
from aortica.edge.validation import (
    EdgeValidationReport,
    TaskValidation,
    validate_edge,
)

__all__ = [
    "DutyCycledModelLoader",
    "PowerValidationReport",
    "SiteMonitor",
    "SiteStatus",
    "compute_sustained_power",
    "validate_power_consumption",
    "DistillationConfig",
    "DistillationEpochMetrics",
    "DistillationResult",
    "ECGCalibrationDataReader",
    "EdgeValidationReport",
    "HardwareBenchmarkReport",
    "InferenceProfile",
    "KeyFinding",
    "MetricResult",
    "MobileNetBackbone1D",
    "PlatformProfile",
    "QuantizationReport",
    "RaspberryPiProfile",
    "SimplifiedReport",
    "TaskValidation",
    "TierThresholds",
    "benchmark_all_platforms",
    "consolidated_csv",
    "consolidated_markdown_table",
    "distillation_loss_classification",
    "distillation_loss_regression",
    "export_onnx",
    "generate_pi_image_script",
    "generate_systemd_service",
    "hardware_benchmark",
    "load_locale",
    "load_platform_profiles",
    "profile_inference",
    "quantize_int8",
    "simplify_output",
    "train_distillation",
    "validate_edge",
    "validate_onnx",
    "write_pi_image_script",
    "write_systemd_service",
]
