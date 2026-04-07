"""Deep learning models: backbone, task heads, training pipelines."""

from aortica.models.aortica_model import AorticaModel, MultiTaskOutput
from aortica.models.attention import CrossLeadAttention
from aortica.models.backbone import AorticaBackbone
from aortica.models.conformal_prediction import ConformalPredictor, UncertaintyReport
from aortica.models.ischaemia_head import IschaemiaHead
from aortica.models.registry import (
    ChecksumError,
    ModelNotFoundError,
    PretrainedModelInfo,
    get_model_info,
    list_available_versions,
    load_pretrained,
)
from aortica.models.resnet1d import ResNet1D
from aortica.models.rhythm_head import RhythmHead
from aortica.models.risk_head import RiskHead
from aortica.models.structural_head import StructuralHead
from aortica.models.temperature_scaling import (
    CalibratedModel,
    TemperatureScaling,
)

__all__ = [
    "AorticaBackbone",
    "AorticaModel",
    "CalibratedModel",
    "ChecksumError",
    "ConformalPredictor",
    "CrossLeadAttention",
    "IschaemiaHead",
    "ModelNotFoundError",
    "MultiTaskOutput",
    "PretrainedModelInfo",
    "ResNet1D",
    "RhythmHead",
    "RiskHead",
    "StructuralHead",
    "TemperatureScaling",
    "UncertaintyReport",
    "get_model_info",
    "list_available_versions",
    "load_pretrained",
]
