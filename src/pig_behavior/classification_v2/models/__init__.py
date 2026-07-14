"""Model components for classification_v2."""

from pig_behavior.classification_v2.models.model_factory import (
    MODEL_MODE_NAMES,
    build_multimodal_model,
    model_mode_contract,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    SUPPORTED_VISUAL_BACKBONES,
    VISUAL_BACKBONE_CONTRACT_VERSION,
    visual_backbone_contract,
)

__all__ = [
    "MODEL_MODE_NAMES",
    "SUPPORTED_VISUAL_BACKBONES",
    "VISUAL_BACKBONE_CONTRACT_VERSION",
    "build_multimodal_model",
    "model_mode_contract",
    "model_parameter_report",
    "visual_backbone_contract",
]
