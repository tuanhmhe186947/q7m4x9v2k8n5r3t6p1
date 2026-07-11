"""Typed multitask wrapper around the audited multimodal fusion backbone.

Auxiliary heads are deterministic decompositions of behavior y used only as
regularization targets. They are never model inputs and are optional at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.multitask_heads import (
    AUXILIARY_LABEL_ORDER,
    AuxiliaryHeadConfig,
    AuxiliaryPredictionHeads,
)

MULTITASK_ARCHITECTURE_VERSION = "multimodal_temporal_conv_v3_visual_context_multitask_v1"


@dataclass(frozen=True, slots=True)
class MultitaskFusionOutput:
    """Logits returned by behavior and hierarchical auxiliary heads."""

    behavior: torch.Tensor
    posture: torch.Tensor
    motion_context: torch.Tensor
    roi_intent: torch.Tensor
    interaction: torch.Tensor

    def auxiliary_logits(self) -> dict[str, torch.Tensor]:
        return {
            "posture": self.posture,
            "motion_context": self.motion_context,
            "roi_intent": self.roi_intent,
            "interaction": self.interaction,
        }


class MultitaskFusionClassifier(nn.Module):
    """Apply behavior and auxiliary heads to one shared multimodal embedding."""

    def __init__(self, backbone_config: MultimodalFusionConfig) -> None:
        super().__init__()
        self.backbone = MultimodalFusionClassifier(backbone_config)
        fused_dim = self.backbone.fused_embedding_dim
        self.auxiliary_heads = AuxiliaryPredictionHeads(
            input_dim=fused_dim,
            heads=[
                AuxiliaryHeadConfig(name=name, num_classes=len(labels))
                for name, labels in AUXILIARY_LABEL_ORDER.items()
            ],
        )

    def forward(self, **model_inputs: torch.Tensor | dict[str, torch.Tensor] | None) -> MultitaskFusionOutput:
        """Return typed logits while accepting the fusion backbone's keyword API."""

        fused = self.backbone.encode_fused(**model_inputs)
        behavior = self.backbone.classifier(fused)
        auxiliary = self.auxiliary_heads(fused)
        return MultitaskFusionOutput(
            behavior=behavior,
            posture=auxiliary["posture"],
            motion_context=auxiliary["motion_context"],
            roi_intent=auxiliary["roi_intent"],
            interaction=auxiliary["interaction"],
        )
