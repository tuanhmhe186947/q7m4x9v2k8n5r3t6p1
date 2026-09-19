"""M3-FG: Functional-group auxiliary head model for classification_v2.

M3-FG extends M2-VFT with one auxiliary 4-class functional-group head on top
of the existing 256D fused hidden representation. The functional groups are:
  0 Resource: drink, eat
  1 Social interaction: fight, social-nose
  2 Environment interaction: explore, playwithtoy
  3 Posture and locomotion: lying, stand, move, sitting
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

FUNCTIONAL_GROUPS: tuple[str, ...] = (
    "resource",
    "social_interaction",
    "environment_interaction",
    "posture_locomotion",
)

BEHAVIOR_TO_GROUP_MAP: dict[str, int] = {
    "drink": 0,
    "eat": 0,
    "fight": 1,
    "social-nose": 1,
    "explore": 2,
    "lying": 3,
    "stand": 3,
    "move": 3,
    "sitting": 3,
    "playwithtoy": 2,
}

BEHAVIOR_INDEX_TO_GROUP: tuple[int, ...] = tuple(
    BEHAVIOR_TO_GROUP_MAP[name] for name in VALID_BEHAVIORS
)


@dataclass(frozen=True, slots=True)
class M3FGOutput:
    """Logits returned by main behavior head and functional-group auxiliary head."""

    behavior: torch.Tensor  # [B, 10]
    group: torch.Tensor  # [B, 4]


class M3FunctionalGroupClassifier(nn.Module):
    """M2-VFT backbone + 4-class functional-group auxiliary head."""

    def __init__(self, backbone_config: MultimodalFusionConfig) -> None:
        super().__init__()
        self.backbone = MultimodalFusionClassifier(backbone_config)
        hidden_dim = backbone_config.fusion_hidden_dim  # 256 in M2-VFT
        self.group_head = nn.Linear(hidden_dim, len(FUNCTIONAL_GROUPS))

    def forward(
        self,
        **model_inputs: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> M3FGOutput:
        """Return behavior logits [B, 10] and functional group logits [B, 4]."""
        fused = self.backbone.encode_fused(**model_inputs)
        # Pass through FusionHead (fused_dim -> 256)
        fusion_hidden = self.backbone.classifier[0](fused)
        # Pass through FinalBehaviorHead (256 -> 10)
        behavior_logits = self.backbone.classifier[1](fusion_hidden)
        # Pass through FunctionalGroupHead (256 -> 4)
        group_logits = self.group_head(fusion_hidden)
        return M3FGOutput(behavior=behavior_logits, group=group_logits)
