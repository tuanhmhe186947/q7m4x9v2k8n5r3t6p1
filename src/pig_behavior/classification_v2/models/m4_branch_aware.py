"""M4-BAS model architecture with branch-aware auxiliary supervision heads.

Branch-aware auxiliary heads:
1. Locomotion Head: Linear(spatial_embedding_dim, 4)
   - Input: Structured temporal pooled representation (spatial_embedding, 128D)
   - Target (masked): lying (0), stand (1), move (2), sitting (3)
   - Other behaviors: masked out (zero loss contribution)
2. Social Head: Linear(visual_context_dim + interaction_dim, 3)
   - Input: Concatenation of union visual embedding (128D) and interaction context (64D) -> 192D
   - Target: fight (0), social-nose (1), all other behaviors (2)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)

# -----------------------------------------------------------------------------
# Auxiliary Target Definitions & Maps
# -----------------------------------------------------------------------------

# Locomotion 4-class mapping (active only for posture/locomotion classes)
LOCOMOTION_BEHAVIORS = ["lying", "stand", "move", "sitting"]
BEHAVIOR_TO_LOCOMOTION_MAP: dict[str, int | None] = {
    "lying": 0,
    "stand": 1,
    "move": 2,
    "sitting": 3,
    "drink": None,
    "eat": None,
    "fight": None,
    "social-nose": None,
    "explore": None,
    "playwithtoy": None,
}

# 10-class integer index -> (target, is_active_mask)
# VALID_BEHAVIORS:
# ['drink', 'eat', 'fight', 'social-nose', 'explore',
#  'lying', 'stand', 'move', 'sitting', 'playwithtoy']
BEHAVIOR_INDEX_TO_LOCOMOTION_TARGET: list[int] = [0, 0, 0, 0, 0, 0, 1, 2, 3, 0]
BEHAVIOR_INDEX_TO_LOCOMOTION_MASK: list[bool] = [
    False,  # 0: drink
    False,  # 1: eat
    False,  # 2: fight
    False,  # 3: social-nose
    False,  # 4: explore
    True,   # 5: lying -> 0
    True,   # 6: stand -> 1
    True,   # 7: move -> 2
    True,   # 8: sitting -> 3
    False,  # 9: playwithtoy
]

# Social 3-class mapping
# 0: fight, 1: social-nose, 2: all other behaviors
BEHAVIOR_TO_SOCIAL_MAP: dict[str, int] = {
    "fight": 0,
    "social-nose": 1,
    "drink": 2,
    "eat": 2,
    "explore": 2,
    "lying": 2,
    "stand": 2,
    "move": 2,
    "sitting": 2,
    "playwithtoy": 2,
}
BEHAVIOR_INDEX_TO_SOCIAL_TARGET: list[int] = [
    2,  # 0: drink
    2,  # 1: eat
    0,  # 2: fight -> 0
    1,  # 3: social-nose -> 1
    2,  # 4: explore
    2,  # 5: lying
    2,  # 6: stand
    2,  # 7: move
    2,  # 8: sitting
    2,  # 9: playwithtoy
]


class M4BranchAwareOutput(NamedTuple):
    """Output tuple holding behavior logits and branch-specific auxiliary logits."""
    behavior: torch.Tensor    # [B, 10]
    locomotion: torch.Tensor  # [B, 4] (from structured spatial branch)
    social: torch.Tensor      # [B, 3] (from union visual + interaction branch)


class M4BranchAwareClassifier(MultimodalFusionClassifier):
    """M2 Multimodal Fusion Classifier augmented with two branch-aware auxiliary heads."""

    def __init__(self, config: MultimodalFusionConfig) -> None:
        super().__init__(config)
        
        # 1. Locomotion Auxiliary Head: from structured spatial branch (spatial_embedding_dim = 128)
        if self.spatial_encoder is None:
            raise ValueError("spatial_encoder must be enabled for M4-BAS")
        self.locomotion_head = nn.Linear(config.spatial_embedding_dim, 4)

        # 2. Social Auxiliary Head: from union visual (128D) + interaction context (64D) = 192D
        if self.visual_context_encoder is None:
            raise ValueError("visual_context_encoder must be enabled for M4-BAS")
        if self.interaction_context_encoder is None:
            raise ValueError("interaction_context_encoder must be enabled for M4-BAS")
        social_in_dim = config.visual_context_embedding_dim + config.interaction_embedding_dim
        self.social_head = nn.Linear(social_in_dim, 3)

    def forward(
        self,
        *,
        image: torch.Tensor,
        spatial_features: dict[str, torch.Tensor],
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        image_length_mask: torch.Tensor | None = None,
        image_observed_mask: torch.Tensor | None = None,
        image_available_mask: torch.Tensor | None = None,
        image_quality_mask: torch.Tensor | None = None,
        image_time_delta: torch.Tensor | None = None,
        spatial_length_mask: torch.Tensor | None = None,
        spatial_observed_mask: torch.Tensor | None = None,
        spatial_available_mask: torch.Tensor | None = None,
        spatial_quality_mask: torch.Tensor | None = None,
        spatial_time_delta: torch.Tensor | None = None,
        spatial_feature_validity_masks: Mapping[str, torch.Tensor] | None = None,
        interaction_context_features: torch.Tensor | None = None,
        interaction_context_available_mask: torch.Tensor | None = None,
        interaction_context_quality_mask: torch.Tensor | None = None,
        visual_context_image: torch.Tensor | None = None,
        visual_context_length_mask: torch.Tensor | None = None,
        visual_context_observed_mask: torch.Tensor | None = None,
        visual_context_available_mask: torch.Tensor | None = None,
        visual_context_quality_mask: torch.Tensor | None = None,
        visual_context_time_delta: torch.Tensor | None = None,
        partner_tokens: torch.Tensor | None = None,
        partner_valid_mask: torch.Tensor | None = None,
        partner_length_mask: torch.Tensor | None = None,
        partner_observed_mask: torch.Tensor | None = None,
        partner_available_mask: torch.Tensor | None = None,
        partner_quality_mask: torch.Tensor | None = None,
        partner_time_delta: torch.Tensor | None = None,
    ) -> M4BranchAwareOutput | torch.Tensor:
        """Forward pass computing main behavior logits and branch-specific auxiliary logits."""
        
        # 1. Actor Image Sequence Embedding [B, 128]
        image_embedding = self.image_encoder(
            image,
            length_mask=(
                image_length_mask
                if image_length_mask is not None
                else length_mask
            ),
            observed_mask=(
                image_observed_mask
                if image_observed_mask is not None
                else observed_mask
            ),
            available_mask=image_available_mask,
            quality_mask=image_quality_mask,
            time_delta=image_time_delta,
        )

        # 2. Structured Spatial Sequence Embedding [B, 128]
        spatial_embedding = self.spatial_encoder(
            spatial_features,
            length_mask=(
                spatial_length_mask
                if spatial_length_mask is not None
                else length_mask
            ),
            observed_mask=(
                spatial_observed_mask
                if spatial_observed_mask is not None
                else observed_mask
            ),
            available_mask=spatial_available_mask,
            quality_mask=spatial_quality_mask,
            time_delta=spatial_time_delta,
            feature_validity_masks=spatial_feature_validity_masks,
        )

        # 3. Interaction Context Embedding [B, 64]
        interaction_embedding = self.interaction_context_encoder(
            interaction_context_features,
            available_mask=interaction_context_available_mask,
            quality_mask=interaction_context_quality_mask,
        )

        # 4. Union Visual Context Embedding [B, 128]
        visual_embedding = self.visual_context_encoder(
            visual_context_image,
            length_mask=visual_context_length_mask,
            observed_mask=visual_context_observed_mask,
            available_mask=visual_context_available_mask,
            quality_mask=visual_context_quality_mask,
            time_delta=visual_context_time_delta,
        )

        # Full multimodal concatenation [B, 448]
        fused = torch.cat(
            [image_embedding, spatial_embedding, interaction_embedding, visual_embedding],
            dim=-1,
        )
        
        # Main behavior logits [B, 10]
        behavior_logits = self.classifier(fused)

        # During training: compute auxiliary heads
        # Branch A: Locomotion head from structured spatial embedding [B, 128] -> [B, 4]
        locomotion_logits = self.locomotion_head(spatial_embedding)

        # Branch B: Social head from union visual (128D) + interaction context (64D) -> [B, 3]
        social_input = torch.cat([visual_embedding, interaction_embedding], dim=-1)
        social_logits = self.social_head(social_input)

        return M4BranchAwareOutput(
            behavior=behavior_logits,
            locomotion=locomotion_logits,
            social=social_logits,
        )
