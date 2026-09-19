"""F2 Causal Pre-target Actor RGB H5 Multimodal Classifier Model.

Wraps the audited M2 multimodal fusion backbone to encode pre-target actor RGB (H5)
using the SAME shared actor ImageSequenceEncoder weights, followed by an identity/zero
initialized Linear(256, 128) adapter to achieve exact evaluation-mode step-0 M2 parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)


@dataclass(frozen=True, slots=True)
class F2ModelOutput:
    """Typed container for F2 behavior logits and fused embedding."""

    behavior_logits: torch.Tensor
    fused_embedding: torch.Tensor
    current_actor_embedding: torch.Tensor
    history_actor_embedding: torch.Tensor
    adapted_actor_embedding: torch.Tensor


class F2H5RgbClassifier(nn.Module):
    """M2 late-fusion backbone + shared actor encoder for H5 + Linear(256, 128) adapter."""

    def __init__(
        self,
        backbone_config: MultimodalFusionConfig,
    ) -> None:
        super().__init__()
        self.config = backbone_config
        self.backbone = MultimodalFusionClassifier(backbone_config)

        actor_dim = backbone_config.image_embedding_dim
        if actor_dim <= 0:
            raise ValueError(f"image_embedding_dim must be positive, got {actor_dim}")

        # Linear(256, 128) adapter
        self.h5_actor_adapter = nn.Linear(actor_dim * 2, actor_dim)

        # Exact Step-0 Identity/Zero initialization
        with torch.no_grad():
            self.h5_actor_adapter.weight.zero_()
            self.h5_actor_adapter.weight[:, 0:actor_dim].copy_(torch.eye(actor_dim))
            self.h5_actor_adapter.bias.zero_()

    def encode_f2_actor(
        self,
        target_image: torch.Tensor,
        history_image: torch.Tensor | None = None,
        target_length_mask: torch.Tensor | None = None,
        target_observed_mask: torch.Tensor | None = None,
        target_available_mask: torch.Tensor | None = None,
        target_quality_mask: torch.Tensor | None = None,
        target_time_delta: torch.Tensor | None = None,
        history_length_mask: torch.Tensor | None = None,
        history_observed_mask: torch.Tensor | None = None,
        history_available_mask: torch.Tensor | None = None,
        history_quality_mask: torch.Tensor | None = None,
        history_time_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode target actor T6 and history actor H5 with the SAME shared actor encoder."""
        actor_encoder = self.backbone.image_encoder
        if actor_encoder is None:
            raise ValueError("Actor image encoder is not enabled in backbone config")

        if target_time_delta is None:
            t_len = target_image.shape[1]
            target_time_delta = (
                torch.arange(t_len, dtype=torch.float32, device=target_image.device)
                .unsqueeze(0)
                .expand(target_image.shape[0], -1)
                / 30.0
            )

        current_actor_128 = actor_encoder(
            target_image,
            length_mask=target_length_mask,
            observed_mask=target_observed_mask,
            available_mask=target_available_mask,
            quality_mask=target_quality_mask,
            time_delta=target_time_delta,
        )

        if history_image is not None and history_available_mask is not None:
            if history_time_delta is None:
                h_len = history_image.shape[1]
                history_time_delta = (
                    torch.arange(
                        h_len, dtype=torch.float32, device=history_image.device
                    )
                    .unsqueeze(0)
                    .expand(history_image.shape[0], -1)
                    / 30.0
                )
            history_actor_128 = actor_encoder(
                history_image,
                length_mask=history_length_mask,
                observed_mask=history_observed_mask,
                available_mask=history_available_mask,
                quality_mask=history_quality_mask,
                time_delta=history_time_delta,
            )
        else:
            history_actor_128 = torch.zeros_like(current_actor_128)

        cat_actor = torch.cat([current_actor_128, history_actor_128], dim=-1)
        adapted_actor_128 = self.h5_actor_adapter(cat_actor)
        return adapted_actor_128, current_actor_128, history_actor_128

    def encode_fused(
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
        history_image: torch.Tensor | None = None,
        history_length_mask: torch.Tensor | None = None,
        history_observed_mask: torch.Tensor | None = None,
        history_available_mask: torch.Tensor | None = None,
        history_quality_mask: torch.Tensor | None = None,
        history_time_delta: torch.Tensor | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all multimodal branches into the late-fusion representation."""
        embeddings: list[torch.Tensor] = []
        batch_size: int | None = None

        # 1. Adapted Actor branch
        if self.backbone.image_encoder is not None:
            adapted_actor, cur_actor, hist_actor = self.encode_f2_actor(
                target_image=image,
                history_image=history_image,
                target_length_mask=(
                    image_length_mask if image_length_mask is not None else length_mask
                ),
                target_observed_mask=(
                    image_observed_mask if image_observed_mask is not None else observed_mask
                ),
                target_available_mask=image_available_mask,
                target_quality_mask=image_quality_mask,
                target_time_delta=image_time_delta,
                history_length_mask=history_length_mask,
                history_observed_mask=history_observed_mask,
                history_available_mask=history_available_mask,
                history_quality_mask=history_quality_mask,
                history_time_delta=history_time_delta,
            )
            embeddings.append(adapted_actor)
            batch_size = int(adapted_actor.shape[0])
        else:
            raise ValueError("image_encoder must be enabled for F2 architecture")

        # 2. Spatial branch (unchanged)
        if self.backbone.spatial_encoder is not None:
            spatial_embedding = self.backbone.spatial_encoder(
                spatial_features,
                length_mask=(
                    spatial_length_mask if spatial_length_mask is not None else length_mask
                ),
                observed_mask=(
                    spatial_observed_mask if spatial_observed_mask is not None else observed_mask
                ),
                available_mask=spatial_available_mask,
                quality_mask=spatial_quality_mask,
                time_delta=spatial_time_delta,
                feature_validity_masks=spatial_feature_validity_masks,
            )
            embeddings.append(spatial_embedding)
            batch_size = int(spatial_embedding.shape[0])

        # 3. Interaction branch (if configured)
        if self.backbone.interaction_context_encoder is not None:
            if interaction_context_features is None:
                raise ValueError("interaction_context_features required by model config")
            if interaction_context_available_mask is None:
                raise ValueError("interaction_context_available_mask required by model config")
            if batch_size is not None and interaction_context_features.shape[0] != batch_size:
                raise ValueError("interaction_context_features batch size mismatch")
            interaction_embedding = self.backbone.interaction_context_encoder(
                interaction_context_features,
                available_mask=interaction_context_available_mask,
                quality_mask=interaction_context_quality_mask,
            )
            embeddings.append(interaction_embedding)
            batch_size = int(interaction_embedding.shape[0])

        # 4. Visual context (union) branch (unchanged)
        if self.backbone.visual_context_encoder is not None:
            if visual_context_image is None or visual_context_length_mask is None:
                raise ValueError("visual context image and length mask required by model config")
            union_embedding = self.backbone.visual_context_encoder(
                visual_context_image,
                length_mask=visual_context_length_mask,
                observed_mask=visual_context_observed_mask,
                available_mask=visual_context_available_mask,
                quality_mask=visual_context_quality_mask,
                time_delta=visual_context_time_delta,
            )
            embeddings.append(union_embedding)
            batch_size = int(union_embedding.shape[0])

        # 5. Partner branch (if configured)
        if self.backbone.partner_encoder is not None:
            if partner_tokens is None or partner_valid_mask is None:
                raise ValueError("partner_tokens and partner_valid_mask required by model config")
            partner_embedding = self.backbone.partner_encoder(
                partner_tokens,
                partner_mask=partner_valid_mask,
                length_mask=(
                    partner_length_mask if partner_length_mask is not None else length_mask
                ),
                observed_mask=(
                    partner_observed_mask if partner_observed_mask is not None else observed_mask
                ),
                available_mask=partner_available_mask,
                quality_mask=partner_quality_mask,
                time_delta=partner_time_delta,
            )
            embeddings.append(partner_embedding)

        fused = torch.cat(embeddings, dim=-1)
        return fused, cur_actor, hist_actor, adapted_actor

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
        history_image: torch.Tensor | None = None,
        history_length_mask: torch.Tensor | None = None,
        history_observed_mask: torch.Tensor | None = None,
        history_available_mask: torch.Tensor | None = None,
        history_quality_mask: torch.Tensor | None = None,
        history_time_delta: torch.Tensor | None = None,
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
    ) -> torch.Tensor:
        """Return behavior classification logits [B, num_classes]."""
        fused, _, _, _ = self.encode_fused(
            image=image,
            spatial_features=spatial_features,
            length_mask=length_mask,
            observed_mask=observed_mask,
            image_length_mask=image_length_mask,
            image_observed_mask=image_observed_mask,
            image_available_mask=image_available_mask,
            image_quality_mask=image_quality_mask,
            image_time_delta=image_time_delta,
            history_image=history_image,
            history_length_mask=history_length_mask,
            history_observed_mask=history_observed_mask,
            history_available_mask=history_available_mask,
            history_quality_mask=history_quality_mask,
            history_time_delta=history_time_delta,
            spatial_length_mask=spatial_length_mask,
            spatial_observed_mask=spatial_observed_mask,
            spatial_available_mask=spatial_available_mask,
            spatial_quality_mask=spatial_quality_mask,
            spatial_time_delta=spatial_time_delta,
            spatial_feature_validity_masks=spatial_feature_validity_masks,
            interaction_context_features=interaction_context_features,
            interaction_context_available_mask=interaction_context_available_mask,
            interaction_context_quality_mask=interaction_context_quality_mask,
            visual_context_image=visual_context_image,
            visual_context_length_mask=visual_context_length_mask,
            visual_context_observed_mask=visual_context_observed_mask,
            visual_context_available_mask=visual_context_available_mask,
            visual_context_quality_mask=visual_context_quality_mask,
            visual_context_time_delta=visual_context_time_delta,
            partner_tokens=partner_tokens,
            partner_valid_mask=partner_valid_mask,
            partner_length_mask=partner_length_mask,
            partner_observed_mask=partner_observed_mask,
            partner_available_mask=partner_available_mask,
            partner_quality_mask=partner_quality_mask,
            partner_time_delta=partner_time_delta,
        )
        return self.backbone.classifier(fused)
