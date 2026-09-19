"""Source-aware context-conditioned branch FiLM over the unchanged M2 core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.multimodal_fusion import (
    M2BranchEmbeddings,
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
    PartnerSetEncoder,
)

JOINT_ARCHITECTURE_VERSION = "source_aware_context_conditioned_branch_film_v1"
POSTURE_CLASSES: tuple[str, ...] = ("lying", "sitting", "upright")
POSTURE_LABEL_MAP: dict[str, int] = {c: i for i, c in enumerate(POSTURE_CLASSES)}
POSTURE_NUM_CLASSES: int = 3
POSTURE_LATENT_DIM: int = 32
ROI_LATENT_DIM: int = 32
H5_LATENT_DIM: int = 32
PARTNER_LATENT_DIM: int = 32
PARTNER_COUNT: int = 2
H5_SLOTS: int = 5
H5_FEATURE_DIM: int = 46
ROI_SLOTS: int = 6
ROI_FEATURE_DIM: int = 18
ROI_VALIDITY_DIM: int = 3


@dataclass(frozen=True, slots=True)
class JointModelOutput:
    """Typed outputs for behavior, posture, branches, and FiLM audits."""

    behavior_logits: torch.Tensor
    posture_logits: torch.Tensor
    fused_embedding: torch.Tensor
    m2_fused_448: torch.Tensor
    m2_branches: M2BranchEmbeddings
    modulated_branches: M2BranchEmbeddings
    source_modulations: Mapping[str, torch.Tensor]
    partner_context: torch.Tensor
    partner_hidden_context: torch.Tensor
    h5_context: torch.Tensor
    roi_context: torch.Tensor
    posture_context: torch.Tensor
    local_representation: torch.Tensor | None = None


class OrderedH5Encoder(nn.Module):
    """Encode ordered masked structured H5 tensors without temporal pooling."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.slot_encoder = nn.Sequential(
            nn.Linear(H5_FEATURE_DIM + 1, H5_LATENT_DIM),
            nn.LayerNorm(H5_LATENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ordered_projection = nn.Sequential(
            nn.Linear(H5_SLOTS * H5_LATENT_DIM, H5_LATENT_DIM),
            nn.LayerNorm(H5_LATENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (
            H5_SLOTS,
            H5_FEATURE_DIM,
        ):
            raise ValueError("h5_structured must have shape [B,5,46]")
        valid = _binary_mask(mask, value.shape[:2], name="h5_mask")
        if not torch.isfinite(value[valid]).all():
            raise ValueError("h5_structured valid values must be finite")
        clean = torch.where(
            valid.unsqueeze(-1),
            value,
            torch.zeros_like(value),
        )
        clean = clean.clone()
        clean[..., 35] = 0.0
        slot_input = torch.cat([clean, valid.unsqueeze(-1).to(value.dtype)], dim=-1)
        slot_encoded = self.slot_encoder(slot_input)
        context = self.ordered_projection(slot_encoded.flatten(start_dim=1))
        return _gate_context(context, valid.any(dim=1))


class OrderedROIEncoder(nn.Module):
    """Encode six ordered ROI18 slots and their three validity channels."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.slot_encoder = nn.Sequential(
            nn.Linear(ROI_FEATURE_DIM + ROI_VALIDITY_DIM, ROI_LATENT_DIM),
            nn.LayerNorm(ROI_LATENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ordered_projection = nn.Sequential(
            nn.Linear(ROI_SLOTS * ROI_LATENT_DIM, ROI_LATENT_DIM),
            nn.LayerNorm(ROI_LATENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (
            ROI_SLOTS,
            ROI_FEATURE_DIM,
        ):
            raise ValueError("roi_sequence must have shape [B,6,18]")
        valid = _binary_mask(
            validity,
            (value.shape[0], ROI_SLOTS, ROI_VALIDITY_DIM),
            name="roi_validity",
        )
        if not torch.isfinite(value).all():
            raise ValueError("roi_sequence values must be finite")
        clean = value.clone()
        clean[..., 17] = 0.0
        slot_input = torch.cat([clean, valid.to(value.dtype)], dim=-1)
        slot_encoded = self.slot_encoder(slot_input)
        context = self.ordered_projection(slot_encoded.flatten(start_dim=1))
        return _gate_context(context, valid.any(dim=(1, 2)))


class ZeroInitFiLM(nn.Module):
    """Generate one bias-free zero-initialized FiLM modulation."""

    def __init__(self, context_dim: int, branch_dim: int) -> None:
        super().__init__()
        self.branch_dim = branch_dim
        self.projection = nn.Linear(context_dim, 2 * branch_dim, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        context: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.projection(context).split(self.branch_dim, dim=-1)
        return gamma * branch + beta


class JointRepresentationClassifier(MultimodalFusionClassifier):
    """Condition the four unchanged M2 branches with source-aware FiLM."""

    def __init__(self, config: MultimodalFusionConfig) -> None:
        super().__init__(config)
        if (
            self.image_encoder is None
            or self.spatial_encoder is None
            or self.interaction_context_encoder is None
            or self.visual_context_encoder is None
            or self.partner_encoder is not None
            or self.fused_embedding_dim != 448
            or config.image_embedding_dim != 128
            or config.spatial_embedding_dim != 128
            or config.interaction_embedding_dim != 64
            or config.visual_context_embedding_dim != 128
        ):
            raise ValueError(
                "Joint model requires locked M2 branches 128+128+64+128"
            )

        self.partner_behavior_encoder = PartnerSetEncoder(
            input_dim=config.num_classes,
            embedding_dim=PARTNER_LATENT_DIM,
            dropout=config.dropout,
        )
        self.partner_hidden_encoder = PartnerSetEncoder(
            input_dim=256,
            embedding_dim=PARTNER_LATENT_DIM,
            dropout=config.dropout,
        )
        self.h5_encoder = OrderedH5Encoder(config.dropout)
        self.roi_encoder = OrderedROIEncoder(config.dropout)
        self.posture_predictor = nn.Sequential(
            nn.Linear(128, POSTURE_LATENT_DIM),
            nn.LayerNorm(POSTURE_LATENT_DIM),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(POSTURE_LATENT_DIM, POSTURE_NUM_CLASSES),
        )
        self.film = nn.ModuleDict(
            {
                "h5_actor": ZeroInitFiLM(H5_LATENT_DIM, 128),
                "h5_structured": ZeroInitFiLM(H5_LATENT_DIM, 128),
                "roi_actor": ZeroInitFiLM(ROI_LATENT_DIM, 128),
                "roi_structured": ZeroInitFiLM(ROI_LATENT_DIM, 128),
                "posture_actor": ZeroInitFiLM(POSTURE_NUM_CLASSES, 128),
                "posture_structured": ZeroInitFiLM(POSTURE_NUM_CLASSES, 128),
                "pb_interaction": ZeroInitFiLM(PARTNER_LATENT_DIM, 64),
                "pb_union": ZeroInitFiLM(PARTNER_LATENT_DIM, 128),
                "pb_hidden_interaction": ZeroInitFiLM(PARTNER_LATENT_DIM, 64),
                "pb_hidden_union": ZeroInitFiLM(PARTNER_LATENT_DIM, 128),
            }
        )

    def _augment_fused_embedding(
        self,
        adapted_fused: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        """Extension seam for challengers; control keeps the exact 448D path."""

        return adapted_fused

    def _encode_partner_context(
        self,
        batch_size: int,
        value: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if value is None and mask is None:
            parameter = next(self.partner_behavior_encoder.parameters())
            return parameter.new_zeros(batch_size, PARTNER_LATENT_DIM)
        if value is None or mask is None:
            raise ValueError("partner behavior values and mask must be provided together")
        expected = (batch_size, PARTNER_COUNT, self.config.num_classes)
        if tuple(value.shape) != expected:
            raise ValueError(
                "partner_behavior_probs must have shape [B,2,num_classes]"
            )
        valid = _binary_mask(
            mask,
            (batch_size, PARTNER_COUNT),
            name="partner_behavior_mask",
        )
        context = self.partner_behavior_encoder(value, available_mask=valid)
        return _gate_context(context, valid.any(dim=1))

    def _encode_partner_hidden_context(
        self,
        batch_size: int,
        value: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if value is None:
            parameter = next(self.partner_hidden_encoder.parameters())
            return parameter.new_zeros(batch_size, PARTNER_LATENT_DIM)
        if mask is None:
            raise ValueError(
                "partner hidden mask must be provided when value is given"
            )
        expected = (batch_size, PARTNER_COUNT, 256)
        if tuple(value.shape) != expected:
            raise ValueError(
                "partner_behavior_hidden must have shape [B,2,256]"
            )
        valid = _binary_mask(
            mask,
            (batch_size, PARTNER_COUNT),
            name="partner_behavior_mask",
        )
        context = self.partner_hidden_encoder(value, available_mask=valid)
        return _gate_context(context, valid.any(dim=1))

    def _encode_h5_context(
        self,
        batch_size: int,
        value: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if value is None and mask is None:
            parameter = next(self.h5_encoder.parameters())
            return parameter.new_zeros(batch_size, H5_LATENT_DIM)
        if value is None or mask is None:
            raise ValueError("H5 structured values and mask must be provided together")
        if value.shape[0] != batch_size:
            raise ValueError("h5_structured batch size mismatch")
        return self.h5_encoder(value, mask)

    def _encode_roi_context(
        self,
        batch_size: int,
        value: torch.Tensor | None,
        validity: torch.Tensor | None,
    ) -> torch.Tensor:
        if value is None and validity is None:
            parameter = next(self.roi_encoder.parameters())
            return parameter.new_zeros(batch_size, ROI_LATENT_DIM)
        if value is None or validity is None:
            raise ValueError("ROI values and validity must be provided together")
        if value.shape[0] != batch_size:
            raise ValueError("roi_sequence batch size mismatch")
        return self.roi_encoder(value, validity)

    def forward(  # type: ignore[override]
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
        partner_behavior_probs: torch.Tensor | None = None,
        partner_behavior_hidden: torch.Tensor | None = None,
        partner_behavior_mask: torch.Tensor | None = None,
        partner_behavior_hidden_mask: torch.Tensor | None = None,
        h5_structured: torch.Tensor | None = None,
        h5_mask: torch.Tensor | None = None,
        roi_sequence: torch.Tensor | None = None,
        roi_validity: torch.Tensor | None = None,
    ) -> JointModelOutput:
        """Return M2 logits after the locked source-aware branch FiLM path."""
        m2_branches = super().encode_fused(
            image=image,
            spatial_features=spatial_features,
            length_mask=length_mask,
            observed_mask=observed_mask,
            image_length_mask=image_length_mask,
            image_observed_mask=image_observed_mask,
            image_available_mask=image_available_mask,
            image_quality_mask=image_quality_mask,
            image_time_delta=image_time_delta,
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
            return_m2_branches=True,
        )
        if not isinstance(m2_branches, M2BranchEmbeddings):
            raise RuntimeError("locked M2 branch exposure failed")
        m2_fused = m2_branches.concatenate()

        partner_context = self._encode_partner_context(
            image.shape[0],
            partner_behavior_probs,
            partner_behavior_mask,
        )
        partner_hidden_context = self._encode_partner_hidden_context(
            image.shape[0],
            partner_behavior_hidden,
            partner_behavior_hidden_mask
            if partner_behavior_hidden_mask is not None
            else partner_behavior_mask,
        )
        h5_context = self._encode_h5_context(
            image.shape[0],
            h5_structured,
            h5_mask,
        )
        roi_context = self._encode_roi_context(
            image.shape[0],
            roi_sequence,
            roi_validity,
        )

        posture_logits = self.posture_predictor(m2_branches.actor128.detach())
        posture_context = F.softmax(posture_logits, dim=-1).detach()

        source_modulations = {
            "h5_actor": self.film["h5_actor"](
                h5_context,
                m2_branches.actor128,
            ),
            "h5_structured": self.film["h5_structured"](
                h5_context,
                m2_branches.structured128,
            ),
            "roi_actor": self.film["roi_actor"](
                roi_context,
                m2_branches.actor128,
            ),
            "roi_structured": self.film["roi_structured"](
                roi_context,
                m2_branches.structured128,
            ),
            "posture_actor": self.film["posture_actor"](
                posture_context,
                m2_branches.actor128,
            ),
            "posture_structured": self.film["posture_structured"](
                posture_context,
                m2_branches.structured128,
            ),
            "pb_interaction": self.film["pb_interaction"](
                partner_context,
                m2_branches.interaction64,
            ),
            "pb_union": self.film["pb_union"](
                partner_context,
                m2_branches.union128,
            ),
            "pb_hidden_interaction": self.film["pb_hidden_interaction"](
                partner_hidden_context,
                m2_branches.interaction64,
            ),
            "pb_hidden_union": self.film["pb_hidden_union"](
                partner_hidden_context,
                m2_branches.union128,
            ),
        }
        modulated_branches = M2BranchEmbeddings(
            actor128=(
                m2_branches.actor128
                + source_modulations["h5_actor"]
                + source_modulations["roi_actor"]
                + source_modulations["posture_actor"]
            ),
            structured128=(
                m2_branches.structured128
                + source_modulations["h5_structured"]
                + source_modulations["roi_structured"]
                + source_modulations["posture_structured"]
            ),
            interaction64=(
                m2_branches.interaction64
                + source_modulations["pb_interaction"]
                + source_modulations["pb_hidden_interaction"]
            ),
            union128=(
                m2_branches.union128
                + source_modulations["pb_union"]
                + source_modulations["pb_hidden_union"]
            ),
        )
        adapted_fused = modulated_branches.concatenate()
        adapted_fused = self._augment_fused_embedding(
            adapted_fused,
            m2_branches=m2_branches,
            partner_context=partner_context,
            partner_hidden_context=partner_hidden_context,
            h5_context=h5_context,
            roi_context=roi_context,
            posture_context=posture_context,
            length_mask=length_mask,
            observed_mask=observed_mask,
            image_length_mask=image_length_mask,
            image_observed_mask=image_observed_mask,
            image_available_mask=image_available_mask,
            image_quality_mask=image_quality_mask,
            image_time_delta=image_time_delta,
            visual_context_length_mask=visual_context_length_mask,
            visual_context_observed_mask=visual_context_observed_mask,
            visual_context_available_mask=visual_context_available_mask,
            visual_context_quality_mask=visual_context_quality_mask,
            visual_context_time_delta=visual_context_time_delta,
        )
        behavior_logits = self.classifier(adapted_fused)

        return JointModelOutput(
            behavior_logits=behavior_logits,
            posture_logits=posture_logits,
            fused_embedding=adapted_fused,
            m2_fused_448=m2_fused,
            m2_branches=m2_branches,
            modulated_branches=modulated_branches,
            source_modulations=source_modulations,
            partner_context=partner_context,
            partner_hidden_context=partner_hidden_context,
            h5_context=h5_context,
            roi_context=roi_context,
            posture_context=posture_context,
            local_representation=getattr(self, "_latest_local_representation", None),
        )

    def forward_behavior(
        self,
        **model_inputs: Any,
    ) -> torch.Tensor:
        """Inference-only entry point returning behavior logits directly."""
        out = self.forward(**model_inputs)
        return out.behavior_logits


def _binary_mask(
    mask: torch.Tensor,
    expected_shape: tuple[int, ...] | torch.Size,
    *,
    name: str,
) -> torch.Tensor:
    expected = tuple(expected_shape)
    if tuple(mask.shape) != expected:
        raise ValueError(f"{name} shape {tuple(mask.shape)} does not match {expected}")
    if not torch.isfinite(mask).all():
        raise ValueError(f"{name} must be finite")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be binary")
    return mask.bool()


def _gate_context(
    context: torch.Tensor,
    source_available: torch.Tensor,
) -> torch.Tensor:
    available = _binary_mask(
        source_available,
        (context.shape[0],),
        name="source_available",
    )
    return torch.where(
        available.unsqueeze(-1),
        context,
        torch.zeros_like(context),
    )


def compute_joint_loss(
    behavior_logits: torch.Tensor,
    behavior_targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    posture_logits: torch.Tensor,
    posture_targets: torch.Tensor | None = None,
    posture_reviewed_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    posture_lambda: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined joint loss: L_total = L_behavior + posture_lambda * L_posture.

    Args:
        behavior_logits: [B, 10] behavior predictions.
        behavior_targets: [B] behavior targets (0..9).
        class_weights: [10] fold-specific behavior class weights.
        posture_logits: [B, 3] auxiliary posture predictions.
        posture_targets: [B] auxiliary posture targets (-1 for unreviewed/censored).
        posture_reviewed_mask: [B] boolean mask (True only for directly reviewed rows).
        sample_weight: [B] sample weights for M2 weighted reduction.
        posture_lambda: weighting factor for posture auxiliary loss.

    Returns:
        tuple of (total_loss, behavior_loss, posture_loss).
    """
    # 1. Behavior loss (exact M2 weighted CE)
    per_row_b_loss = F.cross_entropy(
        behavior_logits,
        behavior_targets,
        weight=class_weights,
        reduction="none",
    )
    if sample_weight is not None:
        total_sw = sample_weight.sum().clamp(min=1e-7)
        behavior_loss = (per_row_b_loss * sample_weight).sum() / total_sw
    else:
        behavior_loss = per_row_b_loss.mean()

    # 2. Posture auxiliary loss (masked to reviewed rows only)
    if (
        posture_targets is not None
        and posture_reviewed_mask is not None
        and posture_reviewed_mask.any()
    ):
        rev_indices = torch.nonzero(posture_reviewed_mask, as_tuple=True)[0]
        rev_posture_logits = posture_logits[rev_indices]
        rev_posture_targets = posture_targets[rev_indices]

        valid_targets = rev_posture_targets >= 0
        if valid_targets.any():
            posture_loss = F.cross_entropy(
                rev_posture_logits[valid_targets],
                rev_posture_targets[valid_targets],
            )
        else:
            posture_loss = torch.tensor(0.0, device=behavior_logits.device)
    else:
        posture_loss = torch.tensor(0.0, device=behavior_logits.device)

    total_loss = behavior_loss + posture_lambda * posture_loss
    return total_loss, behavior_loss, posture_loss
