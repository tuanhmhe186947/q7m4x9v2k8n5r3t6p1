"""Mask-safe multimodal tensor modules for classification_v2.

No identifier, path, review, source, or label value enters these modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from pig_behavior.classification_v2.models.temporal_encoders import (
    MaskedTemporalConvEncoder,
    build_temporal_encoder,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
    VisualBackboneContract,
    build_visual_frame_encoder,
)

MODEL_ARCHITECTURE_VERSION = "multimodal_sequence_factory_v5_spatial_masks"
_FEATURE_MASK_REQUIRED_GROUPS = frozenset({"motion_delta", "social_relation"})


@dataclass(slots=True)
class ImageSequenceEncoderConfig:
    backbone_name: str = "smoke_cnn"
    pretrained_weight_enum: str = NO_PRETRAINED_WEIGHTS
    embedding_dim: int = 64
    dropout: float = 0.0
    temporal_encoder_name: str = "masked_tcn"
    transformer_layers: int = 2
    transformer_heads: int = 4


class ImageSequenceEncoder(nn.Module):
    """Encode actor crop sequences from tensors shaped ``[B, T, 3, H, W]``."""

    def __init__(self, config: ImageSequenceEncoderConfig) -> None:
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        self.frame_encoder, self.backbone_contract = build_visual_frame_encoder(
            config.backbone_name,
            config.pretrained_weight_enum,
        )
        self._register_normalization(self.backbone_contract)
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.backbone_contract.output_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_encoder = build_temporal_encoder(
            config.temporal_encoder_name,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
        )

    def forward(
        self,
        image: torch.Tensor,
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        available_mask: torch.Tensor | None = None,
        quality_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sequence embedding shaped ``[B, embedding_dim]``."""
        if image.ndim != 5:
            raise ValueError("image must have shape [B, T, 3, H, W]")
        if image.shape[2] != 3:
            raise ValueError("image channel dimension must be 3")
        mask = _combined_mask(
            length_mask,
            observed_mask,
            image.shape[:2],
            available_mask=available_mask,
            quality_mask=quality_mask,
            branch_name="image",
        )
        batch_size, sequence_len = image.shape[:2]
        clean_image = _masked_values(image, mask, branch_name="image")
        clean_image = self._normalize(clean_image)
        encoded = self.frame_encoder(
            clean_image.reshape(
                batch_size * sequence_len,
                *image.shape[2:],
            )
        )
        encoded = encoded.reshape(batch_size, sequence_len, -1)
        projected = self.temporal_projection(encoded)
        return self.temporal_encoder(projected, mask, time_delta=time_delta)

    def _register_normalization(
        self,
        contract: VisualBackboneContract,
    ) -> None:
        """Store deterministic RGB normalization outside the checkpoint state."""

        mean = torch.tensor(contract.input_mean).reshape(1, 3, 1, 1)
        std = torch.tensor(contract.input_std).reshape(1, 3, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the backbone contract to cache tensors already scaled to [0, 1]."""

        mean = self._input_mean.to(dtype=image.dtype)
        std = self._input_std.to(dtype=image.dtype)
        return (image - mean) / std


@dataclass(slots=True)
class SpatialSequenceEncoderConfig:
    input_dims: dict[str, int]
    embedding_dim: int = 64
    dropout: float = 0.0
    temporal_encoder_name: str = "masked_tcn"
    transformer_layers: int = 2
    transformer_heads: int = 4


class SpatialSequenceEncoder(nn.Module):
    """Encode whitelisted bbox, motion, ROI, social, and quality sequences."""

    def __init__(self, config: SpatialSequenceEncoderConfig) -> None:
        super().__init__()
        if not config.input_dims:
            raise ValueError("input_dims must not be empty")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        self.branch_order = tuple(config.input_dims)
        branch_dim = max(8, config.embedding_dim // max(1, len(self.branch_order)))
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, branch_dim),
                    nn.LayerNorm(branch_dim),
                    nn.GELU(),
                )
                for name, dim in config.input_dims.items()
            }
        )
        self.mask_branches = nn.ModuleDict(
            {
                name: nn.Linear(dim, branch_dim, bias=False)
                for name, dim in config.input_dims.items()
                if name in _FEATURE_MASK_REQUIRED_GROUPS
            }
        )
        fused_dim = branch_dim * len(self.branch_order)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_encoder = build_temporal_encoder(
            config.temporal_encoder_name,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        available_mask: torch.Tensor | None = None,
        quality_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        feature_validity_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return sequence embedding shaped ``[B, embedding_dim]``."""
        missing = [name for name in self.branch_order if name not in features]
        if missing:
            raise ValueError(f"Missing spatial feature groups: {missing}")
        required_masks = _FEATURE_MASK_REQUIRED_GROUPS.intersection(self.branch_order)
        provided_masks = set() if feature_validity_masks is None else set(feature_validity_masks)
        missing_masks = sorted(required_masks.difference(provided_masks))
        if missing_masks:
            raise ValueError(f"Missing spatial feature validity masks: {missing_masks}")
        first = features[self.branch_order[0]]
        if first.ndim != 3:
            raise ValueError(f"{self.branch_order[0]} must have shape [B, T, D]")
        mask = _combined_mask(
            length_mask,
            observed_mask,
            first.shape[:2],
            available_mask=available_mask,
            quality_mask=quality_mask,
            branch_name="spatial",
        )
        projected = []
        for name in self.branch_order:
            value = features[name]
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [B, T, D]")
            if value.shape[:2] != first.shape[:2]:
                raise ValueError(f"{name} sequence shape does not match first spatial group")
            expected_dim = self.config.input_dims[name]
            if value.shape[-1] != expected_dim:
                msg = (
                    f"{name} feature dim {value.shape[-1]} "
                    f"does not match {expected_dim}"
                )
                raise ValueError(msg)
            explicit = (
                None
                if feature_validity_masks is None
                else feature_validity_masks.get(name)
            )
            base_feature_mask = mask.unsqueeze(-1).to(value.dtype)
            if explicit is None:
                feature_mask = base_feature_mask.expand_as(value)
                mask_delta = None
            else:
                if name not in self.mask_branches:
                    raise ValueError(
                        f"Unexpected feature validity mask for {name}"
                    )
                if explicit.shape != value.shape:
                    raise ValueError(
                        f"{name} feature validity shape does not match values: "
                        f"{tuple(explicit.shape)}:{tuple(value.shape)}"
                    )
                feature_mask = base_feature_mask * explicit.to(value.dtype)
                mask_delta = feature_mask - base_feature_mask
            clean = torch.where(
                feature_mask.bool(),
                value,
                torch.zeros_like(value),
            )
            branch = self.branches[name](clean)
            branch = branch * feature_mask.any(dim=-1, keepdim=True).to(
                branch.dtype
            )
            branch_out = (
                branch
                if mask_delta is None
                else branch + self.mask_branches[name](mask_delta)
            )
            projected.append(branch_out)
        fused = self.projection(torch.cat(projected, dim=-1))
        return self.temporal_encoder(fused, mask, time_delta=time_delta)


class ActorEncoder(ImageSequenceEncoder):
    """Named actor-crop branch sharing the image sequence tensor contract."""


class PartnerSetEncoder(nn.Module):
    """Encode one or more label-independently selected partner feature rows."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0:
            raise ValueError("partner input and embedding dimensions must be positive")
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        available_mask: torch.Tensor,
        quality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B,D]`` while absent partners contribute exactly zero."""

        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[-1] != self.input_dim:
            msg = f"partner features must have shape [B,D] or [B,K,D] with D={self.input_dim}"
            raise ValueError(msg)
        available = _set_mask(
            available_mask,
            value.shape[:2],
            name="interaction_context_available_mask",
        )
        if quality_mask is None:
            quality = available
        else:
            quality = _set_mask(
                quality_mask,
                value.shape[:2],
                name="interaction_context_quality_mask",
            )
            if (quality & ~available).any():
                raise ValueError("interaction quality mask is true outside availability")
        valid = available & quality
        clean = _masked_values(value, valid, branch_name="interaction_context")
        projected = self.projection(clean)
        weights = valid.to(projected.dtype).unsqueeze(-1)
        pooled = (projected * weights).sum(dim=1)
        return pooled / weights.sum(dim=1).clamp_min(1.0)


@dataclass(slots=True)
class RelationalPartnerEncoderConfig:
    token_dim: int = 6
    k: int = 2
    token_embedding_dim: int = 32
    embedding_dim: int = 32
    dropout: float = 0.0
    temporal_encoder_name: str = "masked_tcn"
    transformer_layers: int = 2
    transformer_heads: int = 4


class RelationalPartnerSequenceEncoder(nn.Module):
    """Encode explicit actor-partner relational geometry sequences [B, T, K, D]."""

    def __init__(self, config: RelationalPartnerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_partner_encoder = PartnerSetEncoder(
            input_dim=config.token_dim,
            embedding_dim=config.token_embedding_dim,
            dropout=config.dropout,
        )
        if config.token_embedding_dim != config.embedding_dim:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(config.token_embedding_dim, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.projection = nn.Identity()

        self.temporal_encoder = build_temporal_encoder(
            config.temporal_encoder_name,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        partner_mask: torch.Tensor,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        available_mask: torch.Tensor | None = None,
        quality_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode [B, T, K, D] relational tokens to [B, embedding_dim]."""
        if tokens.ndim != 4:
            raise ValueError("partner tokens must have shape [B, T, K, D]")
        if tokens.shape[-1] != self.config.token_dim:
            msg = f"partner token dimension {tokens.shape[-1]} != expected {self.config.token_dim}"
            raise ValueError(msg)
        batch_size, seq_len, k_dim, token_dim = tokens.shape
        t_mask = _combined_mask(
            length_mask,
            observed_mask,
            tokens.shape[:2],
            available_mask=available_mask,
            quality_mask=quality_mask,
            branch_name="partner_tokens",
        )

        flat_tokens = tokens.reshape(batch_size * seq_len, k_dim, token_dim)
        flat_partner_mask = partner_mask.reshape(batch_size * seq_len, k_dim)

        flat_frame_valid = t_mask.reshape(batch_size * seq_len, 1)
        flat_mask = flat_partner_mask.to(torch.bool) & flat_frame_valid

        frame_encoded = self.frame_partner_encoder(
            flat_tokens,
            available_mask=flat_mask,
        )
        seq_encoded = frame_encoded.reshape(batch_size, seq_len, -1)
        seq_projected = self.projection(seq_encoded)

        return self.temporal_encoder(seq_projected, t_mask, time_delta=time_delta)


class UnionCropEncoder(ImageSequenceEncoder):
    """Encode an actor-partner union/context crop with independent weights."""


class AvailabilityEncoder(nn.Module):
    """Gate embeddings only; masks are never concatenated as behavior features."""

    def forward(
        self,
        embedding: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError("embedding must have shape [B,D]")
        available = _vector_mask(
            available_mask,
            embedding.shape[0],
            name="branch_available_mask",
        )
        return torch.where(
            available.unsqueeze(-1),
            embedding,
            torch.zeros_like(embedding),
        )


class FusionHead(nn.Module):
    """Fuse enabled embeddings without receiving any availability bits."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("fusion input and hidden dimensions must be positive")
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError("fusion embedding must have shape [B,D]")
        return self.layers(embedding)


class FinalBehaviorHead(nn.Module):
    """Directly supervised final behavior logits with no hard cascade."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("behavior head dimensions are invalid")
        self.projection = nn.Linear(input_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError("behavior embedding must have shape [B,D]")
        return self.projection(embedding)


@dataclass(slots=True)
class MultimodalFusionConfig:
    spatial_input_dims: dict[str, int]
    num_classes: int
    interaction_context_dim: int | None = None
    backbone_name: str = "smoke_cnn"
    pretrained_weight_enum: str = NO_PRETRAINED_WEIGHTS
    image_embedding_dim: int = 64
    spatial_embedding_dim: int = 64
    interaction_embedding_dim: int = 32
    visual_context_embedding_dim: int = 64
    partner_token_dim: int = 6
    partner_embedding_dim: int = 32
    fusion_hidden_dim: int = 96
    dropout: float = 0.1
    temporal_encoder_name: str = "masked_tcn"
    transformer_layers: int = 2
    transformer_heads: int = 4
    enable_image: bool = True
    enable_spatial: bool = True
    enable_interaction_context: bool | None = None
    enable_visual_context: bool = False
    enable_partner_tokens: bool = False


class MultimodalFusionClassifier(nn.Module):
    """Late-fusion behavior classifier for image and spatial sequence branches."""

    def __init__(self, config: MultimodalFusionConfig) -> None:
        super().__init__()
        if config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        interaction_enabled = (
            config.interaction_context_dim is not None
            if config.enable_interaction_context is None
            else bool(config.enable_interaction_context)
        )
        if not any(
            [
                config.enable_image,
                config.enable_spatial,
                interaction_enabled,
                config.enable_visual_context,
            ]
        ):
            raise ValueError("at least one multimodal branch must be enabled")
        self.config = config
        self.image_encoder: ImageSequenceEncoder | None = None
        image_dim = 0
        if config.enable_image:
            self.image_encoder = ActorEncoder(
                ImageSequenceEncoderConfig(
                    backbone_name=config.backbone_name,
                    pretrained_weight_enum=config.pretrained_weight_enum,
                    embedding_dim=config.image_embedding_dim,
                    dropout=config.dropout,
                    temporal_encoder_name=config.temporal_encoder_name,
                    transformer_layers=config.transformer_layers,
                    transformer_heads=config.transformer_heads,
                )
            )
            image_dim = config.image_embedding_dim
        self.spatial_encoder: SpatialSequenceEncoder | None = None
        spatial_dim = 0
        if config.enable_spatial:
            self.spatial_encoder = SpatialSequenceEncoder(
                SpatialSequenceEncoderConfig(
                    input_dims=config.spatial_input_dims,
                    embedding_dim=config.spatial_embedding_dim,
                    dropout=config.dropout,
                    temporal_encoder_name=config.temporal_encoder_name,
                    transformer_layers=config.transformer_layers,
                    transformer_heads=config.transformer_heads,
                )
            )
            spatial_dim = config.spatial_embedding_dim
        self.interaction_context_encoder: nn.Module | None = None
        interaction_dim = 0
        if interaction_enabled:
            if config.interaction_context_dim is None:
                msg = "interaction_context_dim required when interaction branch is enabled"
                raise ValueError(msg)
            if config.interaction_context_dim <= 0:
                raise ValueError("interaction_context_dim must be positive when provided")
            if config.interaction_embedding_dim <= 0:
                raise ValueError("interaction_embedding_dim must be positive")
            self.interaction_context_encoder = PartnerSetEncoder(
                config.interaction_context_dim,
                config.interaction_embedding_dim,
                config.dropout,
            )
            interaction_dim = config.interaction_embedding_dim
        self.visual_context_encoder: ImageSequenceEncoder | None = None
        visual_context_dim = 0
        if config.enable_visual_context:
            if config.visual_context_embedding_dim <= 0:
                raise ValueError("visual_context_embedding_dim must be positive")
            # Separate weights prevent the actor crop and actor-partner scene
            # branches from being forced into the same visual representation.
            self.visual_context_encoder = UnionCropEncoder(
                ImageSequenceEncoderConfig(
                    backbone_name=config.backbone_name,
                    pretrained_weight_enum=config.pretrained_weight_enum,
                    embedding_dim=config.visual_context_embedding_dim,
                    dropout=config.dropout,
                    temporal_encoder_name=config.temporal_encoder_name,
                    transformer_layers=config.transformer_layers,
                    transformer_heads=config.transformer_heads,
                )
            )
            visual_context_dim = config.visual_context_embedding_dim
        self.partner_encoder: RelationalPartnerSequenceEncoder | None = None
        partner_dim = 0
        if config.enable_partner_tokens:
            if config.partner_token_dim <= 0:
                raise ValueError("partner_token_dim must be positive")
            if config.partner_embedding_dim <= 0:
                raise ValueError("partner_embedding_dim must be positive")
            self.partner_encoder = RelationalPartnerSequenceEncoder(
                RelationalPartnerEncoderConfig(
                    token_dim=config.partner_token_dim,
                    token_embedding_dim=config.partner_embedding_dim,
                    embedding_dim=config.partner_embedding_dim,
                    dropout=config.dropout,
                    temporal_encoder_name=config.temporal_encoder_name,
                    transformer_layers=config.transformer_layers,
                    transformer_heads=config.transformer_heads,
                )
            )
            partner_dim = config.partner_embedding_dim
        fused_dim = image_dim + spatial_dim + interaction_dim + visual_context_dim + partner_dim
        self.fused_embedding_dim = int(fused_dim)
        self.classifier = nn.Sequential(
            FusionHead(fused_dim, config.fusion_hidden_dim, config.dropout),
            FinalBehaviorHead(config.fusion_hidden_dim, config.num_classes),
        )

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
    ) -> torch.Tensor:
        """Return logits shaped ``[B, num_classes]``.

        ``length_mask``/``observed_mask`` are retained as the common-mask API for
        equal-length branches. The explicit image/spatial masks support the
        current audited data, where image context windows may contain 6 frames
        while spatial arrays are padded to the maximum configured window length.
        """
        fused = self.encode_fused(
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
        )
        return self.classifier(fused)

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
        """Return the shared late-fusion embedding before classification heads."""

        embeddings: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.image_encoder is not None:
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
            embeddings.append(image_embedding)
            batch_size = int(image_embedding.shape[0])
        if self.spatial_encoder is not None:
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
            embeddings.append(spatial_embedding)
            batch_size = int(spatial_embedding.shape[0])
        if self.interaction_context_encoder is not None:
            if interaction_context_features is None:
                raise ValueError("interaction_context_features required by model config")
            if interaction_context_available_mask is None:
                raise ValueError("interaction_context_available_mask required by model config")
            if batch_size is not None and interaction_context_features.shape[0] != batch_size:
                raise ValueError("interaction_context_features batch size mismatch")
            interaction_embedding = self.interaction_context_encoder(
                interaction_context_features,
                available_mask=interaction_context_available_mask,
                quality_mask=interaction_context_quality_mask,
            )
            embeddings.append(interaction_embedding)
            batch_size = int(interaction_embedding.shape[0])
        if self.visual_context_encoder is not None:
            if visual_context_image is None or visual_context_length_mask is None:
                raise ValueError("visual context image and length mask required by model config")
            visual_embedding = self.visual_context_encoder(
                visual_context_image,
                length_mask=visual_context_length_mask,
                observed_mask=visual_context_observed_mask,
                available_mask=visual_context_available_mask,
                quality_mask=visual_context_quality_mask,
                time_delta=visual_context_time_delta,
            )
            if batch_size is not None and visual_embedding.shape[0] != batch_size:
                raise ValueError("visual context batch size mismatch")
            embeddings.append(visual_embedding)
            batch_size = int(visual_embedding.shape[0])
        if self.partner_encoder is not None:
            if partner_tokens is None or partner_valid_mask is None:
                raise ValueError("partner_tokens and partner_valid_mask required by model config")
            effective_partner_time_delta = (
                partner_time_delta
                if partner_time_delta is not None
                else (
                    spatial_time_delta
                    if spatial_time_delta is not None
                    else image_time_delta
                )
            )
            partner_embedding = self.partner_encoder(
                partner_tokens,
                partner_mask=partner_valid_mask,
                length_mask=(
                    partner_length_mask
                    if partner_length_mask is not None
                    else length_mask
                ),
                observed_mask=(
                    partner_observed_mask
                    if partner_observed_mask is not None
                    else observed_mask
                ),
                available_mask=partner_available_mask,
                quality_mask=partner_quality_mask,
                time_delta=effective_partner_time_delta,
            )
            if batch_size is not None and partner_embedding.shape[0] != batch_size:
                raise ValueError("partner embedding batch size mismatch")
            embeddings.append(partner_embedding)
            batch_size = int(partner_embedding.shape[0])
        return torch.cat(embeddings, dim=-1)


def _combined_mask(
    length_mask: torch.Tensor,
    observed_mask: torch.Tensor | None,
    expected_shape: torch.Size | tuple[int, int],
    *,
    available_mask: torch.Tensor | None = None,
    quality_mask: torch.Tensor | None = None,
    branch_name: str,
) -> torch.Tensor:
    length = _set_mask(
        length_mask,
        expected_shape,
        name=f"{branch_name}_length_mask",
    )
    observed = (
        length
        if observed_mask is None
        else _set_mask(
            observed_mask,
            expected_shape,
            name=f"{branch_name}_observed_mask",
        )
    )
    available = (
        observed
        if available_mask is None
        else _set_mask(
            available_mask,
            expected_shape,
            name=f"{branch_name}_available_mask",
        )
    )
    quality = (
        observed
        if quality_mask is None
        else _set_mask(
            quality_mask,
            expected_shape,
            name=f"{branch_name}_quality_mask",
        )
    )
    if (observed & ~length).any():
        raise ValueError(f"{branch_name} observed mask is true outside length")
    if (available & ~observed).any():
        raise ValueError(f"{branch_name} availability is true outside observation")
    if (quality & ~observed).any():
        raise ValueError(f"{branch_name} quality is true outside observation")
    return length & observed & available & quality


def _set_mask(
    mask: torch.Tensor,
    expected_shape: torch.Size | tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    expected = tuple(expected_shape)
    value = mask
    if value.ndim == 1 and expected[1] == 1 and value.shape[0] == expected[0]:
        value = value.unsqueeze(1)
    if value.ndim != 2 or tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {expected}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite entries")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must be binary")
    return value.bool()


def _vector_mask(mask: torch.Tensor, batch_size: int, *, name: str) -> torch.Tensor:
    if mask.ndim != 1 or mask.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [B]")
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be finite and binary")
    return mask.bool()


def _masked_values(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    branch_name: str,
) -> torch.Tensor:
    if tuple(value.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"{branch_name} values do not match mask shape")
    if not torch.isfinite(value[mask]).all():
        raise ValueError(f"{branch_name} observed values contain nonfinite entries")
    expanded = mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
    return torch.where(expanded, value.float(), torch.zeros_like(value).float())


__all__ = [
    "ActorEncoder",
    "AvailabilityEncoder",
    "FinalBehaviorHead",
    "FusionHead",
    "ImageSequenceEncoder",
    "ImageSequenceEncoderConfig",
    "MODEL_ARCHITECTURE_VERSION",
    "MaskedTemporalConvEncoder",
    "MultimodalFusionClassifier",
    "MultimodalFusionConfig",
    "PartnerSetEncoder",
    "RelationalPartnerEncoderConfig",
    "RelationalPartnerSequenceEncoder",
    "SpatialSequenceEncoder",
    "SpatialSequenceEncoderConfig",
    "UnionCropEncoder",
]
