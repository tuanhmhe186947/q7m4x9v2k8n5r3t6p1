"""Mask-aware multimodal fusion smoke model for classification_v2.

This module is intentionally small: it verifies that the audited image sequence
branch and spatial-temporal branch can be fused without introducing identifier,
path, review, source, or label columns into model tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

MODEL_ARCHITECTURE_VERSION = "multimodal_temporal_conv_v3_visual_context"


@dataclass(slots=True)
class ImageSequenceEncoderConfig:
    embedding_dim: int = 64
    dropout: float = 0.0


class ImageSequenceEncoder(nn.Module):
    """Encode actor crop sequences from tensors shaped ``[B, T, 3, H, W]``."""

    def __init__(self, config: ImageSequenceEncoderConfig) -> None:
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(64, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_encoder = MaskedTemporalConvEncoder(
            config.embedding_dim,
            layers=2,
            dropout=config.dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sequence embedding shaped ``[B, embedding_dim]``."""
        if image.ndim != 5:
            raise ValueError("image must have shape [B, T, 3, H, W]")
        if image.shape[2] != 3:
            raise ValueError("image channel dimension must be 3")
        mask = _combined_mask(length_mask, observed_mask, image.shape[:2])
        batch_size, sequence_len = image.shape[:2]
        encoded = self.frame_encoder(image.float().reshape(batch_size * sequence_len, *image.shape[2:]))
        encoded = encoded.reshape(batch_size, sequence_len, -1)
        projected = self.temporal_projection(encoded)
        return self.temporal_encoder(projected, mask)


@dataclass(slots=True)
class SpatialSequenceEncoderConfig:
    input_dims: dict[str, int]
    embedding_dim: int = 64
    dropout: float = 0.0


class SpatialSequenceEncoder(nn.Module):
    """Encode whitelisted bbox, motion, ROI, social, and quality sequences."""

    def __init__(self, config: SpatialSequenceEncoderConfig) -> None:
        super().__init__()
        if not config.input_dims:
            raise ValueError("input_dims must not be empty")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        self.branch_order = tuple(sorted(config.input_dims))
        branch_dim = max(8, config.embedding_dim // max(1, len(self.branch_order)))
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, branch_dim),
                    nn.LayerNorm(branch_dim),
                    nn.GELU(),
                )
                for name, dim in sorted(config.input_dims.items())
            }
        )
        fused_dim = branch_dim * len(self.branch_order)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_encoder = MaskedTemporalConvEncoder(
            config.embedding_dim,
            layers=2,
            dropout=config.dropout,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sequence embedding shaped ``[B, embedding_dim]``."""
        missing = [name for name in self.branch_order if name not in features]
        if missing:
            raise ValueError(f"Missing spatial feature groups: {missing}")
        first = features[self.branch_order[0]]
        if first.ndim != 3:
            raise ValueError(f"{self.branch_order[0]} must have shape [B, T, D]")
        mask = _combined_mask(length_mask, observed_mask, first.shape[:2])
        projected = []
        for name in self.branch_order:
            value = features[name].float()
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [B, T, D]")
            if value.shape[:2] != first.shape[:2]:
                raise ValueError(f"{name} sequence shape does not match first spatial group")
            projected.append(self.branches[name](value))
        fused = self.projection(torch.cat(projected, dim=-1))
        return self.temporal_encoder(fused, mask)


class MaskedTemporalConvEncoder(nn.Module):
    """Learn order-sensitive local dynamics and pool only observed sequence positions."""

    def __init__(self, embedding_dim: int, *, layers: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim <= 0 or layers <= 0:
            raise ValueError("temporal embedding_dim and layers must be positive")
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            embedding_dim,
                            embedding_dim,
                            kernel_size=3,
                            padding=2**layer,
                            dilation=2**layer,
                        ),
                        "norm": nn.LayerNorm(embedding_dim),
                        "dropout": nn.Dropout(dropout),
                    }
                )
                for layer in range(layers)
            ]
        )
        self.pool_score = nn.Linear(embedding_dim, 1)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,T,D]`` values while preventing padded slots from affecting valid slots."""

        if value.ndim != 3 or mask.ndim != 2 or value.shape[:2] != mask.shape:
            raise ValueError("temporal value/mask shapes must be [B,T,D] and [B,T]")
        mask_3d = mask.float().unsqueeze(-1)
        encoded = value.float() * mask_3d
        for block in self.blocks:
            update = block["conv"](encoded.transpose(1, 2)).transpose(1, 2)
            update = block["dropout"](torch.nn.functional.gelu(block["norm"](update)))
            encoded = (encoded + update) * mask_3d
        unnormalized = torch.exp(self.pool_score(encoded).squeeze(-1).clamp(-20.0, 20.0)) * mask.float()
        weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (encoded * weights.unsqueeze(-1)).sum(dim=1)


@dataclass(slots=True)
class MultimodalFusionConfig:
    spatial_input_dims: dict[str, int]
    num_classes: int
    interaction_context_dim: int | None = None
    image_embedding_dim: int = 64
    spatial_embedding_dim: int = 64
    interaction_embedding_dim: int = 32
    visual_context_embedding_dim: int = 64
    fusion_hidden_dim: int = 96
    dropout: float = 0.1
    enable_image: bool = True
    enable_spatial: bool = True
    enable_interaction_context: bool | None = None
    enable_visual_context: bool = False


class MultimodalFusionClassifier(nn.Module):
    """Late-fusion behavior classifier for image and spatial sequence branches."""

    def __init__(self, config: MultimodalFusionConfig) -> None:
        super().__init__()
        if config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        interaction_enabled = (
            config.interaction_context_dim is not None
            if config.enable_interaction_context is None
            else bool(config.enable_interaction_context)
        )
        if not any([config.enable_image, config.enable_spatial, interaction_enabled, config.enable_visual_context]):
            raise ValueError("at least one multimodal branch must be enabled")
        self.config = config
        self.image_encoder: ImageSequenceEncoder | None = None
        image_dim = 0
        if config.enable_image:
            self.image_encoder = ImageSequenceEncoder(
                ImageSequenceEncoderConfig(embedding_dim=config.image_embedding_dim, dropout=config.dropout)
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
                )
            )
            spatial_dim = config.spatial_embedding_dim
        self.interaction_context_encoder: nn.Module | None = None
        interaction_dim = 0
        if interaction_enabled:
            if config.interaction_context_dim is None:
                raise ValueError("interaction_context_dim required when interaction branch is enabled")
            if config.interaction_context_dim <= 0:
                raise ValueError("interaction_context_dim must be positive when provided")
            if config.interaction_embedding_dim <= 0:
                raise ValueError("interaction_embedding_dim must be positive")
            self.interaction_context_encoder = nn.Sequential(
                nn.Linear(config.interaction_context_dim, config.interaction_embedding_dim),
                nn.LayerNorm(config.interaction_embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            interaction_dim = config.interaction_embedding_dim
        self.visual_context_encoder: ImageSequenceEncoder | None = None
        visual_context_dim = 0
        if config.enable_visual_context:
            if config.visual_context_embedding_dim <= 0:
                raise ValueError("visual_context_embedding_dim must be positive")
            # Separate weights prevent the actor crop and actor-partner scene
            # branches from being forced into the same visual representation.
            self.visual_context_encoder = ImageSequenceEncoder(
                ImageSequenceEncoderConfig(
                    embedding_dim=config.visual_context_embedding_dim,
                    dropout=config.dropout,
                )
            )
            visual_context_dim = config.visual_context_embedding_dim
        fused_dim = image_dim + spatial_dim + interaction_dim + visual_context_dim
        self.fused_embedding_dim = int(fused_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),
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
        spatial_length_mask: torch.Tensor | None = None,
        spatial_observed_mask: torch.Tensor | None = None,
        interaction_context_features: torch.Tensor | None = None,
        interaction_context_available_mask: torch.Tensor | None = None,
        visual_context_image: torch.Tensor | None = None,
        visual_context_length_mask: torch.Tensor | None = None,
        visual_context_observed_mask: torch.Tensor | None = None,
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
            spatial_length_mask=spatial_length_mask,
            spatial_observed_mask=spatial_observed_mask,
            interaction_context_features=interaction_context_features,
            interaction_context_available_mask=interaction_context_available_mask,
            visual_context_image=visual_context_image,
            visual_context_length_mask=visual_context_length_mask,
            visual_context_observed_mask=visual_context_observed_mask,
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
        spatial_length_mask: torch.Tensor | None = None,
        spatial_observed_mask: torch.Tensor | None = None,
        interaction_context_features: torch.Tensor | None = None,
        interaction_context_available_mask: torch.Tensor | None = None,
        visual_context_image: torch.Tensor | None = None,
        visual_context_length_mask: torch.Tensor | None = None,
        visual_context_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the shared late-fusion embedding before classification heads."""

        embeddings: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.image_encoder is not None:
            image_embedding = self.image_encoder(
                image,
                length_mask=image_length_mask if image_length_mask is not None else length_mask,
                observed_mask=image_observed_mask if image_observed_mask is not None else observed_mask,
            )
            embeddings.append(image_embedding)
            batch_size = int(image_embedding.shape[0])
        if self.spatial_encoder is not None:
            spatial_embedding = self.spatial_encoder(
                spatial_features,
                length_mask=spatial_length_mask if spatial_length_mask is not None else length_mask,
                observed_mask=spatial_observed_mask if spatial_observed_mask is not None else observed_mask,
            )
            embeddings.append(spatial_embedding)
            batch_size = int(spatial_embedding.shape[0])
        if self.interaction_context_encoder is not None:
            if interaction_context_features is None:
                raise ValueError("interaction_context_features required by model config")
            if interaction_context_features.ndim != 2:
                raise ValueError("interaction_context_features must have shape [B, D]")
            if batch_size is not None and interaction_context_features.shape[0] != batch_size:
                raise ValueError("interaction_context_features batch size mismatch")
            interaction_embedding = self.interaction_context_encoder(interaction_context_features.float())
            if interaction_context_available_mask is not None:
                if interaction_context_available_mask.ndim != 1:
                    raise ValueError("interaction_context_available_mask must have shape [B]")
                interaction_embedding = interaction_embedding * interaction_context_available_mask.float().unsqueeze(-1)
            embeddings.append(interaction_embedding)
        if self.visual_context_encoder is not None:
            if visual_context_image is None or visual_context_length_mask is None:
                raise ValueError("visual context image and length mask required by model config")
            visual_embedding = self.visual_context_encoder(
                visual_context_image,
                length_mask=visual_context_length_mask,
                observed_mask=visual_context_observed_mask,
            )
            if batch_size is not None and visual_embedding.shape[0] != batch_size:
                raise ValueError("visual context batch size mismatch")
            embeddings.append(visual_embedding)
        return torch.cat(embeddings, dim=-1)


def _combined_mask(
    length_mask: torch.Tensor,
    observed_mask: torch.Tensor | None,
    expected_shape: torch.Size | tuple[int, int],
) -> torch.Tensor:
    if length_mask.ndim != 2:
        raise ValueError("length_mask must have shape [B, T]")
    if tuple(length_mask.shape) != tuple(expected_shape):
        raise ValueError(f"length_mask shape {tuple(length_mask.shape)} does not match {tuple(expected_shape)}")
    mask = length_mask.float()
    if observed_mask is not None:
        if observed_mask.shape != length_mask.shape:
            raise ValueError("observed_mask must match length_mask shape")
        mask = mask * observed_mask.float()
    return mask
