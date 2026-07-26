"""Shared visual encoder interface for the balanced causal main model.

The frame encoder itself is reused from ``models.visual_backbones`` so the
weight-enum contract stays in one place. This module only adds the shared
per-frame application used by every baseline: one backbone applied to all slots
of a sequence, with invalid slots zeroed so padding cannot leak into pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
    VisualBackboneContract,
    build_visual_frame_encoder,
    visual_backbone_contract,
)


@dataclass(frozen=True, slots=True)
class VisualEncoderConfig:
    """Validated visual-branch configuration."""

    backbone_name: str = "smoke_cnn"
    pretrained_weight_enum: str = NO_PRETRAINED_WEIGHTS
    embedding_dim: int = 64
    dropout: float = 0.0
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        visual_backbone_contract(self.backbone_name, self.pretrained_weight_enum)
        if self.embedding_dim <= 0:
            raise ValueError("visual embedding_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("visual dropout must be in [0,1)")

    def contract(self) -> VisualBackboneContract:
        return visual_backbone_contract(
            self.backbone_name,
            self.pretrained_weight_enum,
        )

    def to_payload(self) -> dict[str, Any]:
        contract = self.contract()
        return {
            "backbone_name": self.backbone_name,
            "pretrained_weight_enum": self.pretrained_weight_enum,
            "backbone_output_dim": contract.output_dim,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "freeze_backbone": self.freeze_backbone,
            "weights_are_standard_components": True,
        }


class SharedFrameVisualEncoder(nn.Module):
    """Apply one shared frame encoder across every slot of a sequence.

    Returns ``[B, T, embedding_dim]``. Slots whose ``valid_mask`` entry is false
    are returned as exact zeros, which keeps padded frames out of any
    downstream pooling or causal convolution.
    """

    def __init__(self, config: VisualEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder, self.backbone_contract = build_visual_frame_encoder(
            config.backbone_name,
            config.pretrained_weight_enum,
        )
        if config.freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad_(False)
        self.project = nn.Sequential(
            nn.Linear(self.backbone_contract.output_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    @property
    def output_dim(self) -> int:
        return self.config.embedding_dim

    def forward(self, images: Tensor, valid_mask: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError("actor_images must be [B,T,C,H,W]")
        if valid_mask.ndim != 2 or images.shape[:2] != valid_mask.shape:
            raise ValueError("actor_images and valid_mask must share [B,T]")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("actor_images contain nonfinite entries")
        batch, length = images.shape[:2]
        valid = valid_mask.bool()
        flat = images.reshape(batch * length, *images.shape[2:])
        encoded = self.frame_encoder(flat)
        if encoded.ndim != 2:
            encoded = encoded.flatten(1)
        embedded = self.project(encoded).reshape(batch, length, -1)
        return embedded * valid.unsqueeze(-1).to(embedded.dtype)


__all__ = [
    "SharedFrameVisualEncoder",
    "VisualEncoderConfig",
]
