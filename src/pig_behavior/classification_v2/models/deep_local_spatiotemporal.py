"""Deep local spatiotemporal RGB representation for the Final Joint challenger.

The branch consumes feature maps that the production ResNet streams already
compute before global average pooling.  It deliberately does not construct a
second backbone: actor and union maps are tapped from the existing forwards,
conditioned by the already encoded context sources, and reduced over the
canonical temporal window with the project's mask-safe temporal encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from pig_behavior.classification_v2.models.temporal_encoders import (
    build_temporal_encoder,
)

LOCAL_TOKEN_DIM = 64
LOCAL_QUERY_HIDDEN_DIM = 128
LOCAL_CONTEXT_DIM = 323
LOCAL_STREAM_DIM = LOCAL_TOKEN_DIM
LOCAL_COMBINED_DIM = LOCAL_STREAM_DIM * 4


def infer_resnet_stage_channels(frame_encoder: nn.Module) -> tuple[int, int]:
    """Infer the two tapped map widths for ResNet and bounded smoke backbones."""

    layer3 = getattr(frame_encoder, "layer3", None)
    layer4 = getattr(frame_encoder, "layer4", None)
    if isinstance(layer3, nn.Module) and isinstance(layer4, nn.Module):
        layer3_convs = [
            module
            for module in layer3.modules()
            if isinstance(module, nn.Conv2d)
        ]
        layer4_convs = [
            module
            for module in layer4.modules()
            if isinstance(module, nn.Conv2d)
        ]
        if layer3_convs and layer4_convs:
            return layer3_convs[-1].out_channels, layer4_convs[-1].out_channels

    # The smoke CNN has three strided convolutions.  The final two provide two
    # progressively deeper spatial maps and keep the challenger testable on CPU.
    convolutions = [
        module for module in frame_encoder.modules() if isinstance(module, nn.Conv2d)
    ]
    if len(convolutions) >= 2:
        return convolutions[-2].out_channels, convolutions[-1].out_channels
    raise ValueError("local branch requires a two-stage spatial frame encoder")


def spatial_tap_modules(frame_encoder: nn.Module) -> dict[str, nn.Module]:
    """Return deterministic layer3/layer4 tap points for one frame encoder."""

    layer3 = getattr(frame_encoder, "layer3", None)
    layer4 = getattr(frame_encoder, "layer4", None)
    if isinstance(layer3, nn.Module) and isinstance(layer4, nn.Module):
        return {"layer3": layer3, "layer4": layer4}
    convolutions = [
        module for module in frame_encoder.modules() if isinstance(module, nn.Conv2d)
    ]
    if len(convolutions) >= 2:
        return {"layer3": convolutions[-2], "layer4": convolutions[-1]}
    raise ValueError("local branch requires deterministic spatial tap points")


class LocalSpatialTokenAttention(nn.Module):
    """Conditioned content pooling over one spatial feature map per frame."""

    def __init__(self, channel_dims: Mapping[str, int], dropout: float) -> None:
        super().__init__()
        self.key_projections = nn.ModuleDict(
            {
                name: nn.Linear(channels, LOCAL_TOKEN_DIM, bias=False)
                for name, channels in channel_dims.items()
            }
        )
        self.value_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(channels, LOCAL_TOKEN_DIM),
                    nn.LayerNorm(LOCAL_TOKEN_DIM),
                    nn.GELU(),
                )
                for name, channels in channel_dims.items()
            }
        )
        self.scale_fusion = nn.Sequential(
            nn.Linear(2 * LOCAL_TOKEN_DIM, LOCAL_TOKEN_DIM),
            nn.LayerNorm(LOCAL_TOKEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        maps: Mapping[str, torch.Tensor],
        query: torch.Tensor,
        *,
        batch_size: int,
        sequence_len: int,
    ) -> torch.Tensor:
        pooled_scales: list[torch.Tensor] = []
        expected_bt = batch_size * sequence_len
        device = query.device
        dtype = query.dtype

        for stage in ("layer3", "layer4"):
            value = maps.get(stage)
            channels = self.key_projections[stage].in_features
            if value is None or value.ndim != 4:
                value = torch.zeros(
                    (expected_bt, channels, 8, 8),
                    device=device,
                    dtype=dtype,
                )
            elif value.shape[0] != expected_bt:
                if value.shape[0] < expected_bt:
                    pad = torch.zeros(
                        (expected_bt - value.shape[0], *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    value = torch.cat([value, pad], dim=0)
                else:
                    value = value[:expected_bt]

            tokens = value.flatten(start_dim=2).transpose(1, 2)
            keys = self.key_projections[stage](tokens)
            values = self.value_projections[stage](tokens)
            frame_query = query.unsqueeze(1).expand(-1, sequence_len, -1)
            frame_query = frame_query.reshape(batch_size * sequence_len, 1, -1)
            scores = (keys * frame_query).sum(dim=-1)
            scores = scores / math.sqrt(float(LOCAL_TOKEN_DIM))
            weights = torch.softmax(scores, dim=-1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
            pooled_scales.append(pooled.reshape(batch_size, sequence_len, -1))
        return self.scale_fusion(torch.cat(pooled_scales, dim=-1))


class LocalStreamEncoder(nn.Module):
    """Encode one actor or union stream's attended local evidence over T6."""

    def __init__(
        self,
        channel_dims: Mapping[str, int],
        *,
        dropout: float,
        temporal_encoder_name: str,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        self.attention = LocalSpatialTokenAttention(channel_dims, dropout)
        self.temporal_encoder = build_temporal_encoder(
            temporal_encoder_name,
            embedding_dim=LOCAL_STREAM_DIM,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )

    def forward(
        self,
        maps: Mapping[str, torch.Tensor],
        query: torch.Tensor,
        *,
        temporal_mask: torch.Tensor,
        time_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if temporal_mask.ndim != 2:
            raise ValueError("local temporal mask must have shape [B,T]")
        batch_size, sequence_len = temporal_mask.shape
        sequence = self.attention(
            maps,
            query,
            batch_size=batch_size,
            sequence_len=sequence_len,
        )
        return self.temporal_encoder(
            sequence,
            temporal_mask,
            time_delta=time_delta,
        )


class SourceAwareLocalSpatiotemporalBranch(nn.Module):
    """Build actor/union local representations and their explicit interaction."""

    def __init__(
        self,
        actor_channel_dims: Mapping[str, int],
        union_channel_dims: Mapping[str, int],
        *,
        dropout: float,
        temporal_encoder_name: str,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(LOCAL_CONTEXT_DIM, LOCAL_QUERY_HIDDEN_DIM),
            nn.LayerNorm(LOCAL_QUERY_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(LOCAL_QUERY_HIDDEN_DIM, LOCAL_TOKEN_DIM),
        )
        self.actor_stream = LocalStreamEncoder(
            actor_channel_dims,
            dropout=dropout,
            temporal_encoder_name=temporal_encoder_name,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.union_stream = LocalStreamEncoder(
            union_channel_dims,
            dropout=dropout,
            temporal_encoder_name=temporal_encoder_name,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.output_norm = nn.LayerNorm(LOCAL_COMBINED_DIM)

    def forward(
        self,
        *,
        actor_maps: Mapping[str, torch.Tensor],
        union_maps: Mapping[str, torch.Tensor],
        context: torch.Tensor,
        actor_temporal_mask: torch.Tensor,
        union_temporal_mask: torch.Tensor,
        actor_time_delta: torch.Tensor | None,
        union_time_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != LOCAL_CONTEXT_DIM:
            raise ValueError(
                f"local context must have shape [B,{LOCAL_CONTEXT_DIM}]"
            )
        query = self.query_encoder(context)
        actor = self.actor_stream(
            actor_maps,
            query,
            temporal_mask=actor_temporal_mask,
            time_delta=actor_time_delta,
        )
        union = self.union_stream(
            union_maps,
            query,
            temporal_mask=union_temporal_mask,
            time_delta=union_time_delta,
        )
        combined = torch.cat(
            [actor, union, actor - union, actor * union],
            dim=-1,
        )
        return self.output_norm(combined)


def merge_temporal_masks(
    length_mask: torch.Tensor,
    observed_mask: torch.Tensor | None,
    available_mask: torch.Tensor | None,
    quality_mask: torch.Tensor | None,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> torch.Tensor:
    """Apply the same length/observed/availability/quality semantics as M2."""

    def checked(value: torch.Tensor, label: str) -> torch.Tensor:
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name}_{label} shape does not match {expected_shape}")
        if not torch.isfinite(value).all() or not torch.all((value == 0) | (value == 1)):
            raise ValueError(f"{name}_{label} must be finite binary")
        return value.bool()

    length = checked(length_mask, "length")
    observed = length if observed_mask is None else checked(observed_mask, "observed")
    available = observed if available_mask is None else checked(available_mask, "available")
    quality = observed if quality_mask is None else checked(quality_mask, "quality")
    if (observed & ~length).any() or (available & ~observed).any():
        raise ValueError(f"{name} masks violate observation/availability ordering")
    if (quality & ~observed).any():
        raise ValueError(f"{name} quality mask is true outside observation")
    return length & observed & available & quality


__all__ = [
    "LOCAL_COMBINED_DIM",
    "LOCAL_CONTEXT_DIM",
    "LOCAL_STREAM_DIM",
    "LOCAL_TOKEN_DIM",
    "SourceAwareLocalSpatiotemporalBranch",
    "infer_resnet_stage_channels",
    "merge_temporal_masks",
    "spatial_tap_modules",
]
