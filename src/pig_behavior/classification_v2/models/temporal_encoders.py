"""Mask-safe temporal encoders for controlled classification_v2 ablations."""

from __future__ import annotations

import math

import torch
from torch import nn

TEMPORAL_ENCODER_NAMES = frozenset(
    {
        "masked_mean",
        "masked_attention",
        "masked_tcn",
        "small_transformer",
    }
)


class MaskedMeanEncoder(nn.Module):
    """Pool valid sequence slots without learning order or attention weights."""

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del time_delta
        value, valid = _validated_inputs(value, mask)
        weights = valid.to(value.dtype).unsqueeze(-1)
        total = (value * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp_min(1.0)
        return total / count


class MaskedAttentionEncoder(nn.Module):
    """Content-pool valid slots while remaining invariant to masked values."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        _require_positive_dimension(embedding_dim)
        self.pool_score = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del time_delta
        value, valid = _validated_inputs(value, mask)
        scores = self.pool_score(value).squeeze(-1).clamp(-20.0, 20.0)
        unnormalized = torch.exp(scores) * valid.to(value.dtype)
        weights = unnormalized / unnormalized.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        return (value * weights.unsqueeze(-1)).sum(dim=1)


class MaskedTemporalConvEncoder(nn.Module):
    """Learn local ordered dynamics while zeroing every invalid slot per layer."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_positive_dimension(embedding_dim)
        if layers <= 0:
            raise ValueError("temporal convolution layers must be positive")
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
        self.pool = MaskedAttentionEncoder(embedding_dim)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del time_delta
        value, valid = _validated_inputs(value, mask)
        mask_3d = valid.to(value.dtype).unsqueeze(-1)
        encoded = value * mask_3d
        for block in self.blocks:
            update = block["conv"](encoded.transpose(1, 2)).transpose(1, 2)
            update = block["norm"](update)
            update = torch.nn.functional.gelu(update)
            update = block["dropout"](update)
            encoded = (encoded + update) * mask_3d
        return self.pool(encoded, valid)


class SmallMaskedTransformerEncoder(nn.Module):
    """Encode real-time deltas with a small key-padding-aware Transformer."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_positive_dimension(embedding_dim)
        if layers not in {1, 2}:
            raise ValueError("small transformer supports one or two layers")
        if heads <= 0 or embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by transformer heads")
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.pool = MaskedAttentionEncoder(embedding_dim)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value, valid = _validated_inputs(value, mask)
        deltas = _validated_time_delta(time_delta, valid)
        mask_3d = valid.to(value.dtype).unsqueeze(-1)
        encoded = value * mask_3d
        encoded = encoded + _continuous_time_encoding(
            deltas,
            value.shape[-1],
            dtype=value.dtype,
        ) * mask_3d
        safe_valid = valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True
            encoded = encoded.clone()
            encoded[empty_rows, 0] = 0.0
        encoded = self.encoder(encoded, src_key_padding_mask=~safe_valid)
        encoded = encoded * mask_3d
        return self.pool(encoded, valid)


def build_temporal_encoder(
    name: str,
    *,
    embedding_dim: int,
    dropout: float,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
) -> nn.Module:
    """Build one declared temporal family without silently changing its name."""

    if name == "masked_mean":
        return MaskedMeanEncoder()
    if name == "masked_attention":
        return MaskedAttentionEncoder(embedding_dim)
    if name == "masked_tcn":
        return MaskedTemporalConvEncoder(
            embedding_dim,
            layers=2,
            dropout=dropout,
        )
    if name == "small_transformer":
        return SmallMaskedTransformerEncoder(
            embedding_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            dropout=dropout,
        )
    raise ValueError(f"unsupported temporal encoder={name}")


def _validated_inputs(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 3 or mask.ndim != 2 or value.shape[:2] != mask.shape:
        raise ValueError("temporal value/mask shapes must be [B,T,D] and [B,T]")
    if not torch.isfinite(mask).all():
        raise ValueError("temporal mask contains nonfinite entries")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("temporal mask must be binary")
    valid = mask.bool()
    if not torch.isfinite(value[valid]).all():
        raise ValueError("observed temporal values contain nonfinite entries")
    clean = torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))
    return clean.float(), valid


def _validated_time_delta(
    time_delta: torch.Tensor | None,
    valid: torch.Tensor,
) -> torch.Tensor:
    if time_delta is None:
        raise ValueError("small_transformer requires real time_delta [B,T]")
    if time_delta.shape != valid.shape:
        raise ValueError("time_delta must match temporal mask shape")
    if not torch.isfinite(time_delta[valid]).all():
        raise ValueError("observed time_delta contains nonfinite entries")
    if (time_delta[valid] < 0).any():
        raise ValueError("observed time_delta values must be non-negative")
    return torch.where(valid, time_delta, torch.zeros_like(time_delta)).float()


def _continuous_time_encoding(
    time_delta: torch.Tensor,
    embedding_dim: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    elapsed = torch.cumsum(time_delta, dim=1)
    half = max(1, embedding_dim // 2)
    denominator = max(1, half - 1)
    frequencies = torch.exp(
        torch.arange(half, device=elapsed.device, dtype=torch.float32)
        * (-math.log(10_000.0) / denominator)
    )
    angles = elapsed.unsqueeze(-1) * frequencies.view(1, 1, -1)
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if encoded.shape[-1] < embedding_dim:
        encoded = torch.nn.functional.pad(
            encoded,
            (0, embedding_dim - encoded.shape[-1]),
        )
    return encoded[..., :embedding_dim].to(dtype=dtype)


def _require_positive_dimension(embedding_dim: int) -> None:
    if embedding_dim <= 0:
        raise ValueError("temporal embedding_dim must be positive")


__all__ = [
    "TEMPORAL_ENCODER_NAMES",
    "MaskedAttentionEncoder",
    "MaskedMeanEncoder",
    "MaskedTemporalConvEncoder",
    "SmallMaskedTransformerEncoder",
    "build_temporal_encoder",
]
