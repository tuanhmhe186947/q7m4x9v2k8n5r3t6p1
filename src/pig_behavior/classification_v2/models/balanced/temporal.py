"""Causal temporal primitives for the balanced causal main model.

Every encoder here is strictly causal: the representation read out at the
prediction endpoint depends only on slots at or before that endpoint. The
readout is always taken at the last valid slot, so trailing padded slots and
any slot after the endpoint cannot influence the prediction.

The existing ``models.temporal_encoders`` family is mask-safe but not causal
(symmetric convolution padding, unmasked full self-attention), so it is left
untouched and reused only where non-causal pooling is scientifically intended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

CAUSAL_ENCODER_NAMES: tuple[str, ...] = (
    "causal_tcn",
    "causal_gru",
    "causal_transformer",
)

SUPPORTED_TARGET_LENGTHS: tuple[int, ...] = (6, 8, 12, 16)


class CausalityError(ValueError):
    """Raised when a temporal input violates causal endpoint semantics."""


@dataclass(frozen=True, slots=True)
class CausalTemporalConfig:
    """Validated configuration for one causal temporal encoder."""

    name: str = "causal_tcn"
    hidden_dim: int = 64
    layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.0
    heads: int = 4

    def __post_init__(self) -> None:
        if self.name not in CAUSAL_ENCODER_NAMES:
            raise ValueError(
                f"unsupported causal temporal encoder={self.name}; "
                f"expected one of {list(CAUSAL_ENCODER_NAMES)}"
            )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.kernel_size < 2:
            raise ValueError("causal kernel_size must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.name == "causal_transformer":
            if self.heads <= 0 or self.hidden_dim % self.heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by heads for causal_transformer"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "heads": self.heads,
            "causal": True,
            "future_frame_dependence": 0,
        }


def causal_attention_mask(length: int, *, device: torch.device | None = None) -> Tensor:
    """Return an additive triangular mask that blocks every future position."""

    if length <= 0:
        raise ValueError("causal attention mask length must be positive")
    allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    mask = torch.zeros(length, length, dtype=torch.float32, device=device)
    return mask.masked_fill(~allowed, float("-inf"))


def endpoint_index(valid_mask: Tensor) -> Tensor:
    """Return the last valid slot index per row, i.e. the prediction endpoint."""

    valid = _validated_mask(valid_mask)
    if not bool(valid.any(dim=1).all()):
        raise CausalityError(
            "every sample needs at least one valid temporal slot to define a "
            "prediction endpoint"
        )
    positions = torch.arange(valid.shape[1], device=valid.device)
    masked = torch.where(valid, positions.unsqueeze(0), torch.full_like(positions, -1))
    return masked.max(dim=1).values


def gather_endpoint(sequence: Tensor, valid_mask: Tensor) -> Tensor:
    """Read out the encoded state at the prediction endpoint of each row."""

    if sequence.ndim != 3:
        raise ValueError("sequence must be [B,T,D]")
    index = endpoint_index(valid_mask)
    expanded = index.view(-1, 1, 1).expand(-1, 1, sequence.shape[-1])
    return sequence.gather(1, expanded).squeeze(1)


class CausalTemporalConvEncoder(nn.Module):
    """Dilated causal convolution stack with left-only padding."""

    def __init__(self, config: CausalTemporalConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        config.hidden_dim,
                        config.hidden_dim,
                        kernel_size=config.kernel_size,
                        dilation=2**layer,
                    ),
                    "norm": nn.LayerNorm(config.hidden_dim),
                    "dropout": nn.Dropout(config.dropout),
                }
            )
            for layer in range(config.layers)
        )

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        encoded, valid = _prepare(value, valid_mask)
        mask_3d = valid.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * mask_3d
        for layer, block in enumerate(self.blocks):
            pad = (self.config.kernel_size - 1) * (2**layer)
            padded = torch.nn.functional.pad(encoded.transpose(1, 2), (pad, 0))
            update = block["conv"](padded).transpose(1, 2)
            update = block["norm"](update)
            update = torch.nn.functional.gelu(update)
            update = block["dropout"](update)
            encoded = (encoded + update) * mask_3d
        return gather_endpoint(encoded, valid)


class CausalMaskedGRUEncoder(nn.Module):
    """GRU stepped in causal order with mask-gated hidden-state carry."""

    def __init__(self, config: CausalTemporalConfig) -> None:
        super().__init__()
        self.config = config
        self.cells = nn.ModuleList(
            nn.GRUCell(config.hidden_dim, config.hidden_dim)
            for _ in range(config.layers)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        encoded, valid = _prepare(value, valid_mask)
        batch, length, _ = encoded.shape
        for cell in self.cells:
            state = encoded.new_zeros(batch, self.config.hidden_dim)
            outputs: list[Tensor] = []
            for step in range(length):
                gate = valid[:, step].unsqueeze(-1).to(encoded.dtype)
                candidate = cell(encoded[:, step], state)
                state = gate * candidate + (1.0 - gate) * state
                outputs.append(state)
            encoded = self.dropout(torch.stack(outputs, dim=1))
        return gather_endpoint(encoded, valid)


class CausalTransformerEncoder(nn.Module):
    """Transformer encoder with a verified triangular causal mask."""

    def __init__(self, config: CausalTemporalConfig) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        encoded, valid = _prepare(value, valid_mask)
        mask_3d = valid.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * mask_3d
        attention_mask = causal_attention_mask(encoded.shape[1], device=encoded.device)
        encoded = self.encoder(encoded, mask=attention_mask)
        encoded = encoded * mask_3d
        return gather_endpoint(encoded, valid)


def build_causal_temporal_encoder(config: CausalTemporalConfig) -> nn.Module:
    """Build one declared causal encoder without silently renaming the family."""

    if config.name == "causal_tcn":
        return CausalTemporalConvEncoder(config)
    if config.name == "causal_gru":
        return CausalMaskedGRUEncoder(config)
    if config.name == "causal_transformer":
        return CausalTransformerEncoder(config)
    raise ValueError(f"unsupported causal temporal encoder={config.name}")


def _prepare(value: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if value.ndim != 3:
        raise ValueError("temporal value must be [B,T,D]")
    valid = _validated_mask(valid_mask)
    if value.shape[:2] != valid.shape:
        raise ValueError(
            f"temporal value shape {tuple(value.shape[:2])} must match mask "
            f"shape {tuple(valid.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("temporal value contains nonfinite entries")
    clean = torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))
    return clean.float(), valid


def _validated_mask(valid_mask: Tensor) -> Tensor:
    if valid_mask.ndim != 2:
        raise ValueError("temporal mask must be [B,T]")
    if valid_mask.dtype != torch.bool:
        if not bool(torch.isfinite(valid_mask).all()):
            raise ValueError("temporal mask contains nonfinite entries")
        if not bool(torch.all((valid_mask == 0) | (valid_mask == 1))):
            raise ValueError("temporal mask must be boolean or strictly 0/1")
    return valid_mask.bool()


__all__ = [
    "CAUSAL_ENCODER_NAMES",
    "SUPPORTED_TARGET_LENGTHS",
    "CausalMaskedGRUEncoder",
    "CausalTemporalConfig",
    "CausalTemporalConvEncoder",
    "CausalTransformerEncoder",
    "CausalityError",
    "build_causal_temporal_encoder",
    "causal_attention_mask",
    "endpoint_index",
    "gather_endpoint",
]
