"""M7-H5S Structured History Residual Head (Causal H5 + Frozen M2 T6).

Provides a lightweight, causal structured history residual context branch
operating strictly on pre-target same-actor observations (H5) without mutating
or fine-tuning the frozen M2 visual classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class StructuredH5ResidualConfig:
    """Immutable configuration for StructuredH5ResidualHead."""

    num_classes: int = 10
    history_length: int = 5
    feature_dim: int = 46
    slot_hidden_dim: int = 32
    combined_hidden_dim: int = 32
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "history_length": self.history_length,
            "feature_dim": self.feature_dim,
            "slot_hidden_dim": self.slot_hidden_dim,
            "combined_hidden_dim": self.combined_hidden_dim,
            "dropout": self.dropout,
        }


class StructuredH5ResidualHead(nn.Module):
    """Causal Structured H5 Residual Head (6,986 parameters, zero-initialized).

    Architecture:
      Per history slot (5 slots):
        46D mask-zeroed -> Linear(46, 32) -> GELU -> [32]
      Concat 5 slots:
        160D -> Linear(160, 32) -> GELU -> Linear(32, 10) -> [10]

    Exact parameter count:
      slot_encoder: 46 * 32 + 32 = 1,504
      combined_fc1: 160 * 32 + 32 = 5,152
      head: 32 * 10 + 10 = 330
      Total: 6,986 parameters.
    """

    def __init__(self, config: StructuredH5ResidualConfig | None = None) -> None:
        super().__init__()
        self.config = config or StructuredH5ResidualConfig()

        self.slot_encoder = nn.Linear(
            self.config.feature_dim,
            self.config.slot_hidden_dim,
        )
        self.slot_act = nn.GELU()

        concat_dim = self.config.history_length * self.config.slot_hidden_dim
        self.combined_fc1 = nn.Linear(
            concat_dim,
            self.config.combined_hidden_dim,
        )
        self.combined_act = nn.GELU()

        self.dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.dropout > 0.0
            else nn.Identity()
        )
        self.head = nn.Linear(
            self.config.combined_hidden_dim,
            self.config.num_classes,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize slot and hidden layers with standard init, head to exact zero."""
        nn.init.kaiming_uniform_(self.slot_encoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.slot_encoder.bias)

        nn.init.kaiming_uniform_(self.combined_fc1.weight, nonlinearity="linear")
        nn.init.zeros_(self.combined_fc1.bias)

        # Zero-initialization for exact zero residual at initialization
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        h5_structured: torch.Tensor,
        history_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute structured history residual logits [B, num_classes].

        Args:
            h5_structured: [B, 5, 46] structured history features.
            history_observed_mask: [B, 5] boolean or float mask indicating observed slots.

        Returns:
            residual_logits: [B, 10] additive residual logits.
        """
        B, T, D = h5_structured.shape
        if T != self.config.history_length or D != self.config.feature_dim:
            raise ValueError(
                f"Expected h5_structured of shape [B, {self.config.history_length}, "
                f"{self.config.feature_dim}], got {list(h5_structured.shape)}"
            )

        if history_observed_mask is not None:
            mask_2d = history_observed_mask.bool()
            mask_3d = mask_2d.unsqueeze(-1).to(dtype=h5_structured.dtype)
            h5_input = h5_structured * mask_3d
        else:
            mask_2d = torch.ones((B, T), dtype=torch.bool, device=h5_structured.device)
            mask_3d = torch.ones((B, T, 1), dtype=h5_structured.dtype, device=h5_structured.device)
            h5_input = h5_structured

        # Per-slot encoding: [B, 5, 32]
        slot_encoded = self.slot_act(self.slot_encoder(h5_input))
        slot_encoded = slot_encoded * mask_3d

        # Concatenate 5 slots: [B, 160]
        combined = slot_encoded.reshape(B, -1)

        # MLP layers: [B, 32] -> [B, 10]
        hidden = self.combined_act(self.combined_fc1(combined))
        hidden = self.dropout(hidden)
        residual = self.head(hidden)

        # Strict fail-closed short-circuit: rows with no history get exact zero tensor
        has_history = mask_2d.any(dim=-1, keepdim=True)  # [B, 1]
        residual = torch.where(has_history, residual, torch.zeros_like(residual))

        return residual


def combine_m7_logits(
    actor_logits: torch.Tensor,
    history_residual: torch.Tensor,
) -> torch.Tensor:
    """Combine frozen M2 actor logits with history residual.

    final_logits = actor_logits + history_residual
    """
    return actor_logits + history_residual


__all__ = [
    "StructuredH5ResidualConfig",
    "StructuredH5ResidualHead",
    "combine_m7_logits",
]
