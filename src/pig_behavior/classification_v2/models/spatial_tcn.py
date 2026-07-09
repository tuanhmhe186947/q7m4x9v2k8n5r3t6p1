"""Small mask-aware spatial-temporal baseline for classification_v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class SpatialTCNConfig:
    input_dims: dict[str, int]
    num_classes: int
    hidden_dim: int = 64
    dropout: float = 0.1


class SpatialTCNClassifier(nn.Module):
    """Project spatial feature groups per frame, pool with masks, classify behavior.

    The first implementation intentionally uses kernel-size-1 temporal blocks.
    That keeps padded slots fully isolated for the loader/model mask-invariance
    smoke test. Wider temporal kernels can be added later with masked
    convolution semantics.
    """

    def __init__(self, config: SpatialTCNConfig) -> None:
        super().__init__()
        if not config.input_dims:
            raise ValueError("input_dims must not be empty")
        if config.num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        self.config = config
        branch_dim = max(8, config.hidden_dim // max(1, len(config.input_dims)))
        self.branch_order = tuple(sorted(config.input_dims))
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
        self.temporal = nn.Sequential(
            nn.Conv1d(fused_dim, config.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        missing = [name for name in self.branch_order if name not in features]
        if missing:
            raise ValueError(f"Missing spatial feature groups: {missing}")
        if length_mask.ndim != 2:
            raise ValueError("length_mask must have shape [B, T]")
        mask = length_mask.float()
        if observed_mask is not None:
            if observed_mask.shape != length_mask.shape:
                raise ValueError("observed_mask must match length_mask shape")
            mask = mask * observed_mask.float()
        mask_3d = mask.unsqueeze(-1)

        projected = []
        for name in self.branch_order:
            x = features[name].float()
            if x.ndim != 3:
                raise ValueError(f"{name} must have shape [B, T, D]")
            projected.append(self.branches[name](x) * mask_3d)
        fused = torch.cat(projected, dim=-1)
        temporal = self.temporal(fused.transpose(1, 2)).transpose(1, 2) * mask_3d
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = temporal.sum(dim=1) / denom
        return self.classifier(pooled)
