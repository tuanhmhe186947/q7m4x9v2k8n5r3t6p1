"""Small mask-aware spatial-temporal baseline for classification_v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

_FEATURE_MASK_REQUIRED_GROUPS = frozenset(
    {"motion_delta", "social_relation"}
)


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
        self.mask_branches = nn.ModuleDict(
            {
                name: nn.Linear(dim, branch_dim, bias=False)
                for name, dim in sorted(config.input_dims.items())
                if name in _FEATURE_MASK_REQUIRED_GROUPS
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
        feature_validity_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        missing = [name for name in self.branch_order if name not in features]
        if missing:
            raise ValueError(f"Missing spatial feature groups: {missing}")
        required_masks = _FEATURE_MASK_REQUIRED_GROUPS.intersection(
            self.branch_order
        )
        provided_masks = (
            set()
            if feature_validity_masks is None
            else set(feature_validity_masks)
        )
        missing_masks = sorted(required_masks.difference(provided_masks))
        if missing_masks:
            raise ValueError(
                "Missing spatial feature validity masks: "
                f"{missing_masks}"
            )
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
            explicit = (
                None
                if feature_validity_masks is None
                else feature_validity_masks.get(name)
            )
            if explicit is None:
                feature_mask = mask_3d.expand_as(x)
                mask_delta = None
            else:
                if name not in self.mask_branches:
                    raise ValueError(
                        f"Unexpected feature validity mask for {name}"
                    )
                if explicit.shape != x.shape:
                    raise ValueError(
                        f"{name} feature validity shape does not match values: "
                        f"{tuple(explicit.shape)}:{tuple(x.shape)}"
                    )
                feature_mask = mask_3d * explicit.float()
                mask_delta = feature_mask - mask_3d
            clean = torch.where(feature_mask.bool(), x, torch.zeros_like(x))
            branch = self.branches[name](clean)
            branch = branch * feature_mask.any(dim=-1, keepdim=True).to(
                branch.dtype
            )
            projected.append(
                branch
                if mask_delta is None
                else branch + self.mask_branches[name](mask_delta)
            )
        fused = torch.cat(projected, dim=-1)
        temporal = self.temporal(fused.transpose(1, 2)).transpose(1, 2) * mask_3d
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = temporal.sum(dim=1) / denom
        return self.classifier(pooled)
