"""M8-ROI1 & M8-ROI2 Targeted ROI-Conditioned Residual Head.

Provides targeted, label-independent resource-context residual branches
operating on canonical 6-slot ROI18 + validity features (and optional M2
actor_probs conditioning for ROI2) without mutating or fine-tuning the
frozen M2 visual classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Canonical 10 behaviors order:
# 0: drink, 1: eat, 2: fight, 3: social-nose, 4: explore,
# 5: lying, 6: stand, 7: move, 8: sitting, 9: playwithtoy
RESOURCE_CLASS_NAMES: tuple[str, ...] = ("drink", "eat", "explore", "playwithtoy")
RESOURCE_CLASS_INDICES: tuple[int, ...] = (0, 1, 4, 9)
NON_RESOURCE_CLASS_INDICES: tuple[int, ...] = (2, 3, 5, 6, 7, 8)


@dataclass(frozen=True, slots=True)
class ROIResidualConfig:
    """Immutable configuration for TargetedROIResidualHead."""

    num_classes: int = 10
    temporal_slots: int = 6
    roi_feature_dim: int = 18
    roi_mask_dim: int = 3
    slot_input_dim: int = 21  # 18 + 3
    slot_hidden_dim: int = 32
    combined_hidden_dim: int = 32
    num_resource_classes: int = 4
    condition_on_actor_probs: bool = False
    actor_probs_dim: int = 10
    dropout: float = 0.0

    @property
    def combined_input_dim(self) -> int:
        base_dim = self.temporal_slots * self.slot_hidden_dim  # 192
        if self.condition_on_actor_probs:
            return base_dim + self.actor_probs_dim  # 202
        return base_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "temporal_slots": self.temporal_slots,
            "roi_feature_dim": self.roi_feature_dim,
            "roi_mask_dim": self.roi_mask_dim,
            "slot_input_dim": self.slot_input_dim,
            "slot_hidden_dim": self.slot_hidden_dim,
            "combined_hidden_dim": self.combined_hidden_dim,
            "num_resource_classes": self.num_resource_classes,
            "condition_on_actor_probs": self.condition_on_actor_probs,
            "actor_probs_dim": self.actor_probs_dim,
            "dropout": self.dropout,
        }


class TargetedROIResidualHead(nn.Module):
    """Targeted ROI-conditioned Residual Head.

    ROI1 (condition_on_actor_probs=False): 7,012 parameters
      Per slot (6 ordered slots):
        21D -> Linear(21, 32) -> GELU -> [32]
      Concat 6 slots:
        192D -> Linear(192, 32) -> GELU -> Linear(32, 4) -> [4]

    ROI2 (condition_on_actor_probs=True): 7,332 parameters
      Per slot (6 ordered slots):
        21D -> Linear(21, 32) -> GELU -> [32]
      Concat 6 slots + actor_probs:
        (192 + 10) = 202D -> Linear(202, 32) -> GELU -> Linear(32, 4) -> [4]

    Targeted Residual Injection:
      residual[drink] = roi_res[0]
      residual[eat] = roi_res[1]
      residual[explore] = roi_res[2]
      residual[playwithtoy] = roi_res[3]
      residual[other 6 classes] = EXACT ZERO
    """

    def __init__(self, config: ROIResidualConfig | None = None) -> None:
        super().__init__()
        self.config = config or ROIResidualConfig()

        self.slot_encoder = nn.Linear(
            self.config.slot_input_dim,
            self.config.slot_hidden_dim,
        )
        self.slot_act = nn.GELU()

        self.combined_fc1 = nn.Linear(
            self.config.combined_input_dim,
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
            self.config.num_resource_classes,
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
        roi_class_relation: torch.Tensor,
        roi_validity_mask: torch.Tensor,
        actor_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute full 10-class residual logits with exact zero on non-resource classes.

        Args:
            roi_class_relation: [B, 6, 18] ROI class relation tensor.
            roi_validity_mask: [B, 6, 3] ROI validity mask tensor.
            actor_probs: [B, 10] optional M2 actor probabilities for ROI2 conditioning.

        Returns:
            full_residual: [B, 10] additive residual logits with exact zero
              on 6 non-resource classes.
        """
        B, T, D_roi = roi_class_relation.shape
        _, _, D_mask = roi_validity_mask.shape

        if T != self.config.temporal_slots or D_roi != self.config.roi_feature_dim:
            raise ValueError(
                f"Expected roi_class_relation [B, {self.config.temporal_slots}, "
                f"{self.config.roi_feature_dim}], got {list(roi_class_relation.shape)}"
            )
        if D_mask != self.config.roi_mask_dim:
            raise ValueError(
                f"Expected roi_validity_mask [B, {self.config.temporal_slots}, "
                f"{self.config.roi_mask_dim}], got {list(roi_validity_mask.shape)}"
            )

        # Construct 21D per-slot input: [B, 6, 21]
        roi_slot = torch.cat([roi_class_relation, roi_validity_mask], dim=-1)

        # Per-slot encoding: [B, 6, 32]
        slot_encoded = self.slot_act(self.slot_encoder(roi_slot))

        # Concatenate 6 slots: [B, 192]
        roi_ordered_emb = slot_encoded.reshape(B, -1)

        if self.config.condition_on_actor_probs:
            if actor_probs is None:
                raise ValueError("actor_probs is required when condition_on_actor_probs=True")
            if actor_probs.shape != (B, self.config.actor_probs_dim):
                raise ValueError(
                    f"Expected actor_probs [B, {self.config.actor_probs_dim}], "
                    f"got {list(actor_probs.shape)}"
                )
            combined = torch.cat([roi_ordered_emb, actor_probs], dim=-1)  # [B, 202]
        else:
            combined = roi_ordered_emb  # [B, 192]

        # MLP layers: [B, 32] -> [B, 4]
        hidden = self.combined_act(self.combined_fc1(combined))
        hidden = self.dropout(hidden)
        resource_residual = self.head(hidden)  # [B, 4]

        # Construct full [B, 10] residual
        full_residual = torch.zeros(
            (B, self.config.num_classes),
            dtype=roi_class_relation.dtype,
            device=roi_class_relation.device,
        )
        full_residual[:, list(RESOURCE_CLASS_INDICES)] = resource_residual

        return full_residual


def combine_m8_logits(
    actor_logits: torch.Tensor,
    full_roi_residual: torch.Tensor,
) -> torch.Tensor:
    """Combine frozen M2 actor logits with targeted ROI residual.

    final_logits = actor_logits + full_roi_residual
    """
    return actor_logits + full_roi_residual


__all__ = [
    "ROIResidualConfig",
    "TargetedROIResidualHead",
    "combine_m8_logits",
    "RESOURCE_CLASS_NAMES",
    "RESOURCE_CLASS_INDICES",
    "NON_RESOURCE_CLASS_INDICES",
]
