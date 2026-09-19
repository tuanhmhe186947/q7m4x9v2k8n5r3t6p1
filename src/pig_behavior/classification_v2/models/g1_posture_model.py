"""G1 Reviewed Posture Auxiliary Classifier Model.

Wraps the audited M2 multimodal fusion backbone to expose the 256D fused hidden
representation (output of FusionHead) and attach exactly ONE Linear(256, 3)
auxiliary posture head. Behavior logits remain produced by the exact M2 behavior head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)

POSTURE_CLASSES: tuple[str, ...] = ("lying", "sitting", "upright")
POSTURE_LABEL_MAP: dict[str, int] = {c: i for i, c in enumerate(POSTURE_CLASSES)}
POSTURE_HEAD_INPUT_DIM: int = 256
POSTURE_HEAD_OUTPUT_DIM: int = 3
POSTURE_LAMBDA_LOCKED: float = 0.25


@dataclass(frozen=True, slots=True)
class G1ModelOutput:
    """Typed container for G1 behavior and auxiliary posture logits."""

    behavior_logits: torch.Tensor
    posture_logits: torch.Tensor
    fused_embedding: torch.Tensor


class G1PostureAuxiliaryClassifier(nn.Module):
    """M2 multimodal fusion backbone + single Linear(256, 3) auxiliary posture head."""

    def __init__(
        self,
        backbone_config: MultimodalFusionConfig,
    ) -> None:
        super().__init__()
        self.backbone = MultimodalFusionClassifier(backbone_config)
        fusion_hidden_dim = backbone_config.fusion_hidden_dim
        if fusion_hidden_dim != POSTURE_HEAD_INPUT_DIM:
            raise ValueError(
                f"Expected fusion_hidden_dim {POSTURE_HEAD_INPUT_DIM}, got {fusion_hidden_dim}"
            )
        self.posture_head = nn.Linear(POSTURE_HEAD_INPUT_DIM, POSTURE_HEAD_OUTPUT_DIM)

    def forward(
        self,
        **model_inputs: Any,
    ) -> G1ModelOutput:
        """Return behavior logits, posture logits, and fused representation."""
        if "image" not in model_inputs:
            model_inputs["image"] = None
        fused_concat = self.backbone.encode_fused(**model_inputs)
        # Expose 256D fused hidden representation from FusionHead (classifier[0])
        fused_hidden = self.backbone.classifier[0](fused_concat)
        behavior_logits = self.backbone.classifier[1](fused_hidden)
        posture_logits = self.posture_head(fused_hidden)
        return G1ModelOutput(
            behavior_logits=behavior_logits,
            posture_logits=posture_logits,
            fused_embedding=fused_hidden,
        )

    def forward_behavior(
        self,
        **model_inputs: Any,
    ) -> torch.Tensor:
        """Inference-only path returning behavior logits without posture dependencies."""
        if "image" not in model_inputs:
            model_inputs["image"] = None
        fused_concat = self.backbone.encode_fused(**model_inputs)
        return self.backbone.classifier(fused_concat)


def compute_g1_loss(
    behavior_logits: torch.Tensor,
    behavior_targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    posture_logits: torch.Tensor,
    posture_targets: torch.Tensor,
    posture_reviewed_mask: torch.Tensor,
    posture_lambda: float = POSTURE_LAMBDA_LOCKED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined G1 loss: L_total = L_behavior + posture_lambda * L_posture.

    Args:
        behavior_logits: [B, 10] behavior predictions.
        behavior_targets: [B] behavior ground truth labels.
        class_weights: [10] M2 fold-specific behavior class weights.
        posture_logits: [B, 3] auxiliary posture predictions.
        posture_targets: [B] auxiliary posture targets (-1 for unreviewed/censored).
        posture_reviewed_mask: [B] boolean mask (True only for directly reviewed rows).
        posture_lambda: Auxiliary loss weight (locked at 0.25).

    Returns:
        total_loss, loss_behavior, loss_posture
    """
    loss_behavior = F.cross_entropy(
        behavior_logits,
        behavior_targets,
        weight=class_weights,
    )

    if posture_reviewed_mask is not None and posture_reviewed_mask.any():
        active_logits = posture_logits[posture_reviewed_mask]
        active_targets = posture_targets[posture_reviewed_mask]
        loss_posture = F.cross_entropy(active_logits, active_targets)
    else:
        # Exact differentiable zero when no reviewed rows in batch
        loss_posture = 0.0 * posture_logits.sum()

    total_loss = loss_behavior + posture_lambda * loss_posture
    return total_loss, loss_behavior, loss_posture


__all__ = [
    "G1PostureAuxiliaryClassifier",
    "G1ModelOutput",
    "compute_g1_loss",
    "POSTURE_CLASSES",
    "POSTURE_LABEL_MAP",
    "POSTURE_HEAD_INPUT_DIM",
    "POSTURE_HEAD_OUTPUT_DIM",
    "POSTURE_LAMBDA_LOCKED",
]
