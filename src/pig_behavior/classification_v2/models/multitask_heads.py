"""Auxiliary prediction heads for classification_v2 multitask training.

The heads are small reusable modules that sit on top of a fused representation.
They do not decide which targets are valid; masks from ``y_auxiliary_targets``
and the training loss control whether a task contributes for each sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

AUXILIARY_LABEL_ORDER: dict[str, tuple[str, ...]] = {
    "posture": ("lying", "sitting", "standing_or_other"),
    "motion_context": ("move", "explore", "stand", "other"),
    "roi_intent": ("eat", "drink", "playwithtoy", "none"),
    "interaction": ("fight", "social-nose", "none"),
}


@dataclass(frozen=True, slots=True)
class AuxiliaryHeadConfig:
    """Shape contract for one auxiliary classification head."""

    name: str
    num_classes: int


class AuxiliaryPredictionHeads(nn.Module):
    """Project one fused embedding into multiple auxiliary logits."""

    def __init__(self, *, input_dim: int, heads: list[AuxiliaryHeadConfig]) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not heads:
            raise ValueError("heads must not be empty")
        names = [head.name for head in heads]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate auxiliary head names: {names}")
        for head in heads:
            if head.num_classes <= 1:
                raise ValueError(f"{head.name} num_classes must be greater than 1")
        self.input_dim = int(input_dim)
        self.heads = nn.ModuleDict({head.name: nn.Linear(input_dim, head.num_classes) for head in heads})

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{task_name: logits}`` for a fused embedding batch."""
        if embedding.ndim != 2:
            raise ValueError("embedding must have shape [B, D]")
        if embedding.shape[1] != self.input_dim:
            raise ValueError(f"embedding dim {embedding.shape[1]} does not match {self.input_dim}")
        return {name: head(embedding.float()) for name, head in self.heads.items()}
