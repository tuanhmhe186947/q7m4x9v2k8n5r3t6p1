"""Train-only recording-date adversary for the M1-DG1 challenger."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MultitaskFusionClassifier,
    MultitaskFusionOutput,
)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: Any,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Pass values unchanged and reverse only their upstream gradient."""

    return _GradientReversalFunction.apply(value, scale)


class DateDomainHead(nn.Module):
    """Classify one of the proven training recording dates."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_domains: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_domains <= 1:
            raise ValueError("date-domain head dimensions are invalid")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("date-domain hidden tensor must have shape [B,D]")
        return self.layers(hidden)


class DateAdversarialFusionClassifier(MultitaskFusionClassifier):
    """M0 plus a train-only GRL date-domain head on fusion hidden."""

    def __init__(
        self,
        backbone_config: MultimodalFusionConfig,
        *,
        domain_classes: int = 12,
        grl_lambda: float = 0.10,
    ) -> None:
        super().__init__(backbone_config, enable_auxiliary_heads=False)
        if grl_lambda != 0.10:
            raise ValueError("M1-DG1 requires grl_lambda=0.10")
        if domain_classes != 12:
            raise ValueError("M1-DG1 requires exactly 12 train date domains")
        self.grl_lambda = float(grl_lambda)
        self.domain_head = DateDomainHead(
            backbone_config.fusion_hidden_dim,
            hidden_dim=128,
            num_domains=domain_classes,
            dropout=backbone_config.dropout,
        )

    def forward(
        self,
        **model_inputs: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> MultitaskFusionOutput:
        fused = self.backbone.encode_fused(**model_inputs)
        fusion_hidden = self.backbone.classifier[0](fused)
        behavior = self.backbone.classifier[1](fusion_hidden)
        domain = None
        if self.training:
            reversed_hidden = gradient_reverse(
                fusion_hidden,
                scale=1.0,
            )
            domain = self.domain_head(reversed_hidden)
        empty = behavior.new_empty((behavior.shape[0], 0))
        return MultitaskFusionOutput(
            behavior=behavior,
            posture=empty,
            motion_context=empty,
            roi_intent=empty,
            interaction=empty,
            domain=domain,
        )


__all__ = [
    "DateAdversarialFusionClassifier",
    "DateDomainHead",
    "gradient_reverse",
]
