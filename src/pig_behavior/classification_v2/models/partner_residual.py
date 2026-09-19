"""M6 Partner-Behavior Residual Models (M6-PB1 and M6-PB2).

Combines frozen M2 actor logits with a learned residual logit correction
derived from inference-available predicted partner behavior probabilities.
- PB1: partner tokens are predicted partner probabilities [10].
- PB2: partner tokens are concat(partner_probs [10], actor_probs [10]) [20].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    PartnerSetEncoder,
)


@dataclass(frozen=True, slots=True)
class PartnerBehaviorResidualConfig:
    num_classes: int = 10
    k: int = 2
    partner_input_dim: int = 10  # 10 for PB1, 20 for PB2
    partner_embedding_dim: int = 32
    dropout: float = 0.0


def build_pb2_partner_tokens(
    partner_probs: torch.Tensor,
    actor_logits: torch.Tensor,
) -> torch.Tensor:
    """Constructs PB2 partner tokens [B, K, 20] from partner_probs and actor_logits.
    
    actor_probs = softmax(actor_logits, dim=-1)
    partner_token = concat([partner_probs, actor_probs], dim=-1)
    """
    if partner_probs.ndim != 3 or partner_probs.shape[-1] != 10:
        raise ValueError(
            f"Expected partner_probs shape [B, K, 10], got {partner_probs.shape}"
        )
    if actor_logits.ndim != 2 or actor_logits.shape[-1] != 10:
        raise ValueError(
            f"Expected actor_logits shape [B, 10], got {actor_logits.shape}"
        )
    
    b, k, _ = partner_probs.shape
    actor_probs = torch.softmax(actor_logits, dim=-1).unsqueeze(1).expand(-1, k, -1)
    return torch.cat([partner_probs, actor_probs], dim=-1)


class PartnerBehaviorResidualHead(nn.Module):
    """Encodes predicted partner behavior tokens into a residual logit adjustment."""

    def __init__(self, config: PartnerBehaviorResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.partner_encoder = PartnerSetEncoder(
            input_dim=config.partner_input_dim,
            embedding_dim=config.partner_embedding_dim,
            dropout=config.dropout,
        )
        self.residual_projection = nn.Linear(
            config.partner_embedding_dim,
            config.num_classes,
        )
        # ZERO INITIALIZATION: ensures initial residual logits are exactly zero
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        partner_tokens: torch.Tensor,
        partner_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual logits [B, 10] from partner tokens [B, K, D] and mask [B, K]."""
        if (
            partner_tokens.ndim != 3
            or partner_tokens.shape[-1] != self.config.partner_input_dim
        ):
            msg = (
                f"Expected partner_tokens shape [B, {self.config.k}, "
                f"{self.config.partner_input_dim}], got {partner_tokens.shape}"
            )
            raise ValueError(msg)
        if partner_mask.ndim != 2 or partner_mask.shape[-1] != self.config.k:
            raise ValueError(
                f"Expected partner_mask shape [B, {self.config.k}], got {partner_mask.shape}"
            )
        
        encoded_partner = self.partner_encoder(
            partner_tokens,
            available_mask=partner_mask,
        )  # [B, partner_embedding_dim]
        
        has_any_partner = partner_mask.any(dim=-1, keepdim=True)  # [B, 1]
        residual_logits = self.residual_projection(encoded_partner)  # [B, 10]
        residual_logits = torch.where(
            has_any_partner, residual_logits, torch.zeros_like(residual_logits)
        )
        return residual_logits


class M6PartnerBehaviorModel(nn.Module):
    """Full M6 model with frozen M2 backbone and trainable residual head."""

    def __init__(
        self,
        frozen_m2_model: nn.Module,
        residual_head_or_config: (
            PartnerBehaviorResidualHead
            | PartnerBehaviorResidualConfig
            | None
        ) = None,
    ) -> None:
        super().__init__()
        self.m2_model = frozen_m2_model
        # Strictly freeze M2 parameters
        for p in self.m2_model.parameters():
            p.requires_grad = False
        self.m2_model.eval()

        if isinstance(residual_head_or_config, PartnerBehaviorResidualHead):
            self.residual_head = residual_head_or_config
        else:
            config = residual_head_or_config or PartnerBehaviorResidualConfig()
            self.residual_head = PartnerBehaviorResidualHead(config)

    def forward(
        self,
        actor_inputs: dict[str, Any],
        partner_tokens: torch.Tensor,
        partner_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (final_logits, actor_m2_logits, residual_logits)."""
        with torch.no_grad():
            actor_m2_logits = self.m2_model(**actor_inputs)
        
        residual_logits = self.residual_head(partner_tokens, partner_mask)
        final_logits = actor_m2_logits + residual_logits
        return final_logits, actor_m2_logits, residual_logits
