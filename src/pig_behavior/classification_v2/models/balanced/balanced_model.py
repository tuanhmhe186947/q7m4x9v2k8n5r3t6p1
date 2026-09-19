"""Composable balanced causal model used by the B0-B3 baseline ladder.

The same module implements every baseline; a baseline is a *configuration*, not
a separate class. That keeps the visual, numeric, temporal, fusion and head
interfaces identical across the ladder so an ablation difference cannot be
confounded by an implementation difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor, nn

from pig_behavior.classification_v2.models.balanced.contracts import (
    BatchContract,
    ModelBatch,
    SequenceSegment,
    require_batch,
)
from pig_behavior.classification_v2.models.balanced.fusion import (
    BehaviorClassificationHead,
    FusionConfig,
    MaskedAuxiliaryHead,
    MultimodalFusion,
)
from pig_behavior.classification_v2.models.balanced.numeric import (
    ControlMaskEncoder,
    GroupedNumericEncoder,
    ModalityAvailability,
    NumericEncoderConfig,
)
from pig_behavior.classification_v2.models.balanced.temporal import (
    CausalTemporalConfig,
    build_causal_temporal_encoder,
    gather_endpoint,
)
from pig_behavior.classification_v2.models.balanced.visual import (
    SharedFrameVisualEncoder,
    VisualEncoderConfig,
)


@dataclass(frozen=True, slots=True)
class BalancedModelConfig:
    """Fully validated configuration for one balanced-ladder model."""

    name: str
    batch_contract: BatchContract
    visual: VisualEncoderConfig | None = None
    numeric: NumericEncoderConfig | None = None
    temporal: CausalTemporalConfig | None = None
    fusion: FusionConfig = field(default_factory=FusionConfig)
    hidden_dim: int = 64
    control_names: tuple[str, ...] = ()
    availability_names: tuple[str, ...] = ()
    enable_auxiliary_head: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be blank")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.visual is None and self.numeric is None:
            raise ValueError("a model needs at least one predictive branch")
        required = set(self.batch_contract.required_modalities)
        if self.visual is not None and "actor_images" not in required:
            raise ValueError(
                "visual branch is configured but actor_images is not a required "
                "modality of the batch contract"
            )
        if self.numeric is not None:
            declared = set(self.numeric.groups)
            missing = sorted(declared - required)
            if missing:
                raise ValueError(
                    f"numeric groups {missing} are not declared required "
                    "modalities of the batch contract"
                )
        if self.temporal is None and self.batch_contract.target_length != 1:
            raise ValueError(
                "a model without a temporal encoder must declare target_length=1; "
                f"observed target_length={self.batch_contract.target_length}"
            )
        if self.temporal is not None and self.temporal.hidden_dim != self.hidden_dim:
            raise ValueError(
                "temporal hidden_dim must match model hidden_dim: "
                f"{self.temporal.hidden_dim} != {self.hidden_dim}"
            )

    def with_overrides(self, **overrides: Any) -> BalancedModelConfig:
        return replace(self, **overrides)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_dim": self.hidden_dim,
            "target_length": self.batch_contract.target_length,
            "history_length": self.batch_contract.history_length,
            "required_modalities": list(self.batch_contract.required_modalities),
            "maskable_modalities": list(self.batch_contract.maskable_modalities),
            "visual": None if self.visual is None else self.visual.to_payload(),
            "numeric": None if self.numeric is None else self.numeric.to_payload(),
            "temporal": None if self.temporal is None else self.temporal.to_payload(),
            "fusion": self.fusion.to_payload(),
            "control_names": list(self.control_names),
            "availability_names": list(self.availability_names),
            "enable_auxiliary_head": self.enable_auxiliary_head,
            "num_classes": self.batch_contract.num_classes,
            "future_frame_dependence": 0,
        }


class BalancedCausalModel(nn.Module):
    """Visual + grouped-numeric causal model with a ten-class behavior head."""

    def __init__(self, config: BalancedModelConfig) -> None:
        super().__init__()
        self.config = config
        branch_dims: dict[str, int] = {}

        self.visual_encoder: SharedFrameVisualEncoder | None = None
        self.visual_projection: nn.Module | None = None
        self.visual_temporal: nn.Module | None = None
        if config.visual is not None:
            self.visual_encoder = SharedFrameVisualEncoder(config.visual)
            self.visual_projection = nn.Linear(
                self.visual_encoder.output_dim,
                config.hidden_dim,
            )
            self.visual_temporal = self._build_temporal(config)
            branch_dims["visual"] = config.hidden_dim

        self.numeric_encoder: GroupedNumericEncoder | None = None
        self.numeric_projection: nn.Module | None = None
        self.numeric_temporal: nn.Module | None = None
        if config.numeric is not None and config.numeric.groups:
            self.numeric_encoder = GroupedNumericEncoder(config.numeric)
            self.numeric_projection = nn.Linear(
                self.numeric_encoder.output_dim,
                config.hidden_dim,
            )
            self.numeric_temporal = self._build_temporal(config)
            branch_dims["numeric"] = config.hidden_dim

        control_embedding = 0
        if config.numeric is not None:
            control_embedding = config.numeric.control_embedding_dim
        self.control_encoder = ControlMaskEncoder(
            len(config.control_names),
            control_embedding if config.control_names else 0,
        )
        self.availability = ModalityAvailability(names=config.availability_names)
        control_dim = self.control_encoder.output_dim + len(config.availability_names)

        self.fusion = MultimodalFusion(
            config.fusion,
            branch_dims=branch_dims,
            control_dim=control_dim,
        )
        self.head = BehaviorClassificationHead(
            self.fusion.output_dim,
            config.batch_contract.num_classes,
        )
        self.auxiliary_head: MaskedAuxiliaryHead | None = None
        if config.enable_auxiliary_head:
            self.auxiliary_head = MaskedAuxiliaryHead(self.fusion.output_dim)

    @staticmethod
    def _build_temporal(config: BalancedModelConfig) -> nn.Module | None:
        if config.temporal is None:
            return None
        return build_causal_temporal_encoder(config.temporal)

    def forward(
        self,
        batch: ModelBatch,
        *,
        validate: bool = True,
    ) -> dict[str, Tensor]:
        if validate:
            require_batch(batch, self.config.batch_contract)
        segment = batch.target
        valid = segment.valid_mask.bool()
        branches: dict[str, Tensor] = {}

        if self.visual_encoder is not None:
            if segment.images is None:
                raise ValueError(
                    "model requires actor_images but the target segment has none"
                )
            encoded = self.visual_encoder(segment.images, valid)
            projected = self.visual_projection(encoded)
            branches["visual"] = self._temporal_readout(
                self.visual_temporal,
                projected,
                valid,
            )

        if self.numeric_encoder is not None:
            encoded = self.numeric_encoder(segment.numeric_groups, valid)
            projected = self.numeric_projection(encoded)
            branches["numeric"] = self._temporal_readout(
                self.numeric_temporal,
                projected,
                valid,
            )

        controls = self._controls(batch, segment, valid)
        fused = self.fusion(branches, controls)
        outputs = {"logits": self.head(fused), "fused": fused}
        if self.auxiliary_head is not None:
            mask = batch.modality_availability.get("auxiliary_supervision")
            if mask is None:
                mask = torch.zeros(
                    batch.batch_size,
                    dtype=torch.bool,
                    device=fused.device,
                )
            auxiliary = self.auxiliary_head(fused, mask)
            outputs["auxiliary_logits"] = auxiliary["logits"]
            outputs["auxiliary_supervision_mask"] = auxiliary["supervision_mask"]
        return outputs

    def _controls(
        self,
        batch: ModelBatch,
        segment: SequenceSegment,
        valid: Tensor,
    ) -> Tensor:
        parts: list[Tensor] = []
        quality = self.control_encoder(segment.quality_mask, valid)
        if quality.shape[-1]:
            parts.append(gather_endpoint(quality, valid))
        availability = self.availability.vector(
            batch.modality_availability,
            batch_size=batch.batch_size,
            device=valid.device,
        )
        if availability.shape[-1]:
            parts.append(availability)
        if not parts:
            return torch.zeros(
                (batch.batch_size, 0),
                dtype=torch.float32,
                device=valid.device,
            )
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _temporal_readout(
        encoder: nn.Module | None,
        sequence: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if encoder is None:
            return gather_endpoint(sequence, valid)
        return encoder(sequence, valid)

    def parameter_report(self) -> dict[str, Any]:
        by_module = {
            name: int(sum(p.numel() for p in module.parameters()))
            for name, module in sorted(self.named_children())
        }
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "by_top_level_module": by_module,
        }


__all__ = [
    "BalancedCausalModel",
    "BalancedModelConfig",
]
