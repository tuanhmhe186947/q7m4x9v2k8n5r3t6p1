"""Final Joint challenger with one deep local spatiotemporal RGB residual."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pig_behavior.classification_v2.models.deep_local_spatiotemporal import (
    LOCAL_COMBINED_DIM,
    LOCAL_CONTEXT_DIM,
    SourceAwareLocalSpatiotemporalBranch,
    infer_resnet_stage_channels,
    merge_temporal_masks,
    spatial_tap_modules,
)
from pig_behavior.classification_v2.models.joint_representation_model import (
    JointModelOutput,
    JointRepresentationClassifier,
)

DEEP_LOCAL_ARCHITECTURE_VERSION = "final_joint_ms_actor_union_localst_residual_v1"


class DeepLocalJointRepresentationClassifier(JointRepresentationClassifier):
    """Preserve Final Joint and add source-aware local actor/union reasoning."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if self.image_encoder is None or self.visual_context_encoder is None:
            raise ValueError("deep local challenger requires actor and union RGB branches")

        actor_channels = infer_resnet_stage_channels(self.image_encoder.frame_encoder)
        union_channels = infer_resnet_stage_channels(
            self.visual_context_encoder.frame_encoder
        )
        self.local_branch = SourceAwareLocalSpatiotemporalBranch(
            {"layer3": actor_channels[0], "layer4": actor_channels[1]},
            {"layer3": union_channels[0], "layer4": union_channels[1]},
            dropout=config.dropout,
            temporal_encoder_name=config.temporal_encoder_name,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
        )
        self.local_residual_projection = nn.Linear(
            LOCAL_COMBINED_DIM,
            self.fused_embedding_dim,
            bias=False,
        )
        # A tiny nonzero projection gives the branch a live gradient while
        # keeping the checkpoint-warm-start logits within the step-0 bound.
        nn.init.normal_(
            self.local_residual_projection.weight,
            mean=0.0,
            std=1e-10,
        )
        self._latest_spatial_maps: dict[str, dict[str, torch.Tensor]] = {
            "actor": {},
            "union": {},
        }
        self._latest_local_representation: torch.Tensor | None = None
        self._tap_handles = []
        self._register_spatial_taps(
            "actor",
            self.image_encoder.frame_encoder,
        )
        self._register_spatial_taps(
            "union",
            self.visual_context_encoder.frame_encoder,
        )

    def _register_spatial_taps(
        self,
        stream_name: str,
        frame_encoder: nn.Module,
    ) -> None:
        for stage_name, module in spatial_tap_modules(frame_encoder).items():
            self._tap_handles.append(
                module.register_forward_hook(
                    self._make_spatial_hook(stream_name, stage_name)
                )
            )

    def _make_spatial_hook(self, stream_name: str, stage_name: str):
        def capture(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{stream_name} {stage_name} output is not a tensor")
            self._latest_spatial_maps[stream_name][stage_name] = output

        return capture

    def forward(self, **model_inputs: Any) -> JointModelOutput:
        self._latest_spatial_maps = {"actor": {}, "union": {}}
        self._latest_local_representation = None
        return super().forward(**model_inputs)

    def _augment_fused_embedding(
        self,
        adapted_fused: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        m2_branches = kwargs["m2_branches"]
        context = torch.cat(
            [
                m2_branches.structured128,
                kwargs["h5_context"],
                kwargs["roi_context"],
                kwargs["posture_context"],
                kwargs["partner_context"],
                kwargs["partner_hidden_context"],
                m2_branches.interaction64,
            ],
            dim=-1,
        )
        if context.shape[-1] != LOCAL_CONTEXT_DIM:
            raise RuntimeError(
                f"source-aware local context width drifted: {context.shape[-1]}"
            )

        length_mask = kwargs["length_mask"]
        observed_mask = kwargs["observed_mask"]
        actor_length = (
            kwargs["image_length_mask"]
            if kwargs["image_length_mask"] is not None
            else length_mask
        )
        actor_observed = (
            kwargs["image_observed_mask"]
            if kwargs["image_observed_mask"] is not None
            else observed_mask
        )
        actor_mask = merge_temporal_masks(
            actor_length,
            actor_observed,
            kwargs["image_available_mask"],
            kwargs["image_quality_mask"],
            expected_shape=tuple(actor_length.shape),
            name="actor_local",
        )

        union_length = (
            kwargs["visual_context_length_mask"]
            if kwargs["visual_context_length_mask"] is not None
            else length_mask
        )
        union_observed = (
            kwargs["visual_context_observed_mask"]
            if kwargs["visual_context_observed_mask"] is not None
            else observed_mask
        )
        union_mask = merge_temporal_masks(
            union_length,
            union_observed,
            kwargs["visual_context_available_mask"],
            kwargs["visual_context_quality_mask"],
            expected_shape=tuple(union_length.shape),
            name="union_local",
        )

        actor_time_delta = kwargs["image_time_delta"]
        union_time_delta = kwargs["visual_context_time_delta"]
        if actor_time_delta is None:
            actor_time_delta = torch.zeros_like(actor_mask, dtype=torch.float32)
        if union_time_delta is None:
            union_time_delta = torch.zeros_like(union_mask, dtype=torch.float32)
        local_representation = self.local_branch(
            actor_maps=self._latest_spatial_maps["actor"],
            union_maps=self._latest_spatial_maps["union"],
            context=context,
            actor_temporal_mask=actor_mask,
            union_temporal_mask=union_mask,
            actor_time_delta=actor_time_delta,
            union_time_delta=union_time_delta,
        )
        self._latest_local_representation = local_representation
        return adapted_fused + self.local_residual_projection(local_representation)


__all__ = [
    "DEEP_LOCAL_ARCHITECTURE_VERSION",
    "DeepLocalJointRepresentationClassifier",
]
