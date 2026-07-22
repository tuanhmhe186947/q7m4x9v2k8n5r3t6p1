"""Validated model-mode registry for controlled classification_v2 ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MultitaskFusionClassifier,
)
from pig_behavior.classification_v2.models.temporal_encoders import (
    TEMPORAL_ENCODER_NAMES,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    SUPPORTED_VISUAL_BACKBONES,
    visual_backbone_errors,
)

BASE_GEOMETRY_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
)
MOTION_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
)
PEN_CONTEXT_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "pen_boundary_context",
)
PEN_MOTION_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "pen_boundary_context",
)
ROI_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
)
ALL_SPATIAL_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
    "social_relation",
)
IMPLEMENTED_BACKBONES = SUPPORTED_VISUAL_BACKBONES


@dataclass(frozen=True, slots=True)
class ModelModeSpec:
    """Exact modality contract for one scientifically interpretable mode."""

    name: str
    spatial_feature_groups: tuple[str, ...]
    enable_image: bool
    enable_spatial: bool
    enable_interaction_context: bool
    enable_visual_context: bool
    enable_multitask: bool
    allowed_temporal_encoders: frozenset[str] = TEMPORAL_ENCODER_NAMES


MODEL_MODE_REGISTRY: dict[str, ModelModeSpec] = {
    "actor_only": ModelModeSpec(
        name="actor_only",
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
        allowed_temporal_encoders=frozenset({"masked_mean"}),
    ),
    "actor_temporal": ModelModeSpec(
        name="actor_temporal",
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry": ModelModeSpec(
        name="actor_geometry",
        spatial_feature_groups=BASE_GEOMETRY_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry_motion": ModelModeSpec(
        name="actor_geometry_motion",
        spatial_feature_groups=MOTION_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry_pen": ModelModeSpec(
        name="actor_geometry_pen",
        spatial_feature_groups=PEN_CONTEXT_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry_motion_pen": ModelModeSpec(
        name="actor_geometry_motion_pen",
        spatial_feature_groups=PEN_MOTION_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry_roi": ModelModeSpec(
        name="actor_geometry_roi",
        spatial_feature_groups=ROI_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_geometry_roi_social": ModelModeSpec(
        name="actor_geometry_roi_social",
        spatial_feature_groups=ALL_SPATIAL_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
    "actor_partner_union": ModelModeSpec(
        name="actor_partner_union",
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=True,
        enable_multitask=False,
    ),
    "full_multimodal": ModelModeSpec(
        name="full_multimodal",
        spatial_feature_groups=ALL_SPATIAL_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_multitask=False,
    ),
    "full_multimodal_hierarchy": ModelModeSpec(
        name="full_multimodal_hierarchy",
        spatial_feature_groups=ALL_SPATIAL_GROUPS,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_multitask=True,
    ),
    "spatial_only_control": ModelModeSpec(
        name="spatial_only_control",
        spatial_feature_groups=ALL_SPATIAL_GROUPS,
        enable_image=False,
        enable_spatial=True,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    ),
}
MODEL_MODE_NAMES = frozenset(MODEL_MODE_REGISTRY)


class ModelConfigLike(Protocol):
    """Attributes consumed by the factory without importing training config."""

    model_mode: str
    backbone_name: str
    pretrained_weight_enum: str
    image_size: int
    temporal_encoder_name: str
    hidden_dim: int
    dropout: float
    transformer_layers: int
    transformer_heads: int
    spatial_feature_groups: tuple[str, ...]
    enable_image: bool
    enable_spatial: bool
    enable_interaction_context: bool
    enable_visual_context: bool
    enable_multitask: bool


def model_mode_spec(name: str) -> ModelModeSpec:
    """Return one immutable mode contract or reject an unknown mode."""

    try:
        return MODEL_MODE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model_mode={name}") from exc


def model_mode_errors(config: ModelConfigLike) -> list[str]:
    """Report semantic drift between a declared mode and legacy branch flags."""

    try:
        spec = model_mode_spec(config.model_mode)
    except ValueError as exc:
        return [str(exc)]
    errors: list[str] = []
    observed_groups = tuple(config.spatial_feature_groups)
    if observed_groups != spec.spatial_feature_groups:
        errors.append(
            "model_mode_spatial_groups_mismatch="
            f"mode:{spec.name},expected:{list(spec.spatial_feature_groups)},"
            f"observed:{list(observed_groups)}"
        )
    for field in (
        "enable_image",
        "enable_spatial",
        "enable_interaction_context",
        "enable_visual_context",
        "enable_multitask",
    ):
        expected = getattr(spec, field)
        observed = bool(getattr(config, field))
        if observed != expected:
            errors.append(
                f"model_mode_flag_mismatch={field}:"
                f"expected:{expected},observed:{observed}"
            )
    if config.temporal_encoder_name not in spec.allowed_temporal_encoders:
        errors.append(
            "model_mode_temporal_encoder_mismatch="
            f"mode:{spec.name},encoder:{config.temporal_encoder_name}"
        )
    errors.extend(
        visual_backbone_errors(
            config.backbone_name,
            config.pretrained_weight_enum,
        )
    )
    if config.image_size <= 0:
        errors.append("image_size_must_be_positive")
    return errors


def build_multimodal_model(
    config: ModelConfigLike,
    *,
    spatial_input_dims: dict[str, int],
    interaction_context_dim: int | None,
    num_classes: int,
) -> MultitaskFusionClassifier:
    """Build one mode after exact branch, feature, and tensor-dimension checks."""

    errors = model_mode_errors(config)
    if errors:
        raise ValueError(f"invalid model mode contract: {errors}")
    spec = model_mode_spec(config.model_mode)
    observed_groups = tuple(spatial_input_dims)
    if observed_groups != spec.spatial_feature_groups:
        raise ValueError(
            "model input dimensions do not match mode order: "
            f"expected={list(spec.spatial_feature_groups)}, "
            f"observed={list(observed_groups)}"
        )
    if spec.enable_interaction_context and interaction_context_dim is None:
        raise ValueError("model mode requires interaction_context_dim")
    if not spec.enable_interaction_context:
        interaction_context_dim = None
    return MultitaskFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims=spatial_input_dims,
            num_classes=num_classes,
            interaction_context_dim=interaction_context_dim,
            backbone_name=config.backbone_name,
            pretrained_weight_enum=config.pretrained_weight_enum,
            image_embedding_dim=config.hidden_dim,
            spatial_embedding_dim=config.hidden_dim,
            interaction_embedding_dim=max(8, config.hidden_dim // 2),
            visual_context_embedding_dim=config.hidden_dim,
            fusion_hidden_dim=config.hidden_dim * 2,
            dropout=config.dropout,
            temporal_encoder_name=config.temporal_encoder_name,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            enable_image=spec.enable_image,
            enable_spatial=spec.enable_spatial,
            enable_interaction_context=spec.enable_interaction_context,
            enable_visual_context=spec.enable_visual_context,
        ),
        enable_auxiliary_heads=spec.enable_multitask,
    )


def model_parameter_report(model: nn.Module) -> dict[str, Any]:
    """Return deterministic total/trainable counts by top-level module."""

    by_module = {
        name: int(sum(parameter.numel() for parameter in module.parameters()))
        for name, module in sorted(model.named_children())
    }
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "by_top_level_module": by_module,
    }


def model_mode_contract(name: str) -> dict[str, Any]:
    """Serialize one mode for audits without exposing labels or source metadata."""

    spec = model_mode_spec(name)
    return {
        "model_mode": spec.name,
        "spatial_feature_groups": list(spec.spatial_feature_groups),
        "enable_image": spec.enable_image,
        "enable_spatial": spec.enable_spatial,
        "enable_interaction_context": spec.enable_interaction_context,
        "enable_visual_context": spec.enable_visual_context,
        "enable_multitask": spec.enable_multitask,
        "allowed_temporal_encoders": sorted(spec.allowed_temporal_encoders),
        "availability_encoded_as_behavior_feature": False,
    }


__all__ = [
    "ALL_SPATIAL_GROUPS",
    "BASE_GEOMETRY_GROUPS",
    "IMPLEMENTED_BACKBONES",
    "MODEL_MODE_NAMES",
    "MODEL_MODE_REGISTRY",
    "MOTION_GROUPS",
    "PEN_CONTEXT_GROUPS",
    "PEN_MOTION_GROUPS",
    "ModelModeSpec",
    "ROI_GROUPS",
    "build_multimodal_model",
    "model_mode_contract",
    "model_mode_errors",
    "model_mode_spec",
    "model_parameter_report",
]
