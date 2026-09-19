"""Baseline ladder B0-B3 for the balanced causal main model.

Each baseline is one validated :class:`BalancedModelConfig`. The ladder adds
exactly one thing at a time so that a measured difference is attributable:

* ``B0_ACTOR_SINGLE_FRAME``            actor image at the causal endpoint only;
* ``B1_ACTOR_T6_SEQUENCE``             causal ``T6_TARGET_CONTIGUOUS`` images;
* ``B2_ACTOR_T6_PLUS_GEOMETRY``        B1 + ``bbox_xywh_n`` + ``bbox_shape_n``;
* ``B3_ACTOR_T6_PLUS_GEOMETRY_MOTION`` B2 + canonical ``motion_tensor.v2``.

Geometry and motion go through the grouped numeric encoder, never raw into the
classifier. No baseline uses ROI, social, causal-history or gated-fusion inputs.
"""

from __future__ import annotations

from pig_behavior.classification_v2.models.balanced.balanced_model import (
    BalancedModelConfig,
)
from pig_behavior.classification_v2.models.balanced.contracts import (
    AVAILABILITY_CONTROL_NAMES,
    QUALITY_CONTROL_NAMES,
    BatchContract,
)
from pig_behavior.classification_v2.models.balanced.fusion import FusionConfig
from pig_behavior.classification_v2.models.balanced.numeric import NumericEncoderConfig
from pig_behavior.classification_v2.models.balanced.temporal import (
    CausalTemporalConfig,
)
from pig_behavior.classification_v2.models.balanced.visual import VisualEncoderConfig

BASELINE_NAMES: tuple[str, ...] = (
    "B0_ACTOR_SINGLE_FRAME",
    "B1_ACTOR_T6_SEQUENCE",
    "B2_ACTOR_T6_PLUS_GEOMETRY",
    "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION",
)

#: Temporal view each baseline consumes. ``B0`` reads only the causal endpoint.
BASELINE_TEMPORAL_VIEWS: dict[str, str] = {
    "B0_ACTOR_SINGLE_FRAME": "T6_TARGET_CONTIGUOUS_ENDPOINT_ONLY",
    "B1_ACTOR_T6_SEQUENCE": "T6_TARGET_CONTIGUOUS",
    "B2_ACTOR_T6_PLUS_GEOMETRY": "T6_TARGET_CONTIGUOUS",
    "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION": "T6_TARGET_CONTIGUOUS",
}

BASELINE_NUMERIC_GROUPS: dict[str, tuple[str, ...]] = {
    "B0_ACTOR_SINGLE_FRAME": (),
    "B1_ACTOR_T6_SEQUENCE": (),
    "B2_ACTOR_T6_PLUS_GEOMETRY": ("bbox_xywh_n", "bbox_shape_n"),
    "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION": (
        "bbox_xywh_n",
        "bbox_shape_n",
        "motion_delta",
    ),
}

DEFAULT_TARGET_LENGTH = 6
DEFAULT_HIDDEN_DIM = 64
DEFAULT_TEMPORAL_ENCODER = "causal_tcn"


def baseline_config(
    name: str,
    *,
    target_length: int = DEFAULT_TARGET_LENGTH,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    temporal_encoder: str = DEFAULT_TEMPORAL_ENCODER,
    backbone_name: str = "smoke_cnn",
    pretrained_weight_enum: str = "NONE_RANDOM_INIT",
    image_size: int | None = None,
    dropout: float = 0.0,
    include_controls: bool = True,
) -> BalancedModelConfig:
    """Return one validated baseline configuration.

    ``target_length`` is ignored for ``B0``, which is single-frame by
    definition. Every other baseline may be re-instantiated at T8/T12/T16 for
    the target-view ablation without changing any other component.
    """

    if name not in BASELINE_NAMES:
        raise ValueError(
            f"unknown baseline={name}; expected one of {list(BASELINE_NAMES)}"
        )
    groups = BASELINE_NUMERIC_GROUPS[name]
    single_frame = name == "B0_ACTOR_SINGLE_FRAME"
    effective_length = 1 if single_frame else target_length
    if effective_length <= 0:
        raise ValueError("target_length must be positive")

    control_names = tuple(QUALITY_CONTROL_NAMES) if include_controls else ()
    availability_names = tuple(AVAILABILITY_CONTROL_NAMES) if include_controls else ()

    contract = BatchContract(
        required_modalities=("actor_images", *groups),
        target_length=effective_length,
        history_length=0,
        image_size=image_size,
        maskable_modalities=(),
    )
    numeric = (
        NumericEncoderConfig(
            groups=groups,
            embedding_dim=max(8, hidden_dim // 4),
            dropout=dropout,
        )
        if groups
        else None
    )
    temporal = (
        None
        if single_frame
        else CausalTemporalConfig(
            name=temporal_encoder,
            hidden_dim=hidden_dim,
            layers=2,
            dropout=dropout,
        )
    )
    return BalancedModelConfig(
        name=name,
        batch_contract=contract,
        visual=VisualEncoderConfig(
            backbone_name=backbone_name,
            pretrained_weight_enum=pretrained_weight_enum,
            embedding_dim=hidden_dim,
            dropout=dropout,
        ),
        numeric=numeric,
        temporal=temporal,
        fusion=FusionConfig(
            mode="concat_projection",
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        ),
        hidden_dim=hidden_dim,
        control_names=control_names,
        availability_names=availability_names,
        enable_auxiliary_head=False,
    )


def baseline_contract(name: str) -> dict[str, object]:
    """Serialize a baseline definition for audits without building weights."""

    config = baseline_config(name)
    return {
        "baseline": name,
        "temporal_view": BASELINE_TEMPORAL_VIEWS[name],
        "numeric_groups": list(BASELINE_NUMERIC_GROUPS[name]),
        "uses_roi_relation": False,
        "uses_social_relation": False,
        "uses_causal_history": False,
        "uses_gated_fusion": False,
        "geometry_encoded_through_grouped_encoder": bool(
            BASELINE_NUMERIC_GROUPS[name]
        ),
        "model_config": config.to_payload(),
    }


__all__ = [
    "BASELINE_NAMES",
    "BASELINE_NUMERIC_GROUPS",
    "BASELINE_TEMPORAL_VIEWS",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_TARGET_LENGTH",
    "DEFAULT_TEMPORAL_ENCODER",
    "baseline_config",
    "baseline_contract",
]
