from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    DEFAULT_TEMPORAL_TIERS,
)
from pig_behavior.classification_v2.models.model_factory import (
    MODEL_MODE_NAMES,
    build_multimodal_model,
    model_mode_contract,
    model_mode_spec,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.temporal_encoders import (
    TEMPORAL_ENCODER_NAMES,
)
from pig_behavior.classification_v2.training.config import ModelConfig

GROUP_DIMS = {
    "bbox_xywh_n": 4,
    "bbox_shape_n": 3,
    "motion_delta": 5,
    "roi_class_relation": 6,
    "social_relation": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model factory modes using synthetic masked tensors."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/model_factory_audit.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_model_factory_audit()
    if not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def run_model_factory_audit() -> dict[str, Any]:
    """Build every declared mode and temporal family without model training."""

    torch.manual_seed(20260714)
    errors: list[str] = []
    modes: list[dict[str, Any]] = []
    for mode in sorted(MODEL_MODE_NAMES):
        config = _config(mode)
        try:
            model = _build(config).eval()
            output = model(**_inputs(config))
            shape = list(output.behavior.shape)
            finite = bool(torch.isfinite(output.behavior).all())
            if shape != [2, 10] or not finite:
                errors.append(f"mode_forward_invalid={mode}:{shape}:{finite}")
            modes.append(
                {
                    "model_mode": mode,
                    "contract": model_mode_contract(mode),
                    "behavior_shape": shape,
                    "behavior_finite": finite,
                    "auxiliary_shapes": {
                        name: list(logits.shape)
                        for name, logits in output.auxiliary_logits().items()
                    },
                    "parameters": model_parameter_report(model),
                }
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(f"mode_build_failed={mode}:{type(exc).__name__}:{exc}")
    temporal: list[dict[str, Any]] = []
    for encoder_name in sorted(TEMPORAL_ENCODER_NAMES):
        config = _config("actor_temporal", temporal_encoder=encoder_name)
        try:
            model = _build(config).eval()
            output = model(**_inputs(config))
            temporal.append(
                {
                    "temporal_encoder_name": encoder_name,
                    "behavior_shape": list(output.behavior.shape),
                    "behavior_finite": bool(torch.isfinite(output.behavior).all()),
                    "parameters": model_parameter_report(model),
                }
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(
                "temporal_build_failed="
                f"{encoder_name}:{type(exc).__name__}:{exc}"
            )
    temporal_tiers: list[dict[str, Any]] = []
    for sequence_length in DEFAULT_TEMPORAL_TIERS:
        config = _config(
            "actor_temporal",
            temporal_input_frames=sequence_length,
        )
        try:
            model = _build(config).eval()
            output = model(**_inputs(config))
            shape = list(output.behavior.shape)
            finite = bool(torch.isfinite(output.behavior).all())
            if shape != [2, 10] or not finite:
                errors.append(
                    "temporal_tier_forward_invalid="
                    f"T{sequence_length}:{shape}:{finite}"
                )
            temporal_tiers.append(
                {
                    "temporal_tier": f"T{sequence_length}",
                    "sequence_length": sequence_length,
                    "behavior_shape": shape,
                    "behavior_finite": finite,
                    "parameters": model_parameter_report(model),
                }
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(
                "temporal_tier_forward_failed="
                f"T{sequence_length}:{type(exc).__name__}:{exc}"
            )
    tier_parameter_counts = {
        int(item["parameters"]["total"])
        for item in temporal_tiers
    }
    if len(tier_parameter_counts) > 1:
        errors.append("temporal_tier_parameter_count_drift")
    return {
        "schema_version": "classification_v2.model_factory_audit.v2",
        "model_mode_count": len(modes),
        "temporal_encoder_count": len(temporal),
        "temporal_tier_count": len(temporal_tiers),
        "expected_behavior_shape": [2, 10],
        "pretrained_weight_downloaded": False,
        "optimizer_steps": 0,
        "modes": modes,
        "temporal_encoders": temporal,
        "temporal_tiers": temporal_tiers,
        "errors": errors,
        "valid": not errors,
    }


def _config(
    mode: str,
    *,
    temporal_encoder: str | None = None,
    temporal_input_frames: int = 6,
) -> ModelConfig:
    spec = model_mode_spec(mode)
    encoder = temporal_encoder or (
        "masked_mean" if mode == "actor_only" else "masked_tcn"
    )
    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode=mode,
        temporal_input_frames=temporal_input_frames,
        temporal_encoder_name=encoder,
        hidden_dim=8,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
        spatial_feature_groups=spec.spatial_feature_groups,
        enable_image=spec.enable_image,
        enable_spatial=spec.enable_spatial,
        enable_interaction_context=spec.enable_interaction_context,
        enable_visual_context=spec.enable_visual_context,
        enable_multitask=spec.enable_multitask,
    )


def _build(config: ModelConfig):
    return build_multimodal_model(
        config,
        spatial_input_dims={
            name: GROUP_DIMS[name] for name in config.spatial_feature_groups
        },
        interaction_context_dim=(
            5 if config.enable_interaction_context else None
        ),
        num_classes=10,
    )


def _inputs(config: ModelConfig) -> dict[str, Any]:
    batch_size = 2
    sequence_length = config.temporal_input_frames
    length = torch.ones(batch_size, sequence_length)
    observed = length.clone()
    observed[1, -1] = 0.0
    delta = torch.full((batch_size, sequence_length), 0.2)
    delta[:, 0] = 0.0
    return {
        "image": torch.rand(batch_size, sequence_length, 3, 16, 16),
        "spatial_features": {
            name: torch.rand(batch_size, sequence_length, GROUP_DIMS[name])
            for name in config.spatial_feature_groups
        },
        "length_mask": length,
        "observed_mask": observed,
        "image_length_mask": length,
        "image_observed_mask": observed,
        "image_available_mask": observed,
        "image_quality_mask": observed,
        "image_time_delta": delta,
        "spatial_length_mask": length,
        "spatial_observed_mask": observed,
        "spatial_available_mask": observed,
        "spatial_quality_mask": observed,
        "spatial_time_delta": delta,
        "interaction_context_features": torch.rand(batch_size, 5),
        "interaction_context_available_mask": torch.tensor([1.0, 0.0]),
        "interaction_context_quality_mask": torch.tensor([1.0, 0.0]),
        "visual_context_image": torch.rand(
            batch_size,
            sequence_length,
            3,
            16,
            16,
        ),
        "visual_context_length_mask": length,
        "visual_context_observed_mask": observed,
        "visual_context_available_mask": observed,
        "visual_context_quality_mask": observed,
        "visual_context_time_delta": delta,
    }


if __name__ == "__main__":
    main()
