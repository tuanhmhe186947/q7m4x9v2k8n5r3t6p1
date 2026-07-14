"""Audit production visual-backbone controls without data I/O or weight download."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
    VISUAL_BACKBONE_CONTRACT_VERSION,
    visual_backbone_contract,
)
from pig_behavior.classification_v2.training.config import ModelConfig

CONTROL_MATRIX = (
    ("V0", "resnet18", 160),
    ("V1", "resnet18", 224),
    ("V2", "resnet34", 224),
)
PRETRAINED_ENUMS = (
    ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
    ("resnet34", "ResNet34_Weights.IMAGENET1K_V1"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ResNet resolution/backbone controls without training."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/visual_backbone_audit.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_visual_backbone_audit()
    if not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def run_visual_backbone_audit() -> dict[str, Any]:
    """Run one forward per controlled random-init backbone configuration."""

    torch.manual_seed(20260714)
    errors: list[str] = []
    controls: list[dict[str, Any]] = []
    for control_id, backbone_name, image_size in CONTROL_MATRIX:
        try:
            config = _config(backbone_name, image_size)
            model = build_multimodal_model(
                config,
                spatial_input_dims={},
                interaction_context_dim=None,
                num_classes=10,
            ).eval()
            with torch.no_grad():
                output = model(**_actor_inputs(image_size))
            shape = list(output.behavior.shape)
            finite = bool(torch.isfinite(output.behavior).all())
            if shape != [1, 10] or not finite:
                errors.append(
                    f"invalid_forward={control_id}:shape={shape}:finite={finite}"
                )
            controls.append(
                {
                    "control_id": control_id,
                    "backbone_name": backbone_name,
                    "image_size": image_size,
                    "pretrained_weight_enum": NO_PRETRAINED_WEIGHTS,
                    "behavior_shape": shape,
                    "behavior_finite": finite,
                    "parameters": model_parameter_report(model),
                }
            )
        except (RuntimeError, ValueError) as exc:
            errors.append(
                f"control_failed={control_id}:{type(exc).__name__}:{exc}"
            )
    _audit_controlled_comparisons(controls, errors)
    pretrained_contracts = [
        _contract_payload(backbone_name, weight_enum)
        for backbone_name, weight_enum in PRETRAINED_ENUMS
    ]
    return {
        "schema_version": "classification_v2.visual_backbone_audit.v1",
        "visual_backbone_contract_version": VISUAL_BACKBONE_CONTRACT_VERSION,
        "controls": controls,
        "controlled_comparisons": {
            "resolution": "V0_resnet18_160_to_V1_resnet18_224",
            "backbone": "V1_resnet18_224_to_V2_resnet34_224",
        },
        "pretrained_contracts_resolved_without_model_build": pretrained_contracts,
        "pretrained_weight_downloaded": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": not errors,
    }


def _config(backbone_name: str, image_size: int) -> ModelConfig:
    """Return the actor-only control while changing only backbone/resolution."""

    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode="actor_only",
        backbone_name=backbone_name,
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        temporal_encoder_name="masked_mean",
        image_size=image_size,
        hidden_dim=16,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    )


def _actor_inputs(image_size: int) -> dict[str, Any]:
    """Build one deterministic tensor batch matching the inference interface."""

    mask = torch.ones(1, 1)
    return {
        "image": torch.rand(1, 1, 3, image_size, image_size),
        "spatial_features": {},
        "length_mask": mask,
        "observed_mask": mask,
        "image_length_mask": mask,
        "image_observed_mask": mask,
        "image_available_mask": mask,
        "image_quality_mask": mask,
        "image_time_delta": torch.zeros(1, 1),
        "spatial_length_mask": mask,
        "spatial_observed_mask": mask,
        "spatial_available_mask": mask,
        "spatial_quality_mask": mask,
        "spatial_time_delta": torch.zeros(1, 1),
        "interaction_context_features": torch.zeros(1, 1),
        "interaction_context_available_mask": torch.zeros(1),
        "interaction_context_quality_mask": torch.zeros(1),
        "visual_context_image": torch.zeros(1, 1, 3, 1, 1),
        "visual_context_length_mask": mask,
        "visual_context_observed_mask": mask,
        "visual_context_available_mask": torch.zeros(1, 1),
        "visual_context_quality_mask": torch.zeros(1, 1),
        "visual_context_time_delta": torch.zeros(1, 1),
    }


def _audit_controlled_comparisons(
    controls: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """Prove resolution does not change capacity and ResNet34 is the larger model."""

    if len(controls) != len(CONTROL_MATRIX):
        errors.append(f"missing_controls={len(controls)}<{len(CONTROL_MATRIX)}")
        return
    parameters = {
        row["control_id"]: int(row["parameters"]["total"])
        for row in controls
    }
    if parameters["V0"] != parameters["V1"]:
        errors.append("resnet18_parameter_count_changed_with_resolution")
    if parameters["V2"] <= parameters["V1"]:
        errors.append("resnet34_parameter_count_not_greater_than_resnet18")


def _contract_payload(backbone_name: str, weight_enum: str) -> dict[str, Any]:
    """Serialize enum and normalization metadata without constructing a model."""

    contract = visual_backbone_contract(backbone_name, weight_enum)
    return {
        "backbone_name": contract.name,
        "pretrained_weight_enum": contract.pretrained_weight_enum,
        "normalization_name": contract.normalization_name,
        "input_mean": list(contract.input_mean),
        "input_std": list(contract.input_std),
        "output_dim": contract.output_dim,
        "uses_pretrained_weights": contract.uses_pretrained_weights,
    }


if __name__ == "__main__":
    main()
