from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
    visual_backbone_contract,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.config import ModelConfig
from pig_behavior.classification_v2.training.visual_freeze import (
    build_visual_optimizer_groups,
    configure_visual_train_stage,
    named_visual_frame_encoders,
    optimizer_group_report,
    visual_freeze_schedule_payload,
)

MATRIX = (
    {
        "id": "V0",
        "backbone_name": "resnet18",
        "image_size": 160,
        "intended_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
    },
    {
        "id": "V1",
        "backbone_name": "resnet18",
        "image_size": 224,
        "intended_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
    },
    {
        "id": "V2",
        "backbone_name": "resnet34",
        "image_size": 224,
        "intended_weight_enum": "ResNet34_Weights.IMAGENET1K_V1",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit V0/V1/V2 visual freeze schedules without training."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "visual_freeze_schedule_audit.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_audit()
    if not args.dry_run:
        require_output_paths_available([args.output_json], overwrite=args.overwrite)
        _write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def run_audit() -> dict[str, object]:
    """Build random-init structures while auditing intended pretrained metadata."""

    rows = [_audit_candidate(spec) for spec in MATRIX]
    schedules = {_canonical(row["freeze_schedule"]) for row in rows}
    normalizations = {
        _canonical(row["intended_backbone_contract"]["normalization"])
        for row in rows
    }
    head_signatures = {str(row["nonvisual_signature_sha256"]) for row in rows}
    errors: list[str] = []
    if len(schedules) != 1:
        errors.append("visual_matrix_freeze_schedule_drift")
    if len(normalizations) != 1:
        errors.append("visual_matrix_normalization_drift")
    if len(head_signatures) != 1:
        errors.append("visual_matrix_trainable_head_design_drift")
    if not all(row["intended_backbone_contract"]["uses_pretrained"] for row in rows):
        errors.append("visual_matrix_pretrained_status_mismatch")
    if any(row["optimizer_steps"] != 0 for row in rows):
        errors.append("visual_matrix_optimizer_step_detected")
    return {
        "schema_version": "classification_v2_visual_freeze_schedule_audit_v1",
        "matrix": rows,
        "comparison_controls": {
            "V0_to_V1_changed_family": "resolution_only",
            "V1_to_V2_changed_family": "backbone_only",
            "same_pretrained_status": True,
            "same_normalization": len(normalizations) == 1,
            "same_freeze_schedule": len(schedules) == 1,
            "same_trainable_head_design": len(head_signatures) == 1,
        },
        "synthetic_or_structural_only": True,
        "project_data_rows_read": 0,
        "pretrained_downloads": 0,
        "optimizer_steps": 0,
        "training_snapshot_allowed": False,
        "full_oof_allowed": False,
        "errors": errors,
        "valid": not errors,
    }


def _audit_candidate(spec: dict[str, object]) -> dict[str, object]:
    intended = visual_backbone_contract(
        str(spec["backbone_name"]),
        str(spec["intended_weight_enum"]),
    )
    config = _model_config(spec)
    model = build_multimodal_model(
        config,
        spatial_input_dims={},
        interaction_context_dim=None,
        num_classes=len(VALID_BEHAVIORS),
    )
    groups, group_contract = build_visual_optimizer_groups(
        model,
        learning_rate=1e-3,
        backbone_lr_multiplier=config.visual_backbone_lr_multiplier,
        weight_decay=1e-4,
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-4)
    stage_rows = []
    for epoch in range(3):
        model.train()
        report = configure_visual_train_stage(model, config, epoch=epoch)
        stage_rows.append(
            {
                "epoch": epoch,
                "stage": report["stage"],
                "visual_parameter_count": report["visual_parameter_count"],
                "visual_trainable_parameter_count": (
                    report["visual_trainable_parameter_count"]
                ),
                "encoder_count": report["visual_encoder_count"],
                "batch_norm_training_count": sum(
                    len(row["training_batch_norm_names"])
                    for row in report["encoders"]
                ),
            }
        )
    result = {
        **spec,
        "structural_weight_enum": NO_PRETRAINED_WEIGHTS,
        "intended_backbone_contract": {
            "uses_pretrained": intended.uses_pretrained_weights,
            "weight_enum": intended.pretrained_weight_enum,
            "normalization": {
                "name": intended.normalization_name,
                "mean": list(intended.input_mean),
                "std": list(intended.input_std),
            },
        },
        "freeze_schedule": visual_freeze_schedule_payload(config, total_epochs=3),
        "stages": stage_rows,
        "optimizer_groups": optimizer_group_report(optimizer),
        "optimizer_group_contract": group_contract,
        "nonvisual_signature_sha256": _nonvisual_signature(model),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "optimizer_steps": 0,
    }
    del optimizer
    del model
    gc.collect()
    return result


def _model_config(spec: dict[str, object]) -> ModelConfig:
    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode="actor_temporal",
        backbone_name=str(spec["backbone_name"]),
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        image_size=int(spec["image_size"]),
        temporal_encoder_name="masked_tcn",
        hidden_dim=48,
        dropout=0.1,
        visual_freeze_policy="frozen_then_layer4_then_full",
        visual_frozen_warmup_epochs=1,
        visual_layer4_only_epochs=1,
        visual_backbone_lr_multiplier=0.1,
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    )


def _nonvisual_signature(model: torch.nn.Module) -> str:
    visual_ids = {
        id(parameter)
        for encoder in named_visual_frame_encoders(model)
        for parameter in encoder.module.parameters()
    }
    payload = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if id(parameter) not in visual_ids
    ]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
