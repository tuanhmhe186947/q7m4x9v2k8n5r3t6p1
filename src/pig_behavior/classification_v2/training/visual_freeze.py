"""Audited visual-backbone freeze stages and optimizer parameter groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    ImageSequenceEncoder,
)

VISUAL_FREEZE_CONTRACT_VERSION = "classification_v2_visual_freeze_v1"
VISUAL_FREEZE_POLICIES = frozenset(
    {
        "all_trainable",
        "frozen_then_layer4",
        "frozen_then_layer4_then_full",
    }
)
VISUAL_FREEZE_STAGES = frozenset({"frozen", "layer4_only", "full"})


class VisualFreezeConfigLike(Protocol):
    """Fields needed to resolve one visual fine-tuning schedule."""

    backbone_name: str
    enable_image: bool
    enable_visual_context: bool
    visual_freeze_policy: str
    visual_frozen_warmup_epochs: int
    visual_layer4_only_epochs: int
    visual_backbone_lr_multiplier: float


@dataclass(frozen=True, slots=True)
class NamedVisualEncoder:
    """One actor or union-context frame backbone owned by the model."""

    name: str
    module: nn.Module


def visual_freeze_schedule_errors(
    config: VisualFreezeConfigLike,
    *,
    total_epochs: int | None = None,
) -> list[str]:
    """Return invalid or scientifically ambiguous schedule settings."""

    errors: list[str] = []
    policy = config.visual_freeze_policy
    warmup = config.visual_frozen_warmup_epochs
    layer4_epochs = config.visual_layer4_only_epochs
    multiplier = config.visual_backbone_lr_multiplier
    if policy not in VISUAL_FREEZE_POLICIES:
        errors.append(f"unsupported_visual_freeze_policy={policy}")
    if not _is_nonnegative_int(warmup):
        errors.append("visual_frozen_warmup_epochs_must_be_nonnegative_integer")
    if not _is_nonnegative_int(layer4_epochs):
        errors.append("visual_layer4_only_epochs_must_be_nonnegative_integer")
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        errors.append("visual_backbone_lr_multiplier_must_be_numeric")
    elif not math.isfinite(float(multiplier)) or not 0.0 < float(multiplier) <= 1.0:
        errors.append("visual_backbone_lr_multiplier_must_be_in_0_1")
    if errors:
        return errors
    staged = policy != "all_trainable"
    if policy == "all_trainable" and (warmup != 0 or layer4_epochs != 0):
        errors.append("all_trainable_requires_zero_stage_durations")
    if policy == "frozen_then_layer4":
        if warmup < 1:
            errors.append("frozen_then_layer4_requires_warmup_epoch")
        if layer4_epochs != 0:
            errors.append("frozen_then_layer4_requires_zero_terminal_layer4_duration")
    if policy == "frozen_then_layer4_then_full":
        if warmup < 1:
            errors.append("staged_full_requires_warmup_epoch")
        if layer4_epochs < 1:
            errors.append("staged_full_requires_layer4_epoch")
    if staged and config.backbone_name not in {"resnet18", "resnet34"}:
        errors.append(f"staged_visual_freeze_requires_resnet={config.backbone_name}")
    if staged and not (config.enable_image or config.enable_visual_context):
        errors.append("staged_visual_freeze_requires_visual_branch")
    if staged and float(multiplier) >= 1.0:
        errors.append("staged_visual_freeze_requires_lower_backbone_lr")
    if total_epochs is not None:
        if not _is_positive_int(total_epochs):
            errors.append("visual_freeze_total_epochs_must_be_positive_integer")
        elif policy == "frozen_then_layer4" and total_epochs <= warmup:
            errors.append("declared_epochs_never_reach_layer4_stage")
        elif (
            policy == "frozen_then_layer4_then_full"
            and total_epochs <= warmup + layer4_epochs
        ):
            errors.append("declared_epochs_never_reach_full_stage")
    return errors


def visual_freeze_schedule_payload(
    config: VisualFreezeConfigLike,
    *,
    total_epochs: int | None = None,
) -> dict[str, Any]:
    """Serialize the exact schedule used by config, lineage, and checkpoints."""

    errors = visual_freeze_schedule_errors(config, total_epochs=total_epochs)
    if errors:
        raise ValueError(f"invalid visual freeze schedule: {errors}")
    return {
        "contract_version": VISUAL_FREEZE_CONTRACT_VERSION,
        "policy": config.visual_freeze_policy,
        "frozen_warmup_epochs": config.visual_frozen_warmup_epochs,
        "layer4_only_epochs": config.visual_layer4_only_epochs,
        "backbone_lr_multiplier": float(config.visual_backbone_lr_multiplier),
    }


def visual_freeze_stage_for_epoch(
    config: VisualFreezeConfigLike,
    epoch: int,
) -> str:
    """Resolve a deterministic stage from zero-based epoch and schedule."""

    if not _is_nonnegative_int(epoch):
        raise ValueError("visual freeze epoch must be a nonnegative integer")
    errors = visual_freeze_schedule_errors(config)
    if errors:
        raise ValueError(f"invalid visual freeze schedule: {errors}")
    if config.visual_freeze_policy == "all_trainable":
        return "full"
    if epoch < config.visual_frozen_warmup_epochs:
        return "frozen"
    if config.visual_freeze_policy == "frozen_then_layer4":
        return "layer4_only"
    full_epoch = (
        config.visual_frozen_warmup_epochs
        + config.visual_layer4_only_epochs
    )
    return "layer4_only" if epoch < full_epoch else "full"


def named_visual_frame_encoders(model: nn.Module) -> tuple[NamedVisualEncoder, ...]:
    """Find actor and union frame backbones without relying on wrapper paths."""

    found: list[NamedVisualEncoder] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, ImageSequenceEncoder):
            continue
        frame_encoder = module.frame_encoder
        identity = id(frame_encoder)
        if identity in seen:
            raise ValueError("visual frame encoder is shared across sequence branches")
        seen.add(identity)
        name = f"{module_name}.frame_encoder" if module_name else "frame_encoder"
        found.append(NamedVisualEncoder(name=name, module=frame_encoder))
    return tuple(found)


def configure_visual_train_stage(
    model: nn.Module,
    config: VisualFreezeConfigLike,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Apply requires-grad and BatchNorm mode for one training epoch."""

    if not model.training:
        raise ValueError("configure_visual_train_stage requires model.train() first")
    stage = visual_freeze_stage_for_epoch(config, epoch)
    encoders = named_visual_frame_encoders(model)
    if config.visual_freeze_policy != "all_trainable" and not encoders:
        raise ValueError("staged visual freeze found no visual frame encoders")
    for encoder in encoders:
        _configure_encoder_stage(encoder, stage)
    return visual_freeze_parameter_report(
        model,
        config,
        epoch=epoch,
        expected_stage=stage,
    )


def build_visual_optimizer_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all parameters in stable groups before any later unfreeze."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(backbone_lr_multiplier):
        raise ValueError("backbone_lr_multiplier must be finite")
    if not 0.0 < backbone_lr_multiplier <= 1.0:
        raise ValueError("backbone_lr_multiplier must be in (0, 1]")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    encoders = named_visual_frame_encoders(model)
    visual_ids = {
        id(parameter)
        for encoder in encoders
        for parameter in encoder.module.parameters()
    }
    named_parameters = list(model.named_parameters())
    parameter_ids = [id(parameter) for _, parameter in named_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("model.named_parameters contains shared parameter identities")
    visual = [parameter for _, parameter in named_parameters if id(parameter) in visual_ids]
    nonvisual = [parameter for _, parameter in named_parameters if id(parameter) not in visual_ids]
    if not visual and encoders:
        raise ValueError("visual encoders expose no optimizer parameters")
    if not nonvisual:
        raise ValueError("optimizer requires nonvisual trainable-head parameters")
    groups: list[dict[str, Any]] = []
    if visual:
        groups.append(
            {
                "group_name": "visual_backbone",
                "params": visual,
                "lr": learning_rate * backbone_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
    groups.append(
        {
            "group_name": "nonvisual",
            "params": nonvisual,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    )
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if set(grouped_ids) != set(parameter_ids) or len(grouped_ids) != len(parameter_ids):
        raise ValueError("optimizer groups do not cover each model parameter exactly once")
    report = {
        "contract_version": VISUAL_FREEZE_CONTRACT_VERSION,
        "group_order": [str(group["group_name"]) for group in groups],
        "groups": [
            {
                "group_name": str(group["group_name"]),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
                "tensor_count": len(group["params"]),
            }
            for group in groups
        ],
        "all_parameters_covered_once": True,
    }
    return groups, report


def visual_freeze_parameter_report(
    model: nn.Module,
    config: VisualFreezeConfigLike,
    *,
    epoch: int,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    """Audit trainability and module mode without changing the model."""

    stage = visual_freeze_stage_for_epoch(config, epoch)
    if expected_stage is not None and expected_stage != stage:
        raise ValueError(
            f"visual freeze stage mismatch expected={expected_stage} resolved={stage}"
        )
    encoders = named_visual_frame_encoders(model)
    encoder_rows = [_encoder_report(encoder, stage) for encoder in encoders]
    errors = [error for row in encoder_rows for error in row["errors"]]
    if errors:
        raise ValueError(f"visual freeze application invalid: {errors}")
    visual_ids = {
        id(parameter)
        for encoder in encoders
        for parameter in encoder.module.parameters()
    }
    nonvisual_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in visual_ids
    ]
    nonvisual_parameter_count = int(
        sum(parameter.numel() for parameter in nonvisual_parameters)
    )
    nonvisual_trainable_parameter_count = int(
        sum(
            parameter.numel()
            for parameter in nonvisual_parameters
            if parameter.requires_grad
        )
    )
    if nonvisual_trainable_parameter_count != nonvisual_parameter_count:
        raise ValueError("visual freeze schedule must keep every nonvisual head trainable")
    return {
        **visual_freeze_schedule_payload(config),
        "epoch": epoch,
        "stage": stage,
        "visual_encoder_count": len(encoders),
        "visual_encoder_names": [encoder.name for encoder in encoders],
        "visual_parameter_count": int(
            sum(row["parameter_count"] for row in encoder_rows)
        ),
        "visual_trainable_parameter_count": int(
            sum(row["trainable_parameter_count"] for row in encoder_rows)
        ),
        "nonvisual_parameter_count": nonvisual_parameter_count,
        "nonvisual_trainable_parameter_count": (
            nonvisual_trainable_parameter_count
        ),
        "encoders": encoder_rows,
        "errors": [],
        "valid": True,
    }


def optimizer_group_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Serialize stable optimizer group names, rates, and live parameter counts."""

    rows: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", ""))
        if not name:
            raise ValueError(f"optimizer group {index} lacks group_name")
        parameters = list(group["params"])
        rows.append(
            {
                "index": index,
                "group_name": name,
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
                "tensor_count": len(parameters),
            }
        )
    names = [row["group_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"optimizer group names are not unique={names}")
    return {
        "contract_version": VISUAL_FREEZE_CONTRACT_VERSION,
        "group_order": names,
        "groups": rows,
        "valid": True,
    }


def _configure_encoder_stage(encoder: NamedVisualEncoder, stage: str) -> None:
    if stage not in VISUAL_FREEZE_STAGES:
        raise ValueError(f"unsupported visual freeze stage={stage}")
    for parameter in encoder.module.parameters():
        parameter.requires_grad_(stage == "full")
    if stage == "frozen":
        encoder.module.eval()
        return
    if stage == "full":
        encoder.module.train()
        return
    layer4 = getattr(encoder.module, "layer4", None)
    if not isinstance(layer4, nn.Module):
        raise ValueError(f"visual encoder lacks ResNet layer4={encoder.name}")
    encoder.module.eval()
    for parameter in layer4.parameters():
        parameter.requires_grad_(True)
    layer4.train()


def _encoder_report(encoder: NamedVisualEncoder, stage: str) -> dict[str, Any]:
    named_parameters = list(encoder.module.named_parameters())
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    batch_norm = [
        (name, module)
        for name, module in encoder.module.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    training_batch_norm = [name for name, module in batch_norm if module.training]
    errors: list[str] = []
    if stage == "frozen" and trainable:
        errors.append(f"frozen_encoder_has_trainable_parameters={encoder.name}")
    if stage == "frozen" and training_batch_norm:
        errors.append(f"frozen_encoder_has_training_batchnorm={encoder.name}")
    if stage == "layer4_only":
        if not trainable or any(not name.startswith("layer4.") for name in trainable):
            errors.append(f"layer4_stage_trainable_scope_invalid={encoder.name}")
        if any(not name.startswith("layer4.") for name in training_batch_norm):
            errors.append(f"layer4_stage_batchnorm_scope_invalid={encoder.name}")
    if stage == "full" and len(trainable) != len(named_parameters):
        errors.append(f"full_stage_has_frozen_parameters={encoder.name}")
    return {
        "name": encoder.name,
        "parameter_count": int(
            sum(parameter.numel() for _, parameter in named_parameters)
        ),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for _, parameter in named_parameters
                if parameter.requires_grad
            )
        ),
        "parameter_tensor_count": len(named_parameters),
        "trainable_tensor_count": len(trainable),
        "batch_norm_module_count": len(batch_norm),
        "training_batch_norm_names": training_batch_norm,
        "errors": errors,
    }


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


__all__ = [
    "VISUAL_FREEZE_CONTRACT_VERSION",
    "VISUAL_FREEZE_POLICIES",
    "VISUAL_FREEZE_STAGES",
    "build_visual_optimizer_groups",
    "configure_visual_train_stage",
    "named_visual_frame_encoders",
    "optimizer_group_report",
    "visual_freeze_parameter_report",
    "visual_freeze_schedule_errors",
    "visual_freeze_schedule_payload",
    "visual_freeze_stage_for_epoch",
]
