"""Locked generalization treatment helpers for the final Joint runner."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
)


@dataclass(frozen=True, slots=True)
class LockedGeneralizationRecipe:
    """The one frozen treatment admitted by the final refinement task."""

    recipe_id: str
    control_commit: str
    max_epochs: int
    augmentation_policy: str
    brightness_range: tuple[float, float]
    contrast_range: tuple[float, float]
    saturation_range: tuple[float, float]
    gamma_range: tuple[float, float]
    weight_decay: float
    scheduler: str
    scheduler_floor: float


def load_locked_generalization_recipe(
    path: Path,
    *,
    expected_control_commit: str,
) -> LockedGeneralizationRecipe:
    """Load and fail closed on a recipe that differs from the locked contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "LOCKED":
        raise ValueError("generalization recipe must have status=LOCKED")
    if payload.get("control_commit") != expected_control_commit:
        raise ValueError("generalization recipe control commit mismatch")
    for field_name in ("architecture_changed", "data_changed", "split_changed", "pb_changed"):
        if payload.get(field_name) is not False:
            raise ValueError(f"generalization recipe changed locked field={field_name}")

    components = payload.get("components")
    if not isinstance(components, dict):
        raise ValueError("generalization recipe components are missing")
    augmentation = components.get("augmentation")
    optimizer = components.get("optimizer")
    scheduler = components.get("scheduler")
    if not all(isinstance(value, dict) for value in (augmentation, optimizer, scheduler)):
        raise ValueError("generalization recipe component payload is malformed")
    if augmentation.get("policy") != "video_safe_photometric_v1":
        raise ValueError("unexpected generalization augmentation policy")
    if augmentation.get("train_only") is not True:
        raise ValueError("generalization augmentation must be train-only")
    if augmentation.get("temporal_coherence") != "one_parameter_set_per_window":
        raise ValueError("generalization augmentation is not temporally coherent")
    if augmentation.get("shared_actor_union_parameters") is not True:
        raise ValueError("actor/union augmentation parameters must be shared")
    if any(augmentation.get(name) is not False for name in (
        "geometric_transform",
        "gaussian_noise",
        "blur",
    )):
        raise ValueError("generalization augmentation includes an unapproved transform")
    if optimizer.get("name") != "adamw" or optimizer.get("preserve_parameter_groups") is not True:
        raise ValueError("generalization optimizer contract is not AdamW group-preserving")
    if scheduler.get("name") != "cosine_floor_0.1":
        raise ValueError("unexpected generalization scheduler")
    if scheduler.get("preserve_group_lr_ratio") is not True:
        raise ValueError("scheduler must preserve optimizer group LR ratios")

    def _range(name: str) -> tuple[float, float]:
        value = augmentation.get(name)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"augmentation range is malformed: {name}")
        bounds = (float(value[0]), float(value[1]))
        if not all(math.isfinite(item) and item > 0.0 for item in bounds):
            raise ValueError(f"augmentation range is invalid: {name}")
        if bounds[0] > bounds[1]:
            raise ValueError(f"augmentation range is reversed: {name}")
        return bounds

    max_epochs = int(payload.get("max_epochs", scheduler.get("max_epochs", 0)))
    if max_epochs != int(scheduler.get("max_epochs", -1)) or max_epochs <= 0:
        raise ValueError("generalization scheduler epoch contract is invalid")
    floor = float(scheduler.get("minimum_factor", 0.0))
    if not math.isclose(floor, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("generalization scheduler floor must be 0.1")

    recipe = LockedGeneralizationRecipe(
        recipe_id=str(payload.get("recipe_id", "")),
        control_commit=expected_control_commit,
        max_epochs=max_epochs,
        augmentation_policy=str(augmentation["policy"]),
        brightness_range=_range("brightness_factor"),
        contrast_range=_range("contrast_factor"),
        saturation_range=_range("saturation_factor"),
        gamma_range=_range("gamma_factor"),
        weight_decay=float(optimizer.get("weight_decay", -1.0)),
        scheduler=str(scheduler["name"]),
        scheduler_floor=floor,
    )
    if not math.isclose(recipe.weight_decay, 1e-4, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("generalization weight decay must be 1e-4")
    if not recipe.recipe_id:
        raise ValueError("generalization recipe_id is missing")
    return recipe


def apply_locked_recipe(
    config: ClassificationV2TrainingConfig,
    recipe: LockedGeneralizationRecipe,
) -> ClassificationV2TrainingConfig:
    """Overlay only the declared generalization fields on the control config."""

    dataset = replace(config.dataset, augmentation_policy=recipe.augmentation_policy)
    optimization = replace(
        config.optimization,
        epochs=recipe.max_epochs,
        weight_decay=recipe.weight_decay,
        scheduler=recipe.scheduler,
    )
    return replace(config, dataset=dataset, optimization=optimization)


def cosine_floor_factor(step: int, *, max_steps: int, floor: float) -> float:
    """Return a bounded cosine multiplier with an exact endpoint floor."""

    if max_steps <= 0 or not 0.0 < floor <= 1.0:
        raise ValueError("cosine scheduler bounds are invalid")
    progress = min(max(int(step), 0), max_steps) / max_steps
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_locked_scheduler(
    optimizer: torch.optim.Optimizer,
    recipe: LockedGeneralizationRecipe,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build one group-ratio-preserving cosine scheduler."""

    if recipe.scheduler != "cosine_floor_0.1":
        raise ValueError(f"unsupported locked scheduler={recipe.scheduler}")
    factors = [
        lambda step: cosine_floor_factor(
            step,
            max_steps=recipe.max_epochs,
            floor=recipe.scheduler_floor,
        )
        for _ in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factors)


def scheduler_trajectory(
    optimizer: torch.optim.Optimizer,
    recipe: LockedGeneralizationRecipe,
    *,
    steps: int = 4,
) -> list[list[float]]:
    """Simulate several deterministic scheduler transitions without mutating it."""

    return [
        [
            float(group["lr"])
            * cosine_floor_factor(
                step,
                max_steps=recipe.max_epochs,
                floor=recipe.scheduler_floor,
            )
            for group in optimizer.param_groups
        ]
        for step in range(min(max(steps, 1), recipe.max_epochs) + 1)
    ]


def video_safe_photometric_augment(
    model_inputs: Mapping[str, Any],
    recipe: LockedGeneralizationRecipe,
) -> dict[str, Any]:
    """Apply one shared photometric parameter set to each T6 RGB window.

    The two RGB streams share brightness, contrast, saturation, and gamma draws
    for a window. Invalid/padded temporal slots remain unchanged under their
    corresponding availability mask.
    """

    if recipe.augmentation_policy != "video_safe_photometric_v1":
        raise ValueError("unexpected locked augmentation policy")
    actor = _require_rgb(model_inputs.get("image"), "image")
    union = _require_rgb(model_inputs.get("visual_context_image"), "visual_context_image")
    if actor.shape[0] != union.shape[0] or actor.shape[1] != union.shape[1]:
        raise ValueError("actor and union RGB windows are not aligned")
    batch_size = int(actor.shape[0])
    factors = {
        "brightness": _draw_factor(actor, recipe.brightness_range, batch_size),
        "contrast": _draw_factor(actor, recipe.contrast_range, batch_size),
        "saturation": _draw_factor(actor, recipe.saturation_range, batch_size),
        "gamma": _draw_factor(actor, recipe.gamma_range, batch_size),
    }
    actor_mask = model_inputs.get("image_available_mask", model_inputs.get("image_observed_mask"))
    union_mask = model_inputs.get(
        "visual_context_available_mask",
        model_inputs.get("visual_context_observed_mask"),
    )
    result = dict(model_inputs)
    result["image"] = _transform_stream(actor, actor_mask, factors)
    result["visual_context_image"] = _transform_stream(union, union_mask, factors)
    return result


def _draw_factor(
    reference: torch.Tensor,
    bounds: tuple[float, float],
    batch_size: int,
) -> torch.Tensor:
    return torch.empty(
        (batch_size, 1, 1, 1, 1),
        dtype=reference.dtype,
        device=reference.device,
    ).uniform_(bounds[0], bounds[1])


def _transform_stream(
    value: torch.Tensor,
    mask: Any,
    factors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        raise ValueError("RGB augmentation requires an availability mask")
    availability = torch.as_tensor(mask, device=value.device)
    if tuple(availability.shape) != tuple(value.shape[:2]):
        raise ValueError("RGB availability mask shape does not match RGB window")
    availability = availability.to(dtype=value.dtype).clamp(0.0, 1.0)
    brightness = factors["brightness"]
    contrast = factors["contrast"]
    saturation = factors["saturation"]
    gamma = factors["gamma"]
    transformed = value * brightness
    transformed = (transformed - 0.5) * contrast + 0.5
    gray = (
        transformed[:, :, 0:1] * 0.299
        + transformed[:, :, 1:2] * 0.587
        + transformed[:, :, 2:3] * 0.114
    )
    transformed = gray + saturation * (transformed - gray)
    transformed = transformed.clamp(0.0, 1.0).pow(gamma)
    mask_5d = availability.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return torch.where(mask_5d > 0.0, transformed, value)


def _require_rgb(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be a [B,T,C,H,W] tensor")
    if value.shape[2] != 3 or not value.is_floating_point():
        raise ValueError(f"{name} must be floating-point RGB")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


__all__ = [
    "LockedGeneralizationRecipe",
    "apply_locked_recipe",
    "build_locked_scheduler",
    "cosine_floor_factor",
    "load_locked_generalization_recipe",
    "scheduler_trajectory",
    "video_safe_photometric_augment",
]
