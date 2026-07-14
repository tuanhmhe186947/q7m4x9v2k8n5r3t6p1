"""Bounded synthetic correctness gates for production visual backbones."""

from __future__ import annotations

import io
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.config import ModelConfig

SYNTHETIC_VISUAL_SMOKE_SCHEMA_VERSION = (
    "classification_v2.synthetic_visual_smoke.v1"
)


@dataclass(frozen=True, slots=True)
class SyntheticVisualSmokeConfig:
    """Bounded run settings that cannot reference project data artifacts."""

    backbone_name: str = "resnet18"
    image_size: int = 160
    sequence_length: int = 2
    events_per_class: int = 2
    hidden_dim: int = 32
    steps: int = 30
    learning_rate: float = 0.003
    seed: int = 20260714
    device: str = "auto"
    batch_norm_recalibration_passes: int = 20
    minimum_accuracy: float = 0.95
    maximum_loss_ratio: float = 0.25


def run_synthetic_visual_smoke(
    config: SyntheticVisualSmokeConfig,
) -> dict[str, Any]:
    """Backpropagate, memorize tiny patterns, and round-trip optimizer state."""

    _validate_config(config)
    device = _resolve_device(config.device)
    _seed_all(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    images, targets = build_synthetic_visual_events(config)
    images = images.to(device)
    targets = targets.to(device)
    inputs = _actor_inputs(images)
    model_config = _model_config(config)
    model = _build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()
    initial_loss = _evaluate_loss(model, inputs, targets, loss_fn)
    losses: list[float] = []
    gradient_audit: dict[str, Any] | None = None
    resume_audit: dict[str, Any] | None = None
    resume_step = max(1, config.steps // 2)
    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).behavior
        loss = loss_fn(logits, targets)
        loss.backward()
        if gradient_audit is None:
            gradient_audit = _gradient_audit(model)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step == resume_step:
            model, optimizer, resume_audit = _round_trip_state(
                model,
                optimizer,
                model_config,
                inputs,
                device,
                config.learning_rate,
            )
            model.train()
    batch_norm_audit = _recalibrate_batch_norm(
        model,
        inputs,
        passes=config.batch_norm_recalibration_passes,
    )
    final_loss, final_accuracy = _evaluate(model, inputs, targets, loss_fn)
    runtime_sec = float(time.perf_counter() - started)
    loss_ratio = final_loss / initial_loss if initial_loss > 0 else float("inf")
    errors = _result_errors(
        config,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        losses=losses,
        gradient_audit=gradient_audit,
        resume_audit=resume_audit,
        batch_norm_audit=batch_norm_audit,
    )
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "schema_version": SYNTHETIC_VISUAL_SMOKE_SCHEMA_VERSION,
        "synthetic_only": True,
        "training_snapshot_allowed": False,
        "full_oof_allowed": False,
        "backbone_name": config.backbone_name,
        "pretrained_weight_enum": NO_PRETRAINED_WEIGHTS,
        "batch_norm_policy": "train_then_postfit_recalibration",
        "batch_norm_audit": batch_norm_audit,
        "image_size": config.image_size,
        "sequence_length": config.sequence_length,
        "event_count": int(targets.numel()),
        "native_event_count": int(targets.numel()),
        "class_count": len(VALID_BEHAVIORS),
        "events_per_class": config.events_per_class,
        "label_order": list(VALID_BEHAVIORS),
        "optimizer_steps": config.steps,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "device": str(device),
        "runtime_sec": runtime_sec,
        "peak_vram_bytes": peak_vram,
        "parameters": model_parameter_report(model),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "final_accuracy": final_accuracy,
        "minimum_accuracy": config.minimum_accuracy,
        "maximum_loss_ratio": config.maximum_loss_ratio,
        "losses": losses,
        "gradient_audit": gradient_audit,
        "resume_audit": resume_audit,
        "errors": errors,
        "valid": not errors,
    }


def build_synthetic_visual_events(
    config: SyntheticVisualSmokeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create balanced, deterministic class patterns without reading any file."""

    _validate_config(config)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    event_count = len(VALID_BEHAVIORS) * config.events_per_class
    images = torch.zeros(
        event_count,
        config.sequence_length,
        3,
        config.image_size,
        config.image_size,
    )
    targets = torch.empty(event_count, dtype=torch.long)
    event_index = 0
    for class_index in range(len(VALID_BEHAVIORS)):
        for replicate in range(config.events_per_class):
            for slot in range(config.sequence_length):
                images[event_index, slot] = _class_pattern(
                    class_index,
                    replicate,
                    slot,
                    config.image_size,
                    generator,
                )
            targets[event_index] = class_index
            event_index += 1
    return images, targets


def _class_pattern(
    class_index: int,
    replicate: int,
    slot: int,
    image_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Encode class in color and geometry while retaining small event variation."""

    image = torch.full((3, image_size, image_size), 0.08)
    color = torch.tensor(
        [
            0.25 + 0.65 * ((class_index >> 0) & 1),
            0.25 + 0.65 * ((class_index >> 1) & 1),
            0.25 + 0.65 * ((class_index >> 2) & 1),
        ]
    ).reshape(3, 1, 1)
    band = max(4, image_size // 12)
    x_start = (class_index * (image_size - band)) // 9
    y_start = ((9 - class_index) * (image_size - band)) // 9
    shift = slot * max(1, band // 3)
    x_start = min(image_size - band, x_start + shift)
    image[:, :, x_start : x_start + band] = color
    image[:, y_start : y_start + band, :] = 1.0 - color * 0.6
    block = max(5, image_size // 10)
    block_x = (class_index * 17 + replicate * 7) % (image_size - block + 1)
    block_y = (class_index * 11 + slot * 5) % (image_size - block + 1)
    image[:, block_y : block_y + block, block_x : block_x + block] = color
    noise = torch.rand(image.shape, generator=generator) * 0.015
    return (image + noise).clamp(0.0, 1.0)


def _model_config(config: SyntheticVisualSmokeConfig) -> ModelConfig:
    """Use one production actor-temporal path with direct ten-class supervision."""

    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode="actor_temporal",
        backbone_name=config.backbone_name,
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        temporal_encoder_name="masked_tcn",
        image_size=config.image_size,
        hidden_dim=config.hidden_dim,
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


def _build_model(config: ModelConfig) -> nn.Module:
    return build_multimodal_model(
        config,
        spatial_input_dims={},
        interaction_context_dim=None,
        num_classes=len(VALID_BEHAVIORS),
    )


def _actor_inputs(images: torch.Tensor) -> dict[str, Any]:
    batch_size, sequence_length = images.shape[:2]
    mask = torch.ones(batch_size, sequence_length, device=images.device)
    unavailable = torch.zeros_like(mask)
    time_delta = torch.full_like(mask, 0.2)
    time_delta[:, 0] = 0.0
    return {
        "image": images,
        "spatial_features": {},
        "length_mask": mask,
        "observed_mask": mask,
        "image_length_mask": mask,
        "image_observed_mask": mask,
        "image_available_mask": mask,
        "image_quality_mask": mask,
        "image_time_delta": time_delta,
        "spatial_length_mask": mask,
        "spatial_observed_mask": mask,
        "spatial_available_mask": unavailable,
        "spatial_quality_mask": unavailable,
        "spatial_time_delta": time_delta,
        "interaction_context_features": torch.zeros(
            batch_size,
            1,
            device=images.device,
        ),
        "interaction_context_available_mask": torch.zeros(
            batch_size,
            device=images.device,
        ),
        "interaction_context_quality_mask": torch.zeros(
            batch_size,
            device=images.device,
        ),
        "visual_context_image": torch.zeros(
            batch_size,
            sequence_length,
            3,
            1,
            1,
            device=images.device,
        ),
        "visual_context_length_mask": mask,
        "visual_context_observed_mask": mask,
        "visual_context_available_mask": unavailable,
        "visual_context_quality_mask": unavailable,
        "visual_context_time_delta": time_delta,
    }


def _evaluate_loss(
    model: nn.Module,
    inputs: dict[str, Any],
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    loss, _ = _evaluate(model, inputs, targets, loss_fn)
    return loss


def _evaluate(
    model: nn.Module,
    inputs: dict[str, Any],
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).behavior
        loss = float(loss_fn(logits, targets).cpu().item())
        accuracy = float(logits.argmax(dim=1).eq(targets).float().mean().cpu())
    return loss, accuracy


def _gradient_audit(model: nn.Module) -> dict[str, Any]:
    """Require finite nonzero gradients in both visual backbone and final head."""

    rows: dict[str, dict[str, Any]] = {}
    for group_name, prefix in {
        "visual_backbone": "backbone.image_encoder.frame_encoder",
        "final_behavior_head": "backbone.classifier",
    }.items():
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        finite = bool(gradients) and all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        nonzero_sum = float(
            sum(
                gradient.detach().abs().sum().cpu().item()
                for gradient in gradients
                if gradient is not None
            )
        )
        rows[group_name] = {
            "parameter_tensors": len(gradients),
            "finite": finite,
            "absolute_gradient_sum": nonzero_sum,
            "nonzero": nonzero_sum > 0.0,
        }
    return {
        "groups": rows,
        "valid": all(
            row["finite"] and row["nonzero"] for row in rows.values()
        ),
    }


def _recalibrate_batch_norm(
    model: nn.Module,
    inputs: dict[str, Any],
    *,
    passes: int,
) -> dict[str, Any]:
    """Refresh running statistics after weights stop changing, then evaluate."""

    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    modules = [
        module for module in model.modules() if isinstance(module, batch_norm_types)
    ]
    if not modules:
        return {
            "module_count": 0,
            "passes": 0,
            "running_statistics_finite": True,
            "valid": True,
        }
    for module in modules:
        module.reset_running_stats()
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            model(**inputs)
    statistics = [
        tensor
        for module in modules
        for tensor in (module.running_mean, module.running_var)
        if tensor is not None
    ]
    finite = bool(statistics) and all(
        torch.isfinite(tensor).all() for tensor in statistics
    )
    return {
        "module_count": len(modules),
        "passes": passes,
        "running_statistics_finite": finite,
        "valid": finite,
    }


def _round_trip_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    inputs: dict[str, Any],
    device: torch.device,
    learning_rate: float,
) -> tuple[nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    """Round-trip model and optimizer in memory without creating a fake run artifact."""

    model.eval()
    with torch.no_grad():
        expected = model(**inputs).behavior.detach().clone()
    buffer = io.BytesIO()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    payload = torch.load(buffer, map_location=device, weights_only=False)
    resumed_model = _build_model(config).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    resumed_model.load_state_dict(payload["model_state_dict"])
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    resumed_model.eval()
    with torch.no_grad():
        observed = resumed_model(**inputs).behavior
    max_delta = float((expected - observed).abs().max().cpu().item())
    return resumed_model, resumed_optimizer, {
        "method": "in_memory_model_and_optimizer_state_round_trip",
        "max_logit_delta": max_delta,
        "model_state_loaded": True,
        "optimizer_state_loaded": True,
        "valid": max_delta <= 1e-6,
    }


def _result_errors(
    config: SyntheticVisualSmokeConfig,
    *,
    initial_loss: float,
    final_loss: float,
    final_accuracy: float,
    losses: list[float],
    gradient_audit: dict[str, Any] | None,
    resume_audit: dict[str, Any] | None,
    batch_norm_audit: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not np.isfinite([initial_loss, final_loss, *losses]).all():
        errors.append("nonfinite_loss")
    if final_loss > initial_loss * config.maximum_loss_ratio:
        errors.append(
            f"loss_ratio_too_high={final_loss / initial_loss:.6f}"
        )
    if final_accuracy < config.minimum_accuracy:
        errors.append(f"accuracy_too_low={final_accuracy:.6f}")
    if gradient_audit is None or not gradient_audit["valid"]:
        errors.append("gradient_gate_failed")
    if resume_audit is None or not resume_audit["valid"]:
        errors.append("resume_round_trip_failed")
    if not batch_norm_audit["valid"]:
        errors.append("batch_norm_recalibration_failed")
    return errors


def _validate_config(config: SyntheticVisualSmokeConfig) -> None:
    if config.image_size < 32:
        raise ValueError("image_size must be at least 32")
    if config.sequence_length <= 0 or config.events_per_class <= 0:
        raise ValueError("sequence and event counts must be positive")
    if config.hidden_dim <= 0 or config.steps <= 1:
        raise ValueError("hidden_dim must be positive and steps must exceed one")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.batch_norm_recalibration_passes <= 0:
        raise ValueError("batch_norm_recalibration_passes must be positive")
    if not 0.0 <= config.minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be in [0,1]")
    if not 0.0 < config.maximum_loss_ratio < 1.0:
        raise ValueError("maximum_loss_ratio must be in (0,1)")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "SYNTHETIC_VISUAL_SMOKE_SCHEMA_VERSION",
    "SyntheticVisualSmokeConfig",
    "build_synthetic_visual_events",
    "run_synthetic_visual_smoke",
]
