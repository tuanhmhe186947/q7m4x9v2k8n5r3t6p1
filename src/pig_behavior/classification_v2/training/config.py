"""Versioned, fail-closed configuration for classification_v2 training."""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from pig_behavior.classification_v2.models.multitask_fusion import MULTITASK_ARCHITECTURE_VERSION

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    snapshot_json: Path
    trainer_contract_json: Path
    train_ready_root: Path
    actor_packed_cache: Path
    actor_packed_index: Path
    visual_cache_manifest: Path
    visual_packed_cache: Path
    visual_packed_index: Path
    native_oof_fold_manifest: Path
    auxiliary_targets_csv: Path
    strict_packed_cache: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture_version: str
    image_size: int = 64
    hidden_dim: int = 48
    dropout: float = 0.1
    spatial_feature_groups: tuple[str, ...] = ()
    enable_visual_context: bool = True
    enable_multitask: bool = True


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    epochs: int = 3
    batch_size: int = 128
    eval_batch_size: int = 128
    gradient_clip_norm: float = 1.0
    precision: str = "amp"
    seed: int = 20260710
    deterministic: bool = True
    checkpoint_every_steps: int = 500


@dataclass(frozen=True, slots=True)
class LossConfig:
    behavior_weight: float = 1.0
    posture_weight: float = 0.25
    motion_context_weight: float = 0.25
    roi_intent_weight: float = 0.25
    interaction_weight: float = 0.25
    hierarchy_consistency_weight: float = 0.1
    class_weight_power: float = 0.5
    class_weight_max: float = 5.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    mode: str = "smoke"
    fold_id: str = "native_oof_000"
    smoke_steps: int = 8
    smoke_per_class: int = 1
    output_dir: Path = Path("outputs/classification_v2/model_smoke/strict_multitask")
    resume: bool = True


@dataclass(frozen=True, slots=True)
class ClassificationV2TrainingConfig:
    version: str
    dataset: DatasetConfig
    model: ModelConfig
    optimization: OptimizationConfig
    loss: LossConfig
    execution: ExecutionConfig


def load_training_config(path: Path) -> ClassificationV2TrainingConfig:
    """Read and validate a strict JSON config without accepting unknown keys."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    root_required = {"version", "dataset", "model", "optimization", "loss", "execution"}
    missing = sorted(root_required.difference(payload))
    unknown = sorted(set(payload).difference(root_required))
    if missing or unknown:
        raise ValueError(f"training config root mismatch: missing={missing}, unknown={unknown}")
    config = ClassificationV2TrainingConfig(
        version=str(payload["version"]),
        dataset=_from_mapping(DatasetConfig, payload["dataset"]),
        model=_from_mapping(ModelConfig, payload["model"]),
        optimization=_from_mapping(OptimizationConfig, payload["optimization"]),
        loss=_from_mapping(LossConfig, payload["loss"]),
        execution=_from_mapping(ExecutionConfig, payload["execution"]),
    )
    validate_training_config(config)
    return config


def validate_training_config(config: ClassificationV2TrainingConfig) -> None:
    """Reject unsafe, ambiguous, or unsupported training settings."""

    errors: list[str] = []
    if config.version != "classification_v2_training_config_v1":
        errors.append(f"unsupported_version={config.version}")
    if config.model.architecture_version != MULTITASK_ARCHITECTURE_VERSION:
        errors.append(f"architecture_version_mismatch={config.model.architecture_version}")
    if not config.model.spatial_feature_groups:
        errors.append("spatial_feature_groups_empty")
    if config.model.image_size <= 0 or config.model.hidden_dim <= 0:
        errors.append("model_dimensions_must_be_positive")
    if config.optimization.optimizer != "adamw":
        errors.append(f"unsupported_optimizer={config.optimization.optimizer}")
    if config.optimization.precision not in {"fp32", "amp"}:
        errors.append(f"unsupported_precision={config.optimization.precision}")
    if min(config.optimization.epochs, config.optimization.batch_size, config.optimization.eval_batch_size) <= 0:
        errors.append("epochs_and_batch_sizes_must_be_positive")
    if config.optimization.learning_rate <= 0.0 or config.optimization.gradient_clip_norm <= 0.0:
        errors.append("learning_rate_and_gradient_clip_must_be_positive")
    if config.execution.mode not in {"smoke", "full_oof"}:
        errors.append(f"unsupported_execution_mode={config.execution.mode}")
    if config.execution.mode == "smoke" and (
        config.execution.smoke_steps <= 0 or config.execution.smoke_per_class <= 0
    ):
        errors.append("smoke_limits_must_be_positive")
    if config.execution.mode == "full_oof" and not config.dataset.strict_packed_cache:
        errors.append("full_oof_requires_strict_packed_cache")
    loss_values = [
        config.loss.behavior_weight,
        config.loss.posture_weight,
        config.loss.motion_context_weight,
        config.loss.roi_intent_weight,
        config.loss.interaction_weight,
        config.loss.hierarchy_consistency_weight,
    ]
    if any(value < 0.0 for value in loss_values) or config.loss.behavior_weight <= 0.0:
        errors.append("loss_weights_must_be_nonnegative_and_behavior_positive")
    if errors:
        raise ValueError(f"invalid classification_v2 training config: {errors}")


def training_config_to_jsonable(config: ClassificationV2TrainingConfig) -> dict[str, Any]:
    """Serialize typed config for audit/checkpoint artifacts."""

    return {
        "version": config.version,
        **{
            section: {
                field.name: _jsonable(getattr(getattr(config, section), field.name))
                for field in fields(getattr(config, section))
            }
            for section in ["dataset", "model", "optimization", "loss", "execution"]
        },
    }


def _from_mapping(cls: type[T], payload: dict[str, Any]) -> T:
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__} payload must be an object")
    names = {field.name for field in fields(cls)}
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required.difference(payload))
    unknown = sorted(set(payload).difference(names))
    if missing or unknown:
        raise ValueError(f"{cls.__name__} mismatch: missing={missing}, unknown={unknown}")
    converted: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in payload:
            continue
        value = payload[field.name]
        if "Path" in str(field.type):
            value = Path(value)
        elif field.name == "spatial_feature_groups":
            value = tuple(str(item) for item in value)
        converted[field.name] = value
    return cls(**converted)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value
