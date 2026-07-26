"""Model registry for the balanced causal main-model research package.

The registry keeps the *name* of a scientific configuration separate from the
code that builds it, so an experiment manifest can record
``B3_ACTOR_T6_PLUS_GEOMETRY_MOTION`` rather than a set of constructor keyword
arguments.

``BALANCED_CAUSAL_MAIN_MODEL`` is registered as a declared target that is not
yet buildable: it requires the ROI-conditioned, relation, two-timescale and
quality-gated modules that later phases own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pig_behavior.classification_v2.models.balanced.balanced_model import (
    BalancedCausalModel,
    BalancedModelConfig,
)
from pig_behavior.classification_v2.models.balanced.baselines import (
    BASELINE_NAMES,
    BASELINE_NUMERIC_GROUPS,
    BASELINE_TEMPORAL_VIEWS,
    baseline_config,
)
from pig_behavior.classification_v2.models.balanced.fusion import (
    EXTENSION_POINTS,
    FusionExtensionPointError,
)

MODEL_REGISTRY_VERSION = "classification_v2.balanced_model_registry.v1"

BALANCED_MAIN_MODEL_NAME = "BALANCED_CAUSAL_MAIN_MODEL"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered model: how to configure it and what it is allowed to use."""

    name: str
    builder: Callable[..., BalancedModelConfig] | None
    temporal_view: str
    numeric_groups: tuple[str, ...]
    implemented: bool
    pending_modules: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "temporal_view": self.temporal_view,
            "numeric_groups": list(self.numeric_groups),
            "implemented": self.implemented,
            "pending_modules": list(self.pending_modules),
        }


def _baseline_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        builder=lambda **overrides: baseline_config(name, **overrides),
        temporal_view=BASELINE_TEMPORAL_VIEWS[name],
        numeric_groups=BASELINE_NUMERIC_GROUPS[name],
        implemented=True,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    name: _baseline_spec(name) for name in BASELINE_NAMES
}
MODEL_REGISTRY[BALANCED_MAIN_MODEL_NAME] = ModelSpec(
    name=BALANCED_MAIN_MODEL_NAME,
    builder=None,
    temporal_view="T6_TARGET_PLUS_H12",
    numeric_groups=(
        "bbox_xywh_n",
        "bbox_shape_n",
        "motion_delta",
        "roi_class_relation",
        "social_relation",
    ),
    implemented=False,
    pending_modules=tuple(sorted(EXTENSION_POINTS)),
)

BALANCED_MODEL_NAMES: tuple[str, ...] = tuple(MODEL_REGISTRY)


def model_spec(name: str) -> ModelSpec:
    """Return one registered spec or reject an unknown model name."""

    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown balanced model={name}; registered models are "
            f"{list(BALANCED_MODEL_NAMES)}"
        ) from exc


def build_config(name: str, **overrides: Any) -> BalancedModelConfig:
    """Return the validated configuration for one registered model."""

    spec = model_spec(name)
    if spec.builder is None:
        raise FusionExtensionPointError(
            f"{spec.name} is declared but not implemented; it depends on "
            f"{list(spec.pending_modules)}, which later phases own. Build one "
            f"of {list(BASELINE_NAMES)} for the current baseline ladder."
        )
    return spec.builder(**overrides)


def build_model(name: str, **overrides: Any) -> BalancedCausalModel:
    """Build one registered model with its validated configuration."""

    return BalancedCausalModel(build_config(name, **overrides))


def model_spec_contract(name: str) -> dict[str, Any]:
    """Serialize one registered model for a run manifest."""

    spec = model_spec(name)
    payload = {
        "registry_version": MODEL_REGISTRY_VERSION,
        **spec.to_payload(),
    }
    if spec.builder is not None:
        payload["model_config"] = spec.builder().to_payload()
    return payload


def registry_contract() -> dict[str, Any]:
    """Serialize the whole registry for audits."""

    return {
        "registry_version": MODEL_REGISTRY_VERSION,
        "models": [MODEL_REGISTRY[name].to_payload() for name in BALANCED_MODEL_NAMES],
        "declared_extension_points": dict(EXTENSION_POINTS),
    }


__all__ = [
    "BALANCED_MAIN_MODEL_NAME",
    "BALANCED_MODEL_NAMES",
    "MODEL_REGISTRY",
    "MODEL_REGISTRY_VERSION",
    "ModelSpec",
    "build_config",
    "build_model",
    "model_spec",
    "model_spec_contract",
    "registry_contract",
]
