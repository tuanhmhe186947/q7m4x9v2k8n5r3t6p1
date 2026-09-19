"""Versioned visual-backbone contracts for controlled classifier baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)

VISUAL_BACKBONE_CONTRACT_VERSION = "classification_v2_visual_backbone_v1"
NO_PRETRAINED_WEIGHTS = "NONE_RANDOM_INIT"
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class VisualBackboneContract:
    """Resolved backbone, weight, output, and input-normalization contract."""

    name: str
    pretrained_weight_enum: str
    output_dim: int
    normalization_name: str
    input_mean: tuple[float, float, float]
    input_std: tuple[float, float, float]
    uses_pretrained_weights: bool
    supported_pretrained_weight_enums: tuple[str, ...]


_SUPPORTED_WEIGHT_ENUMS = {
    "smoke_cnn": (NO_PRETRAINED_WEIGHTS,),
    "resnet18": (
        NO_PRETRAINED_WEIGHTS,
        "ResNet18_Weights.IMAGENET1K_V1",
    ),
    "resnet34": (
        NO_PRETRAINED_WEIGHTS,
        "ResNet34_Weights.IMAGENET1K_V1",
    ),
}
SUPPORTED_VISUAL_BACKBONES = frozenset(_SUPPORTED_WEIGHT_ENUMS)

_TORCHVISION_WEIGHTS: dict[str, Any] = {
    "ResNet18_Weights.IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1,
    "ResNet34_Weights.IMAGENET1K_V1": ResNet34_Weights.IMAGENET1K_V1,
}
_RESNET_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
}
_RESNET_OUTPUT_DIMS = {"resnet18": 512, "resnet34": 512}


def visual_backbone_errors(
    backbone_name: str,
    pretrained_weight_enum: str,
) -> list[str]:
    """Return exact contract violations without constructing or downloading a model."""

    if backbone_name not in SUPPORTED_VISUAL_BACKBONES:
        return [f"unimplemented_backbone={backbone_name}"]
    supported = _SUPPORTED_WEIGHT_ENUMS[backbone_name]
    if pretrained_weight_enum not in supported:
        return [
            "unsupported_pretrained_weight_enum="
            f"backbone:{backbone_name},observed:{pretrained_weight_enum},"
            f"supported:{list(supported)}"
        ]
    return []


def visual_backbone_contract(
    backbone_name: str,
    pretrained_weight_enum: str,
) -> VisualBackboneContract:
    """Resolve metadata only; this function never downloads model weights."""

    errors = visual_backbone_errors(backbone_name, pretrained_weight_enum)
    if errors:
        raise ValueError(f"invalid visual backbone contract: {errors}")
    uses_pretrained = pretrained_weight_enum != NO_PRETRAINED_WEIGHTS
    if backbone_name == "smoke_cnn":
        output_dim = 64
        normalization_name = "identity_unit_rgb"
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
    else:
        output_dim = _RESNET_OUTPUT_DIMS[backbone_name]
        normalization_name = "imagenet_1k_rgb"
        mean = IMAGENET_RGB_MEAN
        std = IMAGENET_RGB_STD
    return VisualBackboneContract(
        name=backbone_name,
        pretrained_weight_enum=pretrained_weight_enum,
        output_dim=output_dim,
        normalization_name=normalization_name,
        input_mean=mean,
        input_std=std,
        uses_pretrained_weights=uses_pretrained,
        supported_pretrained_weight_enums=_SUPPORTED_WEIGHT_ENUMS[backbone_name],
    )


def build_visual_frame_encoder(
    backbone_name: str,
    pretrained_weight_enum: str,
) -> tuple[nn.Module, VisualBackboneContract]:
    """Build one frame encoder from an explicit weight enum.

    Selecting a non-random enum is the explicit request that permits torchvision
    to obtain that exact weight artifact. Tests use ``NONE_RANDOM_INIT`` or call
    :func:`visual_backbone_contract` so they never trigger a download.
    """

    contract = visual_backbone_contract(
        backbone_name,
        pretrained_weight_enum,
    )
    if backbone_name == "smoke_cnn":
        return _build_smoke_cnn(), contract
    weights = (
        None
        if pretrained_weight_enum == NO_PRETRAINED_WEIGHTS
        else _TORCHVISION_WEIGHTS[pretrained_weight_enum]
    )
    model = _RESNET_BUILDERS[backbone_name](weights=weights)
    classifier = getattr(model, "fc", None)
    output_dim = getattr(classifier, "in_features", None)
    if output_dim != contract.output_dim:
        raise RuntimeError(
            "torchvision backbone output drift: "
            f"backbone={backbone_name}, expected={contract.output_dim}, "
            f"observed={output_dim}"
        )
    model.fc = nn.Identity()
    return model, contract


def _build_smoke_cnn() -> nn.Sequential:
    """Preserve the original lightweight frame encoder for bounded smoke tests."""

    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(4, 16),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, 32),
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


__all__ = [
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "NO_PRETRAINED_WEIGHTS",
    "SUPPORTED_VISUAL_BACKBONES",
    "VISUAL_BACKBONE_CONTRACT_VERSION",
    "VisualBackboneContract",
    "build_visual_frame_encoder",
    "visual_backbone_contract",
    "visual_backbone_errors",
]
