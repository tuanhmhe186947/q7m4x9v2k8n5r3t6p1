"""Multimodal fusion, classification head, and auxiliary-head interfaces.

Fusion modes are registered by name so an ablation can be declared in config
rather than by editing model code. Only the modes required by the B0-B3 ladder
are implemented here; the later scientific modules
(``roi_film``, ``partner_relation_tokens``, ``two_timescale_history`` and
``quality_aware_gated`` fusion) are registered as explicit extension points that
fail with an actionable message instead of silently degrading to concatenation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pig_behavior.classification_v2.schema import COARSE_BEHAVIORS, VALID_BEHAVIORS

FUSION_MODES: tuple[str, ...] = (
    "concat_projection",
    "availability_gated",
    "quality_aware_gated",
)

#: Modules specified by the scientific protocol but intentionally not part of
#: the first baseline implementation. Each maps to the phase that owns it.
FUSION_EXTENSION_POINTS: dict[str, str] = {
    "quality_aware_gated": "Phase 6 quality-aware gated fusion",
}

EXTENSION_POINTS: dict[str, str] = {
    "roi_conditioned_film": "Phase 4 ROI-conditioned FiLM modulation",
    "actor_partner_relation_tokens": "Phase 5 actor-partner relation tokens",
    "two_timescale_causal_history": "Phase 4 two-timescale causal history",
    "quality_aware_gated_fusion": "Phase 6 quality-aware gated fusion",
}


class FusionExtensionPointError(NotImplementedError):
    """Raised when a declared-but-unimplemented research module is requested."""


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Validated fusion configuration."""

    mode: str = "concat_projection"
    hidden_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in FUSION_MODES:
            raise ValueError(
                f"unsupported fusion mode={self.mode}; "
                f"expected one of {list(FUSION_MODES)}"
            )
        if self.hidden_dim <= 0:
            raise ValueError("fusion hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("fusion dropout must be in [0,1)")

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "implemented_extension_points": sorted(
                set(FUSION_MODES) - set(FUSION_EXTENSION_POINTS)
            ),
            "declared_extension_points": dict(EXTENSION_POINTS),
        }


class MultimodalFusion(nn.Module):
    """Fuse branch embeddings into one representation.

    ``branch_dims`` maps a branch name to its embedding width. The control
    vector carries availability/quality signals only; it never enters the
    predictive branch list.
    """

    def __init__(
        self,
        config: FusionConfig,
        *,
        branch_dims: Mapping[str, int],
        control_dim: int = 0,
    ) -> None:
        super().__init__()
        if config.mode in FUSION_EXTENSION_POINTS:
            raise FusionExtensionPointError(
                f"fusion mode {config.mode} is a declared extension point owned "
                f"by {FUSION_EXTENSION_POINTS[config.mode]}; it is intentionally "
                "not part of the B0-B3 baseline implementation"
            )
        if not branch_dims:
            raise ValueError("fusion requires at least one branch")
        if any(width <= 0 for width in branch_dims.values()):
            raise ValueError(f"fusion branch widths must be positive: {dict(branch_dims)}")
        self.config = config
        self.branch_names = tuple(branch_dims)
        self.branch_dims = dict(branch_dims)
        self.control_dim = int(control_dim)
        total = sum(branch_dims.values()) + self.control_dim
        self.project = nn.Sequential(
            nn.Linear(total, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.gate: nn.Module | None = None
        if config.mode == "availability_gated":
            gate_input = max(1, self.control_dim)
            self.gate = nn.Sequential(
                nn.Linear(gate_input, len(self.branch_names)),
                nn.Sigmoid(),
            )

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    def forward(
        self,
        branches: Mapping[str, Tensor],
        controls: Tensor | None = None,
    ) -> Tensor:
        missing = [name for name in self.branch_names if name not in branches]
        if missing:
            raise ValueError(f"fusion is missing required branches={missing}")
        ordered = [branches[name] for name in self.branch_names]
        batch = ordered[0].shape[0]
        device = ordered[0].device
        if controls is None:
            controls = torch.zeros(
                (batch, self.control_dim),
                dtype=torch.float32,
                device=device,
            )
        if int(controls.shape[-1]) != self.control_dim:
            raise ValueError(
                f"fusion control vector width={int(controls.shape[-1])} "
                f"expected={self.control_dim}"
            )
        if self.gate is not None:
            gate_input = controls
            if self.control_dim == 0:
                gate_input = torch.ones((batch, 1), dtype=torch.float32, device=device)
            weights = self.gate(gate_input)
            ordered = [
                tensor * weights[:, index : index + 1]
                for index, tensor in enumerate(ordered)
            ]
        stacked = torch.cat([*ordered, controls], dim=-1)
        return self.project(stacked)


class BehaviorClassificationHead(nn.Module):
    """Ten-class behavior head bound to the canonical label order."""

    def __init__(self, input_dim: int, num_classes: int = len(VALID_BEHAVIORS)) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("classification head input_dim must be positive")
        if num_classes != len(VALID_BEHAVIORS):
            raise ValueError(
                f"classification head must emit {len(VALID_BEHAVIORS)} classes "
                f"to match the canonical behavior order; requested={num_classes}"
            )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.class_order = tuple(VALID_BEHAVIORS)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features)


class MaskedAuxiliaryHead(nn.Module):
    """Optional coarse-behavior head with an explicit per-sample supervision mask.

    The mask is required: an auxiliary head must never learn from samples whose
    auxiliary target is undefined.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = len(COARSE_BEHAVIORS),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("auxiliary head input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("auxiliary head needs more than one class")
        self.classifier = nn.Linear(input_dim, num_classes)
        self.class_order = tuple(COARSE_BEHAVIORS)

    def forward(self, features: Tensor, supervision_mask: Tensor) -> dict[str, Tensor]:
        if supervision_mask.ndim != 1 or supervision_mask.shape[0] != features.shape[0]:
            raise ValueError("auxiliary supervision_mask must be [B]")
        logits = self.classifier(features)
        return {
            "logits": logits,
            "supervision_mask": supervision_mask.bool(),
        }


def require_extension_point(name: str) -> None:
    """Fail with an actionable message for a declared-but-unbuilt module."""

    owner = EXTENSION_POINTS.get(name)
    if owner is None:
        raise ValueError(f"unknown extension point={name}")
    raise FusionExtensionPointError(
        f"{name} is a declared extension point owned by {owner}; it is not part "
        "of the B0-B3 baseline implementation"
    )


__all__ = [
    "EXTENSION_POINTS",
    "FUSION_EXTENSION_POINTS",
    "FUSION_MODES",
    "BehaviorClassificationHead",
    "FusionConfig",
    "FusionExtensionPointError",
    "MaskedAuxiliaryHead",
    "MultimodalFusion",
    "require_extension_point",
]
