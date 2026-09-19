"""Per-frame grouped numeric encoders and control-mask handling.

Numeric features are never concatenated raw into the classifier. Each declared
group gets its own ``Linear -> LayerNorm -> GELU`` encoder and is projected to a
shared width before fusion, so geometry, motion, ROI relations and social
relations keep separate parameterizations.

Quality and availability masks are controls. They are encoded through a
dedicated control encoder and are never appended to the predictive group list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from pig_behavior.classification_v2.models.balanced.contracts import (
    NUMERIC_GROUP_NAMES,
    numeric_group_feature_names,
)


@dataclass(frozen=True, slots=True)
class NumericEncoderConfig:
    """Validated configuration for the grouped numeric branch."""

    groups: tuple[str, ...]
    embedding_dim: int = 32
    dropout: float = 0.0
    control_embedding_dim: int = 8

    def __post_init__(self) -> None:
        unknown = sorted(set(self.groups) - set(NUMERIC_GROUP_NAMES))
        if unknown:
            raise ValueError(
                f"unknown numeric groups={unknown}; "
                f"expected a subset of {list(NUMERIC_GROUP_NAMES)}"
            )
        if len(set(self.groups)) != len(self.groups):
            raise ValueError(f"duplicated numeric groups={list(self.groups)}")
        if self.embedding_dim <= 0:
            raise ValueError("numeric embedding_dim must be positive")
        if self.control_embedding_dim < 0:
            raise ValueError("control_embedding_dim must not be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("numeric dropout must be in [0,1)")

    def group_dimensions(self) -> dict[str, int]:
        canonical = numeric_group_feature_names()
        return {group: len(canonical[group]) for group in self.groups}

    def to_payload(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "group_dimensions": self.group_dimensions(),
            "embedding_dim": self.embedding_dim,
            "control_embedding_dim": self.control_embedding_dim,
            "dropout": self.dropout,
            "raw_concatenation_into_classifier": False,
            "controls_counted_as_predictive_features": False,
        }


class GroupedNumericEncoder(nn.Module):
    """Encode each declared numeric group with its own normalized projection."""

    def __init__(self, config: NumericEncoderConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = config.group_dimensions()
        self.encoders = nn.ModuleDict(
            {
                group: nn.Sequential(
                    nn.Linear(width, config.embedding_dim),
                    nn.LayerNorm(config.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for group, width in dimensions.items()
            }
        )
        self.expected_dimensions = dict(dimensions)

    @property
    def output_dim(self) -> int:
        return self.config.embedding_dim * len(self.config.groups)

    def forward(
        self,
        groups: Mapping[str, Tensor],
        valid_mask: Tensor,
    ) -> Tensor:
        if not self.config.groups:
            batch, length = valid_mask.shape
            return valid_mask.new_zeros((batch, length, 0), dtype=torch.float32)
        valid = valid_mask.bool()
        encoded: list[Tensor] = []
        for group in self.config.groups:
            tensor = groups.get(group)
            if tensor is None:
                raise ValueError(
                    f"numeric group {group} is required by this model but is "
                    "absent from the batch"
                )
            expected = self.expected_dimensions[group]
            if tensor.ndim != 3 or int(tensor.shape[-1]) != expected:
                raise ValueError(
                    f"numeric group {group} must be [B,T,{expected}]; observed "
                    f"shape={tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"numeric group {group} contains nonfinite entries")
            masked = torch.where(
                valid.unsqueeze(-1),
                tensor.float(),
                torch.zeros_like(tensor, dtype=torch.float32),
            )
            encoded.append(self.encoders[group](masked))
        stacked = torch.cat(encoded, dim=-1)
        return stacked * valid.unsqueeze(-1).to(stacked.dtype)


class ControlMaskEncoder(nn.Module):
    """Encode quality/availability controls without treating them as features.

    The encoder exists so a downstream gate can *see* how trustworthy a slot is.
    Its output is tagged as a control embedding and is kept out of the predictive
    group registry.
    """

    def __init__(self, num_controls: int, embedding_dim: int) -> None:
        super().__init__()
        if num_controls < 0:
            raise ValueError("num_controls must not be negative")
        if embedding_dim < 0:
            raise ValueError("control embedding_dim must not be negative")
        self.num_controls = num_controls
        self.embedding_dim = embedding_dim
        self.encoder: nn.Module | None = None
        if num_controls and embedding_dim:
            self.encoder = nn.Sequential(
                nn.Linear(num_controls, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )

    @property
    def output_dim(self) -> int:
        return self.embedding_dim if self.encoder is not None else 0

    def forward(self, quality_mask: Tensor | None, valid_mask: Tensor) -> Tensor:
        batch, length = valid_mask.shape
        if self.encoder is None:
            return valid_mask.new_zeros((batch, length, 0), dtype=torch.float32)
        if quality_mask is None:
            zeros = valid_mask.new_zeros(
                (batch, length, self.num_controls),
                dtype=torch.float32,
            )
            return self.encoder(zeros) * valid_mask.bool().unsqueeze(-1).to(
                torch.float32
            )
        if quality_mask.ndim != 3 or int(quality_mask.shape[-1]) != self.num_controls:
            raise ValueError(
                f"quality_mask must be [B,T,{self.num_controls}]; observed "
                f"shape={tuple(quality_mask.shape)}"
            )
        encoded = self.encoder(quality_mask.float())
        return encoded * valid_mask.bool().unsqueeze(-1).to(encoded.dtype)


@dataclass(frozen=True, slots=True)
class ModalityAvailability:
    """Per-sample availability of each declared modality (a control signal)."""

    names: tuple[str, ...] = field(default_factory=tuple)

    def vector(
        self,
        availability: Mapping[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Return a ``[B, len(names)]`` float control vector, defaulting to 0."""

        if not self.names:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
        columns: list[Tensor] = []
        for name in self.names:
            tensor = availability.get(name)
            if tensor is None:
                columns.append(
                    torch.zeros(batch_size, dtype=torch.float32, device=device)
                )
                continue
            reduced = tensor.float()
            if reduced.ndim == 2:
                reduced = reduced.amax(dim=1)
            columns.append(reduced.reshape(batch_size))
        return torch.stack(columns, dim=1)


def resolve_quality_control_names(declared: Sequence[str]) -> tuple[str, ...]:
    """Return the loader-declared control names, rejecting blanks/duplicates."""

    names = tuple(str(name).strip() for name in declared)
    if any(not name for name in names):
        raise ValueError("quality mask control names must not be blank")
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicated quality mask control names={duplicates}")
    return names


__all__ = [
    "ControlMaskEncoder",
    "GroupedNumericEncoder",
    "ModalityAvailability",
    "NumericEncoderConfig",
    "resolve_quality_control_names",
]
