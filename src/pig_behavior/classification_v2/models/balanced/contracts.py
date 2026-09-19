"""Schema-driven batch contract for the balanced causal main model.

The contract is deliberately fail-closed. Feature order is never inferred from
the tensor width, the motion dimension is never hard-coded, and quality or
availability masks are always treated as controls rather than ordinary
predictive features.

Sequence layout
---------------
Every segment (target or causal history) is right-padded and carries an integer
``frame_offset`` per slot, expressed relative to the prediction endpoint:

* the target endpoint has ``frame_offset == 0``;
* every earlier frame has a strictly negative offset;
* any positive offset is future information and is rejected.

That single convention is what makes ``FUTURE_FRAME_DEPENDENCE`` checkable by
the validator instead of by convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    SPATIAL_SCHEMA_HASH,
    SPATIAL_SCHEMA_TOTAL_DIMENSION,
    SPATIAL_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_QUALITY_COLUMNS,
)

BATCH_CONTRACT_VERSION = "classification_v2.balanced_batch_contract.v1"

#: Ordered predictive numeric groups. The order here is the concatenation order
#: used by audits; the model itself encodes each group separately.
NUMERIC_GROUP_NAMES = SPATIAL_PREDICTIVE_GROUP_NAMES

#: Availability/quality controls. These are never counted as predictive width.
QUALITY_CONTROL_NAMES: tuple[str, ...] = tuple(SPATIAL_QUALITY_COLUMNS)
AVAILABILITY_CONTROL_NAMES: tuple[str, ...] = (
    "roi_feeder_available",
    "roi_drinker_available",
    "roi_toy_available",
    "social_neighbor_available",
)

MODALITY_NAMES: tuple[str, ...] = ("actor_images", *NUMERIC_GROUP_NAMES)

BATCH_CONTRACT_CHECKS: tuple[str, ...] = (
    "MOTION_DIMENSION_CONTRACT",
    "FEATURE_ORDER_CONTRACT",
    "MASK_SHAPE_CONTRACT",
    "TARGET_LENGTH_CONTRACT",
    "HISTORY_LENGTH_CONTRACT",
    "FORBIDDEN_FEATURE_CONTRACT",
    "FINITE_VALUE_CONTRACT",
    "BATCH_ALIGNMENT_CONTRACT",
)

#: Field families that must never appear in model X, per the scientific
#: contract. Matching is exact-or-substring on lowercased names.
FORBIDDEN_X_SUBSTRINGS: tuple[str, ...] = (
    "behavior",
    "label",
    "reviewed",
    "review",
    "reviewer",
    "manual",
    "source_type",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "object_track_key",
    "recording_group",
    "split",
    "fold",
    "window_id",
    "temporal_unit_key",
    "native_unit_id",
    "frame_uid",
    "path",
    "target_roi",
    "roi_target",
    "future",
)

NUM_CLASSES = len(VALID_BEHAVIORS)


class TensorContractError(ValueError):
    """Raised when a batch violates the fail-closed tensor contract."""


def numeric_group_feature_names() -> dict[str, tuple[str, ...]]:
    """Return the canonical ordered feature names for every numeric group.

    Motion names come from the motion-schema authority, so the motion
    dimension is always ``len(motion_names)`` and never a literal.
    """

    names: dict[str, tuple[str, ...]] = {}
    for group in NUMERIC_GROUP_NAMES:
        if group == "motion_delta":
            names[group] = tuple(MOTION_FEATURE_NAMES)
            continue
        names[group] = SPATIAL_PREDICTIVE_FEATURES[group]
    return names


def numeric_group_dimensions() -> dict[str, int]:
    """Return each group width derived from its canonical ordered names."""

    return {group: len(names) for group, names in numeric_group_feature_names().items()}


SPATIAL_PREDICTIVE_DIMENSION = sum(numeric_group_dimensions().values())


def spatial_predictive_contract() -> dict[str, Any]:
    """Serialize the derived predictive schema for audits and manifests."""

    dimensions = numeric_group_dimensions()
    return {
        "schema_version": BATCH_CONTRACT_VERSION,
        "spatial_schema_version": SPATIAL_SCHEMA_VERSION,
        "spatial_schema_hash": SPATIAL_SCHEMA_HASH,
        "motion_schema_version": MOTION_SCHEMA_VERSION,
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "motion_dimension": len(MOTION_FEATURE_NAMES),
        "group_dimensions": dict(dimensions),
        "spatial_predictive_dimension": int(sum(dimensions.values())),
        "canonical_spatial_predictive_dimension": (
            SPATIAL_SCHEMA_TOTAL_DIMENSION
        ),
        "quality_control_names": list(QUALITY_CONTROL_NAMES),
        "availability_control_names": list(AVAILABILITY_CONTROL_NAMES),
        "controls_counted_as_predictive_features": False,
        "num_classes": NUM_CLASSES,
    }


@dataclass(frozen=True, slots=True)
class SequenceSegment:
    """One causal segment of a sample: either the target or its history."""

    valid_mask: Tensor
    frame_offsets: Tensor
    images: Tensor | None = None
    numeric_groups: Mapping[str, Tensor] = field(default_factory=dict)
    quality_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.valid_mask.shape[0])

    @property
    def length(self) -> int:
        return int(self.valid_mask.shape[1])


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """Machine-readable batch with target/history masks kept separate."""

    target: SequenceSegment
    history: SequenceSegment | None = None
    numeric_feature_names: Mapping[str, Sequence[str]] = field(default_factory=dict)
    quality_mask_names: Sequence[str] = ()
    modality_availability: Mapping[str, Tensor] = field(default_factory=dict)
    labels: Tensor | None = None
    native_unit_id: Sequence[str] = ()
    window_id: Sequence[str] = ()
    motion_schema_hash: str | None = None
    motion_schema_version: str | None = None

    @property
    def batch_size(self) -> int:
        return self.target.batch_size


@dataclass(frozen=True, slots=True)
class BatchContract:
    """Declared expectations for one model/view combination."""

    required_modalities: tuple[str, ...]
    target_length: int
    history_length: int = 0
    image_channels: int = 3
    image_size: int | None = None
    num_classes: int = NUM_CLASSES
    require_labels: bool = False
    maskable_modalities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.required_modalities) - set(MODALITY_NAMES))
        if unknown:
            raise ValueError(f"unknown required modalities={unknown}")
        unknown_maskable = sorted(set(self.maskable_modalities) - set(MODALITY_NAMES))
        if unknown_maskable:
            raise ValueError(f"unknown maskable modalities={unknown_maskable}")
        if self.target_length <= 0:
            raise ValueError("target_length must be positive")
        if self.history_length < 0:
            raise ValueError("history_length must not be negative")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """One named contract check with an actionable failure message."""

    name: str
    passed: bool
    errors: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "passed": self.passed,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class ContractReport:
    """Full validator result, safe to serialize into a run manifest."""

    checks: tuple[ContractCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(error for check in self.checks for error in check.errors)

    def check(self, name: str) -> ContractCheck:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(f"unknown contract check={name}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BATCH_CONTRACT_VERSION,
            "passed": self.passed,
            "checks": [check.to_payload() for check in self.checks],
        }


def validate_batch(batch: ModelBatch, contract: BatchContract) -> ContractReport:
    """Run every declared contract check and return a structured report."""

    checks = (
        _check_batch_alignment(batch, contract),
        _check_motion_dimension(batch, contract),
        _check_feature_order(batch, contract),
        _check_mask_shape(batch, contract),
        _check_target_length(batch, contract),
        _check_history_length(batch, contract),
        _check_forbidden_features(batch),
        _check_finite_values(batch),
    )
    ordered = tuple(
        next(check for check in checks if check.name == name)
        for name in BATCH_CONTRACT_CHECKS
    )
    return ContractReport(checks=ordered)


def require_batch(batch: ModelBatch, contract: BatchContract) -> ContractReport:
    """Return a clean report or raise one precise transactional error."""

    report = validate_batch(batch, contract)
    if not report.passed:
        raise TensorContractError(
            "balanced batch contract failed: " + "; ".join(report.errors)
        )
    return report


def _segments(batch: ModelBatch) -> tuple[tuple[str, SequenceSegment], ...]:
    items: list[tuple[str, SequenceSegment]] = [("target", batch.target)]
    if batch.history is not None:
        items.append(("history", batch.history))
    return tuple(items)


def _check_batch_alignment(
    batch: ModelBatch,
    contract: BatchContract,
) -> ContractCheck:
    errors: list[str] = []
    expected = batch.target.batch_size
    if expected <= 0:
        errors.append("BATCH_ALIGNMENT_CONTRACT: target batch dimension is empty")
    for role, segment in _segments(batch):
        tensors: dict[str, Tensor] = {"valid_mask": segment.valid_mask}
        tensors["frame_offsets"] = segment.frame_offsets
        if segment.images is not None:
            tensors["images"] = segment.images
        if segment.quality_mask is not None:
            tensors["quality_mask"] = segment.quality_mask
        for name, tensor in segment.numeric_groups.items():
            tensors[f"numeric.{name}"] = tensor
        for name, tensor in tensors.items():
            if int(tensor.shape[0]) != expected:
                errors.append(
                    "BATCH_ALIGNMENT_CONTRACT: "
                    f"{role}.{name} batch={int(tensor.shape[0])} expected={expected}"
                )
    for name, tensor in batch.modality_availability.items():
        if int(tensor.shape[0]) != expected:
            errors.append(
                "BATCH_ALIGNMENT_CONTRACT: "
                f"modality_availability[{name}] batch={int(tensor.shape[0])} "
                f"expected={expected}"
            )
    for name, values in (
        ("native_unit_id", batch.native_unit_id),
        ("window_id", batch.window_id),
    ):
        if len(values) != expected:
            errors.append(
                f"BATCH_ALIGNMENT_CONTRACT: {name} length={len(values)} "
                f"expected={expected}"
            )
    if batch.labels is None:
        if contract.require_labels:
            errors.append("BATCH_ALIGNMENT_CONTRACT: labels are required but absent")
    else:
        if tuple(batch.labels.shape) != (expected,):
            errors.append(
                "BATCH_ALIGNMENT_CONTRACT: labels shape="
                f"{tuple(batch.labels.shape)} expected=({expected},)"
            )
        elif batch.labels.numel():
            low = int(batch.labels.min())
            high = int(batch.labels.max())
            if low < 0 or high >= contract.num_classes:
                errors.append(
                    "BATCH_ALIGNMENT_CONTRACT: labels outside "
                    f"[0,{contract.num_classes}) observed=[{low},{high}]"
                )
    return ContractCheck("BATCH_ALIGNMENT_CONTRACT", not errors, tuple(errors))


def _check_motion_dimension(
    batch: ModelBatch,
    contract: BatchContract,
) -> ContractCheck:
    errors: list[str] = []
    motion_names = tuple(MOTION_FEATURE_NAMES)
    motion_dim = len(motion_names)
    if motion_dim != MOTION_SCHEMA_DIMENSION:
        errors.append(
            "MOTION_DIMENSION_CONTRACT: motion schema authority is inconsistent "
            f"names={motion_dim} declared={MOTION_SCHEMA_DIMENSION}"
        )
    if batch.motion_schema_hash is not None and (
        batch.motion_schema_hash != MOTION_SCHEMA_HASH
    ):
        errors.append(
            "MOTION_DIMENSION_CONTRACT: sidecar motion_schema_hash mismatch "
            f"observed={batch.motion_schema_hash} expected={MOTION_SCHEMA_HASH}"
        )
    if batch.motion_schema_version is not None and (
        batch.motion_schema_version != MOTION_SCHEMA_VERSION
    ):
        errors.append(
            "MOTION_DIMENSION_CONTRACT: sidecar motion_schema_version mismatch "
            f"observed={batch.motion_schema_version} expected={MOTION_SCHEMA_VERSION}"
        )
    if "motion_delta" not in contract.required_modalities:
        return ContractCheck("MOTION_DIMENSION_CONTRACT", not errors, tuple(errors))
    for role, segment in _segments(batch):
        tensor = segment.numeric_groups.get("motion_delta")
        if tensor is None:
            errors.append(
                f"MOTION_DIMENSION_CONTRACT: {role}.motion_delta is required "
                "by the model contract but absent from the batch"
            )
            continue
        if int(tensor.shape[-1]) != motion_dim:
            errors.append(
                f"MOTION_DIMENSION_CONTRACT: {role}.motion_delta width="
                f"{int(tensor.shape[-1])} expected={motion_dim} "
                f"(derived from {motion_dim} canonical motion names)"
            )
    return ContractCheck("MOTION_DIMENSION_CONTRACT", not errors, tuple(errors))


def _check_feature_order(
    batch: ModelBatch,
    contract: BatchContract,
) -> ContractCheck:
    errors: list[str] = []
    canonical = numeric_group_feature_names()
    for group in contract.required_modalities:
        if group not in canonical:
            continue
        expected = canonical[group]
        declared = batch.numeric_feature_names.get(group)
        if declared is None:
            errors.append(
                f"FEATURE_ORDER_CONTRACT: {group} has no declared feature names; "
                "the loader must publish the ordered sidecar names"
            )
            continue
        observed = tuple(str(name) for name in declared)
        duplicates = sorted({name for name in observed if observed.count(name) > 1})
        if duplicates:
            errors.append(
                f"FEATURE_ORDER_CONTRACT: {group} duplicated feature names="
                f"{duplicates}"
            )
        missing = [name for name in expected if name not in observed]
        extra = [name for name in observed if name not in expected]
        if missing:
            errors.append(
                f"FEATURE_ORDER_CONTRACT: {group} missing features={missing}"
            )
        if extra:
            errors.append(f"FEATURE_ORDER_CONTRACT: {group} extra features={extra}")
        if not missing and not extra and observed != expected:
            reordered = [
                {"index": index, "expected": want, "observed": got}
                for index, (want, got) in enumerate(zip(expected, observed, strict=True))
                if want != got
            ]
            errors.append(
                f"FEATURE_ORDER_CONTRACT: {group} feature order mismatch="
                f"{reordered}"
            )
        for role, segment in _segments(batch):
            tensor = segment.numeric_groups.get(group)
            if tensor is None:
                continue
            if int(tensor.shape[-1]) != len(observed):
                errors.append(
                    f"FEATURE_ORDER_CONTRACT: {role}.{group} tensor width="
                    f"{int(tensor.shape[-1])} but declared "
                    f"{len(observed)} feature names"
                )
    return ContractCheck("FEATURE_ORDER_CONTRACT", not errors, tuple(errors))


def _check_mask_shape(batch: ModelBatch, contract: BatchContract) -> ContractCheck:
    errors: list[str] = []
    for role, segment in _segments(batch):
        if segment.valid_mask.ndim != 2:
            errors.append(
                f"MASK_SHAPE_CONTRACT: {role}.valid_mask must be [B,T]; "
                f"observed ndim={segment.valid_mask.ndim}"
            )
            continue
        errors.extend(_binary_mask_errors(f"{role}.valid_mask", segment.valid_mask))
        if segment.frame_offsets.shape != segment.valid_mask.shape:
            errors.append(
                f"MASK_SHAPE_CONTRACT: {role}.frame_offsets shape="
                f"{tuple(segment.frame_offsets.shape)} must equal valid_mask "
                f"shape={tuple(segment.valid_mask.shape)}"
            )
        length = segment.length
        for name, tensor in segment.numeric_groups.items():
            if tensor.ndim != 3 or int(tensor.shape[1]) != length:
                errors.append(
                    f"MASK_SHAPE_CONTRACT: {role}.{name} must be [B,{length},D]; "
                    f"observed shape={tuple(tensor.shape)}"
                )
        if segment.images is not None and (
            segment.images.ndim != 5 or int(segment.images.shape[1]) != length
        ):
            errors.append(
                f"MASK_SHAPE_CONTRACT: {role}.images must be [B,{length},C,H,W]; "
                f"observed shape={tuple(segment.images.shape)}"
            )
        if segment.images is not None and segment.images.ndim == 5:
            channels = int(segment.images.shape[2])
            if channels != contract.image_channels:
                errors.append(
                    f"MASK_SHAPE_CONTRACT: {role}.images channels={channels} "
                    f"expected={contract.image_channels}"
                )
            if contract.image_size is not None:
                observed = (int(segment.images.shape[3]), int(segment.images.shape[4]))
                if observed != (contract.image_size, contract.image_size):
                    errors.append(
                        f"MASK_SHAPE_CONTRACT: {role}.images size={observed} "
                        f"expected={(contract.image_size, contract.image_size)}"
                    )
        if segment.quality_mask is not None:
            if segment.quality_mask.ndim != 3 or int(segment.quality_mask.shape[1]) != length:
                errors.append(
                    f"MASK_SHAPE_CONTRACT: {role}.quality_mask must be "
                    f"[B,{length},Q]; observed shape="
                    f"{tuple(segment.quality_mask.shape)}"
                )
            elif int(segment.quality_mask.shape[2]) != len(batch.quality_mask_names):
                errors.append(
                    f"MASK_SHAPE_CONTRACT: {role}.quality_mask width="
                    f"{int(segment.quality_mask.shape[2])} but "
                    f"{len(batch.quality_mask_names)} quality mask names declared"
                )
            errors.extend(
                _binary_mask_errors(f"{role}.quality_mask", segment.quality_mask)
            )
    for name, tensor in batch.modality_availability.items():
        if tensor.ndim not in (1, 2):
            errors.append(
                f"MASK_SHAPE_CONTRACT: modality_availability[{name}] must be "
                f"[B] or [B,T]; observed ndim={tensor.ndim}"
            )
            continue
        errors.extend(
            _binary_mask_errors(f"modality_availability[{name}]", tensor)
        )
    errors.extend(_missing_modality_errors(batch, contract))
    return ContractCheck("MASK_SHAPE_CONTRACT", not errors, tuple(errors))


def _missing_modality_errors(
    batch: ModelBatch,
    contract: BatchContract,
) -> list[str]:
    """Reject silent zero evidence for absent or maskable modalities."""

    errors: list[str] = []
    maskable = set(contract.maskable_modalities)
    for modality in contract.required_modalities:
        present = _modality_present(batch.target, modality)
        availability = batch.modality_availability.get(modality)
        if not present and modality not in maskable:
            errors.append(
                f"MASK_SHAPE_CONTRACT: required modality {modality} is absent and "
                "is not declared maskable by the model contract"
            )
            continue
        if modality in maskable and availability is None:
            errors.append(
                f"MASK_SHAPE_CONTRACT: maskable modality {modality} requires an "
                "explicit modality_availability entry so a missing modality is "
                "never read as valid zero evidence"
            )
            continue
        if not present and availability is not None and bool(availability.any()):
            errors.append(
                f"MASK_SHAPE_CONTRACT: modality {modality} is absent from the "
                "batch but its availability mask claims it is available"
            )
    return errors


def _modality_present(segment: SequenceSegment, modality: str) -> bool:
    if modality == "actor_images":
        return segment.images is not None
    return modality in segment.numeric_groups


def _check_target_length(
    batch: ModelBatch,
    contract: BatchContract,
) -> ContractCheck:
    errors: list[str] = []
    segment = batch.target
    if segment.length != contract.target_length:
        errors.append(
            f"TARGET_LENGTH_CONTRACT: target length={segment.length} "
            f"expected={contract.target_length}"
        )
    if segment.valid_mask.ndim != 2 or segment.frame_offsets.shape != segment.valid_mask.shape:
        return ContractCheck("TARGET_LENGTH_CONTRACT", False, (*errors, _SHAPE_HINT))
    valid = segment.valid_mask.bool()
    empty_rows = (~valid.any(dim=1)).nonzero().flatten().tolist()
    if empty_rows:
        errors.append(
            "TARGET_LENGTH_CONTRACT: target rows without any valid frame="
            f"{empty_rows[:8]}"
        )
    offsets = segment.frame_offsets.to(torch.int64)
    future = (valid & offsets.gt(0)).nonzero().tolist()
    if future:
        errors.append(
            "TARGET_LENGTH_CONTRACT: target contains frames after the prediction "
            f"endpoint (positive frame_offsets) at {future[:8]}"
        )
    masked = torch.where(valid, offsets, torch.full_like(offsets, -(2**30)))
    endpoints = masked.max(dim=1).values
    if valid.any(dim=1).all():
        bad_endpoint = (endpoints != 0).nonzero().flatten().tolist()
        if bad_endpoint:
            errors.append(
                "TARGET_LENGTH_CONTRACT: target frames must end at prediction "
                f"time (max valid frame_offset == 0); rows={bad_endpoint[:8]}"
            )
    return ContractCheck("TARGET_LENGTH_CONTRACT", not errors, tuple(errors))


def _check_history_length(
    batch: ModelBatch,
    contract: BatchContract,
) -> ContractCheck:
    errors: list[str] = []
    if contract.history_length == 0:
        if batch.history is not None:
            errors.append(
                "HISTORY_LENGTH_CONTRACT: model contract declares no causal "
                "history but the batch supplies a history segment"
            )
        return ContractCheck("HISTORY_LENGTH_CONTRACT", not errors, tuple(errors))
    if batch.history is None:
        errors.append(
            "HISTORY_LENGTH_CONTRACT: model contract declares history_length="
            f"{contract.history_length} but the batch has no history segment"
        )
        return ContractCheck("HISTORY_LENGTH_CONTRACT", False, tuple(errors))
    history = batch.history
    if history.length != contract.history_length:
        errors.append(
            f"HISTORY_LENGTH_CONTRACT: history length={history.length} "
            f"expected={contract.history_length}"
        )
    if history.frame_offsets.shape != history.valid_mask.shape:
        return ContractCheck("HISTORY_LENGTH_CONTRACT", False, (*errors, _SHAPE_HINT))
    target_valid = batch.target.valid_mask.bool()
    target_offsets = batch.target.frame_offsets.to(torch.int64)
    history_valid = history.valid_mask.bool()
    history_offsets = history.frame_offsets.to(torch.int64)
    large = torch.full_like(target_offsets, 2**30)
    target_start = torch.where(target_valid, target_offsets, large).min(dim=1).values
    small = torch.full_like(history_offsets, -(2**30))
    history_end = torch.where(history_valid, history_offsets, small).max(dim=1).values
    has_history = history_valid.any(dim=1)
    violating = (has_history & (history_end >= target_start)).nonzero().flatten()
    if violating.numel():
        rows = violating.tolist()
        errors.append(
            "HISTORY_LENGTH_CONTRACT: causal history must end strictly before "
            f"the target start; violating rows={rows[:8]} "
            f"history_end={history_end[violating][:8].tolist()} "
            f"target_start={target_start[violating][:8].tolist()}"
        )
    future = (history_valid & history_offsets.gt(0)).nonzero().tolist()
    if future:
        errors.append(
            "HISTORY_LENGTH_CONTRACT: history contains frames after the "
            f"prediction endpoint at {future[:8]}"
        )
    return ContractCheck("HISTORY_LENGTH_CONTRACT", not errors, tuple(errors))


def _check_forbidden_features(batch: ModelBatch) -> ContractCheck:
    errors: list[str] = []
    candidates: list[tuple[str, str]] = []
    for group, names in batch.numeric_feature_names.items():
        candidates.append(("numeric_group", group))
        candidates.extend(("numeric_feature", str(name)) for name in names)
    candidates.extend(("quality_mask", str(name)) for name in batch.quality_mask_names)
    for role, segment in _segments(batch):
        candidates.extend(
            (f"{role}_tensor", str(name)) for name in segment.numeric_groups
        )
    forbidden = sorted(
        {
            f"{kind}:{name}"
            for kind, name in candidates
            if _is_forbidden_x_name(name)
        }
    )
    if forbidden:
        errors.append(
            "FORBIDDEN_FEATURE_CONTRACT: fields forbidden in model X were "
            f"declared as predictive inputs={forbidden}"
        )
    return ContractCheck("FORBIDDEN_FEATURE_CONTRACT", not errors, tuple(errors))


def _is_forbidden_x_name(name: str) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return False
    if lowered in {"social_relation", "roi_class_relation"}:
        return False
    return any(token in lowered for token in FORBIDDEN_X_SUBSTRINGS)


def _check_finite_values(batch: ModelBatch) -> ContractCheck:
    errors: list[str] = []
    for role, segment in _segments(batch):
        tensors: dict[str, Tensor] = dict(segment.numeric_groups)
        if segment.images is not None:
            tensors["images"] = segment.images
        for name, tensor in tensors.items():
            if not tensor.is_floating_point():
                continue
            if not bool(torch.isfinite(tensor).all()):
                count = int((~torch.isfinite(tensor)).sum())
                errors.append(
                    f"FINITE_VALUE_CONTRACT: {role}.{name} has {count} "
                    "nonfinite entries; the loader must zero-fill invalid slots "
                    "rather than emit NaN/Inf"
                )
    return ContractCheck("FINITE_VALUE_CONTRACT", not errors, tuple(errors))


def _binary_mask_errors(name: str, tensor: Tensor) -> list[str]:
    if tensor.dtype == torch.bool:
        return []
    if not bool(torch.isfinite(tensor).all()):
        return [f"MASK_SHAPE_CONTRACT: {name} contains nonfinite mask entries"]
    if not bool(torch.all((tensor == 0) | (tensor == 1))):
        return [f"MASK_SHAPE_CONTRACT: {name} must be boolean or strictly 0/1"]
    return []


def forbidden_field_names(names: Iterable[str]) -> list[str]:
    """Return the subset of ``names`` that must never appear in model X."""

    return sorted({str(name) for name in names if _is_forbidden_x_name(str(name))})


_SHAPE_HINT = (
    "MASK_SHAPE_CONTRACT must pass before length semantics can be validated"
)


__all__ = [
    "AVAILABILITY_CONTROL_NAMES",
    "BATCH_CONTRACT_CHECKS",
    "BATCH_CONTRACT_VERSION",
    "FORBIDDEN_X_SUBSTRINGS",
    "MODALITY_NAMES",
    "MOTION_FEATURE_NAMES",
    "MOTION_SCHEMA_DIMENSION",
    "NUMERIC_GROUP_NAMES",
    "NUM_CLASSES",
    "QUALITY_CONTROL_NAMES",
    "SPATIAL_PREDICTIVE_DIMENSION",
    "BatchContract",
    "ContractCheck",
    "ContractReport",
    "ModelBatch",
    "SequenceSegment",
    "TensorContractError",
    "forbidden_field_names",
    "numeric_group_dimensions",
    "numeric_group_feature_names",
    "require_batch",
    "spatial_predictive_contract",
    "validate_batch",
]
