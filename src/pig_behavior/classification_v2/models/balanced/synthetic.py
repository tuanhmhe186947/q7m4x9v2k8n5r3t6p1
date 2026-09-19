"""Deterministic synthetic batches for contract, smoke and overfit tests.

Nothing here reads production media, production labels, or any run root. The
generator produces tensors that satisfy the canonical schema so a test can prove
contract behaviour without touching the lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.models.balanced.contracts import (
    AVAILABILITY_CONTROL_NAMES,
    QUALITY_CONTROL_NAMES,
    BatchContract,
    ModelBatch,
    SequenceSegment,
    numeric_group_feature_names,
)


@dataclass(frozen=True, slots=True)
class SyntheticBatchSpec:
    """Declarative description of one synthetic batch."""

    contract: BatchContract
    batch_size: int = 4
    image_size: int = 16
    seed: int = 20260726
    valid_lengths: tuple[int, ...] | None = None
    include_controls: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("synthetic batch_size must be positive")
        if self.image_size <= 0:
            raise ValueError("synthetic image_size must be positive")
        if self.valid_lengths is not None:
            if len(self.valid_lengths) != self.batch_size:
                raise ValueError("valid_lengths must have one entry per sample")
            if any(
                length <= 0 or length > self.contract.target_length
                for length in self.valid_lengths
            ):
                raise ValueError(
                    "each valid length must be in "
                    f"[1,{self.contract.target_length}]"
                )


def synthetic_batch(spec: SyntheticBatchSpec) -> ModelBatch:
    """Build one deterministic synthetic batch that satisfies the contract."""

    generator = torch.Generator().manual_seed(spec.seed)
    contract = spec.contract
    batch = spec.batch_size
    length = contract.target_length
    lengths = spec.valid_lengths or tuple([length] * batch)

    valid = torch.zeros((batch, length), dtype=torch.bool)
    offsets = torch.zeros((batch, length), dtype=torch.int64)
    for row, count in enumerate(lengths):
        valid[row, :count] = True
        # The last valid slot is the prediction endpoint (offset 0); earlier
        # valid slots are strictly negative; padded slots stay after it.
        for slot in range(length):
            offsets[row, slot] = slot - (count - 1)

    canonical = numeric_group_feature_names()
    numeric_groups: dict[str, Tensor] = {}
    feature_names: dict[str, tuple[str, ...]] = {}
    for modality in contract.required_modalities:
        if modality == "actor_images":
            continue
        width = len(canonical[modality])
        values = torch.randn((batch, length, width), generator=generator)
        numeric_groups[modality] = values * valid.unsqueeze(-1)
        feature_names[modality] = canonical[modality]

    images: Tensor | None = None
    if "actor_images" in contract.required_modalities:
        images = torch.rand(
            (batch, length, contract.image_channels, spec.image_size, spec.image_size),
            generator=generator,
        )
        images = images * valid.view(batch, length, 1, 1, 1)

    quality_mask: Tensor | None = None
    quality_names: tuple[str, ...] = ()
    availability: dict[str, Tensor] = {}
    if spec.include_controls:
        quality_names = tuple(QUALITY_CONTROL_NAMES)
        quality_mask = valid.unsqueeze(-1).expand(batch, length, len(quality_names))
        quality_mask = quality_mask.to(torch.float32).clone()
        for name in AVAILABILITY_CONTROL_NAMES:
            availability[name] = torch.ones(batch, dtype=torch.bool)

    labels = torch.randint(
        0,
        contract.num_classes,
        (batch,),
        generator=generator,
        dtype=torch.int64,
    )
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=valid,
            frame_offsets=offsets,
            images=images,
            numeric_groups=numeric_groups,
            quality_mask=quality_mask,
        ),
        history=None,
        numeric_feature_names=feature_names,
        quality_mask_names=quality_names,
        modality_availability=availability,
        labels=labels,
        native_unit_id=tuple(f"synthetic_native_unit_{index:04d}" for index in range(batch)),
        window_id=tuple(f"synthetic_window_{index:04d}" for index in range(batch)),
        motion_schema_hash=MOTION_SCHEMA_HASH,
        motion_schema_version=MOTION_SCHEMA_VERSION,
    )


def perturb_padded_slots(batch: ModelBatch, *, value: float = 7.5) -> ModelBatch:
    """Return a copy whose padded (invalid) slots carry different values.

    A causal model with endpoint readout must produce identical logits for the
    original and the perturbed batch.
    """

    segment = batch.target
    invalid = ~segment.valid_mask.bool()
    numeric = {
        name: tensor + invalid.unsqueeze(-1).to(tensor.dtype) * value
        for name, tensor in segment.numeric_groups.items()
    }
    images = segment.images
    if images is not None:
        images = images + invalid.view(*invalid.shape, 1, 1, 1).to(images.dtype) * value
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=segment.valid_mask,
            frame_offsets=segment.frame_offsets,
            images=images,
            numeric_groups=numeric,
            quality_mask=segment.quality_mask,
        ),
        history=batch.history,
        numeric_feature_names=batch.numeric_feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability=batch.modality_availability,
        labels=batch.labels,
        native_unit_id=batch.native_unit_id,
        window_id=batch.window_id,
        motion_schema_hash=batch.motion_schema_hash,
        motion_schema_version=batch.motion_schema_version,
    )


def replace_numeric_group(
    batch: ModelBatch,
    group: str,
    *,
    tensor: Tensor | None = None,
    names: tuple[str, ...] | None = None,
    drop: bool = False,
) -> ModelBatch:
    """Return a copy with one numeric group replaced, renamed, or dropped."""

    segment = batch.target
    numeric = dict(segment.numeric_groups)
    feature_names = dict(batch.numeric_feature_names)
    if drop:
        numeric.pop(group, None)
        feature_names.pop(group, None)
    else:
        if tensor is not None:
            numeric[group] = tensor
        if names is not None:
            feature_names[group] = names
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=segment.valid_mask,
            frame_offsets=segment.frame_offsets,
            images=segment.images,
            numeric_groups=numeric,
            quality_mask=segment.quality_mask,
        ),
        history=batch.history,
        numeric_feature_names=feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability=batch.modality_availability,
        labels=batch.labels,
        native_unit_id=batch.native_unit_id,
        window_id=batch.window_id,
        motion_schema_hash=batch.motion_schema_hash,
        motion_schema_version=batch.motion_schema_version,
    )


def synthetic_overfit_dataset(
    contract: BatchContract,
    *,
    batch_size: int = 8,
    image_size: int = 12,
    seed: int = 7,
) -> tuple[ModelBatch, dict[str, Any]]:
    """Build a tiny separable dataset used only to prove optimization works.

    The class signal is injected into the numeric groups and the image
    brightness, so a working model can drive the loss down. This says nothing
    about model quality on real data.
    """

    spec = SyntheticBatchSpec(
        contract=contract,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
    )
    batch = synthetic_batch(spec)
    labels = torch.arange(batch_size, dtype=torch.int64) % contract.num_classes
    signal = labels.float().view(batch_size, 1, 1)
    numeric = {
        name: tensor * 0.05 + signal
        for name, tensor in batch.target.numeric_groups.items()
    }
    images = batch.target.images
    if images is not None:
        images = images * 0.05 + signal.view(batch_size, 1, 1, 1, 1)
        images = images * batch.target.valid_mask.view(batch_size, -1, 1, 1, 1)
    overfit = ModelBatch(
        target=SequenceSegment(
            valid_mask=batch.target.valid_mask,
            frame_offsets=batch.target.frame_offsets,
            images=images,
            numeric_groups=numeric,
            quality_mask=batch.target.quality_mask,
        ),
        history=None,
        numeric_feature_names=batch.numeric_feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability=batch.modality_availability,
        labels=labels,
        native_unit_id=batch.native_unit_id,
        window_id=batch.window_id,
        motion_schema_hash=batch.motion_schema_hash,
        motion_schema_version=batch.motion_schema_version,
    )
    metadata = {
        "production_data_used": False,
        "production_labels_used": False,
        "production_media_used": False,
        "claims_model_quality": False,
    }
    return overfit, metadata


__all__ = [
    "SyntheticBatchSpec",
    "perturb_padded_slots",
    "replace_numeric_group",
    "synthetic_batch",
    "synthetic_overfit_dataset",
]
