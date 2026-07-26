"""Controlled sampling strategies (loss candidate ``L7`` support).

Every plan is deterministic for a fixed inventory order and seed so a sampling
ablation is reproducible. Sampling is reported separately from the loss, because
combining a balanced sampler with a balanced loss double-corrects the imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pig_behavior.classification_v2.training.balanced.training_mass import (
    TrainingMassError,
    WindowInventory,
    fixed_windows_per_native_unit,
)

SAMPLING_SCHEMA_VERSION = "classification_v2.balanced_sampling.v1"

SAMPLING_STRATEGIES: tuple[str, ...] = (
    "NATURAL",
    "CLASS_AWARE",
    "NATIVE_UNIT_BALANCED",
)


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """One epoch's ordered row indices plus the audit of what it changed."""

    strategy: str
    indices: tuple[int, ...]
    seed: int
    audit: dict[str, Any]

    def __len__(self) -> int:
        return len(self.indices)


def build_sampling_plan(
    inventory: WindowInventory,
    *,
    strategy: str,
    seed: int,
    samples_per_epoch: int | None = None,
    windows_per_unit: int = 1,
) -> SamplingPlan:
    """Build one deterministic epoch sampling plan."""

    if strategy not in SAMPLING_STRATEGIES:
        raise TrainingMassError(
            f"unknown sampling strategy={strategy}; expected one of "
            f"{list(SAMPLING_STRATEGIES)}"
        )
    rng = np.random.default_rng(seed)
    if strategy == "NATURAL":
        indices = rng.permutation(len(inventory)).tolist()
        if samples_per_epoch is not None:
            indices = indices[:samples_per_epoch]
    elif strategy == "CLASS_AWARE":
        indices = _class_aware_indices(inventory, rng, samples_per_epoch)
    else:
        indices = fixed_windows_per_native_unit(
            inventory,
            windows_per_unit=windows_per_unit,
            seed=seed,
        )
        order = rng.permutation(len(indices))
        indices = [indices[int(position)] for position in order]
    audit = _sampling_audit(inventory, indices, strategy)
    return SamplingPlan(
        strategy=strategy,
        indices=tuple(int(value) for value in indices),
        seed=int(seed),
        audit=audit,
    )


def _class_aware_indices(
    inventory: WindowInventory,
    rng: np.random.Generator,
    samples_per_epoch: int | None,
) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for row, index in enumerate(inventory.class_index):
        by_class.setdefault(int(index), []).append(row)
    present = sorted(by_class)
    total = samples_per_epoch or len(inventory)
    picks = rng.integers(0, len(present), size=total)
    return [
        by_class[present[int(pick)]][int(rng.integers(0, len(by_class[present[int(pick)]])))]
        for pick in picks
    ]


def _sampling_audit(
    inventory: WindowInventory,
    indices: list[int],
    strategy: str,
) -> dict[str, Any]:
    sampled_classes: dict[str, int] = {}
    sampled_units: dict[str, int] = {}
    for row in indices:
        name = inventory.class_order[inventory.class_index[row]]
        sampled_classes[name] = sampled_classes.get(name, 0) + 1
        unit = str(inventory.native_unit_id[row])
        sampled_units[unit] = sampled_units.get(unit, 0) + 1
    counts = np.asarray(list(sampled_units.values()), dtype=np.float64)
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "strategy": strategy,
        "sampled_rows": int(len(indices)),
        "sampled_class_counts": sampled_classes,
        "sampled_native_units": int(len(sampled_units)),
        "windows_per_sampled_native_unit_min": int(counts.min()) if counts.size else 0,
        "windows_per_sampled_native_unit_max": int(counts.max()) if counts.size else 0,
        "deterministic_for_fixed_seed": True,
        "combine_with_balanced_loss_warning": (
            "a balanced sampler and a balanced loss both correct the prior; "
            "report them as separate ablation arms"
        ),
    }


__all__ = [
    "SAMPLING_SCHEMA_VERSION",
    "SAMPLING_STRATEGIES",
    "SamplingPlan",
    "build_sampling_plan",
]
