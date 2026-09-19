"""Overlapping-window training-mass correction and its audit.

Overlapping windows mean a native temporal unit that happens to yield more
windows would otherwise receive more gradient mass than an equally important
unit that yields fewer. Two permitted corrections are implemented:

``FIXED_WINDOWS_PER_NATIVE_UNIT``
    Sample a fixed number of windows per native unit per epoch.

``PER_WINDOW_WEIGHTING``
    ``sample_weight = native_unit_class_weight / windows_from_the_same_unit``.

Both make the intended training mass of a native unit independent of how many
windows it produced.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

TRAINING_MASS_SCHEMA_VERSION = "classification_v2.balanced_training_mass.v1"

TRAINING_MASS_STRATEGIES: tuple[str, ...] = (
    "FIXED_WINDOWS_PER_NATIVE_UNIT",
    "PER_WINDOW_WEIGHTING",
)


class TrainingMassError(ValueError):
    """Raised when a window inventory cannot support a valid mass correction."""


@dataclass(frozen=True, slots=True)
class WindowInventory:
    """Training windows with their owning native unit and class index.

    Only training-fold rows belong here. ``window_id`` and ``native_unit_id``
    are metadata used for correction and audit; neither ever enters model X.
    """

    window_id: tuple[str, ...]
    native_unit_id: tuple[str, ...]
    class_index: tuple[int, ...]
    class_order: tuple[str, ...] = tuple(VALID_BEHAVIORS)

    def __post_init__(self) -> None:
        sizes = {len(self.window_id), len(self.native_unit_id), len(self.class_index)}
        if len(sizes) != 1:
            raise TrainingMassError(
                "window_id, native_unit_id and class_index must be aligned: "
                f"lengths={sorted(sizes)}"
            )
        if not self.window_id:
            raise TrainingMassError("window inventory must not be empty")
        if len(set(self.window_id)) != len(self.window_id):
            raise TrainingMassError("window_id values must be unique")
        if any(not str(value).strip() for value in self.native_unit_id):
            raise TrainingMassError("native_unit_id values must not be blank")
        if any(
            index < 0 or index >= len(self.class_order) for index in self.class_index
        ):
            raise TrainingMassError(
                f"class_index values must be in [0,{len(self.class_order)})"
            )
        for unit, labels in _unit_labels(self).items():
            if len(labels) != 1:
                raise TrainingMassError(
                    f"native unit {unit} carries multiple class labels="
                    f"{sorted(labels)}; a native unit must have one label"
                )

    def __len__(self) -> int:
        return len(self.window_id)

    @property
    def native_units(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for unit in self.native_unit_id:
            seen.setdefault(unit, None)
        return tuple(seen)

    def windows_per_native_unit(self) -> dict[str, int]:
        counter = Counter(self.native_unit_id)
        return {unit: int(counter[unit]) for unit in self.native_units}

    def native_unit_class_index(self) -> dict[str, int]:
        return {unit: next(iter(labels)) for unit, labels in _unit_labels(self).items()}


def _unit_labels(inventory: WindowInventory) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = {}
    for unit, index in zip(inventory.native_unit_id, inventory.class_index, strict=True):
        labels.setdefault(str(unit), set()).add(int(index))
    return labels


def per_window_sample_weights(
    inventory: WindowInventory,
    class_weights: Sequence[float] | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return strategy-B weights: class weight divided by sibling window count.

    Every native unit ends up with a total training mass equal to its class
    weight, regardless of how many windows it produced.
    """

    weights = _resolved_class_weights(inventory, class_weights)
    per_unit = inventory.windows_per_native_unit()
    return np.asarray(
        [
            weights[index] / float(per_unit[unit])
            for unit, index in zip(
                inventory.native_unit_id,
                inventory.class_index,
                strict=True,
            )
        ],
        dtype=np.float64,
    )


def fixed_windows_per_native_unit(
    inventory: WindowInventory,
    *,
    windows_per_unit: int,
    seed: int,
    allow_replacement: bool = True,
) -> list[int]:
    """Return strategy-A row indices: a fixed window budget per native unit.

    The selection is deterministic for a fixed inventory order and seed.
    """

    if windows_per_unit <= 0:
        raise TrainingMassError("windows_per_unit must be positive")
    rows_by_unit: dict[str, list[int]] = {}
    for row, unit in enumerate(inventory.native_unit_id):
        rows_by_unit.setdefault(str(unit), []).append(row)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for unit in inventory.native_units:
        rows = rows_by_unit[unit]
        if len(rows) >= windows_per_unit:
            picked = rng.choice(len(rows), size=windows_per_unit, replace=False)
        elif allow_replacement:
            picked = rng.choice(len(rows), size=windows_per_unit, replace=True)
        else:
            raise TrainingMassError(
                f"native unit {unit} has {len(rows)} windows but "
                f"windows_per_unit={windows_per_unit} and replacement is off"
            )
        selected.extend(rows[int(index)] for index in np.sort(picked))
    return selected


def training_mass_audit(
    inventory: WindowInventory,
    *,
    strategy: str,
    class_weights: Sequence[float] | Mapping[str, float] | None = None,
    windows_per_unit: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Return the mandatory before/after training-mass audit."""

    if strategy not in TRAINING_MASS_STRATEGIES:
        raise TrainingMassError(
            f"unknown training-mass strategy={strategy}; expected one of "
            f"{list(TRAINING_MASS_STRATEGIES)}"
        )
    weights = _resolved_class_weights(inventory, class_weights)
    per_unit = inventory.windows_per_native_unit()
    unit_class = inventory.native_unit_class_index()

    before_unit = {
        unit: weights[unit_class[unit]] * float(count)
        for unit, count in per_unit.items()
    }
    if strategy == "PER_WINDOW_WEIGHTING":
        sample_weights = per_window_sample_weights(inventory, class_weights)
        after_unit: dict[str, float] = dict.fromkeys(per_unit, 0.0)
        for unit, weight in zip(inventory.native_unit_id, sample_weights, strict=True):
            after_unit[str(unit)] += float(weight)
        selected_rows = list(range(len(inventory)))
    else:
        selected_rows = fixed_windows_per_native_unit(
            inventory,
            windows_per_unit=windows_per_unit,
            seed=seed,
        )
        after_unit = dict.fromkeys(per_unit, 0.0)
        for row in selected_rows:
            unit = str(inventory.native_unit_id[row])
            after_unit[unit] += weights[unit_class[unit]] / float(windows_per_unit)
        sample_weights = np.asarray(
            [
                weights[inventory.class_index[row]] / float(windows_per_unit)
                for row in selected_rows
            ],
            dtype=np.float64,
        )

    before_class = _class_mass(inventory.class_index, np.full(len(inventory), 1.0), weights)
    after_class = _class_mass(
        [inventory.class_index[row] for row in selected_rows],
        sample_weights,
        None,
    )
    fixed_strategy = strategy == "FIXED_WINDOWS_PER_NATIVE_UNIT"
    before_values = np.asarray(list(before_unit.values()), dtype=np.float64)
    after_values = np.asarray(list(after_unit.values()), dtype=np.float64)
    counts = np.asarray(list(per_unit.values()), dtype=np.float64)
    return {
        "schema_version": TRAINING_MASS_SCHEMA_VERSION,
        "strategy": strategy,
        "windows_per_unit": int(windows_per_unit) if fixed_strategy else None,
        "window_rows": int(len(inventory)),
        "native_units": int(len(per_unit)),
        "WINDOWS_PER_NATIVE_UNIT_DISTRIBUTION": {
            "per_unit": dict(per_unit),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
        },
        "NATIVE_UNIT_TRAINING_MASS_BEFORE_CORRECTION": before_unit,
        "NATIVE_UNIT_TRAINING_MASS_AFTER_CORRECTION": after_unit,
        "CLASS_TRAINING_MASS_BEFORE_CORRECTION": before_class,
        "CLASS_TRAINING_MASS_AFTER_CORRECTION": after_class,
        "MAX_MIN_NATIVE_UNIT_MASS_RATIO": {
            "before": _ratio(before_values),
            "after": _ratio(after_values),
        },
        "equal_mass_within_class_after_correction": _equal_within_class(
            after_unit,
            unit_class,
        ),
    }


def _ratio(values: np.ndarray) -> float:
    low = float(values.min())
    high = float(values.max())
    if low <= 0.0:
        return float("inf")
    return high / low


def _equal_within_class(
    unit_mass: Mapping[str, float],
    unit_class: Mapping[str, int],
    *,
    tolerance: float = 1e-9,
) -> bool:
    by_class: dict[int, list[float]] = {}
    for unit, mass in unit_mass.items():
        by_class.setdefault(unit_class[unit], []).append(float(mass))
    return all(
        max(values) - min(values) <= tolerance for values in by_class.values()
    )


def _class_mass(
    class_index: Sequence[int],
    sample_weights: np.ndarray,
    class_weights: np.ndarray | None,
) -> dict[str, float]:
    totals: dict[int, float] = {}
    for index, weight in zip(class_index, sample_weights, strict=True):
        factor = 1.0 if class_weights is None else float(class_weights[int(index)])
        totals[int(index)] = totals.get(int(index), 0.0) + float(weight) * factor
    return {
        VALID_BEHAVIORS[index]: float(value)
        for index, value in sorted(totals.items())
    }


def _resolved_class_weights(
    inventory: WindowInventory,
    class_weights: Sequence[float] | Mapping[str, float] | None,
) -> np.ndarray:
    size = len(inventory.class_order)
    if class_weights is None:
        return np.ones(size, dtype=np.float64)
    if isinstance(class_weights, Mapping):
        missing = [name for name in inventory.class_order if name not in class_weights]
        if missing:
            raise TrainingMassError(f"class weights missing classes={missing}")
        return np.asarray(
            [float(class_weights[name]) for name in inventory.class_order],
            dtype=np.float64,
        )
    values = np.asarray(class_weights, dtype=np.float64)
    if values.shape != (size,):
        raise TrainingMassError(
            f"class weights must have shape ({size},); observed {values.shape}"
        )
    return values


__all__ = [
    "TRAINING_MASS_SCHEMA_VERSION",
    "TRAINING_MASS_STRATEGIES",
    "TrainingMassError",
    "WindowInventory",
    "fixed_windows_per_native_unit",
    "per_window_sample_weights",
    "training_mass_audit",
]
