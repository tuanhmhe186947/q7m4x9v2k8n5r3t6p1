"""Overlapping-window training-mass correction tests."""

from __future__ import annotations

import numpy as np
import pytest

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.balanced.sampling import (
    SAMPLING_STRATEGIES,
    build_sampling_plan,
)
from pig_behavior.classification_v2.training.balanced.training_mass import (
    TrainingMassError,
    WindowInventory,
    fixed_windows_per_native_unit,
    per_window_sample_weights,
    training_mass_audit,
)


def _uneven_inventory() -> WindowInventory:
    """Two native units of the same class with 7 and 2 windows."""

    windows: list[str] = []
    units: list[str] = []
    classes: list[int] = []
    for index in range(7):
        windows.append(f"unit_a_window_{index}")
        units.append("native_unit_a")
        classes.append(0)
    for index in range(2):
        windows.append(f"unit_b_window_{index}")
        units.append("native_unit_b")
        classes.append(0)
    return WindowInventory(
        window_id=tuple(windows),
        native_unit_id=tuple(units),
        class_index=tuple(classes),
    )


def _mixed_inventory() -> WindowInventory:
    windows: list[str] = []
    units: list[str] = []
    classes: list[int] = []
    layout = {"common_unit": (0, 9), "rare_unit": (7, 1), "mid_unit": (3, 4)}
    for unit, (class_index, count) in layout.items():
        for index in range(count):
            windows.append(f"{unit}_window_{index}")
            units.append(unit)
            classes.append(class_index)
    return WindowInventory(
        window_id=tuple(windows),
        native_unit_id=tuple(units),
        class_index=tuple(classes),
    )


def test_per_window_weighting_equalizes_native_unit_mass() -> None:
    inventory = _uneven_inventory()
    weights = per_window_sample_weights(inventory)
    mass: dict[str, float] = {}
    for unit, weight in zip(inventory.native_unit_id, weights, strict=True):
        mass[unit] = mass.get(unit, 0.0) + float(weight)
    assert pytest.approx(mass["native_unit_a"]) == mass["native_unit_b"]
    assert pytest.approx(mass["native_unit_a"]) == 1.0


def test_fixed_windows_per_unit_equalizes_native_unit_mass() -> None:
    inventory = _uneven_inventory()
    selected = fixed_windows_per_native_unit(
        inventory,
        windows_per_unit=3,
        seed=11,
    )
    counts: dict[str, int] = {}
    for row in selected:
        unit = inventory.native_unit_id[row]
        counts[unit] = counts.get(unit, 0) + 1
    assert counts == {"native_unit_a": 3, "native_unit_b": 3}
    repeated = fixed_windows_per_native_unit(inventory, windows_per_unit=3, seed=11)
    assert selected == repeated


@pytest.mark.parametrize(
    "strategy",
    ["PER_WINDOW_WEIGHTING", "FIXED_WINDOWS_PER_NATIVE_UNIT"],
)
def test_training_mass_audit_reports_before_and_after(strategy: str) -> None:
    inventory = _uneven_inventory()
    audit = training_mass_audit(
        inventory,
        strategy=strategy,
        windows_per_unit=3,
        seed=5,
    )
    for key in (
        "WINDOWS_PER_NATIVE_UNIT_DISTRIBUTION",
        "NATIVE_UNIT_TRAINING_MASS_BEFORE_CORRECTION",
        "NATIVE_UNIT_TRAINING_MASS_AFTER_CORRECTION",
        "CLASS_TRAINING_MASS_BEFORE_CORRECTION",
        "CLASS_TRAINING_MASS_AFTER_CORRECTION",
        "MAX_MIN_NATIVE_UNIT_MASS_RATIO",
    ):
        assert key in audit

    before = audit["NATIVE_UNIT_TRAINING_MASS_BEFORE_CORRECTION"]
    after = audit["NATIVE_UNIT_TRAINING_MASS_AFTER_CORRECTION"]
    assert before["native_unit_a"] == pytest.approx(7.0)
    assert before["native_unit_b"] == pytest.approx(2.0)
    assert after["native_unit_a"] == pytest.approx(after["native_unit_b"])
    assert audit["MAX_MIN_NATIVE_UNIT_MASS_RATIO"]["before"] == pytest.approx(3.5)
    assert audit["MAX_MIN_NATIVE_UNIT_MASS_RATIO"]["after"] == pytest.approx(1.0)
    assert audit["equal_mass_within_class_after_correction"] is True


def test_class_weights_are_applied_on_top_of_the_correction() -> None:
    inventory = _mixed_inventory()
    weights = np.ones(len(VALID_BEHAVIORS))
    weights[7] = 4.0
    audit = training_mass_audit(
        inventory,
        strategy="PER_WINDOW_WEIGHTING",
        class_weights=weights,
    )
    after = audit["NATIVE_UNIT_TRAINING_MASS_AFTER_CORRECTION"]
    assert after["rare_unit"] == pytest.approx(4.0)
    assert after["common_unit"] == pytest.approx(1.0)
    assert after["mid_unit"] == pytest.approx(1.0)


def test_inventory_rejects_multi_label_native_units() -> None:
    with pytest.raises(TrainingMassError):
        WindowInventory(
            window_id=("w0", "w1"),
            native_unit_id=("unit", "unit"),
            class_index=(0, 1),
        )
    with pytest.raises(TrainingMassError):
        WindowInventory(
            window_id=("w0", "w0"),
            native_unit_id=("unit_a", "unit_b"),
            class_index=(0, 0),
        )


@pytest.mark.parametrize("strategy", SAMPLING_STRATEGIES)
def test_sampling_plans_are_deterministic(strategy: str) -> None:
    inventory = _mixed_inventory()
    first = build_sampling_plan(
        inventory,
        strategy=strategy,
        seed=17,
        windows_per_unit=2,
    )
    second = build_sampling_plan(
        inventory,
        strategy=strategy,
        seed=17,
        windows_per_unit=2,
    )
    assert first.indices == second.indices
    assert first.audit["sampled_rows"] == len(first.indices)


def test_native_unit_balanced_sampling_equalizes_window_counts() -> None:
    inventory = _mixed_inventory()
    plan = build_sampling_plan(
        inventory,
        strategy="NATIVE_UNIT_BALANCED",
        seed=3,
        windows_per_unit=2,
    )
    assert plan.audit["windows_per_sampled_native_unit_min"] == 2
    assert plan.audit["windows_per_sampled_native_unit_max"] == 2
