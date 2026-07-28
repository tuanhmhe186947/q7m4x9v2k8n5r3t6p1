from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from classification_v2_acceleration_reference import (
    calculate_acceleration_reference,
)

from pig_behavior.classification_v2.contracts.model_io import (
    forbidden_x_columns,
)
from pig_behavior.classification_v2.features.motion_schema import (
    GENERIC_ACCELERATION_ALIAS,
    LEGACY_ACCELERATION_AUDIT_ALIAS,
    MOTION_FEATURE_NAMES,
    MotionSchemaError,
    acceleration_compatibility_registry,
    ambiguous_acceleration_names,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_temporal_deltas,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
    export_legacy_development_spatial_sequences,
    export_spatial_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows(
    timestamps: list[float],
    x: list[float],
    y: list[float],
    *,
    units: list[str] | None = None,
    actors: list[str] | None = None,
) -> pd.DataFrame:
    count = len(timestamps)
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * count,
            "dataset_id": ["fixture"] * count,
            "video_key": ["video-a"] * count,
            "object_track_key": actors or ["actor-a"] * count,
            "temporal_unit_key": units or ["unit-a"] * count,
            "frame_index": list(range(count)),
            "timestamp_sec": timestamps,
            "cx_n": x,
            "cy_n": y,
            "bw_n": [0.2] * count,
            "bh_n": [0.1] * count,
            "area_n": [0.02] * count,
            "aspect_ratio": [2.0] * count,
            "box_diag_n": [math.hypot(0.2, 0.1)] * count,
            "bbox_valid": [True] * count,
        }
    )


def _compare_reference_to_production(
    timestamps: list[float],
    x: list[float],
    y: list[float],
    *,
    units: list[str] | None = None,
    actors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = calculate_acceleration_reference(
        timestamps=timestamps,
        x=x,
        y=y,
        temporal_units=units,
        actors=actors,
    )
    production = _add_temporal_deltas(
        _rows(
            timestamps,
            x,
            y,
            units=units,
            actors=actors,
        )
    ).reset_index(drop=True)
    for column in (
        "vx_n_per_second",
        "vy_n_per_second",
        "speed_n_per_second",
        "direction_rad",
        "direction_change_rad",
        "acceleration_delta_t_sec",
        "tangential_acceleration_n_per_second2",
        "ax_n_per_second2",
        "ay_n_per_second2",
        "acceleration_vector_magnitude_n_per_second2",
    ):
        np.testing.assert_allclose(
            production[column],
            reference[column],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    return reference, production


def test_constant_speed_straight_line_matches_independent_reference() -> None:
    reference, _ = _compare_reference_to_production(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
    )
    assert reference.loc[2, "tangential_acceleration_n_per_second2"] == 0.0
    assert reference.loc[2, "ax_n_per_second2"] == 0.0
    assert reference.loc[2, "ay_n_per_second2"] == 0.0
    assert (
        reference.loc[
            2,
            "acceleration_vector_magnitude_n_per_second2",
        ]
        == 0.0
    )


def test_increasing_speed_straight_line_separates_quantities() -> None:
    reference, _ = _compare_reference_to_production(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0],
    )
    assert reference.loc[2, "tangential_acceleration_n_per_second2"] > 0
    assert reference.loc[2, "ax_n_per_second2"] > 0
    assert reference.loc[2, "ay_n_per_second2"] == 0.0
    assert reference.loc[
        2,
        "acceleration_vector_magnitude_n_per_second2",
    ] == pytest.approx(abs(reference.loc[2, "ax_n_per_second2"]))


def test_constant_speed_turn_has_zero_tangential_nonzero_vector() -> None:
    reference, _ = _compare_reference_to_production(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    )
    assert reference.loc[2, "tangential_acceleration_n_per_second2"] == 0.0
    assert reference.loc[2, "ax_n_per_second2"] != 0.0
    assert reference.loc[2, "ay_n_per_second2"] != 0.0
    assert (
        reference.loc[
            2,
            "acceleration_vector_magnitude_n_per_second2",
        ]
        > 0
    )


def test_irregular_timestamps_use_velocity_midpoint_denominator() -> None:
    reference, _ = _compare_reference_to_production(
        [0.0, 1.0, 3.0],
        [0.0, 2.0, 6.0],
        [0.0, 0.0, 0.0],
    )
    assert reference.loc[2, "acceleration_delta_t_sec"] == pytest.approx(1.5)
    assert reference.loc[2, "tangential_acceleration_n_per_second2"] == 0.0
    assert reference.loc[2, "ax_n_per_second2"] == 0.0


def test_missing_first_velocity_is_unavailable_not_zero() -> None:
    _, production = _compare_reference_to_production(
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    )
    assert not bool(production.loc[1, "tangential_acceleration_valid"])
    assert pd.isna(
        production.loc[1, "tangential_acceleration_n_per_second2"]
    )


def test_cross_unit_boundary_resets_velocity_and_acceleration() -> None:
    _, production = _compare_reference_to_production(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        units=["unit-a", "unit-a", "unit-b", "unit-b"],
    )
    assert not bool(production.loc[2, "velocity_valid"])
    assert not bool(production.loc[3, "tangential_acceleration_valid"])
    assert pd.isna(
        production.loc[3, "tangential_acceleration_n_per_second2"]
    )


def test_stationary_valid_pair_has_no_observed_direction() -> None:
    _, production = _compare_reference_to_production(
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    )
    assert bool(production.loc[1, "velocity_valid"])
    assert production.loc[1, "speed_n_per_second"] == 0.0
    assert not bool(production.loc[1, "direction_valid"])
    assert pd.isna(production.loc[1, "direction_rad"])


def test_direction_change_wraps_to_shortest_signed_angle() -> None:
    angle_a = math.radians(179.0)
    angle_b = math.radians(-179.0)
    _, production = _compare_reference_to_production(
        [0.0, 1.0, 2.0],
        [0.0, math.cos(angle_a), math.cos(angle_a) + math.cos(angle_b)],
        [0.0, math.sin(angle_a), math.sin(angle_a) + math.sin(angle_b)],
    )
    assert production.loc[2, "direction_change_rad"] == pytest.approx(
        math.radians(2.0)
    )


def test_generic_alias_is_audit_only_and_absent_from_current_model_x() -> None:
    registry = acceleration_compatibility_registry()
    alias = registry[LEGACY_ACCELERATION_AUDIT_ALIAS]
    assert alias["predictive"] is False
    assert alias["deprecated"] is True
    assert alias["semantic_target"] == (
        "tangential_acceleration_n_per_second2"
    )
    assert alias["allowed_in_current_export"] is False
    assert alias["allowed_in_model_x"] is False
    assert GENERIC_ACCELERATION_ALIAS not in MOTION_FEATURE_NAMES

    for path in (
        PROJECT_ROOT / "configs/classification_v2/trainer_contract_v1.json",
        PROJECT_ROOT
        / "configs/classification_v2/reviewed_q2_tabular_feature_spec_v1.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload.get("tabular_feature_whitelist", payload.get("features"))
        assert ambiguous_acceleration_names(names) == []
        assert LEGACY_ACCELERATION_AUDIT_ALIAS not in names

    assert forbidden_x_columns([GENERIC_ACCELERATION_ALIAS]) == [
        GENERIC_ACCELERATION_ALIAS
    ]
    assert forbidden_x_columns([LEGACY_ACCELERATION_AUDIT_ALIAS]) == [
        LEGACY_ACCELERATION_AUDIT_ALIAS
    ]


def test_producer_emits_explicit_audit_alias_not_generic_name() -> None:
    output = _add_temporal_deltas(
        _rows(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
        )
    )
    assert GENERIC_ACCELERATION_ALIAS not in output
    assert LEGACY_ACCELERATION_AUDIT_ALIAS in output
    pd.testing.assert_series_equal(
        output[LEGACY_ACCELERATION_AUDIT_ALIAS],
        output["tangential_acceleration_n_per_second2"],
        check_names=False,
    )


def test_current_and_legacy_exporters_reject_generic_alias() -> None:
    frames = _rows(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0],
    )
    frames[GENERIC_ACCELERATION_ALIAS] = 0.0
    windows = pd.DataFrame(
        {
            "window_id": ["window-a"],
            "object_track_key": ["actor-a"],
            "window_start_frame": [0],
            "window_end_frame": [2],
            "window_length_frames": [3],
        }
    )
    current_schema = {"motion_delta": [GENERIC_ACCELERATION_ALIAS]}
    with pytest.raises(MotionSchemaError, match="ambiguous acceleration"):
        export_spatial_sequences(
            windows,
            frames,
            feature_schema=current_schema,
        )

    legacy_motion = list(LEGACY_SPATIAL_FRAME_FEATURES["motion_delta"])
    legacy_motion[-1] = GENERIC_ACCELERATION_ALIAS
    with pytest.raises(MotionSchemaError, match="ambiguous acceleration"):
        export_legacy_development_spatial_sequences(
            windows,
            frames,
            feature_schema={"motion_delta": legacy_motion},
        )
