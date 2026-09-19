from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
    MotionSchemaError,
    motion_schema_hash,
    motion_schema_metadata,
    require_motion_schema,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_temporal_deltas,
    _add_temporal_unit_aggregates,
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_FRAME_FEATURES,
    export_spatial_sequences,
)


def _rows(
    *,
    timestamps: list[float],
    x: list[float],
    y: list[float] | None = None,
    width: list[float] | None = None,
    height: list[float] | None = None,
    units: list[str] | None = None,
    actors: list[str] | None = None,
    bbox_valid: list[bool] | None = None,
) -> pd.DataFrame:
    count = len(timestamps)
    y = y or [0.0] * count
    width = width or [0.2] * count
    height = height or [0.1] * count
    units = units or ["unit-a"] * count
    actors = actors or ["actor-a"] * count
    bbox_valid = bbox_valid or [True] * count
    area = [w * h for w, h in zip(width, height, strict=True)]
    aspect = [w / h for w, h in zip(width, height, strict=True)]
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * count,
            "dataset_id": ["fixture"] * count,
            "video_key": ["video-a"] * count,
            "object_track_key": actors,
            "temporal_unit_key": units,
            "frame_index": list(range(count)),
            "timestamp_sec": timestamps,
            "cx_n": x,
            "cy_n": y,
            "bw_n": width,
            "bh_n": height,
            "area_n": area,
            "aspect_ratio": aspect,
            "box_diag_n": np.hypot(width, height),
            "bbox_valid": bbox_valid,
        }
    )


def _native_rows(**kwargs: object) -> pd.DataFrame:
    frame = _rows(**kwargs)
    count = len(frame)
    frame["scene_frame_uid"] = [
        f"scene-{index}" for index in range(count)
    ]
    frame["frame_uid"] = [
        f"scene-{index}|actor-a" for index in range(count)
    ]
    frame["pig_id"] = "ID_1"
    frame["track_id"] = "track-1"
    frame["behavior"] = "move"
    frame["x1"] = (frame["cx_n"] - frame["bw_n"] / 2.0) * 100.0
    frame["x2"] = (frame["cx_n"] + frame["bw_n"] / 2.0) * 100.0
    frame["y1"] = (frame["cy_n"] - frame["bh_n"] / 2.0) * 100.0
    frame["y2"] = (frame["cy_n"] + frame["bh_n"] / 2.0) * 100.0
    frame["bbox_area"] = frame["area_n"] * 10_000.0
    frame["feature_computation_grain"] = "FRAME_LOCAL_PRIMITIVES"
    frame["pair_scope_key"] = ""
    return frame


def _attach_unavailable_relation_contract(
    frames: pd.DataFrame,
) -> pd.DataFrame:
    additions: dict[str, object] = {}
    for group in ("roi_class_relation", "social_relation"):
        for feature_name in SPATIAL_PREDICTIVE_FEATURES[group]:
            if feature_name not in frames:
                additions[feature_name] = 0.0
    for roi_class in ("feeder", "drinker", "toy"):
        column = f"roi_{roi_class}_available"
        if column not in frames:
            additions[column] = False
    if "nearest_partner_key" not in frames:
        additions["nearest_partner_key"] = ""
    return pd.concat(
        [frames, pd.DataFrame(additions, index=frames.index)],
        axis=1,
    ).copy()


def _motion(**kwargs: object) -> pd.DataFrame:
    return _add_temporal_deltas(_rows(**kwargs)).sort_values(
        ["temporal_unit_key", "frame_index"],
        kind="mergesort",
    )


def _windows(timestamps: list[float]) -> pd.DataFrame:
    count = len(timestamps)
    window_id = f"actor-a|win={count}|0-{count - 1}"
    return pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": ["actor-a"],
            "window_start_frame": [0],
            "window_end_frame": [count - 1],
            "window_length_frames": [count],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": [f"T{count}_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": [
                json.dumps(list(range(count)), separators=(",", ":"))
            ],
            "selected_frame_indices": [
                json.dumps(list(range(count)), separators=(",", ":"))
            ],
            "selected_timestamps_seconds": [
                json.dumps(timestamps, separators=(",", ":"))
            ],
            "pair_delta_frames": [
                json.dumps([1] * max(0, count - 1))
            ],
            "pair_delta_seconds": [
                json.dumps(
                    [
                        timestamps[index] - timestamps[index - 1]
                        for index in range(1, count)
                    ]
                )
            ],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )


def test_authoritative_motion_schema_is_exact_and_deterministic() -> None:
    assert MOTION_SCHEMA_ID == "schema.pig_strenet_motion_v2"
    assert MOTION_SCHEMA_VERSION == "classification_v2.motion_tensor.v2"
    assert MOTION_SCHEMA_DIMENSION == 12
    assert len(set(MOTION_FEATURE_NAMES)) == 12
    assert motion_schema_hash() == MOTION_SCHEMA_HASH
    assert SPATIAL_FRAME_FEATURES["motion_delta"] == list(
        MOTION_FEATURE_NAMES
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing_required_motion_features"),
        ("reordered", "motion_feature_order_mismatch"),
        ("duplicate", "duplicate_motion_features"),
        ("wrong_hash", "motion_schema_schema_hash_mismatch"),
        ("wrong_version", "motion_schema_schema_version_mismatch"),
        ("wrong_dimension", "motion_schema_dimension_mismatch"),
    ],
)
def test_motion_schema_negative_preflights_fail_closed(
    mutation: str,
    message: str,
) -> None:
    names = list(MOTION_FEATURE_NAMES)
    columns = [*names, *MOTION_REQUIRED_MASKS]
    metadata = copy.deepcopy(motion_schema_metadata())
    if mutation == "missing":
        columns.remove(names[-1])
    elif mutation == "reordered":
        names[0], names[1] = names[1], names[0]
    elif mutation == "duplicate":
        names[-1] = names[0]
    elif mutation == "wrong_hash":
        metadata["schema_hash"] = "0" * 64
    elif mutation == "wrong_version":
        metadata["schema_version"] = "wrong"
    elif mutation == "wrong_dimension":
        metadata["dimension"] = 11

    with pytest.raises(MotionSchemaError, match=message):
        require_motion_schema(
            source_columns=columns,
            actual_feature_names=names,
            actual_masks=MOTION_REQUIRED_MASKS,
            metadata=metadata,
        )


def test_constant_horizontal_velocity_and_midpoint_time() -> None:
    output = _motion(timestamps=[0.0, 1.0, 3.0], x=[0.0, 2.0, 6.0])
    assert output["velocity_valid"].tolist() == [False, True, True]
    assert output["vx_n_per_second"].tolist()[1:] == pytest.approx([2.0, 2.0])
    assert output["vy_n_per_second"].tolist()[1:] == pytest.approx([0.0, 0.0])
    assert output["speed_n_per_second"].tolist()[1:] == pytest.approx([2.0, 2.0])
    assert output["velocity_sample_time_sec"].tolist()[1:] == pytest.approx(
        [0.5, 2.0]
    )
    assert output.iloc[2]["acceleration_delta_t_sec"] == pytest.approx(1.5)
    assert output.iloc[2]["tangential_acceleration_n_per_second2"] == 0.0
    assert output.iloc[2]["ax_n_per_second2"] == 0.0
    assert output.iloc[2]["acceleration_vector_magnitude_n_per_second2"] == 0.0


def test_constant_diagonal_velocity() -> None:
    output = _motion(
        timestamps=[0.0, 2.0],
        x=[0.0, 6.0],
        y=[0.0, 8.0],
    )
    assert output.iloc[1]["vx_n_per_second"] == pytest.approx(3.0)
    assert output.iloc[1]["vy_n_per_second"] == pytest.approx(4.0)
    assert output.iloc[1]["speed_n_per_second"] == pytest.approx(5.0)


def test_bbox_rates_use_pair_delta_time() -> None:
    output = _motion(
        timestamps=[0.0, 2.0],
        x=[0.0, 0.0],
        width=[0.2, 0.4],
        height=[0.1, 0.2],
    )
    row = output.iloc[1]
    assert bool(row["bbox_rate_valid"])
    assert row["bw_rate_n_per_second"] == pytest.approx(0.1)
    assert row["bh_rate_n_per_second"] == pytest.approx(0.05)
    assert row["area_rate_n_per_second"] == pytest.approx(0.03)
    assert row["aspect_ratio_rate_per_second"] == pytest.approx(0.0)


def test_speed_change_fixed_direction_separates_accelerations() -> None:
    output = _motion(
        timestamps=[0.0, 1.0, 2.0],
        x=[0.0, 1.0, 3.0],
    )
    row = output.iloc[2]
    assert row["tangential_acceleration_n_per_second2"] == pytest.approx(1.0)
    assert row["ax_n_per_second2"] == pytest.approx(1.0)
    assert row["ay_n_per_second2"] == pytest.approx(0.0)
    assert row["acceleration_vector_magnitude_n_per_second2"] == pytest.approx(
        1.0
    )
    assert row["direction_change_rad"] == pytest.approx(0.0)


def test_constant_speed_changing_direction_has_vector_acceleration() -> None:
    output = _motion(
        timestamps=[0.0, 1.0, 2.0],
        x=[0.0, 1.0, 1.0],
        y=[0.0, 0.0, 1.0],
    )
    row = output.iloc[2]
    assert row["tangential_acceleration_n_per_second2"] == pytest.approx(0.0)
    assert row["ax_n_per_second2"] == pytest.approx(-1.0)
    assert row["ay_n_per_second2"] == pytest.approx(1.0)
    assert row["acceleration_vector_magnitude_n_per_second2"] == pytest.approx(
        np.sqrt(2.0)
    )
    assert row["direction_change_rad"] == pytest.approx(np.pi / 2.0)


def test_direction_wrap_uses_signed_shortest_angle() -> None:
    angle = np.deg2rad(179.0)
    output = _motion(
        timestamps=[0.0, 1.0, 2.0],
        x=[0.0, np.cos(angle), 2.0 * np.cos(angle)],
        y=[0.0, np.sin(angle), 0.0],
    )
    assert output.iloc[2]["direction_change_rad"] == pytest.approx(
        np.deg2rad(2.0)
    )


def test_zero_speed_direction_is_unavailable_not_angle_zero() -> None:
    output = _motion(timestamps=[0.0, 1.0], x=[0.5, 0.5])
    row = output.iloc[1]
    assert bool(row["velocity_valid"])
    assert row["speed_n_per_second"] == 0.0
    assert not bool(row["direction_valid"])
    assert np.isnan(row["direction_rad"])
    assert np.isnan(row["direction_change_rad"])


def test_invalid_middle_velocity_invalidates_both_acceleration_sides() -> None:
    output = _motion(
        timestamps=[0.0, 1.0, 2.0, 3.0],
        x=[0.0, 1.0, 2.0, 3.0],
        bbox_valid=[True, True, False, True],
    )
    assert output["velocity_valid"].tolist() == [False, True, False, False]
    assert not output["vector_acceleration_valid"].any()
    assert output["ax_n_per_second2"].isna().all()


def test_cross_unit_and_actor_reset_preserve_phase1_contract() -> None:
    output = _motion(
        timestamps=[0.0, 1.0, 2.0, 3.0],
        x=[0.0, 1.0, 2.0, 3.0],
        units=["unit-a", "unit-a", "unit-b", "unit-b"],
        actors=["actor-a", "actor-a", "actor-a", "actor-b"],
    )
    assert output["velocity_valid"].tolist() == [False, True, False, False]
    assert not output["vector_acceleration_valid"].any()


def test_derivative_coverage_uses_family_specific_denominators() -> None:
    motion = _motion(
        timestamps=[0.0, 1.0, 2.0],
        x=[0.0, 1.0, 2.0],
    )
    output = _add_temporal_unit_aggregates(motion)
    row = output.iloc[0]
    assert row["velocity_possible_count"] == 2
    assert row["velocity_valid_count"] == 2
    assert row["velocity_coverage"] == 1.0
    assert row["acceleration_possible_count"] == 1
    assert row["acceleration_valid_count"] == 1
    assert row["acceleration_coverage"] == 1.0


def test_real_producer_to_real_exporter_is_exactly_12d() -> None:
    timestamps = [0.0, 1.0, 3.0]
    produced = _attach_unavailable_relation_contract(
        build_enhanced_spatiotemporal_features(
            _native_rows(timestamps=timestamps, x=[0.0, 2.0, 6.0])
        )
    )
    exported = export_spatial_sequences(_windows(timestamps), produced)
    assert exported.feature_names["motion_delta"] == list(MOTION_FEATURE_NAMES)
    assert exported.arrays["motion_delta"].shape == (1, 3, 12)
    assert exported.audit["motion_schema_hash"] == MOTION_SCHEMA_HASH
    assert exported.audit["motion_schema_preflight"]["errors"] == []
    assert exported.arrays["velocity_valid_mask"][0].tolist() == [
        0.0,
        1.0,
        1.0,
    ]
    assert exported.arrays["vector_acceleration_valid_mask"][0].tolist() == [
        0.0,
        0.0,
        1.0,
    ]


def test_real_exporter_rejects_missing_producer_motion_feature() -> None:
    timestamps = [0.0, 1.0, 2.0]
    produced = _attach_unavailable_relation_contract(
        build_enhanced_spatiotemporal_features(
            _native_rows(timestamps=timestamps, x=[0.0, 1.0, 2.0])
        )
    ).drop(columns=[MOTION_FEATURE_NAMES[-1]])
    with pytest.raises(
        MotionSchemaError,
        match="missing_required_motion_features",
    ):
        export_spatial_sequences(_windows(timestamps), produced)
