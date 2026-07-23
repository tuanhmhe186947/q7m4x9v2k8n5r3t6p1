"""Create production-side rows for independent Phase 2 arithmetic review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_temporal_deltas,
    _add_temporal_unit_aggregates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [
        _case("stationary", [0, 1, 2], [0, 0, 0]),
        _case("horizontal", [0, 1, 2], [0, 1, 2]),
        _case("diagonal", [0, 1], [0, 3], [0, 4]),
        _case("irregular", [0, 1, 3], [0, 2, 6]),
        _case(
            "width_only",
            [0, 2],
            [0, 0],
            width=[0.2, 0.4],
        ),
        _case(
            "height_only",
            [0, 2],
            [0, 0],
            height=[0.1, 0.2],
        ),
        _case(
            "area_change",
            [0, 2],
            [0, 0],
            width=[0.2, 0.4],
        ),
        _case(
            "aspect_change",
            [0, 2],
            [0, 0],
            width=[0.2, 0.4],
            height=[0.1, 0.1],
        ),
        _case(
            "invalid_pair",
            [0, 1],
            [0, 1],
            bbox_valid=[True, False],
        ),
        _case("first_valid_velocity", [0, 1], [0, 1]),
        _case("constant_velocity", [0, 1, 2], [0, 1, 2]),
        _case("speed_increase", [0, 1, 2], [0, 1, 3]),
        _case("speed_decrease", [0, 1, 2], [0, 2, 3]),
        _case(
            "direction_change",
            [0, 1, 2],
            [0, 1, 1],
            [0, 0, 1],
        ),
        _angle_wrap_case(),
        _case("zero_speed_direction", [0, 1], [0.5, 0.5]),
        _case(
            "invalid_middle_velocity",
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            bbox_valid=[True, True, False, True],
        ),
        _case(
            "cross_unit",
            [0, 1, 2],
            [0, 1, 2],
            units=["unit-a", "unit-a", "unit-b"],
        ),
        _case(
            "actor_discontinuity",
            [0, 1],
            [0, 1],
            actors=["actor-a", "actor-b"],
        ),
        _case(
            "all_velocity_invalid",
            [0, 1, 2],
            [0, 1, 2],
            bbox_valid=[False, False, False],
        ),
        _case("one_velocity_no_acceleration", [0, 1], [0, 1]),
        _case("one_acceleration", [0, 1, 2], [0, 1, 2]),
    ]
    rows: list[pd.DataFrame] = []
    units: list[pd.DataFrame] = []
    for source in scenarios:
        motion = _add_temporal_deltas(source)
        aggregate = _add_temporal_unit_aggregates(motion)
        rows.append(motion)
        units.append(aggregate.drop_duplicates("temporal_unit_key"))
    row_output = pd.concat(rows, ignore_index=True)
    unit_output = pd.concat(units, ignore_index=True)
    row_columns = [
        "case_id",
        "row_index",
        "temporal_unit_key",
        "object_track_key",
        "frame_index",
        "timestamp_sec",
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "bbox_valid",
        "valid_motion_pair",
        "velocity_valid",
        "bbox_rate_valid",
        "velocity_sample_time_sec",
        "vx_n_per_second",
        "vy_n_per_second",
        "bw_rate_n_per_second",
        "bh_rate_n_per_second",
        "area_rate_n_per_second",
        "aspect_ratio_rate_per_second",
        "speed_n_per_second",
        "direction_valid",
        "direction_change_valid",
        "direction_change_rad",
        "acceleration_delta_t_sec",
        "tangential_acceleration_valid",
        "tangential_acceleration_n_per_second2",
        "vector_acceleration_valid",
        "ax_n_per_second2",
        "ay_n_per_second2",
        "acceleration_vector_magnitude_n_per_second2",
    ]
    unit_columns = [
        "case_id",
        "temporal_unit_key",
        "observed_frame_count",
        "possible_pair_count",
        "valid_pair_count",
        "valid_pair_ratio",
        "velocity_possible_count",
        "velocity_valid_count",
        "velocity_coverage",
        "direction_change_possible_count",
        "direction_change_valid_count",
        "direction_change_coverage",
        "acceleration_possible_count",
        "acceleration_valid_count",
        "acceleration_coverage",
        "speed_n_per_second_mean_unit",
        "tangential_acceleration_mean_unit",
        "vector_acceleration_magnitude_mean_unit",
        "motion_feature_available",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_output[row_columns].to_csv(
        args.output_dir / "phase2_golden_rows.csv",
        index=False,
    )
    unit_output[unit_columns].to_csv(
        args.output_dir / "phase2_golden_unit_aggregates.csv",
        index=False,
    )


def _case(
    case_id: str,
    timestamps: list[float],
    x: list[float],
    y: list[float] | None = None,
    *,
    width: list[float] | None = None,
    height: list[float] | None = None,
    bbox_valid: list[bool] | None = None,
    units: list[str] | None = None,
    actors: list[str] | None = None,
) -> pd.DataFrame:
    count = len(timestamps)
    y = y or [0.0] * count
    width = width or [0.2] * count
    height = height or [0.1] * count
    bbox_valid = bbox_valid or [True] * count
    units = units or [f"unit-{case_id}"] * count
    actors = actors or ["actor-a"] * count
    return pd.DataFrame(
        {
            "case_id": [case_id] * count,
            "row_index": list(range(count)),
            "source_type": ["golden"] * count,
            "dataset_id": ["golden"] * count,
            "video_key": ["golden"] * count,
            "object_track_key": actors,
            "temporal_unit_key": units,
            "frame_index": list(range(count)),
            "timestamp_sec": timestamps,
            "cx_n": x,
            "cy_n": y,
            "bw_n": width,
            "bh_n": height,
            "area_n": [
                w * h for w, h in zip(width, height, strict=True)
            ],
            "aspect_ratio": [
                w / h for w, h in zip(width, height, strict=True)
            ],
            "box_diag_n": np.hypot(width, height),
            "bbox_valid": bbox_valid,
        }
    )


def _angle_wrap_case() -> pd.DataFrame:
    angle = np.deg2rad(179.0)
    return _case(
        "angle_wrap",
        [0, 1, 2],
        [0.0, float(np.cos(angle)), float(2.0 * np.cos(angle))],
        [0.0, float(np.sin(angle)), 0.0],
    )


if __name__ == "__main__":
    main()
