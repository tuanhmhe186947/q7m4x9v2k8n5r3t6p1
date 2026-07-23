"""Build a bounded, non-promoted Phase 2 production motion trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_HASH,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--units-per-source", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_columns = [
        "source_type",
        "temporal_unit_key",
        "object_track_key",
        "frame_index",
        "cx_n",
        "cy_n",
    ]
    discovery = pd.read_csv(
        args.input_csv,
        usecols=discovery_columns,
        low_memory=False,
    ).sort_values(
        [
            "source_type",
            "temporal_unit_key",
            "object_track_key",
            "frame_index",
        ],
        kind="mergesort",
    )
    discovery_group = discovery.groupby(
        ["source_type", "temporal_unit_key", "object_track_key"],
        sort=False,
    )
    stationary = (
        discovery_group["cx_n"].diff().eq(0)
        & discovery_group["cy_n"].diff().eq(0)
    )
    selected_units: list[str] = []
    source_counts: dict[str, int] = {}
    stationary_source_counts: dict[str, int] = {}
    for source_type in ("cvat_tracking_xml", "legacy_recovered"):
        all_keys = sorted(
            discovery.loc[
                discovery["source_type"].eq(source_type),
                "temporal_unit_key",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        stationary_keys = sorted(
            discovery.loc[
                discovery["source_type"].eq(source_type) & stationary,
                "temporal_unit_key",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        preferred = stationary_keys[:1]
        keys = [
            *preferred,
            *[
                key for key in all_keys if key not in set(preferred)
            ][: args.units_per_source - len(preferred)],
        ]
        selected_units.extend(keys)
        source_counts[source_type] = len(keys)
        stationary_source_counts[source_type] = len(preferred)
    parts: list[pd.DataFrame] = []
    selected_set = set(selected_units)
    for chunk in pd.read_csv(
        args.input_csv,
        low_memory=False,
        chunksize=20_000,
    ):
        keep = chunk["temporal_unit_key"].astype(str).isin(selected_set)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    selected = pd.concat(parts, ignore_index=True)
    produced = build_enhanced_spatiotemporal_features(selected)
    produced = produced.sort_values(
        ["source_type", "temporal_unit_key", "frame_index"],
        kind="mergesort",
    )

    group = produced.groupby(
        [
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "temporal_unit_key",
        ],
        dropna=False,
        sort=False,
    )
    produced["previous_velocity_sample_time"] = group[
        "velocity_sample_time_sec"
    ].shift(1)
    failed_conditions = _failed_conditions(produced)
    trace = pd.DataFrame(
        {
            "temporal_unit_key": produced["temporal_unit_key"],
            "canonical_actor_key": produced["object_track_key"],
            "source_type": produced["source_type"],
            "source_frame_index": produced["frame_index"],
            "timestamp_sec": produced["timestamp_sec"],
            "previous_timestamp_sec": produced["prev_timestamp_sec"],
            "delta_t": produced["delta_time_prev_sec"],
            "valid_motion_pair": produced["valid_motion_pair"],
            "velocity_valid": produced["velocity_valid"],
            "vx_n_per_second": produced["vx_n_per_second"],
            "vy_n_per_second": produced["vy_n_per_second"],
            "speed_n_per_second": produced["speed_n_per_second"],
            "bbox_width_rate_per_second": produced[
                "bw_rate_n_per_second"
            ],
            "bbox_height_rate_per_second": produced[
                "bh_rate_n_per_second"
            ],
            "bbox_area_rate_per_second": produced[
                "area_rate_n_per_second"
            ],
            "bbox_aspect_ratio_rate_per_second": produced[
                "aspect_ratio_rate_per_second"
            ],
            "direction_valid": produced["direction_valid"],
            "direction_change_valid": produced[
                "direction_change_valid"
            ],
            "direction_change_rad": produced["direction_change_rad"],
            "previous_velocity_sample_time": produced[
                "previous_velocity_sample_time"
            ],
            "current_velocity_sample_time": produced[
                "velocity_sample_time_sec"
            ],
            "acceleration_delta_t_sec": produced[
                "acceleration_delta_t_sec"
            ],
            "tangential_acceleration_valid": produced[
                "tangential_acceleration_valid"
            ],
            "tangential_acceleration_value": produced[
                "tangential_acceleration_n_per_second2"
            ],
            "vector_acceleration_valid": produced[
                "vector_acceleration_valid"
            ],
            "ax_n_per_second2": produced["ax_n_per_second2"],
            "ay_n_per_second2": produced["ay_n_per_second2"],
            "acceleration_vector_magnitude_n_per_second2": produced[
                "acceleration_vector_magnitude_n_per_second2"
            ],
            "pair_failed_conditions": failed_conditions,
            "motion_schema_version": produced["motion_schema_version"],
        }
    )
    unit_columns = [
        "temporal_unit_key",
        "source_type",
        "observed_frame_count",
        "possible_pair_count",
        "valid_pair_count",
        "valid_pair_ratio",
        "velocity_valid_count",
        "velocity_coverage",
        "direction_change_valid_count",
        "direction_change_coverage",
        "acceleration_valid_count",
        "acceleration_coverage",
        "speed_n_per_second_mean_unit",
        "tangential_acceleration_mean_unit",
        "vector_acceleration_magnitude_mean_unit",
        "motion_feature_available",
    ]
    summary = (
        produced[unit_columns]
        .drop_duplicates("temporal_unit_key")
        .rename(
            columns={
                "speed_n_per_second_mean_unit": "mean_speed_valid",
                "tangential_acceleration_mean_unit": (
                    "mean_tangential_acceleration_valid"
                ),
                "vector_acceleration_magnitude_mean_unit": (
                    "mean_vector_acceleration_valid"
                ),
            }
        )
        .sort_values(["source_type", "temporal_unit_key"])
    )

    first_key = sorted(selected_units)[0]
    one_unit = produced[
        produced["temporal_unit_key"].astype(str).eq(first_key)
    ].sort_values("frame_index")
    windows = _one_unit_window(one_unit)
    exported = export_spatial_sequences(windows, one_unit)
    tensor_preflight = {
        **exported.audit["motion_schema_preflight"],
        "array_shape": list(exported.arrays["motion_delta"].shape),
        "errors": exported.audit["motion_schema_preflight"]["errors"],
    }
    consistency = _consistency_checks(trace, summary)
    audit = {
        "schema_version": "classification_v2.phase2_bounded_audit.v1",
        "input_csv": str(args.input_csv),
        "input_rows": int(len(discovery)),
        "selected_rows": int(len(produced)),
        "selected_units": int(produced["temporal_unit_key"].nunique()),
        "source_unit_counts": source_counts,
        "stationary_source_unit_counts": stationary_source_counts,
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "motion_feature_names": list(MOTION_FEATURE_NAMES),
        "consistency_checks": consistency,
        "errors": [
            check_id
            for check_id, passed in consistency.items()
            if not passed
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    produced.to_csv(
        args.output_dir / "phase2_bounded_native_evidence.csv",
        index=False,
    )
    trace.to_csv(
        args.output_dir / "phase2_production_motion_trace.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "phase2_production_unit_summary.csv",
        index=False,
    )
    (args.output_dir / "phase2_tensor_schema_preflight.json").write_text(
        json.dumps(tensor_preflight, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "phase2_bounded_regression_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if audit["errors"] or tensor_preflight["errors"]:
        raise SystemExit(1)


def _failed_conditions(frame: pd.DataFrame) -> pd.Series:
    names = [
        ("previous_observation", "previous_observation_available"),
        ("same_temporal_unit", "same_temporal_unit_pair"),
        ("same_actor", "same_actor_trajectory_pair"),
        ("previous_geometry", "previous_geometry_valid"),
        ("current_geometry", "current_geometry_valid"),
        ("valid_delta_time", "valid_delta_time"),
    ]
    result: list[str] = []
    for _, row in frame.iterrows():
        failed = [
            label for label, column in names if not bool(row[column])
        ]
        result.append("|".join(failed))
    return pd.Series(result, index=frame.index)


def _one_unit_window(unit: pd.DataFrame) -> pd.DataFrame:
    indices = unit["frame_index"].astype(int).tolist()
    timestamps = unit["timestamp_sec"].astype(float).tolist()
    start = indices[0]
    window_id = f"{unit.iloc[0]['object_track_key']}|phase2-preflight"
    return pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": [unit.iloc[0]["object_track_key"]],
            "window_start_frame": [indices[0]],
            "window_end_frame": [indices[-1]],
            "window_length_frames": [len(indices)],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": [f"T{len(indices)}_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": [
                json.dumps([value - start for value in indices])
            ],
            "selected_frame_indices": [json.dumps(indices)],
            "selected_timestamps_seconds": [json.dumps(timestamps)],
            "pair_delta_frames": [json.dumps(np.diff(indices).tolist())],
            "pair_delta_seconds": [
                json.dumps(np.diff(timestamps).tolist())
            ],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )


def _consistency_checks(
    trace: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, bool]:
    first = trace.groupby("temporal_unit_key", sort=False).head(1)
    valid = trace["valid_motion_pair"].astype(bool)
    merged = summary.set_index("temporal_unit_key")
    trace_counts = trace.groupby("temporal_unit_key")[
        "valid_motion_pair"
    ].sum()
    expected_possible = (
        trace.groupby("temporal_unit_key").size() - 1
    ).clip(lower=0)
    ratios = (
        merged["valid_pair_count"]
        / merged["possible_pair_count"].replace(0, np.nan)
    ).fillna(0.0)
    return {
        "first_row_each_unit_invalid": bool(
            (~first["valid_motion_pair"].astype(bool)).all()
        ),
        "valid_pair_positive_dt": bool(
            (trace.loc[valid, "delta_t"] > 0).all()
        ),
        "valid_pair_count_matches_trace": bool(
            merged["valid_pair_count"].astype(int).equals(
                trace_counts.reindex(merged.index).astype(int)
            )
        ),
        "possible_pair_count_matches_denominator": bool(
            merged["possible_pair_count"].astype(int).equals(
                expected_possible.reindex(merged.index).astype(int)
            )
        ),
        "valid_pair_ratio_matches": bool(
            np.allclose(merged["valid_pair_ratio"], ratios)
        ),
        "schema_dimension_12": len(MOTION_FEATURE_NAMES) == 12,
    }


if __name__ == "__main__":
    main()
