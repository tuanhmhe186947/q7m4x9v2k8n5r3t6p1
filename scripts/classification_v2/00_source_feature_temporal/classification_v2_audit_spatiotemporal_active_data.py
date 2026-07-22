"""Run a memory-bounded before/after audit of pair-derived active data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.pen_context import (
    _add_pen_temporal_derivatives,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    _window_timing_summary,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    MOTION_GRAIN_COLUMNS,
    _add_roi_temporal_columns,
    _add_temporal_deltas,
)

IDENTITY_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
    "frame_index",
    "timestamp_sec",
)
MOTION_INPUT_COLUMNS = (
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag_n",
    "bbox_valid",
)
ROI_COLUMNS = (
    "behavior",
    "roi_target_class",
    "roi_target_available",
    "roi_target_contact",
    "roi_target_near",
    "roi_target_center_inside",
    "roi_target_min_dist_n",
    "roi_target_max_overlap_ratio",
    "roi_target_max_iou",
)
SOCIAL_COLUMNS = (
    "nearest_pig_id",
    "nearest_dist_n",
    "pair_contact_with_nearest",
    "social_density_near_count",
)
PEN_COLUMNS = (
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
    "pen_context_available",
    "pen_center_inside",
    "pen_center_signed_distance_n",
    "pen_boundary_inward_normal_x",
    "pen_boundary_inward_normal_y",
)
OLD_DERIVED_COLUMNS = (
    "speed_n_per_frame",
    "speed_n_per_sec",
    "accel_n_per_frame2",
    "roi_target_entry_event",
    "roi_target_exit_event",
    "roi_target_near_entry_event",
    "roi_target_near_exit_event",
    "approach_speed_n_per_frame",
    "separation_speed_n_per_frame",
    "pen_approach_speed_n_per_frame",
    "pen_retreat_speed_n_per_frame",
    "pen_parallel_speed_n_per_frame",
    "speed_mean_unit",
    "path_length_n_unit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-enhanced-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--stationary-threshold-per-second",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--active-threshold-per-second",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--canonical-source-fps",
        type=float,
        default=30.0,
        help=(
            "Canonical MP4/source-frame clock used for the fixed comparison. "
            "The input timestamp column remains read-only before evidence."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.old_enhanced_csv.is_file():
        raise FileNotFoundError(args.old_enhanced_csv)
    requested = (
        *IDENTITY_COLUMNS,
        *MOTION_INPUT_COLUMNS,
        *ROI_COLUMNS,
        *SOCIAL_COLUMNS,
        *PEN_COLUMNS,
        *OLD_DERIVED_COLUMNS,
    )
    header = pd.read_csv(args.old_enhanced_csv, nrows=0).columns
    missing = sorted(set(requested).difference(header))
    if missing:
        raise ValueError(f"active-data audit missing columns: {missing}")
    frame = pd.read_csv(
        args.old_enhanced_csv,
        usecols=list(dict.fromkeys(requested)),
        low_memory=False,
    )
    if not np.isfinite(args.canonical_source_fps):
        raise ValueError("canonical-source-fps must be finite")
    if args.canonical_source_fps <= 0:
        raise ValueError("canonical-source-fps must be > 0")
    fixed_input = frame.copy()
    fixed_input["timestamp_sec"] = (
        pd.to_numeric(fixed_input["frame_index"], errors="raise")
        / args.canonical_source_fps
    )
    fixed = _add_temporal_deltas(fixed_input)
    fixed = _add_roi_temporal_columns(fixed)
    fixed = _add_social_pair_derivatives(fixed)
    fixed = _add_pen_temporal_derivatives(fixed)

    grain = [column for column in MOTION_GRAIN_COLUMNS if column in fixed]
    frame_index = pd.to_numeric(fixed["frame_index"], errors="raise")
    unit_start = frame_index.eq(
        fixed.assign(_frame_index_num=frame_index)
        .groupby(grain, dropna=False, sort=False)["_frame_index_num"]
        .transform("min")
    )
    payload = {
        "schema_version": "classification_v2.active_motion_audit.v1",
        "input_csv": str(args.old_enhanced_csv),
        "rows": int(len(frame)),
        "native_units": int(frame["temporal_unit_key"].nunique()),
        "timestamp_clock": _timestamp_clock_summary(
            frame,
            fixed_input,
            canonical_source_fps=args.canonical_source_fps,
        ),
        "before": _boundary_metrics(frame, unit_start, legacy=True),
        "after": _boundary_metrics(fixed, unit_start, legacy=False),
        "native_unit_aggregate_changes": _aggregate_changes(frame, fixed),
        "pair_counts_after": _pair_counts(fixed),
        "speed_per_second_by_source": _source_speed_summary(fixed),
        "speed_per_second_by_source_behavior": (
            _source_behavior_speed_summary(fixed)
        ),
        "speed_per_second_by_source_native_unit_length": (
            _source_unit_length_speed_summary(fixed)
        ),
        "activity_ratio_by_source": _source_activity_summary(
            fixed,
            stationary=args.stationary_threshold_per_second,
            active=args.active_threshold_per_second,
        ),
        "synthetic_pen_projection": _pen_projection_invariants(),
        "sparse_duration_counterexample": _sparse_duration_counterexample(),
        "errors": [],
    }
    after = payload["after"]
    for name, value in after.items():
        if name.endswith("nonzero_count") and value != 0:
            payload["errors"].append(f"after_{name}={value}")
    if payload["pair_counts_after"]["nonpositive_frame_pairs"]:
        payload["errors"].append("nonpositive_frame_pairs_after")
    if payload["pair_counts_after"]["nonpositive_time_pairs"]:
        payload["errors"].append("nonpositive_time_pairs_after")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    serializable_payload = _json_safe(payload)
    args.output_json.write_text(
        json.dumps(serializable_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(serializable_payload, ensure_ascii=False, indent=2))
    if payload["errors"]:
        raise ValueError(f"active-data audit failed: {payload['errors']}")


def _boundary_metrics(
    frame: pd.DataFrame,
    unit_start: pd.Series,
    *,
    legacy: bool,
) -> dict[str, int]:
    speed_column = "speed_n_per_sec" if legacy else "speed_n_per_second"
    accel_column = (
        "accel_n_per_frame2"
        if legacy
        else "acceleration_n_per_second2"
    )
    roi_columns = (
        "roi_target_entry_event",
        "roi_target_exit_event",
        "roi_target_near_entry_event",
        "roi_target_near_exit_event",
    )
    return {
        "unit_start_rows": int(unit_start.sum()),
        "unit_start_speed_nonzero_count": _nonzero_at(
            frame,
            unit_start,
            speed_column,
        ),
        "unit_start_acceleration_nonzero_count": _nonzero_at(
            frame,
            unit_start,
            accel_column,
        ),
        "cross_unit_roi_transition_nonzero_count": _any_nonzero_at(
            frame,
            unit_start,
            roi_columns,
        ),
        "cross_unit_approach_speed_nonzero_count": _nonzero_at(
            frame,
            unit_start,
            (
                "approach_speed_n_per_frame"
                if legacy
                else "approach_speed_n_per_second"
            ),
        ),
        "cross_unit_pen_motion_nonzero_count": _any_nonzero_at(
            frame,
            unit_start,
            (
                (
                    "pen_approach_speed_n_per_frame",
                    "pen_retreat_speed_n_per_frame",
                    "pen_parallel_speed_n_per_frame",
                )
                if legacy
                else (
                    "pen_approach_speed_n_per_second",
                    "pen_retreat_speed_n_per_second",
                    "pen_parallel_speed_n_per_second",
                )
            ),
        ),
    }


def _nonzero_at(frame: pd.DataFrame, mask: pd.Series, column: str) -> int:
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return int(values.loc[mask].abs().gt(1e-12).sum())


def _any_nonzero_at(
    frame: pd.DataFrame,
    mask: pd.Series,
    columns: tuple[str, ...],
) -> int:
    values = frame.loc[:, list(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return int(values.fillna(0.0).abs().gt(1e-12).any(axis=1).loc[mask].sum())


def _add_social_pair_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grain = [column for column in MOTION_GRAIN_COLUMNS if column in out]
    ordered = out.sort_values([*grain, "frame_index"], kind="mergesort")
    group = ordered.groupby(grain, dropna=False, sort=False)
    ordered["prev_nearest_pig_id"] = group["nearest_pig_id"].shift(1).fillna("")
    ordered["prev_nearest_dist_n"] = group["nearest_dist_n"].shift(1)
    same_partner = (
        ordered["nearest_pig_id"].fillna("").astype(str).ne("")
        & ordered["nearest_pig_id"].fillna("").astype(str).eq(
            ordered["prev_nearest_pig_id"].astype(str)
        )
    )
    finite = (
        pd.to_numeric(ordered["nearest_dist_n"], errors="coerce").notna()
        & pd.to_numeric(
            ordered["prev_nearest_dist_n"],
            errors="coerce",
        ).notna()
    )
    velocity_valid = (
        ordered["motion_velocity_pair_valid"].fillna(False).astype(bool)
        & same_partner
        & finite
    )
    adjacent_valid = (
        ordered["adjacent_motion_pair_valid"].fillna(False).astype(bool)
        & same_partner
        & finite
    )
    raw_delta = (
        pd.to_numeric(ordered["nearest_dist_n"], errors="coerce")
        - pd.to_numeric(ordered["prev_nearest_dist_n"], errors="coerce")
    )
    seconds = pd.to_numeric(
        ordered["motion_delta_seconds"],
        errors="coerce",
    )
    signed_velocity = (raw_delta / seconds).where(velocity_valid, 0.0)
    ordered["social_velocity_pair_valid"] = velocity_valid
    ordered["social_adjacent_pair_valid"] = adjacent_valid
    ordered["partner_distance_delta_n"] = raw_delta.where(
        velocity_valid,
        0.0,
    )
    ordered["approach_speed_n_per_second"] = (-signed_velocity).clip(
        lower=0.0
    )
    ordered["retreat_speed_n_per_second"] = signed_velocity.clip(lower=0.0)
    return ordered.sort_index(kind="mergesort")


def _aggregate_changes(old: pd.DataFrame, fixed: pd.DataFrame) -> dict[str, int]:
    key = "temporal_unit_key"
    old_units = old.groupby(key, sort=False).first()
    fixed_units = fixed.groupby(key, sort=False).agg(
        speed_mean_fixed=("speed_n_per_frame", "mean"),
        path_length_fixed=("displacement_n", "sum"),
    )
    joined = old_units[["speed_mean_unit", "path_length_n_unit"]].join(
        fixed_units,
        how="inner",
        validate="one_to_one",
    )
    speed_changed = ~np.isclose(
        pd.to_numeric(joined["speed_mean_unit"], errors="coerce"),
        joined["speed_mean_fixed"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    path_changed = ~np.isclose(
        pd.to_numeric(joined["path_length_n_unit"], errors="coerce"),
        joined["path_length_fixed"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    return {
        "compared_native_units": int(len(joined)),
        "speed_mean_changed_units": int(speed_changed.sum()),
        "path_length_changed_units": int(path_changed.sum()),
    }


def _pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "adjacent_motion_pairs": int(
            frame["adjacent_motion_pair_valid"].fillna(False).sum()
        ),
        "sparse_velocity_pairs": int(
            frame["sparse_velocity_pair_valid"].fillna(False).sum()
        ),
        "nonpositive_frame_pairs": int(
            frame["motion_pair_invalid_nonpositive_frame"].fillna(False).sum()
        ),
        "nonpositive_time_pairs": int(
            frame["motion_pair_invalid_nonpositive_time"].fillna(False).sum()
        ),
    }


def _source_speed_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    valid = frame["motion_velocity_pair_valid"].fillna(False).astype(bool)
    for source, group in frame.loc[valid].groupby("source_type", sort=True):
        values = pd.to_numeric(
            group["speed_n_per_second"],
            errors="coerce",
        ).dropna()
        output[str(source)] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "p10": float(values.quantile(0.10)),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
        }
    return output


def _source_behavior_speed_summary(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    valid = frame["motion_velocity_pair_valid"].fillna(False).astype(bool)
    records: list[dict[str, Any]] = []
    for (source, behavior), group in frame.loc[valid].groupby(
        ["source_type", "behavior"],
        sort=True,
    ):
        speed = pd.to_numeric(
            group["speed_n_per_second"],
            errors="coerce",
        ).dropna()
        records.append(
            {
                "source_type": str(source),
                "behavior": str(behavior),
                "valid_pairs": int(len(speed)),
                "p50": float(speed.quantile(0.50)),
                "p90": float(speed.quantile(0.90)),
            }
        )
    return records


def _source_unit_length_speed_summary(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    grain = [column for column in MOTION_GRAIN_COLUMNS if column in frame]
    unit_length = frame.groupby(
        grain,
        dropna=False,
        sort=False,
    )["frame_index"].transform("nunique")
    valid = frame["motion_velocity_pair_valid"].fillna(False).astype(bool)
    work = frame.loc[valid].assign(native_unit_length=unit_length.loc[valid])
    records: list[dict[str, Any]] = []
    for (source, length), group in work.groupby(
        ["source_type", "native_unit_length"],
        sort=True,
    ):
        speed = pd.to_numeric(
            group["speed_n_per_second"],
            errors="coerce",
        ).dropna()
        records.append(
            {
                "source_type": str(source),
                "native_unit_length": int(length),
                "valid_pairs": int(len(speed)),
                "p50": float(speed.quantile(0.50)),
                "p90": float(speed.quantile(0.90)),
            }
        )
    return records


def _source_activity_summary(
    frame: pd.DataFrame,
    *,
    stationary: float,
    active: float,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    valid = frame["motion_velocity_pair_valid"].fillna(False).astype(bool)
    for source, group in frame.loc[valid].groupby("source_type", sort=True):
        speed = pd.to_numeric(group["speed_n_per_second"], errors="coerce")
        finite = speed[np.isfinite(speed)]
        output[str(source)] = {
            "valid_pairs": int(len(finite)),
            "stationary_ratio": float(finite.le(stationary).mean()),
            "active_ratio": float(finite.ge(active).mean()),
            "intermediate_ratio": float(
                finite.gt(stationary).where(finite.lt(active), False).mean()
            ),
        }
    return output


def _timestamp_clock_summary(
    old: pd.DataFrame,
    fixed_input: pd.DataFrame,
    *,
    canonical_source_fps: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "canonical_source_fps": float(canonical_source_fps),
        "canonical_formula": "timestamp_sec=frame_index/source_fps",
        "by_source": {},
    }
    grain = [column for column in MOTION_GRAIN_COLUMNS if column in old]
    for source, old_group in old.groupby("source_type", sort=True):
        positions = old_group.index
        old_ordered = old_group.sort_values([*grain, "frame_index"])
        old_delta = old_ordered.groupby(
            grain,
            dropna=False,
            sort=False,
        )["timestamp_sec"].diff()
        fixed_group = fixed_input.loc[positions]
        fixed_ordered = fixed_group.sort_values([*grain, "frame_index"])
        fixed_delta = fixed_ordered.groupby(
            grain,
            dropna=False,
            sort=False,
        )["timestamp_sec"].diff()
        old_positive = pd.to_numeric(old_delta, errors="coerce")
        old_positive = old_positive[old_positive.gt(0)]
        fixed_positive = pd.to_numeric(fixed_delta, errors="coerce")
        fixed_positive = fixed_positive[fixed_positive.gt(0)]
        output["by_source"][str(source)] = {
            "old_positive_pair_count": int(len(old_positive)),
            "old_median_delta_seconds": (
                float(old_positive.median()) if len(old_positive) else None
            ),
            "fixed_positive_pair_count": int(len(fixed_positive)),
            "fixed_median_delta_seconds": (
                float(fixed_positive.median()) if len(fixed_positive) else None
            ),
        }
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _pen_projection_invariants() -> dict[str, float]:
    def rows(centers: list[tuple[float, float]]) -> pd.DataFrame:
        records = []
        for index, (center_x, center_y) in enumerate(centers):
            records.append(
                {
                    "source_type": "synthetic",
                    "dataset_id": "synthetic",
                    "video_key": "synthetic",
                    "object_track_key": "track",
                    "temporal_unit_key": "unit",
                    "frame_index": index,
                    "timestamp_sec": index / 10.0,
                    "x1": center_x - 1.0,
                    "y1": center_y - 1.0,
                    "x2": center_x + 1.0,
                    "y2": center_y + 1.0,
                    "image_width": 100.0,
                    "image_height": 100.0,
                    "pen_context_available": True,
                    "pen_center_inside": True,
                    "pen_center_signed_distance_n": center_x
                    / np.hypot(100.0, 100.0),
                    "pen_boundary_inward_normal_x": 1.0,
                    "pen_boundary_inward_normal_y": 0.0,
                }
            )
        return pd.DataFrame(records)

    normal = _add_pen_temporal_derivatives(
        rows([(10.0, 10.0), (12.0, 10.0)])
    )
    tangent = _add_pen_temporal_derivatives(
        rows([(10.0, 10.0), (10.0, 12.0)])
    )
    return {
        "pure_normal_parallel_abs_error": float(
            abs(normal.loc[1, "pen_parallel_speed_n_per_second"])
        ),
        "pure_tangent_normal_abs_error": float(
            abs(tangent.loc[1, "pen_normal_speed_n_per_second"])
        ),
    }


def _sparse_duration_counterexample() -> dict[str, Any]:
    rows = pd.DataFrame(
        {
            "frame_index": [0, 5],
            "timestamp_sec": [0.0, 5.0 / 30.0],
        }
    )
    return _window_timing_summary(
        rows,
        start=0,
        end=5,
        expected_slot_count=6,
        default_fps=None,
    )


if __name__ == "__main__":
    main()
