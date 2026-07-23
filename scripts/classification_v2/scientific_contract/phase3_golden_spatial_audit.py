"""Generate production-side Phase 3 rows for independent arithmetic review."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.social import (
    build_static_social_context_features,
)
from pig_behavior.classification_v2.features.spatial_semantics import (
    axis_normalized_image_distance,
    diagonal_normalized_image_distance,
    is_target_roi_model_forbidden,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    EnhancedFeatureConfig,
    _add_social_context_columns,
    _add_temporal_deltas,
    _add_temporal_unit_aggregates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _distance_cases().to_csv(
        args.output_dir / "phase3_distance_golden_cases.csv",
        index=False,
    )
    _social_cases().to_csv(
        args.output_dir / "phase3_social_golden_cases.csv",
        index=False,
    )
    _roi_cases().to_csv(
        args.output_dir / "phase3_roi_golden_cases.csv",
        index=False,
    )


def _distance_cases() -> pd.DataFrame:
    inputs = [
        ("distance_square_horizontal", 100, 0, 1000, 1000, True),
        ("distance_square_vertical", 0, 100, 1000, 1000, True),
        ("distance_non_square_horizontal", 100, 0, 1000, 500, True),
        ("distance_non_square_vertical", 0, 100, 1000, 500, True),
        ("distance_diagonal_equal_horizontal", 100, 0, 1000, 500, True),
        ("distance_diagonal_equal_vertical", 0, 100, 1000, 500, True),
        ("distance_coincident_centers", 0, 0, 1000, 500, True),
        ("distance_invalid_width", 100, 0, 0, 500, True),
        ("distance_invalid_height", 0, 100, 1000, 0, True),
        ("distance_invalid_actor_geometry", 100, 0, 1000, 500, False),
        ("distance_invalid_partner_geometry", 100, 0, 1000, 500, False),
    ]
    records: list[dict[str, object]] = []
    for case_id, dx, dy, width, height, geometry_valid in inputs:
        axis = axis_normalized_image_distance(dx, dy, width, height)
        diagonal = diagonal_normalized_image_distance(
            dx,
            dy,
            width,
            height,
        )
        available = bool(
            geometry_valid
            and math.isfinite(axis)
            and math.isfinite(diagonal)
        )
        records.append(
            {
                "case_id": case_id,
                "dx_px": dx,
                "dy_px": dy,
                "image_width_px": width,
                "image_height_px": height,
                "actor_geometry_valid": geometry_valid,
                "partner_geometry_valid": geometry_valid,
                "distance_available": available,
                "axis_normalized_distance": axis if available else np.nan,
                "diagonal_normalized_distance": (
                    diagonal if available else np.nan
                ),
                "is_physical_measurement": False,
            }
        )
    return pd.DataFrame(records)


def _social_cases() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scenarios = [
        (
            "social_one_valid_neighbor",
            [_row("A", 0, 0.5), _row("B", 0, 0.6)],
            "A",
        ),
        (
            "social_equal_distance_neighbors",
            [_row("A", 0, 0.5), _row("B", 0, 0.4), _row("C", 0, 0.6)],
            "A",
        ),
        (
            "social_equal_distance_permutation",
            [_row("C", 0, 0.6), _row("A", 0, 0.5), _row("B", 0, 0.4)],
            "A",
        ),
        (
            "social_actor_self_row_present",
            [_row("A", 0, 0.5), _row("B", 0, 0.6)],
            "A",
        ),
        (
            "social_blank_pig_id",
            [_row("A", 0, 0.5, pig_id=""), _row("B", 0, 0.6, pig_id="")],
            "A",
        ),
        (
            "social_duplicate_pig_id",
            [
                _row("A", 0, 0.5),
                _row("B", 0, 0.4, pig_id="duplicate"),
                _row("C", 0, 0.6, pig_id="duplicate"),
            ],
            "A",
        ),
        (
            "social_no_valid_neighbor",
            [_row("A", 0, 0.5)],
            "A",
        ),
    ]
    for case_id, rows, actor in scenarios:
        output = build_static_social_context_features(pd.DataFrame(rows))
        selected = output.loc[
            output["object_track_key"].eq(f"video-a|track={actor}")
        ].iloc[0]
        records.append(_social_record(case_id, selected))

    cross_video = build_static_social_context_features(
        pd.DataFrame(
            [
                _row("A", 0, 0.5, video="video-a", pig_id="same"),
                _row("A", 0, 0.5, video="video-b", pig_id="same"),
            ]
        )
    )
    records.append(
        _social_record(
            "social_same_pig_id_cross_video",
            cross_video.iloc[0],
        )
    )
    records.extend(_continuity_cases())
    return pd.DataFrame(records)


def _continuity_cases() -> list[dict[str, object]]:
    cases = [
        (
            "social_partner_b_remains_b",
            [
                _row("A", 0, 0.5),
                _row("B", 0, 0.6),
                _row("A", 1, 0.5),
                _row("B", 1, 0.6),
            ],
            "A",
        ),
        (
            "social_partner_b_changes_c",
            [
                _row("A", 0, 0.5),
                _row("B", 0, 0.6),
                _row("A", 1, 0.5),
                _row("C", 1, 0.6),
            ],
            "A",
        ),
        (
            "social_partner_b_unavailable",
            [
                _row("A", 0, 0.5),
                _row("B", 0, 0.6),
                _row("A", 1, 0.5),
            ],
            "A",
        ),
        (
            "social_temporal_unit_reset",
            [
                _row("A", 0, 0.5, unit="unit-a"),
                _row("B", 0, 0.6, unit="unit-b"),
                _row("A", 1, 0.5, unit="unit-a-next"),
                _row("B", 1, 0.6, unit="unit-b-next"),
            ],
            "A",
        ),
        (
            "social_actor_identity_reset",
            [
                _row("A0", 0, 0.5),
                _row("B", 0, 0.6),
                _row("A1", 1, 0.5),
                _row("B", 1, 0.6),
            ],
            "A1",
        ),
    ]
    records: list[dict[str, object]] = []
    for case_id, rows, actor in cases:
        output = _add_social_context_columns(
            _pair_support(pd.DataFrame(rows)),
            EnhancedFeatureConfig(),
        )
        selected = output.loc[
            output["object_track_key"].eq(f"video-a|track={actor}")
        ].sort_values("frame_index").iloc[-1]
        records.append(_social_record(case_id, selected))
    return records


def _social_record(
    case_id: str,
    row: pd.Series,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "actor_key": row["object_track_key"],
        "nearest_partner_key": row["nearest_partner_key"],
        "nearest_pig_id": row["nearest_pig_id"],
        "nearest_distance_axis": row["nearest_dist_n"],
        "nearest_distance_diagonal": row["nearest_distance_diagonal"],
        "nearest_neighbor_available": row["nearest_neighbor_available"],
        "nearest_tie_count": row["nearest_tie_count"],
        "same_partner_as_previous": row.get(
            "same_partner_as_previous",
            False,
        ),
        "partner_switch": row.get("partner_switch", False),
        "partner_continuity_valid": row.get(
            "partner_continuity_valid",
            False,
        ),
        "no_neighbor": row.get(
            "no_neighbor",
            not bool(row["nearest_neighbor_available"]),
        ),
    }


def _roi_cases() -> pd.DataFrame:
    cases = [
        (
            "roi_three_available_two_contact",
            [True, True, True, False, False],
            [True, True, False, False, False],
        ),
        ("roi_all_unavailable", [False] * 5, [False] * 5),
        ("roi_all_available_none_contact", [True] * 5, [False] * 5),
        ("roi_all_available_all_contact", [True] * 5, [True] * 5),
        (
            "roi_mixed_invalid_geometry",
            [True, False, True, False, False],
            [True, False, False, False, False],
        ),
        (
            "roi_availability_ratio_denominator",
            [True, True, True, False, False],
            [False] * 5,
        ),
        (
            "roi_contact_ratio_denominator",
            [True, True, True, False, False],
            [True, True, False, False, False],
        ),
        ("roi_zero_placeholder_unavailable", [False] * 3, [False] * 3),
    ]
    records: list[dict[str, object]] = []
    for case_id, available, contact in cases:
        result = _roi_aggregate(available, contact).iloc[0]
        records.append(
            {
                "case_id": case_id,
                "observed_frame_count": result["observed_frame_count"],
                "roi_available_frame_count": result[
                    "target_roi_available_frame_count"
                ],
                "roi_contact_frame_count": result[
                    "target_roi_contact_frame_count"
                ],
                "roi_availability_ratio": result[
                    "target_roi_availability_ratio_unit"
                ],
                "roi_contact_ratio": result[
                    "target_roi_contact_ratio_unit"
                ],
                "target_roi_unit_available": result[
                    "target_roi_unit_available"
                ],
                "model_export_allowed": False,
            }
        )
    records.extend(
        [
            {
                "case_id": "roi_review_only_requested_by_model",
                "model_export_allowed": not is_target_roi_model_forbidden(
                    "target_roi_contact_ratio_unit"
                ),
            },
            {
                "case_id": "roi_label_independent_allowed",
                "model_export_allowed": not is_target_roi_model_forbidden(
                    "roi_feeder_contact"
                ),
            },
        ]
    )
    return pd.DataFrame(records)


def _roi_aggregate(
    available: list[bool],
    contact: list[bool],
) -> pd.DataFrame:
    source = pd.DataFrame(
        [_row("A", index, 0.5) for index in range(len(available))]
    )
    temporal = _add_temporal_deltas(source)
    temporal["roi_target_available"] = available
    temporal["roi_target_contact"] = contact
    temporal["roi_target_near"] = contact
    return _add_temporal_unit_aggregates(temporal)


def _row(
    actor: str,
    frame: int,
    cx_n: float,
    *,
    video: str = "video-a",
    unit: str | None = None,
    pig_id: str | None = None,
) -> dict[str, object]:
    width_px = 1000.0
    height_px = 500.0
    x = cx_n * width_px
    y = 0.5 * height_px
    return {
        "source_type": "golden",
        "dataset_id": "golden",
        "video_key": video,
        "scene_frame_uid": f"{video}|frame={frame}",
        "frame_uid": f"{video}|frame={frame}|actor={actor}",
        "frame_index": frame,
        "timestamp_sec": float(frame),
        "object_track_key": f"{video}|track={actor}",
        "temporal_unit_key": unit or f"{video}|unit={actor}",
        "pig_id": actor if pig_id is None else pig_id,
        "track_id": actor,
        "object_id": f"object-{actor}",
        "behavior": "stand",
        "bbox_valid": True,
        "x1": x - 10,
        "y1": y - 10,
        "x2": x + 10,
        "y2": y + 10,
        "bbox_area": 400.0,
        "image_width": width_px,
        "image_height": height_px,
        "cx_n": cx_n,
        "cy_n": 0.5,
        "bw_n": 0.02,
        "bh_n": 0.04,
        "area_n": 0.0008,
        "aspect_ratio": 1.0,
        "box_diag_n": math.hypot(0.02, 0.04),
        "observed_mask": True,
    }


def _pair_support(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    grain = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "temporal_unit_key",
    ]
    out["delta_frame_prev"] = out.groupby(grain)["frame_index"].diff()
    out["motion_delta_seconds"] = out.groupby(grain)["timestamp_sec"].diff()
    out["adjacent_motion_pair_valid"] = out["delta_frame_prev"].eq(1)
    out["motion_velocity_pair_valid"] = (
        out["delta_frame_prev"].gt(0)
        & out["motion_delta_seconds"].gt(0)
    )
    out["speed_n_per_frame"] = 0.0
    out["speed_n_per_second"] = 0.0
    return out


if __name__ == "__main__":
    main()
