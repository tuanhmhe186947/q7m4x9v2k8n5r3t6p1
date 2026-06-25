"""Quality report export for fixed-ID tracking annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.tracking.config import (
    TrackingConfig,
    tracking_rule_flags_enabled,
)
from pig_behavior.tracking.constants import TRACKING_TELEMETRY_KEYS
from pig_behavior.tracking.refinement import (
    _shape_attributes_dict,
    clean_training_shapes,
)


def _shape_attribute_value(
    shape: dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    return _shape_attributes_dict(shape).get(name, default)


def build_quality_report(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
    video_path: Path,
    source_fps: float,
    source_frame_count: int,
    telemetry: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize frames/tracks that need manual review."""
    frames = sorted({int(shape["frame"]) for shape in shapes})
    shapes_by_frame = {
        frame: [shape for shape in shapes if int(shape["frame"]) == frame]
        for frame in frames
    }
    frame_rows: list[dict[str, Any]] = []
    issue_frames: list[int] = []

    for frame in frames:
        frame_shapes = shapes_by_frame[frame]
        hidden_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
        ]
        predicted_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_track_source") != "detected"
        ]
        refined_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_refined")
        ]
        ambiguous_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_ambiguous_occlusion")
        ]
        hold_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_occlusion_hold")
        ]
        area_occluded_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_area_occluded")
        ]
        merged_split_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_merged_box_split")
        ]
        identity_swap_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_identity_swap_guard")
        ]
        moving_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state") == "moving"
        ]
        stationary_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state") == "stationary"
        ]
        unknown_motion_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state", "unknown") == "unknown"
        ]
        low_score_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if float(shape.get("score", 0.0)) < cfg.review_conf
        ]
        review_ids = sorted(
            set(
                hidden_ids
                + predicted_ids
                + low_score_ids
                + refined_ids
                + ambiguous_ids
                + hold_ids
                + area_occluded_ids
                + merged_split_ids
                + identity_swap_ids
            )
        )
        detected_count = sum(
            1 for shape in frame_shapes if shape.get("_track_source") == "detected"
        )
        min_score = min(
            (float(shape.get("score", 0.0)) for shape in frame_shapes),
            default=0.0,
        )
        row = {
            "frame": frame,
            "time_sec": round(frame / max(source_fps, 1e-6), 3),
            "shape_count": len(frame_shapes),
            "detected_count": detected_count,
            "predicted_count": len(predicted_ids),
            "refined_count": len(refined_ids),
            "ambiguous_occlusion_count": len(ambiguous_ids),
            "occlusion_hold_count": len(hold_ids),
            "area_occluded_count": len(area_occluded_ids),
            "merged_box_split_count": len(merged_split_ids),
            "identity_swap_guard_count": len(identity_swap_ids),
            "hidden_count": len(hidden_ids),
            "low_score_count": len(low_score_ids),
            "min_score": round(min_score, 4),
            "hidden_ids": hidden_ids,
            "predicted_ids": predicted_ids,
            "refined_ids": refined_ids,
            "ambiguous_occlusion_ids": ambiguous_ids,
            "occlusion_hold_ids": hold_ids,
            "area_occluded_ids": area_occluded_ids,
            "merged_box_split_ids": merged_split_ids,
            "identity_swap_guard_ids": identity_swap_ids,
            "moving_ids": moving_ids,
            "stationary_ids": stationary_ids,
            "unknown_motion_ids": unknown_motion_ids,
            "low_score_ids": low_score_ids,
            "review_ids": review_ids,
            "needs_review": bool(review_ids or len(frame_shapes) != cfg.expected_pigs),
        }
        if row["needs_review"]:
            issue_frames.append(frame)
        frame_rows.append(row)

    track_rows: list[dict[str, Any]] = []
    for fixed_id in range(1, cfg.expected_pigs + 1):
        track_shapes = [
            shape
            for shape in shapes
            if str(shape["label"]) == f"Pig_{fixed_id}"
        ]
        scores = [float(shape.get("score", 0.0)) for shape in track_shapes]
        detected_frames = sum(
            1 for shape in track_shapes if shape.get("_track_source") == "detected"
        )
        predicted_frames = len(track_shapes) - detected_frames
        hidden_frames = sum(
            1
            for shape in track_shapes
            if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
        )
        refined_frames = sum(1 for shape in track_shapes if shape.get("_refined"))
        ambiguous_frames = sum(
            1 for shape in track_shapes if shape.get("_ambiguous_occlusion")
        )
        hold_frames = sum(1 for shape in track_shapes if shape.get("_occlusion_hold"))
        area_occluded_frames = sum(
            1 for shape in track_shapes if shape.get("_area_occluded")
        )
        merged_split_frames = sum(
            1 for shape in track_shapes if shape.get("_merged_box_split")
        )
        identity_swap_frames = sum(
            1 for shape in track_shapes if shape.get("_identity_swap_guard")
        )
        moving_frames = sum(
            1 for shape in track_shapes if shape.get("_motion_state") == "moving"
        )
        stationary_frames = sum(
            1 for shape in track_shapes if shape.get("_motion_state") == "stationary"
        )
        unknown_motion_frames = sum(
            1
            for shape in track_shapes
            if shape.get("_motion_state", "unknown") == "unknown"
        )
        review_frames = sum(
            1
            for shape in track_shapes
            if (
                shape.get("_needs_review")
                or shape.get("_refined")
                or shape.get("_ambiguous_occlusion")
                or shape.get("_occlusion_hold")
                or shape.get("_area_occluded")
                or shape.get("_merged_box_split")
                or shape.get("_identity_swap_guard")
            )
        )
        track_rows.append(
            {
                "fixed_id": fixed_id,
                "label": f"Pig_{fixed_id}",
                "id_attribute": f"ID_{fixed_id}",
                "frames": len(track_shapes),
                "detected_frames": detected_frames,
                "predicted_frames": predicted_frames,
                "refined_frames": refined_frames,
                "ambiguous_occlusion_frames": ambiguous_frames,
                "occlusion_hold_frames": hold_frames,
                "area_occluded_frames": area_occluded_frames,
                "merged_box_split_frames": merged_split_frames,
                "identity_swap_guard_frames": identity_swap_frames,
                "moving_frames": moving_frames,
                "stationary_frames": stationary_frames,
                "unknown_motion_frames": unknown_motion_frames,
                "hidden_frames": hidden_frames,
                "review_frames": review_frames,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            }
        )

    clean_shape_count = len(clean_training_shapes(shapes, cfg))
    report = {
        "video": str(video_path),
        "video_name": video_path.name,
        "source_fps": round(float(source_fps), 4),
        "source_frame_count": int(source_frame_count),
        "start_frame": int(cfg.start_frame),
        "processed_frames": len(frames),
        "expected_pigs": int(cfg.expected_pigs),
        "thresholds": {
            "det_conf": cfg.det_conf,
            "track_high_conf": cfg.track_high_conf,
            "review_conf": cfg.review_conf,
            "nms_iou": cfg.nms_iou,
            "track_match_iou": cfg.track_match_iou,
            "use_mask_iou": cfg.use_mask_iou,
            "mask_iou_max_missed": cfg.mask_iou_max_missed,
            "mask_iou_min_area": cfg.mask_iou_min_area,
            "match_cost_threshold": cfg.match_cost_threshold,
            "unseen_track_cost_threshold": cfg.unseen_track_cost_threshold,
            "lost_track_cost_threshold": cfg.lost_track_cost_threshold,
            "lost_track_reid_appearance_threshold": (
                cfg.lost_track_reid_appearance_threshold
            ),
            "initial_track_conf": cfg.initial_track_conf,
            "motion_gate_confidence": cfg.motion_gate_confidence,
            "low_conf_motion_gate": cfg.low_conf_motion_gate,
            "low_conf_max_center_jump": cfg.low_conf_max_center_jump,
            "low_conf_max_box_jump_scale": cfg.low_conf_max_box_jump_scale,
            "low_conf_min_iou": cfg.low_conf_min_iou,
            "occlusion_aware_matching": cfg.occlusion_aware_matching,
            "occlusion_track_iom_threshold": cfg.occlusion_track_iom_threshold,
            "occlusion_detection_iom_threshold": (
                cfg.occlusion_detection_iom_threshold
            ),
            "occlusion_stationary_speed": cfg.occlusion_stationary_speed,
            "occlusion_stationary_max_center_jump": (
                cfg.occlusion_stationary_max_center_jump
            ),
            "occlusion_switch_penalty": cfg.occlusion_switch_penalty,
            "occlusion_competitor_margin": cfg.occlusion_competitor_margin,
            "occlusion_appearance_penalty": cfg.occlusion_appearance_penalty,
            "occlusion_appearance_margin": cfg.occlusion_appearance_margin,
            "occlusion_stationary_lock": cfg.occlusion_stationary_lock,
            "freeze_identity_in_occlusion": cfg.freeze_identity_in_occlusion,
            "hold_occluded_box": cfg.hold_occluded_box,
            "occlusion_hold_max_frames": cfg.occlusion_hold_max_frames,
            "occlusion_hold_hidden_frames": cfg.occlusion_hold_hidden_frames,
            "USE_IOU_FALLBACK": cfg.USE_IOU_FALLBACK,
            "USE_AREA_OCCLUSION_FREEZE": cfg.USE_AREA_OCCLUSION_FREEZE,
            "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": (
                cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE
            ),
            "USE_MERGED_BOX_SPLIT": cfg.USE_MERGED_BOX_SPLIT,
            "iou_fallback_threshold": cfg.iou_fallback_threshold,
            "area_occlusion_shrink_ratio": cfg.area_occlusion_shrink_ratio,
            "area_occlusion_freeze_frames": cfg.area_occlusion_freeze_frames,
            "merged_box_growth_ratio": cfg.merged_box_growth_ratio,
            "merged_box_neighbor_distance": cfg.merged_box_neighbor_distance,
            "merged_box_split_max_tracks": cfg.merged_box_split_max_tracks,
            "hard_occlusion_track_iom_threshold": (
                cfg.hard_occlusion_track_iom_threshold
            ),
            "hard_occlusion_detection_iom_threshold": (
                cfg.hard_occlusion_detection_iom_threshold
            ),
            "hard_occlusion_min_frames": cfg.hard_occlusion_min_frames,
            "hard_occlusion_recovery_frames": cfg.hard_occlusion_recovery_frames,
            "hard_occlusion_score_threshold": cfg.hard_occlusion_score_threshold,
            "identity_swap_guard": cfg.identity_swap_guard,
            "identity_swap_min_gain": cfg.identity_swap_min_gain,
            "identity_swap_iom_threshold": cfg.identity_swap_iom_threshold,
            "hidden_motion_model": cfg.hidden_motion_model,
            "hidden_velocity_alpha": cfg.hidden_velocity_alpha,
            "hidden_acceleration_alpha": cfg.hidden_acceleration_alpha,
            "hidden_stationary_speed": cfg.hidden_stationary_speed,
            "hidden_motion_history": cfg.hidden_motion_history,
            "hidden_min_motion_history": cfg.hidden_min_motion_history,
            "hidden_stationary_displacement": cfg.hidden_stationary_displacement,
            "hidden_moving_displacement": cfg.hidden_moving_displacement,
            "hidden_motion_consistency": cfg.hidden_motion_consistency,
            "hidden_stationary_lock_frames": cfg.hidden_stationary_lock_frames,
            "hidden_max_motion_step_box_scale": cfg.hidden_max_motion_step_box_scale,
        },
        "summary": {
            "total_shapes": len(shapes),
            "clean_training_shapes": clean_shape_count,
            "review_shapes": sum(
                1
                for shape in shapes
                if shape.get("_needs_review") or shape.get("_refined")
            ),
            "refined_shapes": sum(1 for shape in shapes if shape.get("_refined")),
            "ambiguous_occlusion_shapes": sum(
                1 for shape in shapes if shape.get("_ambiguous_occlusion")
            ),
            "occlusion_hold_shapes": sum(
                1 for shape in shapes if shape.get("_occlusion_hold")
            ),
            "area_occluded_shapes": sum(
                1 for shape in shapes if shape.get("_area_occluded")
            ),
            "merged_box_split_shapes": sum(
                1 for shape in shapes if shape.get("_merged_box_split")
            ),
            "identity_swap_guard_shapes": sum(
                1 for shape in shapes if shape.get("_identity_swap_guard")
            ),
            "hidden_shapes": sum(
                1
                for shape in shapes
                if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
            ),
            "issue_frame_count": len(issue_frames),
            "issue_frames": issue_frames,
        },
        "telemetry": (
            {key: int((telemetry or {}).get(key, 0)) for key in TRACKING_TELEMETRY_KEYS}
        ),
        "frames": frame_rows,
        "tracks": track_rows,
    }
    if not tracking_rule_flags_enabled(cfg):
        rule_frame_keys = (
            "area_occluded_count",
            "merged_box_split_count",
            "area_occluded_ids",
            "merged_box_split_ids",
        )
        rule_track_keys = (
            "area_occluded_frames",
            "merged_box_split_frames",
        )
        rule_threshold_keys = (
            "USE_IOU_FALLBACK",
            "USE_AREA_OCCLUSION_FREEZE",
            "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE",
            "USE_MERGED_BOX_SPLIT",
            "iou_fallback_threshold",
            "area_occlusion_shrink_ratio",
            "area_occlusion_freeze_frames",
            "merged_box_growth_ratio",
            "merged_box_neighbor_distance",
            "merged_box_split_max_tracks",
            "hard_occlusion_track_iom_threshold",
            "hard_occlusion_detection_iom_threshold",
            "hard_occlusion_min_frames",
            "hard_occlusion_recovery_frames",
            "hard_occlusion_score_threshold",
        )
        for row in report["frames"]:
            for key in rule_frame_keys:
                row.pop(key, None)
        for row in report["tracks"]:
            for key in rule_track_keys:
                row.pop(key, None)
        for key in rule_threshold_keys:
            report["thresholds"].pop(key, None)
        report["summary"].pop("area_occluded_shapes", None)
        report["summary"].pop("merged_box_split_shapes", None)
        report.pop("telemetry", None)
    return report


def write_quality_report_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def write_quality_report_csv(path: Path, report: dict[str, Any]) -> None:
    include_rule_fields = "area_occluded_shapes" in report.get("summary", {})
    fieldnames = [
        "frame",
        "time_sec",
        "shape_count",
        "detected_count",
        "predicted_count",
        "refined_count",
        "ambiguous_occlusion_count",
        "occlusion_hold_count",
        "identity_swap_guard_count",
        "hidden_count",
        "low_score_count",
        "min_score",
        "hidden_ids",
        "predicted_ids",
        "refined_ids",
        "ambiguous_occlusion_ids",
        "occlusion_hold_ids",
        "identity_swap_guard_ids",
        "moving_ids",
        "stationary_ids",
        "unknown_motion_ids",
        "low_score_ids",
        "review_ids",
        "needs_review",
    ]
    if include_rule_fields:
        count_index = fieldnames.index("identity_swap_guard_count")
        id_index = fieldnames.index("identity_swap_guard_ids")
        fieldnames[count_index:count_index] = [
            "area_occluded_count",
            "merged_box_split_count",
        ]
        fieldnames[id_index + 2 : id_index + 2] = [
            "area_occluded_ids",
            "merged_box_split_ids",
        ]
    list_fields = [
        "hidden_ids",
        "predicted_ids",
        "refined_ids",
        "ambiguous_occlusion_ids",
        "occlusion_hold_ids",
        "identity_swap_guard_ids",
        "moving_ids",
        "stationary_ids",
        "unknown_motion_ids",
        "low_score_ids",
        "review_ids",
    ]
    if include_rule_fields:
        insert_at = list_fields.index("identity_swap_guard_ids")
        list_fields[insert_at:insert_at] = [
            "area_occluded_ids",
            "merged_box_split_ids",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["frames"]:
            serialized = row.copy()
            for key in list_fields:
                serialized[key] = " ".join(str(value) for value in row[key])
            writer.writerow(serialized)


__all__ = [
    "_shape_attribute_value",
    "build_quality_report",
    "write_quality_report_csv",
    "write_quality_report_json",
]
