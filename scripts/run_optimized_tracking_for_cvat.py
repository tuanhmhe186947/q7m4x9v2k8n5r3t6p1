#!/usr/bin/env python3
# ruff: noqa
"""Run optimized pig tracking with extended linear interpolation and sparse keyframe CVAT XML export."""

# ruff: noqa

import sys
from pathlib import Path

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import datetime as dt
from typing import Any
from xml.etree import ElementTree as ET
from xml.dom import minidom

# Import existing modules
import pig_behavior.tracking.refinement as refinement
import pig_behavior.tracking.exporters.cvat_xml as cvat_xml
import pig_behavior.tracking.runner as runner
from pig_behavior.tracking.constants import ID_VALUES, PIG_LABEL_SCHEMA
from pig_behavior.tracking.refinement import (
    _shape_attributes_dict,
    shape_box,
    set_shape_box,
    shape_hidden_value,
    shape_is_stable_anchor,
    interpolate_box,
    size_jump_ratio,
    refine_original_weight,
)
from pig_behavior.tracking.exporters.cvat_xml import _xml_child, _append_cvat_xml_label
from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.cli import parse_args, _profile_from_args, _tracking_config_from_args, print_tracking_summary, display_tracked_video, _video_paths_from_args


PURE_INTERPOLATION = True


def custom_refine_shapes_temporally(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Enhanced refinement that enforces 100% linear interpolation on hidden/predicted/unstable gaps."""
    if not cfg.refine_boxes:
        return shapes

    refined_shapes = [shape.copy() for shape in shapes]
    for shape in refined_shapes:
        shape["_refined"] = False
        shape["_refine_reason"] = ""

    # Allow custom max gap (e.g. from cfg or default to 90 frames)
    max_gap = getattr(cfg, "refine_max_gap_frames", 90)
    pure_interpolate = PURE_INTERPOLATION

    for fixed_id in range(1, cfg.expected_pigs + 1):
        track_shapes = sorted(
            [
                shape
                for shape in refined_shapes
                if str(shape["label"]) == f"Pig_{fixed_id}"
            ],
            key=lambda item: int(item["frame"]),
        )
        stable_indices = [
            idx
            for idx, shape in enumerate(track_shapes)
            if shape_is_stable_anchor(shape, cfg)
        ]
        if not stable_indices:
            continue

        for idx, shape in enumerate(track_shapes):
            frame = int(shape["frame"])
            
            # Find nearby anchors with the custom max_gap
            previous_idx = None
            next_idx = None
            for s_idx in reversed(stable_indices):
                if s_idx >= idx:
                    continue
                if frame - int(track_shapes[s_idx]["frame"]) <= max_gap:
                    previous_idx = s_idx
                break
            for s_idx in stable_indices:
                if s_idx <= idx:
                    continue
                if int(track_shapes[s_idx]["frame"]) - frame <= max_gap:
                    next_idx = s_idx
                break

            if previous_idx is None and next_idx is None:
                continue

            original = shape_box(shape)
            expected = None
            if previous_idx is not None and next_idx is not None:
                expected = interpolate_box(
                    track_shapes[previous_idx],
                    track_shapes[next_idx],
                    frame,
                )
            elif not shape_is_stable_anchor(shape, cfg):
                anchor_idx = previous_idx if previous_idx is not None else next_idx
                if anchor_idx is not None:
                    expected = shape_box(track_shapes[anchor_idx])
            
            if expected is None:
                continue

            source = str(shape.get("_track_source", "unknown"))
            unstable_detection = (
                source != "detected"
                or shape_hidden_value(shape) == "Yes"
                or float(shape.get("score", 0.0)) < cfg.review_conf
            )
            size_jump = size_jump_ratio(original, expected)
            size_outlier = size_jump > cfg.refine_size_jump_threshold
            if not unstable_detection and not size_outlier:
                continue

            # Override original weight to 0.0 (100% interpolation) for predicted or hidden frames
            if unstable_detection and pure_interpolate:
                original_weight = 0.0
                reason = f"interpolated_{source}"
            else:
                original_weight = refine_original_weight(shape, cfg)
                if size_outlier:
                    original_weight = min(original_weight, 0.35)
                reason = f"size_jump>{cfg.refine_size_jump_threshold:.2f}"

            refined = original_weight * original + (1.0 - original_weight) * expected
            shape["_original_points"] = [round(float(value), 2) for value in original]
            shape["_refined"] = True
            shape["_refine_reason"] = reason
            shape["_refine_size_jump"] = round(float(size_jump), 4)
            set_shape_box(shape, refined, width, height)

    return refined_shapes


def simplify_track_keyframes(
    track_shapes: list[dict[str, Any]],
    error_threshold: float = 1.5,
) -> set[int]:
    """Identify sparse keyframes using linear interpolation error thresholding (Douglas-Peucker variant)."""
    n = len(track_shapes)
    if n <= 2:
        return set(range(n))

    # Always mark frame 0, frame n-1, and any frame where state/attributes change as keyframes
    must_keyframes = {0, n - 1}
    for idx in range(1, n):
        prev_attr = _shape_attributes_dict(track_shapes[idx - 1])
        curr_attr = _shape_attributes_dict(track_shapes[idx])
        
        # State transitions trigger keyframes (Hidden / Behavior / ID)
        if (
            prev_attr.get("Hidden") != curr_attr.get("Hidden")
            or prev_attr.get("Behavior") != curr_attr.get("Behavior")
        ):
            must_keyframes.add(idx - 1)
            must_keyframes.add(idx)

    must_keyframes_list = sorted(list(must_keyframes))
    final_keyframes = set(must_keyframes_list)

    # Recursive check for deviations
    def simplify_segment(start_idx: int, end_idx: int):
        if end_idx - start_idx <= 1:
            return
        
        box_start = shape_box(track_shapes[start_idx])
        box_end = shape_box(track_shapes[end_idx])
        frame_start = int(track_shapes[start_idx]["frame"])
        frame_end = int(track_shapes[end_idx]["frame"])
        
        max_err = 0.0
        max_idx = -1
        
        for idx in range(start_idx + 1, end_idx):
            frame = int(track_shapes[idx]["frame"])
            actual_box = shape_box(track_shapes[idx])
            
            # Interpolate
            ratio = (frame - frame_start) / float(frame_end - frame_start)
            interp_box = (1.0 - ratio) * box_start + ratio * box_end
            
            # L-infinity norm error (max coordinate absolute difference)
            err = np.max(np.abs(actual_box - interp_box))
            if err > max_err:
                max_err = err
                max_idx = idx
                
        if max_err > error_threshold:
            final_keyframes.add(max_idx)
            # Recursively check subsets
            simplify_segment(start_idx, max_idx)
            simplify_segment(max_idx, end_idx)

    # Simplify all sub-intervals
    for i in range(len(must_keyframes_list) - 1):
        simplify_segment(must_keyframes_list[i], must_keyframes_list[i + 1])
        
    return final_keyframes


def custom_write_cvat_video_xml(
    path: Path,
    shapes: list[dict[str, Any]],
    video_path: Path,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> None:
    """Write native CVAT for video 1.1 XML using optimized sparse keyframes."""
    # We read error threshold from a global/config level or default to 1.5 pixels
    error_threshold = getattr(custom_write_cvat_video_xml, "error_threshold", 1.5)
    
    root = ET.Element("annotations")
    _xml_child(root, "version", "1.1")

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    _xml_child(task, "id", 0)
    _xml_child(task, "name", video_path.stem)
    _xml_child(task, "size", frame_count)
    _xml_child(task, "mode", "interpolation")
    _xml_child(task, "overlap", 0)
    _xml_child(task, "bugtracker", "")
    _xml_child(task, "flipped", "False")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _xml_child(task, "created", now)
    _xml_child(task, "updated", now)

    labels_el = ET.SubElement(task, "labels")
    for label in PIG_LABEL_SCHEMA:
        _append_cvat_xml_label(labels_el, label)

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    _xml_child(segment, "id", 0)
    _xml_child(segment, "start", 0)
    _xml_child(segment, "stop", max(0, frame_count - 1))
    _xml_child(segment, "url", "")

    owner = ET.SubElement(task, "owner")
    _xml_child(owner, "username", "auto")
    _xml_child(owner, "email", "")

    original_size = ET.SubElement(task, "original_size")
    _xml_child(original_size, "width", int(frame_width))
    _xml_child(original_size, "height", int(frame_height))
    _xml_child(meta, "dumped", now)

    shapes_by_track: dict[int, list[dict[str, Any]]] = {
        fixed_id: [] for fixed_id in range(1, len(ID_VALUES) + 1)
    }
    for shape in shapes:
        fixed_id = int(str(shape["label"]).removeprefix("Pig_"))
        shapes_by_track[fixed_id].append(shape)

    total_exported_keyframes = 0
    total_raw_frames = len(shapes)

    for fixed_id in range(1, len(ID_VALUES) + 1):
        track = ET.SubElement(
            root,
            "track",
            {
                "id": str(fixed_id),
                "label": f"Pig_{fixed_id}",
                "source": "auto",
            },
        )
        
        # Sort shapes by frame
        track_shapes_sorted = sorted(shapes_by_track[fixed_id], key=lambda item: item["frame"])
        if not track_shapes_sorted:
            continue
            
        # Get sparse keyframe indices
        keyframe_indices = simplify_track_keyframes(track_shapes_sorted, error_threshold)
        total_exported_keyframes += len(keyframe_indices)
        
        for idx, shape in enumerate(track_shapes_sorted):
            # Only export frames that are selected as keyframes
            if idx not in keyframe_indices:
                continue
                
            x1, y1, x2, y2 = [float(value) for value in shape["points"]]
            attributes = _shape_attributes_dict(shape)
            hidden = str(attributes.get("Hidden", "No"))
            
            box = ET.SubElement(
                track,
                "box",
                {
                    "frame": str(int(shape["frame"])),
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "outside": "0",
                    "occluded": "1" if hidden == "Yes" else "0",
                    "keyframe": "1",  # Export as keyframe for CVAT
                },
            )
            for name in ("ID", "Behavior", "Hidden"):
                _xml_child(box, "attribute", attributes.get(name, "")).set(
                    "name",
                    name,
                )

    print(f"[Custom CVAT XML Export] Optimized: total keyframes = {total_exported_keyframes} (out of {total_raw_frames} raw boxes across all tracks)")
    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(
        indent="  ",
        encoding="utf-8",
    )
    path.write_bytes(pretty_xml)


# Monkeypatch references in the packages
refinement.refine_shapes_temporally = custom_refine_shapes_temporally
runner.refine_shapes_temporally = custom_refine_shapes_temporally

cvat_xml.write_cvat_video_xml = custom_write_cvat_video_xml
runner.write_cvat_video_xml = custom_write_cvat_video_xml


# Custom tracker configuration helper (supporting ByteTrack and BoT-SORT)
import pig_behavior.tracking.config as tracking_config

TRACKER_TYPE = "bytetrack"

def custom_write_tracker_yaml(path: Path, cfg: TrackingConfig) -> None:
    """Write tracker config tuned for pig videos (supports bytetrack and botsort)."""
    track_low_thresh = min(cfg.det_conf, cfg.track_high_conf)
    global TRACKER_TYPE
    
    if TRACKER_TYPE == "botsort":
        path.write_text(
            "\n".join(
                [
                    "tracker_type: botsort",
                    "model: auto",  # Fixes AttributeError: 'IterableSimpleNamespace' object has no attribute 'model'
                    f"track_high_thresh: {cfg.track_high_conf:.2f}",
                    f"track_low_thresh: {track_low_thresh:.2f}",
                    f"new_track_thresh: {cfg.track_high_conf:.2f}",
                    f"track_thresh: {cfg.track_high_conf:.2f}",
                    f"match_thresh: {cfg.iou:.2f}",
                    "track_buffer: 90",
                    "min_box_area: 10",
                    "mot20: false",
                    "distance_metric: iou",
                    "match_metric: iou",
                    "gmc_method: none",
                    "with_reid: false",  # Turn off ReID to avoid downloading model weights
                    "proximity_thresh: 0.5",
                    "appearance_thresh: 0.25",
                    "fuse_score: true",
                    "max_age: 90",
                    "n_init: 3",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
    else:
        path.write_text(
            "\n".join(
                [
                    "tracker_type: bytetrack",
                    f"track_high_thresh: {cfg.track_high_conf:.2f}",
                    f"track_low_thresh: {track_low_thresh:.2f}",
                    f"new_track_thresh: {cfg.track_high_conf:.2f}",
                    f"track_thresh: {cfg.track_high_conf:.2f}",
                    f"match_thresh: {cfg.iou:.2f}",
                    "track_buffer: 90",
                    "min_box_area: 10",
                    "mot20: false",
                    "fuse_score: true",
                    "proximity_thresh: 0.5",
                    "appearance_thresh: 0.25",
                    "max_age: 90",
                    "n_init: 3",
                    "with_reid: true",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )

tracking_config.write_tracker_yaml = custom_write_tracker_yaml
runner.write_tracker_yaml = custom_write_tracker_yaml


def run_main():
    """Wrapper main that parses cli arguments and sets custom config attributes before running main."""
    # We parse standard arguments plus custom options
    import argparse
    parser = argparse.ArgumentParser(description="Run optimized pig tracking with extended linear interpolation and sparse keyframes.")
    parser.add_argument("--keyframe-error-threshold", type=float, default=1.5, help="Max pixel deviation error allowed before creating a keyframe in CVAT XML.")
    parser.add_argument("--no-pure-interpolation", action="store_true", help="Do not force 100%% linear interpolation for unstable/hidden tracking boxes.")
    parser.add_argument("--tracker-type", choices=["bytetrack", "botsort"], default="bytetrack", help="Front-end tracker type (bytetrack or botsort).")
    
    # We parse known args for custom parameters, then forward the rest to the standard main
    args, standard_args = parser.parse_known_args()
    
    # Set the error threshold for XML exporter
    custom_write_cvat_video_xml.error_threshold = args.keyframe_error_threshold
    
    # Set the tracker type
    global TRACKER_TYPE
    TRACKER_TYPE = args.tracker_type
    
    # Parse standard arguments using cli
    cli_args = parse_args(standard_args)
    profile = _profile_from_args(cli_args)
    
    # Run the tracking for each video
    for video_path in _video_paths_from_args(cli_args, profile):
        cfg = _tracking_config_from_args(cli_args, profile, video_path)
        
        # Override config parameters
        cfg.refine_max_gap_frames = cli_args.refine_max_gap  # Ensure cli custom max gap is used
        global PURE_INTERPOLATION
        PURE_INTERPOLATION = not args.no_pure_interpolation
        
        print(f"\n==================================================")
        print(f"Running Optimized Tracking on: {cfg.video_path.name}")
        print(f"Keyframe Error Threshold: {args.keyframe_error_threshold} px")
        print(f"Pure Interpolation on Occlusion: {not args.no_pure_interpolation}")
        print(f"Front-end Tracker Type: {args.tracker_type}")
        print(f"==================================================")
        
        summary = runner.run_tracking(cfg)
        print_tracking_summary(cfg, summary)
        if cli_args.display_inline:
            display_tracked_video(summary.output_video)


if __name__ == "__main__":
    run_main()
