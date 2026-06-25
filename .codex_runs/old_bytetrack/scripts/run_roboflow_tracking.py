#!/usr/bin/env python3
# ruff: noqa: E402
"""Run tracking using Roboflow Workflow detections."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
from tqdm import tqdm

from pig_behavior.roboflow_client import RoboflowError, detect_pigs_roboflow
from pig_behavior.tracking.association import match_and_update_tracks
from pig_behavior.tracking.config import (
    TrackingConfig,
    get_telemetry_summary,
    resolve_output_paths,
    validate_config,
    write_tracker_yaml,
)
from pig_behavior.tracking.constants import (
    DEFAULT_DET_CONF_THRESHOLD,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_REVIEW_CONF_THRESHOLD,
    DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    DEFAULT_VISUAL_OPACITY,
)
from pig_behavior.tracking.detections import Detection, adaptive_confidence_filter
from pig_behavior.tracking.exporters.annotation import write_annotation_json
from pig_behavior.tracking.exporters.coco import write_coco_annotation_json
from pig_behavior.tracking.exporters.cvat_xml import write_cvat_video_xml
from pig_behavior.tracking.exporters.labels import write_labels_json
from pig_behavior.tracking.exporters.quality import (
    build_quality_report,
    write_quality_report_csv,
    write_quality_report_json,
)
from pig_behavior.tracking.masks import load_mask
from pig_behavior.tracking.refinement import (
    apply_identity_swap_guard,
    clean_training_shapes,
    refine_shapes_temporally,
    shape_hidden_value,
)
from pig_behavior.tracking.schemas import (
    FixedTrack,
    TrackingRuntimeState,
    TrackingSummary,
)
from pig_behavior.tracking.tracks import frame_shapes, initialize_tracks
from pig_behavior.tracking.visualization import render_annotation_video

# Configure logger
logger = logging.getLogger("roboflow_tracking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tracking using Roboflow Detections.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/roboflow_tracking"), help="Output directory."
    )
    parser.add_argument("--roboflow-api-key", type=str, default=None, help="Roboflow API key.")
    parser.add_argument(
        "--workspace", type=str, default="projectdetectpigbehaviorvideoprocess", help="Workspace slug."
    )
    parser.add_argument(
        "--workflow-id", type=str, default="detect-count-and-visualize-3", help="Workflow slug."
    )
    parser.add_argument(
        "--mode", type=str, choices=["realtime", "gt_export"], default="realtime", help="Tracking mode."
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames.")
    parser.add_argument(
        "--detect-every-n-frames", type=int, default=1, help="Skip YOLO/Roboflow detection every N frames."
    )
    return parser.parse_args()


def run_roboflow_tracking(
    cfg: TrackingConfig, api_key: str | None, workspace: str, workflow_id: str
) -> TrackingSummary:
    validate_config(cfg)
    
    (
        output_video,
        annotations_json,
        coco_annotations_json,
        clean_coco_annotations_json,
        cvat_video_xml,
        labels_json,
        tracker_yaml,
        quality_report_json,
        quality_report_csv,
    ) = resolve_output_paths(cfg)
    write_tracker_yaml(tracker_yaml, cfg)
    write_labels_json(labels_json)

    capture = cv2.VideoCapture(str(cfg.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Initialize components
    mask = load_mask(cfg.mask_path, width, height, cfg)
    tracks: dict[int, FixedTrack] | None = None
    runtime = TrackingRuntimeState()
    shapes: list[dict[str, Any]] = []
    
    frame_index = -1
    frames_written = 0
    prev_frame: np.ndarray | None = None
    total_process_time = 0.0

    progress_total = cfg.max_frames or total_frames
    progress = tqdm(total=progress_total, desc="Tracking 8 pigs (Roboflow)")

    try:
        while True:
            if cfg.max_frames is not None and frames_written >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            t_start = time.perf_counter()

            is_detect_frame = frame_index % cfg.detect_every_n_frames == 0

            if is_detect_frame:
                # Query Roboflow Workflow for detections
                try:
                    res = detect_pigs_roboflow(
                        frame=frame,
                        api_key=api_key,
                        workspace_name=workspace,
                        workflow_id=workflow_id
                    )
                except RoboflowError as err:
                    logger.error("Roboflow request failed at frame %d: %s", frame_index, err)
                    # Graceful fallback: treat as skipped frame
                    is_detect_frame = False

            if is_detect_frame:
                # Map Roboflow predictions to standard Detection objects
                from pig_behavior.tracking.detections import extract_hist_hsv
                detections = []
                for pred in res.get("predictions", []):
                    box = pred["box"]
                    detections.append(Detection(
                        box=box,
                        score=pred["score"],
                        raw_id=None,
                        class_id=pred["class_id"],
                        hist=extract_hist_hsv(frame, box),
                        mask=None
                    ))
                
                # Apply confidence and masking filters
                detections = adaptive_confidence_filter(detections, cfg)

                if tracks is None:
                    tracks = initialize_tracks(detections, mask, width, height, cfg)
                else:
                    match_and_update_tracks(
                        tracks,
                        detections,
                        frame,
                        prev_frame,
                        cfg,
                        runtime,
                    )
            else:
                # Update tracks using prediction/motion model on skipped frames
                if tracks is not None:
                    from pig_behavior.tracking.tracks import lk_predict_box
                    for track in tracks.values():
                        lk_box = lk_predict_box(prev_frame, frame, track.last_box, width, height)
                        if lk_box is None:
                            lk_box = track.predicted_box(width, height)
                        track.update_predicted(lk_box, width, height, cfg=cfg)

            current_shapes = frame_shapes(tracks, frame_index, cfg)
            shapes.extend(current_shapes)
            frames_written += 1
            prev_frame = frame.copy()

            t_end = time.perf_counter()
            frame_time = t_end - t_start
            total_process_time += frame_time

            # Log frame stats on progress bar
            if tracks is not None:
                v_count = sum(1 for trk in tracks.values() if trk.get_state() == "VISIBLE")
                o_count = sum(1 for trk in tracks.values() if trk.get_state() == "OCCLUDED")
                progress.set_postfix(
                    lat=f"{frame_time * 1000.0:.1f}ms",
                    vis=v_count,
                    occ=o_count,
                )

            progress.update(1)
    finally:
        capture.release()
        progress.close()
        if frames_written > 0:
            avg_fps = frames_written / max(total_process_time, 1e-6)
            logger.info("Tracking finished. Total Frames: %d, Average FPS: %.2f", frames_written, avg_fps)

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    # Apply offline refinements if gt_export is enabled
    if cfg.enable_offline_smoothing:
        shapes = apply_identity_swap_guard(shapes, width, height, cfg)
        shapes = refine_shapes_temporally(shapes, width, height, cfg)
    
    hidden_shape_count = sum(
        1 for shape in shapes if shape_hidden_value(shape) == "Yes"
    )
    review_shape_count = sum(
        1 for shape in shapes if (
            shape.get("_needs_review")
            or shape.get("_refined")
            or shape.get("_occlusion_hold")
            or shape.get("_identity_swap_guard")
        )
    )

    # Render visualized track video
    render_annotation_video(
        cfg.video_path,
        output_video,
        shapes,
        cfg,
        frame_limit=frames_written,
    )
    
    # Export formats
    write_annotation_json(annotations_json, shapes)
    write_coco_annotation_json(
        coco_annotations_json,
        shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior
    )
    clean_shapes = clean_training_shapes(shapes, cfg)
    write_coco_annotation_json(
        clean_coco_annotations_json,
        clean_shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior
    )
    
    source_frame_count = max(total_frames, frame_index + 1)
    write_cvat_video_xml(
        cvat_video_xml,
        shapes,
        cfg.video_path,
        width,
        height,
        source_frame_count
    )
    
    quality_report = build_quality_report(
        shapes,
        cfg,
        cfg.video_path,
        source_fps,
        source_frame_count,
        get_telemetry_summary(runtime)
    )
    write_quality_report_json(quality_report_json, quality_report)
    write_quality_report_csv(quality_report_csv, quality_report)

    return TrackingSummary(
        output_video=output_video,
        annotations_json=annotations_json,
        coco_annotations_json=coco_annotations_json,
        clean_coco_annotations_json=clean_coco_annotations_json,
        cvat_video_xml=cvat_video_xml,
        labels_json=labels_json,
        quality_report_json=quality_report_json,
        quality_report_csv=quality_report_csv,
        frames_read=frames_written,
        frames_written=frames_written,
        shape_count=len(shapes),
        hidden_shape_count=hidden_shape_count,
        review_shape_count=review_shape_count,
        start_frame=cfg.start_frame,
        source_fps=source_fps,
        output_fps=cfg.output_fps,
        telemetry=get_telemetry_summary(runtime),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    args = parse_args()
    
    # Set default config matching options
    cfg = TrackingConfig(
        video_path=args.video,
        weights_path=Path("models/detector/pig_detector_yolov8.pt"),  # dummy path for constructor
        mask_path=Path("data/annotations/scene/mask.png"),
        output_dir=args.output_dir,
        mode=args.mode,
        max_frames=args.max_frames,
        detect_every_n_frames=args.detect_every_n_frames,
        det_conf=DEFAULT_DET_CONF_THRESHOLD,
        track_high_conf=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
        review_conf=DEFAULT_REVIEW_CONF_THRESHOLD,
        nms_iou=DEFAULT_NMS_IOU_THRESHOLD,
        visual_opacity=DEFAULT_VISUAL_OPACITY,
    )
    
    run_roboflow_tracking(
        cfg=cfg,
        api_key=args.roboflow_api_key,
        workspace=args.workspace,
        workflow_id=args.workflow_id
    )


if __name__ == "__main__":
    main()
