"""Top-level tracking runner for fixed-ID pig annotation export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.tracking.association import match_and_update_tracks
from pig_behavior.tracking.config import (
    TrackingConfig,
    get_telemetry_summary,
    resolve_output_paths,
    validate_config,
    write_tracker_yaml,
)
from pig_behavior.tracking.detections import (
    adaptive_confidence_filter,
    parse_detections,
)
from pig_behavior.tracking.exporters.annotation import write_annotation_json
from pig_behavior.tracking.exporters.coco import write_coco_annotation_json
from pig_behavior.tracking.exporters.cvat_xml import write_cvat_video_xml
from pig_behavior.tracking.exporters.labels import write_labels_json
from pig_behavior.tracking.exporters.quality import (
    build_quality_report,
    write_quality_report_csv,
    write_quality_report_json,
)
from pig_behavior.tracking.masks import apply_mask_to_frame, load_mask
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
from pig_behavior.tracking.visualization import (
    draw_tracks,
    render_annotation_video,
)


def run_tracking(cfg: TrackingConfig) -> TrackingSummary:
    """Run YOLOv8 + mask + stabilized eight-ID tracking."""
    validate_config(cfg)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Install tracking dependencies first: pip install -e .[tracking]"
        ) from exc

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

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or cfg.output_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video frame size.")
    if total_frames and cfg.start_frame >= total_frames:
        raise ValueError(
            f"start_frame={cfg.start_frame} is outside video with "
            f"{total_frames} frames."
        )
    if cfg.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    mask = load_mask(cfg.mask_path, width, height, cfg)
    model = YOLO(str(cfg.weights_path))
    tracks: dict[int, FixedTrack] | None = None
    runtime = TrackingRuntimeState()
    shapes: list[dict[str, Any]] = []
    hidden_shape_count = 0
    review_shape_count = 0
    frame_index = cfg.start_frame - 1
    frames_written = 0
    prev_frame: np.ndarray | None = None
    show_enabled = cfg.show

    try:
        from tqdm import tqdm

        remaining_frames = (
            max(0, total_frames - cfg.start_frame) if total_frames else None
        )
        progress_total = cfg.max_frames
        if progress_total is None:
            progress_total = remaining_frames
        elif remaining_frames is not None:
            progress_total = min(progress_total, remaining_frames)
        progress = tqdm(
            total=progress_total,
            desc="Tracking 8 pigs",
        )
    except ImportError:
        progress = None

    try:
        while True:
            if cfg.max_frames is not None and frames_written >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            frame_h, frame_w = frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(cfg.mask_path, width, height, cfg)

            detector_frame = (
                apply_mask_to_frame(frame, mask)
                if cfg.mask_input_frame and mask is not None
                else frame
            )
            results = model.track(
                source=detector_frame,
                persist=True,
                conf=cfg.det_conf,
                iou=cfg.iou,
                tracker=str(tracker_yaml),
                verbose=False,
                device=cfg.device,
                half=cfg.half,
            )
            detections = adaptive_confidence_filter(
                parse_detections(results[0], frame, mask, cfg),
                cfg,
            )

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

            current_shapes = frame_shapes(tracks, frame_index, cfg)
            shapes.extend(current_shapes)
            frames_written += 1
            prev_frame = frame.copy()

            if progress is not None:
                progress.update(1)

            if show_enabled:
                try:
                    annotated = draw_tracks(frame, tracks, mask, frame_index, cfg)
                    cv2.imshow("Pig ID tracking", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
                except cv2.error:
                    print("OpenCV GUI preview is unavailable; continuing headless.")
                    show_enabled = False
    finally:
        capture.release()
        if progress is not None:
            progress.close()
        if show_enabled:
            cv2.destroyAllWindows()

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    shapes = apply_identity_swap_guard(shapes, width, height, cfg)
    shapes = refine_shapes_temporally(shapes, width, height, cfg)
    hidden_shape_count = sum(
        1 for shape in shapes if shape_hidden_value(shape) == "Yes"
    )
    review_shape_count = sum(
        1
        for shape in shapes
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
    rendered_frames = render_annotation_video(
        cfg.video_path,
        output_video,
        shapes,
        cfg,
        frame_limit=frames_written,
    )
    if rendered_frames != frames_written:
        raise RuntimeError(
            f"Rendered {rendered_frames} frames, but tracked {frames_written} frames."
        )

    write_annotation_json(annotations_json, shapes)
    write_coco_annotation_json(
        coco_annotations_json,
        shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior,
    )
    clean_shapes = clean_training_shapes(shapes, cfg)
    write_coco_annotation_json(
        clean_coco_annotations_json,
        clean_shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior,
        description=(
            "Clean pig training annotations exported as COCO 1.0 "
            "from detected, non-hidden, high-confidence boxes only"
        ),
    )
    max_shape_frame = max(int(shape["frame"]) for shape in shapes)
    source_frame_count = max(total_frames, max_shape_frame + 1)
    write_cvat_video_xml(
        cvat_video_xml,
        shapes,
        cfg.video_path,
        width,
        height,
        source_frame_count,
    )
    quality_report = build_quality_report(
        shapes,
        cfg,
        cfg.video_path,
        source_fps,
        source_frame_count,
        get_telemetry_summary(runtime),
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


def display_tracked_video(
    video_path: Path,
    width: int = 900,
    embed: bool = False,
) -> None:
    """Display the tracked MP4 directly in a notebook output cell."""
    from IPython.display import Video, display

    display(Video(str(video_path), embed=embed, width=width))


__all__ = [
    "display_tracked_video",
    "run_tracking",
]
