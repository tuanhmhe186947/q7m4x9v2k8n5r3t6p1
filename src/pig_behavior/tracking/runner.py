"""Top-level tracking runner for fixed-ID pig annotation export."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
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
    refine_far_camera_hidden_geometry,
    refine_near_wall_hidden_geometry,
    refine_shapes_temporally,
    repair_episode_pair_swaps,
    repair_hidden_suffix_id_swaps,
    repair_local_pair_swaps,
    repair_long_pair_swaps,
    repair_suffix_pair_swaps,
    shape_hidden_value,
    stabilize_overlap_hidden_islands,
    stabilize_realtime_motion_pairs,
    suppress_overlapped_small_low_confidence_boxes,
)
from pig_behavior.tracking.rf_hybrid_transfer import (
    apply_rf_hybrid_transfer,
    canonical_transfer_hash,
    validate_rf_hybrid_core_config,
    write_rf_hybrid_artifacts,
)
from pig_behavior.tracking.schemas import (
    FixedTrack,
    TrackingRuntimeState,
    TrackingSummary,
)
from pig_behavior.tracking.telemetry import (
    peak_process_rss_bytes,
    record_timing_sample,
    resolve_output_timing_contract,
)
from pig_behavior.tracking.tracks import frame_shapes, initialize_tracks
from pig_behavior.tracking.visualization import (
    draw_tracks,
    render_annotation_video,
)

logger = logging.getLogger(__name__)


def _model_to_device(device_str: str) -> str:
    normalized = device_str.strip().lower()
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def _cuda_telemetry_device(torch: Any, device_str: str) -> str | None:
    if not torch.cuda.is_available():
        return None
    normalized = device_str.strip().lower()
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return normalized
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return None


def _resolve_detector(model: object | None, weights_path: Path) -> object:
    """Return an injected detector or construct the unchanged default model."""

    if model is not None:
        return model
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Install tracking dependencies first: pip install -e .[tracking]"
        ) from exc
    return YOLO(str(weights_path))


def _render_output_video(
    cfg: TrackingConfig,
    output_video: Path,
    shapes: list[dict[str, Any]],
    frames_written: int,
) -> None:
    if not cfg.write_output_video:
        logger.info("MP4 rendering disabled; annotation exports remain enabled.")
        return
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


def run_tracking(
    cfg: TrackingConfig,
    model: object | None = None,
) -> TrackingSummary:
    """Run YOLOv8 + mask + stabilized eight-ID tracking.

    Supplying ``model`` is an opt-in test/replay seam.  The default ``None``
    path constructs and invokes Ultralytics exactly as before.
    """
    validate_config(cfg)
    if cfg.rf_hybrid_transfer:
        validate_rf_hybrid_core_config(cfg)
    logger.info(
        "tracking mode=%s tracker_type=%s cvat_video_xml=%s",
        cfg.mode,
        cfg.tracker_type,
        bool(cfg.cvat_video_xml),
    )
    if cfg.mode == "hybrid_bytetrack":
        logger.info(
            "hybrid modules: iou_fallback=%s, occlusion_aware_matching=%s, "
            "identity_swap_guard=%s, merged_box_split=%s, smoothing=%s, refinement=%s",
            cfg.USE_IOU_FALLBACK,
            cfg.occlusion_aware_matching,
            cfg.identity_swap_guard,
            cfg.USE_MERGED_BOX_SPLIT,
            cfg.smooth_boxes,
            cfg.refine_boxes,
        )
    elif cfg.mode == "bytetrack_raw":
        logger.info(
            "raw ByteTrack baseline: smoothing=%s, refinement=%s, "
            "occlusion_aware_matching=%s, identity_swap_guard=%s",
            cfg.smooth_boxes,
            cfg.refine_boxes,
            cfg.occlusion_aware_matching,
            cfg.identity_swap_guard,
        )

    try:
        import cv2
        import torch
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
    device_str = cfg.device
    if device_str is None or device_str == "":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device_str)

    model = _resolve_detector(model, cfg.weights_path)
    try:
        model.to(_model_to_device(device_str))
    except Exception as e:
        if "cuda" in device_str.lower() or "cuda" in str(e).lower():
            logger.warning("CUDA initialization failed, falling back to CPU: %s", e)
            device_str = "cpu"
            model.to(device_str)
            cfg.device = "cpu"
        else:
            raise
    tracks: dict[int, FixedTrack] | None = None
    runtime = TrackingRuntimeState()
    output_timing_contract, declared_delay_frames = (
        resolve_output_timing_contract(cfg)
    )
    runtime.telemetry.update(
        {
            "output_timing_contract": output_timing_contract,
            "declared_delay_frames": declared_delay_frames,
            "source_fps": source_fps,
        }
    )
    cuda_telemetry_device = _cuda_telemetry_device(torch, device_str)
    if cuda_telemetry_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_telemetry_device)
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

    import time

    total_process_time = 0.0
    try:
        while True:
            if cfg.max_frames is not None and frames_written >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            t_start = time.perf_counter()

            frame_h, frame_w = frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(cfg.mask_path, width, height, cfg)

            detector_frame = (
                apply_mask_to_frame(frame, mask)
                if cfg.mask_input_frame and mask is not None
                else frame
            )

            is_detect_frame = (
                cfg.mode in {"bytetrack_raw", "hybrid_bytetrack"}
                or (frame_index - cfg.start_frame) % cfg.detect_every_n_frames == 0
            )
            num_dets = 0

            if is_detect_frame:
                set_frame_context = getattr(model, "set_frame_context", None)
                if callable(set_frame_context):
                    set_frame_context(frame_index, (frame_h, frame_w))
                runtime.telemetry["detection_frames"] = (
                    int(runtime.telemetry["detection_frames"]) + 1
                )
                detector_start = time.perf_counter()
                inference_args = {
                    "source": detector_frame,
                    "conf": cfg.det_conf,
                    "iou": cfg.nms_iou,
                    "max_det": cfg.max_raw_detections,
                    "imgsz": cfg.imgsz,
                    "verbose": False,
                    "device": cfg.device,
                    "half": cfg.half,
                }
                if cfg.mode in {"bytetrack_raw", "hybrid_bytetrack"}:
                    results = model.track(
                        source=detector_frame,
                        persist=True,
                        conf=cfg.det_conf,
                        iou=cfg.nms_iou,
                        tracker=str(tracker_yaml),
                        verbose=False,
                        device=cfg.device,
                        half=cfg.half,
                    )
                else:
                    results = model.predict(**inference_args)
                detections = adaptive_confidence_filter(
                    parse_detections(results[0], frame, mask, cfg),
                    cfg,
                )
                record_timing_sample(
                    runtime,
                    "detector",
                    time.perf_counter() - detector_start,
                )
                num_dets = len(detections)

                association_start = time.perf_counter()
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
                        frame_index=frame_index,
                    )
                record_timing_sample(
                    runtime,
                    "association",
                    time.perf_counter() - association_start,
                )
            else:
                runtime.telemetry["skipped_detection_frames"] = (
                    int(runtime.telemetry["skipped_detection_frames"]) + 1
                )
                # Skip detection: update tracks using motion prediction only
                if tracks is not None:
                    from pig_behavior.tracking.tracks import (
                        lk_predict_box,
                        lk_predict_boxes_batched,
                    )

                    if cfg.realtime_lk_point_batching:
                        track_values = list(tracks.values())
                        lk_boxes = lk_predict_boxes_batched(
                            prev_frame,
                            frame,
                            [track.last_box for track in track_values],
                            width,
                            height,
                        )
                        track_boxes = zip(track_values, lk_boxes, strict=True)
                    else:
                        track_boxes = (
                            (
                                track,
                                lk_predict_box(
                                    prev_frame,
                                    frame,
                                    track.last_box,
                                    width,
                                    height,
                                ),
                            )
                            for track in tracks.values()
                        )
                    for track, predicted_box in track_boxes:
                        if predicted_box is None:
                            predicted_box = track.predicted_box(width, height)
                        track.update_predicted(
                            predicted_box,
                            width,
                            height,
                            cfg=cfg,
                            is_skip_frame=True,
                        )

            current_shapes = frame_shapes(tracks, frame_index, cfg)
            shapes.extend(current_shapes)
            frames_written += 1
            prev_frame = frame.copy()

            t_end = time.perf_counter()
            frame_time = t_end - t_start
            total_process_time += frame_time
            runtime.telemetry["frames_processed"] = (
                int(runtime.telemetry["frames_processed"]) + 1
            )
            record_timing_sample(runtime, "frame", frame_time)

            # Log frame statistics
            if tracks is not None:
                v_count = sum(1 for trk in tracks.values() if trk.get_state() == "VISIBLE")
                m_count = sum(
                    1
                    for trk in tracks.values()
                    if trk.get_state() in ("MISSING", "LOST")
                )
                o_count = sum(1 for trk in tracks.values() if trk.get_state() == "OCCLUDED")
                logger.debug(
                    "Frame %d: Latency=%.2fms | Dets=%d | Visible=%d | Missing=%d | Occluded=%d",
                    frame_index,
                    frame_time * 1000.0,
                    num_dets,
                    v_count,
                    m_count,
                    o_count,
                )
                if progress is not None:
                    progress.set_postfix(
                        lat=f"{frame_time * 1000.0:.1f}ms",
                        vis=v_count,
                        occ=o_count,
                    )

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
        if frames_written > 0:
            avg_fps = frames_written / max(total_process_time, 1e-6)
            logger.info(
                "Tracking finished. Total Frames: %d, Average FPS: %.2f",
                frames_written,
                avg_fps,
            )

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    postprocess_start = time.perf_counter()
    realtime_fast_output_json: Path | None = None
    rf_hybrid_output_json: Path | None = None
    rf_hybrid_change_ledger_json: Path | None = None
    if cfg.rf_hybrid_transfer:
        raw_snapshot = deepcopy(shapes)
        raw_hash = canonical_transfer_hash(raw_snapshot)
        transfer_result = apply_rf_hybrid_transfer(
            raw_snapshot,
            width,
            height,
            mask,
            video=cfg.video_path.stem,
        )
        if canonical_transfer_hash(raw_snapshot) != raw_hash:
            raise RuntimeError("frozen realtime_fast output was mutated")
        shapes = transfer_result.shapes
        realtime_fast_output_json = annotations_json.with_name(
            "realtime_fast_output.json"
        )
        rf_hybrid_output_json = annotations_json.with_name(
            "rf_hybrid_output.json"
        )
        rf_hybrid_change_ledger_json = annotations_json.with_name(
            "rf_hybrid_change_ledger.json"
        )
        write_rf_hybrid_artifacts(
            realtime_fast_path=realtime_fast_output_json,
            rf_hybrid_path=rf_hybrid_output_json,
            ledger_path=rf_hybrid_change_ledger_json,
            video=cfg.video_path.stem,
            raw_shapes=raw_snapshot,
            result=transfer_result,
        )
    else:
        if cfg.enable_offline_smoothing and cfg.identity_swap_guard:
            shapes = apply_identity_swap_guard(shapes, width, height, cfg)
        if cfg.enable_offline_smoothing and (
            cfg.smooth_boxes or cfg.refine_boxes
        ):
            shapes = refine_shapes_temporally(shapes, width, height, cfg)
            shapes = stabilize_overlap_hidden_islands(shapes, cfg)
            shapes = repair_local_pair_swaps(shapes, width, height, cfg)
            shapes = repair_episode_pair_swaps(shapes, width, height, cfg)
            shapes = repair_long_pair_swaps(shapes, width, height, cfg)
            shapes = repair_suffix_pair_swaps(shapes, width, height, cfg)
            shapes = suppress_overlapped_small_low_confidence_boxes(
                shapes,
                cfg,
            )
            shapes = repair_hidden_suffix_id_swaps(shapes, cfg)
        shapes = stabilize_realtime_motion_pairs(shapes, width, height, cfg)
        shapes = refine_near_wall_hidden_geometry(
            shapes,
            width,
            height,
            mask,
            cfg,
        )
        shapes = refine_far_camera_hidden_geometry(
            shapes,
            width,
            height,
            cfg,
        )
    runtime.telemetry["postprocess_time_ms_total"] = (
        time.perf_counter() - postprocess_start
    ) * 1000.0
    runtime.telemetry["peak_process_rss_bytes"] = peak_process_rss_bytes()
    if cuda_telemetry_device is not None:
        runtime.telemetry["peak_cuda_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(cuda_telemetry_device)
        )
        runtime.telemetry["peak_cuda_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(cuda_telemetry_device)
        )
    telemetry_summary = get_telemetry_summary(runtime)
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
            or shape.get("_near_wall_hidden_geometry_refined")
            or shape.get("_far_camera_hidden_geometry_refined")
        )
    )
    _render_output_video(cfg, output_video, shapes, frames_written)

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
        telemetry_summary,
    )
    write_quality_report_json(quality_report_json, quality_report)
    write_quality_report_csv(quality_report_csv, quality_report)
    if cfg.association_debug:
        debug_path = quality_report_csv.with_name("association_debug_events.csv")
        debug_events = runtime.association_debug_events
        if debug_events:
            fieldnames = sorted(
                {key for event in debug_events for key in event.keys()}
            )
            with debug_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(debug_events)
        else:
            debug_path.write_text("event\n", encoding="utf-8")
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
        telemetry=telemetry_summary,
        realtime_fast_output_json=realtime_fast_output_json,
        rf_hybrid_output_json=rf_hybrid_output_json,
        rf_hybrid_change_ledger_json=rf_hybrid_change_ledger_json,
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
