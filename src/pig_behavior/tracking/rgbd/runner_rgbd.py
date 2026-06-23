# ruff: noqa
"""RGB-D Bird's-Eye-View tracking runner.

# ruff: noqa

``run_rgbd_tracking`` is a self-contained alternative to the existing
:func:`~pig_behavior.tracking.runner.run_tracking`.  It reuses the same
YOLO detector, mask handling, ``FixedTrack`` state, and post-processing
(temporal refinement, CVAT/COCO export) but replaces the association step
with a BEV Kalman Filter + Euclidean-distance matcher.

The function never modifies the behaviour of the existing 2-D pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pig_behavior.tracking.config import (
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
from pig_behavior.tracking.rgbd.association_bev import match_bev_tracks
from pig_behavior.tracking.rgbd.config import (
    RGBDTrackingConfig,
    validate_rgbd_config,
)
from pig_behavior.tracking.rgbd.depth import (
    depth_frame_to_meters,
    load_calibration,
)
from pig_behavior.tracking.rgbd.kalman import (
    bev_position,
    bev_velocity,
    create_bev_kalman,
    predict_bev,
    update_bev,
)
from pig_behavior.tracking.rgbd.occlusion import (
    infer_occlusions,
    track_is_occluded,
    update_occlusion_age,
)
from pig_behavior.tracking.rgbd.projector import RGBDProjector
from pig_behavior.tracking.rgbd.reporting import (
    write_association_log_csv,
    write_tracking_csv,
)
from pig_behavior.tracking.rgbd.reporting import (
    write_quality_report_csv as write_rgbd_quality_csv,
)
from pig_behavior.tracking.rgbd.reporting import (
    write_quality_report_json as write_rgbd_quality_json,
)
from pig_behavior.tracking.rgbd.sanity import validate_rgbd_update_with_frame_size
from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    BEVTrackState,
    Detection2D,
    Detection3D,
    FrameTrackRow,
    RGBDQualityMetrics,
)
from pig_behavior.tracking.rgbd.sync import RGBDFrameSynchronizer
from pig_behavior.tracking.schemas import (
    Detection,
    FixedTrack,
    TrackingRuntimeState,
    TrackingSummary,
)
from pig_behavior.tracking.tracks import frame_shapes, initialize_tracks
from pig_behavior.tracking.visualization import render_annotation_video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter: existing Detection → Detection2D
# ---------------------------------------------------------------------------


def _detection_to_2d(det: Detection) -> Detection2D:
    """Convert an existing ``Detection`` to the RGB-D adapter schema."""
    box = det.box
    return Detection2D(
        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        confidence=det.score,
        class_id=det.class_id,
        mask=det.mask.copy() if det.mask is not None else None,
        hist=det.hist.copy() if det.hist is not None else None,
        raw_id=det.raw_id,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_rgbd_tracking(cfg: RGBDTrackingConfig) -> TrackingSummary:
    """Run RGB-D BEV-based tracking pipeline.

    This function mirrors the structure of ``run_tracking`` but uses depth
    data for association.  It produces the same output artifacts (CVAT XML,
    COCO JSON, quality report) plus RGB-D-specific CSV and JSON reports.
    """
    tc = cfg.tracking_config
    validate_config(tc)
    validate_rgbd_config(cfg)

    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Install tracking dependencies first: pip install -e .[tracking]"
        ) from exc

    # ---- output paths (reuse existing 2-D output resolver) -----------------
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
    ) = resolve_output_paths(tc)

    run_output_dir = output_video.parent
    rgbd_csv_path = run_output_dir / "tracking_result.csv"
    rgbd_quality_json = run_output_dir / "rgbd_quality_report.json"
    rgbd_quality_csv = run_output_dir / "rgbd_quality_report.csv"
    association_log_path = run_output_dir / "association_log.csv"

    write_tracker_yaml(tracker_yaml, tc)
    write_labels_json(labels_json)

    # ---- calibration & synchronisation -------------------------------------
    calibration = load_calibration(cfg)
    projector = RGBDProjector(calibration, cfg)
    sync = RGBDFrameSynchronizer(
        tc.video_path,
        cfg.depth_video_path,
        cfg.times_path,
    )

    # ---- video info --------------------------------------------------------
    capture = cv2.VideoCapture(str(tc.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {tc.video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or tc.output_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()  # we'll use the synchroniser for frame reading

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video frame size.")
    if total_frames and tc.start_frame >= total_frames:
        raise ValueError(
            f"start_frame={tc.start_frame} is outside video with "
            f"{total_frames} frames."
        )

    # ---- model & state -----------------------------------------------------
    mask = load_mask(tc.mask_path, width, height, tc)
    device_str = tc.device
    if device_str is None or device_str == "":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device_str)

    model = YOLO(str(tc.weights_path))
    try:
        model.to(device_str)
    except Exception as e:
        if "cuda" in device_str.lower() or "cuda" in str(e).lower():
            logger.warning("CUDA initialization failed, falling back to CPU: %s", e)
            device_str = "cpu"
            model.to(device_str)
            tc.device = "cpu"
        else:
            raise
    tracks: dict[int, FixedTrack] | None = None
    bev_states: dict[int, BEVTrackState] = {}
    runtime = TrackingRuntimeState()
    shapes: list[dict[str, Any]] = []
    csv_rows: list[FrameTrackRow] = []
    all_decisions: list[AssociationDecision] = []
    metrics = RGBDQualityMetrics()
    hidden_shape_count = 0
    review_shape_count = 0
    frame_index = tc.start_frame - 1
    frames_written = 0

    try:
        from tqdm import tqdm

        remaining = max(0, total_frames - tc.start_frame) if total_frames else None
        prog_total = tc.max_frames
        if prog_total is None:
            prog_total = remaining
        elif remaining is not None:
            prog_total = min(prog_total, remaining)
        progress = tqdm(total=prog_total, desc="RGB-D Tracking")
    except ImportError:
        progress = None

    try:
        while True:
            if tc.max_frames is not None and frames_written >= tc.max_frames:
                break

            frame_index += 1
            if frame_index >= (total_frames or float("inf")):
                break

            color_frame, depth_frame_raw, depth_idx = sync.read_synced(frame_index)
            if color_frame is None:
                break

            # Ensure frame dimensions match
            frame_h, frame_w = color_frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(tc.mask_path, width, height, tc)

            metrics.total_frames += 1

            # ---- YOLO detection (identical to 2-D pipeline) ----------------
            detector_frame = (
                apply_mask_to_frame(color_frame, mask)
                if tc.mask_input_frame and mask is not None
                else color_frame
            )
            results = model.predict(
                source=detector_frame,
                conf=tc.det_conf,
                iou=tc.nms_iou,
                max_det=tc.max_raw_detections,
                imgsz=tc.imgsz,
                verbose=False,
                device=tc.device,
                half=tc.half,
            )
            detections = adaptive_confidence_filter(
                parse_detections(results[0], color_frame, mask, tc),
                tc,
            )

            # ---- depth handling --------------------------------------------
            depth_available = depth_frame_raw is not None
            depth_frame_m: np.ndarray | None = None
            if depth_available:
                depth_frame_m = depth_frame_to_meters(
                    depth_frame_raw, calibration.depth_scale
                )

            # Convert detections
            dets_2d = [_detection_to_2d(d) for d in detections]

            # Project to 3D
            dets_3d: list[Detection3D] = []
            if depth_frame_m is not None:
                dets_3d = projector.project_detections(dets_2d, depth_frame_m)
            else:
                # No depth: create Detection3D with depth_valid=False
                for d2 in dets_2d:
                    dets_3d.append(
                        Detection3D(
                            detection_2d=d2,
                            depth_valid=False,
                            invalid_reason="no_depth_frame",
                        )
                    )

            # Count depth stats
            for d3 in dets_3d:
                if not d3.depth_valid:
                    metrics.depth_invalid_count += 1
                if d3.depth_ambiguous:
                    metrics.depth_ambiguous_count += 1

            # ---- initialise tracks on first frame -------------------------
            if tracks is None:
                tracks = initialize_tracks(detections, mask, width, height, tc)
                for fid, trk in tracks.items():
                    cx = float((trk.last_box[0] + trk.last_box[2]) / 2.0)
                    cy = float((trk.last_box[1] + trk.last_box[3]) / 2.0)

                    initial_bev = None
                    if trk.ever_detected and dets_3d:
                        matched_idx = None
                        for idx, det in enumerate(detections):
                            if np.allclose(trk.last_box, det.box):
                                matched_idx = idx
                                break
                        if matched_idx is not None and matched_idx < len(dets_3d):
                            det3d = dets_3d[matched_idx]
                            if det3d.bev_xy is not None:
                                initial_bev = det3d.bev_xy.copy()

                    if initial_bev is None:
                        # Fallback: project at a default depth of 2.0 meters
                        camera_xyz, world_xyz = projector.project_single_point(cx, cy, 2.0)
                        initial_bev = np.array(
                            [world_xyz[cfg.bev_axes[0]], world_xyz[cfg.bev_axes[1]]],
                            dtype=np.float64,
                        )

                    kf = create_bev_kalman(initial_bev, cfg)
                    bev_states[fid] = BEVTrackState(
                        fixed_id=fid,
                        kf=kf,
                        bev_position=initial_bev.copy(),
                        bev_velocity=np.zeros(2, dtype=np.float64),
                        state="tentative",
                    )

            # ---- depth failure mode ----------------------------------------
            all_depth_invalid = all(not d.depth_valid for d in dets_3d)
            skip_this_frame = False

            if all_depth_invalid and dets_3d:
                if cfg.depth_failure_mode == "predict_only":
                    metrics.predict_only_frame_count += 1
                    # Predict all tracks, no association
                    for fid in sorted(bev_states.keys()):
                        bev = bev_states[fid]
                        predict_bev(bev.kf)
                        bev.bev_position = bev_position(bev.kf)
                        bev.bev_velocity = bev_velocity(bev.kf)
                        bev.missed += 1
                        bev.age += 1
                        trk = tracks[fid]
                        predicted_box = trk.predicted_box(width, height)
                        trk.update_predicted(predicted_box, width, height)
                    skip_this_frame = True

                elif cfg.depth_failure_mode == "fallback_2d":
                    metrics.fallback_2d_count += 1
                    from pig_behavior.tracking.association import (
                        match_and_update_tracks,
                    )
                    match_and_update_tracks(
                        tracks, detections, color_frame, None, tc, runtime
                    )
                    skip_this_frame = True

                elif cfg.depth_failure_mode == "skip_frame":
                    metrics.predict_only_frame_count += 1
                    for fid in sorted(bev_states.keys()):
                        bev = bev_states[fid]
                        predict_bev(bev.kf)
                        bev.bev_position = bev_position(bev.kf)
                        bev.bev_velocity = bev_velocity(bev.kf)
                        bev.missed += 1
                        bev.age += 1
                        trk = tracks[fid]
                        predicted_box = trk.predicted_box(width, height)
                        trk.update_predicted(predicted_box, width, height)
                    skip_this_frame = True

            if not skip_this_frame and dets_3d:
                # ---- occlusion inference -----------------------------------
                occlusion_flags = infer_occlusions(dets_3d, cfg)
                occ_count = sum(1 for v in occlusion_flags.values() if v)
                if occ_count > 0:
                    metrics.occlusion_frame_count += 1

                # ---- predict all BEV states --------------------------------
                for fid in sorted(bev_states.keys()):
                    predict_bev(bev_states[fid].kf)

                # ---- BEV association ---------------------------------------
                assignments, decisions = match_bev_tracks(
                    tracks,
                    bev_states,
                    dets_3d,
                    occlusion_flags,
                    frame_index,
                    cfg,
                )
                all_decisions.extend(decisions)

                # ---- sanity gate + track update ----------------------------
                matched_track_ids: set[int] = set()
                occluded_track_ids: set[int] = set()

                for fid, det_idx in assignments.items():
                    trk = tracks[fid]
                    bev = bev_states[fid]
                    det3d = dets_3d[det_idx]

                    # Find the decision for this assignment
                    decision = next(
                        (d for d in decisions if d.track_id == fid and d.detection_index == det_idx),
                        AssociationDecision(
                            frame_index=frame_index,
                            track_id=fid,
                            detection_index=det_idx,
                            accepted=True,
                        ),
                    )

                    accepted, reject_reason = validate_rgbd_update_with_frame_size(
                        trk, bev, det3d, decision, cfg, width, height
                    )

                    if accepted:
                        # Update FixedTrack
                        orig_det = detections[det_idx] if det_idx < len(detections) else None
                        if orig_det is not None:
                            trk.update_detected(orig_det, width, height, tc)
                        # Update Kalman
                        if det3d.bev_xy is not None:
                            update_bev(bev.kf, det3d.bev_xy)
                            bev.bev_position = bev_position(bev.kf)
                            bev.bev_velocity = bev_velocity(bev.kf)
                            bev.last_depth_m = det3d.depth_m
                        bev.missed = 0
                        bev.hits += 1
                        bev.age += 1
                        if bev.hits >= 3:
                            bev.state = "confirmed"
                        matched_track_ids.add(fid)
                        if decision.bev_distance_m is not None:
                            metrics.association_distances.append(decision.bev_distance_m)
                    else:
                        # Sanity rejected → predict-only
                        metrics.rejected_update_count += 1
                        decision.accepted = False
                        decision.reject_reason = reject_reason
                        _count_rejection_reason(metrics, reject_reason)

                        bev.bev_position = bev_position(bev.kf)
                        bev.bev_velocity = bev_velocity(bev.kf)
                        bev.missed += 1
                        bev.age += 1
                        predicted_box = trk.predicted_box(width, height)
                        trk.update_predicted(predicted_box, width, height)

                    if decision.is_occluded:
                        occluded_track_ids.add(fid)

                # Unmatched tracks → predict only
                for fid in sorted(bev_states.keys()):
                    if fid in matched_track_ids:
                        continue
                    bev = bev_states[fid]
                    trk = tracks[fid]
                    bev.bev_position = bev_position(bev.kf)
                    bev.bev_velocity = bev_velocity(bev.kf)
                    bev.missed += 1
                    bev.age += 1
                    predicted_box = trk.predicted_box(width, height)
                    trk.update_predicted(predicted_box, width, height)

                # Update occlusion ages
                update_occlusion_age(
                    bev_states, matched_track_ids, occluded_track_ids, cfg
                )
            elif not skip_this_frame:
                # No detections at all
                for fid in sorted(bev_states.keys()):
                    bev = bev_states[fid]
                    predict_bev(bev.kf)
                    bev.bev_position = bev_position(bev.kf)
                    bev.bev_velocity = bev_velocity(bev.kf)
                    bev.missed += 1
                    bev.age += 1
                    trk = tracks[fid]
                    predicted_box = trk.predicted_box(width, height)
                    trk.update_predicted(predicted_box, width, height)

            # ---- emit shapes (compatible with existing exporter) -----------
            current_shapes = frame_shapes(tracks, frame_index, tc)
            shapes.extend(current_shapes)

            # ---- emit CSV rows ---------------------------------------------
            for fid in sorted(tracks.keys()):
                trk = tracks[fid]
                bev = bev_states.get(fid)
                world = None
                depth_val = None
                d_valid = True
                d_ambig = False
                assoc_dist = None
                rej_reason = None
                is_occ = False
                is_pred_only = False

                if bev is not None:
                    is_occ = track_is_occluded(bev)
                    is_pred_only = bev.missed > 0

                # Try to find the decision for this track in this frame
                frame_decisions = [
                    d for d in all_decisions
                    if d.frame_index == frame_index and d.track_id == fid
                ]
                if frame_decisions:
                    fd = frame_decisions[-1]
                    assoc_dist = fd.bev_distance_m
                    d_valid = fd.depth_valid
                    d_ambig = fd.depth_ambiguous
                    rej_reason = fd.reject_reason

                csv_rows.append(
                    FrameTrackRow(
                        frame=frame_index,
                        track_id=fid,
                        x1=float(trk.last_box[0]),
                        y1=float(trk.last_box[1]),
                        x2=float(trk.last_box[2]),
                        y2=float(trk.last_box[3]),
                        world_x=float(bev.bev_position[0]) if bev is not None else None,
                        world_y=float(bev.bev_position[1]) if bev is not None else None,
                        depth_m=bev.last_depth_m if bev is not None else None,
                        state=bev.state if bev is not None else "unknown",
                        confidence=float(trk.last_score),
                        is_occluded=is_occ,
                        is_predict_only=is_pred_only,
                        depth_valid=d_valid,
                        depth_ambiguous=d_ambig,
                        association_distance_m=assoc_dist,
                        reject_reason=rej_reason,
                    )
                )

            frames_written += 1
            if progress is not None:
                progress.update(1)

    finally:
        sync.release()
        if progress is not None:
            progress.close()

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    # ---- post-processing (reuse existing pipeline) -------------------------
    shapes = apply_identity_swap_guard(shapes, width, height, tc)
    shapes = refine_shapes_temporally(shapes, width, height, tc)
    hidden_shape_count = sum(
        1 for s in shapes if shape_hidden_value(s) == "Yes"
    )
    review_shape_count = sum(
        1
        for s in shapes
        if (
            s.get("_needs_review")
            or s.get("_refined")
            or s.get("_ambiguous_occlusion")
            or s.get("_occlusion_hold")
            or s.get("_area_occluded")
            or s.get("_merged_box_split")
            or s.get("_identity_swap_guard")
        )
    )

    # ---- render video ------------------------------------------------------
    if cfg.render:
        rendered_frames = render_annotation_video(
            tc.video_path,
            output_video,
            shapes,
            tc,
            frame_limit=frames_written,
        )
        if rendered_frames != frames_written:
            logger.warning(
                "Rendered %d frames, but tracked %d frames.",
                rendered_frames,
                frames_written,
            )

    # ---- export: existing formats ------------------------------------------
    write_annotation_json(annotations_json, shapes)
    write_coco_annotation_json(
        coco_annotations_json,
        shapes,
        tc.video_path,
        width,
        height,
        tc.default_behavior,
    )
    clean_shapes = clean_training_shapes(shapes, tc)
    write_coco_annotation_json(
        clean_coco_annotations_json,
        clean_shapes,
        tc.video_path,
        width,
        height,
        tc.default_behavior,
        description=(
            "Clean pig training annotations exported as COCO 1.0 "
            "from detected, non-hidden, high-confidence boxes only"
        ),
    )
    max_shape_frame = max(int(s["frame"]) for s in shapes)
    source_frame_count = max(total_frames, max_shape_frame + 1)
    if tc.mode == "gt_export":
        write_cvat_video_xml(
            cvat_video_xml,
            shapes,
            tc.video_path,
            width,
            height,
            source_frame_count,
        )
    write_labels_json(labels_json)

    # ---- export: RGB-D specific --------------------------------------------
    metrics.total_tracks = len(bev_states)
    metrics.confirmed_tracks = sum(
        1 for b in bev_states.values() if b.state == "confirmed"
    )
    metrics.lost_tracks = sum(
        1 for b in bev_states.values() if b.state == "lost"
    )
    metrics.ambiguous_match_count = sum(
        1
        for d in all_decisions
        if d.score_margin is not None and d.score_margin < cfg.min_score_margin
    )

    write_tracking_csv(rgbd_csv_path, csv_rows)
    write_rgbd_quality_json(rgbd_quality_json, metrics)
    write_rgbd_quality_csv(rgbd_quality_csv, metrics)

    if cfg.debug:
        write_association_log_csv(association_log_path, all_decisions)

    # ---- existing quality report -------------------------------------------
    quality_report = build_quality_report(
        shapes,
        tc,
        tc.video_path,
        source_fps,
        source_frame_count,
        {"rgbd_mode": 1},
    )
    write_quality_report_json(quality_report_json, quality_report)
    write_quality_report_csv(quality_report_csv, quality_report)

    # ---- summary -----------------------------------------------------------
    logger.info(
        "[OK] RGB-D tracking complete: frames=%d tracks=%d "
        "confirmed=%d lost=%d depth_invalid=%d rejected=%d",
        metrics.total_frames,
        metrics.total_tracks,
        metrics.confirmed_tracks,
        metrics.lost_tracks,
        metrics.depth_invalid_count,
        metrics.rejected_update_count,
    )

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
        start_frame=tc.start_frame,
        source_fps=source_fps,
        output_fps=tc.output_fps,
        telemetry={"rgbd_mode": 1},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_rejection_reason(
    metrics: RGBDQualityMetrics,
    reason: str | None,
) -> None:
    """Increment the specific rejection counter."""
    if reason is None:
        return
    counter_map = {
        "invalid_depth": "rejected_by_invalid_depth",
        "depth_ambiguous": "rejected_by_depth_ambiguous",
        "bev_distance_too_large": "rejected_by_bev_distance",
        "center_jump_too_large": "rejected_by_center_jump",
        "area_ratio_invalid": "rejected_by_area_ratio",
        "aspect_ratio_invalid": "rejected_by_aspect_ratio",
        "ambiguous_assignment": "rejected_by_score_margin",
    }
    attr = counter_map.get(reason)
    if attr is not None:
        setattr(metrics, attr, getattr(metrics, attr) + 1)


__all__ = [
    "run_rgbd_tracking",
]
