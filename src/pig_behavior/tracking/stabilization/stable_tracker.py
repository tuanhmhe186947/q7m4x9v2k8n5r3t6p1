"""Stable annotation online tracker and offline stitching/refinement pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pig_behavior.tracking.config import (
    resolve_output_paths,
    validate_config,
    write_tracker_yaml,
)
from pig_behavior.tracking.constants import TRACK_COLORS_BGR
from pig_behavior.tracking.detections import (
    adaptive_confidence_filter,
    hist_distance,
    parse_detections,
)
from pig_behavior.tracking.exporters.labels import write_labels_json
from pig_behavior.tracking.geometry import (
    area_log_ratio,
    bbox_center,
    bbox_iou,
    bbox_size,
    center_distance_norm,
)
from pig_behavior.tracking.masks import apply_mask_to_frame, load_mask
from pig_behavior.tracking.rgbd.config import validate_rgbd_config
from pig_behavior.tracking.rgbd.depth import depth_frame_to_meters, load_calibration
from pig_behavior.tracking.rgbd.kalman import (
    bev_position,
    create_bev_kalman,
    predict_bev,
    update_bev,
)
from pig_behavior.tracking.rgbd.projector import RGBDProjector
from pig_behavior.tracking.rgbd.runner_rgbd import _detection_to_2d
from pig_behavior.tracking.rgbd.sync import RGBDFrameSynchronizer
from pig_behavior.tracking.schemas import TrackingSummary
from pig_behavior.tracking.stabilization.bbox_smoothing import smooth_trajectory_boxes
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig
from pig_behavior.tracking.stabilization.cvat_export import write_stable_cvat_xml
from pig_behavior.tracking.stabilization.diagnostics import (
    FrameDiagnosticRow,
    write_frame_diagnostics,
    write_stability_summary,
    write_stitching_report,
    write_swap_candidates,
)
from pig_behavior.tracking.stabilization.swap_detection import (
    detect_and_optionally_fix_swaps,
)
from pig_behavior.tracking.stabilization.tracklet_stitching import (
    StableTrackletRecord,
    stitch_tracklets,
)

logger = logging.getLogger(__name__)


class OnlineTracklet:
    """Active online tracklet state during conservative online tracking."""

    def __init__(
        self,
        tracklet_id: int,
        start_frame: int,
        initial_bbox: np.ndarray,
        initial_hist: np.ndarray | None,
        initial_score: float,
        initial_bev_xy: np.ndarray | None = None,
        initial_depth_m: float | None = None,
        config: AnnotationStableConfig | None = None,
    ):
        self.tracklet_id = tracklet_id
        self.start_frame = start_frame
        self.bbox_sequence: list[np.ndarray] = [initial_bbox]
        self.hist_sequence: list[np.ndarray | None] = [initial_hist]
        self.confidence_sequence: list[float] = [initial_score]
        self.frames: list[int] = [start_frame]
        self.predict_only_streak = 0
        self.depth_valid_count = 1 if initial_depth_m is not None else 0
        self.bev_valid_count = 1 if initial_bev_xy is not None else 0
        self.bev_sequence: list[np.ndarray | None] = [initial_bev_xy]

        # Simple Kalman filter for BEV coordinates if depth is available
        self.kf = None
        if initial_bev_xy is not None and config is not None and config.rgbd_config is not None:
            self.kf = create_bev_kalman(initial_bev_xy, config.rgbd_config)

    def last_bbox(self) -> np.ndarray:
        return self.bbox_sequence[-1]

    def last_hist(self) -> np.ndarray | None:
        return self.hist_sequence[-1]

    def update_detected(
        self,
        frame_idx: int,
        bbox: np.ndarray,
        hist: np.ndarray | None,
        score: float,
        bev_xy: np.ndarray | None = None,
        depth_m: float | None = None,
    ) -> None:
        self.bbox_sequence.append(bbox)
        self.hist_sequence.append(hist)
        self.confidence_sequence.append(score)
        self.frames.append(frame_idx)
        self.bev_sequence.append(bev_xy)
        self.predict_only_streak = 0
        if depth_m is not None:
            self.depth_valid_count += 1
        if bev_xy is not None:
            self.bev_valid_count += 1
            if self.kf is not None:
                update_bev(self.kf, bev_xy)

    def update_predicted(self, frame_idx: int, bbox: np.ndarray) -> None:
        self.bbox_sequence.append(bbox)
        self.hist_sequence.append(self.last_hist())
        self.confidence_sequence.append(0.0)  # zero confidence for predicted
        self.frames.append(frame_idx)
        self.bev_sequence.append(None)
        self.predict_only_streak += 1
        if self.kf is not None:
            predict_bev(self.kf)

    def to_record(self, fixed_id: int) -> StableTrackletRecord:
        hists = [h for h in self.hist_sequence if h is not None]
        hist_summary = np.mean(hists, axis=0) if hists else None
        length = len(self.frames)
        return StableTrackletRecord(
            tracklet_id=self.tracklet_id,
            fixed_id=fixed_id,
            start_frame=self.start_frame,
            end_frame=self.frames[-1],
            bbox_sequence=self.bbox_sequence,
            center_sequence=[bbox_center(b) for b in self.bbox_sequence],
            area_sequence=[bbox_size(b)[0] * bbox_size(b)[1] for b in self.bbox_sequence],
            confidence_sequence=self.confidence_sequence,
            hist_summary=hist_summary,
            depth_valid_ratio=self.depth_valid_count / max(length, 1),
            bev_valid_ratio=self.bev_valid_count / max(length, 1),
            mean_confidence=float(np.mean(self.confidence_sequence)),
            length=length,
            bev_sequence=self.bev_sequence,
            frames=self.frames,
        )


def run_stable_tracking(config: AnnotationStableConfig) -> TrackingSummary:
    """Runs conservative online tracking followed by offline tracklet stitching,

    swap detection, box smoothing, and CVAT XML export.
    """
    tc = config.tracking_config
    validate_config(tc)
    if config.rgbd_config is not None:
        validate_rgbd_config(config.rgbd_config)

    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install tracking dependencies first: pip install -e .[tracking]") from exc

    # Resolve output paths using 2D base config resolver
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
    write_tracker_yaml(tracker_yaml, tc)
    write_labels_json(labels_json)

    # Output paths for new diagnostics reports
    stability_summary_path = run_output_dir / "tracking_stability_summary.csv"
    frame_diagnostics_path = run_output_dir / "tracking_stability_diagnostics.csv"
    stitching_report_path = run_output_dir / "tracklet_stitching_report.csv"
    swap_candidates_path = run_output_dir / "candidate_id_swaps.csv"

    # Setup calibration and synchronization if RGB-D mode is enabled
    projector = None
    sync = None
    if config.rgbd_config is not None:
        calibration = load_calibration(config.rgbd_config)
        projector = RGBDProjector(calibration, config.rgbd_config)
        sync = RGBDFrameSynchronizer(
            tc.video_path,
            config.rgbd_config.depth_video_path,
            config.rgbd_config.times_path,
        )

    capture = cv2.VideoCapture(str(tc.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {tc.video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or tc.output_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video frame size.")

    # Initialize tracking structures
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

    active_tracklets: list[OnlineTracklet] = []
    all_tracklets: list[OnlineTracklet] = []
    diagnostics_rows: list[FrameDiagnosticRow] = []

    frame_index = tc.start_frame - 1
    frames_written = 0
    next_tracklet_id = 1

    try:
        from tqdm import tqdm

        remaining = max(0, total_frames - tc.start_frame) if total_frames else None
        prog_total = tc.max_frames
        if prog_total is None:
            prog_total = remaining
        elif remaining is not None:
            prog_total = min(prog_total, remaining)
        progress = tqdm(total=prog_total, desc="Stable Tracker Online Phase")
    except ImportError:
        progress = None

    # Re-open or synchronize video frame loop
    capture = cv2.VideoCapture(str(tc.video_path))
    if tc.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, tc.start_frame)

    try:
        while True:
            if tc.max_frames is not None and frames_written >= tc.max_frames:
                break

            ok, frame = capture.read() if sync is None else (True, None)
            depth_frame_raw = None

            if sync is not None:
                frame_index += 1
                if frame_index >= (total_frames or float("inf")):
                    break
                frame, depth_frame_raw, _ = sync.read_synced(frame_index)
                if frame is None:
                    break
            else:
                if not ok:
                    break
                frame_index += 1

            frame_h, frame_w = frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(tc.mask_path, width, height, tc)

            # YOLO detection and filtering
            detector_frame = apply_mask_to_frame(frame, mask) if tc.mask_input_frame and mask is not None else frame
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
                parse_detections(results[0], frame, mask, tc),
                tc,
            )

            # Extract depth data
            depth_valid_for_frame = False
            depth_frame_m = None
            if depth_frame_raw is not None and config.rgbd_config is not None:
                depth_frame_m = depth_frame_to_meters(depth_frame_raw, calibration.depth_scale)
                depth_valid_for_frame = True

            # Prepare 3D projections if possible
            dets_2d = [_detection_to_2d(d) for d in detections]
            bev_positions_dets: list[np.ndarray | None] = [None] * len(detections)
            depth_vals_dets: list[float | None] = [None] * len(detections)

            if depth_valid_for_frame and projector is not None and depth_frame_m is not None:
                dets_3d = projector.project_detections(dets_2d, depth_frame_m)
                for i, d3d in enumerate(dets_3d):
                    if d3d.depth_valid:
                        bev_positions_dets[i] = d3d.bev_xy
                        depth_vals_dets[i] = d3d.depth_m

            # Association
            if not active_tracklets:
                # First frame initialization or recovery
                for i, det in enumerate(detections):
                    tracklet = OnlineTracklet(
                        tracklet_id=next_tracklet_id,
                        start_frame=frame_index,
                        initial_bbox=det.box.copy(),
                        initial_hist=det.hist.copy(),
                        initial_score=det.score,
                        initial_bev_xy=bev_positions_dets[i],
                        initial_depth_m=depth_vals_dets[i],
                        config=config,
                    )
                    active_tracklets.append(tracklet)
                    next_tracklet_id += 1
            else:
                # Build cost matrix
                m = len(active_tracklets)
                d = len(detections)
                cost_matrix = np.full((m, d), 1e6, dtype=np.float32)

                # Cache tracklet data for matching
                for r_idx, tracklet in enumerate(active_tracklets):
                    # Predict BEV Kalman if available
                    if tracklet.kf is not None:
                        predict_bev(tracklet.kf)

                    last_box = tracklet.last_bbox()
                    last_hist = tracklet.last_hist()

                    for c_idx, det in enumerate(detections):
                        # Computes components
                        iou_2d = bbox_iou(last_box, det.box)
                        cost_iou = 1.0 - iou_2d
                        cost_area = area_log_ratio(last_box, det.box)
                        cost_hist = hist_distance(last_hist, det.hist)

                        det_bev = bev_positions_dets[c_idx]

                        # Check depth availability
                        if tracklet.kf is not None and det_bev is not None:
                            # 3D BEV Euclidean distance cost
                            trk_bev = bev_position(tracklet.kf)
                            dist_bev = float(np.linalg.norm(trk_bev - det_bev))
                            cost_bev = min(1.0, dist_bev / 2.0)  # normalized

                            # Hybrid weight calculation
                            cost = (
                                config.w_bev * cost_bev
                                + config.w_iou_2d * cost_iou
                                + config.w_area * min(1.0, cost_area)
                                + config.w_hist * cost_hist
                                + config.w_conf * (1.0 - det.score)
                            )
                        else:
                            # Fallback 2D weights
                            cost = (
                                config.w_iou_2d_fallback * cost_iou
                                + config.w_area_fallback * min(1.0, cost_area)
                                + config.w_hist_fallback * cost_hist
                            )

                        cost_matrix[r_idx, c_idx] = cost

                # Solve Hungarian assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_tracklets_idx = set()
                matched_detections_idx = set()

                for r, c in zip(row_ind, col_ind, strict=False):
                    tracklet = active_tracklets[r]
                    det = detections[c]
                    cost = cost_matrix[r, c]

                    # Gating verification
                    last_box = tracklet.last_bbox()
                    iou_2d = bbox_iou(last_box, det.box)
                    area_ratio = area_log_ratio(last_box, det.box)
                    c_jump = center_distance_norm(last_box, det.box, width, height)

                    reject_reason = None
                    if iou_2d < config.min_iou_2d_for_match:
                        reject_reason = f"iou_2d_too_low_{iou_2d:.2f}"
                    elif c_jump > config.max_center_jump_norm:
                        reject_reason = f"center_jump_too_large_{c_jump:.2f}"
                    elif area_ratio > config.max_area_log_ratio:
                        reject_reason = f"area_log_ratio_too_large_{area_ratio:.2f}"

                    # Margin check for ambiguity
                    if reject_reason is None and config.prefer_gap_over_bad_match:
                        # Find second best cost for this tracklet
                        tracklet_costs = cost_matrix[r, :]
                        sorted_costs = np.sort(tracklet_costs)
                        if len(sorted_costs) > 1:
                            margin = sorted_costs[1] - cost
                            if margin < config.min_assignment_margin:
                                reject_reason = f"ambiguous_match_margin_{margin:.3f}"

                    if reject_reason is None:
                        # Accept match
                        matched_by = (
                            "bev_3d"
                            if (tracklet.kf is not None and bev_positions_dets[c] is not None)
                            else "2d_fallback"
                        )

                        # Store diagnostics
                        diagnostics_rows.append(
                            FrameDiagnosticRow(
                                frame=frame_index,
                                track_id=-1,  # will be assigned post-stitching
                                tracklet_id=tracklet.tracklet_id,
                                bbox=(
                                    float(det.box[0]),
                                    float(det.box[1]),
                                    float(det.box[2]),
                                    float(det.box[3]),
                                ),
                                center_jump_norm=c_jump,
                                area_ratio_prev=area_ratio,
                                depth_valid=(depth_vals_dets[c] is not None),
                                bev_valid=(bev_positions_dets[c] is not None),
                                matched_by=matched_by,
                                cost_bev=cost_matrix[r, c] if matched_by == "bev_3d" else None,
                                cost_iou_2d=1.0 - iou_2d,
                                cost_area=area_ratio,
                                cost_hist=hist_distance(tracklet.last_hist(), det.hist),
                                final_cost=float(cost),
                                assignment_margin=float(sorted_costs[1] - cost) if len(sorted_costs) > 1 else None,
                                is_ambiguous=False,
                                reject_reason=None,
                            )
                        )

                        tracklet.update_detected(
                            frame_index,
                            det.box.copy(),
                            det.hist.copy(),
                            det.score,
                            bev_positions_dets[c],
                            depth_vals_dets[c],
                        )
                        matched_tracklets_idx.add(r)
                        matched_detections_idx.add(c)
                    else:
                        # Rejected by gates -> goes into gap/predict-only
                        diagnostics_rows.append(
                            FrameDiagnosticRow(
                                frame=frame_index,
                                track_id=-1,
                                tracklet_id=tracklet.tracklet_id,
                                bbox=(
                                    float(last_box[0]),
                                    float(last_box[1]),
                                    float(last_box[2]),
                                    float(last_box[3]),
                                ),
                                center_jump_norm=c_jump,
                                area_ratio_prev=area_ratio,
                                depth_valid=False,
                                bev_valid=False,
                                matched_by="gap",
                                cost_bev=None,
                                cost_iou_2d=1.0 - iou_2d,
                                cost_area=area_ratio,
                                cost_hist=1.0,
                                final_cost=float(cost),
                                assignment_margin=None,
                                is_ambiguous=True,
                                reject_reason=reject_reason,
                            )
                        )

                # Process unmatched tracklets (predict-only)
                for r, tracklet in enumerate(active_tracklets):
                    if r in matched_tracklets_idx:
                        continue

                    last_box = tracklet.last_bbox()
                    # Keep static prediction or motion prediction (here we keep constant box for gap stability)
                    tracklet.update_predicted(frame_index, last_box.copy())

                    diagnostics_rows.append(
                        FrameDiagnosticRow(
                            frame=frame_index,
                            track_id=-1,
                            tracklet_id=tracklet.tracklet_id,
                            bbox=(
                                float(last_box[0]),
                                float(last_box[1]),
                                float(last_box[2]),
                                float(last_box[3]),
                            ),
                            center_jump_norm=0.0,
                            area_ratio_prev=0.0,
                            depth_valid=False,
                            bev_valid=False,
                            matched_by="predict_only",
                            cost_bev=None,
                            cost_iou_2d=1.0,
                            cost_area=0.0,
                            cost_hist=0.0,
                            final_cost=None,
                            assignment_margin=None,
                            is_ambiguous=False,
                            reject_reason=None,
                        )
                    )

                # Process unmatched detections (create new tracklets)
                for c in range(d):
                    if c in matched_detections_idx:
                        continue
                    det = detections[c]
                    tracklet = OnlineTracklet(
                        tracklet_id=next_tracklet_id,
                        start_frame=frame_index,
                        initial_bbox=det.box.copy(),
                        initial_hist=det.hist.copy(),
                        initial_score=det.score,
                        initial_bev_xy=bev_positions_dets[c],
                        initial_depth_m=depth_vals_dets[c],
                        config=config,
                    )
                    active_tracklets.append(tracklet)
                    next_tracklet_id += 1

            # Filter/terminate tracklets that reached predict-only streak limit
            still_active = []
            for tracklet in active_tracklets:
                if tracklet.predict_only_streak > config.max_predict_only_streak:
                    # Terminate tracklet
                    all_tracklets.append(tracklet)
                else:
                    still_active.append(tracklet)
            active_tracklets = still_active

            frames_written += 1
            if progress is not None:
                progress.update(1)

    finally:
        capture.release()
        if sync is not None:
            sync.release()
        if progress is not None:
            progress.close()

    # Move any remaining active tracklets to finished pool
    for tracklet in active_tracklets:
        all_tracklets.append(tracklet)

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    # Convert OnlineTracklets to StableTrackletRecord
    # Map them temporarily to standard IDs for offline phase
    records: list[StableTrackletRecord] = []
    for t in all_tracklets:
        # Assign dummy fixed_id initially, stitching will resolve it
        records.append(t.to_record(fixed_id=t.tracklet_id))

    # Filter tracklets: remove very short tracklets to clean noise
    filtered_records = [r for r in records if r.length >= config.min_tracklet_length]

    # Offline Stitching Phase
    tracklet_to_stable_id, stitching_rows = stitch_tracklets(filtered_records, config, width, height)

    # Sort tracks by duration to map the 8 longest tracks to Pig_1 .. Pig_8
    stable_ids = list(set(tracklet_to_stable_id.values()))
    stable_id_lengths = {sid: 0 for sid in stable_ids}
    for r in filtered_records:
        sid = tracklet_to_stable_id[r.tracklet_id]
        stable_id_lengths[sid] += r.length

    sorted_stable_ids = sorted(stable_ids, key=lambda sid: stable_id_lengths[sid], reverse=True)

    # Mapping dict to ensure main 8 IDs are 1..expected_pigs
    final_id_mapping = {}
    for new_id, old_id in enumerate(sorted_stable_ids, start=1):
        final_id_mapping[old_id] = new_id

    # Apply mapping
    tracklet_to_stable_id = {k: final_id_mapping[v] for k, v in tracklet_to_stable_id.items()}

    # Construct stable tracks structure: mapping stable_id -> {frame: (bbox, behavior, is_hidden)}
    stable_tracks: dict[int, dict[int, tuple[np.ndarray, str, bool]]] = {}
    for r in filtered_records:
        stable_id = tracklet_to_stable_id[r.tracklet_id]
        if stable_id not in stable_tracks:
            stable_tracks[stable_id] = {}

        # Reconstruct frame-by-frame data
        for i, frame in enumerate(r.frames):
            is_pred_only = r.confidence_sequence[i] == 0.0
            stable_tracks[stable_id][frame] = (
                r.bbox_sequence[i],
                r.hist_summary,
                tc.default_behavior,
                is_pred_only,
            )

    # Fill in diagnostics track_ids based on final stitched stable IDs
    # Map tracklet_id to final stable track ID
    for row in diagnostics_rows:
        if row.tracklet_id in tracklet_to_stable_id:
            row.track_id = tracklet_to_stable_id[row.tracklet_id]

    # Post-processing Phase 5: Swap Detection
    stable_tracks, swap_candidates = detect_and_optionally_fix_swaps(stable_tracks, config, width, height)

    # Post-processing Phase 6: Bbox Smoothing
    if config.smooth_bbox:
        for _stable_id, track_data in stable_tracks.items():
            # Build contiguous segment boxes
            sorted_frames = sorted(list(track_data.keys()))
            if not sorted_frames:
                continue

            # We process continuous chunks without gaps to smooth properly
            chunks = []
            current_chunk = [sorted_frames[0]]
            for f in sorted_frames[1:]:
                if f == current_chunk[-1] + 1:
                    current_chunk.append(f)
                else:
                    chunks.append(current_chunk)
                    current_chunk = [f]
            chunks.append(current_chunk)

            # Smooth each chunk
            for chunk in chunks:
                if len(chunk) < 3:
                    continue
                boxes_chunk = np.array([track_data[f][0] for f in chunk], dtype=np.float32)
                smoothed_chunk = smooth_trajectory_boxes(boxes_chunk, config, width, height)
                for idx, f in enumerate(chunk):
                    # update smoothed box
                    orig_data = track_data[f]
                    track_data[f] = (
                        smoothed_chunk[idx],
                        orig_data[1],
                        orig_data[2],
                        orig_data[3],
                    )

    # Export final stable CVAT XML
    write_stable_cvat_xml(
        cvat_video_xml,
        stable_tracks,
        tc.video_path,
        width,
        height,
        frame_count=frame_index + 1,
        expected_pigs=config.tracking_config.expected_pigs,
    )

    # Write CSV diagnostics
    write_frame_diagnostics(diagnostics_rows, frame_diagnostics_path)
    write_stitching_report(stitching_rows, stitching_report_path)
    write_swap_candidates(swap_candidates, swap_candidates_path)

    # Calculate global stability summary metrics
    total_detections = len(diagnostics_rows)
    total_jumps = sum(1 for row in diagnostics_rows if row.center_jump_norm > config.max_center_jump_norm)
    total_gaps = sum(1 for row in diagnostics_rows if row.matched_by in ("gap", "predict_only"))

    summary_data = {
        "total_frames": frames_written,
        "total_detections_parsed": total_detections,
        "total_tracklets_online": len(records),
        "total_tracklets_after_filtering": len(filtered_records),
        "total_final_stable_tracks": len(stable_tracks),
        "total_center_jumps": total_jumps,
        "total_gap_or_predicted_frames": total_gaps,
        "total_id_swaps_detected": len(swap_candidates),
        "total_id_swaps_fixed": sum(1 for c in swap_candidates if c.is_fixed),
    }
    write_stability_summary(summary_data, stability_summary_path)

    # Phase 9: Render Optional Debug Video with HUD and visual overlays
    if config.export_debug_video:
        _render_stable_debug_video(
            tc.video_path,
            output_video,
            stable_tracks,
            diagnostics_rows,
            config,
            frames_written,
        )

    # Build generic return shape list for standard metrics logic
    dummy_shapes = []
    for sid, tdata in stable_tracks.items():
        for f, val in tdata.items():
            bbox = val[0]
            hid = val[3]
            dummy_shapes.append(
                {
                    "frame": f,
                    "label": f"Pig_{sid}",
                    "points": [float(val) for val in bbox],
                    "attributes": [{"name": "Hidden", "value": "Yes" if hid else "No"}],
                }
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
        shape_count=len(dummy_shapes),
        hidden_shape_count=sum(1 for s in dummy_shapes if s["attributes"][0]["value"] == "Yes"),
        review_shape_count=0,
        start_frame=tc.start_frame,
        source_fps=source_fps,
        output_fps=tc.output_fps,
        telemetry=summary_data,
    )


def _render_stable_debug_video(
    video_path: Path,
    output_video: Path,
    stable_tracks: dict[int, dict[int, tuple[np.ndarray, str, bool]]],
    diagnostics_rows: list[FrameDiagnosticRow],
    config: AnnotationStableConfig,
    frame_limit: int,
) -> None:
    """Renders debug overlay video showing active tracklet status, matched_by colors,

    smoothed boxes, and diagnostic HUD.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning("Could not reopen video for stable debug overlay rendering.")
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config.tracking_config.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, config.tracking_config.start_frame)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.debug_video_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        logger.warning(f"Could not create output video for debug overlay: {output_video}")
        return

    # Index diagnostics by frame and track
    diag_by_frame_track: dict[int, dict[int, FrameDiagnosticRow]] = {}
    for row in diagnostics_rows:
        if row.track_id != -1:
            diag_by_frame_track.setdefault(row.frame, {})[row.track_id] = row

    frames_rendered = 0
    try:
        while frames_rendered < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break

            frame_idx = config.tracking_config.start_frame + frames_rendered
            overlay = frame.copy()

            # Render HUD text
            cv2.putText(
                overlay,
                f"STABLE TRACKING - Frame {frame_idx}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw pigs bboxes
            for pid in range(1, config.tracking_config.expected_pigs + 1):
                track_data = stable_tracks.get(pid, {})
                if frame_idx in track_data:
                    bbox, _, is_hidden = (
                        track_data[frame_idx][0],
                        track_data[frame_idx][2],
                        track_data[frame_idx][3],
                    )
                    x1, y1, x2, y2 = bbox.astype(int)
                    color = TRACK_COLORS_BGR.get(pid, (0, 255, 0))

                    # Determine matched_by color code
                    match_type = "gap"
                    diag_row = diag_by_frame_track.get(frame_idx, {}).get(pid)
                    if diag_row is not None:
                        match_type = diag_row.matched_by

                    # Border color based on match_type
                    # Green = BEV, Yellow = 2D fallback, Red = predict/gap
                    border_color = (
                        (0, 255, 0)
                        if match_type == "bev_3d"
                        else (0, 255, 255)
                        if match_type == "2d_fallback"
                        else (0, 0, 255)
                    )

                    if is_hidden:
                        # Draw dashed/thin box
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 1)
                    else:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 3)

                    # Draw text label
                    label = f"Pig_{pid} ({match_type})"
                    if is_hidden:
                        label += " HIDDEN"

                    cv2.putText(
                        overlay,
                        label,
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # Blend overlay
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
            writer.write(frame)
            frames_rendered += 1

    finally:
        capture.release()
        writer.release()
