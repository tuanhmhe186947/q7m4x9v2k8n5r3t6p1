from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .anchor_builder import build_anchor_records
from .config import RecoveryConfig
from .detector import YoloPigDetector
from .extractor import frame_filename, output_dir, safe_video_id, write_crop, write_full_frame
from .legacy_gt_loader import LegacyGtMap
from .mask_utils import bbox_mask_metrics, filter_detections_by_mask, load_scene_mask
from .path_utils import SourceResources
from .runtime import RuntimeReporter, write_progress_state
from .sequence_view_builder import build_sequence_views
from .timestamp_utils import timestamp_at
from .tracker import track_dense_range
from .video_utils import VideoReader


def _append_failed_manifest_rows(
    rows: list[dict[str, object]],
    anchor: dict[str, object],
    resources: SourceResources,
    timestamps: list[float],
    reason: str,
) -> None:
    support_frames = set(anchor["gt_support_frames"])
    legacy_gt_mode = "multi_anchor" if anchor.get("legacy_gt_by_frame") else "single_anchor"
    legacy_gt_by_frame = anchor.get("legacy_gt_by_frame", {})
    legacy_gt_support_frames = sorted(int(frame) for frame in legacy_gt_by_frame)
    x1, y1, x2, y2 = anchor["anchor_bbox"]
    for frame_index in anchor["dense_frame_indices"]:
        gt_available = int(frame_index) in legacy_gt_by_frame
        rows.append(
            {
                "tracklet_id": anchor["tracklet_id"],
                "group_id": anchor["group_id"],
                "sample_id": anchor["sample_id"],
                "pig_id": anchor["pig_id"],
                "behavior": anchor["behavior"],
                "hidden": anchor["hidden"],
                "day_final": anchor["day_final"],
                "source_video_original": resources.source_video_original,
                "source_video_resolved": resources.source_video_resolved,
                "source_folder": resources.source_folder,
                "timestamp_file_resolved": resources.times_txt_path,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_at(timestamps, int(frame_index)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox_source": (
                    "gt_legacy"
                    if gt_available
                    else ("gt_anchor" if frame_index == anchor["legacy_anchor_frame"] else "tracker")
                ),
                "det_confidence": None,
                "track_confidence": 0.0,
                "is_anchor_frame": frame_index == anchor["legacy_anchor_frame"],
                "is_gt_support_frame": gt_available or frame_index in support_frames,
                "is_interpolated": False,
                "tracking_status": "failed",
                "qa_status": "needs_review",
                "qa_notes": reason,
                "legacy_gt_mode": legacy_gt_mode,
                "legacy_gt_bbox_available": gt_available,
                "legacy_gt_support_count": len(legacy_gt_support_frames),
                "legacy_gt_support_frames": "|".join(map(str, legacy_gt_support_frames)),
                "detector_best_iou_with_legacy_gt": None,
                "detector_disagrees_with_legacy_gt": False,
                "segment_start_gt_frame": None,
                "segment_end_gt_frame": None,
                "segment_tracking_status": "failed",
                "id_switch_risk_score": 0.0,
                "num_detections_raw": 0,
                "num_detections_after_mask": 0,
                "num_detections_outside_mask": 0,
                "selected_det_center_in_mask": None,
                "selected_det_bbox_mask_coverage": None,
                "mask_filter_applied": False,
                "scene_mask_path": "",
                "crop_path": "",
                "full_frame_path": "",
                "legacy_anchor_frame": anchor["legacy_anchor_frame"],
                "legacy_anchor_time_sec": anchor["legacy_anchor_time_sec"],
                "legacy_anchor_frame_mod_6": anchor["legacy_anchor_frame_mod_6"],
                "legacy_interval_frame_list": "|".join(map(str, anchor["legacy_interval_frame_list"])),
                "legacy_interval_timestamp_list": "|".join(
                    "" if v is None else str(v) for v in anchor["legacy_interval_timestamp_list"]
                ),
                "legacy_interval_start_frame": anchor["legacy_interval_start_frame"],
                "legacy_interval_end_frame": anchor["legacy_interval_end_frame"],
                "legacy_interval_start_time_sec": anchor["legacy_interval_start_time_sec"],
                "legacy_interval_end_time_sec": anchor["legacy_interval_end_time_sec"],
                "depth_sync_status": "not_verified",
                "color_video_path": resources.color_video_path,
                "depth_video_path": resources.depth_video_path,
                "times_txt_path": resources.times_txt_path,
                "background_path": resources.background_path,
                "background_depth_path": resources.background_depth_path,
                "mask_path": resources.mask_path,
                "depth_scale_path": resources.depth_scale_path,
                "inverse_intrinsic_path": resources.inverse_intrinsic_path,
                "rot_path": resources.rot_path,
            }
        )


def build_dense_tracklets(
    accepted_df: pd.DataFrame,
    resources_by_video: dict[str, SourceResources],
    timestamps_by_video: dict[str, list[float]],
    config: RecoveryConfig,
    *,
    sequence_views: list[str] | None = None,
    reporter: RuntimeReporter | None = None,
    legacy_gt_map: LegacyGtMap | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[int, str]], dict[str, dict[int, str]], list[dict[str, object]]]:
    detector = None
    no_detection = config.no_detect_manifest_only
    if not no_detection:
        if config.detector_weights is None:
            raise ValueError("--detector-weights is required unless --no-detect-manifest-only is set")
        detector = YoloPigDetector(config.detector_weights)

    existing_dense = pd.DataFrame()
    skip_tracklet_ids: set[str] = set()
    dense_path = config.output_root / "legacy_dense_tracklet_map.csv"
    sequence_path = config.output_root / "legacy_training_sequence_manifest.csv"
    if config.resume and dense_path.exists():
        existing_dense = pd.read_csv(dense_path)
        if "tracklet_id" in existing_dense:
            skip_tracklet_ids = set(existing_dense["tracklet_id"].astype(str).unique())
        if reporter:
            reporter.log(f"RESUME loaded {len(skip_tracklet_ids)} processed tracklets from {dense_path}")

    anchors = build_anchor_records(
        accepted_df,
        timestamps_by_video,
        config.track_end_mode,
        show_progress=config.progress,
        skip_tracklet_ids=skip_tracklet_ids,
    )
    legacy_gt_map = legacy_gt_map or {}
    legacy_gt_mode = "multi_anchor" if config.legacy_burst_bbox_csv is not None else "single_anchor"
    for anchor in anchors:
        key = (str(anchor["group_id"]), str(anchor["pig_id"]))
        anchor["legacy_gt_by_frame"] = legacy_gt_map.get(key, {}) if legacy_gt_mode == "multi_anchor" else {}
    rows: list[dict[str, object]] = existing_dense.to_dict(orient="records") if not existing_dense.empty else []
    crop_paths_by_tracklet: dict[str, dict[int, str]] = {}
    full_paths_by_tracklet: dict[str, dict[int, str]] = {}
    failures: list[dict[str, object]] = []
    mask_cache = {}
    mask_filter_applied = bool(config.scene_mask is not None and config.mask_filter_detections)

    processed_this_run = 0

    def flush_partial(anchor_row: dict[str, object], stage: str) -> None:
        if not config.manifest_only or config.flush_every <= 0:
            return
        dense_df = pd.DataFrame(rows)
        dense_df.to_csv(dense_path, index=False)
        sequence_df = build_sequence_views(dense_df, sequence_views)
        sequence_df.to_csv(sequence_path, index=False)
        progress_path = write_progress_state(
            config.output_root,
            len(set(dense_df["tracklet_id"].astype(str))) if not dense_df.empty else 0,
            anchor_row.get("group_id"),
            anchor_row.get("video_final"),
            stage,
            reporter.total_sec() if reporter else 0.0,
        )
        if reporter:
            reporter.log(f"WROTE partial {dense_path}")
            reporter.log(f"WROTE partial {sequence_path}")
            reporter.log(f"WROTE {progress_path}")

    for anchor in tqdm(anchors, desc="Generating dense tracklet manifest", disable=not config.progress):
        processed_this_run += 1
        resources = resources_by_video.get(str(anchor["video_final"]))
        if (
            resources is None
            or not resources.source_video_resolved
            or not Path(resources.source_video_resolved).exists()
        ):
            failures.append({"tracklet_id": anchor["tracklet_id"], "reason": "missing_source_video"})
            if no_detection and resources is not None:
                _append_failed_manifest_rows(
                    rows,
                    anchor,
                    resources,
                    timestamps_by_video.get(str(anchor["video_final"]), []),
                    "missing_source_video",
                )
                if config.flush_every > 0 and processed_this_run % config.flush_every == 0:
                    flush_partial(anchor, "dense_manifest")
            continue

        raw_detections_by_frame = {}
        detections_by_frame = {}
        rejected_detections_by_frame = {}
        frame_cache = {}
        try:
            with VideoReader(resources.source_video_resolved) as reader:
                for frame_index in anchor["dense_frame_indices"]:
                    frame = reader.read(int(frame_index))
                    if frame is None:
                        failures.append(
                            {
                                "tracklet_id": anchor["tracklet_id"],
                                "frame_index": frame_index,
                                "reason": "frame_read_failed",
                            }
                        )
                        continue
                    frame_cache[int(frame_index)] = frame
                    if detector is not None:
                        raw_detections = detector.detect(frame, int(frame_index))
                        raw_detections_by_frame[int(frame_index)] = raw_detections
                        if mask_filter_applied:
                            frame_height, frame_width = frame.shape[:2]
                            mask_key = (frame_width, frame_height)
                            if mask_key not in mask_cache:
                                mask_cache[mask_key] = load_scene_mask(config.scene_mask, frame_width, frame_height)
                            kept_detections, rejected_detections = filter_detections_by_mask(
                                raw_detections,
                                mask_cache[mask_key],
                                min_bbox_coverage=config.mask_min_bbox_coverage,
                                require_center_inside=config.mask_require_center_inside,
                            )
                            detections_by_frame[int(frame_index)] = kept_detections
                            rejected_detections_by_frame[int(frame_index)] = rejected_detections
                        else:
                            detections_by_frame[int(frame_index)] = raw_detections
                            rejected_detections_by_frame[int(frame_index)] = []
        except Exception as exc:
            failures.append({"tracklet_id": anchor["tracklet_id"], "reason": f"video_or_detection_failed:{exc}"})
            if no_detection:
                _append_failed_manifest_rows(
                    rows,
                    anchor,
                    resources,
                    timestamps_by_video.get(str(anchor["video_final"]), []),
                    f"video_or_detection_failed:{exc}",
                )
                if config.flush_every > 0 and processed_this_run % config.flush_every == 0:
                    flush_partial(anchor, "dense_manifest")
            continue

        tracked = track_dense_range(
            list(anchor["dense_frame_indices"]),
            anchor["anchor_bbox"],
            detections_by_frame,
            list(anchor["gt_support_frames"]),
            no_detection_mode=no_detection,
            legacy_gt_by_frame=anchor.get("legacy_gt_by_frame", {}),
            legacy_gt_mode=legacy_gt_mode,
        )
        tracklet_id = str(anchor["tracklet_id"])
        crop_paths_by_tracklet[tracklet_id] = {}
        full_paths_by_tracklet[tracklet_id] = {}
        video_id = safe_video_id(resources.source_folder, resources.source_video_resolved)
        dense_crop_dir = output_dir(
            config.output_root,
            "crops",
            "dense_tracklet_0_to_12",
            str(anchor["day_final"]),
            video_id,
            str(anchor["group_id"]),
            str(anchor["pig_id"]),
        )
        dense_full_dir = output_dir(
            config.output_root,
            "full_frames",
            "dense_tracklet_0_to_12",
            str(anchor["day_final"]),
            video_id,
            str(anchor["group_id"]),
            str(anchor["pig_id"]),
        )

        tracked_iterator = tqdm(
            tracked,
            desc=f"Extracting crops/full frames {tracklet_id}",
            disable=not (
                config.progress
                and not config.manifest_only
                and (config.extract_crops or config.extract_full_frames)
            ),
            leave=False,
        )
        for tracked_box in tracked_iterator:
            frame_index = int(tracked_box.frame_index)
            frame = frame_cache.get(frame_index)
            crop_path = ""
            full_path = ""
            if frame is not None and not config.manifest_only:
                if config.extract_crops:
                    crop_path = write_crop(frame, tracked_box.bbox, dense_crop_dir / frame_filename(frame_index))
                if config.extract_full_frames:
                    full_path = write_full_frame(frame, dense_full_dir / frame_filename(frame_index))
            elif frame is not None:
                crop_path = str(dense_crop_dir / frame_filename(frame_index))
                full_path = str(dense_full_dir / frame_filename(frame_index))
            crop_paths_by_tracklet[tracklet_id][frame_index] = crop_path
            full_paths_by_tracklet[tracklet_id][frame_index] = full_path

            x1, y1, x2, y2 = tracked_box.bbox
            legacy_gt_by_frame = anchor.get("legacy_gt_by_frame", {})
            legacy_gt_support_frames = sorted(int(frame) for frame in legacy_gt_by_frame)
            raw_detection_count = len(raw_detections_by_frame.get(frame_index, []))
            masked_detection_count = len(detections_by_frame.get(frame_index, []))
            rejected_detection_count = len(rejected_detections_by_frame.get(frame_index, []))
            selected_center_in_mask = None
            selected_bbox_mask_coverage = None
            qa_status = tracked_box.qa_status
            qa_notes = tracked_box.qa_notes
            if mask_filter_applied and frame is not None:
                frame_height, frame_width = frame.shape[:2]
                mask_key = (frame_width, frame_height)
                if mask_key not in mask_cache:
                    mask_cache[mask_key] = load_scene_mask(config.scene_mask, frame_width, frame_height)
                selected_metrics = bbox_mask_metrics(mask_cache[mask_key], tracked_box.bbox)
                selected_center_in_mask = selected_metrics.center_in_mask
                selected_bbox_mask_coverage = selected_metrics.bbox_mask_coverage
                selected_outside_mask = (
                    (config.mask_require_center_inside and not selected_metrics.center_in_mask)
                    or selected_metrics.bbox_mask_coverage < config.mask_min_bbox_coverage
                )
                if selected_outside_mask:
                    qa_status = "review"
                    if "selected_bbox_outside_scene_mask" not in qa_notes:
                        qa_notes = ";".join(filter(None, [qa_notes, "selected_bbox_outside_scene_mask"]))
            row = {
                "tracklet_id": tracklet_id,
                "group_id": anchor["group_id"],
                "sample_id": anchor["sample_id"],
                "pig_id": anchor["pig_id"],
                "behavior": anchor["behavior"],
                "hidden": anchor["hidden"],
                "day_final": anchor["day_final"],
                "source_video_original": resources.source_video_original,
                "source_video_resolved": resources.source_video_resolved,
                "source_folder": resources.source_folder,
                "timestamp_file_resolved": resources.times_txt_path,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_at(timestamps_by_video.get(str(anchor["video_final"]), []), frame_index),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox_source": tracked_box.bbox_source,
                "det_confidence": tracked_box.det_confidence,
                "track_confidence": tracked_box.track_confidence,
                "is_anchor_frame": tracked_box.is_anchor_frame,
                "is_gt_support_frame": tracked_box.is_gt_support_frame,
                "is_interpolated": tracked_box.is_interpolated,
                "tracking_status": tracked_box.tracking_status,
                "qa_status": qa_status,
                "qa_notes": qa_notes,
                "legacy_gt_mode": legacy_gt_mode,
                "legacy_gt_bbox_available": tracked_box.legacy_gt_bbox_available,
                "legacy_gt_support_count": len(legacy_gt_support_frames),
                "legacy_gt_support_frames": "|".join(map(str, legacy_gt_support_frames)),
                "detector_best_iou_with_legacy_gt": tracked_box.detector_best_iou_with_legacy_gt,
                "detector_disagrees_with_legacy_gt": tracked_box.detector_disagrees_with_legacy_gt,
                "segment_start_gt_frame": tracked_box.segment_start_gt_frame,
                "segment_end_gt_frame": tracked_box.segment_end_gt_frame,
                "segment_tracking_status": tracked_box.segment_tracking_status,
                "id_switch_risk_score": tracked_box.id_switch_risk_score,
                "num_detections_raw": raw_detection_count,
                "num_detections_after_mask": masked_detection_count,
                "num_detections_outside_mask": rejected_detection_count,
                "selected_det_center_in_mask": selected_center_in_mask,
                "selected_det_bbox_mask_coverage": selected_bbox_mask_coverage,
                "mask_filter_applied": mask_filter_applied,
                "scene_mask_path": "" if config.scene_mask is None else str(config.scene_mask),
                "crop_path": crop_path,
                "full_frame_path": full_path,
                "legacy_anchor_frame": anchor["legacy_anchor_frame"],
                "legacy_anchor_time_sec": anchor["legacy_anchor_time_sec"],
                "legacy_anchor_frame_mod_6": anchor["legacy_anchor_frame_mod_6"],
                "legacy_interval_frame_list": "|".join(map(str, anchor["legacy_interval_frame_list"])),
                "legacy_interval_timestamp_list": "|".join(
                    "" if v is None else str(v) for v in anchor["legacy_interval_timestamp_list"]
                ),
                "legacy_interval_start_frame": anchor["legacy_interval_start_frame"],
                "legacy_interval_end_frame": anchor["legacy_interval_end_frame"],
                "legacy_interval_start_time_sec": anchor["legacy_interval_start_time_sec"],
                "legacy_interval_end_time_sec": anchor["legacy_interval_end_time_sec"],
                "depth_sync_status": "not_verified",
                "color_video_path": resources.color_video_path,
                "depth_video_path": resources.depth_video_path,
                "times_txt_path": resources.times_txt_path,
                "background_path": resources.background_path,
                "background_depth_path": resources.background_depth_path,
                "mask_path": resources.mask_path,
                "depth_scale_path": resources.depth_scale_path,
                "inverse_intrinsic_path": resources.inverse_intrinsic_path,
                "rot_path": resources.rot_path,
            }
            rows.append(row)

            if config.save_debug_visuals and frame is not None and not config.manifest_only:
                debug_dir = config.output_root / "debug_visuals" / str(tracklet_id)
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug = frame.copy()
                if mask_filter_applied:
                    frame_height, frame_width = frame.shape[:2]
                    scene_mask = mask_cache.get((frame_width, frame_height))
                    if scene_mask is not None:
                        overlay = debug.copy()
                        mask2d = np.asarray(scene_mask)
                        if mask2d.ndim == 3:
                            if mask2d.shape[2] == 1:
                                mask2d = mask2d[:, :, 0]
                            else:
                                mask2d = np.any(mask2d > 0, axis=2)

                        mask2d = mask2d.astype(bool)
                        overlay[mask2d] = (0, 80, 0)
                        debug = cv2.addWeighted(overlay, 0.25, debug, 0.75, 0)
                        mask_uint8 = scene_mask.astype("uint8") * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(debug, contours, -1, (0, 140, 0), 1)
                if config.debug_draw_all_detections:
                    for detection in detections_by_frame.get(frame_index, []):
                        cv2.rectangle(
                            debug,
                            (int(detection.x1), int(detection.y1)),
                            (int(detection.x2), int(detection.y2)),
                            (0, 180, 0),
                            1,
                        )
                    if config.debug_draw_filtered_detections:
                        for detection in rejected_detections_by_frame.get(frame_index, []):
                            cv2.rectangle(
                                debug,
                                (int(detection.x1), int(detection.y1)),
                                (int(detection.x2), int(detection.y2)),
                                (80, 80, 180),
                                1,
                            )
                gt_record = anchor.get("legacy_gt_by_frame", {}).get(frame_index)
                if gt_record is not None:
                    gx1, gy1, gx2, gy2 = gt_record["bbox"]
                    cv2.rectangle(debug, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (255, 0, 0), 2)
                    cv2.putText(
                        debug,
                        "GT",
                        (int(gx1), max(0, int(gy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                status_text = [
                    f"src={tracked_box.bbox_source}",
                    f"status={tracked_box.tracking_status}",
                    f"mode={legacy_gt_mode}",
                    f"track={tracked_box.track_confidence:.2f}",
                    f"det={'' if tracked_box.det_confidence is None else f'{tracked_box.det_confidence:.2f}'}",
                    f"det_gt_disagree={tracked_box.detector_disagrees_with_legacy_gt}",
                    f"raw_det={raw_detection_count}",
                    f"mask_det={masked_detection_count}",
                    f"outside_mask={rejected_detection_count}",
                    f"mask_cov={'' if selected_bbox_mask_coverage is None else f'{selected_bbox_mask_coverage:.2f}'}",
                ]
                for text_idx, text in enumerate(status_text):
                    cv2.putText(
                        debug,
                        text,
                        (8, 18 + text_idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(str(debug_dir / frame_filename(frame_index)), debug)

        if config.flush_every > 0 and processed_this_run % config.flush_every == 0:
            flush_partial(anchor, "dense_manifest")

    if anchors:
        flush_partial(anchors[-1], "dense_manifest_complete")

    return pd.DataFrame(rows), crop_paths_by_tracklet, full_paths_by_tracklet, failures
