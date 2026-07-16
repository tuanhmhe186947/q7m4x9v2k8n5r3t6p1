"""Command-line interface for fixed-ID pig tracking."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from pig_behavior.tracking.config import (
    TRACKING_MODE_CHOICES,
    TrackingConfig,
    get_telemetry_summary,
    tracking_rule_flags_enabled,
)
from pig_behavior.tracking.constants import (
    DEFAULT_DET_CONF_THRESHOLD,
    DEFAULT_DETECT_EVERY_N_FRAMES,
    DEFAULT_DUP_AREA_RATIO_THRESHOLD,
    DEFAULT_DUP_CENTER_THRESHOLD,
    DEFAULT_DUP_CONTAINMENT_THRESHOLD,
    DEFAULT_DUP_IOU_THRESHOLD,
    DEFAULT_MASK_PATH,
    DEFAULT_MAX_LOST_FRAMES,
    DEFAULT_MAX_RAW_DETECTIONS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REVIEW_CONF_THRESHOLD,
    DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACK_MATCH_IOU_THRESHOLD,
    DEFAULT_VIDEO_PATH,
    DEFAULT_VISUAL_OPACITY,
    DEFAULT_WEIGHTS_PATH,
)
from pig_behavior.tracking.runner import display_tracked_video, run_tracking
from pig_behavior.tracking.schemas import TrackingSummary
from pig_behavior.tracking_path_config import (
    DEFAULT_TRACKING_PATH_CONFIG,
    load_tracking_path_profile,
    profile_path,
    profile_video_path,
    profile_video_paths,
)


def _parse_profile_override(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"profile override must use KEY=VALUE format: {raw!r}"
        )
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"profile override key cannot be empty: {raw!r}"
        )
    return key, value.strip()


def _coerce_profile_override_value(current: object, value: str) -> object:
    if isinstance(current, bool):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise argparse.ArgumentTypeError(
            f"invalid boolean override value {value!r}"
        )
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value)
    if current is None:
        return value
    return type(current)(value)


def _apply_profile_overrides(
    cfg: TrackingConfig, overrides: list[tuple[str, str]]
) -> None:
    valid_fields = {field.name for field in fields(TrackingConfig)}
    for key, value in overrides:
        if key not in valid_fields:
            raise argparse.ArgumentTypeError(
                f"unknown TrackingConfig override {key!r}"
            )
        current = getattr(cfg, key)
        setattr(cfg, key, _coerce_profile_override_value(current, value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-key", type=str, default=None)
    parser.add_argument(
        "--all-config-videos",
        action="store_true",
        help="Run every video listed in the selected tracking path profile.",
    )
    parser.add_argument(
        "--path-config",
        type=Path,
        default=DEFAULT_TRACKING_PATH_CONFIG,
        help="JSON path profile file for fast video/weights/mask/output switching.",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument(
        "--no-output-video",
        action="store_true",
        help="Skip MP4 rendering while retaining annotation and quality exports.",
    )
    parser.add_argument(
        "--profile-override",
        action="append",
        default=[],
        type=_parse_profile_override,
        metavar="KEY=VALUE",
        help="Override a TrackingConfig field. May be supplied multiple times.",
    )
    parser.add_argument(
        "--no-emit-hidden-tracks",
        action="store_true",
        help="Compatibility flag for batch scripts that suppress Hidden=Yes labels.",
    )
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--coco-json", type=Path, default=None)
    parser.add_argument("--clean-coco-json", type=Path, default=None)
    parser.add_argument("--cvat-video-xml", type=Path, default=None)
    parser.add_argument("--labels-json", type=Path, default=None)
    parser.add_argument("--tracker-yaml", type=Path, default=None)
    parser.add_argument("--quality-report-json", type=Path, default=None)
    parser.add_argument("--quality-report-csv", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO inference, for example '0' or 'cpu'.",
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--tracker-type",
        type=str,
        choices=["bytetrack", "botsort"],
        default="bytetrack",
        help="YOLO tracker algorithm to use: 'bytetrack' or 'botsort'.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Deprecated alias for --review-conf.",
    )
    parser.add_argument("--det-conf", type=float, default=DEFAULT_DET_CONF_THRESHOLD)
    parser.add_argument(
        "--track-high-conf",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument(
        "--review-conf",
        type=float,
        default=DEFAULT_REVIEW_CONF_THRESHOLD,
    )
    parser.add_argument("--adaptive-conf-step", type=float, default=0.05)
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Legacy alias/override for --nms-iou.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=DEFAULT_NMS_IOU_THRESHOLD,
        help="YOLO inference NMS IoU threshold.",
    )
    parser.add_argument(
        "--track-match-iou",
        type=float,
        default=DEFAULT_TRACK_MATCH_IOU_THRESHOLD,
        help="ByteTrack/BoT-SORT association match threshold.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(TRACKING_MODE_CHOICES),
        default="realtime",
        help=(
            "Tracking mode: realtime, bytetrack_raw baseline, "
            "or hybrid_bytetrack improved pipeline."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--detect-every-n-frames",
        type=int,
        default=DEFAULT_DETECT_EVERY_N_FRAMES,
        help="Run detection every N frames.",
    )
    parser.add_argument(
        "--max-raw-detections",
        type=int,
        default=DEFAULT_MAX_RAW_DETECTIONS,
        help="Maximum number of raw YOLO detections.",
    )
    parser.add_argument(
        "--enable-offline-smoothing",
        action="store_true",
        help="Enable offline smoothing/refinement (realtime mode ignores this).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--default-behavior", type=str, default="lying")
    parser.add_argument("--roi-mode", choices=["center", "cover"], default="center")
    parser.add_argument("--roi-min-cover", type=float, default=0.10)
    parser.add_argument("--roi-dilate-px", type=int, default=8)
    parser.add_argument("--hidden-missed-frames", type=int, default=5)
    parser.add_argument("--hidden-score-threshold", type=float, default=0.15)
    parser.add_argument("--mask-iou-max-missed", type=int, default=10)
    parser.add_argument("--mask-iou-min-area", type=int, default=64)
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=DEFAULT_MAX_LOST_FRAMES,
    )
    parser.add_argument("--match-cost-threshold", type=float, default=0.78)
    parser.add_argument("--unseen-track-cost-threshold", type=float, default=1.10)
    parser.add_argument("--lost-track-cost-threshold", type=float, default=0.95)
    parser.add_argument(
        "--lost-track-reid-appearance-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--initial-track-conf",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument(
        "--motion-gate-confidence",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument("--low-conf-max-center-jump", type=float, default=0.08)
    parser.add_argument("--low-conf-max-box-jump-scale", type=float, default=1.75)
    parser.add_argument("--low-conf-min-iou", type=float, default=0.01)
    parser.add_argument("--occlusion-track-iom-threshold", type=float, default=0.20)
    parser.add_argument("--occlusion-detection-iom-threshold", type=float, default=0.30)
    parser.add_argument("--occlusion-stationary-speed", type=float, default=0.006)
    parser.add_argument(
        "--occlusion-stationary-max-center-jump",
        type=float,
        default=0.045,
    )
    parser.add_argument("--occlusion-switch-penalty", type=float, default=0.45)
    parser.add_argument("--occlusion-competitor-margin", type=float, default=0.12)
    parser.add_argument("--occlusion-appearance-penalty", type=float, default=0.30)
    parser.add_argument("--occlusion-appearance-margin", type=float, default=0.08)
    parser.add_argument("--directional-y-penalty-weight", type=float, default=0.12)
    parser.add_argument("--directional-y-velocity-epsilon-px", type=float, default=3.0)
    parser.add_argument("--directional-y-margin-px", type=float, default=5.0)
    parser.add_argument("--occlusion-hold-max-frames", type=int, default=30)
    parser.add_argument("--occlusion-hold-hidden-frames", type=int, default=2)
    parser.add_argument("--use-iou-fallback", action="store_true")
    parser.add_argument("--use-area-occlusion-freeze", action="store_true")
    parser.add_argument(
        "--use-conditional-area-occlusion-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-merged-box-split", action="store_true")
    parser.add_argument("--iou-fallback-threshold", type=float, default=0.45)
    parser.add_argument("--area-occlusion-shrink-ratio", type=float, default=0.60)
    parser.add_argument("--area-occlusion-freeze-frames", type=int, default=15)
    parser.add_argument("--merged-box-growth-ratio", type=float, default=1.50)
    parser.add_argument("--merged-box-neighbor-distance", type=float, default=0.12)
    parser.add_argument("--merged-box-split-max-tracks", type=int, default=2)
    parser.add_argument(
        "--hard-occlusion-track-iom-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--hard-occlusion-detection-iom-threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument("--hard-occlusion-min-frames", type=int, default=2)
    parser.add_argument("--hard-occlusion-recovery-frames", type=int, default=4)
    parser.add_argument("--hard-occlusion-score-threshold", type=float, default=0.65)
    parser.add_argument("--identity-swap-min-gain", type=float, default=0.015)
    parser.add_argument("--identity-swap-iom-threshold", type=float, default=0.10)
    parser.add_argument("--hidden-velocity-alpha", type=float, default=0.65)
    parser.add_argument("--hidden-acceleration-alpha", type=float, default=0.35)
    parser.add_argument("--hidden-stationary-speed", type=float, default=0.006)
    parser.add_argument("--hidden-motion-history", type=int, default=8)
    parser.add_argument("--hidden-min-motion-history", type=int, default=4)
    parser.add_argument("--hidden-stationary-displacement", type=float, default=0.015)
    parser.add_argument("--hidden-moving-displacement", type=float, default=0.035)
    parser.add_argument("--hidden-motion-consistency", type=float, default=0.55)
    parser.add_argument("--hidden-stationary-lock-frames", type=int, default=8)
    parser.add_argument("--hidden-max-motion-step-box-scale", type=float, default=1.50)
    parser.add_argument(
        "--dup-iou-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=None,
        help="Deprecated alias for --dup-iou-threshold.",
    )
    parser.add_argument(
        "--dup-containment-threshold",
        type=float,
        default=DEFAULT_DUP_CONTAINMENT_THRESHOLD,
    )
    parser.add_argument(
        "--dup-center-threshold",
        type=float,
        default=DEFAULT_DUP_CENTER_THRESHOLD,
    )
    parser.add_argument(
        "--dup-area-ratio-threshold",
        type=float,
        default=DEFAULT_DUP_AREA_RATIO_THRESHOLD,
    )
    parser.add_argument("--max-box-scale-change", type=float, default=0.25)
    parser.add_argument("--max-box-scale-change-after-gap", type=float, default=0.75)
    parser.add_argument("--high-conf-smooth-alpha", type=float, default=0.75)
    parser.add_argument("--mid-conf-smooth-alpha", type=float, default=0.55)
    parser.add_argument("--low-conf-smooth-alpha", type=float, default=0.35)
    parser.add_argument("--refine-max-gap", type=int, default=15)
    parser.add_argument("--refine-size-jump-threshold", type=float, default=0.45)
    parser.add_argument("--visual-opacity", type=float, default=DEFAULT_VISUAL_OPACITY)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-mask-input", action="store_true")
    parser.add_argument("--no-mask-iou", action="store_true")
    parser.add_argument("--no-smooth-boxes", action="store_true")
    parser.add_argument("--no-refine-boxes", action="store_true")
    parser.add_argument("--no-low-conf-motion-gate", action="store_true")
    parser.add_argument("--no-occlusion-aware-matching", action="store_true")
    parser.add_argument("--no-directional-y-prior", action="store_true")
    parser.add_argument("--learn-identity-in-occlusion", action="store_true")
    parser.add_argument("--no-hold-occluded-box", action="store_true")
    parser.add_argument("--no-identity-swap-guard", action="store_true")
    parser.add_argument("--no-occlusion-stationary-lock", action="store_true")
    parser.add_argument("--no-hidden-motion-model", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--display-inline", action="store_true")

    # ---- RGB-D mode arguments ----------------------------------------------
    rgbd_group = parser.add_argument_group("RGB-D tracking (optional)")
    rgbd_group.add_argument(
        "--rgbd",
        action="store_true",
        help="Enable RGB-D BEV tracking.",
    )
    rgbd_group.add_argument("--depth-video", type=Path, default=None)
    rgbd_group.add_argument("--times-file", type=Path, default=None)
    rgbd_group.add_argument("--depth-scale-file", type=Path, default=None)
    rgbd_group.add_argument("--inverse-intrinsic-file", type=Path, default=None)
    rgbd_group.add_argument("--rotation-file", type=Path, default=None)
    rgbd_group.add_argument("--background-depth-file", type=Path, default=None)
    rgbd_group.add_argument("--background-filter-m", type=float, default=0.15)
    rgbd_group.add_argument("--center-crop-ratio", type=float, default=0.50)
    rgbd_group.add_argument(
        "--depth-strategy",
        choices=[
            "median_center_crop",
            "lower_center_crop",
            "foreground_median",
            "foreground_points_median",
        ],
        default="foreground_points_median",
    )
    rgbd_group.add_argument(
        "--depth-failure-mode",
        choices=["predict_only", "fallback_2d", "skip_frame"],
        default="predict_only",
    )
    rgbd_group.add_argument("--bev-gate", type=float, default=0.40)
    rgbd_group.add_argument(
        "--bev-axes",
        type=str,
        default="0,1",
        help="Comma-separated pair of world-space axis indices for BEV, e.g. '0,1'.",
    )
    rgbd_group.add_argument("--occlusion-iou-threshold-rgbd", type=float, default=0.40)
    rgbd_group.add_argument(
        "--larger-depth-is-farther",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    rgbd_group.add_argument("--max-occlusion-age", type=int, default=45)
    rgbd_group.add_argument("--min-score-margin", type=float, default=0.05)
    rgbd_group.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    rgbd_group.add_argument("--debug", action="store_true")

    # First parse to get normal namespace with defaults
    args = parser.parse_args(argv)

    # In-place modify defaults on the actions of the parser to SUPPRESS
    for action in parser._actions:
        action.default = argparse.SUPPRESS

    # Parse again to see which keys were explicitly provided by the user
    suppressed_args = parser.parse_args(argv)
    overrides = set(vars(suppressed_args).keys())

    # Handle alias and default for dup_iou_threshold
    if "dup_iou_threshold" in overrides:
        # User explicitly specified --dup-iou-threshold
        pass
    elif "duplicate_iou_threshold" in overrides:
        # User explicitly specified --duplicate-iou-threshold
        args.dup_iou_threshold = args.duplicate_iou_threshold
        overrides.add("dup_iou_threshold")
    else:
        # Neither was specified, use default
        args.dup_iou_threshold = DEFAULT_DUP_IOU_THRESHOLD

    args.overrides = overrides
    return args


def _profile_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_tracking_path_profile(args.path_config, args.profile)


def _tracking_config_from_args(
    args: argparse.Namespace,
    profile: dict[str, Any],
    video_path: Path | None = None,
) -> TrackingConfig:
    selected_video = (
        video_path
        or args.video
        or profile_video_path(profile, args.video_key, DEFAULT_VIDEO_PATH)
        or DEFAULT_VIDEO_PATH
    )
    weights_path = (
        args.weights
        or profile_path(profile, "weights", DEFAULT_WEIGHTS_PATH)
        or DEFAULT_WEIGHTS_PATH
    )
    mask_path = None
    if not args.no_mask:
        mask_path = (
            args.mask
            or profile_path(profile, "mask", DEFAULT_MASK_PATH)
            or DEFAULT_MASK_PATH
        )
    output_dir = (
        args.output_dir
        or profile_path(profile, "output_dir", DEFAULT_OUTPUT_DIR)
        or DEFAULT_OUTPUT_DIR
    )
    # Construct config overrides set
    cfg_overrides = set()
    raw_overrides = getattr(args, "overrides", set())
    for key in raw_overrides:
        cfg_overrides.add(key)
        if key == "no_mask":
            cfg_overrides.add("use_mask")
        elif key == "no_mask_input":
            cfg_overrides.add("mask_input_frame")
        elif key == "no_mask_iou":
            cfg_overrides.add("use_mask_iou")
        elif key == "no_smooth_boxes":
            cfg_overrides.add("smooth_boxes")
        elif key == "no_refine_boxes":
            cfg_overrides.add("refine_boxes")
        elif key == "no_low_conf_motion_gate":
            cfg_overrides.add("low_conf_motion_gate")
        elif key == "no_occlusion_aware_matching":
            cfg_overrides.add("occlusion_aware_matching")
        elif key == "no_directional_y_prior":
            cfg_overrides.add("directional_y_prior")
        elif key == "learn_identity_in_occlusion":
            cfg_overrides.add("freeze_identity_in_occlusion")
        elif key == "no_hold_occluded_box":
            cfg_overrides.add("hold_occluded_box")
        elif key == "no_identity_swap_guard":
            cfg_overrides.add("identity_swap_guard")
        elif key == "no_hidden_motion_model":
            cfg_overrides.add("hidden_motion_model")
        elif key == "fps":
            cfg_overrides.add("output_fps")
        elif key == "class_name":
            cfg_overrides.add("allowed_class_name")
        elif key == "duplicate_iou_threshold":
            cfg_overrides.add("dup_iou_threshold")
        elif key == "refine_max_gap":
            cfg_overrides.add("refine_max_gap_frames")
        elif key == "max_box_scale_change":
            cfg_overrides.add("max_box_scale_change_per_frame")

    cfg = TrackingConfig(
        video_path=selected_video,
        weights_path=weights_path,
        mask_path=mask_path,
        output_dir=output_dir,
        output_video=args.output_video,
        write_output_video=not args.no_output_video,
        annotations_json=args.annotations_json,
        coco_annotations_json=args.coco_json,
        clean_coco_annotations_json=args.clean_coco_json,
        cvat_video_xml=args.cvat_video_xml,
        labels_json=args.labels_json,
        tracker_yaml=args.tracker_yaml,
        quality_report_json=args.quality_report_json,
        quality_report_csv=args.quality_report_csv,
        device=args.device,
        half=args.half,
        tracker_type=args.tracker_type,
        start_frame=args.start_frame,
        output_fps=args.fps,
        det_conf=args.det_conf,
        track_high_conf=args.track_high_conf,
        review_conf=args.review_conf,
        adaptive_conf_step=args.adaptive_conf_step,
        conf=args.conf,
        nms_iou=args.nms_iou,
        iou=args.iou,
        track_match_iou=args.track_match_iou,
        mode=args.mode,
        imgsz=args.imgsz,
        detect_every_n_frames=args.detect_every_n_frames,
        max_raw_detections=args.max_raw_detections,
        enable_offline_smoothing=args.enable_offline_smoothing,
        class_id=args.class_id,
        allowed_class_name=args.class_name,
        use_mask=not args.no_mask,
        mask_input_frame=not args.no_mask_input,
        roi_mode=args.roi_mode,
        roi_min_cover=args.roi_min_cover,
        roi_dilate_px=args.roi_dilate_px,
        hidden_missed_frames=args.hidden_missed_frames,
        hidden_score_threshold=args.hidden_score_threshold,
        use_mask_iou=not args.no_mask_iou,
        mask_iou_max_missed=args.mask_iou_max_missed,
        mask_iou_min_area=args.mask_iou_min_area,
        max_missing_frames=args.max_missing_frames,
        match_cost_threshold=args.match_cost_threshold,
        unseen_track_cost_threshold=args.unseen_track_cost_threshold,
        lost_track_cost_threshold=args.lost_track_cost_threshold,
        lost_track_reid_appearance_threshold=(
            args.lost_track_reid_appearance_threshold
        ),
        initial_track_conf=args.initial_track_conf,
        low_conf_motion_gate=not args.no_low_conf_motion_gate,
        motion_gate_confidence=args.motion_gate_confidence,
        low_conf_max_center_jump=args.low_conf_max_center_jump,
        low_conf_max_box_jump_scale=args.low_conf_max_box_jump_scale,
        low_conf_min_iou=args.low_conf_min_iou,
        occlusion_aware_matching=not args.no_occlusion_aware_matching,
        occlusion_track_iom_threshold=args.occlusion_track_iom_threshold,
        occlusion_detection_iom_threshold=args.occlusion_detection_iom_threshold,
        occlusion_stationary_speed=args.occlusion_stationary_speed,
        occlusion_stationary_max_center_jump=(
            args.occlusion_stationary_max_center_jump
        ),
        occlusion_switch_penalty=args.occlusion_switch_penalty,
        occlusion_competitor_margin=args.occlusion_competitor_margin,
        occlusion_appearance_penalty=args.occlusion_appearance_penalty,
        occlusion_appearance_margin=args.occlusion_appearance_margin,
        directional_y_prior=not args.no_directional_y_prior,
        directional_y_penalty_weight=args.directional_y_penalty_weight,
        directional_y_velocity_epsilon_px=args.directional_y_velocity_epsilon_px,
        directional_y_margin_px=args.directional_y_margin_px,
        occlusion_stationary_lock=not args.no_occlusion_stationary_lock,
        freeze_identity_in_occlusion=not args.learn_identity_in_occlusion,
        hold_occluded_box=not args.no_hold_occluded_box,
        occlusion_hold_max_frames=args.occlusion_hold_max_frames,
        occlusion_hold_hidden_frames=args.occlusion_hold_hidden_frames,
        USE_IOU_FALLBACK=args.use_iou_fallback,
        USE_AREA_OCCLUSION_FREEZE=args.use_area_occlusion_freeze,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=(
            args.use_conditional_area_occlusion_freeze
        ),
        USE_MERGED_BOX_SPLIT=args.use_merged_box_split,
        iou_fallback_threshold=args.iou_fallback_threshold,
        area_occlusion_shrink_ratio=args.area_occlusion_shrink_ratio,
        area_occlusion_freeze_frames=args.area_occlusion_freeze_frames,
        merged_box_growth_ratio=args.merged_box_growth_ratio,
        merged_box_neighbor_distance=args.merged_box_neighbor_distance,
        merged_box_split_max_tracks=args.merged_box_split_max_tracks,
        hard_occlusion_track_iom_threshold=args.hard_occlusion_track_iom_threshold,
        hard_occlusion_detection_iom_threshold=(
            args.hard_occlusion_detection_iom_threshold
        ),
        hard_occlusion_min_frames=args.hard_occlusion_min_frames,
        hard_occlusion_recovery_frames=args.hard_occlusion_recovery_frames,
        hard_occlusion_score_threshold=args.hard_occlusion_score_threshold,
        identity_swap_guard=not args.no_identity_swap_guard,
        identity_swap_min_gain=args.identity_swap_min_gain,
        identity_swap_iom_threshold=args.identity_swap_iom_threshold,
        hidden_motion_model=not args.no_hidden_motion_model,
        hidden_velocity_alpha=args.hidden_velocity_alpha,
        hidden_acceleration_alpha=args.hidden_acceleration_alpha,
        hidden_stationary_speed=args.hidden_stationary_speed,
        hidden_motion_history=args.hidden_motion_history,
        hidden_min_motion_history=args.hidden_min_motion_history,
        hidden_stationary_displacement=args.hidden_stationary_displacement,
        hidden_moving_displacement=args.hidden_moving_displacement,
        hidden_motion_consistency=args.hidden_motion_consistency,
        hidden_stationary_lock_frames=args.hidden_stationary_lock_frames,
        hidden_max_motion_step_box_scale=args.hidden_max_motion_step_box_scale,
        dup_iou_threshold=args.dup_iou_threshold,
        dup_containment_threshold=args.dup_containment_threshold,
        dup_center_threshold=args.dup_center_threshold,
        dup_area_ratio_threshold=args.dup_area_ratio_threshold,
        default_behavior=args.default_behavior,
        smooth_boxes=not args.no_smooth_boxes,
        refine_boxes=not args.no_refine_boxes,
        refine_max_gap_frames=args.refine_max_gap,
        refine_size_jump_threshold=args.refine_size_jump_threshold,
        max_box_scale_change_per_frame=args.max_box_scale_change,
        max_box_scale_change_after_gap=args.max_box_scale_change_after_gap,
        high_conf_smooth_alpha=args.high_conf_smooth_alpha,
        mid_conf_smooth_alpha=args.mid_conf_smooth_alpha,
        low_conf_smooth_alpha=args.low_conf_smooth_alpha,
        max_frames=args.max_frames,
        visual_opacity=args.visual_opacity,
        show=args.show,
        display_inline=args.display_inline,
        overrides=cfg_overrides,
    )

    tracker_fields = set(TrackingConfig.__dataclass_fields__.keys())

    exclude_profile_keys = {
        "video_path", "weights_path", "mask_path", "output_dir",
        "output_video", "write_output_video", "annotations_json",
        "coco_annotations_json",
        "clean_coco_annotations_json", "cvat_video_xml", "labels_json",
        "tracker_yaml", "quality_report_json", "quality_report_csv",
        "overrides",
    }

    profile_mapped = {}
    for p_key, p_val in profile.items():
        if p_key == "duplicate_iou_threshold":
            profile_mapped["dup_iou_threshold"] = p_val
        elif p_key == "max_lost_frames":
            profile_mapped["max_missing_frames"] = p_val
            profile_mapped["max_lost_frames"] = p_val
        else:
            profile_mapped[p_key] = p_val

    for p_key, p_val in profile_mapped.items():
        if p_key in tracker_fields and p_key not in exclude_profile_keys:
            if p_key not in cfg.overrides:
                setattr(cfg, p_key, p_val)

    if args.no_emit_hidden_tracks:
        cfg.emit_hidden_tracks = False

    _apply_profile_overrides(cfg, args.profile_override)
    if args.no_output_video:
        cfg.write_output_video = False

    return cfg


def _video_paths_from_args(
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> list[Path | None]:
    if args.all_config_videos:
        return profile_video_paths(profile)
    return [None]


def print_tracking_summary(cfg: TrackingConfig, summary: TrackingSummary) -> None:
    print(f"[OK] input video: {cfg.video_path}")
    if cfg.write_output_video:
        print(f"[OK] video: {summary.output_video}")
    else:
        print("[OK] video: disabled (no MP4 written)")
    print(f"[OK] cvat json annotations: {summary.annotations_json}")
    print(f"[OK] cvat video xml: {summary.cvat_video_xml}")
    print(f"[OK] coco annotations: {summary.coco_annotations_json}")
    print(f"[OK] clean train coco: {summary.clean_coco_annotations_json}")
    print(f"[OK] labels: {summary.labels_json}")
    print(f"[OK] quality report json: {summary.quality_report_json}")
    print(f"[OK] quality report csv: {summary.quality_report_csv}")
    print(
        "[OK] frames="
        f"{summary.frames_written}, shapes={summary.shape_count}, "
        f"hidden={summary.hidden_shape_count}, "
        f"review={summary.review_shape_count}, "
        f"start_frame={summary.start_frame}, "
        f"source_fps={summary.source_fps:.2f}, output_fps={summary.output_fps:.2f}"
    )
    print(
        "[OK] thresholds="
        f"det_conf={cfg.det_conf:.2f}, "
        f"track_high_conf={cfg.track_high_conf:.2f}, "
        f"review_conf={cfg.review_conf:.2f}, "
        f"nms_iou={cfg.nms_iou:.2f}, "
        f"track_match_iou={cfg.track_match_iou:.2f}, "
        f"visual_opacity={cfg.visual_opacity:.2f}"
    )
    print(
        "[OK] low_conf_gate="
        f"enabled={cfg.low_conf_motion_gate}, "
        f"gate_conf={cfg.motion_gate_confidence:.2f}, "
        f"initial_track_conf={cfg.initial_track_conf:.2f}, "
        f"max_center_jump={cfg.low_conf_max_center_jump:.2f}"
    )
    print(
        "[OK] association="
        f"use_mask_iou={cfg.use_mask_iou}, "
        f"mask_iou_max_missed={cfg.mask_iou_max_missed}, "
        f"mask_iou_min_area={cfg.mask_iou_min_area}, "
        f"bbox_fallback=True"
    )
    print(
        "[OK] inference="
        f"device={cfg.device if cfg.device is not None else 'auto'}, "
        f"half={cfg.half}"
    )
    print(
        "[OK] occlusion_matching="
        f"enabled={cfg.occlusion_aware_matching}, "
        f"track_iom={cfg.occlusion_track_iom_threshold:.2f}, "
        f"detection_iom={cfg.occlusion_detection_iom_threshold:.2f}, "
        f"stationary_jump={cfg.occlusion_stationary_max_center_jump:.3f}, "
        f"stationary_lock={cfg.occlusion_stationary_lock}, "
        f"freeze_identity={cfg.freeze_identity_in_occlusion}, "
        f"hold_box={cfg.hold_occluded_box}, "
        f"hold_hidden_frames={cfg.occlusion_hold_hidden_frames}"
    )
    if tracking_rule_flags_enabled(cfg):
        print(
            "[OK] rule_flags="
            f"iou_fallback={cfg.USE_IOU_FALLBACK}, "
            f"area_freeze={cfg.USE_AREA_OCCLUSION_FREEZE}, "
            "conditional_area_freeze="
            f"{cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE}, "
            f"merged_split={cfg.USE_MERGED_BOX_SPLIT}, "
            f"iou_threshold={cfg.iou_fallback_threshold:.2f}, "
            f"shrink_ratio={cfg.area_occlusion_shrink_ratio:.2f}, "
            f"growth_ratio={cfg.merged_box_growth_ratio:.2f}"
        )
        telemetry = get_telemetry_summary(summary)
        print(
            "[OK] telemetry="
            f"hard_merges_triggered={telemetry['hard_merges_triggered']}, "
            "detections_intentionally_ignored="
            f"{telemetry['detections_intentionally_ignored']}, "
            f"recovery_frames_applied={telemetry['recovery_frames_applied']}"
        )
    print(
        "[OK] directional_y_prior="
        f"enabled={cfg.directional_y_prior}, "
        f"penalty={cfg.directional_y_penalty_weight:.3f}, "
        f"velocity_epsilon_px={cfg.directional_y_velocity_epsilon_px:.1f}, "
        f"margin_px={cfg.directional_y_margin_px:.1f}"
    )
    print(
        "[OK] identity_swap_guard="
        f"enabled={cfg.identity_swap_guard}, "
        f"min_gain={cfg.identity_swap_min_gain:.3f}, "
        f"iom={cfg.identity_swap_iom_threshold:.2f}"
    )
    print(
        "[OK] hidden_motion="
        f"enabled={cfg.hidden_motion_model}, "
        f"stationary_speed={cfg.hidden_stationary_speed:.3f}, "
        f"history={cfg.hidden_motion_history}, "
        f"min_history={cfg.hidden_min_motion_history}, "
        f"moving_disp={cfg.hidden_moving_displacement:.3f}, "
        f"lock_frames={cfg.hidden_stationary_lock_frames}, "
        f"max_step_scale={cfg.hidden_max_motion_step_box_scale:.2f}"
    )


def _build_rgbd_config(
    args: argparse.Namespace,
    tracking_config: TrackingConfig,
) -> object:
    """Construct an ``RGBDTrackingConfig`` from CLI arguments."""
    from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig

    if args.depth_video is None:
        raise ValueError("--depth-video is required when --rgbd is set.")
    if args.depth_scale_file is None:
        raise ValueError("--depth-scale-file is required when --rgbd is set.")
    if args.inverse_intrinsic_file is None:
        raise ValueError("--inverse-intrinsic-file is required when --rgbd is set.")
    if args.rotation_file is None:
        raise ValueError("--rotation-file is required when --rgbd is set.")

    bev_axes_parts = args.bev_axes.split(",")
    if len(bev_axes_parts) != 2:
        raise ValueError(
            f"--bev-axes must be two comma-separated ints, got: {args.bev_axes}"
        )
    bev_axes = (int(bev_axes_parts[0].strip()), int(bev_axes_parts[1].strip()))

    return RGBDTrackingConfig(
        tracking_config=tracking_config,
        depth_video_path=args.depth_video,
        times_path=args.times_file,
        depth_scale_path=args.depth_scale_file,
        inverse_intrinsic_path=args.inverse_intrinsic_file,
        rotation_path=args.rotation_file,
        background_depth_path=args.background_depth_file,
        background_filter_m=args.background_filter_m,
        center_crop_ratio=args.center_crop_ratio,
        depth_strategy=args.depth_strategy,
        depth_failure_mode=args.depth_failure_mode,
        bev_association_gate_m=args.bev_gate,
        bev_axes=bev_axes,
        occlusion_iou_threshold=args.occlusion_iou_threshold_rgbd,
        larger_depth_is_farther=args.larger_depth_is_farther,
        max_occlusion_age=args.max_occlusion_age,
        min_score_margin=args.min_score_margin,
        render=args.render,
        debug=args.debug,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args(argv)
    if args.all_config_videos and any(
        path is not None
        for path in (
            args.output_video,
            args.annotations_json,
            args.coco_json,
            args.clean_coco_json,
            args.cvat_video_xml,
            args.labels_json,
            args.tracker_yaml,
            args.quality_report_json,
            args.quality_report_csv,
        )
    ):
        raise ValueError(
            "Do not use single-output file arguments with --all-config-videos."
        )

    profile = _profile_from_args(args)
    summaries: list[TrackingSummary] = []
    for video_path in _video_paths_from_args(args, profile):
        cfg = _tracking_config_from_args(args, profile, video_path)
        if args.display_inline and not cfg.write_output_video:
            raise ValueError("--display-inline requires MP4 output.")

        if args.rgbd:
            from pig_behavior.tracking.rgbd.runner_rgbd import run_rgbd_tracking

            rgbd_cfg = _build_rgbd_config(args, cfg)
            summary = run_rgbd_tracking(rgbd_cfg)
        else:
            summary = run_tracking(cfg)

        summaries.append(summary)
        print_tracking_summary(cfg, summary)
        if args.display_inline:
            display_tracked_video(summary.output_video)

    if len(summaries) > 1:
        print(f"[OK] processed videos: {len(summaries)}")
    return 0


__all__ = [
    "_profile_from_args",
    "_tracking_config_from_args",
    "_video_paths_from_args",
    "main",
    "parse_args",
    "print_tracking_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
