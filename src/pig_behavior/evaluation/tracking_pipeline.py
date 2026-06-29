"""Run tracking predictions and evaluate them against CVAT video XML labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.benchmarking import (  # noqa: E402
    run_tracking_detector_benchmark,
    run_tracking_rule_benchmark,
)
from pig_behavior.evaluation.tracking.config import (  # noqa: E402
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.pipeline import (  # noqa: E402
    run_pipeline,
)
from pig_behavior.evaluation.tracking_metrics import (  # noqa: E402
    DETECTOR_WEIGHTS_V8,
    DETECTOR_WEIGHTS_V26,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
)
from pig_behavior.tracking_path_config import (  # noqa: E402
    DEFAULT_TRACKING_PATH_CONFIG,
    load_tracking_path_profile,
    profile_path,
    profile_video_path,
    profile_video_paths,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run pig tracking prediction and evaluate against CVAT XML labels.",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-key", type=str, default=None)
    parser.add_argument(
        "--all-config-videos",
        action="store_true",
        help="Evaluate every video listed in the selected tracking path profile.",
    )
    parser.add_argument(
        "--path-config",
        type=Path,
        default=DEFAULT_TRACKING_PATH_CONFIG,
        help="JSON path profile file for fast video/dir switching.",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--gt-xml", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--weights-v26",
        type=Path,
        default=DETECTOR_WEIGHTS_V26,
        help="YOLOv26 detector weights used only by --benchmark-detectors.",
    )
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument(
        "--gap-tolerance-frames",
        type=int,
        default=15,
        help=(
            "Merge continuity gaps up to this many frames for gap-tolerant "
            "fragment/tracklet metrics. Use 0 for strict frame-by-frame scoring."
        ),
    )
    parser.add_argument(
        "--run-missing-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the package tracking annotator for videos without prediction XML. "
            "Use --no-run-missing-tracker to evaluate only existing predictions."
        ),
    )
    parser.add_argument(
        "--force-track",
        action="store_true",
        help="Always rerun tracker before evaluation, even if prediction XML exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO inference, for example '0' or 'cpu'.",
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--enable-offline-smoothing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tracker offline smoothing for generated prediction XML.",
    )
    parser.add_argument(
        "--identity-swap-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tracker identity swap guard for generated prediction XML.",
    )
    parser.add_argument(
        "--smooth-boxes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tracker temporal box smoothing for generated prediction XML.",
    )
    parser.add_argument(
        "--refine-boxes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tracker temporal box refinement for generated prediction XML.",
    )
    parser.add_argument("--use-iou-fallback", action="store_true")
    parser.add_argument("--use-area-occlusion-freeze", action="store_true")
    parser.add_argument(
        "--use-conditional-area-occlusion-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-merged-box-split", action="store_true")
    parser.add_argument(
        "--benchmark-rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run all 16 tracking rule flag combinations. This is the default when "
            "executing tracking_pipeline.py directly. Use --no-benchmark-rules "
            "for a single evaluation run with the explicit flags above."
        ),
    )
    parser.add_argument(
        "--benchmark-detectors",
        action="store_true",
        help=(
            "Run the full rule benchmark separately for YOLOv8 and YOLOv26, "
            "using isolated prediction/output folders."
        ),
    )
    parser.add_argument(
        "--tracking-mode",
        type=str,
        choices=["realtime", "bytetrack_raw", "hybrid_bytetrack", "bytetrack", "gt_export"],
        default="hybrid_bytetrack",
        help=(
            "Mode to run the tracker in "
            "(realtime, bytetrack_raw, hybrid_bytetrack, or legacy aliases)."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--profile-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a TrackingConfig profile field for this run. "
            "May be repeated; VALUE is parsed as bool, int, float, or string."
        ),
    )
    return parser.parse_args(argv)


def parse_profile_override_value(raw_value: str) -> object:
    normalized = raw_value.strip()
    lowered = normalized.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        return raw_value


def parse_profile_overrides(
    raw_overrides: list[str],
    allowed_fields: set[str],
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"--profile-override must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in allowed_fields:
            raise ValueError(f"Unknown TrackingConfig override: {key}")
        parsed[key] = parse_profile_override_value(raw_value)
    return parsed


def config_from_args(args: argparse.Namespace) -> TrackingEvaluationPipelineConfig:
    """Build config from parsed args."""
    profile = load_tracking_path_profile(args.path_config, args.profile)
    video_path = args.video
    video_paths = None
    if args.all_config_videos:
        video_paths = profile_video_paths(profile)
    elif video_path is not None:
        video_path_str = str(video_path)
        if "," in video_path_str:
            video_paths = []
            for part in video_path_str.split(","):
                part = part.strip()
                try:
                    resolved = profile_video_path(profile, part)
                    if resolved:
                        video_paths.append(resolved)
                except Exception:
                    direct = Path(part)
                    if direct.exists() and direct.is_file():
                        video_paths.append(direct)
            video_path = None
    elif video_path is None and args.video_key:
        video_path = profile_video_path(profile, args.video_key)

    from pig_behavior.tracking.config import TrackingConfig
    tracker_fields = set(TrackingConfig.__dataclass_fields__.keys())
    profile_overrides = {k: v for k, v in profile.items() if k in tracker_fields}
    profile_overrides.update(
        parse_profile_overrides(args.profile_override, tracker_fields)
    )

    tracking_mode = args.tracking_mode
    if "mode" in profile_overrides:
        # If --tracking-mode was not explicitly passed on CLI, use profile's mode
        if not any(arg.startswith("--tracking-mode") for arg in sys.argv):
            tracking_mode = profile_overrides["mode"]

    # Exclude arguments explicitly passed by run_tracker_for_pair or tracking_rule_overrides
    exclude_keys = [
        "video_path", "weights_path", "mask_path", "output_dir",
        "max_frames", "display_inline", "show", "device", "half", "mode"
    ]
    for key in exclude_keys:
        profile_overrides.pop(key, None)
    postprocess_overrides = {
        "enable_offline_smoothing": args.enable_offline_smoothing,
        "identity_swap_guard": args.identity_swap_guard,
        "smooth_boxes": args.smooth_boxes,
        "refine_boxes": args.refine_boxes,
    }
    profile_overrides.update(
        {
            key: value
            for key, value in postprocess_overrides.items()
            if value is not None
        }
    )

    return TrackingEvaluationPipelineConfig(
        video_path=video_path,
        video_paths=video_paths,
        gt_xml=args.gt_xml,
        gt_dir=args.gt_dir
        or profile_path(profile, "gt_dir", TRACKING_GT_DIR)
        or TRACKING_GT_DIR,
        video_dir=args.video_dir
        or profile_path(profile, "video_dir", VIDEO_DIR)
        or VIDEO_DIR,
        prediction_root=args.prediction_root
        or profile_path(profile, "prediction_root", PREDICTION_ROOT)
        or PREDICTION_ROOT,
        output_root=args.output_root
        or profile_path(profile, "evaluation_output_root", EVAL_OUTPUT_ROOT)
        or EVAL_OUTPUT_ROOT,
        weights_path=args.weights
        or profile_path(profile, "weights", DETECTOR_WEIGHTS_V8)
        or DETECTOR_WEIGHTS_V8,
        weights_v26_path=args.weights_v26,
        mask_path=args.mask or profile_path(profile, "mask", None),
        iou_threshold=args.iou_threshold,
        include_hidden=args.include_hidden,
        gap_tolerance_frames=args.gap_tolerance_frames,
        run_missing_tracker=args.run_missing_tracker,
        force_track=args.force_track,
        max_frames=args.max_frames,
        device=args.device,
        half=args.half,
        USE_IOU_FALLBACK=args.use_iou_fallback,
        USE_AREA_OCCLUSION_FREEZE=args.use_area_occlusion_freeze,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=(
            args.use_conditional_area_occlusion_freeze
        ),
        USE_MERGED_BOX_SPLIT=args.use_merged_box_split,
        tracking_mode=tracking_mode,
        profile_overrides=profile_overrides,
    )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config = config_from_args(args)
    if args.benchmark_detectors:
        summary_df, detailed_metrics_df, output_dir = run_tracking_detector_benchmark(
            config,
        )
        print("[detector-benchmark-summary]")
        print(summary_df.to_string(index=False))
        print("[detector-benchmark-detailed-metrics]")
        print(detailed_metrics_df.to_string(index=False))
        print("[detector-benchmark-output]", output_dir)
        return 0

    if args.benchmark_rules:
        summary_df, detailed_metrics_df, output_dir = run_tracking_rule_benchmark(
            config,
        )
        print("[benchmark-summary]")
        print(summary_df.to_string(index=False))
        print("[benchmark-detailed-metrics]")
        print(detailed_metrics_df.to_string(index=False))
        print("[benchmark-output]", output_dir)
        return 0

    asset_df, metrics_df, output_dir = run_pipeline(config)
    print("[assets]")
    print(asset_df.to_string(index=False))
    print("[metrics]")
    print(metrics_df.to_string(index=False))
    print("[output]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
