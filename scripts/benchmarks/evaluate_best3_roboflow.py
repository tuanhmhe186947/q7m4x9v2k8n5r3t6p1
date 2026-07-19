#!/usr/bin/env python3
"""Canonical entrypoint for the fixed best-3 Roboflow benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.assets import DETECTOR_WEIGHTS_V8  # noqa: E402
from pig_behavior.evaluation.tracking.config import TrackingEvaluationPipelineConfig  # noqa: E402
from pig_behavior.evaluation.tracking.pipeline import run_pipeline  # noqa: E402
from pig_behavior.output_layout import evaluation_root, prediction_root  # noqa: E402

BEST3_VIDEOS = (
    "Pigs281119_000085_30fps.mp4",
    "Pigs291119_000263_30fps.mp4",
    "Pigs291119_000302_30fps.mp4",
)
TRACKING_MODE = "hybrid_bytetrack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed best-3 tracking/evaluation benchmark.",
    )
    parser.add_argument(
        "--weight",
        "--weights",
        dest="weight",
        type=Path,
        default=PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8_roboflow.pt",
        help="Detector weight path.",
    )
    parser.add_argument(
        "--tag",
        default="best3-roboflow",
        help="Experiment tag under outputs/pred and outputs/eval.",
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument("--smooth", dest="smooth", action="store_true")
    smoothing_group.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.set_defaults(smooth=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smooth_tag = "smooth" if args.smooth else "nosmooth"
    weight_path = args.weight.resolve()
    video_paths = [PROJECT_ROOT / "data" / "videos" / name for name in BEST3_VIDEOS]
    pred_root = prediction_root(PROJECT_ROOT, args.tag, TRACKING_MODE, smooth_tag)
    eval_root = evaluation_root(PROJECT_ROOT, args.tag, TRACKING_MODE, smooth_tag)
    profile_overrides = {
        "enable_offline_smoothing": args.smooth,
        "identity_swap_guard": args.smooth,
        "smooth_boxes": args.smooth,
        "refine_boxes": args.smooth,
    }
    config = TrackingEvaluationPipelineConfig(
        video_paths=video_paths,
        gt_dir=PROJECT_ROOT / "data" / "annotations" / "tracking",
        video_dir=PROJECT_ROOT / "data" / "videos",
        prediction_root=pred_root,
        output_root=eval_root,
        weights_path=weight_path if args.weight else DETECTOR_WEIGHTS_V8,
        detector_name="yolov8",
        mask_path=PROJECT_ROOT / "data" / "annotations" / "scene" / "mask.png",
        iou_threshold=0.5,
        gap_tolerance_frames=15,
        run_missing_tracker=True,
        force_track=True,
        max_frames=None,
        device=None,
        half=False,
        USE_IOU_FALLBACK=False,
        USE_AREA_OCCLUSION_FREEZE=False,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=False,
        USE_MERGED_BOX_SPLIT=False,
        tracking_mode=TRACKING_MODE,
        profile_overrides=profile_overrides,
    )
    asset_df, metrics_df, output_dir = run_pipeline(config)
    print(asset_df.to_string(index=False))
    print(metrics_df.to_string(index=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
