from __future__ import annotations

import argparse
import sys
from pathlib import Path

BEST3_VIDEOS = (
    "Pigs281119_000085_30fps.mp4",
    "Pigs291119_000263_30fps.mp4",
    "Pigs291119_000302_30fps.mp4",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run best3 tracking/evaluation for YOLOv8 Roboflow weights.",
    )
    parser.add_argument(
        "--weight",
        "--weights",
        dest="weight",
        type=Path,
        default=root / "models" / "detector" / "pig_detector_yolov8_roboflow.pt",
        help="Path to detector weight file.",
    )
    parser.add_argument(
        "--tag",
        default="yolov8_roboflow_best3",
        help="Output tag under outputs/evaluation/weight_ablation.",
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        "--smooth",
        dest="smooth",
        action="store_true",
        help="Enable offline smoothing post-processing (default).",
    )
    smoothing_group.add_argument(
        "--no-smooth",
        dest="smooth",
        action="store_false",
        help="Disable offline smoothing post-processing.",
    )
    parser.set_defaults(smooth=True)
    return parser.parse_args()


def main() -> int:
    root = project_root()
    src_root = root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pig_behavior.evaluation.tracking.assets import (  # noqa: WPS433
        DETECTOR_WEIGHTS_V8,
    )
    from pig_behavior.evaluation.tracking.config import (  # noqa: WPS433
        TrackingEvaluationPipelineConfig,
    )
    from pig_behavior.evaluation.tracking.pipeline import run_pipeline  # noqa: WPS433

    args = parse_args()
    weight_path = args.weight.resolve()
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    video_paths = [root / "data" / "videos" / name for name in BEST3_VIDEOS]
    output_tag = args.tag
    prediction_root = (
        root
        / "outputs"
        / "id_tracking"
        / "weight_ablation"
        / output_tag
        / "hybrid_bytetrack"
        / "yolov8_roboflow"
        / "iou0_area0_condarea0_merge0"
    )
    output_root = (
        root
        / "outputs"
        / "evaluation"
        / "tracking_metrics"
        / "weight_ablation"
        / output_tag
        / "hybrid_bytetrack"
        / "yolov8_roboflow"
        / "iou0_area0_condarea0_merge0"
    )

    profile_overrides = {}
    if args.smooth:
        profile_overrides.update(
            {
                "enable_offline_smoothing": True,
                "identity_swap_guard": True,
                "smooth_boxes": True,
                "refine_boxes": True,
            }
        )

    config = TrackingEvaluationPipelineConfig(
        video_paths=video_paths,
        gt_dir=root / "data" / "annotations" / "tracking",
        video_dir=root / "data" / "videos",
        prediction_root=prediction_root,
        output_root=output_root,
        weights_path=weight_path if args.weight else DETECTOR_WEIGHTS_V8,
        detector_name="yolov8",
        mask_path=root / "data" / "annotations" / "scene" / "mask.png",
        iou_threshold=0.5,
        include_hidden=False,
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
        tracking_mode="hybrid_bytetrack",
        profile_overrides=profile_overrides,
    )

    asset_df, metrics_df, output_dir = run_pipeline(config)
    print(asset_df.to_string(index=False))
    print(metrics_df.to_string(index=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
