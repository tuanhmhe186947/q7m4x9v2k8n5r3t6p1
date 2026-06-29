#!/usr/bin/env python3
"""Canonical entrypoint for tracking evaluation runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.pipeline import find_gt_xml_for_video  # noqa: E402
from pig_behavior.tracking_path_config import (  # noqa: E402
    load_tracking_path_profile,
    profile_video_path,
    profile_video_paths,
)

BASE_EVAL_CONFIG: dict[str, object] = {
    "USE_IOU_FALLBACK": False,
    "USE_AREA_OCCLUSION_FREEZE": False,
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
    "USE_MERGED_BOX_SPLIT": False,
    "enable_offline_smoothing": True,
    "identity_swap_guard": True,
    "smooth_boxes": True,
    "refine_boxes": True,
}

RULE_BENCHMARK_OVERRIDE_KEYS = {
    "USE_IOU_FALLBACK",
    "USE_AREA_OCCLUSION_FREEZE",
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE",
    "USE_MERGED_BOX_SPLIT",
}

EVAL_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "base": dict(BASE_EVAL_CONFIG),
    "smooth_conservative": {
        **BASE_EVAL_CONFIG,
        "high_conf_smooth_alpha": 0.85,
        "mid_conf_smooth_alpha": 0.65,
        "low_conf_smooth_alpha": 0.45,
    },
    "smooth_responsive": {
        **BASE_EVAL_CONFIG,
        "high_conf_smooth_alpha": 0.65,
        "mid_conf_smooth_alpha": 0.45,
        "low_conf_smooth_alpha": 0.25,
    },
    "smooth_det020_loose": {
        **BASE_EVAL_CONFIG,
        "det_conf": 0.20,
        "low_conf_max_center_jump": 0.10,
        "low_conf_max_box_jump_scale": 2.00,
        "max_raw_detections": 64,
    },
    "smooth_responsive_det020": {
        **BASE_EVAL_CONFIG,
        "high_conf_smooth_alpha": 0.65,
        "mid_conf_smooth_alpha": 0.45,
        "low_conf_smooth_alpha": 0.25,
        "det_conf": 0.20,
    },
    # Backward-compatible alias for older long-form command lines.
    "iou0_area0_condarea0_merge0_smooth_det020_loose_motion": {
        **BASE_EVAL_CONFIG,
        "det_conf": 0.20,
        "low_conf_max_center_jump": 0.10,
        "low_conf_max_box_jump_scale": 2.00,
        "max_raw_detections": 64,
    },
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run tracking evaluation on one or more videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_tracking.py -a
  python scripts/evaluate_tracking.py -v data/videos/Pigs291119_000263_30fps.mp4
  python scripts/evaluate_tracking.py -v data/videos/Pigs291119_000263_30fps.mp4 --mode realtime
""",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-v", "--video", type=str, help="Comma-separated names, paths, keys, or aliases.")
    group.add_argument("-a", "--all-videos", action="store_true", help="Run evaluation on all configured videos.")
    parser.add_argument("-p", "--profile", type=str, default=None, help="Path profile name.")
    parser.add_argument("--path-config", type=str, default=None, help="Custom tracking_paths.json path.")
    parser.add_argument(
        "--mode",
        choices=["realtime", "bytetrack_raw", "hybrid_bytetrack", "bytetrack", "gt_export"],
        default="hybrid_bytetrack",
    )
    parser.add_argument("--skip-missing-gt", action="store_true")
    parser.add_argument("--fail-missing-gt", action="store_true")
    parser.add_argument(
        "--single-config",
        action="store_true",
        help=(
            "Run one exact evaluation config. This is now the default; the flag "
            "is kept for older command lines."
        ),
    )
    parser.add_argument(
        "--benchmark-compatible",
        action="store_true",
        help=(
            "Run the legacy benchmark-compatible detector/rule matrix. "
            "Use this for benchmark-suite reproduction, not single-config reporting."
        ),
    )
    parser.add_argument(
        "--benchmark-rules",
        action="store_true",
        help=(
            "Enable benchmark rule expansion for the selected run(s) without "
            "forcing the legacy benchmark-compatible bundle."
        ),
    )
    parser.add_argument(
        "--benchmark-detectors",
        action="store_true",
        help=(
            "Enable benchmark detector expansion for the selected run(s) without "
            "forcing the legacy benchmark-compatible bundle."
        ),
    )
    parser.add_argument(
        "--eval-config",
        action="append",
        default=None,
        help=(
            "Run a named direct evaluation config. Can be repeated. "
            "May be combined with --benchmark-rules or --benchmark-detectors."
        ),
    )
    parser.add_argument(
        "--list-eval-configs",
        action="store_true",
        help="List named direct evaluation configs and exit.",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the same conservative offline smoothing/refinement used by "
            "the best3 Roboflow benchmark for hybrid_bytetrack runs."
        ),
    )
    args, extra_args = parser.parse_known_args(argv)
    if not args.list_eval_configs and not args.video and not args.all_videos:
        parser.error("one of the arguments -v/--video -a/--all-videos is required")
    return args, extra_args


def _resolve_videos(args: argparse.Namespace, profile: dict[str, object]) -> list[Path]:
    if args.all_videos:
        return profile_video_paths(profile)

    video_paths: list[Path] = []
    for key in (part.strip() for part in args.video.split(",") if part.strip()):
        try:
            video_paths.append(profile_video_path(profile, key))
        except Exception:
            direct = Path(key)
            if direct.exists() and direct.is_file():
                video_paths.append(direct.resolve())
            else:
                raise
    return video_paths


def _filter_videos_with_gt(video_paths: list[Path], gt_dir: Path) -> tuple[list[Path], list[Path]]:
    valid: list[Path] = []
    skipped: list[Path] = []
    for video_path in video_paths:
        if find_gt_xml_for_video(video_path, gt_dir) is not None:
            valid.append(video_path)
        else:
            skipped.append(video_path)
    return valid, skipped


def _extra_arg_value(extra_args: list[str], name: str) -> str | None:
    try:
        return extra_args[extra_args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _selected_eval_configs(raw_configs: list[str] | None) -> list[str]:
    if not raw_configs:
        return []
    selected: list[str] = []
    for raw_config in raw_configs:
        selected.extend(part.strip() for part in raw_config.split(",") if part.strip())
    unknown = sorted(set(selected) - set(EVAL_CONFIG_OVERRIDES))
    if unknown:
        raise ValueError(f"Unknown eval config(s): {', '.join(unknown)}")
    return selected


def _list_eval_configs() -> None:
    print("Available eval configs:")
    for name, overrides in EVAL_CONFIG_OVERRIDES.items():
        summary = ", ".join(f"{key}={value}" for key, value in overrides.items())
        print(f" - {name}: {summary}")


def _format_profile_override_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def main() -> int:
    args, pipeline_extra_args = parse_args()
    if args.list_eval_configs:
        _list_eval_configs()
        return 0
    path_config = Path(args.path_config) if args.path_config else None
    try:
        profile = load_tracking_path_profile(path_config, args.profile)
        video_paths = _resolve_videos(args, profile)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gt_dir = Path(profile.get("gt_dir") or PROJECT_ROOT / "data" / "annotations" / "tracking")
    valid_video_paths, skipped_video_paths = _filter_videos_with_gt(video_paths, gt_dir)
    if skipped_video_paths:
        if args.fail_missing_gt:
            print("Error: Missing GT XML for requested videos.", file=sys.stderr)
            for video_path in skipped_video_paths:
                print(f" - {video_path.name}", file=sys.stderr)
            return 1
        label = "Skipping" if args.skip_missing_gt else "Warning"
        print(f"{label}: videos without GT XML:")
        for video_path in skipped_video_paths:
            print(f" - {video_path.name}")

    if not valid_video_paths:
        print("Error: No videos matching GT XML found.", file=sys.stderr)
        return 1
    try:
        selected_eval_configs = _selected_eval_configs(args.eval_config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if selected_eval_configs and args.benchmark_compatible:
        print(
            "Error: --eval-config evaluates exact configs and cannot be combined "
            "with --benchmark-compatible.",
            file=sys.stderr,
        )
        return 1

    script_path = PROJECT_ROOT / "src" / "pig_behavior" / "evaluation" / "tracking_pipeline.py"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_prediction_root = PROJECT_ROOT / "outputs" / "pred" / args.mode / run_timestamp
    default_output_root = PROJECT_ROOT / "outputs" / "eval" / args.mode / run_timestamp
    has_prediction_root = "--prediction-root" in pipeline_extra_args
    has_output_root = "--output-root" in pipeline_extra_args
    has_force_track_arg = "--force-track" in pipeline_extra_args
    has_benchmark_arg = any(arg in {"--benchmark-rules", "--no-benchmark-rules"} for arg in pipeline_extra_args)
    has_benchmark_detectors_arg = "--benchmark-detectors" in pipeline_extra_args
    has_smoothing_arg = any(
        arg
        in {
            "--enable-offline-smoothing",
            "--no-enable-offline-smoothing",
            "--identity-swap-guard",
            "--no-identity-swap-guard",
            "--smooth-boxes",
            "--no-smooth-boxes",
            "--refine-boxes",
            "--no-refine-boxes",
        }
        for arg in pipeline_extra_args
    )

    prediction_root = Path(_extra_arg_value(pipeline_extra_args, "--prediction-root") or default_prediction_root)
    output_root = Path(_extra_arg_value(pipeline_extra_args, "--output-root") or default_output_root)

    cmd = [
        sys.executable,
        str(script_path),
        "--video",
        ",".join(str(video_path) for video_path in valid_video_paths),
        "--tracking-mode",
        args.mode,
    ]
    if args.profile:
        cmd.extend(["--profile", args.profile])
    if args.path_config:
        cmd.extend(["--path-config", args.path_config])
    if not has_force_track_arg:
        cmd.append("--force-track")
    enable_benchmark_rules = args.benchmark_compatible or args.benchmark_rules
    enable_benchmark_detectors = (
        args.benchmark_compatible or args.benchmark_detectors
    )
    if enable_benchmark_rules and not has_benchmark_arg:
        cmd.append("--benchmark-rules")
    if enable_benchmark_detectors and not has_benchmark_detectors_arg:
        cmd.append("--benchmark-detectors")
    if args.mode in {"hybrid_bytetrack", "bytetrack", "gt_export"} and args.smooth and not has_smoothing_arg:
        cmd.extend(
            [
                "--enable-offline-smoothing",
                "--identity-swap-guard",
                "--smooth-boxes",
                "--refine-boxes",
            ]
        )
    commands: list[tuple[str | None, list[str], Path]] = []
    if selected_eval_configs:
        for config_name in selected_eval_configs:
            config_cmd = list(cmd)
            for key, value in EVAL_CONFIG_OVERRIDES[config_name].items():
                if (
                    (enable_benchmark_rules or enable_benchmark_detectors)
                    and key in RULE_BENCHMARK_OVERRIDE_KEYS
                ):
                    continue
                config_cmd.extend(
                    [
                        "--profile-override",
                        f"{key}={_format_profile_override_value(value)}",
                    ]
                )
            config_output_root = output_root / config_name
            config_prediction_root = prediction_root / config_name
            if not has_prediction_root:
                config_cmd.extend(["--prediction-root", str(config_prediction_root)])
            if not has_output_root:
                config_cmd.extend(["--output-root", str(config_output_root)])
            config_cmd.extend(pipeline_extra_args)
            commands.append((config_name, config_cmd, config_output_root))
    else:
        if not has_prediction_root:
            cmd.extend(["--prediction-root", str(prediction_root)])
        if not has_output_root:
            cmd.extend(["--output-root", str(output_root)])
        cmd.extend(pipeline_extra_args)
        commands.append((None, cmd, output_root))

    return_code = 0
    for config_name, run_cmd, run_output_root in commands:
        label = f" [{config_name}]" if config_name else ""
        print(f"Command{label}: {' '.join(run_cmd)}")
        result = subprocess.run(run_cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            return_code = result.returncode
            break
        print(f"\nBenchmark output{label}: {run_output_root}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
