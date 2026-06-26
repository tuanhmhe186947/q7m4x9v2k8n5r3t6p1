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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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
    group = parser.add_mutually_exclusive_group(required=True)
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
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the same conservative offline smoothing/refinement used by "
            "the best3 Roboflow benchmark for hybrid_bytetrack runs."
        ),
    )
    return parser.parse_known_args()


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


def main() -> int:
    args, pipeline_extra_args = parse_args()
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

    script_path = PROJECT_ROOT / "src" / "pig_behavior" / "evaluation" / "tracking_pipeline.py"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_prediction_root = PROJECT_ROOT / "outputs" / "pred" / args.mode / run_timestamp
    default_output_root = PROJECT_ROOT / "outputs" / "eval" / args.mode / run_timestamp
    has_prediction_root = "--prediction-root" in pipeline_extra_args
    has_output_root = "--output-root" in pipeline_extra_args
    has_force_track_arg = "--force-track" in pipeline_extra_args
    has_benchmark_arg = any(arg in {"--benchmark-rules", "--no-benchmark-rules"} for arg in pipeline_extra_args)
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
    if not has_benchmark_arg:
        cmd.append("--benchmark-rules")
    if "--benchmark-detectors" not in pipeline_extra_args:
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
    if not has_prediction_root:
        cmd.extend(["--prediction-root", str(prediction_root)])
    if not has_output_root:
        cmd.extend(["--output-root", str(output_root)])
    cmd.extend(pipeline_extra_args)

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"\nBenchmark output: {output_root}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
