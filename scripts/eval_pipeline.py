#!/usr/bin/env python3
"""Run tracking evaluation and benchmark one or more videos."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.evaluation.tracking.pipeline import find_gt_xml_for_video
from pig_behavior.tracking_path_config import (
    load_tracking_path_profile,
    profile_video_path,
    profile_video_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tracking evaluation and benchmarking on one or more videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_pipeline.py -a
  python scripts/eval_pipeline.py -v data/videos/Pigs291119_000263_30fps.mp4
  python scripts/eval_pipeline.py -v data/videos/Pigs291119_000263_30fps.mp4 --mode realtime
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-v",
        "--video",
        type=str,
        help="Comma-separated video names, paths, or keys/aliases.",
    )
    group.add_argument(
        "-a",
        "--all-videos",
        action="store_true",
        help="Run evaluation on all videos resolved by the path profile.",
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="Path profile name from configs/tracking_paths.json (default active).",
    )
    parser.add_argument(
        "--path-config",
        type=str,
        default=None,
        help="Path to custom configs/tracking_paths.json file.",
    )
    parser.add_argument(
        "--mode",
        choices=["realtime", "bytetrack_raw", "hybrid_bytetrack", "bytetrack", "gt_export"],
        default="hybrid_bytetrack",
        help="Tracking mode used for benchmark generation (default: hybrid_bytetrack).",
    )
    parser.add_argument(
        "--skip-missing-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos without matching GT XML instead of failing the batch.",
    )
    parser.add_argument(
        "--fail-missing-gt",
        action="store_true",
        help="Fail immediately if any requested video is missing GT XML.",
    )
    return parser.parse_known_args()


def _resolve_videos(args, profile):
    video_paths: list[Path] = []
    if args.all_videos:
        video_paths = profile_video_paths(profile)
        print(f"Resolved all videos from profile: {[p.name for p in video_paths]}")
    elif args.video:
        for key in [part.strip() for part in args.video.split(",")]:
            try:
                resolved = profile_video_path(profile, key)
                if resolved:
                    video_paths.append(resolved)
                    continue
            except Exception as exc:
                direct = Path(key)
                if direct.exists() and direct.is_file():
                    video_paths.append(direct)
                    continue
                print(
                    f"Warning: Could not resolve video '{key}': {exc}",
                    file=sys.stderr,
                )
    return video_paths


def _filter_videos_with_gt(video_paths: list[Path], gt_dir: Path):
    valid_video_paths: list[Path] = []
    skipped_video_paths: list[Path] = []
    for video_path in video_paths:
        if find_gt_xml_for_video(video_path, gt_dir) is not None:
            valid_video_paths.append(video_path)
        else:
            skipped_video_paths.append(video_path)
    return valid_video_paths, skipped_video_paths


def _extra_arg_value(extra_args: list[str], name: str) -> str | None:
    """Return the value passed after a forwarded CLI option."""
    try:
        return extra_args[extra_args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def main():
    args, pipeline_extra_args = parse_args()

    path_config = Path(args.path_config) if args.path_config else None
    try:
        profile = load_tracking_path_profile(path_config, args.profile)
    except Exception as exc:
        print(f"Error loading path profile: {exc}", file=sys.stderr)
        sys.exit(1)

    video_paths = _resolve_videos(args, profile)
    if not video_paths:
        print("Error: No valid videos resolved.", file=sys.stderr)
        sys.exit(1)

    gt_dir = Path(
        profile.get("gt_dir") or PROJECT_ROOT / "data" / "annotations" / "tracking"
    )
    valid_video_paths, skipped_video_paths = _filter_videos_with_gt(video_paths, gt_dir)

    if skipped_video_paths:
        if args.fail_missing_gt:
            print("Error: Missing GT XML for requested videos.", file=sys.stderr)
            for video_path in skipped_video_paths:
                print(f"  - {video_path.name}", file=sys.stderr)
            sys.exit(1)
        if args.skip_missing_gt:
            print("Skipping videos without GT XML:")
            for video_path in skipped_video_paths:
                print(f"  - {video_path.name}")
        else:
            print("Warning: Missing GT XML for some requested videos:")
            for video_path in skipped_video_paths:
                print(f"  - {video_path.name}")

    if not valid_video_paths:
        print("Error: No videos with matching GT XML were found.", file=sys.stderr)
        sys.exit(1)

    print("Videos with GT XML:")
    for video_path in valid_video_paths:
        gt_xml = find_gt_xml_for_video(video_path, gt_dir)
        if gt_xml is not None:
            print(f"  - {video_path.name} -> {gt_xml.name}")

    script_path = (
        PROJECT_ROOT / "src" / "pig_behavior" / "evaluation" / "tracking_pipeline.py"
    )
    has_force_track_arg = "--force-track" in pipeline_extra_args
    has_benchmark_arg = any(
        arg in {"--benchmark-rules", "--no-benchmark-rules"}
        for arg in pipeline_extra_args
    )
    has_prediction_root = "--prediction-root" in pipeline_extra_args
    has_output_root = "--output-root" in pipeline_extra_args

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_prediction_root = (
        PROJECT_ROOT / "outputs" / "id_tracking" / args.mode / run_timestamp
    )
    default_output_root = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "tracking_metrics"
        / args.mode
        / run_timestamp
    )
    prediction_root = Path(
        _extra_arg_value(pipeline_extra_args, "--prediction-root")
        or default_prediction_root
    )
    output_root = Path(
        _extra_arg_value(pipeline_extra_args, "--output-root")
        or default_output_root
    )

    print("\n==================================================")
    print(
        f"Running {args.mode} benchmark on {len(valid_video_paths)} video(s)"
    )
    print("==================================================")

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
    if not has_prediction_root:
        cmd.extend(["--prediction-root", str(prediction_root)])
    if not has_output_root:
        cmd.extend(["--output-root", str(output_root)])

    cmd.extend(pipeline_extra_args)

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"Error: Evaluation failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode

    for detector_name in ("yolov8", "yolov26"):
        for video_path in valid_video_paths:
            print(
                f"\n[*] Running {detector_name} hard-scene evaluation for: "
                f"{video_path.name}"
            )
            hard_scene_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_hard_scenes.py"),
                "--video", video_path.stem,
                "--prediction-root", str(prediction_root / detector_name),
                "--output-dir", str(output_root / detector_name / "hard_scenes"),
            ]
            if args.profile:
                hard_scene_cmd.extend(["--profile", args.profile])
            if args.path_config:
                hard_scene_cmd.extend(["--path-config", args.path_config])

            print(f"Hard-Scene Command: {' '.join(hard_scene_cmd)}")
            subprocess.run(hard_scene_cmd)

    print(f"\nBenchmark output: {output_root}")
    print("\nEvaluation batch execution finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
