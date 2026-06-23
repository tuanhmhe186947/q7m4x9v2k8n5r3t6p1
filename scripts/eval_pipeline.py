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
  python scripts/run_pipeline_eval.py -v Pigs281119_000085_30fps
  python scripts/run_pipeline_eval.py -v Pigs281119_000085_30fps
      --no-benchmark-rules --use-conditional-area-occlusion-freeze
  python scripts/run_pipeline_eval.py -v Pigs281119_000085_30fps --benchmark-detectors
  python scripts/run_pipeline_eval.py -a
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

    # 1. Resolve weights path to determine detector name
    weights_path = None
    # Check if --weights was passed in pipeline_extra_args
    for i, arg in enumerate(pipeline_extra_args):
        if arg == "--weights" and i + 1 < len(pipeline_extra_args):
            weights_path = Path(pipeline_extra_args[i + 1])
            break

    # If not in args, check profile
    if not weights_path and profile:
        profile_weights = profile.get("weights")
        if profile_weights:
            weights_path = Path(profile_weights)

    # Fallback to default
    if not weights_path:
        from pig_behavior.evaluation.tracking.assets import DETECTOR_WEIGHTS_V8
        weights_path = DETECTOR_WEIGHTS_V8

    # 2. Determine detector name (yolov8 or yolov26)
    weights_stem = weights_path.name.lower() if weights_path else ""
    if "yolov26" in weights_stem or "v26" in weights_stem:
        detector_name = "yolov26"
    else:
        detector_name = "yolov8"

    # Generate timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    script_path = (
        PROJECT_ROOT / "src" / "pig_behavior" / "evaluation" / "tracking_pipeline.py"
    )

    for idx, video_path in enumerate(valid_video_paths, 1):
        print("\n==================================================")
        print(
            f"[{idx}/{len(valid_video_paths)}] Running tracking evaluation on: "
            f"{video_path.name}"
        )
        print("==================================================")

        # Output folder for tracking metrics under evaluation/tracking_metrics
        custom_output_root = (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "tracking_metrics"
            / detector_name
            / timestamp_str
            / video_path.stem
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--video",
            str(video_path),
            "--output-root",
            str(custom_output_root),
        ]

        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.path_config:
            cmd.extend(["--path-config", args.path_config])

        cmd.extend(pipeline_extra_args)

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Error: Evaluation failed for {video_path.name} with exit code "
                f"{result.returncode}",
                file=sys.stderr,
            )
            continue

        # Automatically trigger hard-scene evaluation for the video
        print(f"\n[*] Running hard-scene evaluation for: {video_path.name}")
        custom_hard_scene_dir = (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "hard_scene_output"
            / detector_name
            / timestamp_str
        )
        hard_scene_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_hard_scenes.py"),
            "--video", video_path.stem,
            "--output-dir", str(custom_hard_scene_dir),
        ]
        if args.profile:
            hard_scene_cmd.extend(["--profile", args.profile])
        if args.path_config:
            hard_scene_cmd.extend(["--path-config", args.path_config])

        print(f"Hard-Scene Command: {' '.join(hard_scene_cmd)}")
        subprocess.run(hard_scene_cmd)

    print("\nEvaluation batch execution finished.")


if __name__ == "__main__":
    main()
