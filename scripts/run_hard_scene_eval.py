#!/usr/bin/env python3
"""Script to run hard-scene identity evaluation and configuration comparisons on one or more videos."""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add src/ to path so we can load configurations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.tracking_path_config import (
    load_tracking_path_profile,
    profile_video_path,
    profile_video_paths,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hard-scene identity evaluation or comparisons on one or more videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate hard scenes on a single video (auto-maps GT XML and prediction XML):
  python scripts/run_hard_scene_eval.py -v Pigs281119_000085_30fps

  # Compare multiple configs for a video (auto-resolves predictions from latest benchmark run):
  python scripts/run_hard_scene_eval.py -v Pigs281119_000085_30fps --compare

  # Compare specific named prediction files:
  python scripts/run_hard_scene_eval.py -v Pigs281119_000085_30fps --compare --pred base=outputs/id_tracking/base/Pigs281119_000085_30fps/annotations_cvat_shapes.json --pred strict=outputs/id_tracking/strict_assoc_1/Pigs281119_000085_30fps/annotations_cvat_shapes.json

  # Run hard-scene evaluation on all videos:
  python scripts/run_hard_scene_eval.py --all-videos
"""
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="Comma-separated video names, paths, or keys/aliases."
    )
    parser.add_argument(
        "-a", "--all-videos",
        action="store_true",
        help="Run on all videos in active profile."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run in configuration comparison mode (compare multiple tracking runs/presets)."
    )
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        help="For compare mode, key=path pairs of prediction files (e.g. name=path)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Override output directory."
    )
    parser.add_argument(
        "-p", "--profile",
        type=str,
        default=None,
        help="Path profile name from configs/tracking_paths.json (default active)."
    )
    parser.add_argument(
        "--path-config",
        type=str,
        default=None,
        help="Path to custom configs/tracking_paths.json file."
    )
    
    # Accept any extra args to forward to the hard-scene evaluator CLI
    return parser.parse_known_args()

def main():
    args, extra_eval_args = parse_args()
    
    # Load profile
    path_config = Path(args.path_config) if args.path_config else None
    try:
        profile = load_tracking_path_profile(path_config, args.profile)
    except Exception as e:
        print(f"Error loading path profile: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve videos
    video_paths = []
    if args.all_videos:
        video_paths = profile_video_paths(profile)
        print(f"Resolved all videos from profile: {[p.name for p in video_paths]}")
    elif args.video:
        video_keys = [k.strip() for k in args.video.split(",")]
        for key in video_keys:
            try:
                v_path = profile_video_path(profile, key)
                if v_path:
                    video_paths.append(v_path)
            except Exception as e:
                # Try direct path
                direct = Path(key)
                if direct.exists() and direct.is_file():
                    video_paths.append(direct)
                else:
                    print(f"Warning: Could not resolve video '{key}': {e}", file=sys.stderr)
                    
    if not video_paths:
        print("Error: No valid videos resolved.", file=sys.stderr)
        sys.exit(1)

    # Resolve GT directory
    gt_dir = Path(profile.get("annotations", {}).get("tracking", "data/annotations/tracking"))
    video_dir = Path(profile.get("videos", "data/videos"))
    prediction_root = Path(profile.get("outputs", {}).get("id_tracking", "outputs/id_tracking"))
    
    for idx, video_path in enumerate(video_paths, 1):
        print(f"\n==================================================")
        print(f"[{idx}/{len(video_paths)}] Running hard-scene evaluation on: {video_path.name}")
        print(f"==================================================")
        
        # Build commands
        if args.compare:
            cmd = [
                sys.executable,
                "-m", "pig_behavior.evaluation.tracking_hard_scene_evaluator",
                "--compare",
                "--video", video_path.stem,
                "--gt-dir", str(gt_dir),
                "--video-dir", str(video_dir),
                "--prediction-root", str(prediction_root),
            ]
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])
            for pred_pair in args.pred:
                cmd.extend(["--pred", pred_pair])
        else:
            cmd = [
                sys.executable,
                "-m", "pig_behavior.evaluation.tracking_hard_scene_evaluator",
                "--video", video_path.stem,
                "--gt-dir", str(gt_dir),
                "--video-dir", str(video_dir),
                "--prediction-root", str(prediction_root),
            ]
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])
                
        cmd.extend(extra_eval_args)
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error: Hard-scene eval failed for {video_path.name} with exit code {result.returncode}", file=sys.stderr)
            if len(video_paths) == 1:
                sys.exit(result.returncode)

    print("\nHard-scene evaluation batch execution finished.")

if __name__ == "__main__":
    main()