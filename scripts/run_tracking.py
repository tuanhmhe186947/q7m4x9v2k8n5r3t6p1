#!/usr/bin/env python3
"""Script to run tracking on one or more videos using pig-track-for-annotation (tracking_annotation.py)."""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src/ to path so we can load configurations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.tracking_path_config import (  # noqa: E402
    load_tracking_path_profile,
    profile_video_path,
    profile_video_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tracking on one or more videos with optional custom tracking arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track a single video by alias/path:
  python scripts/run_tracking.py -v Pigs281119_000085_30fps

  # Track multiple videos:
  python scripts/run_tracking.py -v Pigs281119_000085_30fps,Pigs291119_000226_30fps

  # Track all videos in the active profile video directory:
  python scripts/run_tracking.py -a

  # Track with custom tracker parameters (passed directly to the tracker):
  python scripts/run_tracking.py -v Pigs281119_000085_30fps --det-conf 0.30 --use-merged-box-split
"""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-v", "--video",
        type=str,
        help="Comma-separated video names, paths, or keys/aliases."
    )
    group.add_argument(
        "-a", "--all-videos",
        action="store_true",
        help="Track all videos."
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
    
    # Accept any extra args to forward to the tracker
    return parser.parse_known_args()

def main():
    args, tracker_extra_args = parse_args()
    
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
        print("Error: No valid videos resolved for tracking.", file=sys.stderr)
        sys.exit(1)

    script_path = PROJECT_ROOT / "src" / "pig_behavior" / "data_preparation" / "tracking_annotation.py"
    
    # Run tracking for each video
    for idx, video_path in enumerate(video_paths, 1):
        print("\n==================================================")
        print(f"[{idx}/{len(video_paths)}] Running tracking on: {video_path.name}")
        print("==================================================")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--video", str(video_path),
            "--use-iou-fallback",
            "--no-use-conditional-area-occlusion-freeze",
        ]
        
        # Add profile or path-config if specified to ensure consistent output paths
        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.path_config:
            cmd.extend(["--path-config", args.path_config])
            
        cmd.extend(tracker_extra_args)
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error: Tracking failed for {video_path.name} with exit code {result.returncode}", file=sys.stderr)
            if len(video_paths) == 1:
                sys.exit(result.returncode)

    print("\nTracking batch execution finished.")

if __name__ == "__main__":
    main()
