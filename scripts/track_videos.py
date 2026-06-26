#!/usr/bin/env python3
"""Canonical entrypoint for running tracking on one or more videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.tracking_path_config import (  # noqa: E402
    load_tracking_path_profile,
    profile_video_path,
    profile_video_paths,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run tracking on one or more videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/track_videos.py -v Pigs281119_000085_30fps
  python scripts/track_videos.py -v Pigs281119_000085_30fps,Pigs291119_000226_30fps
  python scripts/track_videos.py -a
  python scripts/track_videos.py -v Pigs281119_000085_30fps --det-conf 0.30
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-v", "--video", type=str, help="Comma-separated names, paths, keys, or aliases.")
    group.add_argument("-a", "--all-videos", action="store_true", help="Track all configured videos.")
    parser.add_argument("-p", "--profile", type=str, default=None, help="Path profile name.")
    parser.add_argument("--path-config", type=str, default=None, help="Custom tracking_paths.json path.")
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


def main() -> int:
    args, tracker_extra_args = parse_args()
    path_config = Path(args.path_config) if args.path_config else None
    try:
        profile = load_tracking_path_profile(path_config, args.profile)
        video_paths = _resolve_videos(args, profile)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not video_paths:
        print("Error: No valid videos resolved.", file=sys.stderr)
        return 1

    exit_code = 0
    for video_path in video_paths:
        cmd = [
            sys.executable,
            "-m",
            "pig_behavior.tracking.cli",
            "--video",
            str(video_path),
        ]
        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.path_config:
            cmd.extend(["--path-config", args.path_config])
        cmd.extend(tracker_extra_args)
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(
                f"Error: Tracking failed for {video_path.name} with exit code {result.returncode}",
                file=sys.stderr,
            )
            exit_code = result.returncode
            if len(video_paths) == 1:
                return exit_code

    print("\nTracking batch execution finished.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
