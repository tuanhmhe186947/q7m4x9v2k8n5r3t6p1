#!/usr/bin/env python3
"""Canonical entrypoint for running tracking on one or more videos."""

from __future__ import annotations

import argparse
import os
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

try:  # noqa: E402
    from scripts.evaluate_tracking import (
        EVAL_CONFIG_OVERRIDES,
        _format_profile_override_value,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from evaluate_tracking import (
        EVAL_CONFIG_OVERRIDES,
        _format_profile_override_value,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--video", type=str, help="Comma-separated names, paths, keys, or aliases.")
    group.add_argument("-a", "--all-videos", action="store_true", help="Track all configured videos.")
    parser.add_argument("-p", "--profile", type=str, default=None, help="Path profile name.")
    parser.add_argument("--path-config", type=str, default=None, help="Custom tracking_paths.json path.")
    parser.add_argument(
        "--eval-config",
        choices=sorted(EVAL_CONFIG_OVERRIDES),
        help="Named evaluation config preset to apply as tracking profile overrides.",
    )
    parser.add_argument(
        "--list-eval-configs",
        action="store_true",
        help="List available named evaluation config presets and exit.",
    )
    args, tracker_extra_args = parser.parse_known_args()
    if not args.list_eval_configs and not args.video and not args.all_videos:
        parser.error("one of the arguments -v/--video -a/--all-videos is required")
    return args, tracker_extra_args


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
    if args.list_eval_configs:
        for name in sorted(EVAL_CONFIG_OVERRIDES):
            print(name)
        return 0

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
        if args.eval_config:
            for key, value in EVAL_CONFIG_OVERRIDES[args.eval_config].items():
                cmd.extend(
                    [
                        "--profile-override",
                        f"{key}={_format_profile_override_value(value)}",
                    ]
                )
        if args.path_config:
            cmd.extend(["--path-config", args.path_config])
        cmd.extend(tracker_extra_args)
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(SRC_ROOT)
            if not existing_pythonpath
            else f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
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
