"""Public entry point for the offline tracking-annotation workflow."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pig_behavior.tracking import (
    DEFAULT_MASK_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    DEFAULT_WEIGHTS_PATH,
    TrackingConfig,
    TrackingSummary,
    display_tracked_video,
    run_tracking,
)
from pig_behavior.tracking.cli import main
from pig_behavior.tracking_path_config import (
    DEFAULT_TRACKING_PATH_CONFIG,
    load_tracking_path_profile,
    profile_path,
    profile_video_path,
    profile_video_paths,
)

__all__ = [
    "DEFAULT_TRACKING_PATH_CONFIG",
    "DEFAULT_MASK_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_VIDEO_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "TrackingConfig",
    "TrackingSummary",
    "display_tracked_video",
    "load_tracking_path_profile",
    "main",
    "profile_path",
    "profile_video_path",
    "profile_video_paths",
    "run_tracking",
]


if __name__ == "__main__":
    raise SystemExit(main())
