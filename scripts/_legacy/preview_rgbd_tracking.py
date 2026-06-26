#!/usr/bin/env python3
"""Legacy preview wrapper for RGB-D tracking."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.tracking.cli import main as tracking_main  # noqa: E402


def run_tracking() -> int:
    cli_args = [
        "pig-track-for-annotation",
        "--video",
        "data/videos/Pigs291119_000226_30fps.mp4",
        "--rgbd",
        "--depth-video",
        "Help_Pigs291119_000226_30fps/depth.mp4",
        "--times-file",
        "Help_Pigs291119_000226_30fps/times.txt",
        "--depth-scale-file",
        "Help_Pigs291119_000226_30fps/depth_scale.npy",
        "--inverse-intrinsic-file",
        "Help_Pigs291119_000226_30fps/inverse_intrinsic.npy",
        "--rotation-file",
        "Help_Pigs291119_000226_30fps/rot.npy",
        "--output-dir",
        "outputs/pred/rgbd-preview",
        "--use-iou-fallback",
        "--no-use-conditional-area-occlusion-freeze",
    ]
    user_args = sys.argv[1:]
    if "--full" not in user_args and not any(arg.startswith("--max-frames") for arg in user_args):
        cli_args.extend(["--max-frames", "150"])
        print("[*] Running in preview mode (first 150 frames). Use --full to process the entire video.")
    elif "--full" in user_args:
        user_args.remove("--full")
        print("[*] Running in full-video mode.")
    cli_args.extend(user_args)
    return tracking_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(run_tracking())
