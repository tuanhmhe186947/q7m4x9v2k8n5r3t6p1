#!/usr/bin/env python3
"""Script to run 3D RGB-D BEV tracking on a pig video and export annotations and visualization."""

import sys
from pathlib import Path

# Add project root and src/ to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.tracking.cli import main as tracking_main  # noqa: E402


def run_tracking():
    # Set default arguments for the tracking CLI
    cli_args = [
        "pig-track-for-annotation",  # dummy arg0
        "--video", "data/videos/Pigs291119_000226_30fps.mp4",
        "--rgbd",
        "--depth-video", "Help_Pigs291119_000226_30fps/depth.mp4",
        "--times-file", "Help_Pigs291119_000226_30fps/times.txt",
        "--depth-scale-file", "Help_Pigs291119_000226_30fps/depth_scale.npy",
        "--inverse-intrinsic-file", "Help_Pigs291119_000226_30fps/inverse_intrinsic.npy",
        "--rotation-file", "Help_Pigs291119_000226_30fps/rot.npy",
        "--output-dir", "outputs",
        "--use-iou-fallback",
        "--no-use-conditional-area-occlusion-freeze",
    ]

    # Parse helper args from user input
    user_args = sys.argv[1:]
    
    # If user wants a preview (default 150 frames)
    if "--full" not in user_args:
        # Check if max-frames is already specified in user args
        if not any(arg.startswith("--max-frames") for arg in user_args):
            cli_args.extend(["--max-frames", "150"])
            print("[*] Running in preview mode (first 150 frames). Use --full to process the entire video.")
    else:
        user_args.remove("--full")
        print("[*] Running in full-video mode.")

    # Forward any other user arguments
    cli_args.extend(user_args)

    print(f"[*] Executing Tracking CLI with args:\n{cli_args}\n")
    
    # Replace sys.argv and execute the main CLI entrypoint
    sys.argv = cli_args
    sys.exit(tracking_main())

if __name__ == "__main__":
    run_tracking()
