#!/usr/bin/env python3
"""CLI script to run the Stable Annotation Tracking pipeline.

Designed to produce clean, stable CVAT XML outputs with minimal ID swaps
and spurious tracks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and src/ to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.tracking.config import TrackingConfig  # noqa: E402
from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig  # noqa: E402
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig  # noqa: E402
from pig_behavior.tracking.stabilization.stable_tracker import run_stable_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stable CVAT Tracking Pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Path to input color video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for CSVs, XML, and debug video",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt",
        help="Path to detector weights",
    )
    parser.add_argument("--mask", type=Path, default=None, help="Path to mask image")
    
    # Optional Depth/RGBD options
    parser.add_argument("--depth-video", type=Path, default=None, help="Path to depth video")
    parser.add_argument("--times-file", type=Path, default=None, help="Path to depth sync times text file")
    parser.add_argument("--depth-scale-file", type=Path, default=None, help="Path to depth scale npy file")
    parser.add_argument("--inverse-intrinsic-file", type=Path, default=None, help="Path to intrinsic npy matrix")
    parser.add_argument("--rotation-file", type=Path, default=None, help="Path to camera rotation npy matrix")

    # Pipeline execution settings
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--expected-pigs", type=int, default=8, help="Number of pigs to track")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable debug overlay video generation")
    parser.add_argument("--no-stitch", action="store_true", help="Disable offline tracklet stitching")
    parser.add_argument("--no-smooth", action="store_true", help="Disable bounding box smoothing")
    parser.add_argument("--full", action="store_true", help="Process the entire video without preview limit")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine preview mode
    max_frames = args.max_frames
    if not args.full and max_frames is None:
        max_frames = 150
        print("[*] Running in preview mode (first 150 frames). Use --full to process the entire video.")
    elif args.full:
        print("[*] Running in full-video mode.")

    # Initialize basic 2D tracking configuration
    tc = TrackingConfig(
        video_path=args.video,
        weights_path=args.weights,
        mask_path=args.mask,
        output_dir=args.output_dir,
        expected_pigs=args.expected_pigs,
        det_conf=args.det_conf,
        max_frames=max_frames,
    )

    # Initialize optional RGBD tracking configuration
    rc = None
    if args.depth_video is not None:
        # Check required depth files
        missing = []
        for name, val in [
            ("depth_scale_file", args.depth_scale_file),
            ("inverse_intrinsic_file", args.inverse_intrinsic_file),
            ("rotation_file", args.rotation_file)
        ]:
            if val is None:
                missing.append(name)
        if missing:
            print(f"[!] Error: Using --depth-video requires also specifying: {', '.join(missing)}")
            return 1

        rc = RGBDTrackingConfig(
            tracking_config=tc,
            depth_video_path=args.depth_video,
            times_path=args.times_file,
            depth_scale_path=args.depth_scale_file,
            inverse_intrinsic_path=args.inverse_intrinsic_file,
            rotation_path=args.rotation_file,
        )

    # Construct stable annotation configuration
    config = AnnotationStableConfig(
        tracking_config=tc,
        rgbd_config=rc,
        export_debug_video=not args.no_debug_video,
        smooth_bbox=not args.no_smooth,
    )

    if args.no_stitch:
        # Disable stitching by setting max gap to 0
        config.stitch_max_gap = 0

    print(f"[*] Running stable tracking on video: {tc.video_path}")
    if rc is not None:
        print(f"[*] Hybrid RGB-D mode active. Depth video: {rc.depth_video_path}")
    else:
        print("[*] Pure 2D mode active.")

    try:
        summary = run_stable_tracking(config)
        print("\n[*] Tracking Completed Successfully!")
        print(f"[*] Output CVAT XML: {summary.cvat_video_xml}")
        if config.export_debug_video:
            print(f"[*] Debug Video: {summary.output_video}")
        print(f"[*] Diagnostics folder: {tc.output_dir}")
        return 0
    except Exception as e:
        import traceback
        print(f"\n[!] Error during execution: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
