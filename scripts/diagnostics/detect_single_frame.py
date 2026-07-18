#!/usr/bin/env python3
"""Script to run object detection on a specific frame of a video, applying a mask and exporting the result image."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ultralytics import YOLO

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.constants import TRACK_COLORS_BGR
from pig_behavior.tracking.detections import parse_detections
from pig_behavior.tracking.masks import apply_mask_to_frame, load_mask, shade_outside_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect pigs in one or more frames of a video with mask applied."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\ironh\Downloads\PIG_Behavior_Project\data\videos\Pigs291119_000226_30fps.mp4"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=979,
        help="0-indexed frame number to start pig detection.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="0-indexed frame number to end pig detection (inclusive). If None, only --start-frame is processed.",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="Deprecated alias for --start-frame.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save individual frame images when processing a range of frames.",
    )
    parser.add_argument(
        "--expected-pigs",
        type=int,
        default=8,
        help="Number of pigs expected in the frame.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=Path(r"C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png"),
        help="Path to the binary mask image.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8_roboflow_2.pt",
        help="Path to the detector model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detections",
        help="Directory to save the output visualization.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.80,
        help="Overlap threshold for YOLO NMS.",
    )
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="Use Roboflow Workflows for pig detection instead of local YOLOv8.",
    )
    parser.add_argument(
        "--roboflow-api-key",
        type=str,
        default=None,
        help="Optional API key for Roboflow (otherwise uses ROBOFLOW_API_KEY environment variable).",
    )
    return parser.parse_args()


def draw_detections(
    frame: NDArray[np.uint8],
    detections: list[Any],
    mask: NDArray[np.uint8] | None,
    frame_index: int,
) -> NDArray[np.uint8]:
    """Draw bounding boxes and confidence scores on the frame with mask overlay."""
    # Ensure BGR frame format is treated correctly for drawing
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected 3D BGR image, got shape {frame.shape}"
    
    # Shade the area outside the mask for high-contrast ROI
    vis = shade_outside_roi(frame, mask)
    
    # Draw contour outline of the mask in white
    if mask is not None:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

    # Overlay frame number
    cv2.putText(
        vis,
        f"Frame: {frame_index}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw top-K detections with distinct colors
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.box.astype(int)
        color = TRACK_COLORS_BGR.get(idx, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        label = f"Pig {idx}: {det.score:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        
    return vis


def main() -> int:
    args = parse_args()

    # Validate paths
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 1
    if not args.mask.exists():
        print(f"Error: Mask file not found: {args.mask}", file=sys.stderr)
        return 1
    if not args.roboflow and not args.weights.exists():
        print(f"Error: Model weights not found: {args.weights}", file=sys.stderr)
        return 1

    start_frame = args.start_frame
    if args.frame_idx is not None:
        start_frame = args.frame_idx

    # Load configuration
    cfg = TrackingConfig(
        video_path=args.video,
        weights_path=args.weights,
        mask_path=args.mask,
        expected_pigs=args.expected_pigs,
        det_conf=args.det_conf,
        iou=args.iou,
    )

    # Setup device & Model if not running Roboflow Workflows
    if not args.roboflow:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Target device: {device}")

        # Load YOLO Model
        model = YOLO(str(cfg.weights_path))
        try:
            model.to(device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print("[!] CUDA initialization failed, falling back to CPU", file=sys.stderr)
                device = torch.device("cpu")
                model.to(device)

        device_str = "cpu" if device.type == "cpu" else str(device.index or 0)
        print(f"[*] Running inference using weights: {cfg.weights_path.name}")
    else:
        print("[*] Running in Roboflow Workflows mode.")
        try:
            from pig_behavior.roboflow_client import get_roboflow_client
            # Verify client can be obtained (raises RoboflowError if key is missing)
            get_roboflow_client(args.roboflow_api_key)
            device_str = "cloud"
        except Exception as e:
            print(f"Error initializing Roboflow client: {e}", file=sys.stderr)
            return 1

    # Determine execution mode (single frame vs range)
    is_range = args.end_frame is not None and args.end_frame > start_frame

    if not is_range:
        # Load video capture
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"Error: Could not open video: {args.video}", file=sys.stderr)
            return 1

        try:
            # Seek to the target frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame = cap.read()
            if not ok:
                print(f"Error: Could not read frame {start_frame} from video.", file=sys.stderr)
                return 1
        finally:
            cap.release()

        height, width = frame.shape[:2]
        print(f"[*] Successfully loaded frame {start_frame} (dimensions: {width}x{height})")

        # Load mask and apply it to the frame
        mask = load_mask(cfg.mask_path, width, height, cfg)
        detector_frame = apply_mask_to_frame(frame, mask) if mask is not None else frame

        if args.roboflow:
            from pig_behavior.roboflow_client import detect_pigs_roboflow
            try:
                res = detect_pigs_roboflow(
                    frame=detector_frame,
                    api_key=args.roboflow_api_key
                )
            except Exception as e:
                print(f"Error running Roboflow Workflow: {e}", file=sys.stderr)
                return 1
            
            count = res["count_objects"]
            detections = res["predictions"]
            visualized_frame = res["visualized_frame"]
            
            # Print predictions in the same format
            print(f"[*] Roboflow Workflow detected {count} pigs total:")
            for idx, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det["box"]
                w, h = x2 - x1, y2 - y1
                print(
                    f"    - Pig {idx}: conf={det['score']:.4f}, "
                    f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                    f"(width={w:.1f}, height={h:.1f})"
                )
            
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"detect_frame_{start_frame}_roboflow.png"
            cv2.imwrite(str(output_path), visualized_frame)
            print(f"[*] Saved Roboflow visualization to: {output_path}")

        else:
            results = model(
                source=detector_frame,
                conf=cfg.det_conf,
                iou=cfg.iou,
                verbose=False,
                device=device_str,
            )

            if not results or len(results) == 0:
                print("[!] No detection results from YOLO.", file=sys.stderr)
                return 1

            # Parse and filter detections
            detections = parse_detections(results[0], frame, mask, cfg)
            
            # Sort by confidence score (descending) and limit to expected number of pigs
            detections.sort(key=lambda det: det.score, reverse=True)
            top_detections = detections[:args.expected_pigs]

            print(f"[*] Detected {len(detections)} pigs total. Keeping top {len(top_detections)}:")
            for idx, det in enumerate(top_detections, start=1):
                x1, y1, x2, y2 = det.box
                w, h = x2 - x1, y2 - y1
                print(
                    f"    - Pig {idx}: conf={det.score:.4f}, "
                    f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                    f"(width={w:.1f}, height={h:.1f})"
                )

            # Draw visual overlays
            visualized_frame = draw_detections(frame, top_detections, mask, start_frame)

            # Save output frame
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"detect_frame_{start_frame}.png"
            cv2.imwrite(str(output_path), visualized_frame)
            print(f"[*] Saved output image: {output_path}")

    else:
        end_frame = args.end_frame
        print(f"[*] Processing frame range: {start_frame} to {end_frame} (inclusive)")
        
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"Error: Could not open video: {args.video}", file=sys.stderr)
            return 1

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            
            # Read first frame to get size
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame = cap.read()
            if not ok:
                print(f"Error: Could not read start frame {start_frame} from video.", file=sys.stderr)
                return 1
            
            height, width = frame.shape[:2]
            mask = load_mask(cfg.mask_path, width, height, cfg)
            
            # Setup output video writer
            args.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_roboflow" if args.roboflow else ""
            output_video_path = args.output_dir / f"detect_range_{start_frame}_{end_frame}{suffix}.mp4"
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                print(f"Error: Could not open video writer for: {output_video_path}", file=sys.stderr)
                return 1

            # Setup directory for saving individual frames if requested
            range_dir = None
            if args.save_images:
                range_dir = args.output_dir / f"detect_range_{start_frame}_{end_frame}{suffix}"
                range_dir.mkdir(parents=True, exist_ok=True)
                print(f"[*] Saving individual images to: {range_dir}")

            # Reset pointer to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Loop and process
            for frame_idx in range(start_frame, end_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    print(f"\n[*] End of video stream or error reading frame {frame_idx}.")
                    break

                detector_frame = apply_mask_to_frame(frame, mask) if mask is not None else frame
                
                if args.roboflow:
                    from pig_behavior.roboflow_client import detect_pigs_roboflow
                    try:
                        res = detect_pigs_roboflow(
                            frame=detector_frame,
                            api_key=args.roboflow_api_key
                        )
                        top_detections = res["predictions"]
                        visualized_frame = res["visualized_frame"]
                    except Exception as e:
                        print(f"\nError running Roboflow Workflow for frame {frame_idx}: {e}", file=sys.stderr)
                        # On error, fallback to original frame
                        top_detections = []
                        visualized_frame = frame.copy()
                else:
                    results = model(
                        source=detector_frame,
                        conf=cfg.det_conf,
                        iou=cfg.iou,
                        verbose=False,
                        device=device_str,
                    )

                    if results and len(results) > 0:
                        detections = parse_detections(results[0], frame, mask, cfg)
                        detections.sort(key=lambda det: det.score, reverse=True)
                        top_detections = detections[:args.expected_pigs]
                    else:
                        top_detections = []

                    visualized_frame = draw_detections(frame, top_detections, mask, frame_idx)
                
                writer.write(visualized_frame)

                if range_dir is not None:
                    img_path = range_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(img_path), visualized_frame)

                # Show status/progress
                msg = f"\r[*] Processed frame {frame_idx}/{end_frame} (detected {len(top_detections)} pigs)"
                print(msg, end="", flush=True)

            print()  # Newline after loop
            writer.release()
            print(f"[*] Successfully saved output range video: {output_video_path}")

        finally:
            cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
