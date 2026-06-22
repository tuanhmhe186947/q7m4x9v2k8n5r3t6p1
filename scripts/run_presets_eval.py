#!/usr/bin/env python3
"""Script to run tracking annotation pipeline over multiple presets and compile results."""

import os
import sys
import json
import subprocess
import csv
from pathlib import Path

def main():
    # Paths configuration
    video = "data/videos/Pigs291119_000226_30fps.mp4"
    out_root = "outputs/id_tracking/no_gt_assoc_eval_assoc_only"
    video_stem = Path(video).stem

    # Keep this False for normal sweeps. Merged-box split is a local hard-scene
    # rescue rule and can make ordinary frames worse when enabled for a full video.
    enable_merged_split = False
    enable_area_freeze = False

    presets = [
        {"name": "base", "det": 0.25, "jump": 0.08, "stationary": 0.045, "gain": 0.015, "iom": 0.10, "detection_iom": 0.30, "growth": 1.50},
        {"name": "strict_assoc_1", "det": 0.25, "jump": 0.06, "stationary": 0.040, "gain": 0.020, "iom": 0.10, "detection_iom": 0.30, "growth": 1.50},
        {"name": "strict_assoc_2", "det": 0.25, "jump": 0.05, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.30, "growth": 1.40},
        {"name": "strict_assoc_3", "det": 0.30, "jump": 0.05, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.30, "growth": 1.40},
        {"name": "conservative", "det": 0.30, "jump": 0.05, "stationary": 0.030, "gain": 0.030, "iom": 0.15, "detection_iom": 0.40, "growth": 1.30},
        {"name": "soft_detection", "det": 0.25, "jump": 0.06, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.20, "growth": 1.40}
    ]

    summary_rows = []

    for p in presets:
        print(f"\n--- Running preset: {p['name']} ---")
        preset_out = Path(out_root) / p["name"]
        
        # Build command arguments
        cmd = [
            sys.executable,
            "src/pig_behavior/data_preparation/tracking_annotation.py",
            "--video", video,
            "--output-dir", str(preset_out),
            "--det-conf", str(p["det"]),
            "--low-conf-max-center-jump", str(p["jump"]),
            "--bbox-sanity-max-center-jump", str(p["jump"]),
            "--occlusion-stationary-max-center-jump", str(p["stationary"]),
            "--identity-swap-min-gain", str(p["gain"]),
            "--identity-swap-iom-threshold", str(p["iom"]),
            "--occlusion-detection-iom-threshold", str(p["detection_iom"]),
            "--merged-box-growth-ratio", str(p["growth"]),
        ]
        if enable_merged_split:
            cmd.append("--use-merged-box-split")
        if enable_area_freeze:
            cmd.append("--use-conditional-area-occlusion-freeze")

        subprocess.run(cmd, check=True)

        report_path = preset_out / video_stem / "tracking_quality_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        row = {
            "preset": p["name"],
            "det_conf": p["det"],
            "jump_threshold": p["jump"],
            "stationary_threshold": p["stationary"],
            "gain_threshold": p["gain"],
            "iom_threshold": p["iom"],
            "detection_iom": p["detection_iom"],
            "growth_ratio": p["growth"],
            "total_frames": summary.get("total_frames"),
            "review_frame_count": summary.get("review_frame_count"),
            "unassigned_detection_count": summary.get("unassigned_detection_count"),
            "identity_swap_count": summary.get("identity_swap_count"),
            "bbox_jump_count": summary.get("bbox_jump_count"),
            "bbox_jump_rate": summary.get("bbox_jump_rate"),
            "area_spike_count": summary.get("area_spike_count"),
            "aspect_ratio_spike_count": summary.get("aspect_ratio_spike_count"),
            "unstable_track_frames": summary.get("unstable_track_frames"),
            "low_conf_update_count": summary.get("low_conf_update_count"),
            "ambiguous_match_count": summary.get("ambiguous_match_count"),
            "rejected_candidate_count": summary.get("rejected_candidate_count"),
            "rejected_by_center_jump": summary.get("rejected_by_center_jump"),
            "rejected_by_area_ratio": summary.get("rejected_by_area_ratio"),
            "rejected_by_aspect_ratio": summary.get("rejected_by_aspect_ratio"),
            "rejected_by_score_margin": summary.get("rejected_by_score_margin"),
            "issue_frame_count": summary.get("issue_frame_count"),
            "quality_report": str(report_path),
            "output_dir": str(preset_out)
        }
        summary_rows.append(row)

    # Write summary CSV
    if summary_rows:
        Path(out_root).mkdir(parents=True, exist_ok=True)
        summary_csv = Path(out_root) / "no_gt_assoc_summary.csv"
        
        fieldnames = list(summary_rows[0].keys())
        try:
            with open(summary_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\nSummary successfully written to: {summary_csv}")
        except Exception as e:
            print(f"Error: Failed to write CSV summary: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
