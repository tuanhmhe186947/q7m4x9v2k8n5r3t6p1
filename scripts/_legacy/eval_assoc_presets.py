#!/usr/bin/env python3
"""Legacy local sweep for association-only preset comparisons."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACK_SCRIPT = PROJECT_ROOT / "scripts" / "track_videos.py"


def main() -> int:
    video = PROJECT_ROOT / "data" / "videos" / "Pigs291119_000226_30fps.mp4"
    out_root = PROJECT_ROOT / "outputs" / "pred" / "legacy-presets" / "assoc-only"
    video_stem = video.stem
    enable_merged_split = False
    enable_area_freeze = False
    presets = [
        {"name": "base", "det": 0.25, "jump": 0.08, "stationary": 0.045, "gain": 0.015, "iom": 0.10, "detection_iom": 0.30, "growth": 1.50},
        {"name": "strict_assoc_1", "det": 0.25, "jump": 0.06, "stationary": 0.040, "gain": 0.020, "iom": 0.10, "detection_iom": 0.30, "growth": 1.50},
        {"name": "strict_assoc_2", "det": 0.25, "jump": 0.05, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.30, "growth": 1.40},
        {"name": "strict_assoc_3", "det": 0.30, "jump": 0.05, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.30, "growth": 1.40},
        {"name": "conservative", "det": 0.30, "jump": 0.05, "stationary": 0.030, "gain": 0.030, "iom": 0.15, "detection_iom": 0.40, "growth": 1.30},
        {"name": "soft_detection", "det": 0.25, "jump": 0.06, "stationary": 0.035, "gain": 0.020, "iom": 0.15, "detection_iom": 0.20, "growth": 1.40},
    ]

    summary_rows: list[dict[str, object]] = []
    for preset in presets:
        print(f"\n--- Running preset: {preset['name']} ---")
        preset_out = out_root / preset["name"]
        cmd = [
            sys.executable,
            str(TRACK_SCRIPT),
            "--video",
            str(video),
            "--output-dir",
            str(preset_out),
            "--det-conf",
            str(preset["det"]),
            "--low-conf-max-center-jump",
            str(preset["jump"]),
            "--bbox-sanity-max-center-jump",
            str(preset["jump"]),
            "--occlusion-stationary-max-center-jump",
            str(preset["stationary"]),
            "--identity-swap-min-gain",
            str(preset["gain"]),
            "--identity-swap-iom-threshold",
            str(preset["iom"]),
            "--occlusion-detection-iom-threshold",
            str(preset["detection_iom"]),
            "--merged-box-growth-ratio",
            str(preset["growth"]),
            "--use-iou-fallback",
            "--no-use-conditional-area-occlusion-freeze",
        ]
        if enable_merged_split:
            cmd.append("--use-merged-box-split")
        if enable_area_freeze:
            cmd.append("--use-conditional-area-occlusion-freeze")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        report_path = preset_out / "hybrid_bytetrack" / video_stem / "tracking_quality_report.json"
        with report_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summary_rows.append(
            {
                "preset": preset["name"],
                "det_conf": preset["det"],
                "jump_threshold": preset["jump"],
                "stationary_threshold": preset["stationary"],
                "gain_threshold": preset["gain"],
                "iom_threshold": preset["iom"],
                "detection_iom": preset["detection_iom"],
                "growth_ratio": preset["growth"],
                "total_frames": summary.get("total_frames"),
                "review_frame_count": summary.get("review_frame_count"),
                "unassigned_detection_count": summary.get("unassigned_detection_count"),
                "identity_swap_count": summary.get("identity_swap_count"),
                "bbox_jump_count": summary.get("bbox_jump_count"),
            }
        )

    summary_csv = out_root / f"{video_stem}_preset_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
