from __future__ import annotations

import csv
from pathlib import Path

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.refinement import (
    local_swap_motion_cost,
    shape_iou,
)


RUN_DIR = Path("outputs/eval/hybrid_bytetrack/20260703_222520")
CASES = {
    "Pigs291119_000231_30fps": (401, 3, 4, 5),
    "Pigs291119_000263_30fps": (193, 2, 3, 4),
    "Pigs301119_000328_30fps": (1346, 3, 5, 7),
}


def load_pred_shapes(pred_xml: Path) -> dict[int, dict[int, dict[str, object]]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(pred_xml).getroot()
    out: dict[int, dict[int, dict[str, object]]] = {}
    for track in root.findall("track"):
        label = str(track.attrib["label"])
        fixed_id = int(label.removeprefix("Pig_"))
        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue
            frame = int(box.attrib["frame"])
            attrs = [
                {"name": attr.attrib.get("name", ""), "value": attr.text or ""}
                for attr in box.findall("attribute")
            ]
            out.setdefault(frame, {})[fixed_id] = {
                "frame": frame,
                "label": label,
                "points": [
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                ],
                "attributes": attrs,
            }
    return out


def main() -> None:
    cfg = TrackingConfig()
    metrics_path = next(RUN_DIR.rglob("tracking_metrics.csv"))
    rows = {row["video_stem"]: row for row in csv.DictReader(metrics_path.open(encoding="utf-8-sig"))}
    for video, (frame, _gt_id, first_id, second_id) in CASES.items():
        pred_xml = Path(rows[video]["pred_xml"])
        by_frame = load_pred_shapes(pred_xml)
        print(f"\n{video} f{frame} ids {first_id}/{second_id}")
        for prev_frame in range(frame - 3, frame):
            if prev_frame not in by_frame or frame not in by_frame:
                continue
            if first_id not in by_frame[prev_frame] or second_id not in by_frame[prev_frame]:
                continue
            if first_id not in by_frame[frame] or second_id not in by_frame[frame]:
                continue
            first_prev = by_frame[prev_frame][first_id]
            second_prev = by_frame[prev_frame][second_id]
            first_now = by_frame[frame][first_id]
            second_now = by_frame[frame][second_id]
            overlap = max(shape_iou(first_now, second_now), shape_iou(first_prev, second_prev))
            keep = local_swap_motion_cost(first_prev, second_prev, first_now, second_now, 1920, 1080, swapped=False)
            swap = local_swap_motion_cost(first_prev, second_prev, first_now, second_now, 1920, 1080, swapped=True)
            print(
                f"  prev f{prev_frame}: overlap={overlap:.3f} keep={keep:.4f} "
                f"swap={swap:.4f} gain={keep - swap:.4f}"
            )
        print(
            "  gates:",
            f"min_overlap={cfg.local_pair_swap_min_overlap_iou}",
            f"min_gain={cfg.local_pair_swap_min_motion_gain}",
            f"max_gap={min(cfg.local_pair_swap_max_gap_frames, cfg.local_pair_swap_window_frames)}",
        )


if __name__ == "__main__":
    main()
