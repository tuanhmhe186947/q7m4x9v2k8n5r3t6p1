from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml
from pig_behavior.evaluation.tracking.matching import match_frame


RUN_DIR = Path("outputs/eval/hybrid_bytetrack/20260703_194929/smooth_det020_loose")
TARGET_VIDEOS = {
    "Pigs291119_000231_30fps",
    "Pigs291119_000263_30fps",
    "Pigs301119_000328_30fps",
    "Pigs301119_000329_30fps",
}


def object_id(obj: object) -> str:
    for name in ("object_id", "id", "track_id", "source_track_id"):
        value = getattr(obj, name, None)
        if value not in (None, ""):
            return str(value)
    return repr(obj)


def summarize_frames(frames: list[int]) -> str:
    if not frames:
        return ""
    ranges: list[tuple[int, int]] = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame <= prev + 2:
            prev = frame
            continue
        ranges.append((start, prev))
        start = prev = frame
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def analyze_pair(video: str, gt_xml: Path, pred_xml: Path) -> list[dict[str, str]]:
    gt_by_frame = parse_cvat_video_xml(gt_xml, include_hidden=True)
    pred_by_frame = parse_cvat_video_xml(pred_xml, include_hidden=True)
    prev_match_for_gt: dict[str, str] = {}
    switches: list[dict[str, str]] = []

    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_objs = gt_by_frame.get(frame, [])
        pred_objs = pred_by_frame.get(frame, [])
        matches = match_frame(gt_objs, pred_objs, iou_threshold=0.5)
        for gt_idx, pred_idx, iou in matches:
            gt_id = object_id(gt_objs[gt_idx])
            pred_id = object_id(pred_objs[pred_idx])
            prev_pred = prev_match_for_gt.get(gt_id)
            if prev_pred is not None and prev_pred != pred_id:
                switches.append(
                    {
                        "video": video,
                        "frame": str(frame),
                        "gt_id": gt_id,
                        "from_pred": prev_pred,
                        "to_pred": pred_id,
                        "iou": f"{iou:.3f}",
                    }
                )
            prev_match_for_gt[gt_id] = pred_id

    return switches


def main() -> None:
    metrics_path = next(RUN_DIR.rglob("tracking_metrics.csv"))
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8-sig")))
    by_video = {row["video_stem"]: row for row in rows}
    all_switches: list[dict[str, str]] = []

    for video in sorted(TARGET_VIDEOS):
        row = by_video[video]
        switches = analyze_pair(video, Path(row["gt_xml"]), Path(row["pred_xml"]))
        all_switches.extend(switches)

        by_gt: dict[str, list[dict[str, str]]] = defaultdict(list)
        for switch in switches:
            by_gt[switch["gt_id"]].append(switch)

        print(f"\n{video} metric_idsw={row['idsw']} detected_switches={len(switches)}")
        for gt_id, gt_switches in sorted(by_gt.items()):
            frames = [int(item["frame"]) for item in gt_switches]
            details = "; ".join(
                f"f{item['frame']} {item['from_pred']}->{item['to_pred']} iou={item['iou']}"
                for item in gt_switches
            )
            print(f"  {gt_id}: frames {summarize_frames(frames)} | {details}")

    print(f"\nTOTAL detected_switches={len(all_switches)}")
    out_path = Path(".tmp/idsw_frames_194929.tsv")
    with out_path.open("w", encoding="utf-8", newline="") as out:
        out.write("video\tframe\tgt_id\tfrom_pred\tto_pred\tiou\n")
        for item in all_switches:
            out.write(
                "\t".join(
                    [
                        item["video"],
                        item["frame"],
                        item["gt_id"],
                        item["from_pred"],
                        item["to_pred"],
                        item["iou"],
                    ]
                )
                + "\n"
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
