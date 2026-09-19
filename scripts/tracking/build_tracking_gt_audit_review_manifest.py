"""Build deterministic human-review units without modifying source evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import atomic_write_json, load_rows, parse_span, sha256

ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "docs/tracking/development_evidence_defense/DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv"
)
AUTH = (
    ROOT
    / "docs/tracking/development_evidence_defense"
    / "DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json"
)
MAP = ROOT / "docs/tracking/development_evidence_defense/DEVELOPMENT_VIDEO_SESSION_MAP_20260730.csv"
POP = (
    ROOT
    / "docs/tracking/b0_b1_r0_standard_v2/B0_B1_R0_STANDARD_V2_POPULATION_MANIFEST_20260728.json"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(
            ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv"
        ),
    )
    ap.add_argument(
        "--discovery-output",
        default=str(
            ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_GUI_INPUT_DISCOVERY_20260730.json"
        ),
    )
    args = ap.parse_args()
    rows = load_rows(AUDIT)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    videos = {r["video_id"]: r for r in load_rows(MAP)}
    population = {r["video_key"]: r for r in json.loads(POP.read_text(encoding="utf-8"))["videos"]}
    gt = {r["video_id"]: r for r in auth["ground_truth_authorities"]}
    preds = auth["prediction_authorities"]
    pred_files = {
        m: {Path(x["relative_path"]).name: x for x in d["files"]} for m, d in preds.items()
    }
    method_aliases = {
        "realtime_fast_vs_bytetrack_raw": "realtime_fast",
        "hybrid_bytetrack_vs_realtime_fast": "hybrid_bytetrack",
        "rf_hybrid_vs_realtime_fast": "rf_hybrid",
    }
    prepared = []
    for source_row in rows:
        row = dict(source_row)
        row["method_id"] = method_aliases.get(row["method_id"], row["method_id"])
        start, end = parse_span(row["frame_or_span"])
        purpose = row["error_category"] + "|" + row["selection_reason"]
        key = (row["video_id"], row["method_id"], row.get("episode_id", ""), purpose)
        prepared.append((key, start, end, row))
    groups = []
    for key, start, end, row in sorted(prepared, key=lambda item: (item[0], item[1], item[2])):
        previous = groups[-1] if groups else None
        if previous and previous["key"] == key and start <= previous["end"] + 1:
            previous["rows"].append(row)
            previous["end"] = max(previous["end"], end)
        else:
            groups.append({"key": key, "rows": [row], "start": start, "end": end})
    source_hash = sha256(AUDIT)
    out = []
    for i, item in enumerate(
        sorted(
            groups,
            key=lambda value: (
                value["rows"][0]["video_id"],
                value["rows"][0]["method_id"],
                value["start"],
            ),
        ),
        1,
    ):
        row = item["rows"][0]
        vid = row["video_id"]
        vm = videos.get(vid)
        gm = gt.get(vid)
        if not vm or not gm or vid not in population or row["method_id"] not in pred_files:
            raise SystemExit(
                f"TRACKING_GT_AUDIT_INPUT_AUTHORITY_INCOMPLETE:{vid}:{row['method_id']}"
            )
        pred = pred_files[row["method_id"]].get(vid + ".xml")
        if not pred:
            raise SystemExit(
                f"TRACKING_GT_AUDIT_INPUT_AUTHORITY_INCOMPLETE:prediction:{vid}:{row['method_id']}"
            )
        pop = population[vid]
        context_start = max(0, item["start"] - round(3 * float(pop["frames_per_second"])))
        context_end = min(
            int(pop["frame_count"]) - 1, item["end"] + round(3 * float(pop["frames_per_second"]))
        )
        actual_root = Path(gm["path"]).parents[3]
        out.append(
            {
                "review_unit_id": f"GT-AUDIT-{i:04d}",
                "linked_audit_item_ids": ";".join(x["audit_item_id"] for x in item["rows"]),
                "video_id": vid,
                "video_path": vm["video_path_or_authority_id"],
                "video_sha256": pop["source_video_sha256"],
                "primary_method_id": row["method_id"],
                "prediction_path": str(actual_root / pred["relative_path"]),
                "prediction_sha256": pred["sha256"],
                "GT_path": gm["path"],
                "GT_sha256": gm["sha256"],
                "anchor_frame": str((item["start"] + item["end"]) // 2),
                "event_start_frame": item["start"],
                "event_end_frame": item["end"],
                "context_start_frame": context_start,
                "context_end_frame": context_end,
                "FPS": vm["FPS"],
                "context_seconds_before": 3,
                "context_seconds_after": 3,
                "episode_id": row.get("episode_id", ""),
                "GT_identity": row["GT_identity"],
                "predicted_identity": row["predicted_identity"],
                "Hidden_status": row["Hidden_status"],
                "matching_eligibilities": ";".join(
                    sorted({x["matching_eligibility"] for x in item["rows"]})
                ),
                "error_category": row["error_category"],
                "selection_reasons": ";".join(
                    sorted({x["selection_reason"] for x in item["rows"]})
                ),
                "metric_contributions": ";".join(
                    sorted({x["metric_contribution"] for x in item["rows"]})
                ),
                "review_priority": "HIGH" if row["method_id"] == "hybrid_bytetrack" else "NORMAL",
                "source_item_count": len(item["rows"]),
                "source_manifest_sha256": source_hash,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    actual_root = Path(gt[next(iter(gt))]["path"]).parents[3]
    continuation_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    current_dirty = []
    for relative_path in (
        "data/annotations/roi/ROI_annotations.toy_adjusted.coco.json",
        "data/annotations/roi/ROI_annotations.toy_adjusted.manifest.json",
    ):
        candidate = actual_root / relative_path
        if candidate.exists():
            current_dirty.append({"relative_path": relative_path, "sha256": sha256(candidate)})
    discovered = {
        "starting_main_sha": continuation_sha,
        "authority_starting_main_sha": auth.get("starting_main_sha"),
        "continuation_authorized_from_non_doc_descendants": True,
        "audit_items_path": str(AUDIT),
        "audit_items_sha256": source_hash,
        "input_authority_path": str(AUTH),
        "input_authority_sha256": sha256(AUTH),
        "video_count": len(videos),
        "source_item_count": len(rows),
        "referenced_method_ids": sorted({r["primary_method_id"] for r in out}),
        "GT_authority_count": len(gt),
        "prediction_authority_count": len(preds),
        "existing_GUI_components_reused": ["atomic-write pattern only"],
        "unresolved_input_fields": [],
        "protected_dirty_file_hashes": auth.get("protected_dirty_files", {}),
        "current_dirty_file_hashes": current_dirty,
    }
    atomic_write_json(args.discovery_output, discovered)
    atomic_write_json(
        output.with_name("TRACKING_GT_AUDIT_REVIEW_MANIFEST_INTEGRITY_20260730.json"),
        {
            "SOURCE_AUDIT_ITEMS": len(rows),
            "MAPPED_SOURCE_ITEMS": sum(x["source_item_count"] for x in out),
            "UNMAPPED_SOURCE_ITEMS": 0,
            "DUPLICATELY_MAPPED_SOURCE_ITEMS": 0,
            "REVIEW_UNITS": len(out),
            "UNKNOWN_METHODS": 0,
            "INVALID_FRAME_RANGES": sum(
                x["event_start_frame"] < 0 or x["event_end_frame"] >= 1800 for x in out
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
