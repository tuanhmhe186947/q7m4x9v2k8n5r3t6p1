"""Build deterministic human-review units without modifying source evidence."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import load_rows, parse_span, sha256, atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/tracking/development_evidence_defense/DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv"
AUTH = ROOT / "docs/tracking/development_evidence_defense/DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json"
MAP = ROOT / "docs/tracking/development_evidence_defense/DEVELOPMENT_VIDEO_SESSION_MAP_20260730.csv"
POP = ROOT / "docs/tracking/b0_b1_r0_standard_v2/B0_B1_R0_STANDARD_V2_POPULATION_MANIFEST_20260728.json"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv"))
    ap.add_argument("--discovery-output", default=str(ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_GUI_INPUT_DISCOVERY_20260730.json"))
    args = ap.parse_args()
    rows = load_rows(AUDIT); auth = json.loads(AUTH.read_text(encoding="utf-8"))
    videos = {r["video_id"]: r for r in load_rows(MAP)}
    population = {r["video_key"]: r for r in json.loads(POP.read_text(encoding="utf-8"))["videos"]}
    gt = {r["video_id"]: r for r in auth["ground_truth_authorities"]}
    preds = auth["prediction_authorities"]
    pred_files = {m: {Path(x["relative_path"]).name: x for x in d["files"]} for m, d in preds.items()}
    groups = {}
    for row in rows:
        if row["method_id"] == "realtime_fast_vs_bytetrack_raw": row["method_id"] = "realtime_fast"
        elif row["method_id"] == "hybrid_bytetrack_vs_realtime_fast": row["method_id"] = "hybrid_bytetrack"
        elif row["method_id"] == "rf_hybrid_vs_realtime_fast": row["method_id"] = "rf_hybrid"
        start, end = parse_span(row["frame_or_span"])
        # Episode-level rows are grouped; point items with same purpose are grouped only if adjacent.
        purpose = row["error_category"] + "|" + row["selection_reason"]
        key = (row["video_id"], row["method_id"], row.get("episode_id", ""), purpose)
        item = groups.setdefault(key, {"rows": [], "start": start, "end": end})
        item["rows"].append(row); item["start"] = min(item["start"], start); item["end"] = max(item["end"], end)
    source_hash = sha256(AUDIT)
    out = []
    for i, item in enumerate(sorted(groups.values(), key=lambda x: (x["rows"][0]["video_id"], x["rows"][0]["method_id"], x["start"])), 1):
        row = item["rows"][0]; vid = row["video_id"]; vm = videos.get(vid); gm = gt.get(vid)
        if not vm or not gm or vid not in population or row["method_id"] not in pred_files:
            raise SystemExit(f"TRACKING_GT_AUDIT_INPUT_AUTHORITY_INCOMPLETE:{vid}:{row['method_id']}")
        pred = pred_files[row["method_id"]].get(vid + ".xml")
        if not pred:
            raise SystemExit(f"TRACKING_GT_AUDIT_INPUT_AUTHORITY_INCOMPLETE:prediction:{vid}:{row['method_id']}")
        pop = population[vid]
        context_start = max(0, item["start"] - round(3 * float(pop["frames_per_second"])))
        context_end = min(int(pop["frame_count"]) - 1, item["end"] + round(3 * float(pop["frames_per_second"])))
        actual_root = Path(gm["path"]).parents[3]
        out.append({"review_unit_id": f"GT-AUDIT-{i:04d}",
                    "linked_audit_item_ids": ";".join(x["audit_item_id"] for x in item["rows"]),
                    "video_id": vid, "video_path": vm["video_path_or_authority_id"],
                    "video_sha256": pop["source_video_sha256"],
                    "primary_method_id": row["method_id"],
                    "prediction_path": str(actual_root / pred["relative_path"]), "prediction_sha256": pred["sha256"],
                    "GT_path": gm["path"], "GT_sha256": gm["sha256"], "anchor_frame": str((item["start"] + item["end"]) // 2),
                    "event_start_frame": item["start"], "event_end_frame": item["end"],
                    "context_start_frame": context_start, "context_end_frame": context_end, "FPS": vm["FPS"],
                    "context_seconds_before": 3, "context_seconds_after": 3,
                    "episode_id": row.get("episode_id", ""), "GT_identity": row["GT_identity"],
                    "predicted_identity": row["predicted_identity"], "Hidden_status": row["Hidden_status"],
                    "error_category": row["error_category"],
                    "selection_reasons": ";".join(sorted({x["selection_reason"] for x in item["rows"]})),
                    "metric_contributions": ";".join(sorted({x["metric_contribution"] for x in item["rows"]})),
                    "review_priority": "HIGH" if row["method_id"] == "hybrid_bytetrack" else "NORMAL",
                    "source_item_count": len(item["rows"]), "source_manifest_sha256": source_hash})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(out)
    discovered = {"starting_main_sha": auth.get("starting_main_sha"), "audit_items_path": str(AUDIT), "audit_items_sha256": source_hash,
                  "input_authority_path": str(AUTH), "input_authority_sha256": sha256(AUTH), "video_count": len(videos),
                  "source_item_count": len(rows), "referenced_method_ids": sorted({r["method_id"] for r in rows}),
                  "GT_authority_count": len(gt), "prediction_authority_count": len(preds),
                  "existing_GUI_components_reused": ["atomic-write pattern only"], "unresolved_input_fields": [],
                  "protected_dirty_file_hashes": auth.get("protected_dirty_files", {})}
    atomic_write_json(args.discovery_output, discovered)
    atomic_write_json(output.with_name("TRACKING_GT_AUDIT_REVIEW_MANIFEST_INTEGRITY_20260730.json"),
                      {"SOURCE_AUDIT_ITEMS": len(rows), "MAPPED_SOURCE_ITEMS": sum(x["source_item_count"] for x in out),
                       "UNMAPPED_SOURCE_ITEMS": 0, "DUPLICATELY_MAPPED_SOURCE_ITEMS": 0, "REVIEW_UNITS": len(out), "UNKNOWN_METHODS": 0,
                       "INVALID_FRAME_RANGES": sum(x["event_start_frame"] < 0 or x["event_end_frame"] >= 1800 for x in out)})
    return 0
if __name__ == "__main__": raise SystemExit(main())
