"""Produce a bounded current-authority H5 feasibility and cohort audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.temporal_views.h5_feasibility import (
    build_h5_targets,
    evaluate_h5_targets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-window-index", type=Path, required=True)
    parser.add_argument("--reviewed-frame-features", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-snapshot", required=True)
    parser.add_argument("--reviewed-snapshot-sha256", required=True)
    parser.add_argument("--split-hash", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.effective_window_index, low_memory=False)
    split = pd.read_csv(args.split_manifest, low_memory=False)
    targets = build_h5_targets(windows, split)
    frame_columns = [
        "object_track_key", "frame_index", "source_type", "video_key",
        "temporal_unit_key", "timestamp_sec", "bbox_valid", "actor_bbox_valid",
        "behavior_reviewed_final",
    ]
    frames = pd.read_csv(args.reviewed_frame_features, usecols=frame_columns, low_memory=False)
    result, audit = evaluate_h5_targets(targets, frames)
    cohort = targets.merge(result, on="h5_target_id", validate="one_to_one")
    cohort = cohort.loc[cohort["h5_valid"]].copy()
    cohort_path = args.output_dir / "common_h5_matched_cohort.csv"
    cohort.to_csv(cohort_path, index=False, lineterminator="\n")
    payload = {
        "schema_version": "classification_v2.temporal_h5_feasibility.v1",
        "status": "CURRENT_AUTHORITY_READ_ONLY_AUDIT",
        "reviewed_snapshot": args.reviewed_snapshot,
        "reviewed_snapshot_sha256": args.reviewed_snapshot_sha256,
        "split_hash": args.split_hash,
        "inputs": {
            "effective_window_index_sha256": _sha256(args.effective_window_index),
            "reviewed_frame_features_sha256": _sha256(args.reviewed_frame_features),
            "split_manifest_sha256": _sha256(args.split_manifest),
        },
        "audit": audit,
        "common_h5_matched_cohort": {
            "rows": int(len(cohort)), "path": cohort_path.name, "sha256": _sha256(cohort_path),
        },
    }
    path = args.output_dir / "h5_feasibility_and_retention_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
