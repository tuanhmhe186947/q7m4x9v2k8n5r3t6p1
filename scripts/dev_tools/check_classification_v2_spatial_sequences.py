from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_NPZ = Path("outputs/classification_v2/train_ready_windows/X_spatial_sequences.npz")
DEFAULT_AUDIT = Path("outputs/classification_v2/train_ready_windows/spatial_sequence_audit.json")
DEFAULT_WINDOWS = Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 spatial sequence export.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--window-manifest-csv", type=Path, default=DEFAULT_WINDOWS)
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    window_rows = sum(1 for _ in open(args.window_manifest_csv, encoding="utf-8")) - 1
    errors: list[str] = []

    if audit.get("rows") != window_rows:
        errors.append(f"row_count_mismatch audit={audit.get('rows')} manifest={window_rows}")
    if audit.get("forbidden_selected"):
        errors.append(f"forbidden_selected={audit.get('forbidden_selected')}")
    if audit.get("errors"):
        errors.extend(str(e) for e in audit.get("errors", []))

    data = np.load(args.npz)
    required_arrays = {
        "bbox_xywh_n",
        "bbox_shape_n",
        "motion_delta",
        "roi_class_relation",
        "social_relation",
        "quality_mask",
        "observed_mask",
        "frame_index_sequence",
    }
    missing = sorted(required_arrays.difference(data.files))
    if missing:
        errors.append(f"missing_arrays={missing}")

    shapes = {}
    for name in data.files:
        arr = data[name]
        shapes[name] = list(arr.shape)
        if arr.shape[0] != window_rows:
            errors.append(f"{name}_row_mismatch={arr.shape[0]} expected={window_rows}")
        if name != "frame_index_sequence":
            if not np.isfinite(arr).all():
                errors.append(f"{name}_has_nan_or_inf")

    observed = data["observed_mask"] if "observed_mask" in data.files else np.zeros((0, 0), dtype=np.float32)
    if observed.size and not ((observed == 0.0) | (observed == 1.0)).all():
        errors.append("observed_mask_not_binary")

    result = {
        "npz": str(args.npz),
        "audit_json": str(args.audit_json),
        "window_rows": int(window_rows),
        "arrays": shapes,
        "observed_slots": int(observed.sum()) if observed.size else 0,
        "total_slots": int(observed.size),
        "observed_ratio": float(observed.sum() / max(1, observed.size)) if observed.size else 0.0,
        "audit_missing_frame_slots": audit.get("missing_frame_slots"),
        "forbidden_selected": audit.get("forbidden_selected"),
        "errors": errors,
        "warnings": audit.get("warnings", []),
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
