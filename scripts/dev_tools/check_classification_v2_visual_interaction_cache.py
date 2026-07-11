from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Check visual interaction cache shape, masks, and lineage.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--require-cvat-ready", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.cache_dir / "visual_context_manifest.csv", low_memory=False)
    errors: list[str] = []
    required = {"visual_context_id", "image_context_id", "source_type", "visual_context_available", "visual_context_status", "cache_path", "resize_policy"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        errors.append(f"missing_columns={missing}")
    duplicate_ids = int(manifest["visual_context_id"].duplicated().sum()) if not missing else -1
    if duplicate_ids:
        errors.append(f"duplicate_visual_context_id={duplicate_ids}")
    available = manifest["visual_context_available"].astype(str).str.lower().isin({"true", "1"})
    checked = 0
    for row in manifest[available].head(100).itertuples(index=False):
        path = args.cache_dir / str(row.cache_path)
        if not path.exists():
            errors.append(f"missing_cache_file={path}"); continue
        value = np.load(path, mmap_mode="r")
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] != value.shape[1]:
            errors.append(f"invalid_cache_tensor={path}:{value.shape}:{value.dtype}")
        checked += 1
    cvat = manifest[manifest["source_type"].astype(str).eq("cvat_tracking_xml")]
    legacy = manifest[manifest["source_type"].astype(str).eq("legacy_recovered")]
    cvat_ready = int(cvat["visual_context_available"].astype(str).str.lower().isin({"true", "1"}).sum())
    legacy_ready = int(legacy["visual_context_available"].astype(str).str.lower().isin({"true", "1"}).sum())
    if args.require_cvat_ready and cvat_ready == 0:
        errors.append("no_cvat_visual_context_ready")
    if legacy_ready:
        errors.append(f"legacy_visual_context_must_be_masked={legacy_ready}")
    result = {"rows": len(manifest), "cvat_rows": len(cvat), "cvat_ready_rows": cvat_ready, "legacy_rows": len(legacy), "legacy_ready_rows": legacy_ready, "checked_cache_tensors": checked, "duplicate_visual_context_id": duplicate_ids, "errors": errors, "valid": not errors}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
