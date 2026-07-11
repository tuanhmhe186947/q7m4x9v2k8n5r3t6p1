from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 image cache manifest and loader path.")
    parser.add_argument("--cache-manifest", type=Path, default=Path("outputs/classification_v2/image_cache_v2/manifest.csv"))
    parser.add_argument(
        "--frame-context-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_frame_context_manifest.csv"),
    )
    parser.add_argument(
        "--window-context-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_window_context_manifest.csv"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sample-windows", type=int, default=24)
    args = parser.parse_args()
    audit = check_image_cache(
        cache_manifest=args.cache_manifest,
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        image_size=args.image_size,
        sample_windows=args.sample_windows,
    )
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_image_cache(
    *,
    cache_manifest: Path,
    frame_context_csv: Path,
    window_context_csv: Path,
    image_size: int,
    sample_windows: int,
) -> dict[str, Any]:
    """Validate cache files and the cache-aware dataset path on a bounded sample."""

    errors: list[str] = []
    if not cache_manifest.exists():
        errors.append(f"cache_manifest_missing={cache_manifest}")
        return {"schema_version": "classification_v2_image_cache_check_v1", "errors": errors, "valid": False}
    manifest = pd.read_csv(cache_manifest, low_memory=False)
    required = {"image_context_id", "cache_path", "image_size", "cache_format"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        errors.append(f"missing_manifest_columns={missing}")
    duplicate_context = int(manifest["image_context_id"].duplicated().sum()) if "image_context_id" in manifest else -1
    if duplicate_context:
        errors.append(f"duplicate_image_context_id={duplicate_context}")
    if "image_size" in manifest:
        size_mismatch = int(pd.to_numeric(manifest["image_size"], errors="coerce").ne(image_size).sum())
        if size_mismatch:
            errors.append(f"image_size_mismatch_rows={size_mismatch}")
    resize_policies = (
        sorted(manifest["resize_policy"].fillna("").astype(str).unique().tolist())
        if "resize_policy" in manifest
        else []
    )
    base = cache_manifest.parent
    checked_files = 0
    missing_files = 0
    for row in manifest.head(min(len(manifest), 1000)).itertuples(index=False):
        cache_path = Path(str(row.cache_path))
        if not cache_path.is_absolute():
            cache_path = base / cache_path
        checked_files += 1
        if not cache_path.exists():
            missing_files += 1
    if missing_files:
        errors.append(f"sample_missing_cache_files={missing_files}")

    dataset_rows = 0
    observed_frames = 0
    loader_errors: list[Any] = []
    if not errors:
        dataset = ClassificationV2ImageSequenceDataset(
            ImageSequenceDatasetConfig(
                frame_context_csv=frame_context_csv,
                window_context_csv=window_context_csv,
                image_cache_manifest_csv=cache_manifest,
                image_size=image_size,
                max_windows=sample_windows,
                require_complete=True,
            )
        )
        try:
            items = [dataset[index] for index in range(len(dataset))]
            batch = image_sequence_collate(items)
            dataset_rows = int(len(items))
            observed_frames = int(batch["observed_mask"].sum().item())
            loader_errors = [err for item_errors in batch["errors"] for err in item_errors]
            if loader_errors:
                errors.append(f"loader_errors={loader_errors[:10]}")
        finally:
            dataset.close()
    return {
        "schema_version": "classification_v2_image_cache_check_v1",
        "cache_manifest": str(cache_manifest),
        "manifest_rows": int(len(manifest)),
        "checked_files": int(checked_files),
        "sample_windows": int(sample_windows),
        "dataset_rows": int(dataset_rows),
        "observed_frames": int(observed_frames),
        "duplicate_image_context_id": int(duplicate_context),
        "resize_policies": resize_policies,
        "errors": errors,
        "valid": not errors,
    }


if __name__ == "__main__":
    main()
