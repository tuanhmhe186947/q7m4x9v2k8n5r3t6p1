from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)


EXPECTED_RESIZE_POLICY = "letterbox_preserve_aspect_rgb_pad_black_v1"
REQUIRED_LETTERBOX_COLUMNS = {
    "source_crop_width",
    "source_crop_height",
    "source_crop_aspect_ratio",
    "letterbox_scale",
    "letterbox_resized_width",
    "letterbox_resized_height",
    "letterbox_pad_left",
    "letterbox_pad_top",
    "letterbox_pad_right",
    "letterbox_pad_bottom",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 image cache manifest and loader path."
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument(
        "--frame-context-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/image_frame_context_manifest.csv"
        ),
    )
    parser.add_argument(
        "--window-context-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/image_window_context_manifest.csv"
        ),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sample-windows", type=int, default=24)
    parser.add_argument(
        "--source-equivalence-contexts",
        type=int,
        default=16,
        help="Compare cached pixels with independently loaded source crops.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "image_cache_letterbox_policy_audit.json"
        ),
    )
    args = parser.parse_args()
    audit = check_image_cache(
        cache_manifest=args.cache_manifest,
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        image_size=args.image_size,
        sample_windows=args.sample_windows,
        source_equivalence_contexts=args.source_equivalence_contexts,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
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
    source_equivalence_contexts: int = 16,
) -> dict[str, Any]:
    """Validate cache files and the cache-aware dataset path on a bounded sample."""

    if source_equivalence_contexts < 0:
        raise ValueError("source_equivalence_contexts must be non-negative")
    errors: list[str] = []
    if not cache_manifest.exists():
        errors.append(f"cache_manifest_missing={cache_manifest}")
        return {
            "schema_version": "classification_v2_image_cache_check_v1",
            "errors": errors,
            "valid": False,
        }
    manifest = pd.read_csv(cache_manifest, low_memory=False)
    required = {
        "image_context_id",
        "cache_path",
        "image_size",
        "cache_format",
        "resize_policy",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        errors.append(f"missing_manifest_columns={missing}")
    missing_letterbox = sorted(REQUIRED_LETTERBOX_COLUMNS.difference(manifest.columns))
    if missing_letterbox:
        errors.append(f"missing_letterbox_metadata_columns={missing_letterbox}")
    duplicate_context = (
        int(manifest["image_context_id"].duplicated().sum())
        if "image_context_id" in manifest
        else -1
    )
    if duplicate_context:
        errors.append(f"duplicate_image_context_id={duplicate_context}")
    if "image_size" in manifest:
        size_mismatch = int(
            pd.to_numeric(manifest["image_size"], errors="coerce").ne(image_size).sum()
        )
        if size_mismatch:
            errors.append(f"image_size_mismatch_rows={size_mismatch}")
    resize_policies = (
        sorted(manifest["resize_policy"].fillna("").astype(str).unique().tolist())
        if "resize_policy" in manifest
        else []
    )
    if resize_policies != [EXPECTED_RESIZE_POLICY]:
        errors.append(f"resize_policy_mismatch={resize_policies}")
    letterbox_summary = (
        _letterbox_geometry_summary(manifest, image_size) if not missing_letterbox else {}
    )
    letterbox_geometry_errors = int(letterbox_summary.get("invalid_rows", 0))
    if letterbox_geometry_errors:
        errors.append(f"letterbox_geometry_invalid_rows={letterbox_geometry_errors}")
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

    equivalence_checked = 0
    equivalence_mismatches = 0
    if not errors and source_equivalence_contexts > 0:
        source_dataset = ClassificationV2ImageSequenceDataset(
            ImageSequenceDatasetConfig(
                frame_context_csv=frame_context_csv,
                window_context_csv=window_context_csv,
                image_size=image_size,
                require_complete=False,
                image_cache_size=0,
            )
        )
        try:
            sample_count = min(int(source_equivalence_contexts), len(manifest))
            sample_positions = np.linspace(0, len(manifest) - 1, sample_count, dtype=int)
            for position in sample_positions:
                cache_row = manifest.iloc[int(position)]
                context_id = str(cache_row["image_context_id"])
                frame_row = source_dataset.frame_by_context_id.get(context_id)
                cache_path = Path(str(cache_row["cache_path"]))
                if not cache_path.is_absolute():
                    cache_path = base / cache_path
                source_image = (
                    source_dataset._load_frame_image(frame_row)
                    if frame_row is not None
                    else None
                )
                equivalence_checked += 1
                if source_image is None:
                    equivalence_mismatches += 1
                    continue
                expected = np.transpose(
                    (np.clip(source_image, 0.0, 1.0) * 255.0).round().astype(np.uint8),
                    (1, 2, 0),
                )
                try:
                    cached = np.load(cache_path)
                except Exception:
                    equivalence_mismatches += 1
                    continue
                if (
                    cached.shape != expected.shape
                    or cached.dtype != expected.dtype
                    or not np.array_equal(cached, expected)
                ):
                    equivalence_mismatches += 1
        finally:
            source_dataset.close()
        if equivalence_mismatches:
            errors.append(f"source_cache_pixel_mismatches={equivalence_mismatches}")

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
        "source_equivalence_checked": int(equivalence_checked),
        "source_equivalence_mismatches": int(equivalence_mismatches),
        "resize_policies": resize_policies,
        "expected_resize_policy": EXPECTED_RESIZE_POLICY,
        "letterbox_geometry_summary": letterbox_summary,
        "letterbox_geometry_invalid_rows": int(letterbox_geometry_errors),
        "errors": errors,
        "valid": not errors,
    }


def _letterbox_geometry_summary(manifest: pd.DataFrame, image_size: int) -> dict[str, Any]:
    """Verify recorded scale/padding preserves crop aspect inside square canvas."""

    numeric = manifest[list(REQUIRED_LETTERBOX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    available = numeric["source_crop_width"].gt(0) & numeric["source_crop_height"].gt(0)
    if not bool(available.any()):
        return {
            "available_rows": 0,
            "non_square_source_crop_rows": 0,
            "padded_canvas_rows": 0,
            "invalid_rows": 0,
        }
    rows = numeric[available].copy()
    crop_width = rows["source_crop_width"]
    crop_height = rows["source_crop_height"]
    expected_scale = pd.concat(
        [image_size / crop_width, image_size / crop_height],
        axis=1,
    ).min(axis=1)
    expected_width = (crop_width * expected_scale).round().clip(lower=1)
    expected_height = (crop_height * expected_scale).round().clip(lower=1)
    expected_pad_left = ((image_size - expected_width) // 2).astype(int)
    expected_pad_top = ((image_size - expected_height) // 2).astype(int)
    expected_pad_right = image_size - expected_width.astype(int) - expected_pad_left
    expected_pad_bottom = image_size - expected_height.astype(int) - expected_pad_top
    pad_width = (
        rows["letterbox_pad_left"]
        + rows["letterbox_resized_width"]
        + rows["letterbox_pad_right"]
    )
    pad_height = (
        rows["letterbox_pad_top"]
        + rows["letterbox_resized_height"]
        + rows["letterbox_pad_bottom"]
    )
    invalid = (
        rows["letterbox_scale"].le(0)
        | rows["letterbox_resized_width"].le(0)
        | rows["letterbox_resized_height"].le(0)
        | rows["letterbox_pad_left"].lt(0)
        | rows["letterbox_pad_top"].lt(0)
        | rows["letterbox_pad_right"].lt(0)
        | rows["letterbox_pad_bottom"].lt(0)
        | pad_width.ne(image_size)
        | pad_height.ne(image_size)
        | rows["letterbox_scale"].sub(expected_scale).abs().gt(1e-6)
        | rows["letterbox_resized_width"].ne(expected_width)
        | rows["letterbox_resized_height"].ne(expected_height)
        | rows["letterbox_pad_left"].ne(expected_pad_left)
        | rows["letterbox_pad_top"].ne(expected_pad_top)
        | rows["letterbox_pad_right"].ne(expected_pad_right)
        | rows["letterbox_pad_bottom"].ne(expected_pad_bottom)
    )
    padded = (
        rows["letterbox_pad_left"].gt(0)
        | rows["letterbox_pad_top"].gt(0)
        | rows["letterbox_pad_right"].gt(0)
        | rows["letterbox_pad_bottom"].gt(0)
    )
    return {
        "available_rows": int(len(rows)),
        "non_square_source_crop_rows": int(crop_width.ne(crop_height).sum()),
        "padded_canvas_rows": int(padded.sum()),
        "invalid_rows": int(invalid.sum()),
    }


if __name__ == "__main__":
    main()
