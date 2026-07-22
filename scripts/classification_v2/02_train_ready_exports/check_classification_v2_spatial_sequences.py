from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)

DEFAULT_NPZ = Path("outputs/classification_v2/train_ready_windows/X_spatial_sequences.npz")
DEFAULT_AUDIT = Path("outputs/classification_v2/train_ready_windows/spatial_sequence_audit.json")
DEFAULT_WINDOWS = Path(
    "outputs/classification_v2/sequence_features_reviewed/"
    "sequence_window_manifest.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 spatial sequence export.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--window-manifest-csv", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--train-mask-csv",
        type=Path,
        default=None,
        help="Optional train mask used to reject incomplete trainable windows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional machine-readable checker result.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing checker output JSON explicitly.",
    )
    args = parser.parse_args()
    if args.output_json is not None:
        require_output_paths_available(
            [args.output_json],
            overwrite=args.overwrite,
        )

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
        "length_mask",
        "observed_mask",
        "spatial_quality_mask",
        "roi_validity_mask",
        "social_validity_mask",
        "pen_validity_mask",
        "adjacent_motion_pair_mask",
        "sparse_velocity_pair_mask",
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
        if not np.isfinite(arr).all():
            errors.append(f"{name}_has_nan_or_inf")

    observed = (
        data["observed_mask"]
        if "observed_mask" in data.files
        else np.zeros((0, 0), dtype=np.float32)
    )
    length = (
        data["length_mask"]
        if "length_mask" in data.files
        else np.zeros((0, 0), dtype=np.float32)
    )
    quality = (
        data["spatial_quality_mask"]
        if "spatial_quality_mask" in data.files
        else np.zeros((0, 0), dtype=np.float32)
    )
    if observed.size and not ((observed == 0.0) | (observed == 1.0)).all():
        errors.append("observed_mask_not_binary")
    if length.size and not ((length == 0.0) | (length == 1.0)).all():
        errors.append("length_mask_not_binary")
    if observed.shape != length.shape:
        errors.append(f"mask_shape_mismatch observed={observed.shape} length={length.shape}")
    elif observed.size and (observed > length).any():
        errors.append("observed_mask_has_values_outside_length_mask")
    if quality.size and not ((quality == 0.0) | (quality == 1.0)).all():
        errors.append("spatial_quality_mask_not_binary")
    if quality.shape != observed.shape:
        errors.append(
            f"quality_mask_shape_mismatch quality={quality.shape} "
            f"observed={observed.shape}"
        )
    elif quality.size and (quality > observed).any():
        errors.append("spatial_quality_mask_has_values_outside_observed_mask")
    if quality.size:
        invalid = quality == 0.0
        for group in ["bbox_xywh_n", "bbox_shape_n", "motion_delta"]:
            if group in data.files and np.any(data[group][invalid] != 0.0):
                errors.append(f"{group}_nonzero_at_invalid_spatial_slot")

    train_mask_audit = _audit_train_mask_completeness(
        args.train_mask_csv,
        length,
        observed,
        window_rows,
    )
    errors.extend(train_mask_audit["errors"])

    result = {
        "npz": str(args.npz),
        "audit_json": str(args.audit_json),
        "rows": audit.get("rows"),
        "array_shapes": shapes,
        "valid_length_slots": int(length.sum()) if length.size else 0,
        "observed_frame_slots": int(observed.sum()) if observed.size else 0,
        "total_frame_slots": int(observed.size),
        "observed_ratio": float(observed.sum() / max(1, observed.size)) if observed.size else 0.0,
        "observed_within_length_ratio": audit.get("observed_within_length_ratio"),
        "padding_slots": audit.get("padding_slots"),
        "missing_observed_slots_within_length": audit.get("missing_observed_slots_within_length"),
        "missing_frame_slots": audit.get("missing_frame_slots"),
        "train_mask_completeness": train_mask_audit,
        "forbidden_selected": audit.get("forbidden_selected"),
        "warnings": audit.get("warnings", []),
        "errors": errors,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _audit_train_mask_completeness(
    mask_path: Path | None,
    length: np.ndarray,
    observed: np.ndarray,
    expected_rows: int,
) -> dict[str, object]:
    """Prove that missing spatial slots cannot enter the training subset."""

    result: dict[str, object] = {
        "available": mask_path is not None,
        "train_mask_csv": str(mask_path) if mask_path is not None else None,
        "trainable_rows": None,
        "trainable_rows_with_missing_slots": None,
        "trainable_missing_slots": None,
        "errors": [],
    }
    errors = result["errors"]
    if mask_path is None:
        return result
    if not mask_path.exists():
        errors.append(f"missing_train_mask_csv={mask_path}")
        return result

    mask_frame = pd.read_csv(mask_path)
    if len(mask_frame) != expected_rows:
        errors.append(
            f"train_mask_row_mismatch={len(mask_frame)} expected={expected_rows}"
        )
        return result
    if len(mask_frame.columns) != 1:
        errors.append(f"train_mask_column_count={len(mask_frame.columns)} expected=1")
        return result

    raw = mask_frame.iloc[:, 0]
    normalized = raw.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        errors.append(f"invalid_train_mask_values={int(invalid.sum())}")
        return result
    trainable = normalized.isin({"true", "1", "yes", "y", "t"}).to_numpy()

    if length.shape != observed.shape or length.shape[0] != expected_rows:
        errors.append("spatial_masks_unavailable_for_train_mask_audit")
        return result
    missing_per_row = np.maximum(length - observed, 0.0).sum(axis=1)
    trainable_missing = missing_per_row[trainable]
    rows_with_missing = int((trainable_missing > 0).sum())
    missing_slots = int(trainable_missing.sum())
    result["trainable_rows"] = int(trainable.sum())
    result["trainable_rows_with_missing_slots"] = rows_with_missing
    result["trainable_missing_slots"] = missing_slots
    if rows_with_missing:
        errors.append(
            "trainable_windows_have_missing_spatial_slots="
            f"rows:{rows_with_missing} slots:{missing_slots}"
        )
    return result


if __name__ == "__main__":
    main()
