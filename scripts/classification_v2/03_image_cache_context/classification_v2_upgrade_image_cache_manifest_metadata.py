from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

RESIZE_POLICY = "letterbox_preserve_aspect_rgb_pad_black_v1"
LETTERBOX_COLUMNS = [
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
]


def main() -> None:
    """Add auditable letterbox metadata to an existing derived cache manifest."""

    parser = argparse.ArgumentParser(description="Upgrade classification_v2 image cache manifest metadata.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest_metadata_upgrade_audit.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = upgrade_manifest_metadata(
        manifest_csv=args.manifest_csv,
        audit_json=args.audit_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def upgrade_manifest_metadata(
    *,
    manifest_csv: Path,
    audit_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Populate scale/pad columns without rewriting cached image arrays."""

    errors: list[str] = []
    if not manifest_csv.exists():
        return _audit(
            manifest_csv=manifest_csv,
            audit_json=audit_json,
            errors=[f"missing_manifest={manifest_csv}"],
        )
    before_sha256 = _sha256(manifest_csv)
    manifest = pd.read_csv(manifest_csv, low_memory=False)
    required = {"x1", "y1", "x2", "y2", "image_size", "resize_policy"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        errors.append(f"missing_required_columns={missing}")
    policy_values = (
        sorted(manifest["resize_policy"].fillna("").astype(str).unique().tolist())
        if "resize_policy" in manifest
        else []
    )
    if policy_values != [RESIZE_POLICY]:
        errors.append(f"resize_policy_mismatch={policy_values}")
    existing_metadata = [col for col in LETTERBOX_COLUMNS if col in manifest.columns]
    if existing_metadata and not overwrite:
        errors.append(f"metadata_columns_already_exist={existing_metadata}")
    if errors:
        audit = _audit(
            manifest_csv=manifest_csv,
            audit_json=audit_json,
            rows=len(manifest),
            before_sha256=before_sha256,
            resize_policy_values=policy_values,
            existing_metadata_columns=existing_metadata,
            errors=errors,
        )
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return audit

    upgraded = manifest.copy()
    metadata = _compute_letterbox_metadata(upgraded)
    for col in LETTERBOX_COLUMNS:
        upgraded[col] = metadata[col]
    upgraded.to_csv(manifest_csv, index=False)
    after_sha256 = _sha256(manifest_csv)
    audit = _audit(
        manifest_csv=manifest_csv,
        audit_json=audit_json,
        rows=len(upgraded),
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        resize_policy_values=policy_values,
        existing_metadata_columns=existing_metadata,
        added_metadata_columns=LETTERBOX_COLUMNS,
        invalid_metadata_rows=int(metadata["_invalid"].sum()),
        non_square_source_crop_rows=int(metadata["_non_square_source_crop"].sum()),
        padded_canvas_rows=int(metadata["_padded_canvas"].sum()),
        errors=[],
    )
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _compute_letterbox_metadata(manifest: pd.DataFrame) -> pd.DataFrame:
    x1 = pd.to_numeric(manifest["x1"], errors="coerce")
    y1 = pd.to_numeric(manifest["y1"], errors="coerce")
    x2 = pd.to_numeric(manifest["x2"], errors="coerce")
    y2 = pd.to_numeric(manifest["y2"], errors="coerce")
    image_size = pd.to_numeric(manifest["image_size"], errors="coerce")
    crop_width = (x2 - x1).clip(lower=0)
    crop_height = (y2 - y1).clip(lower=0)
    valid = crop_width.gt(0) & crop_height.gt(0) & image_size.gt(0)
    out = pd.DataFrame(index=manifest.index)
    for col in LETTERBOX_COLUMNS:
        out[col] = None
    out["_invalid"] = ~valid
    out["_non_square_source_crop"] = False
    out["_padded_canvas"] = False
    if not bool(valid.any()):
        return out
    scale = pd.concat(
        [image_size[valid] / crop_width[valid], image_size[valid] / crop_height[valid]],
        axis=1,
    ).min(axis=1)
    resized_width = (crop_width[valid] * scale).round().clip(lower=1).astype(int)
    resized_height = (crop_height[valid] * scale).round().clip(lower=1).astype(int)
    pad_left = ((image_size[valid] - resized_width) // 2).astype(int)
    pad_top = ((image_size[valid] - resized_height) // 2).astype(int)
    pad_right = image_size[valid].astype(int) - resized_width - pad_left
    pad_bottom = image_size[valid].astype(int) - resized_height - pad_top
    out.loc[valid, "source_crop_width"] = crop_width[valid].astype(float)
    out.loc[valid, "source_crop_height"] = crop_height[valid].astype(float)
    out.loc[valid, "source_crop_aspect_ratio"] = (crop_width[valid] / crop_height[valid]).astype(float)
    out.loc[valid, "letterbox_scale"] = scale.astype(float)
    out.loc[valid, "letterbox_resized_width"] = resized_width
    out.loc[valid, "letterbox_resized_height"] = resized_height
    out.loc[valid, "letterbox_pad_left"] = pad_left
    out.loc[valid, "letterbox_pad_top"] = pad_top
    out.loc[valid, "letterbox_pad_right"] = pad_right
    out.loc[valid, "letterbox_pad_bottom"] = pad_bottom
    out.loc[valid, "_non_square_source_crop"] = crop_width[valid].ne(crop_height[valid])
    out.loc[valid, "_padded_canvas"] = pad_left.gt(0) | pad_top.gt(0) | pad_right.gt(0) | pad_bottom.gt(0)
    return out


def _audit(
    *,
    manifest_csv: Path,
    audit_json: Path,
    errors: list[str],
    rows: int = 0,
    before_sha256: str | None = None,
    after_sha256: str | None = None,
    resize_policy_values: list[str] | None = None,
    existing_metadata_columns: list[str] | None = None,
    added_metadata_columns: list[str] | None = None,
    invalid_metadata_rows: int = 0,
    non_square_source_crop_rows: int = 0,
    padded_canvas_rows: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2_image_cache_manifest_metadata_upgrade_v1",
        "manifest_csv": str(manifest_csv),
        "audit_json": str(audit_json),
        "rows": int(rows),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "resize_policy_values": resize_policy_values or [],
        "expected_resize_policy": RESIZE_POLICY,
        "existing_metadata_columns": existing_metadata_columns or [],
        "added_metadata_columns": added_metadata_columns or [],
        "invalid_metadata_rows": int(invalid_metadata_rows),
        "non_square_source_crop_rows": int(non_square_source_crop_rows),
        "padded_canvas_rows": int(padded_canvas_rows),
        "errors": errors,
        "valid": not errors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
