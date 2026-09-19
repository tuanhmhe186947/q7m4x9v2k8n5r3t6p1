from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_RESIZE_POLICY = "actor_nearest_partner_union_letterbox_rgb_pad_black_v1"
DEFAULT_CACHE_DIR = Path("outputs/classification_v2/visual_interaction_cache")
DEFAULT_OUTPUT_JSON = Path(
    "outputs/classification_v2/model_design/visual_interaction_cache_audit.json"
)
REQUIRED_MANIFEST_COLUMNS = {
    "visual_context_id",
    "image_context_id",
    "source_type",
    "visual_context_available",
    "visual_context_status",
    "cache_path",
    "resize_policy",
    "context_kind",
    "partner_track_id",
    "union_x1",
    "union_y1",
    "union_x2",
    "union_y2",
}


def main() -> None:
    """Check canonical full-frame actor-partner visual context artifacts."""

    parser = argparse.ArgumentParser(
        description="Check visual interaction cache shape, masks, and lineage."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--require-cvat-ready", action="store_true", default=True)
    parser.add_argument("--require-legacy-ready", action="store_true")
    parser.add_argument("--sample-tensors", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    audit = check_visual_interaction_cache(
        cache_dir=args.cache_dir,
        require_cvat_ready=args.require_cvat_ready,
        require_legacy_ready=args.require_legacy_ready,
        sample_tensors=args.sample_tensors,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_visual_interaction_cache(
    *,
    cache_dir: Path,
    require_cvat_ready: bool,
    require_legacy_ready: bool,
    sample_tensors: int,
) -> dict[str, Any]:
    """Validate visual context is partner/full-frame context, not label gated."""

    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = cache_dir / "visual_context_manifest.csv"
    build_audit_path = cache_dir / "visual_context_cache_audit.json"
    packed_audit_path = cache_dir / "packed_image_cache_audit.json"
    packed_tensor_path = cache_dir / "packed_rgb_64_letterbox.npy"
    packed_index_path = cache_dir / "packed_image_cache_index.csv"
    if not manifest_path.exists():
        return _missing_audit(cache_dir, [f"missing_manifest={manifest_path}"])

    manifest = pd.read_csv(manifest_path, low_memory=False)
    build_audit = _read_optional_json(build_audit_path, warnings)
    packed_audit = _read_optional_json(packed_audit_path, warnings)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        errors.append(f"missing_columns={missing}")

    if "visual_context_id" in manifest.columns:
        duplicate_ids = int(manifest["visual_context_id"].duplicated().sum())
    else:
        duplicate_ids = -1
    if duplicate_ids:
        errors.append(f"duplicate_visual_context_id={duplicate_ids}")

    available = _bool_series(manifest.get("visual_context_available", pd.Series()))
    cvat = manifest[manifest.get("source_type", pd.Series()).astype(str).eq("cvat_tracking_xml")]
    legacy = manifest[manifest.get("source_type", pd.Series()).astype(str).eq("legacy_recovered")]
    cvat_ready = int(_bool_series(cvat.get("visual_context_available", pd.Series())).sum())
    legacy_ready = int(_bool_series(legacy.get("visual_context_available", pd.Series())).sum())
    if require_cvat_ready and len(cvat) and cvat_ready == 0:
        errors.append("no_cvat_visual_context_ready")
    if require_legacy_ready and len(legacy) and legacy_ready == 0:
        errors.append("no_legacy_visual_context_ready")

    resize_policies = _unique_strings(manifest, "resize_policy")
    if resize_policies != [EXPECTED_RESIZE_POLICY]:
        errors.append(f"resize_policy_mismatch={resize_policies}")

    context_kinds = _unique_strings(manifest, "context_kind")
    if context_kinds != ["actor_nearest_partner_union"]:
        errors.append(f"context_kind_mismatch={context_kinds}")

    ready_geometry_errors = _ready_geometry_error_count(manifest, available)
    if ready_geometry_errors:
        errors.append(f"ready_visual_context_geometry_invalid={ready_geometry_errors}")

    tensor_check = _check_individual_tensors(cache_dir, manifest, available, sample_tensors)
    errors.extend(tensor_check["errors"])
    packed_check = _check_packed_cache(
        packed_tensor_path=packed_tensor_path,
        packed_index_path=packed_index_path,
        packed_audit=packed_audit,
        available_rows=int(available.sum()),
    )
    errors.extend(packed_check["errors"])

    if build_audit.get("label_gated") is not False:
        errors.append(f"visual_context_label_gated={build_audit.get('label_gated')}")
    if build_audit.get("rows_dropped_for_missing_context") not in (0, None):
        errors.append(
            "visual_context_rows_dropped_for_missing_context="
            f"{build_audit.get('rows_dropped_for_missing_context')}"
        )
    if build_audit.get("valid") is not True:
        errors.append(f"visual_context_build_audit_invalid={build_audit_path}")

    return {
        "schema_version": "classification_v2_visual_interaction_cache_audit_v1",
        "cache_dir": str(cache_dir),
        "manifest_csv": str(manifest_path),
        "build_audit_json": str(build_audit_path),
        "packed_audit_json": str(packed_audit_path),
        "rows": int(len(manifest)),
        "cvat_rows": int(len(cvat)),
        "cvat_ready_rows": int(cvat_ready),
        "legacy_rows": int(len(legacy)),
        "legacy_ready_rows": int(legacy_ready),
        "available_rows": int(available.sum()),
        "unavailable_rows": int((~available).sum()),
        "duplicate_visual_context_id": int(duplicate_ids),
        "resize_policies": resize_policies,
        "expected_resize_policy": EXPECTED_RESIZE_POLICY,
        "context_kinds": context_kinds,
        "ready_geometry_error_count": int(ready_geometry_errors),
        "checked_cache_tensors": tensor_check["checked_cache_tensors"],
        "packed_tensor_shape": packed_check["packed_tensor_shape"],
        "packed_tensor_dtype": packed_check["packed_tensor_dtype"],
        "packed_index_rows": packed_check["packed_index_rows"],
        "packed_audit_valid": packed_audit.get("valid"),
        "label_gated": build_audit.get("label_gated"),
        "rows_dropped_for_missing_context": build_audit.get("rows_dropped_for_missing_context"),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _missing_audit(cache_dir: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2_visual_interaction_cache_audit_v1",
        "cache_dir": str(cache_dir),
        "rows": 0,
        "cvat_rows": 0,
        "cvat_ready_rows": 0,
        "legacy_rows": 0,
        "legacy_ready_rows": 0,
        "available_rows": 0,
        "unavailable_rows": 0,
        "duplicate_visual_context_id": 0,
        "resize_policies": [],
        "expected_resize_policy": EXPECTED_RESIZE_POLICY,
        "context_kinds": [],
        "ready_geometry_error_count": 0,
        "checked_cache_tensors": 0,
        "packed_tensor_shape": [],
        "packed_tensor_dtype": "",
        "packed_index_rows": 0,
        "packed_audit_valid": False,
        "label_gated": None,
        "rows_dropped_for_missing_context": None,
        "warnings": [],
        "errors": errors,
        "valid": False,
    }


def _check_individual_tensors(
    cache_dir: Path,
    manifest: pd.DataFrame,
    available: pd.Series,
    sample_tensors: int,
) -> dict[str, Any]:
    """Sample individual .npy tensors so manifest paths are not just nominal."""

    errors: list[str] = []
    checked = 0
    for row in manifest[available].head(max(0, sample_tensors)).itertuples(index=False):
        path = cache_dir / str(row.cache_path)
        if not path.exists():
            errors.append(f"missing_cache_file={path}")
            continue
        value = np.load(path, mmap_mode="r")
        if (
            value.dtype != np.uint8
            or value.ndim != 3
            or value.shape[-1] != 3
            or value.shape[0] != value.shape[1]
        ):
            errors.append(f"invalid_cache_tensor={path}:{value.shape}:{value.dtype}")
        checked += 1
    return {"checked_cache_tensors": int(checked), "errors": errors}


def _check_packed_cache(
    *,
    packed_tensor_path: Path,
    packed_index_path: Path,
    packed_audit: dict[str, Any],
    available_rows: int,
) -> dict[str, Any]:
    """Verify packed visual cache is present and row-aligned to available rows."""

    errors: list[str] = []
    shape: list[int] = []
    dtype = ""
    index_rows = 0
    if not packed_tensor_path.exists():
        errors.append(f"missing_packed_tensor={packed_tensor_path}")
    if not packed_index_path.exists():
        errors.append(f"missing_packed_index={packed_index_path}")
    if errors:
        return {
            "packed_tensor_shape": shape,
            "packed_tensor_dtype": dtype,
            "packed_index_rows": index_rows,
            "errors": errors,
        }
    tensor = np.load(packed_tensor_path, mmap_mode="r")
    index = pd.read_csv(packed_index_path, low_memory=False)
    shape = [int(value) for value in tensor.shape]
    dtype = str(tensor.dtype)
    index_rows = int(len(index))
    if tensor.dtype != np.uint8 or len(tensor.shape) != 4 or tensor.shape[-1] != 3:
        errors.append(f"invalid_packed_tensor={shape}:{dtype}")
    if tensor.shape[0] != available_rows or len(index) != available_rows:
        errors.append(
            "packed_available_row_mismatch="
            f"tensor:{tensor.shape[0]}:index:{len(index)}:"
            f"available:{available_rows}"
        )
    if packed_audit.get("valid") is not True:
        errors.append(f"packed_visual_context_audit_invalid={packed_tensor_path.parent}")
    if packed_audit.get("masked_unavailable_rows") is None:
        errors.append("packed_visual_context_missing_masked_unavailable_rows")
    return {
        "packed_tensor_shape": shape,
        "packed_tensor_dtype": dtype,
        "packed_index_rows": index_rows,
        "errors": errors,
    }


def _ready_geometry_error_count(manifest: pd.DataFrame, available: pd.Series) -> int:
    required = ["union_x1", "union_y1", "union_x2", "union_y2", "partner_track_id"]
    if any(col not in manifest.columns for col in required):
        return int(available.sum())
    ready = manifest[available].copy()
    if ready.empty:
        return 0
    x1 = pd.to_numeric(ready["union_x1"], errors="coerce")
    y1 = pd.to_numeric(ready["union_y1"], errors="coerce")
    x2 = pd.to_numeric(ready["union_x2"], errors="coerce")
    y2 = pd.to_numeric(ready["union_y2"], errors="coerce")
    partner_missing = ready["partner_track_id"].fillna("").astype(str).eq("")
    invalid = x1.isna() | y1.isna() | x2.isna() | y2.isna() | x2.le(x1) | y2.le(y1)
    return int((invalid | partner_missing).sum())


def _read_optional_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing_optional_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _unique_strings(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(df[col].fillna("").astype(str).unique().tolist())


if __name__ == "__main__":
    main()
