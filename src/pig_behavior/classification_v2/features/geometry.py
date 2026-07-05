"""Geometry feature builder for classification_v2.

Input:
    merged_frame_objects/frame_object_annotations_policy.csv

Output:
    Same rows, with bbox geometry recomputed and audited.

This step does not drop rows and does not change context policy.
It only recomputes geometry-related columns from x1/y1/x2/y2/image_width/image_height.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import GEOMETRY_FEATURE_COLUMNS

REQUIRED_GEOMETRY_INPUT_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "bbox_valid",
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
)


def build_geometry_features(frame_objects: pd.DataFrame) -> pd.DataFrame:
    """Recompute bbox geometry features.

    Rules:
    - Do not drop any row.
    - Do not reject actor-only rows.
    - Do not use hidden to affect geometry.
    - Recompute all geometry from bbox and image size.
    - Preserve training/context policy columns from previous step.
    """
    missing = [
        col for col in REQUIRED_GEOMETRY_INPUT_COLUMNS if col not in frame_objects.columns
    ]
    if missing:
        raise ValueError(f"Missing geometry input columns: {missing}")

    out = frame_objects.copy()

    numeric_cols = [
        "x1_raw",
        "y1_raw",
        "x2_raw",
        "y2_raw",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "cx",
        "cy",
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "aspect_ratio",
        "box_diag",
        "box_diag_n",
        "box_compactness",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "x1_raw" not in out.columns:
        out["x1_raw"] = out["x1"]
    if "y1_raw" not in out.columns:
        out["y1_raw"] = out["y1"]
    if "x2_raw" not in out.columns:
        out["x2_raw"] = out["x2"]
    if "y2_raw" not in out.columns:
        out["y2_raw"] = out["y2"]

    image_width = out["image_width"].replace(0, np.nan)
    image_height = out["image_height"].replace(0, np.nan)

    out["bbox_w"] = out["x2"] - out["x1"]
    out["bbox_h"] = out["y2"] - out["y1"]
    out["bbox_area"] = out["bbox_w"] * out["bbox_h"]

    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0

    out["cx_n"] = out["cx"] / image_width
    out["cy_n"] = out["cy"] / image_height
    out["bw_n"] = out["bbox_w"] / image_width
    out["bh_n"] = out["bbox_h"] / image_height
    out["area_n"] = out["bbox_area"] / (image_width * image_height)

    out["aspect_ratio"] = out["bbox_w"] / out["bbox_h"].replace(0, np.nan)
    out["box_diag"] = np.sqrt(out["bbox_w"] ** 2 + out["bbox_h"] ** 2)

    image_diag = np.sqrt(image_width**2 + image_height**2)
    out["box_diag_n"] = out["box_diag"] / image_diag

    # Compactness is a stable shape descriptor:
    # larger when area is large relative to diagonal squared.
    out["box_compactness"] = out["area_n"] / (out["box_diag_n"] ** 2).replace(
        0,
        np.nan,
    )

    original_bbox_valid = _to_bool_series(out["bbox_valid"])

    computed_bbox_valid = (
        out["x1"].notna()
        & out["y1"].notna()
        & out["x2"].notna()
        & out["y2"].notna()
        & out["image_width"].notna()
        & out["image_height"].notna()
        & out["bbox_w"].gt(0)
        & out["bbox_h"].gt(0)
        & out["x1"].ge(0)
        & out["y1"].ge(0)
        & out["x2"].le(out["image_width"])
        & out["y2"].le(out["image_height"])
    )

    out["bbox_valid"] = original_bbox_valid & computed_bbox_valid
    out["actor_bbox_valid"] = out["bbox_valid"]

    out["actor_quality"] = "valid"
    out.loc[~out["actor_bbox_valid"], "actor_quality"] = "invalid_bbox"

    out["bbox_was_clipped"] = (
        out["x1"].ne(out["x1_raw"])
        | out["y1"].ne(out["y1_raw"])
        | out["x2"].ne(out["x2_raw"])
        | out["y2"].ne(out["y2_raw"])
    )

    geometry_cols = list(GEOMETRY_FEATURE_COLUMNS)
    for col in geometry_cols:
        if col not in out.columns:
            out[col] = np.nan

    finite_matrix = np.isfinite(out[geometry_cols].to_numpy(dtype=float))
    out["geometry_nan_count"] = (~finite_matrix).sum(axis=1)
    out["geometry_feature_valid"] = out["bbox_valid"] & out["geometry_nan_count"].eq(0)

    out["geometry_quality"] = "ok"
    out.loc[~out["bbox_valid"], "geometry_quality"] = "invalid_bbox"
    out.loc[
        out["bbox_valid"] & out["geometry_nan_count"].gt(0),
        "geometry_quality",
    ] = "geometry_nan_or_inf"

    return out


def validate_geometry_features(df: pd.DataFrame) -> dict[str, Any]:
    """Audit geometry output."""
    errors: list[str] = []
    warnings: list[str] = []

    required = set(REQUIRED_GEOMETRY_INPUT_COLUMNS)
    required.update(GEOMETRY_FEATURE_COLUMNS)
    required.update(
        {
            "bbox_w",
            "bbox_h",
            "bbox_area",
            "cx",
            "cy",
            "box_diag",
            "bbox_valid",
            "actor_bbox_valid",
            "geometry_feature_valid",
            "geometry_nan_count",
            "geometry_quality",
        }
    )

    missing = sorted(required.difference(df.columns))
    if missing:
        errors.append(f"missing_columns={missing}")

    bbox_valid = _to_bool_series(df["bbox_valid"]) if "bbox_valid" in df.columns else None
    geometry_valid = (
        _to_bool_series(df["geometry_feature_valid"])
        if "geometry_feature_valid" in df.columns
        else None
    )

    invalid_bbox = int((~bbox_valid).sum()) if bbox_valid is not None else -1
    invalid_geometry = (
        int((~geometry_valid).sum()) if geometry_valid is not None else -1
    )

    if invalid_bbox > 0:
        warnings.append(f"invalid_bbox_count={invalid_bbox}")

    if invalid_geometry > 0:
        warnings.append(f"invalid_geometry_count={invalid_geometry}")

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique()) if "frame_uid" in df.columns else 0,
        "sources": _value_counts_dict(df, "source_type"),
        "datasets": _value_counts_dict(df, "dataset_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "bbox_valid": _value_counts_dict(df, "bbox_valid"),
        "actor_bbox_valid": _value_counts_dict(df, "actor_bbox_valid"),
        "geometry_feature_valid": _value_counts_dict(df, "geometry_feature_valid"),
        "geometry_quality": _value_counts_dict(df, "geometry_quality"),
        "invalid_bbox": invalid_bbox,
        "invalid_geometry": invalid_geometry,
        "geometry_nan_count": _value_counts_dict(df, "geometry_nan_count"),
        "errors": errors,
        "warnings": warnings,
    }


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)

    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", ""}

    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return False

    return series.map(parse).astype(bool)


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}