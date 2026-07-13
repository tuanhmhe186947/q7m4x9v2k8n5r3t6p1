"""Merge canonical frame-object sources for classification_v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    CANONICAL_FRAME_OBJECT_COLUMNS,
    SOURCE_TYPES,
    VALID_BEHAVIOR_SET,
)


def merge_frame_object_sources(
    frames: list[pd.DataFrame],
    *,
    source_names: list[str] | None = None,
    strict_schema: bool = True,
    sort_output: bool = True,
) -> pd.DataFrame:
    """Merge canonical frame-object dataframes by stacking rows."""
    if not frames:
        return pd.DataFrame(columns=CANONICAL_FRAME_OBJECT_COLUMNS)

    names = source_names or [f"source_{idx}" for idx in range(len(frames))]
    if len(names) != len(frames):
        raise ValueError("source_names must have the same length as frames.")

    normalized = []
    for name, frame_df in zip(names, frames, strict=True):
        normalized.append(
            normalize_canonical_frame_objects(
                frame_df,
                source_name=name,
                strict_schema=strict_schema,
            )
        )

    expected_rows = sum(len(frame) for frame in normalized)
    merged = pd.concat(normalized, ignore_index=True)
    if len(merged) != expected_rows:
        raise RuntimeError(
            "Merging canonical sources changed row count: "
            f"expected={expected_rows}, actual={len(merged)}"
        )

    if sort_output and not merged.empty:
        sort_cols = [
            col
            for col in [
                "source_type",
                "dataset_id",
                "video_key",
                "frame_index",
                "frame_uid",
                "object_id_in_image",
                "track_id",
                "pig_id",
            ]
            if col in merged.columns
        ]
        merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return merged[CANONICAL_FRAME_OBJECT_COLUMNS]


def normalize_canonical_frame_objects(
    df: pd.DataFrame,
    *,
    source_name: str,
    strict_schema: bool,
) -> pd.DataFrame:
    """Ensure a dataframe follows the canonical frame-object schema."""
    out = df.copy()

    missing = [col for col in CANONICAL_FRAME_OBJECT_COLUMNS if col not in out.columns]
    if missing and strict_schema:
        raise ValueError(
            f"{source_name} is missing canonical columns: {missing}. "
            "Fix the source parser or run with strict_schema=False."
        )

    for col in missing:
        out[col] = pd.NA

    extra = [col for col in out.columns if col not in CANONICAL_FRAME_OBJECT_COLUMNS]
    if extra:
        out = out.drop(columns=extra)

    return out[CANONICAL_FRAME_OBJECT_COLUMNS]


def audit_merged_frame_objects(df: pd.DataFrame) -> dict[str, Any]:
    """Return audit information for merged canonical frame objects."""
    if df.empty:
        return {
            "rows": 0,
            "frames": 0,
            "sources": {},
            "datasets": {},
            "videos": {},
            "behaviors": {},
            "context_pig_count": {},
            "errors": ["empty_dataframe"],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [
        col for col in CANONICAL_FRAME_OBJECT_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        errors.append(f"missing_columns={missing_columns}")

    invalid_source_types = sorted(
        set(df["source_type"].dropna().astype(str)).difference(SOURCE_TYPES)
    )
    if invalid_source_types:
        errors.append(f"invalid_source_types={invalid_source_types}")

    behaviors = set(df["behavior"].dropna().astype(str))
    invalid_behaviors = sorted(
        behavior
        for behavior in behaviors
        if behavior and behavior not in VALID_BEHAVIOR_SET
    )
    if invalid_behaviors:
        warnings.append(f"invalid_or_unknown_behaviors={invalid_behaviors}")

    invalid_bbox_count = 0
    if "bbox_valid" in df.columns:
        bbox_valid = df["bbox_valid"].astype(str).str.lower().isin(["true", "1"])
        invalid_bbox_count = int((~bbox_valid).sum())
    else:
        errors.append("bbox_valid_column_missing")

    duplicate_object_rows = _duplicate_object_row_count(df)
    if duplicate_object_rows:
        errors.append(f"duplicate_frame_object_rows={duplicate_object_rows}")

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique(dropna=True)),
        "sources": _value_counts_dict(df, "source_type"),
        "datasets": _value_counts_dict(df, "dataset_id"),
        "videos": _value_counts_dict(df, "video_key"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "context_pig_count": _value_counts_dict(df, "global_context_pig_count"),
        "annotation_scope": _value_counts_dict(df, "annotation_scope"),
        "social_feature_quality": _value_counts_dict(df, "social_feature_quality"),
        "training_tier": _value_counts_dict(df, "training_tier"),
        "qa_status": _value_counts_dict(df, "qa_status"),
        "bbox_valid": _value_counts_dict(df, "bbox_valid"),
        "invalid_bbox_count": invalid_bbox_count,
        "duplicate_frame_object_rows": duplicate_object_rows,
        "errors": errors,
        "warnings": warnings,
    }


def save_merged_frame_objects(
    df: pd.DataFrame,
    output_csv: str | Path,
    *,
    audit_json: str | Path | None = None,
) -> dict[str, Any]:
    """Save merged frame objects and optional audit JSON."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    audit = audit_merged_frame_objects(df)

    if audit_json is not None:
        audit_path = Path(audit_json)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return audit


def read_canonical_frame_object_csv(path: str | Path) -> pd.DataFrame:
    """Read a saved canonical frame-object CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Canonical frame-object CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    return normalize_canonical_frame_objects(
        df,
        source_name=str(csv_path),
        strict_schema=True,
    )


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return value counts as a JSON-friendly dictionary."""
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _duplicate_object_row_count(df: pd.DataFrame) -> int:
    """Count ambiguous duplicate actor observations within one source frame."""

    preferred = [
        "source_type",
        "dataset_id",
        "video_key",
        "frame_index",
        "track_id",
        "pig_id",
    ]
    if not set(preferred).issubset(df.columns):
        return 0
    key = df[preferred].copy()
    for column in preferred:
        key[column] = key[column].fillna("").astype(str).str.strip()
    return int(key.duplicated(keep=False).sum())
