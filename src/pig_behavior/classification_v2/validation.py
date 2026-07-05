"""Validation helpers for classification_v2."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pig_behavior.classification_v2.features.review_policy import audit_review_policy


def validate_reviewed_frame_features(df: pd.DataFrame) -> dict[str, Any]:
    """Validate frame-level features after review policy and before sequence building."""
    audit = audit_review_policy(df)

    errors = list(audit.get("errors", []))
    warnings = list(audit.get("warnings", []))

    required = [
        "behavior_train",
        "label_strength",
        "ambiguity_group",
        "review_decision",
        "training_action_final",
        "training_weight_final",
        "include_in_training_final",
        "use_for_main_train_final",
        "use_for_robust_train_final",
        "use_for_roi_training_final",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing_reviewed_feature_columns={missing}")

    if "include_in_training_final" in df.columns and "bbox_valid" in df.columns:
        include = _to_bool_series(df["include_in_training_final"])
        bbox_valid = _to_bool_series(df["bbox_valid"])
        included_invalid_bbox = int((include & ~bbox_valid).sum())
        if included_invalid_bbox:
            errors.append(f"included_invalid_bbox_rows={included_invalid_bbox}")

    if "behavior_train" in df.columns:
        missing_behavior_train = int(
            df["behavior_train"].isna().sum()
            + df["behavior_train"].astype(str).str.strip().eq("").sum()
        )
        if missing_behavior_train:
            errors.append(f"missing_behavior_train={missing_behavior_train}")

    audit["errors"] = errors
    audit["warnings"] = warnings
    return audit


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