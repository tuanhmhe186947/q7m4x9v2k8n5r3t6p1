"""Native temporal-unit out-of-fold assignment for classification_v2.

Each recording group is held out exactly once. The resulting manifest is a
prediction/evaluation contract, not a training run: downstream trainers should
train on all other groups and predict the held-out group for that fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class NativeOOFFoldResult:
    manifest: pd.DataFrame
    audit: dict[str, Any]


def build_native_oof_folds(native_split_manifest: pd.DataFrame) -> NativeOOFFoldResult:
    """Assign every native temporal unit to one label-safe held-out fold."""
    required = ["temporal_unit_key", "recording_group_id", "behavior_label", "native_unit_valid_for_main_eval"]
    missing = [col for col in required if col not in native_split_manifest.columns]
    if missing:
        raise ValueError(f"native split manifest missing columns: {missing}")
    passthrough = [col for col in ["source_type", "video_key"] if col in native_split_manifest]
    work = native_split_manifest[required + passthrough].copy()
    work["temporal_unit_key"] = work["temporal_unit_key"].astype(str)
    work["recording_group_id"] = work["recording_group_id"].astype(str)
    groups = sorted(work["recording_group_id"].dropna().astype(str).unique())
    fold_map = {group: f"native_oof_{idx:03d}" for idx, group in enumerate(groups)}
    work["oof_fold_id"] = work["recording_group_id"].map(fold_map)
    work["oof_test_group_key"] = work["recording_group_id"]
    work["oof_role"] = "test"
    work["oof_train_group_count"] = len(groups) - 1
    ordered_cols = [
        "temporal_unit_key",
        "recording_group_id",
        "oof_fold_id",
        "oof_test_group_key",
        "oof_role",
        "oof_train_group_count",
        "behavior_label",
        "native_unit_valid_for_main_eval",
    ]
    ordered_cols.extend([c for c in ["source_type", "video_key"] if c in work.columns])
    manifest = work[ordered_cols].sort_values(["oof_fold_id", "temporal_unit_key"]).reset_index(drop=True)
    audit = audit_native_oof_folds(manifest)
    return NativeOOFFoldResult(manifest=manifest, audit=audit)


def audit_native_oof_folds(manifest: pd.DataFrame) -> dict[str, Any]:
    """Validate that each recording group maps to one fold and every unit appears once."""
    errors: list[str] = []
    duplicate_units = int(manifest["temporal_unit_key"].duplicated().sum()) if "temporal_unit_key" in manifest else 0
    group_fold_counts = (
        manifest.groupby("recording_group_id")["oof_fold_id"].nunique().to_dict()
        if {"recording_group_id", "oof_fold_id"}.issubset(manifest.columns)
        else {}
    )
    leaking_groups = sorted(str(group) for group, count in group_fold_counts.items() if int(count) != 1)
    fold_group_counts = (
        manifest.groupby("oof_fold_id")["recording_group_id"].nunique().to_dict()
        if {"recording_group_id", "oof_fold_id"}.issubset(manifest.columns)
        else {}
    )
    multi_group_folds = sorted(str(fold) for fold, count in fold_group_counts.items() if int(count) != 1)
    if duplicate_units:
        errors.append(f"duplicate_temporal_unit_key={duplicate_units}")
    if leaking_groups:
        errors.append(f"recording_group_in_multiple_folds={len(leaking_groups)}")
    if multi_group_folds:
        errors.append(f"fold_holds_multiple_groups={len(multi_group_folds)}")
    fold_count = int(manifest["oof_fold_id"].nunique()) if "oof_fold_id" in manifest else 0
    if fold_count < 3:
        errors.append(f"too_few_oof_folds={fold_count}")
    return {
        "rows": int(len(manifest)),
        "fold_count": fold_count,
        "recording_group_count": int(manifest["recording_group_id"].nunique())
        if "recording_group_id" in manifest
        else 0,
        "duplicate_temporal_unit_key": duplicate_units,
        "fold_rows": manifest["oof_fold_id"].value_counts(dropna=False).sort_index().to_dict()
        if "oof_fold_id" in manifest
        else {},
        "valid_eval_rows_by_fold": manifest[_to_bool(manifest["native_unit_valid_for_main_eval"])]
        .groupby("oof_fold_id")["temporal_unit_key"]
        .count()
        .to_dict()
        if {"native_unit_valid_for_main_eval", "oof_fold_id", "temporal_unit_key"}.issubset(manifest.columns)
        else {},
        "label_counts": manifest["behavior_label"].value_counts(dropna=False).to_dict()
        if "behavior_label" in manifest
        else {},
        "warnings": [
            "OOF folds are grouped by recording_group_id; pig_id is not treated as cross-video biological identity",
            "Use native temporal-unit macro F1 as primary metric; window metrics are secondary diagnostics",
        ],
        "errors": errors,
    }


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
