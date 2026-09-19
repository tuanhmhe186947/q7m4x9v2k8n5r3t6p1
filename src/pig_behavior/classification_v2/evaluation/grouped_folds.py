"""Leakage-safe grouped outer/inner folds for the classification_v2 Q2 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

GROUPED_FOLD_ALGORITHM_VERSION = "q2_grouped_greedy_v1"


@dataclass(slots=True)
class GroupedFoldResult:
    """One-row OOF assignments plus expanded train/validation/test roles."""

    assignments: pd.DataFrame
    roles: pd.DataFrame
    audit: dict[str, Any]


def build_grouped_folds(
    native_units: pd.DataFrame,
    *,
    requested_folds: int = 5,
    seed: int = 20260710,
) -> GroupedFoldResult:
    """Assign recording dates without treating annotation-local pig IDs as identities."""

    required = {
        "temporal_unit_key",
        "recording_group_id",
        "behavior_label",
        "native_unit_valid_for_main_eval",
        "source_type",
        "video_key",
    }
    missing = sorted(required.difference(native_units.columns))
    if missing:
        raise ValueError(f"Q2 grouped-fold input missing columns: {missing}")
    columns = [
        "temporal_unit_key",
        "recording_group_id",
        "behavior_label",
        "native_unit_valid_for_main_eval",
        "source_type",
        "video_key",
    ]
    work = native_units[columns].copy()
    if work["temporal_unit_key"].duplicated().any():
        raise ValueError("duplicate temporal_unit_key in Q2 grouped-fold input")
    work["recording_group_id"] = work["recording_group_id"].astype(str)
    work["behavior_label"] = work["behavior_label"].astype(str)
    group_count = int(work["recording_group_id"].nunique())
    valid = work.loc[_to_bool(work["native_unit_valid_for_main_eval"])]
    class_group_support = (
        valid.groupby("behavior_label")["recording_group_id"]
        .nunique()
        .reindex(VALID_BEHAVIORS, fill_value=0)
    )
    fold_count = min(requested_folds, group_count, int(class_group_support.min()))
    if fold_count < 3:
        raise ValueError(f"at least three recording groups are required, observed={group_count}")
    group_table = _group_table(valid)
    group_to_fold = _greedy_assignment(group_table, fold_count=fold_count, seed=seed)
    assignments = work.copy()
    assignments["outer_fold_id"] = assignments["recording_group_id"].map(group_to_fold)
    assignments["oof_fold_id"] = assignments["outer_fold_id"]
    assignments["oof_role"] = "test"
    assignments = assignments.sort_values(["outer_fold_id", "temporal_unit_key"]).reset_index(drop=True)
    roles = _expand_roles(assignments, fold_count)
    audit = audit_grouped_folds(assignments, roles, requested_folds=requested_folds, seed=seed)
    audit["class_recording_group_support"] = class_group_support.astype(int).to_dict()
    audit["fold_count_reduced_for_class_coverage"] = fold_count < requested_folds
    return GroupedFoldResult(assignments=assignments, roles=roles, audit=audit)


def audit_grouped_folds(
    assignments: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    requested_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Prove one OOF assignment and zero group/video/unit leakage in every fold."""

    errors: list[str] = []
    duplicate_units = int(assignments["temporal_unit_key"].duplicated().sum())
    if duplicate_units:
        errors.append(f"duplicate_oof_temporal_unit_key={duplicate_units}")
    fold_ids = sorted(assignments["outer_fold_id"].astype(str).unique())
    fold_audits: dict[str, Any] = {}
    for fold_id in fold_ids:
        fold_roles = roles.loc[roles["outer_fold_id"].eq(fold_id)]
        role_sets = {
            role: set(fold_roles.loc[fold_roles["role"].eq(role), "recording_group_id"].astype(str))
            for role in ("train", "validation", "test")
        }
        overlaps = {
            "train_validation": sorted(role_sets["train"] & role_sets["validation"]),
            "train_test": sorted(role_sets["train"] & role_sets["test"]),
            "validation_test": sorted(role_sets["validation"] & role_sets["test"]),
        }
        if any(overlaps.values()):
            errors.append(f"recording_group_overlap={fold_id}:{overlaps}")
        leakage_counts = {
            key: _role_overlap_counts(fold_roles, key)
            for key in ["recording_group_id", "video_key", "temporal_unit_key"]
        }
        if any(any(counts.values()) for counts in leakage_counts.values()):
            errors.append(f"entity_role_overlap={fold_id}:{leakage_counts}")
        test = assignments.loc[assignments["outer_fold_id"].eq(fold_id)]
        valid_test = test.loc[_to_bool(test["native_unit_valid_for_main_eval"])]
        label_counts = valid_test["behavior_label"].value_counts().reindex(VALID_BEHAVIORS, fill_value=0)
        fold_audits[fold_id] = {
            "recording_groups": sorted(test["recording_group_id"].astype(str).unique()),
            "videos": int(test["video_key"].nunique()),
            "sources": test["source_type"].value_counts().sort_index().to_dict(),
            "native_units": int(len(test)),
            "valid_units": int(len(valid_test)),
            "behavior_counts": label_counts.astype(int).to_dict(),
            "missing_classes": label_counts[label_counts.eq(0)].index.tolist(),
            "source_by_behavior": {
                behavior: {source: int(count) for source, count in source_counts.items()}
                for behavior, source_counts in pd.crosstab(
                    valid_test["source_type"], valid_test["behavior_label"]
                ).to_dict().items()
            },
            "role_group_counts": {role: len(values) for role, values in role_sets.items()},
            "group_overlap": overlaps,
            "leakage_overlap_counts": leakage_counts,
        }
    unsupported = {
        label: sorted(
            fold_id
            for fold_id, details in fold_audits.items()
            if label in details["missing_classes"]
        )
        for label in VALID_BEHAVIORS
    }
    unsupported = {label: folds for label, folds in unsupported.items() if folds}
    return {
        "schema_version": "classification_v2_q2_grouped_fold_audit_v1",
        "algorithm_version": GROUPED_FOLD_ALGORITHM_VERSION,
        "seed": seed,
        "requested_fold_count": requested_folds,
        "selected_fold_count": len(fold_ids),
        "recording_group_count": int(assignments["recording_group_id"].nunique()),
        "native_unit_rows": int(len(assignments)),
        "duplicate_temporal_unit_key": duplicate_units,
        "every_unit_has_one_oof_fold": bool(assignments["outer_fold_id"].notna().all()),
        "folds": fold_audits,
        "unsupported_class_fold_combinations": unsupported,
        "pig_id_used_as_cross_video_group": False,
        "errors": errors,
        "valid": not errors,
    }


def _group_table(valid: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(valid["recording_group_id"], valid["behavior_label"]).reindex(
        columns=VALID_BEHAVIORS, fill_value=0
    )
    counts["_rows"] = counts.sum(axis=1)
    return counts


def _greedy_assignment(group_table: pd.DataFrame, *, fold_count: int, seed: int) -> dict[str, str]:
    """Greedily minimize normalized class and row imbalance with deterministic ties."""

    rng = np.random.default_rng(seed)
    class_columns = list(VALID_BEHAVIORS)
    global_counts = group_table[class_columns].sum(axis=0).to_numpy(dtype=float)
    rarity = (group_table[class_columns] / np.maximum(global_counts, 1.0)).sum(axis=1)
    tie = pd.Series(rng.random(len(group_table)), index=group_table.index)
    order = sorted(
        group_table.index,
        key=lambda group: (-float(rarity[group]), -int(group_table.loc[group, "_rows"]), float(tie[group])),
    )
    fold_classes = np.zeros((fold_count, len(class_columns)), dtype=float)
    fold_rows = np.zeros(fold_count, dtype=float)
    numeric_assignment: dict[str, int] = {}
    # A class present in exactly K groups must place those groups in distinct
    # folds; otherwise K-fold coverage is mathematically possible but a purely
    # soft balance objective may still miss it.
    supports = (group_table[class_columns] > 0).sum(axis=0).sort_values()
    for label in supports.index:
        if int(supports[label]) != fold_count:
            continue
        groups = sorted(group_table.index[group_table[label].gt(0)])
        used = {numeric_assignment[group] for group in groups if group in numeric_assignment}
        for group in groups:
            if group in numeric_assignment:
                continue
            available = [fold for fold in range(fold_count) if fold not in used]
            selected = min(available, key=lambda fold: (fold_rows[fold], fold))
            numeric_assignment[str(group)] = selected
            used.add(selected)
            vector = group_table.loc[group, class_columns].to_numpy(dtype=float)
            fold_classes[selected] += vector
            fold_rows[selected] += float(group_table.loc[group, "_rows"])
    for group in order:
        if str(group) in numeric_assignment:
            continue
        vector = group_table.loc[group, class_columns].to_numpy(dtype=float)
        rows = float(group_table.loc[group, "_rows"])
        candidates = range(fold_count)
        scores = []
        for fold in candidates:
            candidate_classes = fold_classes.copy()
            candidate_rows = fold_rows.copy()
            candidate_classes[fold] += vector
            candidate_rows[fold] += rows
            class_balance = np.std(candidate_classes / np.maximum(global_counts, 1.0), axis=0).mean()
            row_balance = np.std(candidate_rows / max(float(group_table["_rows"].sum()), 1.0))
            scores.append((float(class_balance + 0.25 * row_balance), float(candidate_rows[fold]), fold))
        selected = min(scores)[2]
        fold_classes[selected] += vector
        fold_rows[selected] += rows
        numeric_assignment[str(group)] = selected
    return {group: f"q2_outer_{fold:02d}" for group, fold in numeric_assignment.items()}


def _expand_roles(assignments: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    fold_ids = [f"q2_outer_{index:02d}" for index in range(fold_count)]
    parts = []
    for index, fold_id in enumerate(fold_ids):
        validation_fold = fold_ids[(index + 1) % fold_count]
        part = assignments.copy()
        part["outer_fold_id"] = fold_id
        part["role"] = np.where(
            part["oof_fold_id"].eq(fold_id),
            "test",
            np.where(part["oof_fold_id"].eq(validation_fold), "validation", "train"),
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["outer_fold_id", "role", "temporal_unit_key"]
    ).reset_index(drop=True)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _role_overlap_counts(frame: pd.DataFrame, key: str) -> dict[str, int]:
    values = {
        role: set(frame.loc[frame["role"].eq(role), key].astype(str))
        for role in ("train", "validation", "test")
    }
    return {
        "train_validation": len(values["train"] & values["validation"]),
        "train_test": len(values["train"] & values["test"]),
        "validation_test": len(values["validation"] & values["test"]),
    }
