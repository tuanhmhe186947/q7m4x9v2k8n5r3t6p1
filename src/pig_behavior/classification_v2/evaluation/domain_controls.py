"""Source/domain control views that preserve every classification_v2 window."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pig_behavior.classification_v2.contracts.model_io import forbidden_x_columns
from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

LABEL_INDEPENDENT_AVAILABILITY_COLUMNS = (
    "window_image_context_complete",
    "scene_context_ready",
    "scene_partner_context_ready",
)
TARGET_DERIVED_AVAILABILITY_COLUMNS = frozenset(
    {
        "interaction_context_ready",
        "interaction_context_required",
        "is_interaction_window",
    }
)


def build_source_matched_views(windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark source and matched-length controls without dropping or relabeling rows."""

    required = {
        "window_id",
        "source_type",
        "behavior_window_label",
        "window_length_frames",
        "window_valid_for_main_train",
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"source-matched view input missing columns: {missing}")
    if windows["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in source-matched view input")
    result = windows.copy()
    valid = _to_bool(result["window_valid_for_main_train"])
    source = result["source_type"].astype(str)
    length = pd.to_numeric(result["window_length_frames"], errors="coerce")
    result["view_combined"] = valid
    result["view_cvat_only"] = valid & source.eq("cvat_tracking_xml")
    result["view_legacy_only"] = valid & source.eq("legacy_recovered")
    result["view_matched_6frame"] = valid & length.eq(6)
    result["source_class_balance_keep"] = False
    result["source_class_balance_rank"] = 0
    result["source_class_balance_quota"] = 0
    candidates = result.loc[result["view_matched_6frame"]].copy()
    sources = sorted(candidates["source_type"].astype(str).unique())
    quotas: dict[str, int] = {}
    counts = candidates.groupby(["behavior_window_label", "source_type"])["window_id"].count()
    for label in sorted(candidates["behavior_window_label"].astype(str).unique()):
        values = [int(counts.get((label, source_name), 0)) for source_name in sources]
        quotas[label] = (
            min(values)
            if len(sources) >= 2 and all(value > 0 for value in values)
            else 0
        )
    ordered = candidates.sort_values(
        ["behavior_window_label", "source_type", "window_id"], kind="mergesort"
    )
    ranks = ordered.groupby(["behavior_window_label", "source_type"]).cumcount() + 1
    result.loc[ordered.index, "source_class_balance_rank"] = ranks.astype(int)
    result["source_class_balance_quota"] = (
        result["behavior_window_label"].astype(str).map(quotas).fillna(0).astype(int)
    )
    result["source_class_balance_keep"] = (
        result["view_matched_6frame"]
        & result["source_class_balance_quota"].gt(0)
        & result["source_class_balance_rank"].le(result["source_class_balance_quota"])
    )
    result["source_control_exclusion_reason"] = "not_valid_for_main_train"
    result.loc[valid, "source_control_exclusion_reason"] = "valid_not_in_matched_6frame"
    result.loc[result["view_matched_6frame"], "source_control_exclusion_reason"] = (
        "matched_6frame_above_source_class_quota"
    )
    result.loc[result["source_class_balance_keep"], "source_control_exclusion_reason"] = (
        "source_class_matched_keep"
    )
    audit = {
        "schema_version": "classification_v2_source_matched_views_v1",
        "rows_input": int(len(windows)),
        "rows_output": int(len(result)),
        "duplicate_window_id": int(result["window_id"].duplicated().sum()),
        "source_counts": source.value_counts().sort_index().to_dict(),
        "view_counts": {
            column: int(_to_bool(result[column]).sum())
            for column in [
                "view_combined",
                "view_cvat_only",
                "view_legacy_only",
                "view_matched_6frame",
                "source_class_balance_keep",
            ]
        },
        "matched_6frame_source_behavior_counts": _contingency(
            result.loc[result["view_matched_6frame"]]
        ),
        "source_class_balanced_counts": _contingency(
            result.loc[result["source_class_balance_keep"]]
        ),
        "rows_dropped": 0,
        "labels_changed": 0,
        "errors": [],
        "valid": len(result) == len(windows) and not result["window_id"].duplicated().any(),
    }
    return result, audit


def grouped_source_probe(
    features: pd.DataFrame,
    window_metadata: pd.DataFrame,
    native_mapping: pd.DataFrame,
    grouped_roles: pd.DataFrame,
    *,
    feature_whitelist: list[str] | tuple[str, ...],
    expected_ordered_window_id_sha256: str,
    forbidden_patterns: list[str] | tuple[str, ...] | None = None,
    max_iter: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict source from exact model-X fields at native temporal-unit grain."""

    columns = _validate_exact_model_feature_table(
        features,
        feature_whitelist,
        forbidden_patterns=forbidden_patterns,
    )
    native, input_audit = _prepare_native_probe_table(
        features,
        window_metadata,
        native_mapping,
        feature_columns=columns,
        expected_ordered_window_id_sha256=expected_ordered_window_id_sha256,
    )
    roles, role_audit = _validate_grouped_probe_roles(grouped_roles, native)
    out, fold_audits = _run_grouped_native_probe(
        native,
        roles,
        feature_columns=columns,
        target_column="source_type",
        prediction_column="source_predicted",
        max_iter=max_iter,
    )
    sources = sorted(native["source_type"].astype(str).unique())
    if len(sources) != 2:
        raise ValueError(f"source probe requires exactly two sources, observed={sources}")
    pooled = float(
        balanced_accuracy_score(out["source_type"], out["source_predicted"])
    )
    audit = {
        "schema_version": "classification_v2_grouped_source_probe_v2",
        "statistical_unit": "native_temporal_unit",
        "feature_whitelist": columns,
        "feature_whitelist_sha256": _ordered_names_sha256(columns),
        "feature_count": int(len(columns)),
        "forbidden_pattern_contract": (
            list(forbidden_patterns)
            if forbidden_patterns is not None
            else "model_io_default"
        ),
        "input_contract": input_audit,
        "role_contract": role_audit,
        "eligible_window_rows": input_audit["eligible_window_rows"],
        "eligible_native_unit_rows": int(len(native)),
        "oof_prediction_rows": int(len(out)),
        "oof_fold_count": int(out["outer_fold_id"].nunique()),
        "eligible_window_to_native_row_loss": 0,
        "eligible_native_unit_to_oof_row_loss": int(len(native) - len(out)),
        "every_eligible_native_unit_tested_once": True,
        "pooled_balanced_accuracy": pooled,
        "folds": fold_audits,
        "source_identifier_in_features": False,
        "source_and_group_columns_are_output_metadata_only": True,
        "scaler_and_probe_fit_scope": "grouped_training_native_units_only",
        "interpretation": "internal source/domain shortcut diagnostic, not external generalization",
        "warnings": [
            "High source predictability can indicate source-correlated geometry or missingness."
        ],
        "errors": [],
        "valid": True,
    }
    return out, audit


def grouped_availability_behavior_probe(
    availability: pd.DataFrame,
    window_metadata: pd.DataFrame,
    native_mapping: pd.DataFrame,
    grouped_roles: pd.DataFrame,
    *,
    availability_feature_whitelist: list[str] | tuple[str, ...],
    expected_ordered_window_id_sha256: str,
    max_iter: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure behavior predictability from label-independent availability only."""

    columns, numeric = _validate_availability_feature_table(
        availability,
        availability_feature_whitelist,
    )
    native, input_audit = _prepare_native_probe_table(
        numeric,
        window_metadata,
        native_mapping,
        feature_columns=columns,
        expected_ordered_window_id_sha256=expected_ordered_window_id_sha256,
        feature_window_ids=availability["window_id"],
    )
    roles, role_audit = _validate_grouped_probe_roles(grouped_roles, native)
    out, fold_audits = _run_grouped_native_probe(
        native,
        roles,
        feature_columns=columns,
        target_column="behavior_window_label",
        prediction_column="behavior_predicted",
        max_iter=max_iter,
    )
    true = out["behavior_window_label"].astype(str)
    predicted = out["behavior_predicted"].astype(str)
    supported_labels = sorted(true.unique())
    unknown = sorted(set(supported_labels).difference(VALID_BEHAVIORS))
    if unknown:
        raise ValueError(f"availability probe observed invalid behaviors: {unknown}")
    audit = {
        "schema_version": "classification_v2_availability_behavior_probe_v1",
        "statistical_unit": "native_temporal_unit",
        "diagnostic_only": True,
        "availability_columns": columns,
        "availability_whitelist_sha256": _ordered_names_sha256(columns),
        "target_derived_availability_columns": [],
        "availability_features_enter_classifier_x": False,
        "input_contract": input_audit,
        "role_contract": role_audit,
        "eligible_window_rows": input_audit["eligible_window_rows"],
        "eligible_native_unit_rows": int(len(native)),
        "oof_prediction_rows": int(len(out)),
        "oof_fold_count": int(out["outer_fold_id"].nunique()),
        "eligible_window_to_native_row_loss": 0,
        "eligible_native_unit_to_oof_row_loss": int(len(native) - len(out)),
        "every_eligible_native_unit_tested_once": True,
        "pooled_accuracy": float(accuracy_score(true, predicted)),
        "pooled_macro_f1_supported": float(
            f1_score(
                true,
                predicted,
                labels=supported_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "pooled_macro_f1_global_10": float(
            f1_score(
                true,
                predicted,
                labels=VALID_BEHAVIORS,
                average="macro",
                zero_division=0,
            )
        ),
        "folds": fold_audits,
        "scaler_and_probe_fit_scope": "grouped_training_native_units_only",
        "interpretation": (
            "availability-only behavior shortcut diagnostic; not a behavior model"
        ),
        "errors": [],
        "valid": True,
    }
    return out, audit


def _validate_exact_model_feature_table(
    features: pd.DataFrame,
    feature_whitelist: list[str] | tuple[str, ...],
    *,
    forbidden_patterns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Require the exact ordered trainer whitelist and finite numeric values."""

    columns = _validated_ordered_names(feature_whitelist, "feature_whitelist")
    observed = list(features.columns)
    if len(observed) != len(set(observed)):
        raise ValueError("source probe feature table contains duplicate columns")
    if observed != columns:
        missing = [column for column in columns if column not in observed]
        extra = [column for column in observed if column not in columns]
        raise ValueError(
            "source probe feature schema mismatch: "
            f"missing={missing}, extra={extra}, ordered_match={observed == columns}"
        )
    forbidden = forbidden_x_columns(columns, forbidden_patterns)
    if forbidden:
        raise ValueError(f"source probe whitelist contains forbidden X fields: {forbidden}")
    nonnumeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if nonnumeric:
        raise ValueError(f"source probe features are nonnumeric: {nonnumeric}")
    values = features.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise ValueError(f"source probe feature table has nonfinite_values={nonfinite}")
    return columns


def _validate_availability_feature_table(
    availability: pd.DataFrame,
    availability_feature_whitelist: list[str] | tuple[str, ...],
) -> tuple[list[str], pd.DataFrame]:
    """Allow only named, label-independent binary availability diagnostics."""

    columns = _validated_ordered_names(
        availability_feature_whitelist,
        "availability_feature_whitelist",
    )
    target_derived = sorted(set(columns).intersection(TARGET_DERIVED_AVAILABILITY_COLUMNS))
    if target_derived:
        raise ValueError(
            "availability probe contains target-derived fields: "
            f"{target_derived}"
        )
    unsupported = sorted(set(columns).difference(LABEL_INDEPENDENT_AVAILABILITY_COLUMNS))
    if unsupported:
        raise ValueError(
            "availability probe fields are not registered label-independent masks: "
            f"{unsupported}"
        )
    expected = ["window_id", *columns]
    observed = list(availability.columns)
    if observed != expected:
        missing = [column for column in expected if column not in observed]
        extra = [column for column in observed if column not in expected]
        raise ValueError(
            "availability probe schema mismatch: "
            f"missing={missing}, extra={extra}, ordered_match={observed == expected}"
        )
    numeric = pd.DataFrame(index=availability.index)
    for column in columns:
        numeric[column] = _strict_binary_column(availability[column], column)
    return columns, numeric


def _prepare_native_probe_table(
    feature_values: pd.DataFrame,
    window_metadata: pd.DataFrame,
    native_mapping: pd.DataFrame,
    *,
    feature_columns: list[str],
    expected_ordered_window_id_sha256: str,
    feature_window_ids: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bind ordered windows, then collapse repeated windows to native units."""

    metadata_columns = [
        "window_id",
        "source_type",
        "behavior_window_label",
        "window_valid_for_main_train",
        "split_group_key",
        "video_key",
    ]
    missing_metadata = [
        column for column in metadata_columns if column not in window_metadata
    ]
    mapping_columns = [
        "window_id",
        "temporal_unit_keys_window",
        "num_temporal_units_window",
    ]
    missing_mapping = [
        column for column in mapping_columns if column not in native_mapping
    ]
    if missing_metadata or missing_mapping:
        raise ValueError(
            "native probe input schema mismatch: "
            f"metadata_missing={missing_metadata}, mapping_missing={missing_mapping}"
        )
    if not (
        len(feature_values) == len(window_metadata) == len(native_mapping)
    ):
        raise ValueError(
            "native probe row mismatch: "
            f"features={len(feature_values)}, metadata={len(window_metadata)}, "
            f"mapping={len(native_mapping)}"
        )
    candidates = {"native_mapping": native_mapping["window_id"]}
    if feature_window_ids is not None:
        candidates["feature_window_ids"] = feature_window_ids
    alignment = require_ordered_window_ids(
        "window_metadata",
        window_metadata["window_id"],
        candidates,
    )
    observed_hash = alignment["reference_ordered_window_id_sha256"]
    if not _is_sha256(expected_ordered_window_id_sha256):
        raise ValueError("expected ordered-window SHA256 is malformed")
    if observed_hash != expected_ordered_window_id_sha256:
        raise ValueError(
            "native probe ordered-window lineage mismatch: "
            f"expected={expected_ordered_window_id_sha256}, observed={observed_hash}"
        )

    work = window_metadata.loc[:, metadata_columns].reset_index(drop=True).copy()
    work["window_valid_for_main_train"] = _strict_bool_series(
        work["window_valid_for_main_train"],
        "window_valid_for_main_train",
    )
    for column in [
        "window_id",
        "source_type",
        "behavior_window_label",
        "split_group_key",
        "video_key",
    ]:
        work[column] = work[column].fillna("").astype(str).str.strip()
    eligible = work["window_valid_for_main_train"]
    blank_metadata = {
        column: int(work.loc[eligible, column].eq("").sum())
        for column in [
            "window_id",
            "source_type",
            "behavior_window_label",
            "split_group_key",
            "video_key",
        ]
    }
    blank_metadata = {
        column: count for column, count in blank_metadata.items() if count
    }
    if blank_metadata:
        raise ValueError(f"eligible native probe metadata is blank: {blank_metadata}")
    invalid_behaviors = sorted(
        set(work.loc[eligible, "behavior_window_label"]).difference(VALID_BEHAVIORS)
    )
    if invalid_behaviors:
        raise ValueError(
            f"eligible native probe rows have invalid behaviors: {invalid_behaviors}"
        )

    unit_count = pd.to_numeric(
        native_mapping["num_temporal_units_window"],
        errors="coerce",
    )
    invalid_unit_count = eligible & (
        unit_count.isna() | unit_count.ne(1) | np.floor(unit_count).ne(unit_count)
    )
    if invalid_unit_count.any():
        raise ValueError(
            "eligible source-probe windows must map to exactly one native unit: "
            f"invalid_rows={int(invalid_unit_count.sum())}"
        )
    work["temporal_unit_key"] = (
        native_mapping["temporal_unit_keys_window"]
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy()
    )
    blank_unit_keys = int(work.loc[eligible, "temporal_unit_key"].eq("").sum())
    if blank_unit_keys:
        raise ValueError(
            f"eligible source-probe windows have blank native keys={blank_unit_keys}"
        )
    for column in feature_columns:
        work[column] = feature_values[column].to_numpy(dtype=np.float64)
    selected = work.loc[eligible].copy()
    conflict_columns = [
        "source_type",
        "behavior_window_label",
        "split_group_key",
        "video_key",
    ]
    conflicts = {
        column: int(
            selected.groupby("temporal_unit_key", sort=False)[column]
            .nunique(dropna=False)
            .gt(1)
            .sum()
        )
        for column in conflict_columns
    }
    conflicts = {column: count for column, count in conflicts.items() if count}
    if conflicts:
        raise ValueError(f"native-unit metadata conflicts: {conflicts}")

    grouped = selected.groupby("temporal_unit_key", sort=True)
    native_metadata = grouped[conflict_columns].first().reset_index()
    native_features = grouped[feature_columns].mean().reset_index()
    windows_per_native = grouped.size().rename("windows_per_native_unit").reset_index()
    native = native_metadata.merge(
        native_features,
        on="temporal_unit_key",
        how="inner",
        validate="one_to_one",
    ).merge(
        windows_per_native,
        on="temporal_unit_key",
        how="inner",
        validate="one_to_one",
    )
    if native["temporal_unit_key"].duplicated().any():
        raise ValueError("native probe aggregation emitted duplicate temporal_unit_key")
    represented_windows = int(native["windows_per_native_unit"].sum())
    if represented_windows != len(selected):
        raise ValueError(
            "native probe aggregation lost eligible windows: "
            f"eligible={len(selected)}, represented={represented_windows}"
        )
    audit = {
        "schema_version": "classification_v2_native_probe_input_v1",
        "ordered_window_id_sha256_expected": expected_ordered_window_id_sha256,
        "ordered_window_id_sha256_observed": observed_hash,
        "ordered_window_lineage_match": True,
        "window_alignment": alignment,
        "input_window_rows": int(len(work)),
        "invalid_for_main_train_window_rows": int((~eligible).sum()),
        "eligible_window_rows": int(eligible.sum()),
        "eligible_single_native_window_rows": int(len(selected)),
        "eligible_window_row_loss": 0,
        "eligible_native_unit_rows": int(len(native)),
        "windows_per_native_unit_min": int(
            native["windows_per_native_unit"].min()
        ),
        "windows_per_native_unit_max": int(
            native["windows_per_native_unit"].max()
        ),
        "feature_window_ids_bound": feature_window_ids is not None,
        "metadata_conflicts": {},
        "errors": [],
        "valid": True,
    }
    return native, audit


def _validate_grouped_probe_roles(
    grouped_roles: pd.DataFrame,
    native: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate grouped train/validation/test authority for every native unit."""

    required = [
        "temporal_unit_key",
        "recording_group_id",
        "behavior_label",
        "native_unit_valid_for_main_eval",
        "source_type",
        "video_key",
        "outer_fold_id",
        "role",
    ]
    missing = [column for column in required if column not in grouped_roles]
    if missing:
        raise ValueError(f"grouped source-probe roles missing columns: {missing}")
    roles = grouped_roles.loc[:, required].copy()
    for column in required:
        if column == "native_unit_valid_for_main_eval":
            continue
        roles[column] = roles[column].fillna("").astype(str).str.strip()
    blank_counts = {
        column: int(roles[column].eq("").sum())
        for column in required
        if column != "native_unit_valid_for_main_eval"
    }
    blank_counts = {column: count for column, count in blank_counts.items() if count}
    if blank_counts:
        raise ValueError(f"grouped source-probe roles contain blanks: {blank_counts}")
    unknown_roles = sorted(set(roles["role"]).difference({"train", "validation", "test"}))
    if unknown_roles:
        raise ValueError(f"grouped source-probe roles are invalid: {unknown_roles}")
    duplicate = int(roles.duplicated(["outer_fold_id", "temporal_unit_key"]).sum())
    if duplicate:
        raise ValueError(f"duplicate grouped role assignments={duplicate}")

    native_keys = set(native["temporal_unit_key"].astype(str))
    selected = roles.loc[roles["temporal_unit_key"].isin(native_keys)].copy()
    fold_ids = sorted(roles["outer_fold_id"].unique())
    if len(fold_ids) < 2:
        raise ValueError(f"grouped source probe requires at least two folds={fold_ids}")
    expected_keys = native_keys
    for fold_id in fold_ids:
        fold_keys = set(
            selected.loc[selected["outer_fold_id"].eq(fold_id), "temporal_unit_key"]
        )
        if fold_keys != expected_keys:
            raise ValueError(
                "grouped role native-unit coverage mismatch: "
                f"fold={fold_id}, missing={len(expected_keys - fold_keys)}, "
                f"extra={len(fold_keys - expected_keys)}"
            )
    test_counts = (
        selected.loc[selected["role"].eq("test")]
        .groupby("temporal_unit_key")
        .size()
        .reindex(sorted(native_keys), fill_value=0)
    )
    invalid_test_counts = int(test_counts.ne(1).sum())
    if invalid_test_counts:
        raise ValueError(
            "each eligible native unit must be outer-test exactly once: "
            f"invalid_units={invalid_test_counts}"
        )

    authority = native[
        [
            "temporal_unit_key",
            "source_type",
            "behavior_window_label",
            "split_group_key",
            "video_key",
        ]
    ].rename(
        columns={
            "source_type": "metadata_source_type",
            "behavior_window_label": "metadata_behavior_label",
            "split_group_key": "metadata_recording_group_id",
            "video_key": "metadata_video_key",
        }
    )
    checked = selected.merge(
        authority,
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
    )
    comparisons = {
        "source_type": "metadata_source_type",
        "behavior_label": "metadata_behavior_label",
        "recording_group_id": "metadata_recording_group_id",
        "video_key": "metadata_video_key",
    }
    conflicts = {
        role_column: int(checked[role_column].ne(checked[metadata_column]).sum())
        for role_column, metadata_column in comparisons.items()
    }
    conflicts = {column: count for column, count in conflicts.items() if count}
    if conflicts:
        raise ValueError(f"grouped-role/native metadata conflicts: {conflicts}")
    valid_eval = _strict_bool_series(
        checked["native_unit_valid_for_main_eval"],
        "native_unit_valid_for_main_eval",
    )
    if not valid_eval.all():
        raise ValueError(
            "eligible training windows map to native units excluded from main eval: "
            f"rows={int((~valid_eval).sum())}"
        )

    overlap_counts: dict[str, dict[str, int]] = {}
    for fold_id in fold_ids:
        fold = selected.loc[selected["outer_fold_id"].eq(fold_id)]
        for entity in ["recording_group_id", "video_key"]:
            values = {
                role: set(fold.loc[fold["role"].eq(role), entity])
                for role in ["train", "validation", "test"]
            }
            overlaps = {
                "train_validation": len(values["train"] & values["validation"]),
                "train_test": len(values["train"] & values["test"]),
                "validation_test": len(values["validation"] & values["test"]),
            }
            overlap_counts[f"{fold_id}:{entity}"] = overlaps
            if any(overlaps.values()):
                raise ValueError(
                    "grouped source-probe entity role overlap: "
                    f"fold={fold_id}, entity={entity}, overlaps={overlaps}"
                )
    audit = {
        "schema_version": "classification_v2_native_probe_roles_v1",
        "fold_count": int(len(fold_ids)),
        "eligible_native_unit_rows": int(len(native)),
        "selected_role_rows": int(len(selected)),
        "duplicate_fold_native_roles": 0,
        "invalid_test_assignment_units": 0,
        "role_metadata_conflicts": {},
        "entity_role_overlap_counts": overlap_counts,
        "validation_and_test_excluded_from_fit": True,
        "errors": [],
        "valid": True,
    }
    return selected, audit


def _run_grouped_native_probe(
    native: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    prediction_column: str,
    max_iter: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit fold-local probes and emit one outer-test row per native unit."""

    if max_iter <= 0:
        raise ValueError("grouped native probe max_iter must be positive")
    indexed = native.set_index("temporal_unit_key")
    if not indexed.index.is_unique:
        raise ValueError("grouped native probe received duplicate temporal units")
    predictions: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold_id in sorted(roles["outer_fold_id"].unique()):
        fold = roles.loc[roles["outer_fold_id"].eq(fold_id)]
        train_keys = fold.loc[fold["role"].eq("train"), "temporal_unit_key"].tolist()
        test_keys = fold.loc[fold["role"].eq("test"), "temporal_unit_key"].tolist()
        if not train_keys or not test_keys:
            raise ValueError(f"empty grouped native probe split={fold_id}")
        train = indexed.loc[train_keys]
        test = indexed.loc[test_keys]
        y_train = train[target_column].astype(str)
        y_test = test[target_column].astype(str)
        if y_train.nunique() < 2:
            raise ValueError(
                f"grouped native probe training fold has one class={fold_id}"
            )
        unseen = sorted(set(y_test).difference(y_train))
        if unseen:
            raise ValueError(
                f"grouped native probe test classes absent from train={fold_id}:{unseen}"
            )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                random_state=0,
            ),
        )
        model.fit(train[feature_columns], y_train)
        predicted = model.predict(test[feature_columns])
        part = test[
            [
                "source_type",
                "behavior_window_label",
                "split_group_key",
                "video_key",
            ]
        ].reset_index()
        part["outer_fold_id"] = fold_id
        part[prediction_column] = predicted
        predictions.append(part)
        fold_audits.append(
            {
                "outer_fold_id": fold_id,
                "train_native_units": int(len(train)),
                "validation_native_units": int(fold["role"].eq("validation").sum()),
                "test_native_units": int(len(test)),
                "train_target_counts": y_train.value_counts().sort_index().to_dict(),
                "test_target_counts": y_test.value_counts().sort_index().to_dict(),
                "balanced_accuracy_supported": _supported_balanced_accuracy(
                    y_test.reset_index(drop=True),
                    predicted,
                ),
                "scaler_fit_on_train_only": True,
                "validation_and_test_excluded_from_fit": True,
            }
        )
    out = pd.concat(predictions, ignore_index=True)
    duplicate = int(out["temporal_unit_key"].duplicated().sum())
    missing = set(native["temporal_unit_key"]).difference(out["temporal_unit_key"])
    extra = set(out["temporal_unit_key"]).difference(native["temporal_unit_key"])
    if duplicate or missing or extra:
        raise ValueError(
            "grouped native probe OOF coverage mismatch: "
            f"duplicates={duplicate}, missing={len(missing)}, extra={len(extra)}"
        )
    return (
        out.sort_values(["outer_fold_id", "temporal_unit_key"], kind="mergesort")
        .reset_index(drop=True),
        fold_audits,
    )


def audit_domain_feature_shift(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    feature_whitelist: list[str] | tuple[str, ...],
    expected_ordered_window_id_sha256: str,
    forbidden_patterns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Quantify feature missingness and standardized mean shift by source and behavior."""

    columns = _validate_exact_model_feature_table(
        features,
        feature_whitelist,
        forbidden_patterns=forbidden_patterns,
    )
    required = [
        "window_id",
        "source_type",
        "behavior_window_label",
        "window_valid_for_main_train",
    ]
    missing = [column for column in required if column not in metadata]
    if missing or len(features) != len(metadata):
        raise ValueError(
            "domain shift input mismatch: "
            f"missing={missing}, features={len(features)}, metadata={len(metadata)}"
        )
    alignment = require_ordered_window_ids(
        "window_metadata",
        metadata["window_id"],
    )
    observed_hash = alignment["reference_ordered_window_id_sha256"]
    if not _is_sha256(expected_ordered_window_id_sha256):
        raise ValueError("expected ordered-window SHA256 is malformed")
    if observed_hash != expected_ordered_window_id_sha256:
        raise ValueError(
            "domain shift ordered-window lineage mismatch: "
            f"expected={expected_ordered_window_id_sha256}, observed={observed_hash}"
        )
    valid = _strict_bool_series(
        metadata["window_valid_for_main_train"],
        "window_valid_for_main_train",
    )
    x = features.loc[valid, columns].copy()
    meta = metadata.loc[valid].reset_index(drop=True)
    x = x.reset_index(drop=True)
    sources = sorted(meta["source_type"].astype(str).unique())
    if len(sources) != 2:
        raise ValueError(f"domain shift audit requires exactly two sources, observed={sources}")
    source_stats: dict[str, Any] = {}
    for source in sources:
        subset = x.loc[meta["source_type"].astype(str).eq(source)]
        source_stats[source] = {
            "rows": int(len(subset)),
            "missing_rate": subset.isna().mean().astype(float).to_dict(),
            "mean": subset.mean().astype(float).to_dict(),
            "std": subset.std(ddof=0).astype(float).to_dict(),
        }
    left, right = sources
    left_mask = meta["source_type"].astype(str).eq(left)
    right_mask = meta["source_type"].astype(str).eq(right)
    mean_delta = x.loc[left_mask].mean() - x.loc[right_mask].mean()
    pooled_scale = np.sqrt(
        (x.loc[left_mask].var(ddof=0) + x.loc[right_mask].var(ddof=0)) / 2.0
    ).replace(0.0, np.nan)
    smd = (mean_delta / pooled_scale).replace([np.inf, -np.inf], np.nan)
    ranked = smd.abs().sort_values(ascending=False, na_position="last")
    return {
        "schema_version": "classification_v2_domain_feature_shift_v2",
        "eligible_rows": int(valid.sum()),
        "feature_count": int(len(columns)),
        "feature_whitelist": columns,
        "feature_whitelist_sha256": _ordered_names_sha256(columns),
        "forbidden_pattern_contract": (
            list(forbidden_patterns)
            if forbidden_patterns is not None
            else "model_io_default"
        ),
        "ordered_window_id_sha256_expected": expected_ordered_window_id_sha256,
        "ordered_window_id_sha256_observed": observed_hash,
        "ordered_window_lineage_match": True,
        "sources": sources,
        "source_statistics": source_stats,
        "standardized_mean_difference": smd.astype(object).where(smd.notna(), None).to_dict(),
        "top_absolute_smd_features": [
            {"feature": feature, "absolute_smd": float(ranked[feature])}
            for feature in ranked.index[:20]
            if pd.notna(ranked[feature])
        ],
        "source_behavior_counts": {
            behavior: {source: int(count) for source, count in values.items()}
            for behavior, values in pd.crosstab(
                meta["source_type"], meta["behavior_window_label"]
            ).to_dict().items()
        },
        "source_identifier_in_features": False,
        "camera_layout_metadata_status": "not_available_in_current_window_metadata",
        "camera_safe_claim_allowed": False,
        "claim_boundary": "recording-date/video-safe internal validation only",
        "errors": [],
        "valid": True,
    }


def _validated_ordered_names(
    values: list[str] | tuple[str, ...],
    name: str,
) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty ordered list")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{name} contains blank fields")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate fields")
    return normalized


def _strict_binary_column(series: pd.Series, name: str) -> pd.Series:
    """Convert one declared mask to 0/1 without treating unknown text as false."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"availability mask has missing values: {name}")
        return series.astype(np.float64)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"availability mask has nonfinite values: {name}")
        invalid = ~values.isin([0, 1])
        if invalid.any():
            raise ValueError(
                f"availability mask is not binary: {name}, invalid={int(invalid.sum())}"
            )
        return values.astype(np.float64)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": 1.0,
        "1": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "t": 1.0,
        "false": 0.0,
        "0": 0.0,
        "no": 0.0,
        "n": 0.0,
        "f": 0.0,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"availability mask has unknown values: {name}={unknown}")
    return normalized.map(mapping).astype(np.float64)


def _strict_bool_series(series: pd.Series, name: str) -> pd.Series:
    values = _strict_binary_column(series, name)
    return values.eq(1.0)


def _ordered_names_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    normalized = str(value).strip().lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _contingency(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame["source_type"], frame["behavior_window_label"])
    return {
        label: {source: int(count) for source, count in values.items()}
        for label, values in table.to_dict().items()
    }


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _supported_balanced_accuracy(true: pd.Series, predicted: np.ndarray) -> float:
    recalls = []
    predicted_series = pd.Series(predicted, index=true.index)
    for label in sorted(true.astype(str).unique()):
        mask = true.astype(str).eq(label)
        recalls.append(float(predicted_series.loc[mask].astype(str).eq(label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0
