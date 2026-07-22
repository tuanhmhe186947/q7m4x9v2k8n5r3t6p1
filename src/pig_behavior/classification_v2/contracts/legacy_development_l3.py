"""Immutable input contracts for the legacy-only L3 development gate."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_SOURCE,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

LEGACY_L3_SCHEMA_VERSION = "classification_v2.legacy_development_l3.v1"
PREDICTIVE_SPATIAL_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
    "social_relation",
)
MASK_ONLY_SPATIAL_GROUPS = ("quality_mask",)
DERIVED_MASK_FEATURE_INPUTS = {
    "social_neighbor_available": ("nearest_pig_id", "nearest_track_id"),
}
REQUIRED_FORBIDDEN_PATTERNS = (
    "manual_*",
    "review_*",
    "*behavior*",
    "*label*",
    "*path*",
    "*_id",
    "*_key",
    "source_type",
    "*fold*",
    "*split*",
    "*train*",
    "*eval*",
    "*loadable*",
    "*available*",
    "*availability*",
    "*missing*",
    "*_valid",
    "*_mask",
)
FORBIDDEN_PROBE_COLUMNS = (
    "manual_review_decision",
    "review_status",
    "behavior_label",
    "behavior_temporal_final",
    "crop_path",
    "source_video_path",
    "image_context_id",
    "window_id",
    "temporal_unit_key",
    "recording_group_id",
    "oof_fold_id",
    "source_type",
    "video_key",
    "pig_id",
    "image_context_loadable",
    "partner_context_available",
    "social_missing_mask",
    "window_valid_for_main_train",
)
AVAILABILITY_COLUMNS = (
    "roi_feeder_available",
    "roi_drinker_available",
    "roi_toy_available",
    "social_missing_mask",
    "bbox_valid",
    "actor_bbox_valid",
    "geometry_feature_valid",
    "roi_feature_valid",
    "spatiotemporal_feature_valid",
    "use_for_main_eval",
)
IMAGE_AVAILABILITY_COLUMNS = (
    "image_context_loadable",
    "bbox_context_valid",
    "full_frame_context_available",
    "partner_context_available",
)


def audit_legacy_feature_contract(
    contract: dict[str, Any],
    *,
    available_frame_columns: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Validate exact predictive, mask-only, blacklist, and claim contracts."""

    errors: list[str] = []
    if contract.get("lineage_scope") != LEGACY_DEVELOPMENT_SCOPE:
        errors.append("feature_contract_scope_mismatch")
    if contract.get("human_review_complete") is not False:
        errors.append("feature_contract_must_remain_unreviewed")
    boundary = contract.get("claim_boundary", {})
    for field in (
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "external_generalization_claim_allowed",
        "canonical_full_oof_allowed",
    ):
        if boundary.get(field) is not False:
            errors.append(f"feature_contract_{field}_must_be_false")

    selection = contract.get("feature_selection", {})
    if selection.get("never_use_all_numeric_columns") is not True:
        errors.append("never_use_all_numeric_columns_must_be_true")
    if selection.get("fail_closed_on_unknown_columns") is not True:
        errors.append("fail_closed_on_unknown_columns_must_be_true")
    if selection.get("window_tabular_branch_enabled") is not False:
        errors.append("legacy_window_tabular_branch_must_start_disabled")
    predictive_groups = tuple(selection.get("predictive_spatial_groups", []))
    mask_groups = tuple(selection.get("mask_only_spatial_groups", []))
    if predictive_groups != PREDICTIVE_SPATIAL_GROUPS:
        errors.append("predictive_spatial_group_order_mismatch")
    if mask_groups != MASK_ONLY_SPATIAL_GROUPS:
        errors.append("mask_only_spatial_group_order_mismatch")

    predictive_features = _features_for_groups(predictive_groups, errors)
    mask_features = _features_for_groups(mask_groups, errors)
    overlap = sorted(set(predictive_features).intersection(mask_features))
    if overlap:
        errors.append(f"predictive_mask_feature_overlap={overlap}")
    available = set(available_frame_columns)
    derived_mask_features = {
        feature
        for feature, inputs in DERIVED_MASK_FEATURE_INPUTS.items()
        if set(inputs).issubset(available)
    }
    missing_features = sorted(
        set(predictive_features + mask_features).difference(
            available | derived_mask_features
        )
    )
    if missing_features:
        errors.append(f"frozen_features_missing_from_frames={missing_features}")

    patterns = [
        str(value)
        for value in contract.get("forbidden_predictive_x_patterns", [])
    ]
    missing_patterns = [
        pattern for pattern in REQUIRED_FORBIDDEN_PATTERNS if pattern not in patterns
    ]
    if missing_patterns:
        errors.append(f"missing_forbidden_patterns={missing_patterns}")
    unblocked_probes = [
        column for column in FORBIDDEN_PROBE_COLUMNS if not _forbidden(column, patterns)
    ]
    if unblocked_probes:
        errors.append(f"forbidden_probe_columns_not_blocked={unblocked_probes}")
    blocked_predictive = [
        feature for feature in predictive_features if _forbidden(feature, patterns)
    ]
    if blocked_predictive:
        errors.append(f"predictive_features_match_blacklist={blocked_predictive}")

    actor = contract.get("actor_image_sequence", {})
    if actor.get("image_size") != 160:
        errors.append("actor_image_size_must_be_160")
    if actor.get("resize_policy") != (
        "letterbox_preserve_aspect_rgb_pad_black_v1"
    ):
        errors.append("actor_resize_policy_mismatch")
    if actor.get("cache_only_required") is not True:
        errors.append("actor_cache_only_must_be_required")
    if actor.get("source_media_fallback_allowed") is not False:
        errors.append("actor_source_fallback_must_be_false")

    labels = contract.get("target_y", {}).get("labels", [])
    if list(labels) != list(VALID_BEHAVIORS):
        errors.append("legacy_target_label_order_mismatch")
    authorization = contract.get("authorization", {})
    if authorization.get("model_training_authorized") is not False:
        errors.append("feature_freeze_must_not_authorize_training")
    if authorization.get("accuracy_f1_comparison_authorized") is not False:
        errors.append("feature_freeze_must_not_authorize_metric_comparison")

    feature_payload = {
        "predictive_groups": {
            group: LEGACY_SPATIAL_FRAME_FEATURES[group]
            for group in PREDICTIVE_SPATIAL_GROUPS
        },
        "mask_only_groups": {
            group: LEGACY_SPATIAL_FRAME_FEATURES[group]
            for group in MASK_ONLY_SPATIAL_GROUPS
        },
    }
    return {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "window_tabular_branch_enabled": False,
        "predictive_spatial_groups": list(predictive_groups),
        "mask_only_spatial_groups": list(mask_groups),
        "predictive_feature_whitelist": predictive_features,
        "mask_only_feature_whitelist": mask_features,
        "derived_mask_feature_inputs": DERIVED_MASK_FEATURE_INPUTS,
        "forbidden_predictive_x_patterns": patterns,
        "unblocked_forbidden_probe_columns": unblocked_probes,
        "feature_contract_payload_sha256": payload_sha256(feature_payload),
        "predictive_whitelist_sha256": payload_sha256(predictive_features),
        "mask_only_whitelist_sha256": payload_sha256(mask_features),
        "blacklist_sha256": payload_sha256(patterns),
        "errors": errors,
        "valid": not errors,
    }


def _features_for_groups(
    groups: tuple[str, ...],
    errors: list[str],
) -> list[str]:
    features: list[str] = []
    for group in groups:
        if group not in LEGACY_SPATIAL_FRAME_FEATURES:
            errors.append(f"unknown_spatial_feature_group={group}")
            continue
        features.extend(LEGACY_SPATIAL_FRAME_FEATURES[group])
    return features


def _forbidden(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns)


def audit_legacy_shortcuts(
    *,
    native_units: pd.DataFrame,
    temporal_selection: pd.DataFrame,
    temporal_views: dict[str, pd.DataFrame],
    image_frames: pd.DataFrame,
    enhanced_frames: pd.DataFrame,
    feature_contract_audit: dict[str, Any],
) -> dict[str, Any]:
    """Audit structural source, length, padding, and availability shortcuts."""

    errors: list[str] = []
    warnings: list[str] = []
    _require_columns(
        native_units,
        {
            "temporal_unit_key",
            "source_type",
            "behavior_label",
            "lineage_scope",
            "human_review_complete",
        },
        "native_units",
    )
    _require_columns(
        temporal_selection,
        {
            "window_id",
            "temporal_unit_key",
            "window_length_frames",
            "lineage_scope",
            "human_review_complete",
        },
        "temporal_selection",
    )
    _claim_errors(native_units, "native_units", errors)
    _claim_errors(temporal_selection, "temporal_selection", errors)
    if native_units["temporal_unit_key"].duplicated().any():
        errors.append("duplicate_native_temporal_unit_key")

    source_values = sorted(native_units["source_type"].astype(str).unique())
    if source_values != [LEGACY_SOURCE]:
        errors.append(f"legacy_source_values={source_values}")
    joined = temporal_selection.merge(
        native_units[["temporal_unit_key", "behavior_label", "source_type"]],
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    missing_labels = int(joined["behavior_label"].isna().sum())
    if missing_labels:
        errors.append(f"shortcut_rows_without_label={missing_labels}")
    length_report = _signature_target_report(
        joined["window_length_frames"],
        joined["behavior_label"],
    )
    length_distribution = pd.crosstab(
        joined["window_length_frames"],
        joined["behavior_label"],
        normalize="index",
    ).reindex(columns=list(VALID_BEHAVIORS), fill_value=0.0)
    length_distribution_delta = (
        float(
            (length_distribution.max(axis=0) - length_distribution.min(axis=0))
            .max()
        )
        if not length_distribution.empty
        else 0.0
    )
    if length_distribution_delta > 1e-12:
        errors.append(
            "temporal_length_label_distribution_drift="
            f"{length_distribution_delta}"
        )

    view_reports: dict[str, Any] = {}
    missing_views = sorted(
        set(LEGACY_TEMPORAL_MODEL_VIEW_SPECS).difference(temporal_views)
    )
    if missing_views:
        errors.append(f"missing_temporal_views={missing_views}")
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        slots = temporal_views.get(view_name)
        if slots is None:
            continue
        report = _audit_temporal_view(
            view_name,
            int(spec["sequence_length"]),
            slots,
        )
        view_reports[view_name] = report
        errors.extend(report["errors"])

    image_report: dict[str, Any] = {}
    _require_columns(
        image_frames,
        {
            "source_type",
            "lineage_scope",
            "human_review_complete",
            *IMAGE_AVAILABILITY_COLUMNS,
        },
        "image_frames",
    )
    _claim_errors(image_frames, "image_frames", errors)
    for column in IMAGE_AVAILABILITY_COLUMNS:
        values, invalid = _strict_bool(image_frames[column])
        image_report[column] = {
            "true_rows": int(values.sum()),
            "false_rows": int((~values).sum()),
            "invalid_rows": invalid,
        }
        if invalid:
            errors.append(f"invalid_image_availability_values={column}:{invalid}")
    for required_true in ("image_context_loadable", "bbox_context_valid"):
        if image_report[required_true]["false_rows"]:
            errors.append(
                f"required_image_context_false={required_true}:"
                f"{image_report[required_true]['false_rows']}"
            )

    _require_columns(
        enhanced_frames,
        {
            "behavior",
            "source_type",
            "lineage_scope",
            "human_review_complete",
            *AVAILABILITY_COLUMNS,
        },
        "enhanced_frames",
    )
    _claim_errors(enhanced_frames, "enhanced_frames", errors)
    availability_reports: dict[str, Any] = {}
    variable_availability_columns: list[str] = []
    near_direct_columns: list[str] = []
    for column in AVAILABILITY_COLUMNS:
        values = enhanced_frames[column]
        unique_values = int(values.nunique(dropna=False))
        if unique_values > 1:
            variable_availability_columns.append(column)
        mapping = _signature_target_report(values, enhanced_frames["behavior"])
        if mapping["near_direct_target_signature"]:
            near_direct_columns.append(column)
        availability_reports[column] = {
            "unique_values": unique_values,
            "missing_rows": int(values.isna().sum()),
            "target_mapping": mapping,
        }
    if near_direct_columns:
        warnings.append(
            "availability_near_direct_target_signatures_are_control_only="
            f"{near_direct_columns}"
        )
    if feature_contract_audit.get("valid") is not True:
        errors.append("feature_contract_audit_invalid_for_shortcut_gate")
    patterns = feature_contract_audit.get(
        "forbidden_predictive_x_patterns",
        [],
    )
    unprotected_availability = [
        column
        for column in (*AVAILABILITY_COLUMNS, *IMAGE_AVAILABILITY_COLUMNS)
        if not _forbidden(column, list(patterns))
    ]
    if unprotected_availability:
        errors.append(
            "availability_columns_not_blacklisted="
            f"{unprotected_availability}"
        )

    return {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "method": "deterministic_structural_audit_without_model_fit",
        "source_probe": {
            "source_values": source_values,
            "source_count": len(source_values),
            "learned_probe_applicable": False,
            "status": "NOT_APPLICABLE_SINGLE_SOURCE_STRUCTURAL_ONLY",
            "model_fit_performed": False,
        },
        "length_shortcut": {
            **length_report,
            "maximum_class_distribution_delta_across_lengths": (
                length_distribution_delta
            ),
            "class_distribution_by_length": {
                str(index): {
                    str(label): float(value)
                    for label, value in row.items()
                }
                for index, row in length_distribution.to_dict(
                    orient="index"
                ).items()
            },
        },
        "temporal_view_reports": view_reports,
        "image_availability": image_report,
        "availability_and_missingness": availability_reports,
        "variable_availability_columns": variable_availability_columns,
        "near_direct_availability_columns": near_direct_columns,
        "availability_is_predictive_x": False,
        "availability_only_controls_required_for_modality_promotion": True,
        "unprotected_availability_columns": unprotected_availability,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _audit_temporal_view(
    view_name: str,
    sequence_length: int,
    slots: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "temporal_view_name",
        "view_item_id",
        "source_type",
        "slot_index",
        "declared_sequence_length",
        "length_mask",
        "observed_mask",
        "timing_valid_mask",
        "padding_mask",
        "lineage_scope",
        "human_review_complete",
    }
    _require_columns(slots, required, view_name)
    _claim_errors(slots, view_name, errors)
    names = sorted(slots["temporal_view_name"].astype(str).unique())
    if names != [view_name]:
        errors.append(f"{view_name}:view_name_values={names}")
    sources = sorted(slots["source_type"].astype(str).unique())
    if sources != [LEGACY_SOURCE]:
        errors.append(f"{view_name}:source_values={sources}")
    declared = pd.to_numeric(
        slots["declared_sequence_length"],
        errors="coerce",
    )
    declared_mismatch = int(declared.ne(sequence_length).sum())
    if declared_mismatch:
        errors.append(
            f"{view_name}:declared_length_mismatches={declared_mismatch}"
        )
    duplicate_slots = int(
        slots.duplicated(["view_item_id", "slot_index"]).sum()
    )
    if duplicate_slots:
        errors.append(f"{view_name}:duplicate_item_slots={duplicate_slots}")
    group_sizes = slots.groupby("view_item_id", sort=False).size()
    wrong_group_sizes = int(group_sizes.ne(sequence_length).sum())
    if wrong_group_sizes:
        errors.append(f"{view_name}:wrong_item_slot_counts={wrong_group_sizes}")
    slot_index = pd.to_numeric(slots["slot_index"], errors="coerce")
    invalid_slot_indices = int(
        (
            slot_index.isna()
            | slot_index.lt(0)
            | slot_index.ge(sequence_length)
            | slot_index.mod(1).ne(0)
        ).sum()
    )
    if invalid_slot_indices:
        errors.append(
            f"{view_name}:invalid_slot_indices={invalid_slot_indices}"
        )
    mask_counts: dict[str, Any] = {}
    expected_true = {
        "length_mask": True,
        "observed_mask": True,
        "timing_valid_mask": True,
        "padding_mask": False,
    }
    for column, expected in expected_true.items():
        values, invalid = _strict_bool(slots[column])
        mismatches = int(values.ne(expected).sum())
        mask_counts[column] = {
            "true_rows": int(values.sum()),
            "false_rows": int((~values).sum()),
            "invalid_rows": invalid,
            "expected": expected,
            "mismatch_rows": mismatches,
        }
        if invalid or mismatches:
            errors.append(
                f"{view_name}:{column}_contract="
                f"invalid:{invalid},mismatch:{mismatches}"
            )
    return {
        "sequence_length": sequence_length,
        "item_rows": int(len(group_sizes)),
        "slot_rows": int(len(slots)),
        "duplicate_item_slots": duplicate_slots,
        "wrong_item_slot_counts": wrong_group_sizes,
        "invalid_slot_indices": invalid_slot_indices,
        "mask_counts": mask_counts,
        "errors": errors,
        "valid": not errors,
    }


def build_legacy_artifact_manifest(
    artifacts: dict[str, tuple[str, Path]],
) -> pd.DataFrame:
    """Hash every frozen L3 input without creating a self-referential digest."""

    rows: list[dict[str, Any]] = []
    for artifact_name, (artifact_kind, path) in sorted(artifacts.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen artifact={artifact_name}:{path}")
        row_count: int | None = None
        tensor_shape = ""
        tensor_dtype = ""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            row_count = _csv_row_count(path)
        elif suffix == ".npy":
            tensor = np.load(path, mmap_mode="r")
            tensor_shape = "x".join(str(int(value)) for value in tensor.shape)
            tensor_dtype = str(tensor.dtype)
            row_count = int(tensor.shape[0]) if tensor.ndim else None
        rows.append(
            {
                "artifact_name": artifact_name,
                "artifact_kind": artifact_kind,
                "path": path.as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "row_count": row_count,
                "tensor_shape": tensor_shape,
                "tensor_dtype": tensor_dtype,
                "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                "human_review_complete": False,
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest["artifact_name"].duplicated().any():
        raise ValueError("duplicate frozen artifact names")
    if manifest["path"].duplicated().any():
        raise ValueError("duplicate frozen artifact paths")
    return manifest


def verify_legacy_artifact_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    """Recompute frozen file hashes, sizes, row counts, and tensor metadata."""

    required = {
        "artifact_name",
        "artifact_kind",
        "path",
        "sha256",
        "size_bytes",
        "row_count",
        "tensor_shape",
        "tensor_dtype",
        "lineage_scope",
        "human_review_complete",
    }
    _require_columns(manifest, required, "artifact_manifest")
    errors: list[str] = []
    _claim_errors(manifest, "artifact_manifest", errors)
    duplicate_names = int(manifest["artifact_name"].duplicated().sum())
    duplicate_paths = int(manifest["path"].duplicated().sum())
    if duplicate_names:
        errors.append(f"duplicate_artifact_names={duplicate_names}")
    if duplicate_paths:
        errors.append(f"duplicate_artifact_paths={duplicate_paths}")
    rows: dict[str, Any] = {}
    for row in manifest.itertuples(index=False):
        name = str(row.artifact_name)
        path = Path(str(row.path))
        exists = path.is_file()
        observed_sha = file_sha256(path) if exists else ""
        observed_size = int(path.stat().st_size) if exists else -1
        sha_match = observed_sha == str(row.sha256)
        size_match = observed_size == int(row.size_bytes)
        row_count_match = True
        tensor_shape_match = True
        tensor_dtype_match = True
        if exists and path.suffix.lower() == ".csv":
            expected_rows = _optional_int(row.row_count)
            row_count_match = expected_rows == _csv_row_count(path)
        if exists and path.suffix.lower() == ".npy":
            tensor = np.load(path, mmap_mode="r")
            observed_shape = "x".join(str(int(value)) for value in tensor.shape)
            tensor_shape_match = observed_shape == str(row.tensor_shape)
            tensor_dtype_match = str(tensor.dtype) == str(row.tensor_dtype)
            expected_rows = _optional_int(row.row_count)
            row_count_match = expected_rows == int(tensor.shape[0])
        if not exists:
            errors.append(f"frozen_artifact_missing={name}")
        if exists and not sha_match:
            errors.append(f"frozen_artifact_sha_mismatch={name}")
        if exists and not size_match:
            errors.append(f"frozen_artifact_size_mismatch={name}")
        if exists and not row_count_match:
            errors.append(f"frozen_artifact_row_count_mismatch={name}")
        if exists and not tensor_shape_match:
            errors.append(f"frozen_artifact_tensor_shape_mismatch={name}")
        if exists and not tensor_dtype_match:
            errors.append(f"frozen_artifact_tensor_dtype_mismatch={name}")
        rows[name] = {
            "path": str(path),
            "exists": exists,
            "sha256_match": sha_match,
            "size_match": size_match,
            "row_count_match": row_count_match,
            "tensor_shape_match": tensor_shape_match,
            "tensor_dtype_match": tensor_dtype_match,
        }
    return {
        "artifact_rows": int(len(manifest)),
        "verified_artifacts": rows,
        "errors": errors,
        "valid": not errors,
    }


def verify_legacy_snapshot(
    snapshot: dict[str, Any],
    *,
    artifact_manifest_path: Path,
    feature_contract_path: Path,
    feature_audit_path: Path,
    shortcut_audit_path: Path,
) -> dict[str, Any]:
    """Verify the non-cyclic hashes and claim boundary of one L3 snapshot."""

    errors: list[str] = []
    if snapshot.get("schema_version") != LEGACY_L3_SCHEMA_VERSION:
        errors.append("snapshot_schema_version_mismatch")
    if snapshot.get("lineage_scope") != LEGACY_DEVELOPMENT_SCOPE:
        errors.append("snapshot_lineage_scope_mismatch")
    if snapshot.get("human_review_complete") is not False:
        errors.append("snapshot_must_remain_unreviewed")
    for field in (
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "model_training_authorized",
        "accuracy_f1_comparison_authorized",
    ):
        if snapshot.get(field) is not False:
            errors.append(f"snapshot_{field}_must_be_false")
    expected_hashes = {
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "feature_contract_sha256": file_sha256(feature_contract_path),
        "feature_audit_sha256": file_sha256(feature_audit_path),
        "shortcut_audit_sha256": file_sha256(shortcut_audit_path),
    }
    hash_matches = {
        field: snapshot.get(field) == expected
        for field, expected in expected_hashes.items()
    }
    for field, matches in hash_matches.items():
        if not matches:
            errors.append(f"snapshot_hash_mismatch={field}")
    frozen = snapshot.get("frozen_contract", {})
    expected_views = list(LEGACY_TEMPORAL_MODEL_VIEW_SPECS)
    if frozen.get("temporal_views") != expected_views:
        errors.append("snapshot_temporal_view_order_mismatch")
    if frozen.get("image_size") != 160:
        errors.append("snapshot_image_size_mismatch")
    if frozen.get("predictive_spatial_groups") != list(
        PREDICTIVE_SPATIAL_GROUPS
    ):
        errors.append("snapshot_predictive_groups_mismatch")
    if frozen.get("mask_only_spatial_groups") != list(
        MASK_ONLY_SPATIAL_GROUPS
    ):
        errors.append("snapshot_mask_groups_mismatch")
    return {
        "expected_hashes": expected_hashes,
        "hash_matches": hash_matches,
        "errors": errors,
        "valid": not errors,
    }


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


def _claim_errors(
    frame: pd.DataFrame,
    name: str,
    errors: list[str],
) -> None:
    scopes = sorted(frame["lineage_scope"].fillna("").astype(str).unique())
    reviewed, invalid = _strict_bool(frame["human_review_complete"])
    if scopes != [LEGACY_DEVELOPMENT_SCOPE]:
        errors.append(f"{name}:lineage_scope_values={scopes}")
    if invalid or reviewed.any():
        errors.append(
            f"{name}:human_review_complete="
            f"invalid:{invalid},true:{int(reviewed.sum())}"
        )


def _strict_bool(series: pd.Series) -> tuple[pd.Series, int]:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    valid = normalized.isin(truthy | falsy)
    return normalized.isin(truthy), int((~valid).sum())


def _signature_target_report(
    signatures: pd.Series,
    targets: pd.Series,
    *,
    direct_accuracy_threshold: float = 0.95,
    minimum_uplift: float = 0.10,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "signature": signatures.fillna("<NA>").astype(str),
            "target": targets.fillna("").astype(str),
        }
    )
    frame = frame.loc[frame["target"].ne("")]
    if frame.empty:
        return {
            "rows": 0,
            "target_class_count": 0,
            "unique_signatures": 0,
            "majority_target_baseline": 0.0,
            "signature_mapping_accuracy": 0.0,
            "uplift_over_majority": 0.0,
            "near_direct_target_signature": False,
        }
    table = pd.crosstab(frame["signature"], frame["target"])
    rows = int(table.to_numpy().sum())
    baseline = float(frame["target"].value_counts().max() / rows)
    accuracy = float(table.max(axis=1).sum() / rows)
    uplift = accuracy - baseline
    return {
        "rows": rows,
        "target_class_count": int(frame["target"].nunique()),
        "unique_signatures": int(len(table)),
        "majority_target_baseline": baseline,
        "signature_mapping_accuracy": accuracy,
        "uplift_over_majority": uplift,
        "near_direct_target_signature": bool(
            frame["target"].nunique() >= 2
            and accuracy >= direct_accuracy_threshold
            and uplift >= minimum_uplift
        ),
    }


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
