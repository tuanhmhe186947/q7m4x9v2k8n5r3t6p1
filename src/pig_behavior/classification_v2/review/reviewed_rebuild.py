"""Fail-closed contracts for rebuilding after frozen Behavior review."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.lineage_config import (
    _build_source_report,
    _tree_snapshot,
    resolve_source_path,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    PostReviewContractError,
    sha256_file,
    validate_review_close_authority,
)

APPLICATION_AUTHORITY_VERSION = (
    "classification_v2.reviewed_training_application_authority.v1"
)
AUTOCARRY_VERSION = "classification_v2.final_review_autocarry.v1"
OVERLAY_AUDIT_VERSION = "classification_v2.reviewed_label_overlay_audit.v1"
DERIVED_CONFIG_VERSION = "classification_v2.reviewed_lineage_config_derivation.v1"


class ReviewedRebuildContractError(PostReviewContractError):
    """Raised when reviewed rebuild lineage cannot be proven."""


def freeze_reviewed_training_application_authority(
    *,
    review_close_authority: Mapping[str, Any],
    primary_scope: pd.DataFrame,
    primary_decisions: pd.DataFrame,
    primary_quality: pd.DataFrame,
    control_scope: pd.DataFrame,
    control_decisions: pd.DataFrame,
    control_quality: pd.DataFrame,
    composite_scope: pd.DataFrame,
    composite_decisions: pd.DataFrame,
    composite_quality: pd.DataFrame,
    corrected_source_authority: Mapping[str, Any],
    fixed_point_audit: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Bind the exact 3,243-unit overlay without changing review membership."""
    validate_review_close_authority(review_close_authority)
    _validate_review_close_bindings(review_close_authority, artifact_bindings)
    expected_primary = int(
        review_close_authority["primary_review"]["scope_rows"]
    )
    expected_control = int(
        review_close_authority["control_review"]["scope_rows"]
    )
    if len(primary_scope) != expected_primary or len(primary_quality) != expected_primary:
        raise ReviewedRebuildContractError("application_primary_count_drift")
    if len(control_scope) != expected_control or len(control_quality) != expected_control:
        raise ReviewedRebuildContractError("application_control_count_drift")
    expected_total = expected_primary + expected_control
    for name, frame in (
        ("composite_scope", composite_scope),
        ("composite_decisions", composite_decisions),
        ("composite_quality", composite_quality),
    ):
        if len(frame) != expected_total:
            raise ReviewedRebuildContractError(
                f"application_{name}_count={len(frame)}:{expected_total}"
            )
        _require_unique_key(frame, "review_unit_id", name)

    _assert_semantic_union(
        composite_scope,
        primary_scope,
        control_scope,
        columns=(
            "review_unit_id",
            "temporal_unit_key",
            "source_type",
            "video_key",
            "unit_start_frame",
            "unit_end_frame",
            "behavior_label",
        ),
        name="scope",
    )
    _assert_semantic_union(
        composite_decisions,
        primary_decisions,
        control_decisions,
        columns=(
            "review_unit_id",
            "manual_review_decision",
            "manual_corrected_behavior",
            "manual_label_strength",
        ),
        name="decisions",
    )
    _assert_semantic_union(
        composite_quality,
        primary_quality,
        control_quality,
        columns=(
            "review_unit_id",
            "original_behavior",
            "reviewed_behavior",
            "label_status",
            "source_label_error_confirmed",
            "error_pattern",
        ),
        name="quality",
    )
    decision_counts = (
        composite_decisions["manual_review_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )
    if set(decision_counts) - {"accept", "corrected", "exclude"}:
        raise ReviewedRebuildContractError("application_unresolved_decision")
    if int(decision_counts.get("exclude", 0)) != 0:
        raise ReviewedRebuildContractError("application_exclusions_not_supported")
    source_errors = composite_quality["source_label_error_confirmed"].eq("YES")
    changed = composite_quality["original_behavior"].astype(str).ne(
        composite_quality["reviewed_behavior"].astype(str)
    )
    if not source_errors.equals(changed):
        raise ReviewedRebuildContractError("application_quality_change_mismatch")
    if int(fixed_point_audit.get("high_target_rows", -1)) != 0:
        raise ReviewedRebuildContractError("application_fixed_point_high_open")
    if int(fixed_point_audit.get("automatic_label_changes", -1)) != 0:
        raise ReviewedRebuildContractError("application_automatic_label_change")
    if corrected_source_authority.get("status") != "FROZEN":
        raise ReviewedRebuildContractError("application_corrected_source_not_frozen")

    return {
        "schema_version": APPLICATION_AUTHORITY_VERSION,
        "status": "FROZEN",
        "human_reviewed_units": expected_total,
        "primary_units": expected_primary,
        "control_units": expected_control,
        "decision_counts": {str(key): int(value) for key, value in decision_counts.items()},
        "source_label_corrections": int(source_errors.sum()),
        "technical_exclusions": 0,
        "fixed_point_high_targets": 0,
        "application_policy": "FROZEN_REVIEW_OVERLAY_ONLY",
        "unreviewed_policy": "PRESERVE_ORIGINAL_LABEL_EXACTLY",
        "review_membership_recomputed": False,
        "candidate_selection_recomputed": False,
        "automatic_label_changes": 0,
        "model_predictions_used": False,
        "artifacts": {name: dict(value) for name, value in artifact_bindings.items()},
        "active_behavior_ledger_touched": "NO",
    }


def build_final_review_autocarry(
    frame_features: pd.DataFrame,
    reviewed_scope: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an explicit all-unreviewed original-label carry manifest."""
    frame_columns = [
        "temporal_unit_key",
        "source_type",
        "video_key",
        "behavior",
    ]
    _require_columns(frame_features, frame_columns, "frame_features")
    _require_columns(
        reviewed_scope,
        ["review_unit_id", "temporal_unit_key"],
        "reviewed_scope",
    )
    _require_unique_key(reviewed_scope, "review_unit_id", "reviewed_scope")
    _require_unique_key(reviewed_scope, "temporal_unit_key", "reviewed_scope")
    unit_counts = frame_features.groupby("temporal_unit_key").agg(
        source_type_count=("source_type", "nunique"),
        video_key_count=("video_key", "nunique"),
        behavior_count=("behavior", "nunique"),
        source_type=("source_type", "first"),
        video_key=("video_key", "first"),
        behavior_label=("behavior", "first"),
        unit_frame_count=("temporal_unit_key", "size"),
    )
    inconsistent = unit_counts[
        unit_counts[["source_type_count", "video_key_count", "behavior_count"]]
        .ne(1)
        .any(axis=1)
    ]
    if len(inconsistent):
        raise ReviewedRebuildContractError(
            f"autocarry_inconsistent_units={len(inconsistent)}"
        )
    universe = set(unit_counts.index.astype(str))
    reviewed = set(reviewed_scope["temporal_unit_key"].astype(str))
    missing_reviewed = reviewed - universe
    if missing_reviewed:
        raise ReviewedRebuildContractError(
            f"autocarry_reviewed_units_missing={len(missing_reviewed)}"
        )
    carry_keys = sorted(universe - reviewed)
    carry = unit_counts.loc[carry_keys].reset_index()
    carry["review_unit_id"] = carry["temporal_unit_key"].map(
        lambda value: "final_autocarry_" + hashlib.sha256(
            str(value).encode()
        ).hexdigest()[:24]
    )
    carry["include_in_training"] = True
    carry["sample_weight"] = 1.0
    carry["auto_carry_provenance"] = (
        "FROZEN_UNREVIEWED_ORIGINAL_LABEL_PRESERVED"
    )
    output_columns = [
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "video_key",
        "behavior_label",
        "unit_frame_count",
        "include_in_training",
        "sample_weight",
        "auto_carry_provenance",
    ]
    carry = carry[output_columns]
    audit = {
        "schema_version": AUTOCARRY_VERSION,
        "status": "PASS",
        "universe_units": len(universe),
        "reviewed_units": len(reviewed),
        "auto_carry_units": len(carry),
        "reviewed_auto_carry_overlap": 0,
        "missing_partition_units": 0,
        "extra_partition_units": 0,
        "labels_changed": 0,
        "selection_recomputed": False,
    }
    return carry, audit


def build_reviewed_application_views(
    *,
    frame_features: pd.DataFrame,
    composite_scope: pd.DataFrame,
    composite_decisions: pd.DataFrame,
    composite_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Adapt frozen final labels to corrected-source snapshots deterministically."""
    _require_columns(
        frame_features,
        [
            "temporal_unit_key",
            "source_type",
            "video_key",
            "frame_index",
            "behavior",
        ],
        "frame_features",
    )
    _require_columns(
        composite_scope,
        [
            "review_unit_id",
            "temporal_unit_key",
            "source_type",
            "video_key",
            "unit_start_frame",
            "unit_end_frame",
            "unit_frame_count",
            "behavior_label",
        ],
        "composite_scope",
    )
    _require_columns(
        composite_decisions,
        [
            "review_unit_id",
            "behavior_label",
            "manual_review_decision",
            "manual_corrected_behavior",
        ],
        "composite_decisions",
    )
    _require_columns(
        composite_quality,
        ["review_unit_id", "original_behavior", "reviewed_behavior"],
        "composite_quality",
    )
    for name, frame in (
        ("composite_scope", composite_scope),
        ("composite_decisions", composite_decisions),
        ("composite_quality", composite_quality),
    ):
        _require_unique_key(frame, "review_unit_id", name)

    frame = frame_features.copy()
    frame["frame_index_num"] = pd.to_numeric(frame["frame_index"], errors="coerce")
    if frame["frame_index_num"].isna().any():
        raise ReviewedRebuildContractError("application_frame_index_invalid")
    units = frame.groupby("temporal_unit_key").agg(
        fresh_source_type=("source_type", "first"),
        source_type_count=("source_type", "nunique"),
        fresh_video_key=("video_key", "first"),
        video_key_count=("video_key", "nunique"),
        fresh_source_behavior=("behavior", "first"),
        source_behavior_count=("behavior", "nunique"),
        fresh_unit_start=("frame_index_num", "min"),
        fresh_unit_end=("frame_index_num", "max"),
        fresh_unit_count=("frame_index_num", "size"),
        unique_frame_count=("frame_index_num", "nunique"),
    )
    bad_units = units[
        units[
            [
                "source_type_count",
                "video_key_count",
                "source_behavior_count",
            ]
        ]
        .ne(1)
        .any(axis=1)
        | units["fresh_unit_count"].ne(units["unique_frame_count"])
    ]
    if len(bad_units):
        raise ReviewedRebuildContractError(
            f"application_fresh_unit_inconsistent={len(bad_units)}"
        )

    scope = composite_scope.merge(
        units.reset_index(),
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if scope["fresh_source_behavior"].isna().any():
        missing_fresh = int(scope["fresh_source_behavior"].isna().sum())
        raise ReviewedRebuildContractError(
            f"application_scope_missing_fresh_units={missing_fresh}"
        )
    mismatches = []
    for left, right, name in (
        (scope["source_type"].astype(str), scope["fresh_source_type"].astype(str), "source"),
        (scope["video_key"].astype(str), scope["fresh_video_key"].astype(str), "video"),
        (
            pd.to_numeric(scope["unit_start_frame"], errors="coerce"),
            scope["fresh_unit_start"],
            "start",
        ),
        (
            pd.to_numeric(scope["unit_end_frame"], errors="coerce"),
            scope["fresh_unit_end"],
            "end",
        ),
        (
            pd.to_numeric(scope["unit_frame_count"], errors="coerce"),
            scope["fresh_unit_count"],
            "count",
        ),
    ):
        count = int(left.ne(right).sum())
        if count:
            mismatches.append(f"{name}={count}")
    if mismatches:
        raise ReviewedRebuildContractError(
            "application_fresh_scope_mismatch=" + ",".join(mismatches)
        )

    application_scope = composite_scope.copy()
    fresh_by_key = scope.set_index("review_unit_id")["fresh_source_behavior"]
    application_scope["behavior_label"] = application_scope["review_unit_id"].map(
        fresh_by_key
    )
    if "behavior" in application_scope:
        application_scope["behavior"] = application_scope["behavior_label"]

    quality = composite_quality.set_index("review_unit_id")
    decisions = composite_decisions.copy()
    decisions["behavior_label"] = decisions["review_unit_id"].map(fresh_by_key)
    reviewed_label = decisions["review_unit_id"].map(quality["reviewed_behavior"])
    corrected = decisions["behavior_label"].astype(str).ne(reviewed_label.astype(str))
    decisions["manual_review_decision"] = np.where(
        corrected,
        "corrected",
        "accept",
    )
    decisions["manual_corrected_behavior"] = np.where(
        corrected,
        reviewed_label,
        "",
    )
    decisions["review_reason"] = "FROZEN_FINAL_REVIEW_APPLICATION"
    source_snapshot_changed = composite_quality["original_behavior"].astype(str).ne(
        composite_quality["review_unit_id"].map(fresh_by_key).astype(str)
    )
    audit = {
        "schema_version": OVERLAY_AUDIT_VERSION,
        "status": "READY_TO_APPLY",
        "reviewed_units": int(len(application_scope)),
        "fresh_source_snapshot_changed_units": int(source_snapshot_changed.sum()),
        "application_accept_units": int((~corrected).sum()),
        "application_corrected_units": int(corrected.sum()),
        "final_reviewed_labels_changed": 0,
        "review_membership_changed": 0,
        "scope_geometry_mismatches": 0,
        "source_or_video_mismatches": 0,
        "adaptation_rule": (
            "FINAL_REVIEWED_LABEL_FIXED; DECISION_ACTION_REDERIVED_AGAINST_"
            "FRESH_CORRECTED_SOURCE"
        ),
    }
    return application_scope, decisions, audit


def audit_reviewed_label_overlay(
    *,
    before_frames: pd.DataFrame,
    after_frames: pd.DataFrame,
    composite_scope: pd.DataFrame,
    composite_quality: pd.DataFrame,
    apply_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove reviewed labels applied and all unreviewed labels stayed fixed."""
    if len(before_frames) != len(after_frames):
        raise ReviewedRebuildContractError("overlay_frame_count_changed")
    key_columns = ["temporal_unit_key", "frame_index"]
    _require_columns(before_frames, [*key_columns, "behavior"], "before_frames")
    _require_columns(
        after_frames,
        [
            *key_columns,
            "behavior",
            "behavior_after_review",
            "behavior_reviewed_final",
            "behavior_review_label_resolved",
            "behavior_review_auto_carried",
        ],
        "after_frames",
    )
    before = before_frames[[*key_columns, "behavior"]].rename(
        columns={"behavior": "source_behavior"}
    )
    after = after_frames.merge(
        before,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    if after["source_behavior"].isna().any():
        raise ReviewedRebuildContractError("overlay_frame_key_mismatch")
    quality = composite_quality.set_index("review_unit_id")
    reviewed_key_by_id = composite_scope.set_index("review_unit_id")[
        "temporal_unit_key"
    ]
    expected = reviewed_key_by_id.to_frame().join(
        quality[["reviewed_behavior"]],
        how="left",
    )
    expected_by_key = expected.set_index("temporal_unit_key")["reviewed_behavior"]
    expected_label = after["temporal_unit_key"].map(expected_by_key)
    reviewed_mask = expected_label.notna()
    wrong_reviewed = reviewed_mask & after["behavior_reviewed_final"].astype(str).ne(
        expected_label.astype(str)
    )
    if wrong_reviewed.any():
        raise ReviewedRebuildContractError(
            f"overlay_wrong_reviewed_frames={int(wrong_reviewed.sum())}"
        )
    carry_mask = ~reviewed_mask
    changed_carry = carry_mask & after["behavior_after_review"].astype(str).ne(
        after["source_behavior"].astype(str)
    )
    if changed_carry.any():
        raise ReviewedRebuildContractError(
            f"overlay_unreviewed_label_changed={int(changed_carry.sum())}"
        )
    unresolved = ~_bool_series(after["behavior_review_label_resolved"])
    if unresolved.any():
        raise ReviewedRebuildContractError(
            f"overlay_unresolved_frames={int(unresolved.sum())}"
        )
    if int(apply_audit.get("missing_review_unit_count", -1)) != 0:
        raise ReviewedRebuildContractError("overlay_apply_unmatched_units")
    return {
        "schema_version": OVERLAY_AUDIT_VERSION,
        "status": "PASS",
        "frame_rows_before": int(len(before_frames)),
        "frame_rows_after": int(len(after_frames)),
        "reviewed_units": int(len(expected_by_key)),
        "reviewed_frame_rows": int(reviewed_mask.sum()),
        "auto_carry_units": int(
            after.loc[carry_mask, "temporal_unit_key"].nunique()
        ),
        "auto_carry_frame_rows": int(carry_mask.sum()),
        "wrong_reviewed_frames": 0,
        "unreviewed_label_changes": 0,
        "unresolved_frames": 0,
        "active_behavior_ledger_touched": "NO",
        "candidate_selection_recomputed": False,
    }


def derive_reviewed_lineage_config(
    *,
    repository_root: Path,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    lineage_id: str,
    run_root: Path,
    scientific_accepted_sha: str,
    adjusted_roi_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a fresh config; full preflight must still rehash heavy trees."""
    if len(scientific_accepted_sha) != 40:
        raise ReviewedRebuildContractError("derived_config_code_sha_invalid")
    if not lineage_id.strip():
        raise ReviewedRebuildContractError("derived_config_lineage_id_blank")
    if run_root.exists() and any(run_root.iterdir()):
        raise ReviewedRebuildContractError("derived_config_run_root_not_empty")
    root = repository_root.resolve()
    roi = adjusted_roi_path.resolve()
    try:
        roi_relative = roi.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReviewedRebuildContractError(
            "derived_config_roi_outside_repository"
        ) from exc
    if not roi_relative.endswith(
        "data/annotations/roi/ROI_annotations.toy_adjusted.coco.json"
    ):
        raise ReviewedRebuildContractError("derived_config_adjusted_roi_required")

    config = copy.deepcopy(dict(base_config))
    config["lineage_id"] = lineage_id
    config["scientific_accepted_sha"] = scientific_accepted_sha
    config["run_root_default"] = run_root.resolve().as_posix()
    source = config["source"]
    source["roi"] = roi_relative
    if not all(value is False for value in config["authorization"].values()):
        raise ReviewedRebuildContractError(
            "derived_config_authorization_must_remain_false"
        )

    crop_root = resolve_source_path(root, config, "legacy_crop_root")
    video_root = resolve_source_path(root, config, "video_root")
    crop_files, _ = _tree_snapshot(crop_root)
    video_files, _ = _tree_snapshot(video_root)
    report = _build_source_report(
        root=root,
        config=config,
        crop_files=crop_files,
        video_files=video_files,
        crop_fingerprint=str(source["expected_crop_fingerprint"]),
        video_fingerprint=str(source["expected_video_fingerprint"]),
    )
    source["expected_legacy_sha256"] = str(report["legacy_csv_sha256"])
    source["expected_roi_sha256"] = str(report["roi_sha256"])
    source["expected_completion_audit_sha256"] = str(
        report["completion_audit_sha256"]
    )
    source["expected_legacy_rows"] = int(report["legacy_csv_rows"])
    source["expected_legacy_crop_files"] = int(report["crop_file_count"])
    source["expected_cvat_xml_count"] = int(report["cvat_xml_count"])
    source["expected_cvat_box_rows"] = int(report["cvat_box_rows"])
    source["expected_mixed_rows"] = int(report["projected_mixed_rows"])
    source["expected_pen_mask_sha256"] = str(report["pen_mask_sha256"])
    source["expected_cvat_xml_fingerprint"] = str(
        report["cvat_xml_fingerprint"]
    )
    source["expected_bundle_fingerprint"] = str(report["bundle_fingerprint"])
    verified = _build_source_report(
        root=root,
        config=config,
        crop_files=crop_files,
        video_files=video_files,
        crop_fingerprint=str(source["expected_crop_fingerprint"]),
        video_fingerprint=str(source["expected_video_fingerprint"]),
    )
    if not verified["valid"]:
        raise ReviewedRebuildContractError("derived_config_source_report_invalid")
    manifest = {
        "schema_version": DERIVED_CONFIG_VERSION,
        "status": "DERIVED_PENDING_FULL_SOURCE_PREFLIGHT",
        "lineage_id": lineage_id,
        "run_root": str(run_root.resolve()),
        "scientific_accepted_sha": scientific_accepted_sha,
        "base_config": {
            "path": str(base_config_path.resolve()),
            "sha256": sha256_file(base_config_path),
        },
        "adjusted_roi": {
            "path": str(roi),
            "sha256": sha256_file(roi),
        },
        "source_report": {
            key: verified.get(key)
            for key in (
                "legacy_csv_sha256",
                "legacy_csv_rows",
                "completion_audit_sha256",
                "completion_audit_status",
                "crop_file_count",
                "crop_fingerprint",
                "cvat_xml_count",
                "cvat_xml_fingerprint",
                "cvat_box_rows",
                "roi_sha256",
                "pen_mask_sha256",
                "video_fingerprint",
                "projected_mixed_rows",
                "bundle_fingerprint",
            )
        },
        "heavy_tree_fingerprint_policy": (
            "CARRIED_FROM_BASE_FOR_DERIVATION; MUST_PASS_FRESH_FULL_PREFLIGHT"
        ),
        "authorization_flags_all_false": True,
    }
    return config, manifest


def _validate_review_close_bindings(
    review_close_authority: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, str]],
) -> None:
    expected = review_close_authority["artifacts"]
    for name in (
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
    ):
        if name not in bindings or dict(bindings[name]) != dict(expected[name]):
            raise ReviewedRebuildContractError(
                f"application_review_close_binding_drift={name}"
            )


def _assert_semantic_union(
    composite: pd.DataFrame,
    primary: pd.DataFrame,
    control: pd.DataFrame,
    *,
    columns: Sequence[str],
    name: str,
) -> None:
    for frame_name, frame in (
        ("composite", composite),
        ("primary", primary),
        ("control", control),
    ):
        _require_columns(frame, columns, f"{name}_{frame_name}")
        _require_unique_key(frame, "review_unit_id", f"{name}_{frame_name}")
    expected = pd.concat(
        [primary[list(columns)], control[list(columns)]],
        ignore_index=True,
    )
    if expected["review_unit_id"].duplicated().any():
        raise ReviewedRebuildContractError(
            f"application_primary_control_overlap={name}"
        )
    actual = composite[list(columns)].copy()
    for column in columns:
        expected[column] = expected[column].fillna("").astype(str).str.strip()
        actual[column] = actual[column].fillna("").astype(str).str.strip()
    expected = expected.sort_values("review_unit_id", kind="mergesort").reset_index(
        drop=True
    )
    actual = actual.sort_values("review_unit_id", kind="mergesort").reset_index(
        drop=True
    )
    if not actual.equals(expected):
        raise ReviewedRebuildContractError(
            f"application_composite_union_mismatch={name}"
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReviewedRebuildContractError(
            f"reviewed_rebuild_{name}_missing={','.join(missing)}"
        )


def _require_unique_key(frame: pd.DataFrame, key: str, name: str) -> None:
    if frame[key].fillna("").astype(str).str.strip().eq("").any():
        raise ReviewedRebuildContractError(f"reviewed_rebuild_blank_key={name}")
    if frame[key].astype(str).duplicated().any():
        raise ReviewedRebuildContractError(f"reviewed_rebuild_duplicate_key={name}")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible authority payload."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "APPLICATION_AUTHORITY_VERSION",
    "AUTOCARRY_VERSION",
    "DERIVED_CONFIG_VERSION",
    "OVERLAY_AUDIT_VERSION",
    "ReviewedRebuildContractError",
    "audit_reviewed_label_overlay",
    "build_final_review_autocarry",
    "build_reviewed_application_views",
    "derive_reviewed_lineage_config",
    "freeze_reviewed_training_application_authority",
    "stable_payload_hash",
]
