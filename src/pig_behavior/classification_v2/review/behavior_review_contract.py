"""Canonical contract for behavior review decisions and native review units.

This module contains semantics shared by the review-unit builder, GUI, coverage
audit, and apply step. Keeping one contract prevents a decision accepted by one
stage from being interpreted differently by another stage.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CANONICAL_BEHAVIORS = frozenset(
    {
        "drink",
        "eat",
        "fight",
        "social-nose",
        "explore",
        "lying",
        "stand",
        "move",
        "sitting",
        "playwithtoy",
    }
)
VALID_DECISIONS = frozenset({"pending", "accept", "corrected", "exclude"})
VALID_ACTIONS = frozenset(
    {
        "",
        "main_train",
        "correct_and_keep",
        "low_weight_train",
        "exclude",
        "review_later",
    }
)
VALID_LABEL_STRENGTHS = frozenset({"", "strong", "medium", "weak", "boundary"})

LEGACY_DECISION_ALIASES = {"reject": "exclude", "uncertain": "accept"}
LEGACY_ACTION_ALIASES = {"keep": "main_train", "downweight": "low_weight_train"}

REQUIRED_DECISION_COLUMNS = (
    "review_item_id",
    "review_unit_id",
    "review_unit_type",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
    "review_template",
    "behavior_label",
    "original_behavior",
    "review_reason",
    "apply_scope",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
)

DECISION_TEXT_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "review_template",
    "behavior_label",
    "original_behavior",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_note",
)

MANIFEST_SNAPSHOT_PAIRS = (
    ("review_unit_type", "review_unit_type"),
    ("temporal_unit_key", "temporal_unit_key"),
    ("source_type", "source_type"),
    ("dataset_id", "dataset_id"),
    ("video_key", "video_key"),
    ("pig_id", "pig_id"),
    ("track_id", "track_id"),
    ("object_track_key", "object_track_key"),
    ("unit_start_frame", "unit_start_frame"),
    ("unit_end_frame", "unit_end_frame"),
    ("review_template", "review_template"),
    ("behavior_label", "behavior_label"),
    ("original_behavior", "behavior_label"),
    ("apply_scope", "apply_scope"),
)

BEHAVIOR_REVIEW_TEMPLATE = {
    "fight": "interaction",
    "social-nose": "interaction",
    "eat": "roi",
    "drink": "roi",
    "playwithtoy": "roi",
    "move": "motion",
    "explore": "motion",
    "stand": "motion",
    "lying": "posture",
    "sitting": "posture",
}

SOURCE_UNIT_CONTRACTS = {
    "legacy_recovered": {
        "frame_count": 16,
        "review_unit_type": "legacy_burst_16",
        "apply_scope": "whole_legacy_burst_16f",
    },
    "cvat_tracking_xml": {
        "frame_count": 6,
        "review_unit_type": "cvat_interval_6",
        "apply_scope": "cvat_interval_6f",
    },
}


def normalize_text(value: Any) -> str:
    """Return stable text for nullable CSV values."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def canonicalize_decisions(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize legacy aliases and deterministic action/weight defaults.

    Human confidence is never inferred. In particular, this function does not
    manufacture ``manual_label_strength`` when an older file omitted it.
    """
    out = decisions.copy()
    warnings: list[str] = []
    defaults: dict[str, Any] = {
        "review_unit_id": "",
        "temporal_unit_key": "",
        "review_template": "",
        "behavior_label": "",
        "original_behavior": "",
        "manual_review_decision": "pending",
        "manual_corrected_behavior": "",
        "manual_label_strength": "",
        "manual_training_action": "",
        "manual_sample_weight": pd.NA,
        "manual_note": "",
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    for column in DECISION_TEXT_COLUMNS:
        out[column] = out[column].map(normalize_text)
    out["manual_review_decision"] = out["manual_review_decision"].replace(
        "",
        "pending",
    )

    raw_decision = out["manual_review_decision"].copy()
    raw_action = out["manual_training_action"].copy()
    for old, new in LEGACY_DECISION_ALIASES.items():
        count = int(raw_decision.eq(old).sum())
        if count:
            warnings.append(f"legacy_decision_alias={old}->{new}:rows={count}")
    for old, new in LEGACY_ACTION_ALIASES.items():
        count = int(raw_action.eq(old).sum())
        if count:
            warnings.append(f"legacy_action_alias={old}->{new}:rows={count}")

    uncertain = raw_decision.eq("uncertain")
    out["manual_review_decision"] = raw_decision.replace(LEGACY_DECISION_ALIASES)
    out["manual_training_action"] = raw_action.replace(LEGACY_ACTION_ALIASES)
    out.loc[
        uncertain & out["manual_training_action"].eq(""),
        "manual_training_action",
    ] = "low_weight_train"

    decision = out["manual_review_decision"]
    action = out["manual_training_action"]
    default_actions = {
        "accept": "main_train",
        "corrected": "correct_and_keep",
        "exclude": "exclude",
    }
    for value, default_action in default_actions.items():
        mask = decision.eq(value) & action.eq("")
        out.loc[mask, "manual_training_action"] = default_action

    raw_weight = out["manual_sample_weight"].map(normalize_text)
    out["manual_sample_weight"] = pd.to_numeric(raw_weight, errors="coerce")
    invalid_weight_text = raw_weight.ne("") & out["manual_sample_weight"].isna()
    if invalid_weight_text.any():
        warnings.append(f"invalid_sample_weight_text={int(invalid_weight_text.sum())}")
    action = out["manual_training_action"]
    default_weights = {
        "main_train": 1.0,
        "correct_and_keep": 1.0,
        "low_weight_train": 0.5,
        "exclude": 0.0,
        "review_later": 0.0,
    }
    for value, weight in default_weights.items():
        mask = action.eq(value) & raw_weight.eq("")
        out.loc[mask, "manual_sample_weight"] = weight

    pending = decision.eq("pending")
    blank_pending_action = pending & action.eq("")
    out.loc[blank_pending_action, "manual_sample_weight"] = pd.NA
    return out, warnings


def validate_decision_semantics(
    decisions: pd.DataFrame,
    *,
    require_complete: bool,
) -> tuple[list[str], list[str]]:
    """Validate decision, correction, action, strength, and weight coherence."""
    errors: list[str] = []
    warnings: list[str] = []
    if decisions.empty:
        if require_complete:
            errors.append("no_decision_rows")
        return errors, warnings

    ids = decisions["review_unit_id"].map(normalize_text)
    blank_ids = int(ids.eq("").sum())
    duplicate_ids = int(ids.duplicated(keep=False).sum())
    if blank_ids:
        errors.append(f"blank_review_unit_id_rows={blank_ids}")
    if duplicate_ids:
        errors.append(f"duplicate_decision_rows={duplicate_ids}")

    decision = decisions["manual_review_decision"].map(normalize_text)
    correction = decisions["manual_corrected_behavior"].map(normalize_text)
    strength = decisions["manual_label_strength"].map(normalize_text)
    action = decisions["manual_training_action"].map(normalize_text)
    weight = pd.to_numeric(decisions["manual_sample_weight"], errors="coerce")

    invalid_decisions = sorted(set(decision) - VALID_DECISIONS)
    invalid_actions = sorted(set(action) - VALID_ACTIONS)
    invalid_strengths = sorted(set(strength) - VALID_LABEL_STRENGTHS)
    if invalid_decisions:
        errors.append(f"invalid_decisions={invalid_decisions}")
    if invalid_actions:
        errors.append(f"invalid_training_actions={invalid_actions}")
    if invalid_strengths:
        errors.append(f"invalid_label_strengths={invalid_strengths}")

    corrected = decision.eq("corrected")
    invalid_correction = corrected & ~correction.isin(CANONICAL_BEHAVIORS)
    unexpected_correction = ~corrected & correction.ne("")
    if invalid_correction.any():
        errors.append(f"invalid_corrected_behavior_rows={int(invalid_correction.sum())}")
    if unexpected_correction.any():
        errors.append(f"correction_without_corrected_decision={int(unexpected_correction.sum())}")

    allowed_by_decision = {
        "pending": {"", "review_later"},
        "accept": {"main_train", "low_weight_train"},
        "corrected": {"correct_and_keep", "low_weight_train"},
        "exclude": {"exclude"},
    }
    for value, allowed in allowed_by_decision.items():
        invalid = decision.eq(value) & ~action.isin(allowed)
        if invalid.any():
            errors.append(f"decision_action_conflict={value}:rows={int(invalid.sum())}")

    non_pending = ~decision.eq("pending")
    missing_strength = non_pending & strength.eq("")
    if missing_strength.any():
        message = f"active_decision_without_strength={int(missing_strength.sum())}"
        if require_complete:
            errors.append(message)
        else:
            warnings.append(message)

    out_of_range = weight.notna() & ((weight < 0.0) | (weight > 1.0))
    missing_active_weight = non_pending & weight.isna()
    zero_included = action.isin({"main_train", "correct_and_keep", "low_weight_train"}) & (
        weight.isna() | weight.le(0.0)
    )
    nonzero_excluded = action.isin({"exclude", "review_later"}) & weight.fillna(0.0).ne(0.0)
    not_low_weight = action.eq("low_weight_train") & (
        weight.isna() | weight.le(0.0) | weight.ge(1.0)
    )
    if out_of_range.any():
        errors.append(f"sample_weight_out_of_range={int(out_of_range.sum())}")
    if missing_active_weight.any():
        errors.append(f"active_decision_without_numeric_weight={int(missing_active_weight.sum())}")
    if zero_included.any():
        errors.append(f"included_action_without_positive_weight={int(zero_included.sum())}")
    if nonzero_excluded.any():
        errors.append(f"excluded_action_with_nonzero_weight={int(nonzero_excluded.sum())}")
    if not_low_weight.any():
        errors.append(f"low_weight_action_not_between_zero_one={int(not_low_weight.sum())}")

    pending = decision.eq("pending")
    pending_payload = pending & (correction.ne("") | strength.ne(""))
    if pending_payload.any():
        errors.append(f"pending_with_review_payload={int(pending_payload.sum())}")
    if require_complete and pending.any():
        errors.append(f"pending_review_unit_count={int(pending.sum())}")
    return errors, warnings


def audit_manifest_alignment(
    review_manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    allow_blank_snapshot: bool,
) -> tuple[list[str], list[str]]:
    """Detect stale or redirected decisions by comparing GUI snapshots.

    The canonical manifest is the authority for apply scope. Snapshot columns
    in a decision file are evidence that the reviewer saw that same unit.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if "review_unit_id" not in review_manifest.columns:
        return ["review_manifest_missing_review_unit_id"], warnings
    if "review_unit_id" not in decisions.columns:
        return ["decisions_missing_review_unit_id"], warnings

    manifest_ids = review_manifest["review_unit_id"].map(normalize_text)
    decision_ids = decisions["review_unit_id"].map(normalize_text)
    if manifest_ids.eq("").any():
        errors.append(f"blank_manifest_review_unit_id={int(manifest_ids.eq('').sum())}")
    if manifest_ids.duplicated(keep=False).any():
        count = int(manifest_ids.duplicated(keep=False).sum())
        errors.append(f"duplicate_review_manifest_rows={count}")

    expected = set(manifest_ids)
    observed = set(decision_ids)
    unexpected = sorted(observed - expected)
    if unexpected:
        errors.append(f"unexpected_review_unit_count={len(unexpected)}")

    manifest = review_manifest.copy()
    manifest["review_unit_id"] = manifest_ids
    decision_copy = decisions.copy()
    decision_copy["review_unit_id"] = decision_ids
    merged = decision_copy.merge(
        manifest,
        on="review_unit_id",
        how="inner",
        suffixes=("_decision", "_manifest"),
    )

    for decision_column, manifest_column in MANIFEST_SNAPSHOT_PAIRS:
        columns_available = (
            decision_column in decisions.columns
            and manifest_column in review_manifest.columns
        )
        if not columns_available:
            continue
        left_name = (
            f"{decision_column}_decision"
            if decision_column in review_manifest.columns
            else decision_column
        )
        right_name = (
            f"{manifest_column}_manifest"
            if manifest_column in decisions.columns
            else manifest_column
        )
        if left_name not in merged.columns or right_name not in merged.columns:
            continue
        left = _snapshot_values(merged[left_name], decision_column)
        right = _snapshot_values(merged[right_name], manifest_column)
        blank = left.eq("")
        mismatch = ~blank & left.ne(right)
        if blank.any():
            message = f"blank_decision_snapshot={decision_column}:rows={int(blank.sum())}"
            if allow_blank_snapshot:
                warnings.append(message)
            else:
                errors.append(message)
        if mismatch.any():
            errors.append(
                f"stale_decision_snapshot={decision_column}:rows={int(mismatch.sum())}"
            )
    return errors, warnings


def _snapshot_values(series: pd.Series, column: str) -> pd.Series:
    """Normalize numeric frame boundaries without weakening text identity checks."""
    if column not in {"unit_start_frame", "unit_end_frame"}:
        return series.map(normalize_text)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(_format_frame_boundary)


def _format_frame_boundary(value: Any) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else format(numeric, ".17g")


def audit_review_unit_contract(review_units: pd.DataFrame) -> dict[str, Any]:
    """Validate source-specific native-unit and review-template semantics."""
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "review_unit_id",
        "review_unit_type",
        "temporal_unit_key",
        "source_type",
        "unit_start_frame",
        "unit_end_frame",
        "behavior_label",
        "review_template",
        "apply_scope",
    }
    missing = sorted(required - set(review_units.columns))
    if missing:
        return {
            "rows": int(len(review_units)),
            "errors": [f"missing_review_unit_columns={missing}"],
            "warnings": warnings,
        }
    if "window_uid" in review_units.columns:
        errors.append("forbidden_window_uid_column")

    ids = review_units["review_unit_id"].map(normalize_text)
    temporal_keys = review_units["temporal_unit_key"].map(normalize_text)
    for name, values in [
        ("review_unit_id", ids),
        ("temporal_unit_key", temporal_keys),
    ]:
        blank = int(values.eq("").sum())
        duplicate = int(values.duplicated(keep=False).sum())
        if blank:
            errors.append(f"blank_{name}_rows={blank}")
        if duplicate:
            errors.append(f"duplicate_{name}_rows={duplicate}")

    source = review_units["source_type"].map(normalize_text)
    unsupported = sorted(set(source) - set(SOURCE_UNIT_CONTRACTS))
    if unsupported:
        errors.append(f"unsupported_source_type={unsupported}")

    behavior = review_units["behavior_label"].map(normalize_text)
    invalid_behavior = sorted(set(behavior) - CANONICAL_BEHAVIORS)
    if invalid_behavior:
        errors.append(f"invalid_behavior_label={invalid_behavior}")

    start = pd.to_numeric(review_units["unit_start_frame"], errors="coerce")
    end = pd.to_numeric(review_units["unit_end_frame"], errors="coerce")
    invalid_boundary = start.isna() | end.isna() | end.lt(start)
    if invalid_boundary.any():
        errors.append(f"invalid_unit_boundary_rows={int(invalid_boundary.sum())}")
    noninteger_boundary = (
        start.notna() & start.mod(1).ne(0)
    ) | (
        end.notna() & end.mod(1).ne(0)
    )
    if noninteger_boundary.any():
        errors.append(f"noninteger_unit_boundary_rows={int(noninteger_boundary.sum())}")
    observed_count = end - start + 1

    for source_type, contract in SOURCE_UNIT_CONTRACTS.items():
        source_mask = source.eq(source_type)
        wrong_count = source_mask & observed_count.ne(contract["frame_count"])
        wrong_type = source_mask & review_units["review_unit_type"].map(
            normalize_text
        ).ne(contract["review_unit_type"])
        wrong_scope = source_mask & review_units["apply_scope"].map(normalize_text).ne(
            contract["apply_scope"]
        )
        if wrong_count.any():
            errors.append(
                f"wrong_native_frame_count={source_type}:rows={int(wrong_count.sum())}"
            )
        if wrong_type.any():
            errors.append(f"wrong_review_unit_type={source_type}:rows={int(wrong_type.sum())}")
        if wrong_scope.any():
            errors.append(f"wrong_apply_scope={source_type}:rows={int(wrong_scope.sum())}")

    cvat = source.eq("cvat_tracking_xml")
    bad_anchor = cvat & start.notna() & start.mod(6).ne(0)
    if bad_anchor.any():
        errors.append(f"cvat_unit_start_not_anchor_multiple=rows={int(bad_anchor.sum())}")

    expected_template = behavior.map(BEHAVIOR_REVIEW_TEMPLATE)
    actual_template = review_units["review_template"].map(normalize_text)
    wrong_template = expected_template.notna() & actual_template.ne(expected_template)
    if wrong_template.any():
        errors.append(f"wrong_behavior_review_template_rows={int(wrong_template.sum())}")

    if "unit_frame_count" in review_units.columns:
        declared_count = pd.to_numeric(review_units["unit_frame_count"], errors="coerce")
        mismatch = declared_count.ne(observed_count)
        if mismatch.any():
            errors.append(f"unit_frame_count_mismatch_rows={int(mismatch.sum())}")
    if "display_frame_indices" in review_units.columns:
        invalid_display = 0
        for row_index, text in review_units["display_frame_indices"].items():
            expected = _expected_frame_indices(start.loc[row_index], end.loc[row_index])
            observed = _parse_frame_indices(text)
            invalid_display += int(observed != expected)
        if invalid_display:
            errors.append(f"display_frame_indices_mismatch_rows={invalid_display}")

    return {
        "rows": int(len(review_units)),
        "source_counts": source.value_counts(dropna=False).to_dict(),
        "behavior_counts": behavior.value_counts(dropna=False).to_dict(),
        "errors": errors,
        "warnings": warnings,
    }


def _parse_frame_indices(value: Any) -> list[int]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        return [int(float(token.strip())) for token in text.split(",") if token.strip()]
    except ValueError:
        return []


def _expected_frame_indices(start: Any, end: Any) -> list[int]:
    if pd.isna(start) or pd.isna(end):
        return []
    return list(range(int(start), int(end) + 1))
