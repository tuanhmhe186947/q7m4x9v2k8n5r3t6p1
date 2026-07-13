"""Review and label-quality policy for classification_v2.

This step runs after ROI features.

It does not replace context_policy.py. Context policy decides whether a row is
structurally usable. This module interprets behavior-vs-ROI consistency and
optional manual review decisions into final training attributes.

Current focus:
- eat -> feeder
- drink -> drinker
- playwithtoy -> toy
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    ROI_DOMINANT_BEHAVIORS,
    VALID_BEHAVIOR_SET,
)

BEHAVIOR_TO_TARGET_ROI: dict[str, str] = {
    "eat": "feeder",
    "drink": "drinker",
    "playwithtoy": "toy",
}

LABEL_STRENGTHS: set[str] = {
    "strong",
    "medium",
    "weak",
    "boundary",
    "unknown",
}

AMBIGUITY_GROUPS: set[str] = {
    "none",
    "roi_feeding_drinking_toy",
    "aggression_social",
    "motion_state",
    "posture",
    "unknown",
}

REVIEW_DECISIONS: set[str] = {
    "auto_accept",
    "accept",
    "corrected",
    "exclude",
    "pending",
    "not_required",
}

TRAINING_ACTIONS: set[str] = {
    "main_train",
    "low_weight_train",
    "robust_train_only",
    "exclude",
    "pending",
}

REQUIRED_REVIEW_INPUT_COLUMNS: tuple[str, ...] = (
    "behavior",
    "bbox_valid",
    "include_in_training",
    "use_for_roi_training",
    "roi_target_available",
    "roi_target_near",
    "roi_target_contact",
)


MANUAL_REVIEW_COLUMNS: tuple[str, ...] = (
    "manual_review_decision",
    "manual_label_strength",
    "manual_corrected_behavior",
    "manual_ambiguity_group",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
)


def add_roi_label_review_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Add automatic ROI-label consistency and label-strength attributes.

    This function should be called after build_roi_features().
    It does not drop rows and does not modify the original behavior label.
    """
    missing = [c for c in REQUIRED_REVIEW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing review-policy input columns: {missing}")

    out = df.copy()

    if "review_row_index" not in out.columns:
        out["review_row_index"] = range(len(out))

    out["behavior_original"] = out["behavior"]

    is_roi_behavior = out["behavior"].isin(ROI_DOMINANT_BEHAVIORS)
    target_available = _to_bool_series(out["roi_target_available"])
    target_contact = _to_bool_series(out["roi_target_contact"])
    target_near = _to_bool_series(out["roi_target_near"])

    out["roi_consistency_status"] = "not_required"
    out["label_strength_auto"] = "strong"
    out["ambiguity_group_auto"] = "none"
    out["review_reason_auto"] = "not_roi_policy_scope"
    out["review_required_auto"] = False
    out["training_action_auto"] = "main_train"
    out["sample_weight_auto"] = 1.0

    contact_mask = is_roi_behavior & target_available & target_contact
    near_mask = is_roi_behavior & target_available & target_near & ~target_contact
    far_mask = is_roi_behavior & target_available & ~target_near & ~target_contact
    unavailable_mask = is_roi_behavior & ~target_available

    out.loc[contact_mask, "roi_consistency_status"] = "target_roi_contact"
    out.loc[contact_mask, "label_strength_auto"] = "strong"
    out.loc[contact_mask, "ambiguity_group_auto"] = "roi_feeding_drinking_toy"
    out.loc[contact_mask, "review_reason_auto"] = "roi_target_contact"
    out.loc[contact_mask, "review_required_auto"] = False
    out.loc[contact_mask, "training_action_auto"] = "main_train"
    out.loc[contact_mask, "sample_weight_auto"] = 1.0

    out.loc[near_mask, "roi_consistency_status"] = "target_roi_near_no_contact"
    out.loc[near_mask, "label_strength_auto"] = "medium"
    out.loc[near_mask, "ambiguity_group_auto"] = "roi_feeding_drinking_toy"
    out.loc[near_mask, "review_reason_auto"] = "roi_target_near_but_no_contact"
    out.loc[near_mask, "review_required_auto"] = True
    out.loc[near_mask, "training_action_auto"] = "main_train"
    out.loc[near_mask, "sample_weight_auto"] = 0.75

    out.loc[far_mask, "roi_consistency_status"] = "target_roi_far"
    out.loc[far_mask, "label_strength_auto"] = "weak"
    out.loc[far_mask, "ambiguity_group_auto"] = "roi_feeding_drinking_toy"
    out.loc[far_mask, "review_reason_auto"] = "roi_target_far_from_labeled_behavior"
    out.loc[far_mask, "review_required_auto"] = True
    out.loc[far_mask, "training_action_auto"] = "low_weight_train"
    out.loc[far_mask, "sample_weight_auto"] = 0.35

    out.loc[unavailable_mask, "roi_consistency_status"] = "target_roi_unavailable"
    out.loc[unavailable_mask, "label_strength_auto"] = "weak"
    out.loc[unavailable_mask, "ambiguity_group_auto"] = "roi_feeding_drinking_toy"
    out.loc[unavailable_mask, "review_reason_auto"] = "target_roi_unavailable"
    out.loc[unavailable_mask, "review_required_auto"] = True
    out.loc[unavailable_mask, "training_action_auto"] = "low_weight_train"
    out.loc[unavailable_mask, "sample_weight_auto"] = 0.35

    return out


def build_behavior_review_template(
    df: pd.DataFrame,
    *,
    only_review_required: bool = True,
) -> pd.DataFrame:
    """Build a CSV template for manual review.

    Usually this is used for target ROI near/far/unavailable rows.
    """
    out = df.copy()

    if only_review_required:
        out = out[_to_bool_series(out["review_required_auto"])].copy()

    preferred_cols = [
        "review_row_index",
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_key",
        "clip_id",
        "task_id",
        "frame_uid",
        "image_name",
        "image_key",
        "frame_index",
        "relative_frame_index",
        "pig_id",
        "track_id",
        "tracklet_id",
        "behavior",
        "behavior_original",
        "roi_target_class",
        "roi_consistency_status",
        "label_strength_auto",
        "ambiguity_group_auto",
        "review_reason_auto",
        "review_required_auto",
        "roi_target_available",
        "roi_target_near",
        "roi_target_contact",
        "roi_target_min_dist_n",
        "roi_target_max_overlap_ratio",
        "roi_target_max_iou",
        "roi_target_center_inside",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "bbox_w",
        "bbox_h",
        "bbox_valid",
        "include_in_training",
        "use_for_roi_training",
    ]

    cols = [c for c in preferred_cols if c in out.columns]
    template = out[cols].copy()

    for col in MANUAL_REVIEW_COLUMNS:
        if col not in template.columns:
            template[col] = ""

    return template


def apply_behavior_review_decisions(
    df: pd.DataFrame,
    review_decisions: pd.DataFrame | None = None,
    *,
    pending_policy: str = "auto",
    include_weak_in_training: bool = False,
) -> pd.DataFrame:
    """Apply manual review decisions and produce final training attributes.

    Parameters
    ----------
    df:
        Frame-level feature table after add_roi_label_review_attributes().
    review_decisions:
        Manual review CSV. Can be None.
    pending_policy:
        "auto":
            Keep rows using automatic policy if manual review is missing.
            This is recommended while only ROI review is partially finished.
        "exclude":
            Exclude rows that still require manual review.
    include_weak_in_training:
        If True, weak samples can be used for robust training only.
    """
    if pending_policy not in {"auto", "exclude"}:
        raise ValueError("pending_policy must be 'auto' or 'exclude'.")

    out = df.copy()

    for col in MANUAL_REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    if review_decisions is not None and not review_decisions.empty:
        out = _merge_review_decisions(out, review_decisions)

    out["review_decision"] = (
        out["manual_review_decision"].fillna("").astype(str).str.strip()
    )

    missing_manual = out["review_decision"].eq("")
    requires_review = _to_bool_series(out["review_required_auto"])

    out.loc[missing_manual & requires_review, "review_decision"] = "pending"
    out.loc[missing_manual & ~requires_review, "review_decision"] = "auto_accept"

    out["label_strength"] = (
        out["manual_label_strength"].fillna("").astype(str).str.strip()
    )
    out.loc[out["label_strength"].eq(""), "label_strength"] = out[
        "label_strength_auto"
    ]

    out["ambiguity_group"] = (
        out["manual_ambiguity_group"].fillna("").astype(str).str.strip()
    )
    out.loc[out["ambiguity_group"].eq(""), "ambiguity_group"] = out[
        "ambiguity_group_auto"
    ]

    corrected_behavior = (
        out["manual_corrected_behavior"].fillna("").astype(str).str.strip()
    )

    out["behavior_train"] = out["behavior"]
    correction_mask = corrected_behavior.ne("")
    out.loc[correction_mask, "behavior_train"] = corrected_behavior[correction_mask]

    out["training_action_final"] = (
        out["manual_training_action"].fillna("").astype(str).str.strip()
    )
    out.loc[out["training_action_final"].eq(""), "training_action_final"] = out[
        "training_action_auto"
    ]

    manual_weight = pd.to_numeric(out["manual_sample_weight"], errors="coerce")
    auto_weight = pd.to_numeric(out["sample_weight_auto"], errors="coerce").fillna(1.0)

    out["training_weight_final"] = manual_weight.fillna(auto_weight)
    out["training_weight_final"] = out["training_weight_final"].clip(
        lower=0.0,
        upper=1.0,
    )

    base_include = _to_bool_series(out["include_in_training"])
    bbox_valid = _to_bool_series(out["bbox_valid"])
    valid_behavior = out["behavior_train"].isin(VALID_BEHAVIOR_SET)

    exclude_decision = out["review_decision"].isin({"exclude"})
    exclude_action = out["training_action_final"].eq("exclude")
    exclude_mask = exclude_decision | exclude_action

    if pending_policy == "exclude":
        exclude_mask = exclude_mask | out["review_decision"].eq("pending")

    out["include_in_training_final"] = (
        base_include
        & bbox_valid
        & valid_behavior
        & ~exclude_mask
    )

    strong_or_medium = out["label_strength"].isin(["strong", "medium"])
    weak_allowed = include_weak_in_training & out["label_strength"].eq("weak")

    out["use_for_main_train_final"] = (
        out["include_in_training_final"]
        & strong_or_medium
        & out["training_action_final"].isin(["main_train"])
    )

    out["use_for_robust_train_final"] = (
        out["include_in_training_final"]
        & (strong_or_medium | weak_allowed)
        & out["training_action_final"].isin(
            ["main_train", "low_weight_train", "robust_train_only"]
        )
    )

    out["use_for_roi_training_final"] = (
        out["include_in_training_final"]
        & out["behavior_train"].isin(ROI_DOMINANT_BEHAVIORS)
        & out["roi_consistency_status"].isin(
            ["target_roi_contact", "target_roi_near_no_contact"]
        )
        & (strong_or_medium | weak_allowed)
    )

    out.loc[exclude_mask | ~valid_behavior, "training_weight_final"] = 0.0

    out["review_policy_status"] = "ok"
    out.loc[out["review_decision"].eq("pending"), "review_policy_status"] = (
        "pending_review"
    )
    out.loc[exclude_mask, "review_policy_status"] = "excluded"
    out.loc[~valid_behavior, "review_policy_status"] = "invalid_behavior_train"

    return out


def audit_review_policy(df: pd.DataFrame) -> dict[str, Any]:
    """Audit reviewed frame-level features before sequence building."""
    errors: list[str] = []
    warnings: list[str] = []

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
        "roi_consistency_status",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing_review_policy_columns={missing}")

    invalid_behavior_train = 0
    if "behavior_train" in df.columns:
        invalid_behavior_train = int(
            (~df["behavior_train"].isin(VALID_BEHAVIOR_SET)).sum()
        )

    invalid_label_strengths: list[str] = []
    if "label_strength" in df.columns:
        invalid_label_strengths = sorted(
            set(df["label_strength"].dropna().astype(str)).difference(LABEL_STRENGTHS)
        )
        if invalid_label_strengths:
            errors.append(f"invalid_label_strengths={invalid_label_strengths}")

    invalid_ambiguity_groups: list[str] = []
    if "ambiguity_group" in df.columns:
        invalid_ambiguity_groups = sorted(
            set(df["ambiguity_group"].dropna().astype(str)).difference(
                AMBIGUITY_GROUPS
            )
        )
        if invalid_ambiguity_groups:
            errors.append(f"invalid_ambiguity_groups={invalid_ambiguity_groups}")

    invalid_review_decisions: list[str] = []
    if "review_decision" in df.columns:
        invalid_review_decisions = sorted(
            set(df["review_decision"].dropna().astype(str)).difference(
                REVIEW_DECISIONS
            )
        )
        if invalid_review_decisions:
            errors.append(f"invalid_review_decisions={invalid_review_decisions}")

    invalid_training_actions: list[str] = []
    if "training_action_final" in df.columns:
        invalid_training_actions = sorted(
            set(df["training_action_final"].dropna().astype(str)).difference(
                TRAINING_ACTIONS
            )
        )
        if invalid_training_actions:
            errors.append(f"invalid_training_actions={invalid_training_actions}")

    invalid_training_weight = 0
    if "training_weight_final" in df.columns:
        weight = pd.to_numeric(df["training_weight_final"], errors="coerce")
        invalid_training_weight = int((weight.isna() | weight.lt(0) | weight.gt(1)).sum())
        if invalid_training_weight:
            errors.append(f"invalid_training_weight={invalid_training_weight}")

    roi_far_used_for_roi_training = 0
    if {
        "use_for_roi_training_final",
        "roi_consistency_status",
    }.issubset(df.columns):
        roi_far_used_for_roi_training = int(
            (
                _to_bool_series(df["use_for_roi_training_final"])
                & df["roi_consistency_status"].isin(
                    ["target_roi_far", "target_roi_unavailable"]
                )
            ).sum()
        )
        if roi_far_used_for_roi_training:
            errors.append(
                "roi_far_or_unavailable_used_for_roi_training="
                f"{roi_far_used_for_roi_training}"
            )

    excluded_nonzero_weight = 0
    if {
        "include_in_training_final",
        "training_weight_final",
    }.issubset(df.columns):
        excluded_nonzero_weight = int(
            (
                ~_to_bool_series(df["include_in_training_final"])
                & pd.to_numeric(df["training_weight_final"], errors="coerce").fillna(0).gt(0)
            ).sum()
        )
        if excluded_nonzero_weight:
            warnings.append(f"excluded_rows_with_nonzero_weight={excluded_nonzero_weight}")

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique()) if "frame_uid" in df.columns else 0,
        "behaviors": _value_counts_dict(df, "behavior"),
        "behavior_train": _value_counts_dict(df, "behavior_train"),
        "roi_consistency_status": _value_counts_dict(df, "roi_consistency_status"),
        "label_strength": _value_counts_dict(df, "label_strength"),
        "ambiguity_group": _value_counts_dict(df, "ambiguity_group"),
        "review_decision": _value_counts_dict(df, "review_decision"),
        "review_policy_status": _value_counts_dict(df, "review_policy_status"),
        "training_action_final": _value_counts_dict(df, "training_action_final"),
        "include_in_training_final": _value_counts_dict(
            df,
            "include_in_training_final",
        ),
        "use_for_main_train_final": _value_counts_dict(
            df,
            "use_for_main_train_final",
        ),
        "use_for_robust_train_final": _value_counts_dict(
            df,
            "use_for_robust_train_final",
        ),
        "use_for_roi_training_final": _value_counts_dict(
            df,
            "use_for_roi_training_final",
        ),
        "invalid_behavior_train": invalid_behavior_train,
        "invalid_training_weight": invalid_training_weight,
        "roi_far_used_for_roi_training": roi_far_used_for_roi_training,
        "excluded_nonzero_weight": excluded_nonzero_weight,
        "errors": errors,
        "warnings": warnings,
    }


def _merge_review_decisions(
    df: pd.DataFrame,
    review_decisions: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    review = review_decisions.copy()

    for col in MANUAL_REVIEW_COLUMNS:
        if col not in review.columns:
            review[col] = ""

    if "review_row_index" in review.columns and "review_row_index" in out.columns:
        key: str | list[str] = "review_row_index"
    elif "row_index" in review.columns and "review_row_index" in out.columns:
        review = review.rename(columns={"row_index": "review_row_index"})
        key = "review_row_index"
    else:
        key_cols = [
            "source_type",
            "dataset_id",
            "video_key",
            "frame_uid",
            "pig_id",
            "behavior",
        ]
        missing = [c for c in key_cols if c not in review.columns or c not in out.columns]
        if missing:
            raise ValueError(
                "Review decisions need review_row_index/row_index or stable key columns. "
                f"Missing: {missing}"
            )
        key = key_cols

    keep_cols = [key] if isinstance(key, str) else list(key)
    keep_cols += list(MANUAL_REVIEW_COLUMNS)
    review = review[keep_cols].copy()
    key_columns = [key] if isinstance(key, str) else list(key)
    _validate_review_merge_keys(out, review, key_columns)

    out = out.drop(
        columns=[c for c in MANUAL_REVIEW_COLUMNS if c in out.columns],
        errors="ignore",
    )
    input_rows = len(out)
    out = out.merge(
        review,
        on=key,
        how="left",
        validate="one_to_one",
    )
    if len(out) != input_rows:
        raise ValueError(
            f"Review decision merge changed row count: {input_rows} -> {len(out)}"
        )

    for col in MANUAL_REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    return out


def _validate_review_merge_keys(
    frame: pd.DataFrame,
    review: pd.DataFrame,
    key_columns: list[str],
) -> None:
    """Require unique, complete decision keys that exist in the frame table."""

    frame_blank = _blank_key_row_count(frame, key_columns)
    review_blank = _blank_key_row_count(review, key_columns)
    frame_duplicate = int(frame.duplicated(key_columns, keep=False).sum())
    review_duplicate = int(review.duplicated(key_columns, keep=False).sum())
    errors: list[str] = []
    if frame_blank:
        errors.append(f"blank_frame_review_key_rows={frame_blank}")
    if review_blank:
        errors.append(f"blank_decision_review_key_rows={review_blank}")
    if frame_duplicate:
        errors.append(f"duplicate_frame_review_key_rows={frame_duplicate}")
    if review_duplicate:
        errors.append(f"duplicate_decision_review_key_rows={review_duplicate}")

    if not errors:
        frame_keys = pd.MultiIndex.from_frame(frame[key_columns])
        review_keys = pd.MultiIndex.from_frame(review[key_columns])
        unmatched = int((~review_keys.isin(frame_keys)).sum())
        if unmatched:
            errors.append(f"unmatched_decision_review_key_rows={unmatched}")
    if errors:
        raise ValueError("Review decision key contract failed: " + "; ".join(errors))


def _blank_key_row_count(frame: pd.DataFrame, key_columns: list[str]) -> int:
    """Count rows with a missing or blank component in a composite key."""

    blank = pd.DataFrame(index=frame.index)
    for column in key_columns:
        values = frame[column]
        blank[column] = values.isna() | values.astype(str).str.strip().eq("")
    return int(blank.any(axis=1).sum())


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
