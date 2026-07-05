"""Context and training policy normalization for classification_v2.

This module recomputes final policy columns after all sources have been merged.

Design rules:
- Do not reject rows only because global_context_pig_count < 8.
- Keep actor-only rows for non-interaction behaviors.
- Require local partner context only for fight/social-nose.
- Hidden has already been manually reviewed, so it is trusted metadata.
  It does not reject or down-weight a sample by itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    DEFAULT_PIG_IDS,
    INTERACTION_BEHAVIORS,
    MOTION_DOMINANT_BEHAVIORS,
    QA_STATUSES,
    ROI_DOMINANT_BEHAVIORS,
    SHAPE_DOMINANT_BEHAVIORS,
    TRAINING_TIERS,
    VALID_BEHAVIOR_SET,
    behavior_to_coarse,
    normalize_behavior,
    normalize_hidden,
    normalize_pig_id,
)

REQUIRED_POLICY_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "hidden",
    "bbox_valid",
)


def apply_context_policy(
    frame_objects: pd.DataFrame,
    *,
    expected_pig_count: int = 8,
    recompute_context: bool = True,
    require_full_8_for_eval: bool = False,
) -> pd.DataFrame:
    """Apply final context/training policy to merged frame objects.

    Parameters
    ----------
    frame_objects:
        Merged canonical frame-object dataframe.
    expected_pig_count:
        Expected full context pig count. Default is 8.
    recompute_context:
        If True, recompute context columns from frame_uid/pig_id.
        This is recommended after merge.
    require_full_8_for_eval:
        If True, use_for_main_eval is True only for full-context rows.
        Default False because legacy/selected annotations are valid partial context.
    """
    missing = [c for c in REQUIRED_POLICY_COLUMNS if c not in frame_objects.columns]
    if missing:
        raise ValueError(f"Missing required policy columns: {missing}")

    out = frame_objects.copy()

    out = _normalize_labels(out)
    out = _ensure_bbox_valid(out)

    if recompute_context or _needs_context_recompute(out):
        out = _recompute_context_columns(
            out,
            expected_pig_count=expected_pig_count,
        )

    out = _apply_behavior_specific_context(
        out,
        expected_pig_count=expected_pig_count,
    )

    out = _apply_training_flags(
        out,
        require_full_8_for_eval=require_full_8_for_eval,
    )

    return out


def audit_context_policy(df: pd.DataFrame) -> dict[str, Any]:
    """Return audit summary after applying context policy."""
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_POLICY_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing_required_columns={missing}")

    if "training_tier" in df.columns:
        invalid_tiers = sorted(
            set(df["training_tier"].dropna().astype(str)).difference(TRAINING_TIERS)
        )
        if invalid_tiers:
            errors.append(f"invalid_training_tiers={invalid_tiers}")

    if "qa_status" in df.columns:
        invalid_qa = sorted(
            set(df["qa_status"].dropna().astype(str)).difference(QA_STATUSES)
        )
        if invalid_qa:
            errors.append(f"invalid_qa_statuses={invalid_qa}")

    if "bbox_valid" in df.columns:
        bbox_valid = _to_bool_series(df["bbox_valid"])
        invalid_bbox = int((~bbox_valid).sum())
    else:
        invalid_bbox = -1
        errors.append("bbox_valid_missing")

    if "behavior" in df.columns:
        behavior_valid = df["behavior"].isin(VALID_BEHAVIOR_SET)
        invalid_behavior = int((~behavior_valid).sum())
    else:
        invalid_behavior = -1
        errors.append("behavior_missing")

    if "training_tier" in df.columns and df["training_tier"].astype(str).eq(
        "warning"
    ).any():
        errors.append("training_tier_warning_should_not_exist")

    if "social_missing_mask" in df.columns:
        social_missing_count = int(_to_bool_series(df["social_missing_mask"]).sum())
    else:
        social_missing_count = 0

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique()) if "frame_uid" in df.columns else 0,
        "sources": _value_counts_dict(df, "source_type"),
        "datasets": _value_counts_dict(df, "dataset_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "hidden": _value_counts_dict(df, "hidden"),
        "context_pig_count": _value_counts_dict(df, "global_context_pig_count"),
        "annotation_scope": _value_counts_dict(df, "annotation_scope"),
        "local_context_quality": _value_counts_dict(df, "local_context_quality"),
        "social_feature_quality": _value_counts_dict(df, "social_feature_quality"),
        "training_tier": _value_counts_dict(df, "training_tier"),
        "qa_status": _value_counts_dict(df, "qa_status"),
        "include_in_training": _value_counts_dict(df, "include_in_training"),
        "use_for_visual_training": _value_counts_dict(df, "use_for_visual_training"),
        "use_for_shape_training": _value_counts_dict(df, "use_for_shape_training"),
        "use_for_motion_training": _value_counts_dict(df, "use_for_motion_training"),
        "use_for_roi_training": _value_counts_dict(df, "use_for_roi_training"),
        "use_for_social_training": _value_counts_dict(df, "use_for_social_training"),
        "use_for_main_eval": _value_counts_dict(df, "use_for_main_eval"),
        "bbox_valid": _value_counts_dict(df, "bbox_valid"),
        "invalid_bbox": invalid_bbox,
        "invalid_behavior": invalid_behavior,
        "social_missing_count": social_missing_count,
        "errors": errors,
        "warnings": warnings,
    }


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["behavior"] = out["behavior"].map(normalize_behavior)
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = out["hidden"].map(normalize_hidden)
    out["pig_id"] = out["pig_id"].map(normalize_pig_id)

    out["hidden_is_trusted"] = True
    out["visibility_quality"] = "visible"
    out.loc[out["hidden"].eq("Yes"), "visibility_quality"] = "hidden"

    return out


def _ensure_bbox_valid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "bbox_valid" in out.columns:
        out["bbox_valid"] = _to_bool_series(out["bbox_valid"])
        return out

    for col in ["x1", "y1", "x2", "y2"]:
        if col not in out.columns:
            out["bbox_valid"] = False
            return out
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["bbox_valid"] = (
        out["x1"].notna()
        & out["y1"].notna()
        & out["x2"].notna()
        & out["y2"].notna()
        & out["x2"].gt(out["x1"])
        & out["y2"].gt(out["y1"])
    )

    return out


def _needs_context_recompute(df: pd.DataFrame) -> bool:
    required = {
        "global_context_pig_count",
        "global_context_complete_8",
        "local_context_pig_count",
        "duplicate_pig_id_in_frame",
        "context_overfull",
        "missing_global_pig_ids",
    }
    return not required.issubset(set(df.columns))


def _recompute_context_columns(
    df: pd.DataFrame,
    *,
    expected_pig_count: int,
) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "global_context_pig_count",
        "global_context_complete_8",
        "local_context_pig_count",
        "duplicate_pig_id_in_frame",
        "context_overfull",
        "missing_global_pig_ids",
    ]:
        if col in out.columns:
            out = out.drop(columns=[col])

    counts = (
        out.groupby("frame_uid", dropna=False)["pig_id"]
        .nunique(dropna=True)
        .rename("global_context_pig_count")
    )

    pig_sets = (
        out.groupby("frame_uid", dropna=False)["pig_id"]
        .apply(lambda values: set(values.dropna().astype(str)))
        .rename("present_pig_ids")
    )

    duplicate = (
        out.groupby(["frame_uid", "pig_id"], dropna=False)
        .size()
        .gt(1)
        .groupby("frame_uid")
        .any()
        .rename("duplicate_pig_id_in_frame")
    )

    out = out.merge(counts, left_on="frame_uid", right_index=True, how="left")
    out = out.merge(pig_sets, left_on="frame_uid", right_index=True, how="left")
    out = out.merge(duplicate, left_on="frame_uid", right_index=True, how="left")

    expected_ids = set(DEFAULT_PIG_IDS[:expected_pig_count])

    out["global_context_pig_count"] = pd.to_numeric(
        out["global_context_pig_count"],
        errors="coerce",
    ).fillna(0)

    out["global_context_complete_8"] = out["global_context_pig_count"].eq(
        expected_pig_count
    )
    out["context_overfull"] = out["global_context_pig_count"].gt(expected_pig_count)
    out["local_context_pig_count"] = out["global_context_pig_count"]

    out["duplicate_pig_id_in_frame"] = out[
        "duplicate_pig_id_in_frame"
    ].fillna(False)

    out["missing_global_pig_ids"] = out["present_pig_ids"].apply(
        lambda present: "|".join(sorted(expected_ids.difference(present)))
        if isinstance(present, set)
        else "|".join(sorted(expected_ids))
    )

    out = out.drop(columns=["present_pig_ids"])

    return out


def _apply_behavior_specific_context(
    df: pd.DataFrame,
    *,
    expected_pig_count: int,
) -> pd.DataFrame:
    out = df.copy()

    count = pd.to_numeric(out["local_context_pig_count"], errors="coerce").fillna(0)
    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    has_partner = count.ge(2)
    full_context = count.eq(expected_pig_count)

    out["annotation_scope"] = "selected_actor_group"
    out.loc[count.le(1), "annotation_scope"] = "actor_only"
    out.loc[is_social & has_partner, "annotation_scope"] = "interaction_pair_or_group"
    out.loc[full_context, "annotation_scope"] = "full_context"

    out["local_context_quality"] = "selected_context_ok"
    out.loc[count.le(1), "local_context_quality"] = "actor_only_ok"
    out.loc[is_social & has_partner, "local_context_quality"] = "interaction_context_ok"
    out.loc[is_social & ~has_partner, "local_context_quality"] = (
        "missing_interaction_partner"
    )
    out.loc[full_context, "local_context_quality"] = "full_context"

    out["social_feature_required"] = is_social
    out["social_missing_mask"] = is_social & ~has_partner

    out["social_feature_quality"] = "unknown"
    out.loc[is_social & ~has_partner, "social_feature_quality"] = "missing_context"
    out.loc[is_social & has_partner, "social_feature_quality"] = "interaction_context"
    out.loc[is_social & full_context, "social_feature_quality"] = "full_context"

    out["interaction_partner_count"] = 0
    out.loc[is_social, "interaction_partner_count"] = (count[is_social] - 1).clip(
        lower=0
    )

    out["interaction_partner_ids"] = _interaction_partner_ids(out)

    out["context_quality"] = "partial_or_selected_context"
    out.loc[full_context, "context_quality"] = "full_context"
    out.loc[out["context_overfull"].fillna(False), "context_quality"] = (
        "review_overfull_context"
    )

    return out


def _apply_training_flags(
    df: pd.DataFrame,
    *,
    require_full_8_for_eval: bool,
) -> pd.DataFrame:
    out = df.copy()

    bbox_valid = _to_bool_series(out["bbox_valid"])
    behavior_valid = out["behavior"].isin(VALID_BEHAVIOR_SET)

    frame_uid_missing = (
        out["frame_uid"].isna()
        | out["frame_uid"].astype(str).str.strip().eq("")
    )

    pig_id_missing = (
        out["pig_id"].isna()
        | out["pig_id"].astype(str).str.strip().eq("")
    )
    required_missing = frame_uid_missing | pig_id_missing

    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    social_missing = _to_bool_series(out["social_missing_mask"])
    duplicate_pig = _to_bool_series(out["duplicate_pig_id_in_frame"])
    context_overfull = _to_bool_series(out["context_overfull"])
    full_context = _to_bool_series(out["global_context_complete_8"])

    invalid_bbox = ~bbox_valid
    invalid_behavior = ~behavior_valid
    rejected = invalid_bbox | invalid_behavior | required_missing

    review = social_missing | duplicate_pig | context_overfull

    include = ~rejected

    out["include_in_training"] = include
    out["qa_status"] = "ok"
    out["training_tier"] = "clean"
    out["sample_weight"] = 1.0

    non_social = ~is_social
    local_count = pd.to_numeric(out["local_context_pig_count"], errors="coerce").fillna(0)

    out.loc[full_context & include, "training_tier"] = "clean_full_context"
    out.loc[is_social & ~social_missing & include, "training_tier"] = (
        "clean_interaction"
    )
    out.loc[non_social & local_count.le(1) & include, "training_tier"] = "actor_only"
    out.loc[
        non_social & local_count.between(2, 7, inclusive="both") & include,
        "training_tier",
    ] = "partial_context"

    out.loc[review & include, "training_tier"] = "review"
    out.loc[social_missing & include, "qa_status"] = (
        "review_interaction_missing_partner"
    )
    out.loc[(duplicate_pig | context_overfull) & include, "qa_status"] = "review"

    out.loc[invalid_bbox, "qa_status"] = "invalid_bbox"
    out.loc[invalid_behavior, "qa_status"] = "invalid_behavior"
    out.loc[required_missing, "qa_status"] = "missing_required_value"

    out.loc[rejected, "training_tier"] = "rejected"
    out.loc[rejected, "include_in_training"] = False
    out.loc[rejected, "sample_weight"] = 0.0

    # Lower weight only for context uncertainty, not for reviewed Hidden.
    out.loc[social_missing & include, "sample_weight"] = 0.5
    out.loc[(duplicate_pig | context_overfull) & include, "sample_weight"] = 0.5
    out.loc[
        non_social & local_count.le(1) & include & ~review,
        "sample_weight",
    ] = 0.9
    out.loc[
        non_social & local_count.between(2, 7, inclusive="both") & include & ~review,
        "sample_weight",
    ] = 0.95

    out["use_for_visual_training"] = include
    out["use_for_shape_training"] = include & out["behavior"].isin(
        SHAPE_DOMINANT_BEHAVIORS
    )
    out["use_for_motion_training"] = include & out["behavior"].isin(
        MOTION_DOMINANT_BEHAVIORS
    )
    out["use_for_roi_training"] = include & out["behavior"].isin(
        ROI_DOMINANT_BEHAVIORS
    )

    out["use_for_social_training"] = (
        include
        & is_social
        & ~social_missing
        & ~duplicate_pig
        & ~context_overfull
    )

    out["use_for_main_eval"] = (
        include
        & ~social_missing
        & ~duplicate_pig
        & ~context_overfull
    )

    if require_full_8_for_eval:
        out["use_for_main_eval"] = out["use_for_main_eval"] & full_context

    return out


def _interaction_partner_ids(df: pd.DataFrame) -> pd.Series:
    partner_ids: dict[int, str] = {}

    for _, group in df.groupby("frame_uid", dropna=False):
        ids = [str(v) for v in group["pig_id"].dropna().tolist()]

        for idx, pig_id in zip(group.index, group["pig_id"], strict=True):
            partners = sorted(pid for pid in ids if pid != str(pig_id))
            partner_ids[int(idx)] = "|".join(partners)

    return pd.Series(partner_ids)


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