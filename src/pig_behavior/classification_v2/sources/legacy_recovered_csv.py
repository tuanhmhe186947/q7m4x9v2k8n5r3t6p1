"""Parser for legacy recovered burst CSV files.

This module converts legacy burst recovery outputs into the canonical
frame-object schema used by classification_v2.

Supported inputs:
- legacy_frame_object_annotations.csv
- legacy_dense_tracklet_map.csv

Important design rule:
Do not reject rows or frames only because fewer than 8 pigs are present.
Legacy data may intentionally contain actor-only or selected-context
annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    CANONICAL_FRAME_OBJECT_COLUMNS,
    DEFAULT_PIG_IDS,
    SOURCE_TYPE_LEGACY,
    behavior_to_coarse,
    normalize_behavior,
    normalize_hidden,
    normalize_pig_id,
)

DEFAULT_LEGACY_IMAGE_WIDTH = 1280
DEFAULT_LEGACY_IMAGE_HEIGHT = 720


def load_legacy_frame_objects(
    path: str | Path,
    *,
    dataset_id: str = "legacy",
    max_rows: int | None = None,
    default_image_width: int = DEFAULT_LEGACY_IMAGE_WIDTH,
    default_image_height: int = DEFAULT_LEGACY_IMAGE_HEIGHT,
) -> pd.DataFrame:
    """Load a legacy recovered CSV and return canonical frame objects.

    Parameters
    ----------
    path:
        Path to either legacy_frame_object_annotations.csv or
        legacy_dense_tracklet_map.csv.
    dataset_id:
        Dataset identifier stored in the canonical output.
    max_rows:
        Optional row limit for debugging.
    default_image_width:
        Fallback width when the source CSV has no image_width column.
    default_image_height:
        Fallback height when the source CSV has no image_height column.

    Returns
    -------
    pd.DataFrame
        Canonical frame-object dataframe.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Legacy CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path, low_memory=False)
    if max_rows is not None:
        df_raw = df_raw.head(max_rows).copy()

    if df_raw.empty:
        return _empty_canonical_df()

    source_kind = infer_legacy_source_kind(df_raw)

    if source_kind == "frame_object_export":
        canonical = _from_frame_object_export(
            df_raw,
            dataset_id=dataset_id,
            default_image_width=default_image_width,
            default_image_height=default_image_height,
        )
    elif source_kind == "dense_tracklet_map":
        canonical = _from_dense_tracklet_map(
            df_raw,
            dataset_id=dataset_id,
            default_image_width=default_image_width,
            default_image_height=default_image_height,
        )
    else:
        raise ValueError(
            "Unsupported legacy CSV schema. Expected either "
            "legacy_frame_object_annotations.csv or legacy_dense_tracklet_map.csv. "
            f"Columns found: {sorted(df_raw.columns.tolist())}"
        )

    canonical = _normalize_common_fields(canonical)
    canonical = add_legacy_context_counts(canonical)
    canonical = _ensure_canonical_columns(canonical)

    return canonical[CANONICAL_FRAME_OBJECT_COLUMNS]


def infer_legacy_source_kind(df: pd.DataFrame) -> str:
    """Infer supported legacy CSV type from column names."""
    columns = set(df.columns)

    frame_object_export_markers = {
        "image_key",
        "image_name",
        "source_video_key",
        "tracklet_id",
        "pig_id",
        "behavior",
        "frame_index",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    dense_tracklet_map_markers = {
        "tracklet_id",
        "pig_id",
        "behavior",
        "frame_index",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    if frame_object_export_markers.issubset(columns):
        return "frame_object_export"

    if dense_tracklet_map_markers.issubset(columns):
        return "dense_tracklet_map"

    return "unknown"


def add_legacy_context_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add global context counts for legacy recovered data.

    The count is computed per frame_uid. A frame may have 1..8 pigs.
    This function only records the context status; it never rejects rows.
    """
    out = df.copy()

    if out.empty:
        out["global_context_pig_count"] = pd.Series(dtype="Int64")
        out["global_context_complete_8"] = pd.Series(dtype="boolean")
        out["missing_global_pig_ids"] = pd.Series(dtype="object")
        return out

    if "frame_uid" not in out.columns:
        raise ValueError("frame_uid is required before adding context counts.")
    
    has_existing_context = (
        "global_context_pig_count" in out.columns
        and out["global_context_pig_count"].notna().any()
    )

    if has_existing_context:
        out["global_context_pig_count"] = _to_numeric(out["global_context_pig_count"])

        if "global_context_complete_8" not in out.columns:
            out["global_context_complete_8"] = out["global_context_pig_count"].eq(8)
        else:
            out["global_context_complete_8"] = _to_bool_series(
                out["global_context_complete_8"],
                default=False,
            )

        if "missing_global_pig_ids" not in out.columns:
            out["missing_global_pig_ids"] = ""

        if "local_context_pig_count" not in out.columns:
            out["local_context_pig_count"] = out["global_context_pig_count"]

        if "context_quality" not in out.columns:
            out["context_quality"] = out["global_context_complete_8"].map(
                {True: "full_context", False: "partial_or_selected_context"}
            )

        return out

    counts = (
        out.groupby("frame_uid", dropna=False)["pig_id"]
        .nunique(dropna=True)
        .rename("global_context_pig_count")
    )

    pig_sets = (
        out.groupby("frame_uid", dropna=False)["pig_id"]
        .apply(lambda values: set(v for v in values.dropna().astype(str)))
        .rename("present_pig_ids")
    )

    out = out.merge(counts, left_on="frame_uid", right_index=True, how="left")
    out = out.merge(pig_sets, left_on="frame_uid", right_index=True, how="left")

    expected = set(DEFAULT_PIG_IDS)

    out["global_context_complete_8"] = out["global_context_pig_count"].eq(8)
    out["missing_global_pig_ids"] = out["present_pig_ids"].apply(
        lambda present: "|".join(sorted(expected.difference(present)))
        if isinstance(present, set)
        else "|".join(DEFAULT_PIG_IDS)
    )

    out = out.drop(columns=["present_pig_ids"])

    # For step 2, local context equals available context.
    # Later context_policy.py will refine this behavior-specifically.
    out["local_context_pig_count"] = out["global_context_pig_count"]
    out["context_quality"] = out["global_context_complete_8"].map(
        {True: "full_context", False: "partial_or_selected_context"}
    )

    return out


def audit_legacy_frame_objects(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact audit dictionary for logging/reporting."""
    if df.empty:
        return {
            "rows": 0,
            "frames": 0,
            "tracklets": 0,
            "pig_ids": {},
            "behaviors": {},
            "context_pig_count": {},
            "source_types": {},
        }

    audit: dict[str, Any] = {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique(dropna=True))
        if "frame_uid" in df.columns
        else 0,
        "tracklets": int(df["track_id"].nunique(dropna=True))
        if "track_id" in df.columns
        else 0,
        "pig_ids": _value_counts_dict(df, "pig_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "context_pig_count": _value_counts_dict(df, "global_context_pig_count"),
        "source_types": _value_counts_dict(df, "source_type"),
        "annotation_scope": _value_counts_dict(df, "annotation_scope"),
        "training_tier": _value_counts_dict(df, "training_tier"),
        "qa_status": _value_counts_dict(df, "qa_status"),
    }
    return audit


def _from_frame_object_export(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    default_image_width: int,
    default_image_height: int,
) -> pd.DataFrame:
    """Map legacy_frame_object_annotations.csv to canonical columns."""
    out = pd.DataFrame(index=df.index)

    out["source_type"] = _first_existing_series(
    df,
    ["source_type"],
    default=SOURCE_TYPE_LEGACY,
    )

    out["dataset_id"] = _first_existing_series(
        df,
        ["dataset_id"],
        default=dataset_id,
    )

    out["video_key"] = _first_existing_series(
        df,
        ["video_key", "source_video_key", "source_video_resolved", "color_video_path"],
        default="legacy_unknown",
    )

    out["clip_id"] = _first_existing_series(
        df,
        ["clip_id", "group_id"],
        default="",
    )

    out["task_id"] = _first_existing_series(
        df,
        ["task_id"],
        default="",
    )

    out["frame_uid"] = _first_existing_series(
        df,
        ["frame_uid", "image_key"],
        default="",
    )
    missing_frame_uid = out["frame_uid"].isna() | out["frame_uid"].astype(str).eq("")
    if missing_frame_uid.any():
        fallback_uid = _build_frame_uid(
            video_key=out["video_key"],
            group_id=_get_series(df, "group_id", default=""),
            frame_index=_get_series(df, "frame_index", default=0),
        )
        out.loc[missing_frame_uid, "frame_uid"] = fallback_uid.loc[missing_frame_uid]

    out["image_key"] = _first_existing_series(
        df,
        ["image_key", "frame_uid"],
        default=out["frame_uid"],
    )

    out["image_name"] = _get_series(df, "image_name", default="")
    out["frame_index"] = _to_numeric(_get_series(df, "frame_index", default=0))

    out["relative_frame_index"] = _to_numeric(
        _first_existing_series(
            df,
            ["relative_frame_index"],
            default=out["frame_index"],
        )
    )
    out["timestamp_sec"] = _to_numeric(
        _get_series(df, "timestamp_sec", default=pd.NA)
    )

    out["timestamp_source"] = _first_existing_series(
        df,
        ["timestamp_source"],
        default=_timestamp_source(out["timestamp_sec"]),
    )

    out["image_width"] = _to_numeric(
        _get_series(df, "image_width", default=default_image_width)
    ).fillna(default_image_width)
    out["image_height"] = _to_numeric(
        _get_series(df, "image_height", default=default_image_height)
    ).fillna(default_image_height)

    out["pig_id"] = _get_series(df, "pig_id", default="")
    out["track_id"] = _get_series(df, "tracklet_id", default="")
    out["track_label"] = out["pig_id"]

    _copy_bbox_columns(df, out)
    for raw_col, fallback_col in [
        ("x1_raw", "x1"),
        ("y1_raw", "y1"),
        ("x2_raw", "x2"),
        ("y2_raw", "y2"),
    ]:
        out[raw_col] = _to_numeric(
            _first_existing_series(df, [raw_col], default=out[fallback_col])
        )

    out["bbox_valid"] = _to_bool_series(
        _first_existing_series(df, ["bbox_valid"], default=_bbox_valid(out)),
        default=True,
    )

    out["bbox_was_clipped"] = _to_bool_series(
        _first_existing_series(df, ["bbox_was_clipped"], default=False),
        default=False,
    )

    out["actor_bbox_valid"] = out["bbox_valid"]

    out["behavior"] = _get_series(df, "behavior", default="")
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = _get_series(df, "hidden", default="No")
    out["is_actor_label"] = True
    out["label_source"] = "legacy_recovered"
    out["bbox_source"] = _get_series(df, "bbox_source", default="legacy_recovered")

    out["global_context_pig_count"] = _to_numeric(
        _first_existing_series(df, ["global_context_pig_count"], default=pd.NA)
    )

    out["global_context_complete_8"] = _to_bool_series(
        _first_existing_series(df, ["global_context_complete_8"], default=False),
        default=False,
    )

    out["missing_global_pig_ids"] = _first_existing_series(
        df,
        ["missing_global_pig_ids"],
        default="",
    )

    out["duplicate_pig_id_in_frame"] = _to_bool_series(
        _first_existing_series(df, ["duplicate_pig_id_in_frame"], default=False),
        default=False,
    )

    out["context_overfull"] = _to_bool_series(
        _first_existing_series(df, ["context_overfull"], default=False),
        default=False,
    )

    out["local_context_pig_count"] = _to_numeric(
        _first_existing_series(
            df,
            ["local_context_pig_count"],
            default=out["global_context_pig_count"],
        )
    )

    out["annotation_scope"] = _first_existing_series(
        df,
        ["annotation_scope"],
        default="unknown",
    )

    out["local_context_quality"] = _first_existing_series(
        df,
        ["local_context_quality"],
        default="unknown",
    )

    out["interaction_partner_count"] = _to_numeric(
        _first_existing_series(df, ["interaction_partner_count"], default=pd.NA)
    )

    out["interaction_partner_ids"] = _first_existing_series(
        df,
        ["interaction_partner_ids"],
        default="",
    )

    out["context_quality"] = _first_existing_series(
        df,
        ["context_quality"],
        default="unknown",
    )

    out["social_feature_quality"] = _first_existing_series(
        df,
        ["social_feature_quality"],
        default="unknown",
    )

    out["actor_quality"] = out["actor_bbox_valid"].map(
        {True: "valid", False: "invalid_bbox"}
    )

    out["include_in_training"] = _to_bool_series(
        _first_existing_series(df, ["include_in_training"], default=True),
        default=True,
    )

    out["use_for_visual_training"] = _to_bool_series(
        _first_existing_series(
            df,
            ["use_for_visual_training", "include_in_training"],
            default=out["include_in_training"],
        ),
        default=True,
    )

    out["use_for_shape_training"] = _to_bool_series(
        _first_existing_series(
            df,
            ["use_for_shape_training"],
            default=out["use_for_visual_training"],
        ),
        default=True,
    )

    out["use_for_motion_training"] = _to_bool_series(
        _first_existing_series(
            df,
            ["use_for_motion_training"],
            default=out["use_for_visual_training"],
        ),
        default=True,
    )

    out["use_for_roi_training"] = _to_bool_series(
        _first_existing_series(
            df,
            ["use_for_roi_training"],
            default=out["use_for_visual_training"],
        ),
        default=True,
    )

    out["use_for_social_training"] = _to_bool_series(
        _first_existing_series(df, ["use_for_social_training"], default=False),
        default=False,
    )

    out["use_for_main_eval"] = _to_bool_series(
        _first_existing_series(
            df,
            ["use_for_main_eval"],
            default=out["use_for_visual_training"],
        ),
        default=True,
    )

    out["training_tier"] = _get_series(df, "training_tier", default="clean")
    out["qa_status"] = _get_series(df, "qa_status", default="ok")
    out["sample_weight"] = 1.0

    out["crop_path"] = _get_series(df, "crop_path", default="")
    out["source_video_path"] = _first_existing_series(
        df,
        ["source_video_resolved", "color_video_path", "source_video_path"],
        default="",
    )
    out["times_txt_path"] = _get_series(df, "times_txt_path", default="")

    return out


def _from_dense_tracklet_map(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    default_image_width: int,
    default_image_height: int,
) -> pd.DataFrame:
    """Map legacy_dense_tracklet_map.csv to canonical columns."""
    out = pd.DataFrame(index=df.index)

    source_video = _first_existing_series(
        df,
        ["source_video_key", "video_key", "source_video_resolved", "color_video_path"],
        default="legacy_unknown",
    )
    group_id = _get_series(df, "group_id", default="")
    frame_index = _get_series(df, "frame_index", default=0)

    out["source_type"] = SOURCE_TYPE_LEGACY
    out["dataset_id"] = dataset_id
    out["video_key"] = source_video
    out["clip_id"] = group_id
    out["task_id"] = ""

    if "image_key" in df.columns:
        out["frame_uid"] = df["image_key"]
    else:
        out["frame_uid"] = _build_frame_uid(
            video_key=source_video,
            group_id=group_id,
            frame_index=frame_index,
        )

    out["image_key"] = out["frame_uid"]
    out["image_name"] = _get_series(df, "image_name", default="")
    out["frame_index"] = _to_numeric(frame_index)
    out["relative_frame_index"] = out["frame_index"]
    out["timestamp_sec"] = _to_numeric(
        _get_series(df, "timestamp_sec", default=pd.NA)
    )
    out["timestamp_source"] = _timestamp_source(out["timestamp_sec"])

    out["image_width"] = _to_numeric(
        _get_series(df, "image_width", default=default_image_width)
    ).fillna(default_image_width)
    out["image_height"] = _to_numeric(
        _get_series(df, "image_height", default=default_image_height)
    ).fillna(default_image_height)

    out["pig_id"] = _get_series(df, "pig_id", default="")
    out["track_id"] = _get_series(df, "tracklet_id", default="")
    out["track_label"] = out["pig_id"]

    _copy_bbox_columns(df, out)

    out["sequence_frame_count"] = _to_numeric(
        _first_existing_series(df, ["sequence_frame_count"], default=16)
    )

    out["legacy_sequence_mode"] = _first_existing_series(
        df,
        ["legacy_sequence_mode"],
        default="legacy_16f_0_to_15",
    )

    out["legacy_expected_sequence_length"] = _to_numeric(
        _first_existing_series(df, ["legacy_expected_sequence_length"], default=16)
    )

    out["legacy_anchor_relative_frames"] = _first_existing_series(
        df,
        ["legacy_anchor_relative_frames"],
        default="0,3,6,9,12,15",
    )

    out["is_legacy_gt_anchor"] = _to_bool_series(
        _first_existing_series(df, ["is_legacy_gt_anchor"], default=False),
        default=False,
    )

    out["sequence_complete"] = _to_bool_series(
        _first_existing_series(df, ["sequence_complete"], default=True),
        default=True,
    )

    out["sequence_range_valid"] = _to_bool_series(
        _first_existing_series(df, ["sequence_range_valid"], default=True),
        default=True,
    )

    out["behavior"] = _get_series(df, "behavior", default="")
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = _get_series(df, "hidden", default="No")
    out["is_actor_label"] = True
    out["label_source"] = "legacy_recovered"
    out["bbox_source"] = _get_series(df, "bbox_source", default="legacy_recovered")

    out["annotation_scope"] = "unknown"
    out["local_context_quality"] = "unknown"
    out["interaction_partner_count"] = pd.NA
    out["interaction_partner_ids"] = ""

    out["actor_bbox_valid"] = _bbox_valid(out)
    out["actor_quality"] = out["actor_bbox_valid"].map(
        {True: "valid", False: "invalid_bbox"}
    )

    out["use_for_visual_training"] = _get_bool_series(
        df, "include_in_training", default=True
    )
    out["use_for_shape_training"] = out["use_for_visual_training"]
    out["use_for_motion_training"] = out["use_for_visual_training"]
    out["use_for_roi_training"] = out["use_for_visual_training"]
    out["use_for_social_training"] = False
    out["use_for_main_eval"] = False

    out["include_in_training"] = _get_bool_series(
        df, "include_in_training", default=True
    )
    out["training_tier"] = _get_series(df, "training_tier", default="clean")
    out["qa_status"] = _get_series(df, "qa_status", default="ok")
    out["sample_weight"] = 1.0

    out["crop_path"] = _get_series(df, "crop_path", default="")
    out["source_video_path"] = _first_existing_series(
        df,
        ["source_video_resolved", "color_video_path", "source_video_path"],
        default="",
    )
    out["times_txt_path"] = _get_series(df, "times_txt_path", default="")

    return out


def _normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize behavior, hidden, pig_id, and common quality fields."""
    out = df.copy()

    out["pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["behavior"] = out["behavior"].map(normalize_behavior)
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = out["hidden"].map(normalize_hidden)

    invalid_behavior = out["behavior"].isna()
    invalid_bbox = ~out["actor_bbox_valid"].fillna(False)

    out.loc[invalid_behavior, "qa_status"] = "invalid_behavior"
    out.loc[invalid_behavior, "training_tier"] = "rejected"
    out.loc[invalid_behavior, "include_in_training"] = False

    out.loc[invalid_bbox, "qa_status"] = "invalid_bbox"
    out.loc[invalid_bbox, "training_tier"] = "rejected"
    out.loc[invalid_bbox, "include_in_training"] = False

    hidden_yes = out["hidden"].eq("Yes")
    #out.loc[hidden_yes, "qa_status"] = "hidden"
    #out.loc[hidden_yes, "training_tier"] = "review"
    valid_actor_for_hidden_policy = (
        hidden_yes
        & out["behavior"].notna()
        & out["actor_bbox_valid"].fillna(False)
    )

    # If an older export/parser marked these rows as hidden/review only
    # because Hidden=Yes, restore them as usable training rows.
    out.loc[
        valid_actor_for_hidden_policy & out["qa_status"].eq("hidden"),
        "qa_status",
    ] = "ok"

    out.loc[
        valid_actor_for_hidden_policy
        & out["training_tier"].isin(["review", "warning"]),
        "training_tier",
    ] = "clean"

    out.loc[valid_actor_for_hidden_policy, "include_in_training"] = True

    for flag_col in [
        "use_for_visual_training",
        "use_for_shape_training",
        "use_for_motion_training",
        "use_for_roi_training",
        "use_for_main_eval",
    ]:
        if flag_col in out.columns:
            out.loc[valid_actor_for_hidden_policy, flag_col] = True

    if "annotation_scope" not in out.columns:
        out["annotation_scope"] = "selected_actor_group"
    else:
        blank_scope = (
            out["annotation_scope"].isna()
            | out["annotation_scope"].astype(str).str.strip().isin(
                ["", "nan", "None", "unknown"]
            )
        )
        out.loc[blank_scope, "annotation_scope"] = "selected_actor_group"

    return out


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every canonical column exists."""
    out = df.copy()

    for col in CANONICAL_FRAME_OBJECT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # Now that context count exists, refine annotation scope.
    if "global_context_pig_count" in out.columns:
        blank_scope = (
            out["annotation_scope"].isna()
            | out["annotation_scope"].astype(str).str.strip().isin(
                ["", "nan", "None", "unknown"]
            )
        )

        out.loc[
            blank_scope & out["global_context_pig_count"].eq(1),
            "annotation_scope",
        ] = "actor_only"

        out.loc[
            blank_scope
            & out["global_context_pig_count"].between(2, 7, inclusive="both"),
            "annotation_scope",
        ] = "selected_actor_group"

        out.loc[
            blank_scope & out["global_context_pig_count"].eq(8),
            "annotation_scope",
        ] = "full_context"

        blank_social = (
            out["social_feature_quality"].isna()
            | out["social_feature_quality"].astype(str).str.strip().isin(
                ["", "nan", "None", "unknown"]
            )
        )

        non_social = ~out["behavior"].isin(["fight", "social-nose"])

        out.loc[blank_social & non_social, "social_feature_quality"] = "not_required"

        out.loc[
            blank_social & ~non_social & out["global_context_pig_count"].eq(1),
            "social_feature_quality",
        ] = "missing_partner"

        out.loc[
            blank_social
            & ~non_social
            & out["global_context_pig_count"].between(2, 7, inclusive="both"),
            "social_feature_quality",
        ] = "usable_pair_or_group"

        out.loc[
            blank_social & ~non_social & out["global_context_pig_count"].eq(8),
            "social_feature_quality",
        ] = "full_context"

    return out


def _copy_bbox_columns(source: pd.DataFrame, out: pd.DataFrame) -> None:
    """Copy bbox columns and compute basic geometry fallback."""
    for col in ["x1", "y1", "x2", "y2"]:
        out[col] = _to_numeric(_get_series(source, col, default=pd.NA))

    out["bbox_w"] = _to_numeric(
        _get_series(source, "bbox_w", default=out["x2"] - out["x1"])
    )
    out["bbox_h"] = _to_numeric(
        _get_series(source, "bbox_h", default=out["y2"] - out["y1"])
    )
    out["bbox_area"] = _to_numeric(
        _get_series(source, "bbox_area", default=out["bbox_w"] * out["bbox_h"])
    )

    out["cx"] = _to_numeric(
        _get_series(source, "cx", default=(out["x1"] + out["x2"]) / 2.0)
    )
    out["cy"] = _to_numeric(
        _get_series(source, "cy", default=(out["y1"] + out["y2"]) / 2.0)
    )

    image_width = _to_numeric(out["image_width"]).replace(0, pd.NA)
    image_height = _to_numeric(out["image_height"]).replace(0, pd.NA)

    out["cx_n"] = _to_numeric(_get_series(source, "cx_n", default=out["cx"] / image_width))
    out["cy_n"] = _to_numeric(_get_series(source, "cy_n", default=out["cy"] / image_height))
    out["bw_n"] = _to_numeric(
        _get_series(source, "bw_n", default=out["bbox_w"] / image_width)
    )
    out["bh_n"] = _to_numeric(
        _get_series(source, "bh_n", default=out["bbox_h"] / image_height)
    )
    out["area_n"] = _to_numeric(
        _get_series(
            source,
            "area_n",
            default=out["bbox_area"] / (image_width * image_height),
        )
    )

    out["aspect_ratio"] = out["bbox_w"] / out["bbox_h"].replace(0, pd.NA)
    out["box_diag"] = (out["bbox_w"] ** 2 + out["bbox_h"] ** 2) ** 0.5
    out["box_diag_n"] = (
        out["bw_n"].fillna(0) ** 2 + out["bh_n"].fillna(0) ** 2
    ) ** 0.5
    out["box_compactness"] = out["area_n"] / (out["box_diag_n"] ** 2).replace(0, pd.NA)


def _bbox_valid(df: pd.DataFrame) -> pd.Series:
    """Return bbox validity mask."""
    return (
        df["x1"].notna()
        & df["y1"].notna()
        & df["x2"].notna()
        & df["y2"].notna()
        & (df["x2"] > df["x1"])
        & (df["y2"] > df["y1"])
        & (df["x1"] >= 0)
        & (df["y1"] >= 0)
        & (df["x2"] <= df["image_width"])
        & (df["y2"] <= df["image_height"])
    )


def _build_frame_uid(
    *,
    video_key: pd.Series,
    group_id: pd.Series,
    frame_index: pd.Series,
) -> pd.Series:
    """Build a stable frame UID for legacy data."""
    frame_num = _to_numeric(frame_index).fillna(-1).astype(int)
    return (
        video_key.astype(str)
        + "::"
        + group_id.astype(str)
        + "::f"
        + frame_num.astype(str).str.zfill(6)
    )


def _timestamp_source(timestamp_sec: pd.Series) -> pd.Series:
    """Mark timestamp source for parser output."""
    return timestamp_sec.notna().map({True: "input_timestamp", False: "unknown"})

def _get_series(df: pd.DataFrame, column: str, *, default: Any) -> pd.Series:
    """Return a column or a default-valued Series."""
    if column in df.columns:
        return df[column]

    if isinstance(default, pd.Series):
        return default.reindex(df.index)

    return pd.Series([default] * len(df), index=df.index)


def _first_existing_series(
    df: pd.DataFrame,
    columns: list[str],
    *,
    default: Any,
) -> pd.Series:
    """Return the first existing non-empty column among candidates."""
    for column in columns:
        if column in df.columns:
            series = df[column]
            if not series.isna().all():
                return series

    if isinstance(default, pd.Series):
        return default.reindex(df.index)

    return pd.Series([default] * len(df), index=df.index)


def _first_existing_series(
    df: pd.DataFrame,
    columns: list[str],
    *,
    default: Any,
) -> pd.Series:
    """Return the first existing non-empty column among candidates."""
    for column in columns:
        if column in df.columns:
            series = df[column]
            if not series.isna().all():
                return series
    return pd.Series([default] * len(df), index=df.index)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric with invalid values as NA."""
    return pd.to_numeric(series, errors="coerce")

def _to_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    """Convert common CSV truthy/falsy formats to bool."""
    if series.dtype == bool:
        return series.fillna(default).astype(bool)

    truthy = {"true", "1", "yes", "y", "t", "include"}
    falsy = {"false", "0", "no", "n", "f", "exclude"}

    def parse(value: object) -> bool:
        if pd.isna(value):
            return default
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return default

    return series.map(parse).astype(bool)

def _get_bool_series(
    df: pd.DataFrame,
    column: str,
    *,
    default: bool,
) -> pd.Series:
    """Return a boolean Series from common CSV truthy/falsy formats."""
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)

    series = df[column]
    if series.dtype == bool:
        return series.fillna(default)

    truthy = {"true", "1", "yes", "y", "include"}
    falsy = {"false", "0", "no", "n", "exclude"}

    def parse(value: object) -> bool:
        if pd.isna(value):
            return default
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return default

    return series.map(parse).astype(bool)


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return value counts as a JSON-friendly dict."""
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _empty_canonical_df() -> pd.DataFrame:
    """Return an empty canonical dataframe."""
    return pd.DataFrame(columns=CANONICAL_FRAME_OBJECT_COLUMNS)