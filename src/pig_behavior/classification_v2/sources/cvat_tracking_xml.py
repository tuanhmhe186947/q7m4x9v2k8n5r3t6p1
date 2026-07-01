"""Parser for CVAT 1.1 tracking XML files.

This module converts CVAT interpolation/tracking XML annotations into the
canonical frame-object schema used by classification_v2.

Supported XML structure:
- <annotations>
- <meta><task>...</task></meta>
- <track id="..." label="Pig_1" source="manual">
- <box frame="..." outside="0" xtl="..." ytl="..." xbr="..." ybr="...">
- <attribute name="ID">ID_1</attribute>
- <attribute name="Behavior">lying</attribute>
- <attribute name="Hidden">No</attribute>

Important design rule:
Do not reject frames only because fewer than 8 pigs are present.
For full tracking XML, we expect 8 pigs per frame, but the parser records
missing/partial context instead of dropping data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from pig_behavior.classification_v2.schema import (
    CANONICAL_FRAME_OBJECT_COLUMNS,
    DEFAULT_PIG_IDS,
    INTERACTION_BEHAVIORS,
    MOTION_DOMINANT_BEHAVIORS,
    ROI_DOMINANT_BEHAVIORS,
    SHAPE_DOMINANT_BEHAVIORS,
    SOURCE_TYPE_CVAT_TRACKING_XML,
    behavior_to_coarse,
    normalize_behavior,
    normalize_hidden,
    normalize_pig_id,
)


def load_cvat_tracking_xml(
    xml_path: str | Path,
    *,
    video_key: str | None = None,
    dataset_id: str | None = None,
    fps: float | None = None,
    expected_pig_count: int = 8,
    require_full_8_for_eval: bool = False,
    max_rows: int | None = None,
    trust_hidden: bool = False,
) -> pd.DataFrame:
    """Load CVAT 1.1 tracking XML and return canonical frame objects."""
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"CVAT tracking XML not found: {path}")

    root = ET.parse(path).getroot()

    task_name = _text(root, "./meta/task/name", default=path.stem)
    task_id = _text(root, "./meta/task/id", default="")
    task_size = _safe_int(_text(root, "./meta/task/size", default="0"), default=0)

    resolved_video_key = video_key or _normalize_video_key_from_task_name(task_name)
    resolved_dataset_id = dataset_id or f"cvat_tracking_{resolved_video_key}"

    image_width = _safe_int(
        _text(root, "./meta/task/original_size/width", default="0"),
        default=0,
    )
    image_height = _safe_int(
        _text(root, "./meta/task/original_size/height", default="0"),
        default=0,
    )

    rows: list[dict[str, Any]] = []

    for track in root.findall("track"):
        track_id = str(track.attrib.get("id", ""))
        track_label = str(track.attrib.get("label", ""))
        track_source = str(track.attrib.get("source", "manual"))
        fallback_pig_id = _pig_id_from_track_label(track_label)

        for box in track.findall("box"):
            if _is_outside(box.attrib.get("outside", "0")):
                continue

            frame_index = _safe_int(box.attrib.get("frame", "0"), default=0)
            attrs = _box_attributes(box)

            pig_id = attrs.get("ID") or attrs.get("id") or fallback_pig_id
            behavior = attrs.get("Behavior") or attrs.get("behavior") or ""
            hidden = attrs.get("Hidden") or attrs.get("hidden") or "No"

            x1_raw = _safe_float(box.attrib.get("xtl"))
            y1_raw = _safe_float(box.attrib.get("ytl"))
            x2_raw = _safe_float(box.attrib.get("xbr"))
            y2_raw = _safe_float(box.attrib.get("ybr"))

            frame_uid = f"{resolved_video_key}::f{frame_index:06d}"

            rows.append(
                {
                    "source_type": SOURCE_TYPE_CVAT_TRACKING_XML,
                    "dataset_id": resolved_dataset_id,
                    "video_key": resolved_video_key,
                    "source_video_key": resolved_video_key,
                    "clip_id": "",
                    "task_id": task_id,
                    "frame_uid": frame_uid,
                    "image_key": frame_uid,
                    "image_name": f"{resolved_video_key}__f{frame_index:06d}.jpg",
                    "object_id_in_image": pd.NA,
                    "frame_index": frame_index,
                    "relative_frame_index": frame_index,
                    "sequence_frame_count": task_size if task_size > 0 else pd.NA,
                    "legacy_sequence_mode": pd.NA,
                    "legacy_expected_sequence_length": pd.NA,
                    "legacy_anchor_relative_frames": pd.NA,
                    "is_legacy_gt_anchor": False,
                    "sequence_complete": pd.NA,
                    "sequence_range_valid": pd.NA,
                    "timestamp_sec": _timestamp_from_frame(frame_index, fps),
                    "timestamp_source": "fps" if fps and fps > 0 else "unknown",
                    "image_width": image_width if image_width > 0 else pd.NA,
                    "image_height": image_height if image_height > 0 else pd.NA,
                    "pig_id": pig_id,
                    "track_id": track_id,
                    "track_label": track_label,
                    "x1_raw": x1_raw,
                    "y1_raw": y1_raw,
                    "x2_raw": x2_raw,
                    "y2_raw": y2_raw,
                    "x1": x1_raw,
                    "y1": y1_raw,
                    "x2": x2_raw,
                    "y2": y2_raw,
                    "behavior": behavior,
                    "behavior_coarse": None,
                    "hidden": hidden,
                    "is_actor_label": True,
                    "label_source": "cvat_tracking_xml",
                    "bbox_source": track_source,
                    "crop_path": "",
                    "source_video_path": "",
                    "times_txt_path": "",
                }
            )

    out = pd.DataFrame(rows)

    if max_rows is not None:
        out = out.head(max_rows).copy()

    if out.empty:
        return _empty_canonical_df()

    out = _normalize_and_add_geometry(out)
    out = _add_context_columns(out, expected_pig_count=expected_pig_count)
    out = _add_training_policy(
        out,
        require_full_8_for_eval=require_full_8_for_eval,
        trust_hidden=trust_hidden,
    )
    out = _ensure_canonical_columns(out)

    return out[CANONICAL_FRAME_OBJECT_COLUMNS]


def audit_cvat_tracking_xml(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact audit information for CVAT tracking XML dataframe."""
    if df.empty:
        return {
            "rows": 0,
            "frames": 0,
            "tracks": 0,
            "pig_ids": {},
            "behaviors": {},
            "context_pig_count": {},
            "annotation_scope": {},
            "qa_status": {},
        }

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique(dropna=True)),
        "tracks": int(df["track_id"].nunique(dropna=True)),
        "pig_ids": _value_counts_dict(df, "pig_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "context_pig_count": _value_counts_dict(df, "global_context_pig_count"),
        "annotation_scope": _value_counts_dict(df, "annotation_scope"),
        "social_feature_quality": _value_counts_dict(df, "social_feature_quality"),
        "training_tier": _value_counts_dict(df, "training_tier"),
        "qa_status": _value_counts_dict(df, "qa_status"),
        "bbox_valid": _value_counts_dict(df, "bbox_valid"),
    }


def _normalize_and_add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["behavior"] = out["behavior"].map(normalize_behavior)
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = out["hidden"].map(normalize_hidden)

    for col in ["x1_raw", "y1_raw", "x2_raw", "y2_raw"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    image_width = pd.to_numeric(out["image_width"], errors="coerce")
    image_height = pd.to_numeric(out["image_height"], errors="coerce")

    out["x1"] = out["x1_raw"]
    out["y1"] = out["y1_raw"]
    out["x2"] = out["x2_raw"]
    out["y2"] = out["y2_raw"]

    has_width = image_width.notna() & image_width.gt(0)
    has_height = image_height.notna() & image_height.gt(0)

    out.loc[has_width, "x1"] = out.loc[has_width, "x1"].clip(
        lower=0,
        upper=image_width.loc[has_width],
    )
    out.loc[has_width, "x2"] = out.loc[has_width, "x2"].clip(
        lower=0,
        upper=image_width.loc[has_width],
    )
    out.loc[has_height, "y1"] = out.loc[has_height, "y1"].clip(
        lower=0,
        upper=image_height.loc[has_height],
    )
    out.loc[has_height, "y2"] = out.loc[has_height, "y2"].clip(
        lower=0,
        upper=image_height.loc[has_height],
    )

    out["bbox_was_clipped"] = (
        out["x1"].ne(out["x1_raw"])
        | out["y1"].ne(out["y1_raw"])
        | out["x2"].ne(out["x2_raw"])
        | out["y2"].ne(out["y2_raw"])
    )

    out["bbox_w"] = out["x2"] - out["x1"]
    out["bbox_h"] = out["y2"] - out["y1"]
    out["bbox_area"] = out["bbox_w"] * out["bbox_h"]
    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0

    width_safe = image_width.replace(0, pd.NA)
    height_safe = image_height.replace(0, pd.NA)

    out["cx_n"] = out["cx"] / width_safe
    out["cy_n"] = out["cy"] / height_safe
    out["bw_n"] = out["bbox_w"] / width_safe
    out["bh_n"] = out["bbox_h"] / height_safe
    out["area_n"] = out["bbox_area"] / (width_safe * height_safe)

    out["aspect_ratio"] = out["bbox_w"] / out["bbox_h"].replace(0, pd.NA)
    out["box_diag"] = (out["bbox_w"] ** 2 + out["bbox_h"] ** 2) ** 0.5
    out["box_diag_n"] = (
        out["bw_n"].fillna(0) ** 2 + out["bh_n"].fillna(0) ** 2
    ) ** 0.5
    out["box_compactness"] = out["area_n"] / (out["box_diag_n"] ** 2).replace(0, pd.NA)

    out["bbox_valid"] = (
        out["x1"].notna()
        & out["y1"].notna()
        & out["x2"].notna()
        & out["y2"].notna()
        & out["bbox_w"].gt(0)
        & out["bbox_h"].gt(0)
        & out["x1"].ge(0)
        & out["y1"].ge(0)
    )

    out["actor_bbox_valid"] = out["bbox_valid"]
    out["actor_quality"] = out["actor_bbox_valid"].map(
        {True: "valid", False: "invalid_bbox"}
    )

    return out


def _add_context_columns(
    df: pd.DataFrame,
    *,
    expected_pig_count: int,
) -> pd.DataFrame:
    out = df.copy()

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

    duplicates = (
        out.groupby(["frame_uid", "pig_id"], dropna=False)
        .size()
        .gt(1)
        .groupby("frame_uid")
        .any()
        .rename("duplicate_pig_id_in_frame")
    )

    out = out.merge(counts, left_on="frame_uid", right_index=True, how="left")
    out = out.merge(pig_sets, left_on="frame_uid", right_index=True, how="left")
    out = out.merge(duplicates, left_on="frame_uid", right_index=True, how="left")

    expected_ids = set(DEFAULT_PIG_IDS[:expected_pig_count])

    out["global_context_complete_8"] = out["global_context_pig_count"].eq(
        expected_pig_count
    )
    out["context_overfull"] = out["global_context_pig_count"].gt(expected_pig_count)
    out["duplicate_pig_id_in_frame"] = out["duplicate_pig_id_in_frame"].fillna(False)

    out["missing_global_pig_ids"] = out["present_pig_ids"].apply(
        lambda present: "|".join(sorted(expected_ids.difference(present)))
        if isinstance(present, set)
        else "|".join(sorted(expected_ids))
    )

    out = out.drop(columns=["present_pig_ids"])

    out["local_context_pig_count"] = out["global_context_pig_count"]

    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    has_partner = out["local_context_pig_count"].ge(2)

    out["annotation_scope"] = "selected_actor_group"
    out.loc[out["global_context_pig_count"].eq(1), "annotation_scope"] = "actor_only"
    out.loc[
        out["global_context_pig_count"].between(2, expected_pig_count - 1, inclusive="both"),
        "annotation_scope",
    ] = "selected_actor_group"
    out.loc[is_social & has_partner, "annotation_scope"] = "interaction_pair_or_group"
    out.loc[out["global_context_complete_8"], "annotation_scope"] = "full_context"
    out.loc[out["context_overfull"], "annotation_scope"] = "overfull_context"

    out["interaction_partner_count"] = 0
    out.loc[is_social, "interaction_partner_count"] = (
        out.loc[is_social, "local_context_pig_count"] - 1
    ).clip(lower=0)

    out["interaction_partner_ids"] = _interaction_partner_ids(out)

    out["context_quality"] = "partial_or_selected_context"
    out.loc[out["global_context_complete_8"], "context_quality"] = "full_context"
    out.loc[out["context_overfull"], "context_quality"] = "overfull_context"

    out["local_context_quality"] = "sufficient_actor_context"
    out.loc[is_social & has_partner, "local_context_quality"] = (
        "sufficient_interaction_context"
    )
    out.loc[is_social & ~has_partner, "local_context_quality"] = (
        "needs_review_missing_partner"
    )
    out.loc[out["global_context_complete_8"], "local_context_quality"] = "full_context"

    out["social_feature_quality"] = "not_required"
    out.loc[is_social & ~has_partner, "social_feature_quality"] = "missing_partner"
    out.loc[is_social & has_partner, "social_feature_quality"] = "usable_pair_or_group"
    out.loc[is_social & out["global_context_complete_8"], "social_feature_quality"] = (
        "full_context"
    )

    return out


def _add_training_policy(
    df: pd.DataFrame,
    *,
    require_full_8_for_eval: bool,
    trust_hidden: bool,
) -> pd.DataFrame:
    out = df.copy()

    invalid_behavior = out["behavior"].isna()
    invalid_bbox = ~out["bbox_valid"].fillna(False)
    hidden_yes = out["hidden"].eq("Yes")
    trusted_hidden_yes = hidden_yes if trust_hidden else pd.Series(
        False,
        index=out.index,
    )
    duplicate_pig = out["duplicate_pig_id_in_frame"].fillna(False)

    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    social_missing_partner = is_social & out["local_context_pig_count"].lt(2)

    out["include_in_training"] = True
    out["training_tier"] = "clean"
    out["qa_status"] = "ok"
    out["sample_weight"] = 1.0

    out.loc[out["global_context_complete_8"], "training_tier"] = "clean_full_context"
    out.loc[is_social & out["local_context_pig_count"].ge(2), "training_tier"] = (
        "clean_interaction"
    )
    out.loc[
        ~is_social & out["local_context_pig_count"].eq(1),
        "training_tier",
    ] = "actor_only"
    out.loc[
        ~out["global_context_complete_8"] & out["local_context_pig_count"].ge(2),
        "training_tier",
    ] = "partial_context"

    out.loc[trusted_hidden_yes, "training_tier"] = "review"
    out.loc[trusted_hidden_yes, "qa_status"] = "hidden"
    out.loc[trusted_hidden_yes, "sample_weight"] = 0.5


    out.loc[social_missing_partner, "training_tier"] = "review"
    out.loc[social_missing_partner, "qa_status"] = "review_interaction_missing_partner"
    out.loc[social_missing_partner, "sample_weight"] = 0.5

    out.loc[duplicate_pig, "training_tier"] = "review"
    out.loc[duplicate_pig, "qa_status"] = "review"

    out.loc[invalid_behavior, "include_in_training"] = False
    out.loc[invalid_behavior, "training_tier"] = "rejected"
    out.loc[invalid_behavior, "qa_status"] = "invalid_behavior"
    out.loc[invalid_behavior, "sample_weight"] = 0.0

    out.loc[invalid_bbox, "include_in_training"] = False
    out.loc[invalid_bbox, "training_tier"] = "rejected"
    out.loc[invalid_bbox, "qa_status"] = "invalid_bbox"
    out.loc[invalid_bbox, "sample_weight"] = 0.0

    include = out["include_in_training"].fillna(False)

    out["use_for_visual_training"] = include
    out["use_for_shape_training"] = include & out["behavior"].isin(SHAPE_DOMINANT_BEHAVIORS)
    out["use_for_motion_training"] = include & out["behavior"].isin(
        MOTION_DOMINANT_BEHAVIORS
    )
    out["use_for_roi_training"] = include & out["behavior"].isin(ROI_DOMINANT_BEHAVIORS)
    out["use_for_social_training"] = (
        include & is_social & out["local_context_pig_count"].ge(2)
    )

    out["use_for_main_eval"] = (
        include
        & ~trusted_hidden_yes
        & ~invalid_behavior
        & ~invalid_bbox
    )
    if require_full_8_for_eval:
        out["use_for_main_eval"] = out["use_for_main_eval"] & out[
            "global_context_complete_8"
        ]

    return out


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in CANONICAL_FRAME_OBJECT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.sort_values(
        ["video_key", "frame_index", "pig_id", "track_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    out["object_id_in_image"] = (
        out.groupby("frame_uid", dropna=False).cumcount() + 1
    )

    return out


def _interaction_partner_ids(df: pd.DataFrame) -> pd.Series:
    partner_ids: dict[int, str] = {}

    for _, group in df.groupby("frame_uid", dropna=False):
        ids = [str(v) for v in group["pig_id"].dropna().tolist()]

        for idx, pig_id in zip(group.index, group["pig_id"], strict=False):
            partners = sorted(pid for pid in ids if pid != str(pig_id))
            partner_ids[idx] = "|".join(partners)

    return pd.Series(partner_ids)


def _box_attributes(box: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for attr in box.findall("attribute"):
        name = str(attr.attrib.get("name", "")).strip()
        value = "" if attr.text is None else str(attr.text).strip()
        if name:
            attrs[name] = value
    return attrs


def _pig_id_from_track_label(label: str) -> str:
    match = re.search(r"(\d+)$", label.strip())
    if match:
        return f"ID_{match.group(1)}"
    return label


def _normalize_video_key_from_task_name(task_name: str) -> str:
    text = str(task_name).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return text or "cvat_tracking_unknown"


def _timestamp_from_frame(frame_index: int, fps: float | None) -> float | pd.NA:
    if fps is None or fps <= 0:
        return pd.NA
    return frame_index / fps


def _text(root: ET.Element, path: str, *, default: str) -> str:
    value = root.findtext(path)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _is_outside(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_int(value: object, *, default: Any = 0) -> Any:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object) -> float | pd.NA:
    try:
        if pd.isna(value):
            return pd.NA
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _empty_canonical_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_FRAME_OBJECT_COLUMNS)