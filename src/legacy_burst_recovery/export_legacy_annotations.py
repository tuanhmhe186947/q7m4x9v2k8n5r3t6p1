import argparse
import re
from pathlib import Path

#from xml.etree.ElementTree import Element, ElementTree, SubElement
import pandas as pd

BEHAVIOR_CLASSES = [
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
]

BEHAVIOR_TO_COARSE = {
    "lying": "resting",
    "sitting": "resting",
    "eat": "feeding",
    "drink": "feeding",
    "move": "locomotion",
    "stand": "locomotion",
    "explore": "locomotion",
    "playwithtoy": "locomotion",
    "social-nose": "social",
    "fight": "social",
}

INTERACTION_BEHAVIORS = {"fight", "social-nose"}
ROI_DOMINANT_BEHAVIORS = {"eat", "drink", "playwithtoy"}
MOTION_DOMINANT_BEHAVIORS = {"move", "explore", "fight"}
SHAPE_DOMINANT_BEHAVIORS = {"lying", "sitting", "stand"}

DEFAULT_ANCHOR_RELATIVE_FRAMES = [0, 3, 6, 9, 12, 15]


def parse_bool_value(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def parse_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    return s.map(lambda v: parse_bool_value(v, default=default)).astype(bool)


def safe_float(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def normalize_source_video_key(path_value: str) -> str:
    raw = str(path_value).strip()
    lowered = raw.replace("\\", "/").lower()

    if not lowered or lowered in {"nan", "none", "null"}:
        return ""

    # Full legacy path:
    # .../pigs291119/PIGS291119/000302/color.mp4
    m = re.search(r"(pigs\d{6})/pigs\d{6}/(\d{3,6})/color\.mp4", lowered)
    if m:
        return f"{m.group(1)}/{m.group(2).zfill(6)}"

    # Legacy path:
    # .../pigs291119/000302/color.mp4
    m = re.search(r"(pigs\d{6})/(\d{3,6})/color\.mp4", lowered)
    if m:
        return f"{m.group(1)}/{m.group(2).zfill(6)}"

    # Already normalized:
    # pigs291119/000302
    m = re.search(r"(pigs\d{6})/(\d{3,6})(?:$|/)", lowered)
    if m:
        return f"{m.group(1)}/{m.group(2).zfill(6)}"

    # MP4 filename:
    # Pigs291119_000302_30fps.mp4
    m = re.search(r"(pigs\d{6})[_-](\d{3,6})", lowered)
    if m:
        return f"{m.group(1)}/{m.group(2).zfill(6)}"

    return raw


def normalize_video_name(path_value: str) -> str:
    raw = str(path_value).strip().replace("\\", "/")
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    return Path(raw).name


def normalize_pig_id(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    upper = text.upper().replace(" ", "_")

    m = re.fullmatch(r"ID[_-]?(\d+)", upper)
    if m:
        return f"ID_{int(m.group(1))}"

    m = re.fullmatch(r"PIG[_-]?(\d+)", upper)
    if m:
        return f"ID_{int(m.group(1))}"

    m = re.fullmatch(r"(\d+)", upper)
    if m:
        return f"ID_{int(m.group(1))}"

    return text


def normalize_behavior(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""

    text = text.replace("_", "-").replace(" ", "-")
    aliases = {
        "socialnose": "social-nose",
        "social-nosing": "social-nose",
        "social-nose-contact": "social-nose",
        "play-with-toy": "playwithtoy",
        "play-toy": "playwithtoy",
        "standing": "stand",
        "sit": "sitting",
        "lying-down": "lying",
        "lie": "lying",
        "eating": "eat",
        "drinking": "drink",
        "moving": "move",
    }
    return aliases.get(text, text)


def normalize_hidden(value) -> str:
    if pd.isna(value):
        return "No"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "No"

    if text.lower() in {"true", "1", "yes", "y", "hidden"}:
        return "Yes"
    if text.lower() in {"false", "0", "no", "n", "visible"}:
        return "No"

    if text.lower() in {"yes", "no"}:
        return text.capitalize()

    return text


def parse_anchor_frames(value: str) -> list[int]:
    if not value:
        return DEFAULT_ANCHOR_RELATIVE_FRAMES.copy()

    frames = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    return sorted(set(frames))


def choose_source_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "source_video_key",
        "source_video_resolved",
        "color_video_path",
        "source_video_original",
        "video_final",
        "video_key",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def sequence_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "dataset_id",
        "video_key",
        "source_video_key",
        "group_id",
        "sample_id",
        "tracklet_id",
        "pig_id",
    ]
    cols = [c for c in candidates if c in df.columns]

    if cols:
        return cols

    return ["group_id", "pig_id"]


def add_relative_frame_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce").fillna(0).astype(int)

    if "relative_frame_index" in out.columns:
        out["relative_frame_index"] = (
            pd.to_numeric(out["relative_frame_index"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        return out

    group_cols = sequence_group_columns(out)
    out["relative_frame_index"] = (
        out["frame_index"]
        - out.groupby(group_cols, dropna=False)["frame_index"].transform("min")
    ).astype(int)

    return out


def add_sequence_columns(
    df: pd.DataFrame,
    expected_sequence_length: int,
    anchor_relative_frames: list[int],
) -> pd.DataFrame:
    out = df.copy()
    group_cols = sequence_group_columns(out)

    out["sequence_frame_count"] = (
        out.groupby(group_cols, dropna=False)["relative_frame_index"]
        .transform("nunique")
        .astype(int)
    )

    out["legacy_sequence_mode"] = f"legacy_{expected_sequence_length}f_0_to_{expected_sequence_length - 1}"
    out["legacy_expected_sequence_length"] = int(expected_sequence_length)
    out["legacy_anchor_relative_frames"] = ",".join(str(x) for x in anchor_relative_frames)
    out["is_legacy_gt_anchor"] = out["relative_frame_index"].isin(anchor_relative_frames)

    out["sequence_complete"] = out["sequence_frame_count"].eq(expected_sequence_length)
    out["sequence_range_valid"] = out["relative_frame_index"].between(
        0,
        expected_sequence_length - 1,
        inclusive="both",
    )

    return out


def add_context_columns(df: pd.DataFrame, expected_pig_count: int) -> pd.DataFrame:
    out = df.copy()

    frame_col = "frame_uid" if "frame_uid" in out.columns else "image_key"
    expected_pig_ids = [f"ID_{i}" for i in range(1, expected_pig_count + 1)]

    object_count = out.groupby(frame_col, dropna=False)["pig_id"].transform("size")
    unique_count = out.groupby(frame_col, dropna=False)["pig_id"].transform("nunique")

    out["object_count_in_frame"] = object_count.astype(int)
    out["global_context_pig_count"] = unique_count.astype(int)
    out["global_context_complete_8"] = out["global_context_pig_count"].eq(expected_pig_count)
    out["context_overfull"] = (
        out["global_context_pig_count"].gt(expected_pig_count)
        | out["object_count_in_frame"].gt(expected_pig_count)
    )

    duplicate_counts = out.groupby([frame_col, "pig_id"], dropna=False)["pig_id"].transform("size")
    out["duplicate_pig_id_in_frame"] = duplicate_counts.gt(1)

    pig_sets = (
        out.groupby(frame_col, dropna=False)["pig_id"]
        .apply(lambda s: sorted({str(x) for x in s.dropna() if str(x).strip()}))
        .to_dict()
    )

    def missing_pigs(frame_uid: str) -> str:
        present = set(pig_sets.get(frame_uid, []))
        missing = [pig_id for pig_id in expected_pig_ids if pig_id not in present]
        return ",".join(missing)

    out["missing_global_pig_ids"] = out[frame_col].map(missing_pigs).fillna("")

    out["local_context_pig_count"] = out["global_context_pig_count"].astype(int)

    def frame_scope(row) -> str:
        behavior = str(row.get("behavior", ""))
        count = int(row.get("global_context_pig_count", 0))
        complete = bool(row.get("global_context_complete_8", False))

        if complete:
            return "full_context"
        if behavior in INTERACTION_BEHAVIORS and count >= 2:
            return "interaction_pair_or_group"
        if count > 1:
            return "selected_actor_group"
        return "actor_only"

    out["annotation_scope"] = out.apply(frame_scope, axis=1)

    def partner_count(row) -> int:
        behavior = str(row.get("behavior", ""))
        if behavior not in INTERACTION_BEHAVIORS:
            return 0
        return max(int(row.get("local_context_pig_count", 0)) - 1, 0)

    out["interaction_partner_count"] = out.apply(partner_count, axis=1)

    def partner_ids(row) -> str:
        behavior = str(row.get("behavior", ""))
        if behavior not in INTERACTION_BEHAVIORS:
            return ""

        frame_uid = row.get(frame_col, "")
        actor_id = str(row.get("pig_id", ""))
        partners = [pig_id for pig_id in pig_sets.get(frame_uid, []) if pig_id != actor_id]
        return ",".join(partners)

    out["interaction_partner_ids"] = out.apply(partner_ids, axis=1)

    def local_quality(row) -> str:
        behavior = str(row.get("behavior", ""))
        if behavior in INTERACTION_BEHAVIORS:
            if int(row.get("interaction_partner_count", 0)) >= 1:
                return "sufficient_interaction_context"
            return "needs_review_missing_partner"
        return "sufficient_actor_context"

    out["local_context_quality"] = out.apply(local_quality, axis=1)

    def social_quality(row) -> str:
        behavior = str(row.get("behavior", ""))
        if behavior not in INTERACTION_BEHAVIORS:
            return "not_required"
        if int(row.get("interaction_partner_count", 0)) >= 1:
            return "usable_pair_or_group"
        return "missing_partner"

    out["social_feature_quality"] = out.apply(social_quality, axis=1)

    def context_quality(row) -> str:
        if bool(row.get("global_context_complete_8", False)):
            return "full_8_context"
        if int(row.get("global_context_pig_count", 0)) > 1:
            return "partial_context"
        return "actor_only_context"

    out["context_quality"] = out.apply(context_quality, axis=1)

    return out


def add_training_usage_columns(df: pd.DataFrame, require_full_8_for_eval: bool) -> pd.DataFrame:
    out = df.copy()

    if "include_in_training" not in out.columns:
        out["include_in_training"] = True
    else:
        out["include_in_training"] = parse_bool_series(out["include_in_training"], default=True)

    if "training_tier" not in out.columns:
        out["training_tier"] = "legacy_recovered"

    if "qa_status" not in out.columns:
        out["qa_status"] = "ok"

    out.loc[~out["bbox_valid"], "include_in_training"] = False
    out.loc[~out["bbox_valid"], "training_tier"] = "rejected"
    out.loc[~out["bbox_valid"], "qa_status"] = "invalid_bbox"

    out["actor_bbox_valid"] = out["bbox_valid"]
    out["actor_quality"] = out["bbox_valid"].map({True: "valid", False: "invalid"})

    out["use_for_visual_training"] = out["include_in_training"] & out["bbox_valid"]
    out["use_for_shape_training"] = (
        out["use_for_visual_training"] & out["behavior"].isin(SHAPE_DOMINANT_BEHAVIORS)
    )
    out["use_for_motion_training"] = (
        out["use_for_visual_training"]
        & out["behavior"].isin(MOTION_DOMINANT_BEHAVIORS)
        & out["sequence_frame_count"].ge(2)
    )
    out["use_for_roi_training"] = (
        out["use_for_visual_training"] & out["behavior"].isin(ROI_DOMINANT_BEHAVIORS)
    )
    out["use_for_social_training"] = (
        out["use_for_visual_training"]
        & out["behavior"].isin(INTERACTION_BEHAVIORS)
        & out["interaction_partner_count"].ge(1)
    )

    if require_full_8_for_eval:
        out["use_for_main_eval"] = out["use_for_visual_training"] & out["global_context_complete_8"]
    else:
        out["use_for_main_eval"] = out["use_for_visual_training"]

    return out


def add_bbox_columns(df: pd.DataFrame, image_width: int, image_height: int) -> pd.DataFrame:
    out = df.copy()

    for col in ["x1", "y1", "x2", "y2"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["x1_raw"] = out["x1"]
    out["y1_raw"] = out["y1"]
    out["x2_raw"] = out["x2"]
    out["y2_raw"] = out["y2"]

    out["bbox_raw_valid"] = (
        out[["x1_raw", "y1_raw", "x2_raw", "y2_raw"]].notna().all(axis=1)
        & out["x2_raw"].gt(out["x1_raw"])
        & out["y2_raw"].gt(out["y1_raw"])
    )

    out["x1"] = out["x1"].clip(0, image_width - 1)
    out["x2"] = out["x2"].clip(0, image_width - 1)
    out["y1"] = out["y1"].clip(0, image_height - 1)
    out["y2"] = out["y2"].clip(0, image_height - 1)

    out["bbox_w"] = (out["x2"] - out["x1"]).clip(lower=0)
    out["bbox_h"] = (out["y2"] - out["y1"]).clip(lower=0)
    out["bbox_area"] = out["bbox_w"] * out["bbox_h"]

    out["bbox_valid"] = (
        out[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
        & out["bbox_w"].gt(0)
        & out["bbox_h"].gt(0)
    )

    out["bbox_was_clipped"] = (
        out["x1"].ne(out["x1_raw"])
        | out["y1"].ne(out["y1_raw"])
        | out["x2"].ne(out["x2_raw"])
        | out["y2"].ne(out["y2_raw"])
    )

    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0
    out["cx_n"] = out["cx"] / float(image_width)
    out["cy_n"] = out["cy"] / float(image_height)
    out["bw_n"] = out["bbox_w"] / float(image_width)
    out["bh_n"] = out["bbox_h"] / float(image_height)
    out["area_n"] = out["bbox_area"] / float(image_width * image_height)

    return out


def build_frame_object_csv(
    dense_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    training_only: bool,
    dataset_id: str,
    source_type: str,
    expected_sequence_length: int,
    anchor_relative_frames: list[int],
    expected_pig_count: int,
    fps: float | None,
    require_full_8_for_eval: bool,
) -> pd.DataFrame:
    df = dense_df.copy()

    required = ["group_id", "pig_id", "frame_index", "x1", "y1", "x2", "y2", "behavior"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dense CSV: {missing}")

    source_col = choose_source_column(df)
    if source_col is None:
        df["source_video_key"] = "unknown_video"
        df["video_name"] = ""
    else:
        df["source_video_key"] = df[source_col].map(normalize_source_video_key)
        df.loc[df["source_video_key"].eq(""), "source_video_key"] = "unknown_video"
        df["video_name"] = df[source_col].map(normalize_video_name)

    df["source_type"] = source_type
    df["dataset_id"] = dataset_id
    df["video_key"] = df["source_video_key"]

    df["group_id"] = df["group_id"].astype(str)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)
    if "tracklet_id" in df.columns:
        df["tracklet_id"] = df["tracklet_id"].astype(str)
        df["track_id"] = df["tracklet_id"]
    else:
        df["tracklet_id"] = ""
        df["track_id"] = ""

    df["pig_id"] = df["pig_id"].map(normalize_pig_id)
    df["behavior"] = df["behavior"].map(normalize_behavior)
    df["behavior_coarse"] = df["behavior"].map(BEHAVIOR_TO_COARSE).fillna("unknown")

    if "hidden" in df.columns:
        df["hidden"] = df["hidden"].map(normalize_hidden)
    else:
        df["hidden"] = "No"

    df["is_actor_label"] = True
    df["label_source"] = "legacy_dense_behavior"

    if "bbox_source" not in df.columns:
        df["bbox_source"] = "legacy_recovered_bbox"

    if "clip_id" not in df.columns:
        df["clip_id"] = df["group_id"]

    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").fillna(0).astype(int)

    df["image_key"] = (
        df["source_video_key"].astype(str)
        + "::"
        + df["group_id"].astype(str)
        + "::f"
        + df["frame_index"].astype(str).str.zfill(6)
    )
    df["frame_uid"] = df["image_key"]

    df["image_name"] = (
        df["source_video_key"].astype(str).str.replace("/", "_", regex=False)
        + "__"
        + df["group_id"].astype(str)
        + "__f"
        + df["frame_index"].astype(str).str.zfill(6)
        + ".jpg"
    )

    df["image_width"] = int(image_width)
    df["image_height"] = int(image_height)

    if "timestamp_sec" in df.columns:
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
        df["timestamp_source"] = "input_timestamp_sec"
    elif fps is not None and fps > 0:
        df["timestamp_sec"] = df["frame_index"] / float(fps)
        df["timestamp_source"] = "frame_index_div_fps"
    else:
        df["timestamp_sec"] = pd.NA
        df["timestamp_source"] = "absent"

    df = add_relative_frame_index(df)
    df = add_sequence_columns(
        df,
        expected_sequence_length=expected_sequence_length,
        anchor_relative_frames=anchor_relative_frames,
    )
    df = add_bbox_columns(df, image_width=image_width, image_height=image_height)

    sort_cols = [
        c
        for c in [
            "source_video_key",
            "group_id",
            "sample_id",
            "tracklet_id",
            "frame_index",
            "pig_id",
        ]
        if c in df.columns
    ]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df["object_id_in_image"] = df.groupby("frame_uid", dropna=False).cumcount().astype(int)

    df = add_context_columns(df, expected_pig_count=expected_pig_count)
    df = add_training_usage_columns(df, require_full_8_for_eval=require_full_8_for_eval)

    if training_only:
        df = df[df["include_in_training"]].copy()

    preferred_cols = [
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_key",
        "video_name",
        "clip_id",
        "frame_uid",
        "image_key",
        "image_name",
        "object_id_in_image",
        "image_width",
        "image_height",
        "group_id",
        "sample_id",
        "tracklet_id",
        "track_id",
        "pig_id",
        "frame_index",
        "relative_frame_index",
        "timestamp_sec",
        "timestamp_source",
        "sequence_frame_count",
        "legacy_sequence_mode",
        "legacy_expected_sequence_length",
        "legacy_anchor_relative_frames",
        "is_legacy_gt_anchor",
        "sequence_complete",
        "sequence_range_valid",
        "behavior",
        "behavior_coarse",
        "hidden",
        "is_actor_label",
        "label_source",
        "x1_raw",
        "y1_raw",
        "x2_raw",
        "y2_raw",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "cx",
        "cy",
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "bbox_raw_valid",
        "bbox_valid",
        "bbox_was_clipped",
        "object_count_in_frame",
        "global_context_pig_count",
        "global_context_complete_8",
        "missing_global_pig_ids",
        "duplicate_pig_id_in_frame",
        "context_overfull",
        "local_context_pig_count",
        "local_context_quality",
        "annotation_scope",
        "interaction_partner_count",
        "interaction_partner_ids",
        "actor_bbox_valid",
        "actor_quality",
        "context_quality",
        "social_feature_quality",
        "use_for_visual_training",
        "use_for_shape_training",
        "use_for_motion_training",
        "use_for_roi_training",
        "use_for_social_training",
        "use_for_main_eval",
        "bbox_source",
        "tracking_status",
        "qa_status",
        "include_in_training",
        "training_tier",
        "crop_path",
        "source_video_resolved",
        "color_video_path",
        "depth_video_path",
        "times_txt_path",
        "legacy_gt_mode",
        "legacy_gt_support_frames",
        "mask_filter_applied",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in cols]
    return df[cols + extra_cols]

"""
def export_coco(frame_df: pd.DataFrame, out_json: Path) -> None:
    categories = [
        {"id": 1, "name": "pig", "supercategory": "animal"},
    ]

    behavior_to_id = {b: i + 1 for i, b in enumerate(BEHAVIOR_CLASSES)}

    images = []
    annotations = []
    image_id_map = {}
    ann_id = 1

    unique_images = (
        frame_df[
            [
                "image_key",
                "image_name",
                "image_width",
                "image_height",
                "source_video_key",
                "video_key",
                "group_id",
                "frame_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(["source_video_key", "group_id", "frame_index"])
    )

    for image_id, row in enumerate(unique_images.itertuples(index=False), start=1):
        image_id_map[row.image_key] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": row.image_name,
                "width": int(row.image_width),
                "height": int(row.image_height),
                "source_video_key": row.source_video_key,
                "video_key": row.video_key,
                "group_id": row.group_id,
                "frame_index": int(row.frame_index),
            }
        )

    for row in frame_df.itertuples(index=False):
        x1 = safe_float(row.x1)
        y1 = safe_float(row.y1)
        w = safe_float(row.bbox_w)
        h = safe_float(row.bbox_h)

        if w <= 0 or h <= 0:
            continue

        behavior = str(getattr(row, "behavior", ""))
        pig_id = str(getattr(row, "pig_id", ""))

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_map[row.image_key],
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "attributes": {
                    "source_type": str(getattr(row, "source_type", "")),
                    "dataset_id": str(getattr(row, "dataset_id", "")),
                    "video_key": str(getattr(row, "video_key", "")),
                    "frame_uid": str(getattr(row, "frame_uid", "")),
                    "object_id_in_image": safe_int(getattr(row, "object_id_in_image", 0)),
                    "pig_id": pig_id,
                    "behavior": behavior,
                    "behavior_id": behavior_to_id.get(behavior, -1),
                    "behavior_coarse": str(getattr(row, "behavior_coarse", "")),
                    "hidden": str(getattr(row, "hidden", "")),
                    "tracklet_id": str(getattr(row, "tracklet_id", "")),
                    "sample_id": str(getattr(row, "sample_id", "")),
                    "relative_frame_index": safe_int(getattr(row, "relative_frame_index", 0)),
                    "sequence_frame_count": safe_int(getattr(row, "sequence_frame_count", 0)),
                    "legacy_sequence_mode": str(getattr(row, "legacy_sequence_mode", "")),
                    "is_legacy_gt_anchor": str(getattr(row, "is_legacy_gt_anchor", "")),
                    "bbox_source": str(getattr(row, "bbox_source", "")),
                    "bbox_valid": str(getattr(row, "bbox_valid", "")),
                    "bbox_was_clipped": str(getattr(row, "bbox_was_clipped", "")),
                    "global_context_pig_count": safe_int(
                        getattr(row, "global_context_pig_count", 0)
                    ),
                    "annotation_scope": str(getattr(row, "annotation_scope", "")),
                    "tracking_status": str(getattr(row, "tracking_status", "")),
                    "qa_status": str(getattr(row, "qa_status", "")),
                    "include_in_training": str(getattr(row, "include_in_training", "")),
                    "training_tier": str(getattr(row, "training_tier", "")),
                },
            }
        )
        ann_id += 1

    coco = {
        "info": {
            "description": "Legacy recovered pig behavior annotations, canonical 16-frame export",
            "version": "2.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
"""

"""
def export_cvat_1_1(frame_df: pd.DataFrame, out_xml: Path) -> None:
    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "name").text = "legacy_recovered_pig_behavior_16f"

    labels = SubElement(task, "labels")
    label = SubElement(labels, "label")
    SubElement(label, "name").text = "Pig"

    attrs = SubElement(label, "attributes")

    attr_id = SubElement(attrs, "attribute")
    SubElement(attr_id, "name").text = "ID"
    SubElement(attr_id, "mutable").text = "False"
    SubElement(attr_id, "input_type").text = "select"
    SubElement(attr_id, "default_value").text = "ID_1"
    SubElement(attr_id, "values").text = "\n".join([f"ID_{i}" for i in range(1, 9)])

    attr_behavior = SubElement(attrs, "attribute")
    SubElement(attr_behavior, "name").text = "Behavior"
    SubElement(attr_behavior, "mutable").text = "False"
    SubElement(attr_behavior, "input_type").text = "select"
    SubElement(attr_behavior, "default_value").text = BEHAVIOR_CLASSES[0]
    SubElement(attr_behavior, "values").text = "\n".join(BEHAVIOR_CLASSES)

    attr_hidden = SubElement(attrs, "attribute")
    SubElement(attr_hidden, "name").text = "Hidden"
    SubElement(attr_hidden, "mutable").text = "False"
    SubElement(attr_hidden, "input_type").text = "select"
    SubElement(attr_hidden, "default_value").text = "No"
    SubElement(attr_hidden, "values").text = "Yes\nNo"

    work = frame_df.copy()

    work["image_key"] = work["image_key"].astype(str)
    work["image_name"] = work["image_name"].astype(str)
    work["source_video_key"] = work["source_video_key"].astype(str)
    work["group_id"] = work["group_id"].astype(str)
    work["pig_id"] = work["pig_id"].astype(str)

    work["frame_index"] = pd.to_numeric(work["frame_index"], errors="coerce").fillna(0).astype(int)
    work["object_id_in_image"] = (
        pd.to_numeric(work["object_id_in_image"], errors="coerce").fillna(0).astype(int)
    )

    if "bbox_valid" in work.columns:
        bbox_valid_mask = work["bbox_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[bbox_valid_mask].copy()

    image_rows = (
        work[
            [
                "image_key",
                "image_name",
                "image_width",
                "image_height",
                "source_video_key",
                "group_id",
                "frame_index",
            ]
        ]
        .drop_duplicates("image_key")
        .sort_values(["source_video_key", "group_id", "frame_index"], kind="mergesort")
        .reset_index(drop=True)
    )

    work = work.sort_values(
        ["image_key", "pig_id", "object_id_in_image"],
        kind="mergesort",
    )

    boxes_by_image = {
        str(image_key): group
        for image_key, group in work.groupby("image_key", sort=False)
    }

    for image_id, row in image_rows.iterrows():
        image_key = str(row["image_key"])

        image_el = SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": str(row["image_name"]),
                "width": str(int(row["image_width"])),
                "height": str(int(row["image_height"])),
            },
        )

        g = boxes_by_image.get(image_key)
        if g is None or g.empty:
            continue

        for obj in g.itertuples(index=False):
            box = SubElement(
                image_el,
                "box",
                {
                    "label": "Pig",
                    "source": "manual",
                    "occluded": "0",
                    "xtl": f"{safe_float(obj.x1):.2f}",
                    "ytl": f"{safe_float(obj.y1):.2f}",
                    "xbr": f"{safe_float(obj.x2):.2f}",
                    "ybr": f"{safe_float(obj.y2):.2f}",
                    "z_order": "0",
                },
            )

            attr = SubElement(box, "attribute", {"name": "ID"})
            attr.text = str(getattr(obj, "pig_id", ""))

            attr = SubElement(box, "attribute", {"name": "Behavior"})
            attr.text = str(getattr(obj, "behavior", ""))

            attr = SubElement(box, "attribute", {"name": "Hidden"})
            attr.text = normalize_hidden(getattr(obj, "hidden", "No"))

    tree = ElementTree(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
"""

def print_summary(frame_df: pd.DataFrame) -> None:
    print("object rows=", len(frame_df))
    print("frames=", frame_df["frame_uid"].nunique())
    print("videos=", frame_df["video_key"].nunique())

    if "tracklet_id" in frame_df.columns:
        print("tracklets=", frame_df["tracklet_id"].nunique())

    print("\npig boxes per frame:")
    print(
        frame_df.groupby("frame_uid")["pig_id"]
        .nunique()
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nsequence frame count:")
    seq_cols = sequence_group_columns(frame_df)
    print(
        frame_df.groupby(seq_cols, dropna=False)["relative_frame_index"]
        .nunique()
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nrelative frame range:")
    print(
        frame_df.groupby(seq_cols, dropna=False)["relative_frame_index"]
        .agg(["min", "max"])
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nlegacy_sequence_mode:")
    print(frame_df["legacy_sequence_mode"].value_counts(dropna=False).to_string())

    print("\nanchor frames:")
    anchors = frame_df[frame_df["is_legacy_gt_anchor"]]
    if len(anchors):
        print(anchors["relative_frame_index"].value_counts().sort_index().to_string())
    else:
        print("NO ANCHOR ROWS")

    print("\nbehavior distribution by row:")
    print(frame_df["behavior"].value_counts(dropna=False).to_string())

    if "tracklet_id" in frame_df.columns:
        print("\nbehavior distribution by tracklet:")
        print(
            frame_df.drop_duplicates(["video_key", "group_id", "sample_id", "tracklet_id"])[
                "behavior"
            ]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nannotation scope:")
    print(
        frame_df.drop_duplicates("frame_uid")["annotation_scope"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nbbox audit:")
    print("bbox_was_clipped=", int(frame_df["bbox_was_clipped"].sum()))
    print("invalid_bbox=", int((~frame_df["bbox_valid"]).sum()))

    print("\nquality flags:")
    print("sequence_complete false=", int((~frame_df["sequence_complete"]).sum()))
    print("sequence_range_valid false=", int((~frame_df["sequence_range_valid"]).sum()))
    print("duplicate_pig_id_in_frame true=", int(frame_df["duplicate_pig_id_in_frame"].sum()))
    print("context_overfull true=", int(frame_df["context_overfull"].sum()))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dense-csv",
        default=(
            r"C:\Users\ironh\Downloads\PIG_Behavior_Project"
            r"\data\raw\legacy_full_multigt_masked_nodup_16f"
            r"\legacy_dense_tracklet_map.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            r"C:\Users\ironh\Downloads\PIG_Behavior_Project"
            r"\outputs\legacy_full_multigt_masked_nodup_16f"
            r"\exports"
        ),
    )
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)

    parser.add_argument(
        "--dataset-id",
        default="legacy_recovered_16f",
        help="Canonical dataset id written to the output CSV.",
    )
    parser.add_argument(
        "--source-type",
        default="legacy_recovered",
        help="Canonical source type written to the output CSV.",
    )
    parser.add_argument(
        "--expected-sequence-length",
        type=int,
        default=16,
        help="Expected legacy burst length. For current full legacy burst, keep 16.",
    )
    parser.add_argument(
        "--anchor-relative-frames",
        default="0,3,6,9,12,15",
        help="Comma-separated relative frame indices used as legacy GT anchors.",
    )
    parser.add_argument(
        "--expected-pig-count",
        type=int,
        default=8,
        help="Expected full-context pig count in this pen.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS used to compute timestamp_sec if the dense CSV has no timestamp_sec.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help=(
            "Export only rows with include_in_training=True. "
            "Default is to export all rows so context is not removed."
        ),
    )
    parser.add_argument(
        "--require-full-8-for-eval",
        action="store_true",
        help="If set, use_for_main_eval requires global_context_complete_8=True.",
    )

    args = parser.parse_args()

    dense_csv = Path(args.dense_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_relative_frames = parse_anchor_frames(args.anchor_relative_frames)

    print(f"reading dense csv: {dense_csv}", flush=True)
    dense = pd.read_csv(dense_csv, low_memory=False)
    print(f"dense rows: {len(dense)}", flush=True)

    print("building canonical frame-object csv...", flush=True)
    frame_df = build_frame_object_csv(
        dense_df=dense,
        image_width=args.image_width,
        image_height=args.image_height,
        training_only=args.training_only,
        dataset_id=args.dataset_id,
        source_type=args.source_type,
        expected_sequence_length=args.expected_sequence_length,
        anchor_relative_frames=anchor_relative_frames,
        expected_pig_count=args.expected_pig_count,
        fps=args.fps,
        require_full_8_for_eval=args.require_full_8_for_eval,
    )

    out_frame_csv = out_dir / "legacy_frame_object_annotations.csv"

    print(f"writing csv: {out_frame_csv}", flush=True)
    frame_df.to_csv(out_frame_csv, index=False)

    print("saved:", out_frame_csv, flush=True)
    print_summary(frame_df)


if __name__ == "__main__":
    main()