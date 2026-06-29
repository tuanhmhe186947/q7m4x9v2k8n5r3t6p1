"""Parser for CVAT selected native annotations.

This parser converts selected CVAT native annotations, usually from an
annotations.json file, into the canonical frame-object schema used by
classification_v2.

Important design rule:
Selected CVAT annotations are not full 8-pig tracking annotations.
Do not reject rows or frames only because fewer than 8 pigs are present.
Actor-only annotations are valid for non-interaction behaviors.
Interaction behaviors such as fight/social-nose need at least one partner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    CANONICAL_FRAME_OBJECT_COLUMNS,
    DEFAULT_PIG_IDS,
    INTERACTION_BEHAVIORS,
    MOTION_DOMINANT_BEHAVIORS,
    ROI_DOMINANT_BEHAVIORS,
    SHAPE_DOMINANT_BEHAVIORS,
    SOURCE_TYPE_CVAT_SELECTED_NATIVE,
    behavior_to_coarse,
    normalize_behavior,
    normalize_hidden,
    normalize_pig_id,
)


def load_cvat_selected_native(
    task_dir: str | Path | None = None,
    annotations_json: str | Path | None = None,
    *,
    video_key: str | None = None,
    dataset_id: str = "cvat_selected",
    image_width: int | None = None,
    image_height: int | None = None,
    fps: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load selected CVAT native annotations and return canonical frame objects.

    Parameters
    ----------
    task_dir:
        Optional CVAT task folder. If provided, this parser will look for
        task.json, annotations.json, and data/manifest.jsonl.
    annotations_json:
        Optional explicit path to annotations.json.
    video_key:
        Stable video/task key. If omitted, it is inferred from task metadata
        or file/folder name.
    dataset_id:
        Dataset identifier written to the canonical dataframe.
    image_width:
        Optional fallback image width.
    image_height:
        Optional fallback image height.
    fps:
        Optional FPS used to compute timestamp_sec.
    max_rows:
        Optional row limit for debugging.

    Returns
    -------
    pd.DataFrame
        Canonical frame-object dataframe.
    """
    task_path = Path(task_dir) if task_dir is not None else None

    if annotations_json is None:
        if task_path is None:
            raise ValueError("Either task_dir or annotations_json must be provided.")
        annotations_path = task_path / "annotations.json"
    else:
        annotations_path = Path(annotations_json)

    if not annotations_path.exists():
        raise FileNotFoundError(f"CVAT annotations.json not found: {annotations_path}")

    task_meta = _read_optional_json(task_path / "task.json") if task_path else {}
    annotations = _read_json(annotations_path)
    manifest = _read_manifest(task_path / "data" / "manifest.jsonl") if task_path else {}

    label_id_to_name, attr_id_to_name = _build_label_and_attribute_maps(
        annotations=annotations,
        task_meta=task_meta,
    )

    resolved_video_key = _resolve_video_key(
        video_key=video_key,
        task_meta=task_meta,
        annotations_path=annotations_path,
        task_path=task_path,
    )

    resolved_width, resolved_height = _resolve_image_size(
        task_meta=task_meta,
        image_width=image_width,
        image_height=image_height,
    )

    shapes = _extract_shapes(annotations)
    rows: list[dict[str, Any]] = []

    for shape in shapes:
        if _shape_is_outside(shape):
            continue

        shape_type = str(shape.get("type", "rectangle")).lower()
        if shape_type not in {"rectangle", "box"}:
            continue

        points = shape.get("points")
        bbox = _points_to_bbox(points)
        if bbox is None:
            continue

        frame_index = _safe_int(
            _first_existing_value(shape, ["frame", "frame_index", "image_id"], 0),
            default=0,
        )

        label_name = _shape_label_name(shape, label_id_to_name)
        attrs = _shape_attributes(shape, attr_id_to_name)

        pig_id = _first_existing_attr(attrs, ["ID", "Pig ID", "pig_id", "pig", "id"], "")
        behavior = _first_existing_attr(
            attrs,
            ["Behavior", "behavior", "Action", "action"],
            "",
        )
        hidden = _first_existing_attr(attrs, ["Hidden", "hidden"], "No")

        if not behavior:
            behavior = _behavior_from_label(label_name)

        x1, y1, x2, y2 = bbox

        image_name = _image_name_for_frame(
            frame_index=frame_index,
            manifest=manifest,
            video_key=resolved_video_key,
        )
        frame_uid = f"{resolved_video_key}::f{frame_index:06d}"

        rows.append(
            {
                "source_type": SOURCE_TYPE_CVAT_SELECTED_NATIVE,
                "dataset_id": dataset_id,
                "video_key": resolved_video_key,
                "source_video_key": resolved_video_key,
                "clip_id": str(_first_existing_value(shape, ["clip_id", "group_id"], "")),
                "task_id": str(task_meta.get("id", task_meta.get("name", ""))),
                "frame_uid": frame_uid,
                "image_key": frame_uid,
                "image_name": image_name,
                "object_id_in_image": pd.NA,
                "frame_index": frame_index,
                "relative_frame_index": frame_index,
                "sequence_frame_count": pd.NA,
                "legacy_sequence_mode": pd.NA,
                "legacy_expected_sequence_length": pd.NA,
                "legacy_anchor_relative_frames": pd.NA,
                "is_legacy_gt_anchor": False,
                "sequence_complete": pd.NA,
                "sequence_range_valid": pd.NA,
                "timestamp_sec": _timestamp_from_frame(frame_index, fps),
                "timestamp_source": "fps" if fps else "unknown",
                "image_width": resolved_width,
                "image_height": resolved_height,
                "pig_id": pig_id,
                "track_id": str(_first_existing_value(shape, ["track_id", "id"], "")),
                "track_label": pig_id,
                "x1_raw": x1,
                "y1_raw": y1,
                "x2_raw": x2,
                "y2_raw": y2,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "behavior": behavior,
                "behavior_coarse": None,
                "hidden": hidden,
                "is_actor_label": True,
                "label_source": "cvat_selected",
                "bbox_source": str(shape.get("source", "manual")),
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
    out = _add_selected_context(out)
    out = _add_training_policy(out)
    out = _ensure_canonical_columns(out)

    return out[CANONICAL_FRAME_OBJECT_COLUMNS]


def audit_cvat_selected_native(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact audit information for selected CVAT annotations."""
    if df.empty:
        return {
            "rows": 0,
            "frames": 0,
            "pig_ids": {},
            "behaviors": {},
            "context_pig_count": {},
            "annotation_scope": {},
            "qa_status": {},
        }

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique(dropna=True)),
        "pig_ids": _value_counts_dict(df, "pig_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "context_pig_count": _value_counts_dict(df, "global_context_pig_count"),
        "annotation_scope": _value_counts_dict(df, "annotation_scope"),
        "social_feature_quality": _value_counts_dict(df, "social_feature_quality"),
        "training_tier": _value_counts_dict(df, "training_tier"),
        "qa_status": _value_counts_dict(df, "qa_status"),
    }


def _normalize_and_add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["behavior"] = out["behavior"].map(normalize_behavior)
    out["behavior_coarse"] = out["behavior"].map(behavior_to_coarse)
    out["hidden"] = out["hidden"].map(normalize_hidden)

    for col in ["x1_raw", "y1_raw", "x2_raw", "y2_raw", "x1", "y1", "x2", "y2"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["bbox_w"] = out["x2"] - out["x1"]
    out["bbox_h"] = out["y2"] - out["y1"]
    out["bbox_area"] = out["bbox_w"] * out["bbox_h"]
    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0

    out["image_width"] = pd.to_numeric(out["image_width"], errors="coerce")
    out["image_height"] = pd.to_numeric(out["image_height"], errors="coerce")

    width = out["image_width"].replace(0, pd.NA)
    height = out["image_height"].replace(0, pd.NA)

    out["cx_n"] = out["cx"] / width
    out["cy_n"] = out["cy"] / height
    out["bw_n"] = out["bbox_w"] / width
    out["bh_n"] = out["bbox_h"] / height
    out["area_n"] = out["bbox_area"] / (width * height)

    out["aspect_ratio"] = out["bbox_w"] / out["bbox_h"].replace(0, pd.NA)
    out["box_diag"] = (out["bbox_w"] ** 2 + out["bbox_h"] ** 2) ** 0.5
    out["box_diag_n"] = (
        out["bw_n"].fillna(0) ** 2 + out["bh_n"].fillna(0) ** 2
    ) ** 0.5
    out["box_compactness"] = out["area_n"] / (out["box_diag_n"] ** 2).replace(0, pd.NA)

    base_valid = (
        out["x1"].notna()
        & out["y1"].notna()
        & out["x2"].notna()
        & out["y2"].notna()
        & (out["x2"] > out["x1"])
        & (out["y2"] > out["y1"])
        & (out["x1"] >= 0)
        & (out["y1"] >= 0)
    )

    has_size = out["image_width"].notna() & out["image_height"].notna()
    inside_size = (out["x2"] <= out["image_width"]) & (out["y2"] <= out["image_height"])

    out["bbox_valid"] = base_valid & (~has_size | inside_size)
    out["bbox_was_clipped"] = False
    out["actor_bbox_valid"] = out["bbox_valid"]
    out["actor_quality"] = out["actor_bbox_valid"].map(
        {True: "valid", False: "invalid_bbox"}
    )

    return out


def _add_selected_context(df: pd.DataFrame) -> pd.DataFrame:
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

    expected = set(DEFAULT_PIG_IDS)
    out["global_context_complete_8"] = out["global_context_pig_count"].eq(8)
    out["context_overfull"] = out["global_context_pig_count"].gt(8)
    out["missing_global_pig_ids"] = out["present_pig_ids"].apply(
        lambda present: "|".join(sorted(expected.difference(present)))
        if isinstance(present, set)
        else "|".join(DEFAULT_PIG_IDS)
    )

    out = out.drop(columns=["present_pig_ids"])

    out["local_context_pig_count"] = out["global_context_pig_count"]

    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    has_partner = out["local_context_pig_count"].ge(2)

    out["annotation_scope"] = "selected_actor_group"
    out.loc[out["global_context_pig_count"].eq(1), "annotation_scope"] = "actor_only"
    out.loc[
        out["global_context_pig_count"].between(2, 7, inclusive="both"),
        "annotation_scope",
    ] = "selected_actor_group"
    out.loc[is_social & has_partner, "annotation_scope"] = "interaction_pair_or_group"
    out.loc[out["global_context_complete_8"], "annotation_scope"] = "full_context"
    out.loc[out["context_overfull"], "annotation_scope"] = "overfull_context"

    out["interaction_partner_count"] = 0
    out.loc[is_social, "interaction_partner_count"] = (
        out.loc[is_social, "local_context_pig_count"] - 1
    ).clip(lower=0)

    partner_map = _interaction_partner_ids(out)
    out["interaction_partner_ids"] = out.index.map(partner_map).fillna("")

    out["local_context_quality"] = "sufficient_actor_context"
    out.loc[is_social & has_partner, "local_context_quality"] = (
        "sufficient_interaction_context"
    )
    out.loc[is_social & ~has_partner, "local_context_quality"] = (
        "needs_review_missing_partner"
    )
    out.loc[out["global_context_complete_8"], "context_quality"] = "full_context"
    out.loc[~out["global_context_complete_8"], "context_quality"] = (
        "partial_or_selected_context"
    )

    out["social_feature_quality"] = "not_required"
    out.loc[is_social & has_partner, "social_feature_quality"] = "usable_pair_or_group"
    out.loc[is_social & ~has_partner, "social_feature_quality"] = "missing_partner"
    out.loc[is_social & out["global_context_complete_8"], "social_feature_quality"] = (
        "full_context"
    )

    return out


def _add_training_policy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    invalid_behavior = out["behavior"].isna()
    invalid_bbox = ~out["bbox_valid"].fillna(False)
    hidden_yes = out["hidden"].eq("Yes")
    is_social = out["behavior"].isin(INTERACTION_BEHAVIORS)
    social_missing_partner = is_social & out["local_context_pig_count"].lt(2)

    out["include_in_training"] = True
    out["training_tier"] = "clean"
    out["qa_status"] = "ok"
    out["sample_weight"] = 1.0

    out.loc[hidden_yes, "training_tier"] = "review"
    out.loc[hidden_yes, "qa_status"] = "hidden"
    out.loc[hidden_yes, "sample_weight"] = 0.5

    out.loc[social_missing_partner, "training_tier"] = "review"
    out.loc[social_missing_partner, "qa_status"] = "review_interaction_missing_partner"
    out.loc[social_missing_partner, "sample_weight"] = 0.5

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
    out["use_for_main_eval"] = include & ~social_missing_partner & ~hidden_yes

    return out


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in CANONICAL_FRAME_OBJECT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.sort_values(
        ["video_key", "frame_index", "pig_id", "x1", "y1"],
        kind="mergesort",
    ).reset_index(drop=True)

    out["object_id_in_image"] = (
        out.groupby("frame_uid", dropna=False).cumcount() + 1
    )

    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_manifest(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            frame = _safe_int(
                _first_existing_value(obj, ["frame", "frame_index", "number"], idx),
                default=idx,
            )
            name = _first_existing_value(
                obj,
                ["name", "file_name", "filename", "path"],
                "",
            )
            if name:
                mapping[frame] = str(name)
    return mapping


def _extract_shapes(annotations: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(annotations.get("shapes"), list):
        return annotations["shapes"]

    if isinstance(annotations.get("annotations"), list):
        return annotations["annotations"]

    if isinstance(annotations.get("items"), list):
        shapes: list[dict[str, Any]] = []
        for item in annotations["items"]:
            item_frame = _safe_int(
                _first_existing_value(item, ["frame", "frame_index", "id"], 0),
                default=0,
            )
            item_shapes = item.get("shapes", item.get("annotations", []))
            if isinstance(item_shapes, list):
                for shape in item_shapes:
                    if "frame" not in shape:
                        shape = {**shape, "frame": item_frame}
                    shapes.append(shape)
        return shapes

    return []


def _build_label_and_attribute_maps(
    *,
    annotations: dict[str, Any],
    task_meta: dict[str, Any],
) -> tuple[dict[int, str], dict[int, str]]:
    label_id_to_name: dict[int, str] = {}
    attr_id_to_name: dict[int, str] = {}

    for source in [task_meta, annotations]:
        labels = source.get("labels", [])
        if isinstance(labels, dict):
            labels = list(labels.values())

        if not isinstance(labels, list):
            continue

        for label in labels:
            if not isinstance(label, dict):
                continue

            label_id = label.get("id", label.get("pk"))
            label_name = label.get("name")
            if label_id is not None and label_name:
                label_id_to_name[_safe_int(label_id, default=-1)] = str(label_name)

            attrs = label.get("attributes", [])
            if isinstance(attrs, dict):
                attrs = list(attrs.values())

            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                attr_id = attr.get("id", attr.get("spec_id", attr.get("pk")))
                attr_name = attr.get("name")
                if attr_id is not None and attr_name:
                    attr_id_to_name[_safe_int(attr_id, default=-1)] = str(attr_name)

    return label_id_to_name, attr_id_to_name


def _shape_attributes(
    shape: dict[str, Any],
    attr_id_to_name: dict[int, str],
) -> dict[str, str]:
    result: dict[str, str] = {}

    attrs = shape.get("attributes", {})
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            result[str(key)] = str(value)
        return result

    if not isinstance(attrs, list):
        return result

    for attr in attrs:
        if not isinstance(attr, dict):
            continue

        name = attr.get("name")
        if not name and "spec_id" in attr:
            name = attr_id_to_name.get(_safe_int(attr["spec_id"], default=-1))

        value = attr.get("value", "")
        if name:
            result[str(name)] = str(value)

    return result


def _first_existing_attr(
    attrs: dict[str, str],
    names: list[str],
    default: str,
) -> str:
    lower_map = {key.lower(): value for key, value in attrs.items()}

    for name in names:
        if name in attrs and attrs[name] not in {"", None}:
            return str(attrs[name])

        lower = name.lower()
        if lower in lower_map and lower_map[lower] not in {"", None}:
            return str(lower_map[lower])

    return default


def _shape_label_name(
    shape: dict[str, Any],
    label_id_to_name: dict[int, str],
) -> str:
    label = shape.get("label", shape.get("label_name", ""))
    if label:
        return str(label)

    label_id = shape.get("label_id")
    if label_id is None:
        return ""

    return label_id_to_name.get(_safe_int(label_id, default=-1), "")


def _behavior_from_label(label_name: str) -> str:
    normalized = normalize_behavior(label_name)
    return normalized or ""


def _points_to_bbox(points: object) -> tuple[float, float, float, float] | None:
    if not isinstance(points, list) or len(points) < 4:
        return None

    values: list[float] = []
    for value in points:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None

    if len(values) == 4:
        x1, y1, x2, y2 = values
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    xs = values[0::2]
    ys = values[1::2]
    if not xs or not ys:
        return None

    return min(xs), min(ys), max(xs), max(ys)


def _shape_is_outside(shape: dict[str, Any]) -> bool:
    value = shape.get("outside", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _resolve_video_key(
    *,
    video_key: str | None,
    task_meta: dict[str, Any],
    annotations_path: Path,
    task_path: Path | None,
) -> str:
    if video_key:
        return video_key

    for key in ["video_key", "name", "task_name"]:
        value = task_meta.get(key)
        if value:
            return str(value)

    if task_path is not None:
        return task_path.name

    return annotations_path.stem


def _resolve_image_size(
    *,
    task_meta: dict[str, Any],
    image_width: int | None,
    image_height: int | None,
) -> tuple[int | pd.NA, int | pd.NA]:
    if image_width is not None and image_height is not None:
        return image_width, image_height

    size_candidates = [
        task_meta.get("original_size"),
        task_meta.get("size"),
        task_meta.get("image_size"),
    ]

    task = task_meta.get("task")
    if isinstance(task, dict):
        size_candidates.extend(
            [
                task.get("original_size"),
                task.get("size"),
                task.get("image_size"),
            ]
        )

    width = image_width
    height = image_height

    for size in size_candidates:
        if not isinstance(size, dict):
            continue

        width = width or size.get("width")
        height = height or size.get("height")

    return (
        _safe_int(width, default=pd.NA),
        _safe_int(height, default=pd.NA),
    )


def _image_name_for_frame(
    *,
    frame_index: int,
    manifest: dict[int, str],
    video_key: str,
) -> str:
    if frame_index in manifest:
        return manifest[frame_index]
    return f"{video_key}__f{frame_index:06d}.jpg"


def _timestamp_from_frame(frame_index: int, fps: float | None) -> float | pd.NA:
    if fps is None or fps <= 0:
        return pd.NA
    return frame_index / fps


def _interaction_partner_ids(df: pd.DataFrame) -> pd.Series:
    partner_ids: dict[int, str] = {}

    for _, group in df.groupby("frame_uid", dropna=False):
        ids = [str(v) for v in group["pig_id"].dropna().tolist()]

        for idx, pig_id in zip(group.index, group["pig_id"], strict=False):
            partners = sorted(pid for pid in ids if pid != str(pig_id))
            partner_ids[idx] = "|".join(partners)

    return pd.Series(partner_ids)


def _first_existing_value(
    data: dict[str, Any],
    keys: list[str],
    default: Any,
) -> Any:
    for key in keys:
        if key in data and data[key] not in {None, ""}:
            return data[key]
    return default


def _safe_int(value: Any, *, default: Any = 0) -> Any:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _empty_canonical_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_FRAME_OBJECT_COLUMNS)