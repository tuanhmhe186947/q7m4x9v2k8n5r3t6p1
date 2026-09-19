"""Prepare fresh C6 union and full-frame media inputs from one rebuilt lineage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    letterbox_rgb_uint8,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

LINEAGE_SCOPE = "legacy-only-unreviewed-development"
C6_OFFSETS = (5, 6, 7, 8, 9, 10)
IMAGE_SIZE = 224
WINDOW_CONTEXT_DELIMITER = ";;"


@dataclass(frozen=True, slots=True)
class LegacyC6ModalityContextTables:
    """Exact C6 targets plus same-scene metadata needed for visual context."""

    frame_context: pd.DataFrame
    window_context: pd.DataFrame
    union_selection: pd.DataFrame
    full_frame_selection: pd.DataFrame
    audit: dict[str, Any]


def prepare_legacy_c6_modality_context(
    harmonized_frames: pd.DataFrame,
    selected_units: pd.DataFrame,
) -> LegacyC6ModalityContextTables:
    """Bind model-visible C6 slots to video media without loading outer media."""

    _require_columns(
        harmonized_frames,
        {
            "temporal_unit_key",
            "scene_frame_uid",
            "frame_uid",
            "source_type",
            "dataset_id",
            "video_key",
            "clip_id",
            "object_track_key",
            "pig_id",
            "track_id",
            "nearest_track_id",
            "frame_index",
            "relative_frame_index",
            "source_video_path",
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_valid",
            "lineage_scope",
            "human_review_complete",
        },
        "harmonized frames",
    )
    _require_columns(
        selected_units,
        {
            "temporal_unit_key",
            "position",
            "window_id",
            "l5_role",
            "lineage_scope",
            "human_review_complete",
        },
        "selected units",
    )
    _require_unreviewed_claim(harmonized_frames, "harmonized frames")
    _require_unreviewed_claim(selected_units, "selected units")
    if selected_units["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("selected C6 units contain duplicate temporal_unit_key")
    roles = set(selected_units["l5_role"].fillna("").astype(str))
    if roles != {"train", "validation"}:
        raise ValueError(f"selected C6 roles={sorted(roles)}")

    selected_keys = set(selected_units["temporal_unit_key"].astype(str))
    relative = pd.to_numeric(
        harmonized_frames["relative_frame_index"],
        errors="coerce",
    )
    target = harmonized_frames.loc[
        harmonized_frames["temporal_unit_key"].astype(str).isin(selected_keys)
        & relative.isin(C6_OFFSETS)
    ].copy()
    target["relative_frame_index"] = pd.to_numeric(
        target["relative_frame_index"],
        errors="raise",
    ).astype(np.int64)
    target["frame_index"] = pd.to_numeric(
        target["frame_index"],
        errors="raise",
    ).astype(np.int64)
    target = target.drop(
        columns=["position", "window_id", "l5_role"],
        errors="ignore",
    ).merge(
        selected_units[
            ["temporal_unit_key", "position", "window_id", "l5_role"]
        ],
        on="temporal_unit_key",
        how="inner",
        validate="many_to_one",
    )
    target = target.sort_values(
        ["position", "relative_frame_index", "frame_uid"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_c6_targets(target, selected_units)

    scene_keys = set(target["scene_frame_uid"].astype(str))
    context = harmonized_frames.loc[
        harmonized_frames["scene_frame_uid"].astype(str).isin(scene_keys)
    ].copy()
    context["frame_index"] = pd.to_numeric(
        context["frame_index"],
        errors="raise",
    ).astype(np.int64)
    context["resolved_media_path"] = context["source_video_path"].map(
        _resolved_media_path
    )
    context["resolved_media_exists"] = context[
        "resolved_media_path"
    ].map(lambda value: Path(value).is_file())
    context["image_context_id"] = [
        _image_context_id(source, object_key, frame_index)
        for source, object_key, frame_index in context[
            ["source_type", "object_track_key", "frame_index"]
        ].itertuples(index=False, name=None)
    ]
    context["image_context_source"] = "legacy_video_bbox"
    context["bbox_context_valid"] = _strict_bool(context["bbox_valid"])
    context["full_frame_context_available"] = context[
        "resolved_media_exists"
    ]
    context["partner_context_available"] = context[
        "nearest_track_id"
    ].fillna("").astype(str).str.strip().ne("")
    context["image_context_loadable"] = (
        context["resolved_media_exists"]
        & context["bbox_context_valid"]
    )
    context["image_context_error"] = np.where(
        context["resolved_media_exists"],
        "",
        "missing_source_video",
    )
    if not context["resolved_media_exists"].all():
        missing = sorted(
            set(
                context.loc[
                    ~context["resolved_media_exists"],
                    "resolved_media_path",
                ].astype(str)
            )
        )
        raise FileNotFoundError(f"C6 source videos missing={missing[:5]}")
    if context["image_context_id"].astype(str).duplicated().any():
        raise ValueError("C6 frame context duplicates image_context_id")
    lookup_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "clip_id",
        "frame_index",
        "track_id",
    ]
    if context[lookup_columns].astype(str).duplicated().any():
        raise ValueError("C6 same-frame actor lookup contains duplicates")

    id_by_frame = dict(
        zip(
            context["frame_uid"].astype(str),
            context["image_context_id"].astype(str),
            strict=True,
        )
    )
    window_records: list[dict[str, Any]] = []
    for unit in selected_units.sort_values("position").itertuples(index=False):
        rows = target.loc[
            target["temporal_unit_key"].astype(str).eq(
                str(unit.temporal_unit_key)
            )
        ].sort_values("relative_frame_index")
        first = rows.iloc[0]
        frame_uids = rows["frame_uid"].astype(str).tolist()
        scene_uids = rows["scene_frame_uid"].astype(str).tolist()
        context_ids = [id_by_frame[value] for value in frame_uids]
        frame_indices = rows["frame_index"].astype(int).tolist()
        window_records.append(
            {
                "window_id": str(unit.window_id),
                "temporal_unit_key": str(unit.temporal_unit_key),
                "source_type": str(first["source_type"]),
                "dataset_id": str(first["dataset_id"]),
                "video_key": str(first["video_key"]),
                "object_track_key": str(first["object_track_key"]),
                "pig_id": str(first["pig_id"]),
                "track_id": str(first["track_id"]),
                "window_length_frames": len(C6_OFFSETS),
                "window_start_frame": frame_indices[0],
                "window_end_frame": frame_indices[-1],
                "window_valid_for_main_train": True,
                "frame_uid_sequence": "|".join(frame_uids),
                "scene_frame_uid_sequence": "|".join(scene_uids),
                "image_context_id_sequence": (
                    WINDOW_CONTEXT_DELIMITER.join(context_ids)
                ),
                "expected_frame_indices": "|".join(
                    str(value) for value in frame_indices
                ),
                "observed_image_context_rows": len(C6_OFFSETS),
                "loadable_image_context_rows": len(C6_OFFSETS),
                "missing_image_context_slots": 0,
                "window_image_context_complete": True,
                "l5_role": str(unit.l5_role),
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        )
    windows = pd.DataFrame.from_records(window_records)
    union_ids = [
        value
        for sequence in windows["image_context_id_sequence"].astype(str)
        for value in sequence.split(WINDOW_CONTEXT_DELIMITER)
    ]
    if len(union_ids) != len(set(union_ids)):
        raise ValueError("C6 union target slots duplicate image_context_id")
    union_selection = pd.DataFrame({"image_context_id": union_ids})
    full_selection = _full_frame_selection(target)
    frame_context = context.sort_values(
        ["resolved_media_path", "frame_index", "object_track_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    audit = {
        "schema_version": "classification_v2.legacy_c6_modality_context.v1",
        "status": "PASS_LEGACY_C6_MODALITY_CONTEXT",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "selected_native_units": int(len(selected_units)),
        "train_native_units": int(
            selected_units["l5_role"].astype(str).eq("train").sum()
        ),
        "validation_native_units": int(
            selected_units["l5_role"].astype(str).eq("validation").sum()
        ),
        "c6_target_slots": int(len(target)),
        "context_frame_rows": int(len(frame_context)),
        "union_selected_slots": int(len(union_selection)),
        "full_frame_unique_scenes": int(len(full_selection)),
        "outer_holdout_metadata_rows_read": 0,
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "native_frame_offsets": list(C6_OFFSETS),
        "errors": [],
        "valid": True,
    }
    return LegacyC6ModalityContextTables(
        frame_context=frame_context,
        window_context=windows,
        union_selection=union_selection,
        full_frame_selection=full_selection,
        audit=audit,
    )


def build_legacy_c6_full_frame_pixel_cache(
    selection: pd.DataFrame,
    output_dir: Path,
    *,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Any]:
    """Decode each selected scene once into a deterministic packed RGB tensor."""

    if image_size <= 0:
        raise ValueError("C6 full-frame image_size must be positive")
    _require_columns(
        selection,
        {
            "scene_frame_uid",
            "resolved_media_path",
            "frame_index",
            "lineage_scope",
            "human_review_complete",
        },
        "full-frame selection",
    )
    _require_unreviewed_claim(selection, "full-frame selection")
    if selection["scene_frame_uid"].astype(str).duplicated().any():
        raise ValueError("C6 full-frame selection duplicates scene_frame_uid")
    output_dir.mkdir(parents=True, exist_ok=False)
    ordered = selection.sort_values(
        "scene_frame_uid",
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["packed_row"] = np.arange(len(ordered), dtype=np.int64)
    tensor_path = output_dir / f"packed_rgb_{image_size}_letterbox.npy"
    index_path = output_dir / "packed_image_cache_index.csv"
    audit_path = output_dir / "packed_image_cache_audit.json"
    tensor = np.lib.format.open_memmap(
        tensor_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(ordered), image_size, image_size, 3),
    )
    decode_count = 0
    seek_count = 0
    errors: list[str] = []
    try:
        media_order = ordered.sort_values(
            ["resolved_media_path", "frame_index", "scene_frame_uid"],
            kind="mergesort",
        )
        for media_path, rows in media_order.groupby(
            "resolved_media_path",
            sort=True,
        ):
            capture = cv2.VideoCapture(str(media_path))
            if not capture.isOpened():
                errors.append(f"video_open_failed={media_path}")
                capture.release()
                continue
            next_frame: int | None = None
            try:
                for row in rows.itertuples(index=False):
                    target = int(row.frame_index)
                    if next_frame != target:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                        seek_count += 1
                    ok, bgr = capture.read()
                    if not ok or bgr is None:
                        errors.append(
                            "video_decode_failed="
                            f"{media_path}:{target}:{row.scene_frame_uid}"
                        )
                        continue
                    next_frame = target + 1
                    decode_count += 1
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    tensor[int(row.packed_row)] = letterbox_rgb_uint8(
                        rgb,
                        image_size,
                    )
            finally:
                capture.release()
        tensor.flush()
    finally:
        tensor.flush()
        _close_memmap(tensor)
    if errors:
        raise RuntimeError("C6 full-frame cache failed: " + "; ".join(errors[:5]))
    index_columns = [
        "scene_frame_uid",
        "packed_row",
        "resolved_media_path",
        "frame_index",
        "lineage_scope",
        "human_review_complete",
    ]
    ordered[index_columns].to_csv(
        index_path,
        index=False,
        lineterminator="\n",
    )
    audit = {
        "schema_version": "classification_v2.legacy_c6_full_frame_cache.v1",
        "status": "PASS_LEGACY_C6_FULL_FRAME_PIXEL_CACHE",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "image_size": image_size,
        "resize_policy": "full_frame_letterbox_rgb_pad_black_v1",
        "packed_rows": int(len(ordered)),
        "source_media_reads": decode_count,
        "video_seek_count": seek_count,
        "outer_holdout_media_reads": 0,
        "tensor_path": str(tensor_path),
        "tensor_sha256": file_sha256(tensor_path),
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "errors": [],
        "valid": True,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def _validate_c6_targets(
    target: pd.DataFrame,
    selected_units: pd.DataFrame,
) -> None:
    grouped = target.groupby("temporal_unit_key", sort=False)
    if set(grouped.groups) != set(selected_units["temporal_unit_key"].astype(str)):
        raise ValueError("C6 target unit universe differs from selected units")
    counts = grouped.size()
    offsets = grouped["relative_frame_index"].apply(list)
    frame_indices = grouped["frame_index"].apply(list)
    if not counts.eq(len(C6_OFFSETS)).all():
        raise ValueError("C6 target slot count drift")
    if not offsets.map(lambda value: value == list(C6_OFFSETS)).all():
        raise ValueError("C6 target offsets drift")
    if not frame_indices.map(
        lambda value: bool(np.all(np.diff(value) == 1))
    ).all():
        raise ValueError("C6 target source frames are not contiguous")
    if target["frame_uid"].astype(str).duplicated().any():
        raise ValueError("C6 target duplicates frame_uid")


def _full_frame_selection(target: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scene_frame_uid",
        "source_video_path",
        "frame_index",
        "lineage_scope",
        "human_review_complete",
    ]
    work = target[columns].copy()
    work["resolved_media_path"] = work["source_video_path"].map(
        _resolved_media_path
    )
    consistency = work.groupby("scene_frame_uid", sort=False).agg(
        media_count=("resolved_media_path", "nunique"),
        frame_count=("frame_index", "nunique"),
    )
    if not consistency["media_count"].eq(1).all():
        raise ValueError("C6 scene frame maps to multiple media paths")
    if not consistency["frame_count"].eq(1).all():
        raise ValueError("C6 scene frame maps to multiple source frames")
    return (
        work.drop(columns="source_video_path")
        .drop_duplicates("scene_frame_uid")
        .sort_values("scene_frame_uid", kind="mergesort")
        .reset_index(drop=True)
    )


def _image_context_id(
    source_type: object,
    object_track_key: object,
    frame_index: object,
) -> str:
    return (
        f"{source_type}|{object_track_key}|"
        f"f{int(frame_index):06d}"
    )


def _resolved_media_path(value: object) -> str:
    path = str(value).strip()
    if not path or path.lower() in {"nan", "none", "<na>"}:
        raise ValueError("C6 source_video_path is blank")
    return str(Path(path))


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


def _require_unreviewed_claim(frame: pd.DataFrame, name: str) -> None:
    scopes = set(frame["lineage_scope"].fillna("").astype(str))
    if scopes != {LINEAGE_SCOPE}:
        raise ValueError(f"{name} lineage scopes={sorted(scopes)}")
    reviewed = set(_strict_bool(frame["human_review_complete"]))
    if reviewed != {False}:
        raise ValueError(f"{name} human-review claim drift")


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError("C6 boolean column contains missing values")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"C6 boolean values are invalid={unknown}")
    return normalized.map(mapping).astype(bool)


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


__all__ = [
    "C6_OFFSETS",
    "IMAGE_SIZE",
    "LINEAGE_SCOPE",
    "LegacyC6ModalityContextTables",
    "build_legacy_c6_full_frame_pixel_cache",
    "prepare_legacy_c6_modality_context",
]
