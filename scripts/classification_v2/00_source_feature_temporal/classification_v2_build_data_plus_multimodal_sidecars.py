"""Build keyed DATA+ RGB/H5/ROI/posture/46D sidecars on local CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (  # noqa: E402
    letterbox_rgb_uint8,
)
from pig_behavior.classification_v2.features.geometry import (  # noqa: E402
    build_geometry_features,
)
from pig_behavior.classification_v2.features.roi import (  # noqa: E402
    build_roi_features,
)
from pig_behavior.classification_v2.features.spatiotemporal import (  # noqa: E402
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (  # noqa: E402
    export_spatial_sequences,
)
from pig_behavior.classification_v2.training.data_plus_multimodal import (  # noqa: E402
    SPATIAL_SLICES,
    DataPlusSidecarPaths,
    make_data_plus_sample_key,
)

DEFAULT_ROOT = (
    REPO_ROOT
    / "outputs/classification_v2/data_plus_supplemental_t6_v2_20260825"
)
DEFAULT_ROI_COCO = (
    REPO_ROOT
    / "data"
    / "annotations"
    / "roi"
    / "ROI_annotations.toy_adjusted.coco.json"
)
SPATIAL_GROUPS = tuple(SPATIAL_SLICES)
PB_FOLD_COUNTS = {
    "vg1": 657,
    "vg2": 1252,
    "vg3": 1252,
    "vg4": 1244,
    "vg5": 1197,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DATA+ keyed multimodal sidecars without training."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _identity(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    unit_ids = np.asarray(
        manifest["supplemental_unit_id"].astype(str).to_numpy(),
        dtype=str,
    )
    return {
        "sample_key": np.asarray(
            [make_data_plus_sample_key(value) for value in unit_ids]
        ),
        "supplemental_unit_id": unit_ids,
        "target_object_track_key": np.asarray(
            manifest["target_object_track_key"].astype(str).to_numpy(),
            dtype=str,
        ),
    }


def _structured_bundle(
    root: Path,
    manifest: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source_path = root / "data_plus_spatial_46d.npz"
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    expected = len(manifest)
    if any(arrays[name].shape[0] != expected for name in SPATIAL_GROUPS):
        raise ValueError("DATA+ spatial bundle row count does not match manifest")
    frame_indices = arrays.get("frame_index_sequence")
    if frame_indices is None:
        raise ValueError("DATA+ spatial bundle lacks frame_index_sequence")
    groups = {
        unit_id: group.sort_values("frame_index", kind="mergesort")
        for unit_id, group in target_rows.groupby(
            "supplemental_unit_id", sort=False
        )
    }
    candidates: dict[tuple[int, ...], list[tuple[str, np.ndarray]]] = {}
    for unit_id, group in groups.items():
        width = group["image_width"].to_numpy(dtype=np.float64)
        height = group["image_height"].to_numpy(dtype=np.float64)
        boxes = np.stack(
            [
                (group["x1"] + group["x2"]) / (2.0 * width),
                (group["y1"] + group["y2"]) / (2.0 * height),
                (group["x2"] - group["x1"]) / width,
                (group["y2"] - group["y1"]) / height,
            ],
            axis=1,
        )
        frame_key = tuple(group["frame_index"].to_numpy(dtype=np.int32))
        candidates.setdefault(frame_key, []).append((unit_id, boxes))
    reorder: list[int] = []
    used: set[str] = set()
    for row_pos, (frames, boxes) in enumerate(
        zip(frame_indices, arrays["bbox_xywh_n"], strict=True)
    ):
        options = candidates.get(tuple(frames), [])
        scored = sorted(
            (float(np.max(np.abs(expected - boxes))), unit_id, expected)
            for unit_id, expected in options
            if unit_id not in used
        )
        if not scored or scored[0][0] > 1e-4:
            raise ValueError(
                "DATA+ spatial row cannot be keyed by frame/bbox authority "
                f"at row {row_pos}"
            )
        used.add(scored[0][1])
        reorder.append(
            int(manifest.index[manifest["supplemental_unit_id"] == scored[0][1]][0])
        )
    if len(reorder) != len(manifest) or len(set(reorder)) != len(manifest):
        raise ValueError("DATA+ spatial identity mapping is incomplete or duplicated")
    inverse = np.argsort(np.asarray(reorder, dtype=np.int64))
    arrays = {
        name: value[inverse] if value.shape[0] == len(inverse) else value
        for name, value in arrays.items()
    }
    structured = np.concatenate(
        [arrays[name] for name in SPATIAL_GROUPS],
        axis=-1,
    ).astype(np.float32)
    if structured.shape != (expected, 6, 46):
        raise ValueError(f"unexpected DATA+ structured shape: {structured.shape}")
    return arrays, structured


def _time_delta(manifest: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(manifest), 6), dtype=np.float32)
    for row_index, value in enumerate(manifest["selected_timestamps_seconds"]):
        timestamps = np.asarray(json.loads(str(value)), dtype=np.float64)
        if timestamps.shape != (6,) or not np.isfinite(timestamps).all():
            raise ValueError("invalid DATA+ timestamp sequence")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("DATA+ timestamp sequence must be strictly causal")
        result[row_index] = (timestamps - timestamps[0]).astype(np.float32)
    return result


def _target_rows(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    path = root / "supplemental_target_frame_features.csv"
    columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "supplemental_unit_id",
        "object_track_key",
        "frame_index",
        "timestamp_sec",
        "source_video_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    ]
    rows = pd.read_csv(path, usecols=columns, low_memory=False)
    rows["supplemental_unit_id"] = rows["supplemental_unit_id"].astype(str)
    rows = rows.sort_values(
        ["supplemental_unit_id", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    counts = rows.groupby("supplemental_unit_id").size()
    if len(rows) != len(manifest) * 6 or not counts.eq(6).all():
        raise ValueError("target frame features are not exactly T6 per unit")
    return rows


def _crop_box(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    image_size: int,
    *,
    expand_floor_ceil: bool,
) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    if not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = map(float, box)
    if expand_floor_ceil:
        ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
        ix2, iy2 = int(np.ceil(x2)), int(np.ceil(y2))
    else:
        ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
    ix1 = max(0, min(width, ix1))
    iy1 = max(0, min(height, iy1))
    ix2 = max(0, min(width, ix2))
    iy2 = max(0, min(height, iy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return letterbox_rgb_uint8(rgb, image_size)


def _union_box(actor: np.ndarray, partner: np.ndarray) -> np.ndarray:
    x1 = min(float(actor[0]), float(partner[0]))
    y1 = min(float(actor[1]), float(partner[1]))
    x2 = max(float(actor[2]), float(partner[2]))
    y2 = max(float(actor[3]), float(partner[3]))
    pad_x = (x2 - x1) * 0.1
    pad_y = (y2 - y1) * 0.1
    return np.asarray(
        [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
        dtype=np.float32,
    )


def _build_rgb(
    root: Path,
    manifest: pd.DataFrame,
    target_rows: pd.DataFrame,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / "data_plus_partner_context_v2.npz") as partner:
        partner_unit_ids = partner["supplemental_unit_id"].astype(str)
        expected_ids = manifest["supplemental_unit_id"].astype(str).to_numpy()
        if len(set(partner_unit_ids)) != len(partner_unit_ids):
            raise ValueError("partner context contains duplicate unit identities")
        partner_positions = {value: pos for pos, value in enumerate(partner_unit_ids)}
        if set(partner_positions) != set(expected_ids):
            raise ValueError("partner context identity does not match DATA+ manifest")
        reorder = np.asarray(
            [partner_positions[value] for value in expected_ids],
            dtype=np.int64,
        )
        partner_bbox = partner["partner_bbox_xyxy"][reorder].copy()
        partner_mask = partner["partner_mask"][reorder].astype(bool)

    n_rows = len(manifest)
    rgb = np.zeros(
        (n_rows, 2, 6, image_size, image_size, 3),
        dtype=np.uint8,
    )
    valid = np.zeros((n_rows, 2, 6), dtype=bool)
    unit_positions = {
        str(unit_id): position
        for position, unit_id in enumerate(manifest["supplemental_unit_id"])
    }
    rows = target_rows.copy()
    rows["unit_pos"] = rows["supplemental_unit_id"].map(unit_positions)
    rows["unit_step"] = rows.groupby(
        "supplemental_unit_id", sort=False
    ).cumcount()
    rows["video_path_resolved"] = rows["source_video_path"].map(
        lambda value: str((REPO_ROOT / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    for resolved, video_rows in rows.groupby("video_path_resolved", sort=False):
        capture = cv2.VideoCapture(resolved)
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode DATA+ video: {resolved}")
        try:
            next_frame: int | None = None
            for frame_index_value, frame_rows in video_rows.groupby(
                "frame_index", sort=True
            ):
                frame_index = int(frame_index_value)
                if next_frame != frame_index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"cannot decode {resolved} frame {frame_index}"
                    )
                next_frame = frame_index + 1
                for row in frame_rows.itertuples(index=False):
                    unit_pos = int(row.unit_pos)
                    step = int(row.unit_step)
                    actor = np.asarray(
                        [row.x1, row.y1, row.x2, row.y2], dtype=np.float32
                    )
                    actor_crop = _crop_box(
                        frame_bgr, actor, image_size, expand_floor_ceil=False
                    )
                    if actor_crop is None:
                        raise RuntimeError(
                            f"invalid actor crop: {row.supplemental_unit_id}"
                        )
                    rgb[unit_pos, 0, step] = actor_crop
                    valid[unit_pos, 0, step] = True
                    observed_slots = np.flatnonzero(
                        partner_mask[unit_pos, step]
                    )
                    if observed_slots.size:
                        nearest = partner_bbox[
                            unit_pos, step, int(observed_slots[0])
                        ]
                        union_crop = _crop_box(
                            frame_bgr,
                            _union_box(actor, nearest),
                            image_size,
                            expand_floor_ceil=True,
                        )
                        if union_crop is not None:
                            rgb[unit_pos, 1, step] = union_crop
                            valid[unit_pos, 1, step] = True
        finally:
            capture.release()
    return rgb, valid


def _build_h5(
    manifest: pd.DataFrame,
    structured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.zeros((len(manifest), 5, 46), dtype=np.float32)
    valid = np.zeros((len(manifest), 5), dtype=bool)
    observations: dict[tuple[str, str, int], np.ndarray] = {}
    for row_pos, unit in enumerate(manifest.itertuples(index=False)):
        frames = list(json.loads(str(unit.source_frame_indices)))
        for step, frame_index in enumerate(frames):
            observations[
                (
                    str(unit.source_clip_id),
                    str(unit.target_object_track_key),
                    int(frame_index),
                )
            ] = structured[row_pos, step]
    for row_pos, unit in enumerate(manifest.itertuples(index=False)):
        frames = list(json.loads(str(unit.source_frame_indices)))
        first_frame = int(frames[0])
        for slot, frame_index in enumerate(range(first_frame - 5, first_frame)):
            if frame_index < 0:
                continue
            key = (
                str(unit.source_clip_id),
                str(unit.target_object_track_key),
                frame_index,
            )
            value = observations.get(key)
            if value is not None:
                history[row_pos, slot] = value
                valid[row_pos, slot] = True
    return history, valid


def _partner_windows_and_frames(
    root: Path,
    manifest: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    slots = pd.read_csv(root / "supplemental_partner_slots.csv", low_memory=False)
    slots = slots[slots["partner_mask"].fillna(False).astype(bool)].copy()
    slots["supplemental_unit_id"] = slots["supplemental_unit_id"].astype(str)
    slots["partner_track_key"] = slots["partner_track_key"].astype(str)
    slots_by_frame = {
        (str(unit_id), int(frame_index)): group
        for (unit_id, frame_index), group in slots.groupby(
            ["supplemental_unit_id", "frame_index"], sort=False
        )
    }
    exploded: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    window_positions: list[tuple[int, int]] = []
    unit_position = {
        str(unit_id): pos
        for pos, unit_id in enumerate(manifest["supplemental_unit_id"])
    }
    target_groups = {
        str(unit_id): group.sort_values("frame_index", kind="mergesort")
        for unit_id, group in target_rows.groupby("supplemental_unit_id", sort=False)
    }

    for unit in manifest.itertuples(index=False):
        unit_id = str(unit.supplemental_unit_id)
        target = target_groups[unit_id]
        frame_values = tuple(json.loads(str(unit.source_frame_indices)))
        valid_slots: list[tuple[int, str, str]] = []
        for slot in range(2):
            selected = slots[
                slots["supplemental_unit_id"].eq(unit_id)
                & slots["partner_slot"].eq(slot)
            ].sort_values("frame_index", kind="mergesort")
            keys = tuple(selected["partner_track_key"])
            frames = tuple(selected["frame_index"].astype(int))
            if len(selected) == 6 and len(set(keys)) == 1 and frames == frame_values:
                partner_key = str(keys[0])
                export_key = f"{partner_key}::pb_unit={unit_id}::slot={slot}"
                valid_slots.append((slot, partner_key, export_key))
                record = unit._asdict()
                record["window_id"] = f"{unit_id}::pb_slot={slot}"
                record["object_track_key"] = export_key
                record["target_object_track_key"] = export_key
                record["pair_scope_key"] = record["window_id"]
                window_rows.append(record)
                window_positions.append((unit_position[unit_id], slot))

        valid_by_key = {
            key: (slot, export_key)
            for slot, key, export_key in valid_slots
        }
        for target_row in target.to_dict("records"):
            frame_index = int(target_row["frame_index"])
            scene_uid = f"data_plus_pb::{unit_id}::frame={frame_index:06d}"
            target_record = dict(target_row)
            target_record["scene_frame_uid"] = scene_uid
            target_record["frame_uid"] = scene_uid + "::target"
            target_record["temporal_unit_key"] = unit_id
            target_record["annotation_role"] = "pb_target_context"
            exploded.append(target_record)
            detected = slots_by_frame.get((unit_id, frame_index))
            if detected is None:
                continue
            for partner in detected.to_dict("records"):
                partner_key = str(partner["partner_track_key"])
                if partner_key not in valid_by_key:
                    continue
                slot, export_key = valid_by_key[partner_key]
                partner_record = dict(target_record)
                partner_record.update(
                    {
                        "object_track_key": export_key,
                        "track_id": f"detector_{partner['partner_tracker_id']}",
                        "pig_id": "DETECTED_PARTNER",
                        "behavior": "",
                        "x1_raw": float(partner["x1"]),
                        "y1_raw": float(partner["y1"]),
                        "x2_raw": float(partner["x2"]),
                        "y2_raw": float(partner["y2"]),
                        "x1": float(partner["x1"]),
                        "y1": float(partner["y1"]),
                        "x2": float(partner["x2"]),
                        "y2": float(partner["y2"]),
                        "bbox_valid": True,
                        "observed_mask": True,
                        "include_in_training": False,
                        "annotation_role": "pb_partner_actor",
                        "confidence": float(partner["partner_confidence"]),
                        "bbox_source": "hybrid_bytetrack_observation_only",
                        "temporal_unit_key": f"{unit_id}::pb_slot={slot}",
                        "frame_uid": (
                            scene_uid
                            + "::partner="
                            + quote(partner_key, safe="-_.~")
                        ),
                    }
                )
                exploded.append(partner_record)

    if not window_rows:
        return pd.DataFrame(), pd.DataFrame(), np.empty((0, 2), dtype=np.int64)
    context = pd.DataFrame(exploded)
    context = build_geometry_features(context)
    context = build_roi_features(context, roi_coco_path=DEFAULT_ROI_COCO)
    frames = build_enhanced_spatiotemporal_features(context)
    windows = pd.DataFrame(window_rows).reset_index(drop=True)
    positions = np.asarray(window_positions, dtype=np.int64)
    return windows, frames, positions


def _empty_partner_f2(n_rows: int, image_size: int) -> dict[str, np.ndarray]:
    return {
        "bbox_xywh_n": np.zeros((n_rows, 2, 6, 4), dtype=np.float32),
        "bbox_shape_n": np.zeros((n_rows, 2, 6, 2), dtype=np.float32),
        "motion_delta": np.zeros((n_rows, 2, 6, 12), dtype=np.float32),
        "roi_class_relation": np.zeros((n_rows, 2, 6, 18), dtype=np.float32),
        "social_relation": np.zeros((n_rows, 2, 6, 10), dtype=np.float32),
        "partner_actor_rgb": np.zeros(
            (n_rows, 2, 6, image_size, image_size, 3), dtype=np.uint8
        ),
        "partner_union_rgb": np.zeros(
            (n_rows, 2, 6, image_size, image_size, 3), dtype=np.uint8
        ),
        "partner_history_rgb": np.zeros(
            (n_rows, 2, 5, image_size, image_size, 3), dtype=np.uint8
        ),
        "partner_actor_valid": np.zeros((n_rows, 2, 6), dtype=bool),
        "partner_union_valid": np.zeros((n_rows, 2, 6), dtype=bool),
        "partner_history_valid": np.zeros((n_rows, 2, 5), dtype=bool),
        "partner_input_valid": np.zeros((n_rows, 2), dtype=bool),
        "observed_mask": np.zeros((n_rows, 2, 6), dtype=bool),
        "motion_feature_validity_mask": np.zeros(
            (n_rows, 2, 6, 12), dtype=bool
        ),
        "social_feature_validity_mask": np.zeros(
            (n_rows, 2, 6, 10), dtype=bool
        ),
        "roi_validity_mask": np.zeros((n_rows, 2, 6, 3), dtype=bool),
    }


def _build_partner_f2_inputs(
    root: Path,
    manifest: pd.DataFrame,
    target_rows: pd.DataFrame,
    image_size: int,
) -> dict[str, np.ndarray]:
    windows, frames, positions = _partner_windows_and_frames(
        root, manifest, target_rows
    )
    result = _empty_partner_f2(len(manifest), image_size)
    if windows.empty:
        return result
    exported = export_spatial_sequences(windows, frames, max_window_length=6)
    for export_pos, (unit_pos, slot) in enumerate(positions):
        for name in (
            "bbox_xywh_n",
            "bbox_shape_n",
            "motion_delta",
            "roi_class_relation",
            "social_relation",
        ):
            result[name][unit_pos, slot] = exported.arrays[name][export_pos]
        result["observed_mask"][unit_pos, slot] = (
            exported.arrays["observed_mask"][export_pos] > 0.5
        )
        result["motion_feature_validity_mask"][unit_pos, slot] = (
            exported.arrays["motion_feature_validity_mask"][export_pos] > 0.5
        )
        result["social_feature_validity_mask"][unit_pos, slot] = (
            exported.arrays["social_feature_validity_mask"][export_pos] > 0.5
        )
        result["roi_validity_mask"][unit_pos, slot] = (
            exported.arrays["roi_validity_mask"][export_pos] > 0.5
        )

    slots = pd.read_csv(root / "supplemental_partner_slots.csv", low_memory=False)
    detections = pd.read_csv(root / "partner_detections.csv", low_memory=False)
    detection_lookup = {
        (str(row.clip_id), int(row.tracker_id), int(row.frame_index)): np.asarray(
            [row.x1, row.y1, row.x2, row.y2], dtype=np.float32
        )
        for row in detections.itertuples(index=False)
    }
    target_lookup = {
        (str(row.supplemental_unit_id), int(row.frame_index)): np.asarray(
            [row.x1, row.y1, row.x2, row.y2], dtype=np.float32
        )
        for row in target_rows.itertuples(index=False)
    }
    slot_lookup = {
        (
            str(row.supplemental_unit_id),
            int(row.partner_slot),
            int(row.frame_index),
        ): row
        for row in slots.itertuples(index=False)
        if bool(row.partner_mask)
    }
    requests: dict[
        tuple[str, int], list[tuple[str, int, int, int, np.ndarray]]
    ] = {}
    for unit_pos, slot in positions:
        unit = manifest.iloc[int(unit_pos)]
        unit_id = str(unit["supplemental_unit_id"])
        clip_id = str(unit["source_clip_id"])
        video_path = Path(str(unit["source_video_path"]))
        if not video_path.is_absolute():
            video_path = (REPO_ROOT / video_path).resolve()
        frame_ids = tuple(json.loads(str(unit["source_frame_indices"])))
        slot_rows = [
            slot_lookup.get((unit_id, int(slot), int(frame_id)))
            for frame_id in frame_ids
        ]
        if any(row is None for row in slot_rows):
            continue
        tracker_ids = {int(row.partner_tracker_id) for row in slot_rows}
        partner_keys = {str(row.partner_track_key) for row in slot_rows}
        if len(tracker_ids) != 1 or len(partner_keys) != 1:
            continue
        tracker_id = next(iter(tracker_ids))
        first_frame = int(frame_ids[0])
        history_boxes = [
            detection_lookup.get((clip_id, tracker_id, frame_index))
            for frame_index in range(first_frame - 5, first_frame)
        ]
        if first_frame < 5 or any(box is None for box in history_boxes):
            continue
        if not result["observed_mask"][unit_pos, slot].all():
            continue
        if not result["roi_validity_mask"][unit_pos, slot].all():
            continue
        for step, (frame_id, row) in enumerate(
            zip(frame_ids, slot_rows, strict=True)
        ):
            partner_box = np.asarray(
                [row.x1, row.y1, row.x2, row.y2], dtype=np.float32
            )
            target_box = target_lookup[(unit_id, int(frame_id))]
            request_key = (str(video_path), int(frame_id))
            requests.setdefault(request_key, []).extend(
                [
                    ("actor", int(unit_pos), int(slot), step, partner_box),
                    (
                        "union",
                        int(unit_pos),
                        int(slot),
                        step,
                        _union_box(target_box, partner_box),
                    ),
                ]
            )
        for step, (frame_id, box) in enumerate(
            zip(
                range(first_frame - 5, first_frame),
                history_boxes,
                strict=True,
            )
        ):
            requests.setdefault((str(video_path), frame_id), []).append(
                ("history", int(unit_pos), int(slot), step, box)
            )

    by_video: dict[str, list[int]] = {}
    for video_path, frame_index in requests:
        by_video.setdefault(video_path, []).append(frame_index)
    for video_path, frame_indices in by_video.items():
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode DATA+ video: {video_path}")
        try:
            next_frame: int | None = None
            for frame_index in sorted(set(frame_indices)):
                if next_frame != frame_index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"cannot decode {video_path} frame {frame_index}"
                    )
                next_frame = frame_index + 1
                for kind, unit_pos, slot, step, box in requests[
                    (video_path, frame_index)
                ]:
                    crop = _crop_box(
                        frame_bgr,
                        box,
                        image_size,
                        expand_floor_ceil=True,
                    )
                    if crop is None:
                        continue
                    if kind == "actor":
                        result["partner_actor_rgb"][unit_pos, slot, step] = crop
                        result["partner_actor_valid"][unit_pos, slot, step] = True
                    elif kind == "union":
                        result["partner_union_rgb"][unit_pos, slot, step] = crop
                        result["partner_union_valid"][unit_pos, slot, step] = True
                    else:
                        result["partner_history_rgb"][unit_pos, slot, step] = crop
                        result["partner_history_valid"][unit_pos, slot, step] = True
        finally:
            capture.release()

    result["partner_input_valid"] = (
        result["partner_actor_valid"].all(axis=2)
        & result["partner_union_valid"].all(axis=2)
        & result["partner_history_valid"].all(axis=2)
        & result["observed_mask"].all(axis=2)
        & result["roi_validity_mask"].all(axis=(2, 3))
    )
    for unit_pos, slot in np.argwhere(~result["partner_input_valid"]):
        for name in (
            "bbox_xywh_n",
            "bbox_shape_n",
            "motion_delta",
            "roi_class_relation",
            "social_relation",
            "partner_actor_rgb",
            "partner_union_rgb",
            "partner_history_rgb",
            "partner_actor_valid",
            "partner_union_valid",
            "partner_history_valid",
            "observed_mask",
            "motion_feature_validity_mask",
            "social_feature_validity_mask",
            "roi_validity_mask",
        ):
            result[name][unit_pos, slot] = 0
    return result


def _pb_fold_coverage(root: Path) -> dict[str, Any]:
    eligibility = pd.read_csv(root / "supplemental_fold_eligibility.csv")
    pb_root = REPO_ROOT / "outputs/classification_v2/pb_teacher_f2_v1"
    fold_coverage: dict[str, Any] = {}
    observed_union: set[str] = set()
    for fold, expected_count in PB_FOLD_COUNTS.items():
        fold_rows = eligibility[
            eligibility["fold"].astype(str).str.lower().eq(fold)
        ]
        if (
            fold_rows["eligible_for_inner_validation"].astype(bool).any()
            or fold_rows["eligible_for_outer_test"].astype(bool).any()
        ):
            raise ValueError(f"{fold} DATA+ eligibility reaches validation/test")
        unit_ids = fold_rows.loc[
            fold_rows["eligible_for_training"].astype(bool),
            "supplemental_unit_id",
        ].astype(str)
        required = {make_data_plus_sample_key(value) for value in unit_ids}
        if len(required) != expected_count:
            raise ValueError(
                f"{fold} DATA+ train count {len(required)} != {expected_count}"
            )
        cache = torch.load(
            pb_root / fold / "pb_teacher_features.pt",
            map_location="cpu",
            weights_only=False,
        )
        sample_keys = tuple(str(value) for value in cache["sample_key"])
        positions = {key: pos for pos, key in enumerate(sample_keys)}
        missing = required.difference(positions)
        if missing:
            raise ValueError(f"{fold} PB cache misses DATA+ keys: {sorted(missing)[:5]}")
        for name, tail in (
            ("partner_probs", (2, 10)),
            ("partner_hidden", (2, 256)),
            ("partner_mask", (2,)),
        ):
            tensor = cache.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{fold} PB cache lacks tensor {name}")
            if tuple(tensor.shape) != (len(sample_keys), *tail):
                raise ValueError(f"{fold} PB tensor shape mismatch for {name}")
        observed_union.update(required)
        fold_coverage[fold.upper()] = {
            "eligible_train_keys": len(required),
            "pb_probability_keys": len(required),
            "pb_hidden_keys": len(required),
            "inner_validation_keys": 0,
            "outer_test_keys": 0,
        }
    return {
        "unique_data_plus_keys": len(observed_union),
        "fold_eligible_key_total": sum(PB_FOLD_COUNTS.values()),
        "folds": fold_coverage,
    }


def _canonical_guard(root: Path) -> dict[str, Any]:
    source = json.loads((root / "data_plus_build_audit.json").read_text())
    authorities = source["authorities"]
    hash_matches = {
        name: sha256_file(Path(record["path"])) == record["sha256"]
        for name, record in authorities.items()
    }
    if not all(hash_matches.values()):
        raise ValueError("canonical authority hash changed during DATA+ build")
    if source["canonical_population_modified"]:
        raise ValueError("canonical population guard is not clean")
    if source["frozen_fold_manifests_modified"]:
        raise ValueError("frozen fold manifest guard is not clean")
    return {
        "authority_hash_matches": hash_matches,
        "canonical_population_modified": False,
        "frozen_fold_manifests_modified": False,
    }


def build(root: Path, image_size: int) -> dict[str, Any]:
    if image_size != 128:
        raise ValueError("final Joint DATA+ RGB authority requires image_size=128")
    root = root.resolve()
    manifest = pd.read_csv(root / "supplemental_t6_manifest.csv", low_memory=False)
    manifest = manifest.sort_values(
        ["supplemental_unit_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(manifest) != 1252:
        raise ValueError(f"expected 1252 DATA+ units, observed {len(manifest)}")
    identity = _identity(manifest)
    target_rows = _target_rows(root, manifest)
    arrays, structured = _structured_bundle(root, manifest, target_rows)
    rgb, rgb_valid = _build_rgb(root, manifest, target_rows, image_size)
    h5, h5_valid = _build_h5(manifest, structured)
    partner_f2 = _build_partner_f2_inputs(
        root, manifest, target_rows, image_size
    )
    roi = structured[:, :, SPATIAL_SLICES["roi_class_relation"]].copy()
    roi_valid = arrays["roi_validity_mask"].astype(bool)
    paths = DataPlusSidecarPaths.from_root(root)

    np.savez(
        paths.rgb,
        **identity,
        feature_tensor=rgb,
        validity_mask=rgb_valid,
        time_delta_seconds=_time_delta(manifest),
    )
    np.savez_compressed(
        paths.spatial,
        **identity,
        class_label=np.asarray(
            manifest["class_label"].astype(str).to_numpy(),
            dtype=str,
        ),
        feature_tensor=structured,
        validity_mask=arrays["spatial_quality_mask"].astype(bool),
        roi_validity_mask=roi_valid,
        social_feature_validity_mask=arrays[
            "social_feature_validity_mask"
        ].astype(bool),
        motion_feature_validity_mask=arrays[
            "motion_feature_validity_mask"
        ].astype(bool),
    )
    np.savez_compressed(
        paths.h5,
        **identity,
        feature_tensor=h5,
        validity_mask=h5_valid,
    )
    np.savez_compressed(
        paths.roi,
        **identity,
        feature_tensor=roi,
        validity_mask=roi_valid,
    )
    np.savez_compressed(
        paths.posture,
        **identity,
        feature_tensor=np.full(len(manifest), -1, dtype=np.int64),
        validity_mask=np.zeros(len(manifest), dtype=bool),
    )
    partner_f2_path = root / "data_plus_partner_f2_inputs.npz"
    np.savez_compressed(partner_f2_path, **identity, **partner_f2)
    artifacts = [
        paths.rgb,
        paths.spatial,
        paths.h5,
        paths.roi,
        paths.posture,
        partner_f2_path,
    ]
    pb_coverage = _pb_fold_coverage(root)
    canonical_guard = _canonical_guard(root)
    audit = {
        "schema_version": "classification_v2.data_plus_multimodal_sidecars.v1",
        "status": "PASS",
        "data_plus_multimodal_alignment_ready": "PASS",
        "sample_count": int(len(manifest)),
        "sample_key_unique": bool(len(set(identity["sample_key"])) == len(manifest)),
        "rgb_actor_frame_coverage": int(rgb_valid[:, 0].sum()),
        "rgb_actor_unit_coverage": int(rgb_valid[:, 0].all(axis=1).sum()),
        "rgb_union_frame_coverage": int(rgb_valid[:, 1].sum()),
        "rgb_union_unit_any_coverage": int(rgb_valid[:, 1].any(axis=1).sum()),
        "h5_observed_slot_coverage": int(h5_valid.sum()),
        "h5_full_unit_coverage": int(h5_valid.all(axis=1).sum()),
        "roi_valid_channel_coverage": int(roi_valid.sum()),
        "roi_unit_any_coverage": int(roi_valid.any(axis=(1, 2)).sum()),
        "posture_reviewed_coverage": 0,
        "posture_policy": "target=-1 and mask=false; no fabricated posture label",
        "structured46d_unit_coverage": int(
            arrays["spatial_quality_mask"].astype(bool).all(axis=1).sum()
        ),
        "partner_f2_valid_slots": int(
            partner_f2["partner_input_valid"].sum()
        ),
        "partner_f2_zero_rgb_valid_slots": int(
            (
                partner_f2["partner_input_valid"]
                & ~partner_f2["partner_actor_rgb"].any(axis=(2, 3, 4, 5))
            ).sum()
        ),
        "partner_f2_missing_required_h5_valid_slots": int(
            (
                partner_f2["partner_input_valid"]
                & ~partner_f2["partner_history_valid"].all(axis=2)
            ).sum()
        ),
        "pb_coverage": pb_coverage,
        "canonical_guard": canonical_guard,
        "h5_policy": "five strictly previous same-clip same-target frames only",
        "gpu_used": False,
        "training_runs": 0,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        },
    }
    audit_path = root / "data_plus_multimodal_alignment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    args = parse_args()
    audit = build(args.root, args.image_size)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
