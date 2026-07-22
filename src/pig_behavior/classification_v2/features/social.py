"""Same-frame social geometry for Classification V2.

This module deliberately contains no shift, diff, rolling, or temporal pair
logic. Native-unit and final-view partner deltas are computed by their own
grain-specific stages from these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import scene_frame_key


@dataclass(frozen=True, slots=True)
class StaticSocialConfig:
    near_distance_n: float = 0.08
    contact_iou_threshold: float = 0.01
    contact_overlap_threshold: float = 0.05
    max_frame_group_size: int = 64

    def validate(self) -> None:
        if self.near_distance_n <= 0:
            raise ValueError("near_distance_n must be > 0")
        if self.contact_iou_threshold < 0:
            raise ValueError("contact_iou_threshold must be >= 0")
        if self.contact_overlap_threshold < 0:
            raise ValueError("contact_overlap_threshold must be >= 0")
        if self.max_frame_group_size <= 1:
            raise ValueError("max_frame_group_size must be > 1")


STATIC_SOCIAL_COLUMNS: tuple[str, ...] = (
    "nearest_pig_id",
    "nearest_track_id",
    "nearest_dist_n",
    "nearest_pair_iou",
    "nearest_pair_overlap_ratio",
    "social_density_near_count",
    "social_contact_count",
    "social_context_frame_size",
    "pair_contact_with_nearest",
    "social_context_valid",
)


def build_static_social_context_features(
    frame_features: pd.DataFrame,
    *,
    near_distance_n: float = 0.08,
    contact_iou_threshold: float = 0.01,
    contact_overlap_threshold: float = 0.05,
    max_frame_group_size: int = 64,
) -> pd.DataFrame:
    """Compute deterministic partner geometry using only rows in one frame."""

    config = StaticSocialConfig(
        near_distance_n=near_distance_n,
        contact_iou_threshold=contact_iou_threshold,
        contact_overlap_threshold=contact_overlap_threshold,
        max_frame_group_size=max_frame_group_size,
    )
    config.validate()
    required = {
        "source_type",
        "dataset_id",
        "video_key",
        "frame_index",
        "pig_id",
        "track_id",
        "bbox_valid",
        "cx_n",
        "cy_n",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_area",
    }
    missing = sorted(required.difference(frame_features.columns))
    if missing:
        raise ValueError(f"Missing static social input columns: {missing}")

    out = frame_features.copy()
    source_index = out.index
    work = out.reset_index(drop=True)
    n = len(work)
    nearest_pig = np.full(n, "", dtype=object)
    nearest_track = np.full(n, "", dtype=object)
    nearest_dist = np.full(n, np.nan, dtype="float64")
    nearest_iou = np.zeros(n, dtype="float64")
    nearest_overlap = np.zeros(n, dtype="float64")
    near_count = np.zeros(n, dtype="int64")
    contact_count = np.zeros(n, dtype="int64")
    frame_size = np.zeros(n, dtype="int64")
    social_valid = np.zeros(n, dtype=bool)

    numeric_columns = ("cx_n", "cy_n", "x1", "y1", "x2", "y2", "bbox_area")
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_social_frame_group_key"] = _social_frame_group_key(work)
    valid_bbox = _to_bool(work["bbox_valid"]).to_numpy(dtype=bool)
    pig_ids = work["pig_id"].fillna("").astype(str).to_numpy(dtype=object)
    track_ids = work["track_id"].fillna("").astype(str).to_numpy(dtype=object)
    object_keys = work.get(
        "object_track_key",
        work["track_id"],
    ).fillna("").astype(str).to_numpy(dtype=object)
    centers = work[["cx_n", "cy_n"]].to_numpy(dtype="float64")
    boxes = work[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float64")
    areas = work["bbox_area"].to_numpy(dtype="float64")

    groups = work.groupby(
        "_social_frame_group_key",
        dropna=False,
        sort=False,
    ).indices.values()
    for raw_indices in groups:
        indices = np.asarray(raw_indices, dtype="int64")
        count = indices.size
        frame_size[indices] = int(count)
        if count == 0 or count > config.max_frame_group_size:
            continue
        finite = (
            np.isfinite(centers[indices]).all(axis=1)
            & np.isfinite(boxes[indices]).all(axis=1)
            & np.isfinite(areas[indices])
            & (areas[indices] > 0)
            & valid_bbox[indices]
        )
        social_valid[indices[finite]] = True
        if count == 1 or not finite.any():
            continue
        delta = centers[indices, None, :] - centers[indices][None, :, :]
        distance = np.sqrt(np.square(delta).sum(axis=2))
        np.fill_diagonal(distance, np.inf)
        distance[~finite, :] = np.inf
        distance[:, ~finite] = np.inf
        nearest_position = np.zeros(count, dtype="int64")
        for local in range(count):
            candidates = [
                candidate
                for candidate in range(count)
                if np.isfinite(distance[local, candidate])
            ]
            if candidates:
                nearest_position[local] = min(
                    candidates,
                    key=lambda candidate: (
                        distance[local, candidate],
                        str(object_keys[indices[candidate]]),
                        str(track_ids[indices[candidate]]),
                        str(pig_ids[indices[candidate]]),
                    ),
                )
        nearest_value = distance[np.arange(count), nearest_position]
        has_neighbor = np.isfinite(nearest_value) & finite
        iou, overlap = _pairwise_box_overlap(boxes[indices], areas[indices])
        iou[~finite, :] = 0.0
        iou[:, ~finite] = 0.0
        overlap[~finite, :] = 0.0
        overlap[:, ~finite] = 0.0
        contact = (iou >= config.contact_iou_threshold) | (
            overlap >= config.contact_overlap_threshold
        )
        np.fill_diagonal(contact, False)
        valid_global = indices[finite]
        near_count[valid_global] = (
            distance[finite] <= config.near_distance_n
        ).sum(axis=1)
        contact_count[valid_global] = contact[finite].sum(axis=1)
        for local in np.where(has_neighbor)[0]:
            target = indices[local]
            neighbor = nearest_position[local]
            nearest_pig[target] = pig_ids[indices[neighbor]]
            nearest_track[target] = track_ids[indices[neighbor]]
            nearest_dist[target] = nearest_value[local]
            nearest_iou[target] = iou[local, neighbor]
            nearest_overlap[target] = overlap[local, neighbor]

    work["nearest_pig_id"] = nearest_pig
    work["nearest_track_id"] = nearest_track
    work["nearest_dist_n"] = nearest_dist
    work["nearest_pair_iou"] = nearest_iou
    work["nearest_pair_overlap_ratio"] = nearest_overlap
    work["social_density_near_count"] = near_count
    work["social_contact_count"] = contact_count
    work["social_context_frame_size"] = frame_size
    work["pair_contact_with_nearest"] = (
        work["nearest_pair_iou"].ge(config.contact_iou_threshold)
        | work["nearest_pair_overlap_ratio"].ge(
            config.contact_overlap_threshold
        )
    )
    work["social_context_valid"] = social_valid
    work = work.drop(columns=["_social_frame_group_key"])
    work.index = source_index
    return work


def _social_frame_group_key(rows: pd.DataFrame) -> pd.Series:
    frame_uid = scene_frame_key(rows).fillna("").astype(str).str.strip()
    frame_index = pd.to_numeric(rows["frame_index"], errors="coerce")
    invalid = frame_uid.eq("") & (
        frame_index.isna() | frame_index.mod(1).ne(0) | frame_index.lt(0)
    )
    if invalid.any():
        raise ValueError(
            "Social frame grouping requires a UID or valid frame index: "
            f"invalid_rows={int(invalid.sum())}"
        )
    fallback = "frame=" + frame_index.round().astype("Int64").astype(str)
    local = ("uid=" + frame_uid).where(frame_uid.ne(""), fallback)
    return (
        rows["source_type"].astype(str)
        + "|"
        + rows["dataset_id"].astype(str)
        + "|"
        + rows["video_key"].astype(str)
        + "|"
        + local
    )


def _pairwise_box_overlap(
    boxes: np.ndarray,
    areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = areas[:, None] + areas[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    minimum_area = np.minimum(areas[:, None], areas[None, :])
    overlap = np.divide(
        intersection,
        minimum_area,
        out=np.zeros_like(intersection),
        where=minimum_area > 0,
    )
    return iou, overlap


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.casefold().isin(
        {"1", "true", "yes", "y"}
    )


__all__ = [
    "STATIC_SOCIAL_COLUMNS",
    "StaticSocialConfig",
    "build_static_social_context_features",
]
