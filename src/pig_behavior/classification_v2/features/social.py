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
from pig_behavior.classification_v2.features.spatial_semantics import (
    AXIS_DISTANCE_METRIC_ID,
    AXIS_DISTANCE_METRIC_VERSION,
    DIAGONAL_DISTANCE_METRIC_ID,
    DIAGONAL_DISTANCE_METRIC_VERSION,
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_NEAR_THRESHOLD_ID,
    SOCIAL_NEAR_THRESHOLD_UNITS,
    SOCIAL_NEAR_THRESHOLD_VALUE,
    SOCIAL_TIE_BREAK_RULE,
    SOCIAL_TIE_BREAK_VERSION,
    canonical_social_identity_columns,
    pairwise_image_distance_matrices,
)


@dataclass(frozen=True, slots=True)
class StaticSocialConfig:
    near_distance_n: float = SOCIAL_NEAR_THRESHOLD_VALUE
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
    "nearest_partner_key",
    "nearest_stable_partner_key",
    "nearest_pig_id",
    "nearest_track_id",
    "nearest_object_id",
    "nearest_dist_n",
    "nearest_distance_axis",
    "nearest_distance_diagonal",
    "pair_distance_diagonal_n",
    "distance_available",
    "nearest_distance_available",
    "nearest_neighbor_available",
    "nearest_partner_available",
    "nearest_tie_count",
    "nearest_tie_break_rule",
    "nearest_pair_iou",
    "nearest_pair_overlap_ratio",
    "social_density_near_count",
    "social_density_available",
    "social_contact_count",
    "social_context_frame_size",
    "partner_candidate_count",
    "pair_contact_with_nearest",
    "social_context_valid",
    "social_exclusion_reason",
)


def build_static_social_context_features(
    frame_features: pd.DataFrame,
    *,
    near_distance_n: float = SOCIAL_NEAR_THRESHOLD_VALUE,
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
    for column in ("image_width", "image_height"):
        if column not in work.columns:
            work[column] = np.nan
    n = len(work)
    nearest_pig = np.full(n, "", dtype=object)
    nearest_track = np.full(n, "", dtype=object)
    nearest_object = np.full(n, "", dtype=object)
    nearest_partner_key = np.full(n, "", dtype=object)
    nearest_identity_field = np.full(n, "", dtype=object)
    nearest_axis = np.full(n, np.nan, dtype="float64")
    nearest_diagonal = np.full(n, np.nan, dtype="float64")
    nearest_tie_count = np.zeros(n, dtype="int64")
    nearest_iou = np.zeros(n, dtype="float64")
    nearest_overlap = np.zeros(n, dtype="float64")
    near_count = np.zeros(n, dtype="int64")
    contact_count = np.zeros(n, dtype="int64")
    candidate_count = np.zeros(n, dtype="int64")
    frame_size = np.zeros(n, dtype="int64")
    social_valid = np.zeros(n, dtype=bool)
    social_exclusion_reason = np.full(
        n,
        "invalid_actor_geometry_or_dimensions",
        dtype=object,
    )

    numeric_columns = (
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_area",
        "cx_n",
        "cy_n",
        "image_width",
        "image_height",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_social_frame_group_key"] = _social_frame_group_key(work)
    valid_bbox = _to_bool(work["bbox_valid"]).to_numpy(dtype=bool)
    pig_ids = work["pig_id"].fillna("").astype(str).to_numpy(dtype=object)
    track_ids = work["track_id"].fillna("").astype(str).to_numpy(dtype=object)
    object_ids = work.get(
        "object_id",
        pd.Series("", index=work.index),
    ).fillna("").astype(str).to_numpy(dtype=object)
    canonical_keys, identity_fields = canonical_social_identity_columns(work)
    canonical_key_values = canonical_keys.to_numpy(dtype=object)
    identity_field_values = identity_fields.to_numpy(dtype=object)
    boxes = work[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float64")
    areas = work["bbox_area"].to_numpy(dtype="float64")
    image_widths = work["image_width"].to_numpy(dtype="float64")
    image_heights = work["image_height"].to_numpy(dtype="float64")
    centers_px = np.column_stack(
        (
            work["cx_n"].to_numpy(dtype="float64") * image_widths,
            work["cy_n"].to_numpy(dtype="float64") * image_heights,
        )
    )

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
            if count > config.max_frame_group_size:
                social_exclusion_reason[indices] = "frame_group_too_large"
            continue
        geometry_valid = (
            np.isfinite(centers_px[indices]).all(axis=1)
            & np.isfinite(boxes[indices]).all(axis=1)
            & np.isfinite(areas[indices])
            & (areas[indices] > 0)
            & np.isfinite(image_widths[indices])
            & np.isfinite(image_heights[indices])
            & (image_widths[indices] > 0)
            & (image_heights[indices] > 0)
            & (boxes[indices, 2] > boxes[indices, 0])
            & (boxes[indices, 3] > boxes[indices, 1])
            & valid_bbox[indices]
        )
        identity_valid = np.asarray(
            [bool(str(value).strip()) for value in canonical_key_values[indices]],
            dtype=bool,
        )
        duplicate_identity = pd.Series(
            canonical_key_values[indices][identity_valid],
            dtype="string",
        ).duplicated(keep=False)
        if duplicate_identity.any():
            duplicates = sorted(
                set(
                    pd.Series(
                        canonical_key_values[indices][identity_valid],
                        dtype="string",
                    )[duplicate_identity].astype(str)
                )
            )
            raise ValueError(
                "Duplicate canonical social identities in one frame: "
                f"{duplicates}"
            )
        actor_valid = geometry_valid & identity_valid
        social_valid[indices[actor_valid]] = True
        social_exclusion_reason[indices[geometry_valid & ~identity_valid]] = (
            "invalid_actor_identity"
        )
        social_exclusion_reason[indices[actor_valid]] = "no_valid_neighbor"
        if count == 1 or not actor_valid.any():
            continue
        axis_distance, diagonal_distance, pair_valid = (
            pairwise_image_distance_matrices(
                centers_px[indices],
                image_widths[indices],
                image_heights[indices],
                actor_valid,
                [str(value) for value in canonical_key_values[indices]],
            )
        )
        nearest_position = np.zeros(count, dtype="int64")
        for local in range(count):
            candidates = [
                candidate
                for candidate in range(count)
                if pair_valid[local, candidate]
                and np.isfinite(axis_distance[local, candidate])
            ]
            candidate_count[indices[local]] = len(candidates)
            if candidates:
                nearest_position[local] = min(
                    candidates,
                    key=lambda candidate: (
                        axis_distance[local, candidate],
                        str(canonical_key_values[indices[candidate]]),
                    ),
                )
        nearest_value = axis_distance[np.arange(count), nearest_position]
        has_neighbor = np.isfinite(nearest_value) & actor_valid
        iou, overlap = _pairwise_box_overlap(boxes[indices], areas[indices])
        iou[~pair_valid] = 0.0
        overlap[~pair_valid] = 0.0
        contact = (iou >= config.contact_iou_threshold) | (
            overlap >= config.contact_overlap_threshold
        )
        contact &= pair_valid
        valid_global = indices[actor_valid]
        near_count[valid_global] = (
            np.isfinite(axis_distance[actor_valid])
            & (axis_distance[actor_valid] <= config.near_distance_n)
        ).sum(axis=1)
        contact_count[valid_global] = contact[actor_valid].sum(axis=1)
        for local in np.where(has_neighbor)[0]:
            target = indices[local]
            neighbor = nearest_position[local]
            nearest_pig[target] = pig_ids[indices[neighbor]]
            nearest_track[target] = track_ids[indices[neighbor]]
            nearest_object[target] = object_ids[indices[neighbor]]
            nearest_partner_key[target] = canonical_key_values[
                indices[neighbor]
            ]
            nearest_identity_field[target] = identity_field_values[
                indices[neighbor]
            ]
            nearest_axis[target] = nearest_value[local]
            nearest_diagonal[target] = diagonal_distance[local, neighbor]
            nearest_tie_count[target] = int(
                np.count_nonzero(
                    np.isfinite(axis_distance[local])
                    & (axis_distance[local] == nearest_value[local])
                )
            )
            nearest_iou[target] = iou[local, neighbor]
            nearest_overlap[target] = overlap[local, neighbor]
            social_exclusion_reason[target] = "available"

    work["nearest_partner_key"] = nearest_partner_key
    work["nearest_stable_partner_key"] = nearest_partner_key
    work["nearest_partner_identity_field"] = nearest_identity_field
    work["nearest_pig_id"] = nearest_pig
    work["nearest_track_id"] = nearest_track
    work["nearest_object_id"] = nearest_object
    work["nearest_dist_n"] = nearest_axis
    work["nearest_distance_axis"] = nearest_axis
    work["nearest_distance_diagonal"] = nearest_diagonal
    work["pair_distance_diagonal_n"] = nearest_diagonal
    work["nearest_tie_count"] = nearest_tie_count
    work["nearest_tie_break_rule"] = SOCIAL_TIE_BREAK_RULE
    work["nearest_pair_iou"] = nearest_iou
    work["nearest_pair_overlap_ratio"] = nearest_overlap
    work["social_density_near_count"] = near_count
    work["partner_candidate_count"] = candidate_count
    work["social_contact_count"] = contact_count
    work["social_context_frame_size"] = frame_size
    work["nearest_neighbor_available"] = work["nearest_partner_key"].ne("")
    work["nearest_partner_available"] = work["nearest_neighbor_available"]
    work["nearest_distance_available"] = work["nearest_neighbor_available"]
    work["distance_available"] = work["nearest_neighbor_available"]
    work["social_neighbor_available"] = work["nearest_neighbor_available"]
    work["social_density_available"] = social_valid
    work["pair_contact_with_nearest"] = (
        work["nearest_neighbor_available"]
        & (
            work["nearest_pair_iou"].ge(config.contact_iou_threshold)
            | work["nearest_pair_overlap_ratio"].ge(
                config.contact_overlap_threshold
            )
        )
    )
    work["social_context_valid"] = social_valid
    work["social_exclusion_reason"] = social_exclusion_reason
    work["distance_metric_id"] = AXIS_DISTANCE_METRIC_ID
    work["distance_metric_version"] = AXIS_DISTANCE_METRIC_VERSION
    work["diagonal_distance_metric_id"] = DIAGONAL_DISTANCE_METRIC_ID
    work["diagonal_distance_metric_version"] = (
        DIAGONAL_DISTANCE_METRIC_VERSION
    )
    work["social_distance_threshold_id"] = SOCIAL_NEAR_THRESHOLD_ID
    work["social_distance_threshold_value"] = float(config.near_distance_n)
    work["social_distance_threshold_units"] = SOCIAL_NEAR_THRESHOLD_UNITS
    work["social_distance_threshold_metric_id"] = AXIS_DISTANCE_METRIC_ID
    work["social_distance_threshold_metric_version"] = (
        AXIS_DISTANCE_METRIC_VERSION
    )
    work["social_identity_version"] = SOCIAL_IDENTITY_VERSION
    work["social_tie_break_version"] = SOCIAL_TIE_BREAK_VERSION
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
