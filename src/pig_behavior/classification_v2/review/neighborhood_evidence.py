"""Label-independent set-level geometry for Behavior Review diagnostics.

This additive review-only contract does not alter the canonical spatial 46D,
does not enter model-X, and has no candidate-selection binding in v1.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REVIEW_NEIGHBORHOOD_SCHEMA_ID = (
    "schema.classification_v2_review_neighborhood_evidence_v1"
)
REVIEW_NEIGHBORHOOD_SCHEMA_VERSION = (
    "classification_v2.review_neighborhood_evidence.v1"
)
REVIEW_NEIGHBORHOOD_METRIC_VERSION = (
    "classification_v2.review_neighborhood_geometry.v1"
)
REVIEW_NEIGHBORHOOD_MAX_NEIGHBORS = 7
REVIEW_NEIGHBORHOOD_RESET_SCOPE = "temporal_unit_key"
REVIEW_NEIGHBORHOOD_NEAR_EDGE_DISTANCE_N = 0.08
REVIEW_NEIGHBORHOOD_CONTACT_IOU_PROXY_THRESHOLD = 0.01
REVIEW_NEIGHBORHOOD_CONTACT_OVERLAP_PROXY_THRESHOLD = 0.05

REVIEW_NEIGHBORHOOD_FRAME_FIELDS: tuple[str, ...] = (
    "valid_neighbor_count",
    "min_edge_distance_n",
    "min_center_distance_n",
    "max_pair_iou",
    "max_pair_overlap_min_area",
    "near_neighbor_count",
    "contact_proxy_count",
    "any_contact_proxy",
    "crowding_level",
    "neighborhood_evidence_available",
)

REVIEW_NEIGHBORHOOD_UNIT_FIELDS: tuple[str, ...] = (
    "observed_frame_count",
    "frames_with_valid_neighbors",
    "max_valid_neighbor_count",
    "neighbor_valid_ratio",
    "any_contact_proxy_ratio",
    "near_neighbor_ratio",
    "max_concurrent_contact_proxy_count",
    "min_edge_distance_over_unit",
    "median_min_edge_distance",
    "min_center_distance_over_unit",
    "median_min_center_distance",
    "overlap_present_ratio",
    "crowding_ratio",
    "neighborhood_evidence_available",
    "neighborhood_evidence_availability_reason",
)

REVIEW_NEIGHBORHOOD_FRAME_COLUMNS: tuple[str, ...] = (
    "temporal_unit_key",
    "scene_frame_uid",
    "frame_index",
    "actor_key_audit_only",
    *REVIEW_NEIGHBORHOOD_FRAME_FIELDS,
)

REVIEW_NEIGHBORHOOD_UNIT_COLUMNS: tuple[str, ...] = (
    "temporal_unit_key",
    *REVIEW_NEIGHBORHOOD_UNIT_FIELDS,
)

_REQUIRED_INPUT_COLUMNS = frozenset(
    {
        "temporal_unit_key",
        "scene_frame_uid",
        "frame_index",
        "object_track_key",
        "bbox_valid",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ReviewNeighborhoodEvidenceError(ValueError):
    """Raised when review-neighborhood evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class ReviewNeighborhoodEvidenceResult:
    """In-memory frame and unit evidence plus hash-bound metadata."""

    frame_evidence: pd.DataFrame
    unit_evidence: pd.DataFrame
    metadata: dict[str, Any]


def canonical_review_neighborhood_schema_payload() -> dict[str, Any]:
    """Return the exact semantic payload defining review evidence v1."""

    return {
        "schema_id": REVIEW_NEIGHBORHOOD_SCHEMA_ID,
        "schema_version": REVIEW_NEIGHBORHOOD_SCHEMA_VERSION,
        "metric_version": REVIEW_NEIGHBORHOOD_METRIC_VERSION,
        "frame_metadata_fields": [
            "temporal_unit_key",
            "scene_frame_uid",
            "frame_index",
            "actor_key_audit_only",
        ],
        "ordered_frame_fields": list(REVIEW_NEIGHBORHOOD_FRAME_FIELDS),
        "ordered_unit_fields": list(REVIEW_NEIGHBORHOOD_UNIT_FIELDS),
        "field_units": {
            "valid_neighbor_count": "count",
            "min_edge_distance_n": "axis_normalized_image_distance",
            "min_center_distance_n": "axis_normalized_image_distance",
            "max_pair_iou": "ratio",
            "max_pair_overlap_min_area": "ratio",
            "near_neighbor_count": "count",
            "contact_proxy_count": "count",
            "any_contact_proxy": "boolean",
            "crowding_level": "ratio_of_current_seven_neighbor_cap",
            "neighborhood_evidence_available": "boolean",
            "observed_frame_count": "count",
            "frames_with_valid_neighbors": "count",
            "max_valid_neighbor_count": "count",
            "neighbor_valid_ratio": "ratio",
            "any_contact_proxy_ratio": "ratio_over_available_frames",
            "near_neighbor_ratio": "ratio_over_available_frames",
            "max_concurrent_contact_proxy_count": "count",
            "min_edge_distance_over_unit": (
                "axis_normalized_image_distance"
            ),
            "median_min_edge_distance": "axis_normalized_image_distance",
            "min_center_distance_over_unit": (
                "axis_normalized_image_distance"
            ),
            "median_min_center_distance": (
                "axis_normalized_image_distance"
            ),
            "overlap_present_ratio": "ratio_over_available_frames",
            "crowding_ratio": "ratio_over_available_frames",
            "neighborhood_evidence_availability_reason": "enum",
        },
        "edge_distance_formula": (
            "hypot(max(signed_sep_x,0)/image_width,"
            "max(signed_sep_y,0)/image_height)"
        ),
        "near_edge_distance_n": (
            REVIEW_NEIGHBORHOOD_NEAR_EDGE_DISTANCE_N
        ),
        "contact_proxy": {
            "name": "bbox_overlap_contact_proxy",
            "pair_iou_threshold": (
                REVIEW_NEIGHBORHOOD_CONTACT_IOU_PROXY_THRESHOLD
            ),
            "pair_overlap_min_area_threshold": (
                REVIEW_NEIGHBORHOOD_CONTACT_OVERLAP_PROXY_THRESHOLD
            ),
            "physical_contact_claimed": False,
        },
        "neighbor_scope": (
            "all available non-self neighbors, capped at seven under the "
            "current eight-pig data authority"
        ),
        "arbitrary_pen_size_generalization_claimed": False,
        "missingness_policy": {
            "no_valid_neighbor": (
                "available=false; counts and ratios zero; distances NaN"
            ),
            "unavailable_is_far": False,
            "unavailable_is_observed_zero_relation": False,
        },
        "temporal_pair_dynamics": "none_in_v1",
        "reset_scope": REVIEW_NEIGHBORHOOD_RESET_SCOPE,
        "absolute_identity_numeric_evidence": "forbidden",
        "behavior_label_dependency": "forbidden",
        "review_decision_dependency": "forbidden",
        "candidate_selection_binding": "none_in_v1",
        "model_x_usage": "forbidden",
    }


def review_neighborhood_schema_hash() -> str:
    """Return SHA-256 over all review-evidence semantics."""

    encoded = json.dumps(
        canonical_review_neighborhood_schema_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


REVIEW_NEIGHBORHOOD_SCHEMA_HASH = review_neighborhood_schema_hash()


def _to_bool(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(bool)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"1", "true", "yes", "y"})
        .to_numpy(bool)
    )


def _validate_hashes(
    producer_sha: str,
    input_hashes: Mapping[str, str],
) -> dict[str, str]:
    normalized_sha = str(producer_sha).strip().casefold()
    if not _GIT_SHA_PATTERN.fullmatch(normalized_sha):
        raise ReviewNeighborhoodEvidenceError(
            "producer_sha must be a full 40-character Git SHA"
        )
    normalized: dict[str, str] = {}
    for raw_name, raw_value in sorted(input_hashes.items()):
        name = str(raw_name).strip()
        value = str(raw_value).strip().casefold()
        if not name or not _SHA256_PATTERN.fullmatch(value):
            raise ReviewNeighborhoodEvidenceError(
                f"invalid input hash binding: {raw_name!r}"
            )
        normalized[name] = value
    if not normalized:
        raise ReviewNeighborhoodEvidenceError("input_hashes must not be empty")
    return normalized


def _validate_inputs(rows: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_INPUT_COLUMNS.difference(rows.columns))
    if missing:
        raise ReviewNeighborhoodEvidenceError(
            f"missing review-neighborhood input columns: {missing}"
        )
    if rows.empty:
        raise ReviewNeighborhoodEvidenceError(
            "review-neighborhood input must not be empty"
        )
    blank_unit = rows["temporal_unit_key"].fillna("").astype(str).str.strip().eq("")
    blank_frame = rows["scene_frame_uid"].fillna("").astype(str).str.strip().eq("")
    blank_actor = rows["object_track_key"].fillna("").astype(str).str.strip().eq("")
    if blank_unit.any() or blank_frame.any() or blank_actor.any():
        raise ReviewNeighborhoodEvidenceError(
            "temporal_unit_key, scene_frame_uid, and object_track_key "
            "must be nonblank"
        )


def _build_frame_evidence(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.reset_index(drop=True).copy()
    numeric_columns = (
        "frame_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    keys = work["object_track_key"].astype(str).to_numpy(object)
    boxes = work[["x1", "y1", "x2", "y2"]].to_numpy(float)
    widths = work["image_width"].to_numpy(float)
    heights = work["image_height"].to_numpy(float)
    bbox_valid = _to_bool(work["bbox_valid"])
    geometry_valid = (
        bbox_valid
        & np.isfinite(boxes).all(axis=1)
        & np.isfinite(widths)
        & np.isfinite(heights)
        & (widths > 0)
        & (heights > 0)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    row_count = len(work)
    valid_neighbor_count = np.zeros(row_count, np.int64)
    min_edge_distance = np.full(row_count, np.nan)
    min_center_distance = np.full(row_count, np.nan)
    max_iou = np.full(row_count, np.nan)
    max_overlap = np.full(row_count, np.nan)
    near_count = np.zeros(row_count, np.int64)
    contact_count = np.zeros(row_count, np.int64)
    available = np.zeros(row_count, bool)

    groups = work.groupby(
        "scene_frame_uid",
        sort=False,
        dropna=False,
    ).indices
    for raw_indices in groups.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        group_keys = keys[indices]
        group_valid = geometry_valid[indices]
        valid_keys = pd.Series(
            group_keys[group_valid],
            dtype="string",
        )
        if valid_keys.duplicated().any():
            duplicates = sorted(
                valid_keys[valid_keys.duplicated(False)].unique()
            )
            raise ReviewNeighborhoodEvidenceError(
                "duplicate actor identity in scene frame: "
                f"{duplicates[:10]}"
            )
        group_boxes = boxes[indices]
        x1 = group_boxes[:, 0]
        y1 = group_boxes[:, 1]
        x2 = group_boxes[:, 2]
        y2 = group_boxes[:, 3]
        pair_valid = (
            group_valid[:, None]
            & group_valid[None, :]
            & (group_keys[:, None] != group_keys[None, :])
        )
        neighbor_counts = pair_valid.sum(axis=1)
        if np.any(neighbor_counts > REVIEW_NEIGHBORHOOD_MAX_NEIGHBORS):
            raise ReviewNeighborhoodEvidenceError(
                "more than seven valid non-self neighbors exceeds the "
                "current eight-pig data authority"
            )
        signed_sep_x = np.maximum(x1[:, None], x1[None, :]) - np.minimum(
            x2[:, None],
            x2[None, :],
        )
        signed_sep_y = np.maximum(y1[:, None], y1[None, :]) - np.minimum(
            y2[:, None],
            y2[None, :],
        )
        gap_x = np.maximum(signed_sep_x, 0.0)
        gap_y = np.maximum(signed_sep_y, 0.0)
        edge = np.hypot(
            gap_x / widths[indices, None],
            gap_y / heights[indices, None],
        )
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center = np.hypot(
            (center_x[None, :] - center_x[:, None])
            / widths[indices, None],
            (center_y[None, :] - center_y[:, None])
            / heights[indices, None],
        )
        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[:, None], x2[None, :])
            - np.maximum(x1[:, None], x1[None, :]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[:, None], y2[None, :])
            - np.maximum(y1[:, None], y1[None, :]),
        )
        intersection = intersection_width * intersection_height
        area = (x2 - x1) * (y2 - y1)
        union = area[:, None] + area[None, :] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        minimum_area = np.minimum(area[:, None], area[None, :])
        overlap = np.divide(
            intersection,
            minimum_area,
            out=np.zeros_like(intersection),
            where=minimum_area > 0,
        )
        contact_proxy = (
            iou >= REVIEW_NEIGHBORHOOD_CONTACT_IOU_PROXY_THRESHOLD
        ) | (
            overlap
            >= REVIEW_NEIGHBORHOOD_CONTACT_OVERLAP_PROXY_THRESHOLD
        )
        contact_proxy &= pair_valid
        edge = np.where(pair_valid, edge, np.nan)
        center = np.where(pair_valid, center, np.nan)
        iou = np.where(pair_valid, iou, np.nan)
        overlap = np.where(pair_valid, overlap, np.nan)
        group_available = neighbor_counts > 0
        valid_neighbor_count[indices] = neighbor_counts
        available[indices] = group_available
        for local in np.flatnonzero(group_available):
            global_index = indices[local]
            min_edge_distance[global_index] = np.nanmin(edge[local])
            min_center_distance[global_index] = np.nanmin(center[local])
            max_iou[global_index] = np.nanmax(iou[local])
            max_overlap[global_index] = np.nanmax(overlap[local])
            near_count[global_index] = int(
                np.count_nonzero(
                    pair_valid[local]
                    & (
                        edge[local]
                        <= REVIEW_NEIGHBORHOOD_NEAR_EDGE_DISTANCE_N
                    )
                )
            )
            contact_count[global_index] = int(
                np.count_nonzero(contact_proxy[local])
            )

    frame = pd.DataFrame(
        {
            "temporal_unit_key": work["temporal_unit_key"].astype(str),
            "scene_frame_uid": work["scene_frame_uid"].astype(str),
            "frame_index": work["frame_index"],
            "actor_key_audit_only": work["object_track_key"].astype(str),
            "valid_neighbor_count": valid_neighbor_count,
            "min_edge_distance_n": min_edge_distance,
            "min_center_distance_n": min_center_distance,
            "max_pair_iou": max_iou,
            "max_pair_overlap_min_area": max_overlap,
            "near_neighbor_count": near_count,
            "contact_proxy_count": contact_count,
            "any_contact_proxy": contact_count > 0,
            "crowding_level": (
                valid_neighbor_count
                / float(REVIEW_NEIGHBORHOOD_MAX_NEIGHBORS)
            ),
            "neighborhood_evidence_available": available,
        }
    )
    return frame.sort_values(
        ["temporal_unit_key", "frame_index", "actor_key_audit_only"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_unit_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    available = work["neighborhood_evidence_available"].astype(bool)
    work["_available"] = available.astype(int)
    work["_contact"] = (
        available & work["any_contact_proxy"].astype(bool)
    ).astype(int)
    work["_near"] = (
        available
        & work["min_edge_distance_n"].le(
            REVIEW_NEIGHBORHOOD_NEAR_EDGE_DISTANCE_N
        )
    ).astype(int)
    work["_overlap"] = (
        available & work["max_pair_overlap_min_area"].gt(0.0)
    ).astype(int)
    work["_crowded"] = (
        available & work["valid_neighbor_count"].ge(2)
    ).astype(int)
    grouped = work.groupby(
        "temporal_unit_key",
        sort=True,
        dropna=False,
    )
    unit = grouped.agg(
        observed_frame_count=("frame_index", "size"),
        frames_with_valid_neighbors=("_available", "sum"),
        max_valid_neighbor_count=("valid_neighbor_count", "max"),
        contact_proxy_frame_count=("_contact", "sum"),
        near_neighbor_frame_count=("_near", "sum"),
        max_concurrent_contact_proxy_count=("contact_proxy_count", "max"),
        min_edge_distance_over_unit=("min_edge_distance_n", "min"),
        median_min_edge_distance=("min_edge_distance_n", "median"),
        min_center_distance_over_unit=("min_center_distance_n", "min"),
        median_min_center_distance=("min_center_distance_n", "median"),
        overlap_present_frame_count=("_overlap", "sum"),
        crowded_frame_count=("_crowded", "sum"),
    ).reset_index()
    observed = unit["observed_frame_count"].replace(0, np.nan)
    valid = unit["frames_with_valid_neighbors"].replace(0, np.nan)
    unit["neighbor_valid_ratio"] = (
        unit["frames_with_valid_neighbors"] / observed
    ).fillna(0.0)
    unit["any_contact_proxy_ratio"] = (
        unit.pop("contact_proxy_frame_count") / valid
    ).fillna(0.0)
    unit["near_neighbor_ratio"] = (
        unit.pop("near_neighbor_frame_count") / valid
    ).fillna(0.0)
    unit["overlap_present_ratio"] = (
        unit.pop("overlap_present_frame_count") / valid
    ).fillna(0.0)
    unit["crowding_ratio"] = (
        unit.pop("crowded_frame_count") / valid
    ).fillna(0.0)
    unit["neighborhood_evidence_available"] = (
        unit["frames_with_valid_neighbors"] > 0
    )
    unit["neighborhood_evidence_availability_reason"] = np.where(
        unit["neighborhood_evidence_available"],
        "available",
        "no_valid_neighbor_observation",
    )
    return unit.loc[:, REVIEW_NEIGHBORHOOD_UNIT_COLUMNS]


def build_review_neighborhood_evidence(
    frame_features: pd.DataFrame,
    *,
    producer_sha: str,
    input_hashes: Mapping[str, str],
) -> ReviewNeighborhoodEvidenceResult:
    """Build deterministic set-level review evidence without target labels."""

    _validate_inputs(frame_features)
    normalized_hashes = _validate_hashes(producer_sha, input_hashes)
    frame = _build_frame_evidence(frame_features)
    unit = _build_unit_evidence(frame)
    metadata = {
        **canonical_review_neighborhood_schema_payload(),
        "schema_hash": REVIEW_NEIGHBORHOOD_SCHEMA_HASH,
        "producer_sha": producer_sha.strip().casefold(),
        "input_sha256": normalized_hashes,
        "authoritative_status": (
            "PRE_REVIEW_LABEL_INDEPENDENT_INFRASTRUCTURE"
        ),
    }
    require_review_neighborhood_evidence(unit, metadata)
    return ReviewNeighborhoodEvidenceResult(
        frame_evidence=frame,
        unit_evidence=unit,
        metadata=metadata,
    )


def audit_review_neighborhood_evidence(
    evidence: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate order, schema, masks, and missingness."""

    errors: list[str] = []
    if list(evidence.columns) != list(REVIEW_NEIGHBORHOOD_UNIT_COLUMNS):
        errors.append(
            "ordered_unit_columns_mismatch="
            f"{list(evidence.columns)!r}:"
            f"{list(REVIEW_NEIGHBORHOOD_UNIT_COLUMNS)!r}"
        )
    if evidence.empty:
        errors.append("unit_evidence_empty")
    if "temporal_unit_key" in evidence:
        keys = evidence["temporal_unit_key"].fillna("").astype(str)
        if keys.str.strip().eq("").any():
            errors.append("blank_temporal_unit_key")
        if keys.duplicated().any():
            errors.append("duplicate_temporal_unit_key")
    expected_metadata = {
        **canonical_review_neighborhood_schema_payload(),
        "schema_hash": REVIEW_NEIGHBORHOOD_SCHEMA_HASH,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            errors.append(f"metadata_{field}_mismatch")
    try:
        _validate_hashes(
            str(metadata.get("producer_sha", "")),
            metadata.get("input_sha256", {}),
        )
    except (ReviewNeighborhoodEvidenceError, AttributeError):
        errors.append("invalid_producer_or_input_hash_binding")

    if set(REVIEW_NEIGHBORHOOD_UNIT_COLUMNS).issubset(evidence.columns):
        observed = pd.to_numeric(
            evidence["observed_frame_count"],
            errors="coerce",
        )
        valid_frames = pd.to_numeric(
            evidence["frames_with_valid_neighbors"],
            errors="coerce",
        )
        max_neighbors = pd.to_numeric(
            evidence["max_valid_neighbor_count"],
            errors="coerce",
        )
        available = _to_bool(
            evidence["neighborhood_evidence_available"]
        )
        if (
            observed.isna().any()
            or observed.le(0).any()
            or valid_frames.isna().any()
            or valid_frames.lt(0).any()
            or valid_frames.gt(observed).any()
        ):
            errors.append("invalid_frame_counts")
        if (
            max_neighbors.isna().any()
            or max_neighbors.lt(0).any()
            or max_neighbors.gt(REVIEW_NEIGHBORHOOD_MAX_NEIGHBORS).any()
        ):
            errors.append("invalid_neighbor_counts")
        expected_available = valid_frames.gt(0).to_numpy()
        if not np.array_equal(available, expected_available):
            errors.append("availability_mask_count_mismatch")
        expected_ratio = (
            valid_frames / observed.replace(0, np.nan)
        ).fillna(0.0)
        actual_ratio = pd.to_numeric(
            evidence["neighbor_valid_ratio"],
            errors="coerce",
        )
        if not np.allclose(actual_ratio, expected_ratio):
            errors.append("neighbor_valid_ratio_mismatch")
        ratio_fields = (
            "neighbor_valid_ratio",
            "any_contact_proxy_ratio",
            "near_neighbor_ratio",
            "overlap_present_ratio",
            "crowding_ratio",
        )
        for field in ratio_fields:
            values = pd.to_numeric(evidence[field], errors="coerce")
            if values.isna().any() or values.lt(0).any() or values.gt(1).any():
                errors.append(f"invalid_ratio={field}")
        distance_fields = (
            "min_edge_distance_over_unit",
            "median_min_edge_distance",
            "min_center_distance_over_unit",
            "median_min_center_distance",
        )
        for field in distance_fields:
            values = pd.to_numeric(evidence[field], errors="coerce")
            if values[available].isna().any() or values[available].lt(0).any():
                errors.append(f"invalid_available_distance={field}")
            if values[~available].notna().any():
                errors.append(f"unavailable_distance_not_nan={field}")
        unavailable = ~available
        zero_when_unavailable = (
            "frames_with_valid_neighbors",
            "max_valid_neighbor_count",
            "neighbor_valid_ratio",
            "any_contact_proxy_ratio",
            "near_neighbor_ratio",
            "max_concurrent_contact_proxy_count",
            "overlap_present_ratio",
            "crowding_ratio",
        )
        for field in zero_when_unavailable:
            values = pd.to_numeric(evidence[field], errors="coerce")
            if values[unavailable].ne(0).any():
                errors.append(f"unavailable_placeholder_nonzero={field}")
        reasons = evidence[
            "neighborhood_evidence_availability_reason"
        ].astype(str)
        expected_reasons = np.where(
            available,
            "available",
            "no_valid_neighbor_observation",
        )
        if not np.array_equal(reasons.to_numpy(), expected_reasons):
            errors.append("availability_reason_mismatch")
    forbidden_columns = [
        column
        for column in evidence.columns
        if any(
            token in column.casefold()
            for token in (
                "behavior",
                "decision",
                "candidate",
                "partner_id",
                "track_id",
                "video",
                "source",
                "date",
            )
        )
    ]
    if forbidden_columns:
        errors.append(f"forbidden_evidence_columns={forbidden_columns}")
    return {
        "schema_id": REVIEW_NEIGHBORHOOD_SCHEMA_ID,
        "schema_version": REVIEW_NEIGHBORHOOD_SCHEMA_VERSION,
        "schema_hash": REVIEW_NEIGHBORHOOD_SCHEMA_HASH,
        "row_count": int(len(evidence)),
        "errors": errors,
        "valid": not errors,
    }


def require_review_neighborhood_evidence(
    evidence: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a valid audit or fail closed."""

    audit = audit_review_neighborhood_evidence(evidence, metadata)
    if audit["errors"]:
        raise ReviewNeighborhoodEvidenceError(
            "review-neighborhood evidence failed: "
            + "; ".join(audit["errors"])
        )
    return audit


__all__ = [
    "REVIEW_NEIGHBORHOOD_CONTACT_IOU_PROXY_THRESHOLD",
    "REVIEW_NEIGHBORHOOD_CONTACT_OVERLAP_PROXY_THRESHOLD",
    "REVIEW_NEIGHBORHOOD_FRAME_COLUMNS",
    "REVIEW_NEIGHBORHOOD_FRAME_FIELDS",
    "REVIEW_NEIGHBORHOOD_MAX_NEIGHBORS",
    "REVIEW_NEIGHBORHOOD_METRIC_VERSION",
    "REVIEW_NEIGHBORHOOD_NEAR_EDGE_DISTANCE_N",
    "REVIEW_NEIGHBORHOOD_RESET_SCOPE",
    "REVIEW_NEIGHBORHOOD_SCHEMA_HASH",
    "REVIEW_NEIGHBORHOOD_SCHEMA_ID",
    "REVIEW_NEIGHBORHOOD_SCHEMA_VERSION",
    "REVIEW_NEIGHBORHOOD_UNIT_COLUMNS",
    "REVIEW_NEIGHBORHOOD_UNIT_FIELDS",
    "ReviewNeighborhoodEvidenceError",
    "ReviewNeighborhoodEvidenceResult",
    "audit_review_neighborhood_evidence",
    "build_review_neighborhood_evidence",
    "canonical_review_neighborhood_schema_payload",
    "require_review_neighborhood_evidence",
    "review_neighborhood_schema_hash",
]
