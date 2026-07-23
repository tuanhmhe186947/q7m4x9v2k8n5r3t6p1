"""Shared scientific authorities for Classification V2 spatial semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

AXIS_DISTANCE_METRIC_ID = "image_axis_normalized_distance"
AXIS_DISTANCE_METRIC_VERSION = (
    "classification_v2.image_axis_normalized_distance.v1"
)
DIAGONAL_DISTANCE_METRIC_ID = "image_diagonal_normalized_distance"
DIAGONAL_DISTANCE_METRIC_VERSION = (
    "classification_v2.image_diagonal_normalized_distance.v1"
)

SOCIAL_IDENTITY_FIELDS: tuple[str, ...] = (
    "object_track_key",
    "track_id",
    "object_id",
)
SOCIAL_IDENTITY_VERSION = "classification_v2.social_identity.v1"
SOCIAL_TIE_BREAK_VERSION = "classification_v2.social_tie_break.v1"
SOCIAL_TIE_BREAK_RULE = (
    "axis_distance_ascending_then_canonical_partner_key_ascending"
)

SOCIAL_NEAR_THRESHOLD_ID = "social_near_distance_n_v1"
SOCIAL_NEAR_THRESHOLD_VALUE = 0.08
SOCIAL_NEAR_THRESHOLD_UNITS = "axis_normalized_image_distance"

ROI_AGGREGATION_VERSION = (
    "classification_v2.roi_aggregation.available_frames.v1"
)
ROI_NEAR_THRESHOLD_ID = "roi_near_distance_diagonal_n_v1"
ROI_NEAR_THRESHOLD_VALUE = 0.08
ROI_CONTACT_THRESHOLD_ID = "roi_contact_distance_diagonal_n_v1"
ROI_CONTACT_THRESHOLD_VALUE = 0.02
ROI_DISTANCE_THRESHOLD_UNITS = "diagonal_normalized_image_distance"
ROI_TARGET_MODEL_POLICY_VERSION = (
    "classification_v2.target_roi_model_forbidden.v1"
)
TARGET_ROI_MODEL_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "target_roi_",
    "roi_target_",
)
TARGET_ROI_MODEL_FORBIDDEN_EXACT: frozenset[str] = frozenset(
    {
        "label_selected_roi_class_indicator",
        "target_roi_contact",
        "target_roi_distance",
        "target_roi_contact_ratio_unit",
    }
)
TARGET_ROI_MODEL_FORBIDDEN_REASON = (
    "behavior-selected target ROI is label-derived review evidence"
)


@dataclass(frozen=True, slots=True)
class DistanceMetricContract:
    """Machine-readable contract for one image-coordinate distance metric."""

    metric_id: str
    version: str
    formula: str
    coordinate_system: str
    units: str
    isotropic_in_pixel_space: bool
    is_physical_measurement: bool = False


AXIS_DISTANCE_CONTRACT = DistanceMetricContract(
    metric_id=AXIS_DISTANCE_METRIC_ID,
    version=AXIS_DISTANCE_METRIC_VERSION,
    formula="sqrt((dx_px/image_width_px)^2+(dy_px/image_height_px)^2)",
    coordinate_system="image_pixel_axes_normalized_separately",
    units="axis_normalized_image_distance",
    isotropic_in_pixel_space=False,
)
DIAGONAL_DISTANCE_CONTRACT = DistanceMetricContract(
    metric_id=DIAGONAL_DISTANCE_METRIC_ID,
    version=DIAGONAL_DISTANCE_METRIC_VERSION,
    formula=(
        "sqrt(dx_px^2+dy_px^2)"
        "/sqrt(image_width_px^2+image_height_px^2)"
    ),
    coordinate_system="image_pixel_euclidean_normalized_by_diagonal",
    units="diagonal_normalized_image_distance",
    isotropic_in_pixel_space=True,
)


def distance_metric_registry() -> list[dict[str, Any]]:
    """Return the authoritative ordered distance-metric registry."""

    return [
        asdict(AXIS_DISTANCE_CONTRACT),
        asdict(DIAGONAL_DISTANCE_CONTRACT),
    ]


def axis_normalized_image_distance(
    dx_px: float,
    dy_px: float,
    image_width_px: float,
    image_height_px: float,
) -> float:
    """Return separately axis-normalized image distance or NaN."""

    values = (dx_px, dy_px, image_width_px, image_height_px)
    if not all(math.isfinite(float(value)) for value in values):
        return math.nan
    if image_width_px <= 0 or image_height_px <= 0:
        return math.nan
    return math.hypot(
        float(dx_px) / float(image_width_px),
        float(dy_px) / float(image_height_px),
    )


def diagonal_normalized_image_distance(
    dx_px: float,
    dy_px: float,
    image_width_px: float,
    image_height_px: float,
) -> float:
    """Return isotropic diagonal-normalized image distance or NaN."""

    values = (dx_px, dy_px, image_width_px, image_height_px)
    if not all(math.isfinite(float(value)) for value in values):
        return math.nan
    diagonal = math.hypot(float(image_width_px), float(image_height_px))
    if image_width_px <= 0 or image_height_px <= 0 or diagonal <= 0:
        return math.nan
    return math.hypot(float(dx_px), float(dy_px)) / diagonal


def pairwise_image_distance_matrices(
    centers_px: np.ndarray,
    image_widths_px: np.ndarray,
    image_heights_px: np.ndarray,
    row_valid: np.ndarray,
    identity_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise metrics and validity for one scene frame."""

    centers = np.asarray(centers_px, dtype="float64")
    widths = np.asarray(image_widths_px, dtype="float64")
    heights = np.asarray(image_heights_px, dtype="float64")
    valid = np.asarray(row_valid, dtype=bool)
    count = len(centers)
    if centers.shape != (count, 2):
        raise ValueError("centers_px must have shape (n, 2)")
    if len(widths) != count or len(heights) != count:
        raise ValueError("image-dimension arrays must match centers")
    if len(identity_keys) != count:
        raise ValueError("identity_keys must match centers")

    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    same_dimensions = (
        np.isclose(widths[:, None], widths[None, :], rtol=0.0, atol=0.0)
        & np.isclose(
            heights[:, None],
            heights[None, :],
            rtol=0.0,
            atol=0.0,
        )
    )
    dimension_valid = (
        np.isfinite(widths)
        & np.isfinite(heights)
        & (widths > 0)
        & (heights > 0)
    )
    identities = np.asarray(identity_keys, dtype=str)
    distinct_identity = (
        identities[:, None] != identities[None, :]
    ) & (identities[:, None] != "") & (identities[None, :] != "")
    pair_valid = (
        valid[:, None]
        & valid[None, :]
        & dimension_valid[:, None]
        & dimension_valid[None, :]
        & same_dimensions
        & distinct_identity
    )

    axis = np.hypot(
        np.divide(
            dx,
            widths[:, None],
            out=np.full_like(dx, np.nan),
            where=widths[:, None] > 0,
        ),
        np.divide(
            dy,
            heights[:, None],
            out=np.full_like(dy, np.nan),
            where=heights[:, None] > 0,
        ),
    )
    diagonals = np.hypot(widths, heights)
    diagonal = np.divide(
        np.hypot(dx, dy),
        diagonals[:, None],
        out=np.full_like(dx, np.nan),
        where=diagonals[:, None] > 0,
    )
    axis[~pair_valid] = np.nan
    diagonal[~pair_valid] = np.nan
    return axis, diagonal, pair_valid


def canonical_social_identity(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    """Return canonical stable identity and selected hierarchy field."""

    for field in SOCIAL_IDENTITY_FIELDS:
        value = _stable_text(row.get(field))
        if not value:
            continue
        if field == "object_track_key":
            return value, field
        scope = "|".join(
            (
                f"source={_stable_text(row.get('source_type'))}",
                f"dataset={_stable_text(row.get('dataset_id'))}",
                f"video={_stable_text(row.get('video_key'))}",
            )
        )
        return f"{scope}|{field}={value}", field
    return "", ""


def canonical_social_identity_columns(
    rows: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Vectorize canonical social identity resolution."""

    resolved = [
        canonical_social_identity(record)
        for record in rows.to_dict(orient="records")
    ]
    return (
        pd.Series(
            [item[0] for item in resolved],
            index=rows.index,
            dtype="string",
        ).fillna(""),
        pd.Series(
            [item[1] for item in resolved],
            index=rows.index,
            dtype="string",
        ).fillna(""),
    )


def is_target_roi_model_forbidden(column: str) -> bool:
    """Return whether a label-selected ROI feature is forbidden from model X."""

    normalized = str(column).strip().lower()
    return normalized in TARGET_ROI_MODEL_FORBIDDEN_EXACT or normalized.startswith(
        TARGET_ROI_MODEL_FORBIDDEN_PREFIXES
    )


def target_roi_policy_metadata(column: str) -> dict[str, Any]:
    """Return explicit review/model/leakage policy for a target ROI feature."""

    forbidden = is_target_roi_model_forbidden(column)
    return {
        "feature_name": column,
        "review_eligible": True,
        "model_eligible": not forbidden,
        "model_forbidden_reason": (
            TARGET_ROI_MODEL_FORBIDDEN_REASON if forbidden else ""
        ),
        "leakage_risk": "CRITICAL_LABEL_SELECTED" if forbidden else "LOW",
        "semantics_version": ROI_TARGET_MODEL_POLICY_VERSION,
    }


def _stable_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "<na>"} else text


__all__ = [
    "AXIS_DISTANCE_CONTRACT",
    "AXIS_DISTANCE_METRIC_ID",
    "AXIS_DISTANCE_METRIC_VERSION",
    "DIAGONAL_DISTANCE_CONTRACT",
    "DIAGONAL_DISTANCE_METRIC_ID",
    "DIAGONAL_DISTANCE_METRIC_VERSION",
    "ROI_AGGREGATION_VERSION",
    "ROI_CONTACT_THRESHOLD_ID",
    "ROI_CONTACT_THRESHOLD_VALUE",
    "ROI_DISTANCE_THRESHOLD_UNITS",
    "ROI_NEAR_THRESHOLD_ID",
    "ROI_NEAR_THRESHOLD_VALUE",
    "ROI_TARGET_MODEL_POLICY_VERSION",
    "SOCIAL_IDENTITY_FIELDS",
    "SOCIAL_IDENTITY_VERSION",
    "SOCIAL_NEAR_THRESHOLD_ID",
    "SOCIAL_NEAR_THRESHOLD_UNITS",
    "SOCIAL_NEAR_THRESHOLD_VALUE",
    "SOCIAL_TIE_BREAK_RULE",
    "SOCIAL_TIE_BREAK_VERSION",
    "TARGET_ROI_MODEL_FORBIDDEN_EXACT",
    "TARGET_ROI_MODEL_FORBIDDEN_PREFIXES",
    "TARGET_ROI_MODEL_FORBIDDEN_REASON",
    "axis_normalized_image_distance",
    "canonical_social_identity",
    "canonical_social_identity_columns",
    "diagonal_normalized_image_distance",
    "distance_metric_registry",
    "is_target_roi_model_forbidden",
    "pairwise_image_distance_matrices",
    "target_roi_policy_metadata",
]
