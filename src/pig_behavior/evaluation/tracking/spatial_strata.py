"""Mask- and perspective-aware strata for tracking diagnostics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PERSPECTIVE_AXIS = "pen_relative_x_left_near_right_far_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SpatialStrataThresholds:
    """Predeclared thresholds for spatial tracking diagnostics."""

    near_wall_distance_bbox_scale: float = 0.25
    far_pen_relative_x: float = 0.67
    absolute_small_area_ratio: float = 0.02
    perspective_small_quantile: float = 0.10

    def __post_init__(self) -> None:
        unit_interval_values = {
            "near_wall_distance_bbox_scale": (
                self.near_wall_distance_bbox_scale
            ),
            "far_pen_relative_x": self.far_pen_relative_x,
            "absolute_small_area_ratio": self.absolute_small_area_ratio,
            "perspective_small_quantile": self.perspective_small_quantile,
        }
        for name, value in unit_interval_values.items():
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class SpatialSceneContext:
    """Binary pen mask and its distance-to-wall transform."""

    mask_path: Path
    pen_mask: np.ndarray
    wall_distance_px: np.ndarray

    @property
    def height(self) -> int:
        return int(self.pen_mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.pen_mask.shape[1])


def load_spatial_scene_context(mask_path: Path) -> SpatialSceneContext:
    """Load one binary pen mask and calculate distance to its boundary."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("OpenCV is required for mask spatial strata.") from exc

    resolved = Path(mask_path).resolve()
    mask = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read spatial mask: {resolved}")
    pen_mask = mask >= 128
    if not np.any(pen_mask):
        raise ValueError(f"Spatial mask has no pen pixels: {resolved}")
    wall_distance = cv2.distanceTransform(
        pen_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    return SpatialSceneContext(
        mask_path=resolved,
        pen_mask=pen_mask,
        wall_distance_px=wall_distance,
    )


def _clipped_bbox_indices(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def _pen_relative_x(
    context: SpatialSceneContext,
    center_x: float,
    center_y: float,
) -> float:
    row_index = max(0, min(context.height - 1, int(round(center_y))))
    pen_columns = np.flatnonzero(context.pen_mask[row_index])
    if pen_columns.size < 2:
        pen_columns = np.flatnonzero(np.any(context.pen_mask, axis=0))
    if pen_columns.size < 2:
        return 0.5
    left = float(pen_columns[0])
    right = float(pen_columns[-1])
    relative = (center_x - left) / max(right - left, 1.0)
    return float(np.clip(relative, 0.0, 1.0))


def spatial_features_for_bbox(
    context: SpatialSceneContext,
    bbox: tuple[float, float, float, float],
    thresholds: SpatialStrataThresholds,
) -> dict[str, float | bool | str]:
    """Return wall, far-camera proxy, and absolute-size features."""
    x1, y1, x2, y2 = bbox
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    bbox_area = bbox_width * bbox_height
    if bbox_area <= 0.0:
        raise ValueError(f"Spatial strata received invalid bbox: {bbox}")

    left, top, right, bottom = _clipped_bbox_indices(
        bbox,
        context.width,
        context.height,
    )
    bbox_mask = context.pen_mask[top:bottom, left:right]
    bbox_wall_distance = context.wall_distance_px[top:bottom, left:right]
    wall_distance_px = float(np.quantile(bbox_wall_distance, 0.10))
    bbox_scale = math.sqrt(bbox_area)
    wall_distance_bbox_scale = wall_distance_px / max(bbox_scale, 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    pen_relative_x = _pen_relative_x(context, center_x, center_y)
    bbox_area_ratio = bbox_area / float(context.width * context.height)

    return {
        "perspective_axis": PERSPECTIVE_AXIS,
        "pen_relative_x": round(pen_relative_x, 6),
        "far_camera_score": round(pen_relative_x, 6),
        "is_far_camera_proxy": (
            pen_relative_x >= thresholds.far_pen_relative_x
        ),
        "bbox_area_ratio": round(bbox_area_ratio, 8),
        "is_absolute_small": (
            bbox_area_ratio <= thresholds.absolute_small_area_ratio
        ),
        "wall_distance_px": round(wall_distance_px, 4),
        "wall_distance_bbox_scale": round(wall_distance_bbox_scale, 6),
        "is_near_wall": (
            wall_distance_bbox_scale
            <= thresholds.near_wall_distance_bbox_scale
        ),
        "bbox_pen_mask_cover_ratio": round(float(np.mean(bbox_mask)), 6),
    }


def calibrate_perspective_small(
    rows: list[dict[str, Any]],
    thresholds: SpatialStrataThresholds,
) -> dict[str, int | float | str]:
    """Fit expected GT bbox size along the camera perspective axis."""
    valid_rows = [
        row
        for row in rows
        if float(row.get("bbox_area_ratio", 0.0)) > 0.0
        and math.isfinite(float(row.get("pen_relative_x", math.nan)))
    ]
    if not valid_rows:
        return {
            "sample_count": 0,
            "perspective_axis": PERSPECTIVE_AXIS,
            "slope": 0.0,
            "intercept": 0.0,
            "residual_threshold": 0.0,
            "small_quantile": thresholds.perspective_small_quantile,
        }

    x_values = np.asarray(
        [float(row["pen_relative_x"]) for row in valid_rows],
        dtype=np.float64,
    )
    log_areas = np.log(
        np.asarray(
            [float(row["bbox_area_ratio"]) for row in valid_rows],
            dtype=np.float64,
        )
    )
    if float(np.ptp(x_values)) > 1e-9:
        slope, intercept = np.polyfit(x_values, log_areas, 1)
    else:
        slope = 0.0
        intercept = float(np.median(log_areas))
    residuals = log_areas - (slope * x_values + intercept)
    residual_threshold = float(
        np.quantile(residuals, thresholds.perspective_small_quantile)
    )

    for row, expected_log_area, residual in zip(
        valid_rows,
        slope * x_values + intercept,
        residuals,
        strict=True,
    ):
        row["perspective_expected_log_area"] = round(
            float(expected_log_area),
            8,
        )
        row["perspective_area_log_residual"] = round(float(residual), 8)
        row["is_perspective_residual_small"] = (
            float(residual) <= residual_threshold
        )

    return {
        "sample_count": len(valid_rows),
        "perspective_axis": PERSPECTIVE_AXIS,
        "slope": round(float(slope), 8),
        "intercept": round(float(intercept), 8),
        "residual_threshold": round(residual_threshold, 8),
        "small_quantile": thresholds.perspective_small_quantile,
    }


def summarize_spatial_strata(
    rows: list[dict[str, Any]],
    thresholds: SpatialStrataThresholds,
    calibration: dict[str, int | float | str],
    context: SpatialSceneContext,
    *,
    source_match_iou_threshold: float = 0.30,
    quality_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Summarize identity, missing, and bbox quality by spatial stratum."""

    for name, value in {
        "source_match_iou_threshold": source_match_iou_threshold,
        "quality_iou_threshold": quality_iou_threshold,
    }.items():
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    def summarize(flag: str | None) -> dict[str, int | float]:
        selected = (
            rows
            if flag is None
            else [row for row in rows if bool(row.get(flag, False))]
        )
        matched = [row for row in selected if bool(row.get("is_matched"))]
        correct = [row for row in matched if bool(row.get("is_id_correct"))]
        wrong = [row for row in matched if bool(row.get("is_id_wrong"))]
        missing = [row for row in selected if bool(row.get("is_missing"))]
        try:
            matched_ious = np.asarray(
                [float(row["matched_iou"]) for row in matched],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Matched spatial rows require numeric matched_iou values."
            ) from exc
        if not np.all(np.isfinite(matched_ious)):
            raise ValueError(
                "Matched spatial rows require finite matched_iou values."
            )
        quality_match_count = int(
            np.count_nonzero(matched_ious >= quality_iou_threshold)
        )
        low_iou_match_count = len(matched) - quality_match_count

        def iou_statistic(kind: str) -> float:
            if not matched_ious.size:
                return 0.0
            if kind == "mean":
                value = float(np.mean(matched_ious))
            elif kind == "median":
                value = float(np.median(matched_ious))
            else:
                value = float(np.quantile(matched_ious, 0.10))
            return round(value, 6)

        return {
            "instance_count": len(selected),
            "matched_count": len(matched),
            "id_accuracy": round(
                len(correct) / len(matched),
                6,
            )
            if matched
            else 0.0,
            "wrong_id_rate": round(
                len(wrong) / len(matched),
                6,
            )
            if matched
            else 0.0,
            "missing_rate": round(
                len(missing) / len(selected),
                6,
            )
            if selected
            else 0.0,
            "mean_matched_iou": iou_statistic("mean"),
            "median_matched_iou": iou_statistic("median"),
            "p10_matched_iou": iou_statistic("p10"),
            "quality_match_count": quality_match_count,
            "quality_match_rate": round(
                quality_match_count / len(selected),
                6,
            )
            if selected
            else 0.0,
            "low_iou_match_count": low_iou_match_count,
            "low_iou_match_rate": round(
                low_iou_match_count / len(selected),
                6,
            )
            if selected
            else 0.0,
        }

    return {
        "enabled": True,
        "mask_path": str(context.mask_path),
        "mask_sha256": _file_sha256(context.mask_path),
        "mask_width": context.width,
        "mask_height": context.height,
        "perspective_axis": PERSPECTIVE_AXIS,
        "geometry_quality_contract": {
            "source_match_iou_threshold": source_match_iou_threshold,
            "quality_iou_threshold": quality_iou_threshold,
        },
        "thresholds": {
            "near_wall_distance_bbox_scale": (
                thresholds.near_wall_distance_bbox_scale
            ),
            "far_pen_relative_x": thresholds.far_pen_relative_x,
            "absolute_small_area_ratio": (
                thresholds.absolute_small_area_ratio
            ),
            "perspective_small_quantile": (
                thresholds.perspective_small_quantile
            ),
        },
        "perspective_size_calibration": calibration,
        "all_instances": summarize(None),
        "near_wall": summarize("is_near_wall"),
        "far_camera_proxy": summarize("is_far_camera_proxy"),
        "absolute_small": summarize("is_absolute_small"),
        "perspective_residual_small": summarize(
            "is_perspective_residual_small"
        ),
    }


__all__ = [
    "PERSPECTIVE_AXIS",
    "SpatialSceneContext",
    "SpatialStrataThresholds",
    "calibrate_perspective_small",
    "load_spatial_scene_context",
    "spatial_features_for_bbox",
    "summarize_spatial_strata",
]
