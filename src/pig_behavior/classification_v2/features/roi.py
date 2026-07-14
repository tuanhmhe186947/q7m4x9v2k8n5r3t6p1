"""ROI feature builder for classification_v2.

This module reads static scene ROI annotations from COCO JSON and computes
actor-to-ROI spatial features for each frame-object row.

The ROI background image contains no pigs. That is expected: it defines
fixed regions of the pen such as feeder, drinker and toy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import scene_frame_key
from pig_behavior.classification_v2.contracts.lineage_claims import (
    add_optional_lineage_claims_to_audit,
    require_lineage_claims_preserved,
    resolve_optional_lineage_claims,
)
from pig_behavior.classification_v2.schema import ROI_DOMINANT_BEHAVIORS

ROI_CLASSES: tuple[str, ...] = ("feeder", "drinker", "toy")

BEHAVIOR_TO_TARGET_ROI: dict[str, str] = {
    "eat": "feeder",
    "drink": "drinker",
    "playwithtoy": "toy",
}

REQUIRED_ROI_INPUT_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "bbox_valid",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "bbox_area",
    "image_width",
    "image_height",
)


@dataclass(frozen=True)
class SceneROI:
    roi_id: int
    category: str
    image_width: float
    image_height: float
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: np.ndarray


def load_scene_rois_from_coco(coco_path: Path) -> list[SceneROI]:
    """Load feeder/drinker/toy ROI annotations from COCO JSON."""
    if not coco_path.exists():
        raise FileNotFoundError(coco_path)

    data = json.loads(coco_path.read_text(encoding="utf-8"))

    categories = {
        int(cat["id"]): str(cat["name"]).strip().lower()
        for cat in data.get("categories", [])
    }

    images = {
        int(img["id"]): img
        for img in data.get("images", [])
    }

    rois: list[SceneROI] = []

    for ann in data.get("annotations", []):
        category_id = int(ann.get("category_id"))
        category = categories.get(category_id, "").strip().lower()

        if category not in ROI_CLASSES:
            continue

        image_id = int(ann.get("image_id"))
        image = images.get(image_id)
        if image is None:
            raise ValueError(f"COCO annotation references unknown image_id={image_id}")

        image_width = float(image["width"])
        image_height = float(image["height"])

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            raise ValueError(f"Invalid ROI bbox in annotation id={ann.get('id')}")

        x, y, w, h = [float(v) for v in bbox]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        polygon = _polygon_from_segmentation(ann.get("segmentation"), x1, y1, x2, y2)

        rois.append(
            SceneROI(
                roi_id=int(ann.get("id")),
                category=category,
                image_width=image_width,
                image_height=image_height,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                polygon=polygon,
            )
        )

    if not rois:
        raise ValueError(f"No feeder/drinker/toy ROI annotations found in {coco_path}")

    return rois


def build_roi_features(
    frame_features: pd.DataFrame,
    *,
    roi_coco_path: Path,
    near_distance_n: float = 0.08,
    contact_distance_n: float = 0.02,
) -> pd.DataFrame:
    """Add static scene ROI features to frame-object rows.

    Parameters
    ----------
    frame_features:
        DataFrame after geometry feature step.
    roi_coco_path:
        COCO JSON containing static scene ROI annotations.
    near_distance_n:
        Normalized bbox-to-ROI distance threshold for "near".
    contact_distance_n:
        Normalized bbox-to-ROI distance threshold for "contact".
    """
    resolve_optional_lineage_claims(
        frame_features,
        artifact_name="ROI input",
    )
    missing = [
        col for col in REQUIRED_ROI_INPUT_COLUMNS if col not in frame_features.columns
    ]
    if missing:
        raise ValueError(f"Missing ROI input columns: {missing}")

    rois = load_scene_rois_from_coco(roi_coco_path)
    out = frame_features.copy()

    for col in [
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "bbox_area",
        "image_width",
        "image_height",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bbox_valid = _to_bool_series(out["bbox_valid"])
    image_diag = np.sqrt(out["image_width"] ** 2 + out["image_height"] ** 2).replace(
        0,
        np.nan,
    )

    for roi_class in ROI_CLASSES:
        out[f"roi_{roi_class}_available"] = False
        out[f"roi_{roi_class}_nearest_id"] = ""
        out[f"roi_{roi_class}_min_dist_px"] = np.inf
        out[f"roi_{roi_class}_min_dist_n"] = np.inf
        out[f"roi_{roi_class}_max_overlap_ratio"] = 0.0
        out[f"roi_{roi_class}_max_iou"] = 0.0
        out[f"roi_{roi_class}_center_inside"] = False

    for roi in rois:
        cls = roi.category

        sx = out["image_width"] / roi.image_width
        sy = out["image_height"] / roi.image_height

        rx1 = roi.x1 * sx
        ry1 = roi.y1 * sy
        rx2 = roi.x2 * sx
        ry2 = roi.y2 * sy

        inter_w = np.maximum(
            0.0,
            np.minimum(out["x2"], rx2) - np.maximum(out["x1"], rx1),
        )
        inter_h = np.maximum(
            0.0,
            np.minimum(out["y2"], ry2) - np.maximum(out["y1"], ry1),
        )
        inter_area = inter_w * inter_h

        roi_area = np.maximum(1.0, (rx2 - rx1) * (ry2 - ry1))
        actor_area = out["bbox_area"].replace(0, np.nan)

        overlap_ratio = (inter_area / actor_area).fillna(0.0)
        iou = (inter_area / (actor_area + roi_area - inter_area).replace(0, np.nan)).fillna(
            0.0
        )

        dx = np.maximum.reduce(
            [
                rx1 - out["x2"],
                out["x1"] - rx2,
                np.zeros(len(out), dtype=float),
            ]
        )
        dy = np.maximum.reduce(
            [
                ry1 - out["y2"],
                out["y1"] - ry2,
                np.zeros(len(out), dtype=float),
            ]
        )

        dist_px = np.sqrt(dx**2 + dy**2)
        dist_n = dist_px / image_diag

        cx_original = out["cx"] / sx.replace(0, np.nan)
        cy_original = out["cy"] / sy.replace(0, np.nan)
        center_inside = _points_in_polygon(
            cx_original.to_numpy(dtype=float),
            cy_original.to_numpy(dtype=float),
            roi.polygon,
        )

        current_dist = out[f"roi_{cls}_min_dist_n"]
        better = dist_n.lt(current_dist)

        out.loc[better, f"roi_{cls}_nearest_id"] = str(roi.roi_id)
        out.loc[better, f"roi_{cls}_min_dist_px"] = dist_px[better]
        out.loc[better, f"roi_{cls}_min_dist_n"] = dist_n[better]

        out[f"roi_{cls}_max_overlap_ratio"] = np.maximum(
            out[f"roi_{cls}_max_overlap_ratio"],
            overlap_ratio,
        )
        out[f"roi_{cls}_max_iou"] = np.maximum(out[f"roi_{cls}_max_iou"], iou)
        out[f"roi_{cls}_center_inside"] = (
            _to_bool_series(out[f"roi_{cls}_center_inside"]) | center_inside
        )
        out[f"roi_{cls}_available"] = True

    for roi_class in ROI_CLASSES:
        for col in [
            f"roi_{roi_class}_min_dist_px",
            f"roi_{roi_class}_min_dist_n",
        ]:
            out[col] = out[col].replace(np.inf, np.nan)

        out[f"roi_{roi_class}_near"] = (
            bbox_valid
            & out[f"roi_{roi_class}_available"]
            & out[f"roi_{roi_class}_min_dist_n"].le(near_distance_n)
        )
        out[f"roi_{roi_class}_contact"] = (
            bbox_valid
            & out[f"roi_{roi_class}_available"]
            & (
                out[f"roi_{roi_class}_min_dist_n"].le(contact_distance_n)
                | out[f"roi_{roi_class}_max_overlap_ratio"].gt(0)
                | _to_bool_series(out[f"roi_{roi_class}_center_inside"])
            )
        )

    out["roi_feature_required"] = out["behavior"].isin(ROI_DOMINANT_BEHAVIORS)
    out["roi_target_class"] = out["behavior"].map(BEHAVIOR_TO_TARGET_ROI).fillna("")

    out["roi_target_available"] = False
    out["roi_target_min_dist_n"] = np.nan
    out["roi_target_min_dist_px"] = np.nan
    out["roi_target_max_overlap_ratio"] = 0.0
    out["roi_target_max_iou"] = 0.0
    out["roi_target_center_inside"] = False
    out["roi_target_near"] = False
    out["roi_target_contact"] = False

    for roi_class in ROI_CLASSES:
        target = out["roi_target_class"].eq(roi_class)

        out.loc[target, "roi_target_available"] = out.loc[
            target,
            f"roi_{roi_class}_available",
        ]
        out.loc[target, "roi_target_min_dist_n"] = out.loc[
            target,
            f"roi_{roi_class}_min_dist_n",
        ]
        out.loc[target, "roi_target_min_dist_px"] = out.loc[
            target,
            f"roi_{roi_class}_min_dist_px",
        ]
        out.loc[target, "roi_target_max_overlap_ratio"] = out.loc[
            target,
            f"roi_{roi_class}_max_overlap_ratio",
        ]
        out.loc[target, "roi_target_max_iou"] = out.loc[
            target,
            f"roi_{roi_class}_max_iou",
        ]
        out.loc[target, "roi_target_center_inside"] = out.loc[
            target,
            f"roi_{roi_class}_center_inside",
        ]
        out.loc[target, "roi_target_near"] = out.loc[
            target,
            f"roi_{roi_class}_near",
        ]
        out.loc[target, "roi_target_contact"] = out.loc[
            target,
            f"roi_{roi_class}_contact",
        ]

    out["roi_context_quality"] = "not_required"
    out.loc[
        out["roi_feature_required"] & ~bbox_valid,
        "roi_context_quality",
    ] = "invalid_bbox"
    out.loc[
        out["roi_feature_required"] & bbox_valid & ~out["roi_target_available"],
        "roi_context_quality",
    ] = "missing_target_roi"
    out.loc[
        out["roi_feature_required"]
        & bbox_valid
        & out["roi_target_available"]
        & ~out["roi_target_near"],
        "roi_context_quality",
    ] = "target_roi_far"
    out.loc[
        out["roi_feature_required"] & bbox_valid & out["roi_target_near"],
        "roi_context_quality",
    ] = "target_roi_near"
    out.loc[
        out["roi_feature_required"] & bbox_valid & out["roi_target_contact"],
        "roi_context_quality",
    ] = "target_roi_contact"

    include = (
        _to_bool_series(out["include_in_training"])
        if "include_in_training" in out.columns
        else bbox_valid
    )

    out["use_for_roi_training"] = (
        include
        & bbox_valid
        & out["roi_feature_required"]
        & out["roi_target_available"]
    )

    out["roi_feature_valid"] = (
        ~out["roi_feature_required"]
        | (
            bbox_valid
            & out["roi_feature_required"]
            & out["roi_target_available"]
            & out["roi_target_min_dist_n"].notna()
        )
    )

    require_lineage_claims_preserved(
        frame_features,
        out,
        source_name="ROI input",
        derived_name="ROI output",
    )
    return out


def validate_roi_features(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    required = {
        "roi_feature_required",
        "roi_target_class",
        "roi_target_available",
        "roi_target_min_dist_n",
        "roi_target_near",
        "roi_target_contact",
        "roi_context_quality",
        "roi_feature_valid",
        "use_for_roi_training",
    }

    for roi_class in ROI_CLASSES:
        required.update(
            {
                f"roi_{roi_class}_available",
                f"roi_{roi_class}_min_dist_n",
                f"roi_{roi_class}_max_overlap_ratio",
                f"roi_{roi_class}_max_iou",
                f"roi_{roi_class}_near",
                f"roi_{roi_class}_contact",
            }
        )

    missing = sorted(required.difference(df.columns))
    if missing:
        errors.append(f"missing_columns={missing}")

    roi_required = (
        _to_bool_series(df["roi_feature_required"])
        if "roi_feature_required" in df.columns
        else pd.Series(False, index=df.index)
    )
    roi_valid = (
        _to_bool_series(df["roi_feature_valid"])
        if "roi_feature_valid" in df.columns
        else pd.Series(False, index=df.index)
    )

    required_invalid = int((roi_required & ~roi_valid).sum())
    if required_invalid > 0:
        warnings.append(f"roi_required_invalid_count={required_invalid}")

    audit = {
        "rows": int(len(df)),
        "frames": int(scene_frame_key(df).nunique()),
        "frame_objects": int(df["frame_uid"].nunique())
        if "frame_uid" in df.columns
        else 0,
        "sources": _value_counts_dict(df, "source_type"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "roi_feature_required": _value_counts_dict(df, "roi_feature_required"),
        "roi_target_class": _value_counts_dict(df, "roi_target_class"),
        "roi_target_available": _value_counts_dict(df, "roi_target_available"),
        "roi_target_near": _value_counts_dict(df, "roi_target_near"),
        "roi_target_contact": _value_counts_dict(df, "roi_target_contact"),
        "roi_context_quality": _value_counts_dict(df, "roi_context_quality"),
        "roi_feature_valid": _value_counts_dict(df, "roi_feature_valid"),
        "use_for_roi_training": _value_counts_dict(df, "use_for_roi_training"),
        "required_invalid": required_invalid,
        "errors": errors,
        "warnings": warnings,
    }
    return add_optional_lineage_claims_to_audit(
        audit,
        df,
        artifact_name="ROI audit frame table",
    )


def _polygon_from_segmentation(
    segmentation: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    if isinstance(segmentation, list) and segmentation:
        first = segmentation[0]
        if isinstance(first, list) and len(first) >= 6:
            pts = np.asarray(first, dtype=float).reshape(-1, 2)
            return pts

    return np.asarray(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=float,
    )


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon."""
    if len(polygon) < 3:
        return np.zeros_like(xs, dtype=bool)

    inside = np.zeros_like(xs, dtype=bool)
    xj, yj = polygon[-1]

    for xi, yi in polygon:
        intersects = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersects
        xj, yj = xi, yi

    return inside


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)

    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", ""}

    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return False

    return series.map(parse).astype(bool)


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}
