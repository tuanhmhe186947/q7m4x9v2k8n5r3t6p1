"""YOLO result parsing and detection filtering for pig tracking."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.geometry import (
    bbox_center,
    bbox_iom,
    bbox_iou,
    center_distance_norm,
    clip_box,
)
from pig_behavior.tracking.masks import detection_overlap_score, roi_keep
from pig_behavior.tracking.schemas import Detection

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def _result_masks(
    result: Any,
    width: int,
    height: int,
) -> list[np.ndarray | None]:
    masks = getattr(result, "masks", None)
    data = _to_numpy(getattr(masks, "data", None)) if masks is not None else None
    if data is None or len(data) == 0:
        return []

    import cv2

    out: list[np.ndarray | None] = []
    for mask_values in data:
        mask = np.asarray(mask_values)
        if mask.ndim != 2:
            out.append(None)
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        out.append(mask > 0.5)
    return out


def extract_hist_hsv(
    frame: np.ndarray,
    box: np.ndarray,
    *,
    foreground_core: bool = False,
) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)

    if foreground_core:
        inset_x = int(round((x2 - x1) * 0.10))
        inset_y = int(round((y2 - y1) * 0.10))
        if x2 - x1 > 2 * inset_x and y2 - y1 > 2 * inset_y:
            x1 += inset_x
            x2 -= inset_x
            y1 += inset_y
            y2 -= inset_y

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_mask = None
    if foreground_core:
        hist_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.ellipse(hist_mask, (48, 48), (43, 38), 0, 0, 360, 255, -1)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        hist_mask,
        [16, 16, 4],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist.flatten()


def hist_distance(first: np.ndarray | None, second: np.ndarray) -> float:
    if first is None:
        return 0.50
    return float(np.clip(1.0 - np.sum(np.sqrt(first * second)), 0.0, 1.0))


def deduplicate_detections(
    detections: list[Detection],
    cfg: TrackingConfig,
    width: int,
    height: int,
    frame_id: int | None = None,
) -> list[Detection]:
    """Remove highly overlapping or contained duplicate detections.
    
    Combines IoU with center distance, and containment with center distance
    to avoid dropping two real pigs overlapping each other.
    """
    kept: list[Detection] = []
    for det in detections:
        is_dup = False
        for other in kept:
            iou = bbox_iou(det.box, other.box)
            iom = bbox_iom(det.box, other.box)
            center_dist = center_distance_norm(det.box, other.box, width, height)

            # Calculate area ratio
            det_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
            other_area = (other.box[2] - other.box[0]) * (other.box[3] - other.box[1])
            min_area = min(det_area, other_area)
            max_area = max(det_area, other_area)
            area_ratio = min_area / max(max_area, 1e-6)

            # High IoU plus similar center and area confirms a duplicate.
            if (
                iou > cfg.dup_iou_threshold
                and center_dist < cfg.dup_center_threshold
                and area_ratio >= cfg.dup_area_ratio_threshold
            ):
                logger.debug(
                    "Deduplication drop [duplicate_iou] at frame %s: "
                    "det %s dropped because of other %s. "
                    "iou=%.3f, center_dist=%.3f, area_ratio=%.3f",
                    frame_id,
                    det.box.tolist(),
                    other.box.tolist(),
                    iou,
                    center_dist,
                    area_ratio,
                )
                is_dup = True
                break

            # Strong containment plus a similar center confirms a duplicate.
            if (
                iom > cfg.dup_containment_threshold
                and center_dist < cfg.dup_center_threshold
            ):
                logger.debug(
                    "Deduplication drop [duplicate_containment] at frame %s: "
                    "det %s dropped because of other %s. "
                    "iom=%.3f, center_dist=%.3f",
                    frame_id,
                    det.box.tolist(),
                    other.box.tolist(),
                    iom,
                    center_dist,
                )
                is_dup = True
                break
        if not is_dup:
            kept.append(det)
    return kept


def parse_detections(
    result: Any,
    frame: np.ndarray,
    mask: np.ndarray | None,
    cfg: TrackingConfig,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = frame.shape[:2]
    names = _names_dict(getattr(result, "names", {}))
    xyxy = _to_numpy(boxes.xyxy)
    conf = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    raw_ids = _to_numpy(getattr(boxes, "id", None))
    masks = _result_masks(result, width, height)
    if xyxy is None or conf is None:
        return []

    detections: list[Detection] = []
    for idx, box_values in enumerate(xyxy):
        class_id = int(classes[idx]) if classes is not None else None
        if cfg.class_id is not None and class_id != cfg.class_id:
            continue
        if cfg.allowed_class_name is not None and class_id is not None:
            class_name = names.get(class_id, "").lower()
            if class_name != cfg.allowed_class_name.lower():
                continue

        box = clip_box(np.asarray(box_values, dtype=np.float32), width, height)
        if not roi_keep(mask, box, cfg):
            continue

        raw_id = None
        if raw_ids is not None and idx < len(raw_ids):
            raw_id = int(raw_ids[idx])
        detections.append(
            Detection(
                box=box,
                score=float(conf[idx]),
                raw_id=raw_id,
                class_id=class_id,
                hist=extract_hist_hsv(
                    frame,
                    box,
                    foreground_core=cfg.appearance_hist_foreground_core,
                ),
                mask=masks[idx] if idx < len(masks) else None,
            )
        )

    detections.sort(key=lambda item: item.score, reverse=True)
    if cfg.mode == "hybrid_bytetrack":
        detections = suppress_duplicate_detections(detections, cfg)
        return detections[: max(cfg.expected_pigs * 3, cfg.expected_pigs)]
    detections = deduplicate_detections(detections, cfg, width, height)
    return detections


def suppress_duplicate_detections(
    detections: list[Detection],
    cfg: TrackingConfig,
) -> list[Detection]:
    kept: list[Detection] = []
    for det in detections:
        if all(
            detection_overlap_score(det, other, cfg) < cfg.dup_iou_threshold
            for other in kept
        ):
            kept.append(det)
    return kept


def confidence_ladder(cfg: TrackingConfig) -> list[float]:
    """Return descending thresholds from review_conf to det_conf."""
    thresholds: list[float] = []
    current = cfg.review_conf
    while current > cfg.det_conf:
        thresholds.append(round(current, 4))
        current -= cfg.adaptive_conf_step
    thresholds.append(round(cfg.det_conf, 4))

    unique_thresholds: list[float] = []
    seen: set[float] = set()
    for threshold in thresholds:
        clipped = float(np.clip(threshold, cfg.det_conf, cfg.review_conf))
        if clipped not in seen:
            unique_thresholds.append(clipped)
            seen.add(clipped)
    return unique_thresholds


def adaptive_confidence_filter(
    detections: list[Detection],
    cfg: TrackingConfig,
) -> list[Detection]:
    """Keep the highest confidence threshold that still gives enough candidates,
    or just filter by det_conf in realtime.
    """
    if cfg.mode in {"realtime", "bytetrack_raw"}:
        return [det for det in detections if det.score >= cfg.det_conf]

    if not detections:
        return []

    max_candidates = max(cfg.expected_pigs * 3, cfg.expected_pigs)
    for threshold in confidence_ladder(cfg):
        selected = [det for det in detections if det.score >= threshold]
        if len(selected) >= cfg.expected_pigs:
            return selected[:max_candidates]
    return [det for det in detections[:max_candidates] if det.score >= cfg.det_conf]


def spatial_sort_detections(detections: list[Detection]) -> list[Detection]:
    return sorted(
        detections,
        key=lambda det: (bbox_center(det.box)[1], bbox_center(det.box)[0]),
    )


__all__ = [
    "_names_dict",
    "_result_masks",
    "_to_numpy",
    "adaptive_confidence_filter",
    "confidence_ladder",
    "extract_hist_hsv",
    "hist_distance",
    "parse_detections",
    "spatial_sort_detections",
    "suppress_duplicate_detections",
    "deduplicate_detections",
]
