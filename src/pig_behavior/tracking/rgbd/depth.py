# ruff: noqa
"""Depth extraction, calibration loading and depth-confidence scoring.

# ruff: noqa

All depth-extraction strategies follow the same contract: given a raw depth
frame (uint16 or float), a bounding box and optional mask/background, return
the median depth in *metres*, an IQR measure for ambiguity detection, and the
number of valid foreground pixels used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RGBDCalibration:
    """Camera calibration data loaded from ``.npy`` files."""

    depth_scale: float
    inverse_intrinsic: np.ndarray  # (3, 3)
    rotation: np.ndarray  # (3, 3)
    background_depth_m: np.ndarray | None  # (H, W) or None


def load_calibration(cfg: RGBDTrackingConfig) -> RGBDCalibration:
    """Load and validate calibration arrays from disk."""
    depth_scale_raw = np.load(str(cfg.depth_scale_path))
    depth_scale = float(depth_scale_raw.flat[0])
    if depth_scale <= 0:
        raise ValueError(f"depth_scale must be > 0, got {depth_scale}")

    k_inv = np.load(str(cfg.inverse_intrinsic_path)).astype(np.float64)
    if k_inv.shape != (3, 3):
        raise ValueError(f"inverse_intrinsic must be (3,3), got {k_inv.shape}")

    rot = np.load(str(cfg.rotation_path)).astype(np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must be (3,3), got {rot.shape}")

    background_depth_m: np.ndarray | None = None
    if cfg.background_depth_path is not None and cfg.background_depth_path.exists():
        import cv2

        bg_raw = cv2.imread(str(cfg.background_depth_path), cv2.IMREAD_UNCHANGED)
        if bg_raw is not None:
            if bg_raw.ndim == 3:
                bg_raw = bg_raw[:, :, 0]
            background_depth_m = bg_raw.astype(np.float64) * depth_scale
            logger.info(
                "Background depth loaded: shape=%s range=[%.3f, %.3f] m",
                background_depth_m.shape,
                float(np.nanmin(background_depth_m)),
                float(np.nanmax(background_depth_m)),
            )

    logger.info(
        "Calibration loaded: depth_scale=%.6f K_inv_det=%.4e R_det=%.4e",
        depth_scale,
        float(np.linalg.det(k_inv)),
        float(np.linalg.det(rot)),
    )
    return RGBDCalibration(
        depth_scale=depth_scale,
        inverse_intrinsic=k_inv,
        rotation=rot,
        background_depth_m=background_depth_m,
    )


# ---------------------------------------------------------------------------
# Depth conversion
# ---------------------------------------------------------------------------


def depth_frame_to_meters(
    raw_depth: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    """Convert a raw integer depth frame to metres (float64)."""
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[:, :, 0]
    return raw_depth.astype(np.float64) * depth_scale


# ---------------------------------------------------------------------------
# Depth extraction result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DepthExtractionResult:
    """Result of depth extraction for a single detection."""

    median_depth_m: float | None
    iqr_m: float | None
    valid_pixel_count: int
    depth_valid: bool
    depth_ambiguous: bool
    invalid_reason: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_bbox(
    x1: float, y1: float, x2: float, y2: float, h: int, w: int
) -> tuple[int, int, int, int]:
    """Clamp bbox to frame boundaries and return integer coords."""
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2 + 0.5))
    cy2 = min(h, int(y2 + 0.5))
    return cx1, cy1, cx2, cy2


def _center_crop_region(
    cx1: int, cy1: int, cx2: int, cy2: int, ratio: float
) -> tuple[int, int, int, int]:
    """Return the centre-crop sub-region of a bbox."""
    bw = cx2 - cx1
    bh = cy2 - cy1
    margin_x = int(bw * (1.0 - ratio) / 2.0)
    margin_y = int(bh * (1.0 - ratio) / 2.0)
    return cx1 + margin_x, cy1 + margin_y, cx2 - margin_x, cy2 - margin_y


def _lower_center_crop_region(
    cx1: int, cy1: int, cx2: int, cy2: int, ratio: float
) -> tuple[int, int, int, int]:
    """Centre-crop horizontally, take the bottom half vertically."""
    bw = cx2 - cx1
    bh = cy2 - cy1
    margin_x = int(bw * (1.0 - ratio) / 2.0)
    top = cy1 + bh // 2
    return cx1 + margin_x, top, cx2 - margin_x, cy2


def _filter_valid_depths(
    depths: np.ndarray,
    cfg: RGBDTrackingConfig,
    background_patch: np.ndarray | None = None,
) -> np.ndarray:
    """Remove invalid and background pixels from a depth array."""
    valid = np.isfinite(depths) & (depths > 0)
    valid &= depths >= cfg.min_depth_m
    valid &= depths <= cfg.max_depth_m

    if background_patch is not None and background_patch.shape == depths.shape:
        bg_valid = np.isfinite(background_patch) & (background_patch > 0)
        foreground = np.abs(depths - background_patch) > cfg.background_filter_m
        valid &= foreground | ~bg_valid

    return depths[valid]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _extract_median_center_crop(
    depth_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RGBDTrackingConfig,
    background_patch: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rx1, ry1, rx2, ry2 = _center_crop_region(x1, y1, x2, y2, cfg.center_crop_ratio)
    patch = depth_m[ry1:ry2, rx1:rx2].ravel()
    bg = background_patch[ry1:ry2, rx1:rx2].ravel() if background_patch is not None else None
    return _filter_valid_depths(patch, cfg, bg), (rx1, ry1, rx2, ry2)


def _extract_lower_center_crop(
    depth_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RGBDTrackingConfig,
    background_patch: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rx1, ry1, rx2, ry2 = _lower_center_crop_region(x1, y1, x2, y2, cfg.center_crop_ratio)
    patch = depth_m[ry1:ry2, rx1:rx2].ravel()
    bg = background_patch[ry1:ry2, rx1:rx2].ravel() if background_patch is not None else None
    return _filter_valid_depths(patch, cfg, bg), (rx1, ry1, rx2, ry2)


def _extract_foreground_median(
    depth_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RGBDTrackingConfig,
    background_patch: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    patch = depth_m[y1:y2, x1:x2].ravel()
    bg = background_patch[y1:y2, x1:x2].ravel() if background_patch is not None else None
    return _filter_valid_depths(patch, cfg, bg), (x1, y1, x2, y2)


def _extract_foreground_points_median(
    depth_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RGBDTrackingConfig,
    background_patch: np.ndarray | None,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Use mask pixels if available, else full bbox foreground."""
    if mask is not None:
        h_d, w_d = depth_m.shape[:2]
        h_m, w_m = mask.shape[:2]
        if (h_m, w_m) != (h_d, w_d):
            import cv2
            mask = cv2.resize(
                mask.astype(np.uint8), (w_d, h_d),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask_region = mask[y1:y2, x1:x2]
        patch = depth_m[y1:y2, x1:x2][mask_region].ravel()
        bg: np.ndarray | None = None
        if background_patch is not None:
            bg = background_patch[y1:y2, x1:x2][mask_region].ravel()
    else:
        patch = depth_m[y1:y2, x1:x2].ravel()
        bg = background_patch[y1:y2, x1:x2].ravel() if background_patch is not None else None
    return _filter_valid_depths(patch, cfg, bg), (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_depth_for_bbox(
    depth_frame_m: np.ndarray,
    bbox: tuple[float, float, float, float],
    cfg: RGBDTrackingConfig,
    background_depth_m: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> DepthExtractionResult:
    """Extract median depth, IQR and validity for a single bounding box.

    Parameters
    ----------
    depth_frame_m:
        Full depth frame in metres (float).
    bbox:
        ``(x1, y1, x2, y2)`` detection bounding box in pixel coords.
    cfg:
        RGB-D tracking configuration.
    background_depth_m:
        Optional background depth map in metres (same resolution as depth).
    mask:
        Optional per-pixel segmentation mask for this detection.

    Returns
    -------
    DepthExtractionResult
    """
    h, w = depth_frame_m.shape[:2]
    bx1, by1, bx2, by2 = _clamp_bbox(*bbox, h, w)
    if bx2 <= bx1 or by2 <= by1:
        return DepthExtractionResult(
            median_depth_m=None,
            iqr_m=None,
            valid_pixel_count=0,
            depth_valid=False,
            depth_ambiguous=False,
            invalid_reason="empty_bbox",
        )

    strategy = cfg.depth_strategy
    if strategy == "median_center_crop":
        valid_depths, _ = _extract_median_center_crop(
            depth_frame_m, bx1, by1, bx2, by2, cfg, background_depth_m
        )
    elif strategy == "lower_center_crop":
        valid_depths, _ = _extract_lower_center_crop(
            depth_frame_m, bx1, by1, bx2, by2, cfg, background_depth_m
        )
    elif strategy == "foreground_median":
        valid_depths, _ = _extract_foreground_median(
            depth_frame_m, bx1, by1, bx2, by2, cfg, background_depth_m
        )
    elif strategy == "foreground_points_median":
        valid_depths, _ = _extract_foreground_points_median(
            depth_frame_m, bx1, by1, bx2, by2, cfg, background_depth_m, mask
        )
    else:
        raise ValueError(f"Unknown depth_strategy: {strategy}")

    count = len(valid_depths)
    if count < cfg.min_valid_depth_pixels:
        return DepthExtractionResult(
            median_depth_m=float(np.median(valid_depths)) if count > 0 else None,
            iqr_m=None,
            valid_pixel_count=count,
            depth_valid=False,
            depth_ambiguous=False,
            invalid_reason="too_few_valid_pixels",
        )

    median_val = float(np.median(valid_depths))
    q25 = float(np.percentile(valid_depths, 25))
    q75 = float(np.percentile(valid_depths, 75))
    iqr = q75 - q25
    ambiguous = iqr > cfg.depth_ambiguity_iqr_m

    return DepthExtractionResult(
        median_depth_m=median_val,
        iqr_m=iqr,
        valid_pixel_count=count,
        depth_valid=True,
        depth_ambiguous=ambiguous,
        invalid_reason=None,
    )


def compute_depth_confidence(result: DepthExtractionResult) -> float:
    """Return a 0-1 confidence score for a depth extraction result.

    Higher is more reliable.
    """
    if not result.depth_valid or result.median_depth_m is None:
        return 0.0
    pixel_score = min(1.0, result.valid_pixel_count / 100.0)
    iqr_score = 1.0
    if result.iqr_m is not None and result.iqr_m > 0:
        iqr_score = max(0.0, 1.0 - result.iqr_m / 0.50)
    ambiguity_penalty = 0.3 if result.depth_ambiguous else 0.0
    return max(0.0, min(1.0, 0.5 * pixel_score + 0.5 * iqr_score - ambiguity_penalty))


__all__ = [
    "DepthExtractionResult",
    "RGBDCalibration",
    "compute_depth_confidence",
    "depth_frame_to_meters",
    "extract_depth_for_bbox",
    "load_calibration",
]
