# ruff: noqa
"""Project 2-D pixel detections into 3-D camera/world/BEV coordinates.

# ruff: noqa

Two projection modes are supported:

1. **Single-point** — project the bottom-centre of the bounding box at the
   median depth.  Fast but sensitive to depth noise.
2. **Foreground-points-median** — project many foreground depth pixels to
   world space and take the component-wise median.  More robust for
   partially occluded objects.
"""

from __future__ import annotations

import logging

import numpy as np

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.depth import (
    DepthExtractionResult,
    RGBDCalibration,
    extract_depth_for_bbox,
)
from pig_behavior.tracking.rgbd.schemas import Detection2D, Detection3D

logger = logging.getLogger(__name__)


class RGBDProjector:
    """Stateless projector that converts 2-D pixel coordinates to 3-D.

    Parameters
    ----------
    calibration:
        Pre-loaded camera calibration (K_inv, R, depth_scale, background).
    cfg:
        RGB-D tracking configuration.
    """

    def __init__(
        self,
        calibration: RGBDCalibration,
        cfg: RGBDTrackingConfig,
    ) -> None:
        self._k_inv = calibration.inverse_intrinsic  # (3, 3)
        self._rot = calibration.rotation  # (3, 3)
        self._depth_scale = calibration.depth_scale
        self._background_m = calibration.background_depth_m
        self._cfg = cfg
        self._bev_ax0, self._bev_ax1 = cfg.bev_axes

    # ------------------------------------------------------------------
    # Low-level projection
    # ------------------------------------------------------------------

    def project_single_point(
        self,
        u: float,
        v: float,
        depth_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a single pixel ``(u, v)`` at *depth_m* metres.

        Returns ``(camera_xyz, world_xyz)`` each as shape ``(3,)``.
        """
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        camera_xyz = depth_m * (self._k_inv @ pixel_h)
        world_xyz = self._rot @ camera_xyz
        return camera_xyz.astype(np.float64), world_xyz.astype(np.float64)

    def project_foreground_pixels(
        self,
        pixel_uv: np.ndarray,
        depth_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project N foreground pixels to camera and world space.

        Parameters
        ----------
        pixel_uv:
            ``(N, 2)`` array of ``(u, v)`` pixel coordinates.
        depth_m:
            ``(N,)`` array of depth values in metres.

        Returns
        -------
        camera_xyz:
            ``(N, 3)`` camera-space points.
        world_xyz:
            ``(N, 3)`` world-space points.
        """
        n = len(pixel_uv)
        ones = np.ones((n, 1), dtype=np.float64)
        # (N, 3) homogeneous pixels
        pixel_h = np.hstack([pixel_uv.astype(np.float64), ones])
        # camera_xyz[i] = depth_m[i] * K_inv @ pixel_h[i]
        camera_xyz = (self._k_inv @ pixel_h.T).T * depth_m[:, None]
        world_xyz = (self._rot @ camera_xyz.T).T
        return camera_xyz, world_xyz

    # ------------------------------------------------------------------
    # High-level: Detection2D → Detection3D
    # ------------------------------------------------------------------

    def project_detection(
        self,
        detection_2d: Detection2D,
        depth_frame_m: np.ndarray,
    ) -> Detection3D:
        """Project one Detection2D to Detection3D using the configured strategy.

        If depth is invalid the detection is still returned with
        ``depth_valid=False`` so the caller can decide what to do.
        """
        result = extract_depth_for_bbox(
            depth_frame_m,
            detection_2d.bbox,
            self._cfg,
            background_depth_m=self._background_m,
            mask=detection_2d.mask,
        )

        if not result.depth_valid or result.median_depth_m is None:
            return Detection3D(
                detection_2d=detection_2d,
                depth_m=result.median_depth_m,
                depth_valid=False,
                depth_ambiguous=result.depth_ambiguous,
                depth_iqr_m=result.iqr_m,
                valid_depth_pixel_count=result.valid_pixel_count,
                invalid_reason=result.invalid_reason,
            )

        strategy = self._cfg.depth_strategy
        if strategy == "foreground_points_median":
            return self._project_foreground_points(detection_2d, depth_frame_m, result)
        else:
            return self._project_bottom_center(detection_2d, result)

    def project_detections(
        self,
        detections_2d: list[Detection2D],
        depth_frame_m: np.ndarray,
    ) -> list[Detection3D]:
        """Batch-project all 2-D detections."""
        return [self.project_detection(d, depth_frame_m) for d in detections_2d]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _project_bottom_center(
        self,
        det: Detection2D,
        depth_result: DepthExtractionResult,
    ) -> Detection3D:
        """Single-point projection from bbox bottom-centre."""
        x1, y1, x2, y2 = det.bbox
        u = (x1 + x2) / 2.0
        v = y2  # bottom-centre
        depth = depth_result.median_depth_m
        assert depth is not None

        camera_xyz, world_xyz = self.project_single_point(u, v, depth)
        bev_xy = np.array(
            [world_xyz[self._bev_ax0], world_xyz[self._bev_ax1]],
            dtype=np.float64,
        )
        return Detection3D(
            detection_2d=det,
            camera_xyz=camera_xyz,
            world_xyz=world_xyz,
            bev_xy=bev_xy,
            depth_m=depth,
            depth_valid=True,
            depth_ambiguous=depth_result.depth_ambiguous,
            depth_iqr_m=depth_result.iqr_m,
            valid_depth_pixel_count=depth_result.valid_pixel_count,
        )

    def _project_foreground_points(
        self,
        det: Detection2D,
        depth_frame_m: np.ndarray,
        depth_result: DepthExtractionResult,
    ) -> Detection3D:
        """Multi-point projection: project foreground pixels, take median."""
        h, w = depth_frame_m.shape[:2]
        x1, y1, x2, y2 = det.bbox
        bx1 = max(0, int(x1))
        by1 = max(0, int(y1))
        bx2 = min(w, int(x2 + 0.5))
        by2 = min(h, int(y2 + 0.5))

        if bx2 <= bx1 or by2 <= by1:
            return self._project_bottom_center(det, depth_result)

        # Build pixel grid inside bbox
        ys, xs = np.mgrid[by1:by2, bx1:bx2]
        depths = depth_frame_m[by1:by2, bx1:bx2]

        # Apply mask if available
        if det.mask is not None:
            mask_region = det.mask
            mh, mw = mask_region.shape[:2]
            if (mh, mw) != (h, w):
                import cv2
                mask_region = cv2.resize(
                    mask_region.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask_crop = mask_region[by1:by2, bx1:bx2]
        else:
            mask_crop = np.ones_like(depths, dtype=bool)

        # Filter valid pixels
        valid = (
            np.isfinite(depths)
            & (depths >= self._cfg.min_depth_m)
            & (depths <= self._cfg.max_depth_m)
            & (depths > 0)
            & mask_crop
        )

        # Background filter
        if self._background_m is not None:
            bg_patch = self._background_m[by1:by2, bx1:bx2]
            bg_valid = np.isfinite(bg_patch) & (bg_patch > 0)
            fg_mask = np.abs(depths - bg_patch) > self._cfg.background_filter_m
            valid &= fg_mask | ~bg_valid

        valid_xs = xs[valid].ravel()
        valid_ys = ys[valid].ravel()
        valid_ds = depths[valid].ravel()

        if len(valid_ds) < self._cfg.min_valid_depth_pixels:
            # Not enough points — fall back to single-point
            return self._project_bottom_center(det, depth_result)

        pixel_uv = np.column_stack([valid_xs, valid_ys]).astype(np.float64)
        _, world_pts = self.project_foreground_pixels(pixel_uv, valid_ds)

        # Component-wise median
        median_world = np.median(world_pts, axis=0)
        bev_xy = np.array(
            [median_world[self._bev_ax0], median_world[self._bev_ax1]],
            dtype=np.float64,
        )

        return Detection3D(
            detection_2d=det,
            camera_xyz=None,  # ambiguous for multi-point
            world_xyz=median_world,
            bev_xy=bev_xy,
            depth_m=depth_result.median_depth_m,
            depth_valid=True,
            depth_ambiguous=depth_result.depth_ambiguous,
            depth_iqr_m=depth_result.iqr_m,
            valid_depth_pixel_count=depth_result.valid_pixel_count,
        )


__all__ = [
    "RGBDProjector",
]
