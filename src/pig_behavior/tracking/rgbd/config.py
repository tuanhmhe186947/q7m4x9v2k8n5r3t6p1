# ruff: noqa
"""Configuration for the optional RGB-D tracking pipeline.

``RGBDTrackingConfig`` uses *composition* rather than inheritance because
:class:`~pig_behavior.tracking.config.TrackingConfig` is defined with
``slots=True``, which disallows subclass extension.  The embedded
``tracking_config`` field carries all existing 2-D pipeline settings so
that YOLO detection, mask handling and export behaviour stay identical.
"""

# ruff: noqa

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pig_behavior.tracking.config import TrackingConfig


@dataclass(slots=True)
class RGBDTrackingConfig:
    """Full configuration for RGB-D Bird's-Eye-View tracking."""

    # ---- embedded 2-D config (detector, mask, output, thresholds) ----------
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)

    # ---- RGB-D data paths --------------------------------------------------
    depth_video_path: Path = Path("depth.mp4")
    times_path: Path | None = None
    depth_scale_path: Path = Path("depth_scale.npy")
    inverse_intrinsic_path: Path = Path("inverse_intrinsic.npy")
    rotation_path: Path = Path("rot.npy")
    background_depth_path: Path | None = None

    # ---- depth extraction --------------------------------------------------
    background_filter_m: float = 0.15
    center_crop_ratio: float = 0.50
    min_valid_depth_pixels: int = 20
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    depth_ambiguity_iqr_m: float = 0.25

    depth_strategy: Literal[
        "median_center_crop",
        "lower_center_crop",
        "foreground_median",
        "foreground_points_median",
    ] = "foreground_points_median"

    # ---- BEV projection ----------------------------------------------------
    bev_axes: tuple[int, int] = (0, 1)
    bev_association_gate_m: float = 0.40
    larger_depth_is_farther: bool = True

    # ---- occlusion ---------------------------------------------------------
    occlusion_iou_threshold: float = 0.40
    max_occlusion_age: int = 45

    # ---- depth failure mode ------------------------------------------------
    depth_failure_mode: Literal[
        "predict_only",
        "fallback_2d",
        "skip_frame",
    ] = "predict_only"

    # ---- sanity gate -------------------------------------------------------
    min_score_margin: float = 0.05
    max_center_jump_norm: float = 0.06
    min_area_ratio: float = 0.60
    max_area_ratio: float = 1.50
    min_aspect_ratio_change: float = 0.50
    max_aspect_ratio_change: float = 2.00

    # ---- BEV association cost weights --------------------------------------
    w_bev: float = 0.50
    w_conf: float = 0.10
    w_depth_ambiguous: float = 0.15
    w_occlusion: float = 0.10
    w_hist: float = 0.15

    # ---- Kalman filter tuning ----------------------------------------------
    kf_process_std: float = 0.10
    kf_measurement_std: float = 0.05

    # ---- rendering / debug -------------------------------------------------
    render: bool = True
    debug: bool = False


def validate_rgbd_config(cfg: RGBDTrackingConfig) -> None:
    """Raise ``ValueError`` / ``FileNotFoundError`` for invalid settings."""
    if not cfg.depth_video_path.exists():
        raise FileNotFoundError(f"Depth video not found: {cfg.depth_video_path}")
    if not cfg.depth_scale_path.exists():
        raise FileNotFoundError(f"depth_scale.npy not found: {cfg.depth_scale_path}")
    if not cfg.inverse_intrinsic_path.exists():
        raise FileNotFoundError(
            f"inverse_intrinsic.npy not found: {cfg.inverse_intrinsic_path}"
        )
    if not cfg.rotation_path.exists():
        raise FileNotFoundError(f"rot.npy not found: {cfg.rotation_path}")
    if cfg.times_path is not None and not cfg.times_path.exists():
        raise FileNotFoundError(f"times.txt not found: {cfg.times_path}")
    if (
        cfg.background_depth_path is not None
        and not cfg.background_depth_path.exists()
    ):
        raise FileNotFoundError(
            f"background_depth not found: {cfg.background_depth_path}"
        )

    ax0, ax1 = cfg.bev_axes
    if not (0 <= ax0 <= 2 and 0 <= ax1 <= 2 and ax0 != ax1):
        raise ValueError(f"bev_axes must be two distinct indices in [0,2]: {cfg.bev_axes}")
    if cfg.background_filter_m < 0:
        raise ValueError("background_filter_m must be >= 0")
    if cfg.bev_association_gate_m <= 0:
        raise ValueError("bev_association_gate_m must be > 0")
    if cfg.min_valid_depth_pixels < 1:
        raise ValueError("min_valid_depth_pixels must be >= 1")
    if cfg.min_depth_m >= cfg.max_depth_m:
        raise ValueError("min_depth_m must be < max_depth_m")
    if cfg.depth_ambiguity_iqr_m <= 0:
        raise ValueError("depth_ambiguity_iqr_m must be > 0")
    if not 0.0 < cfg.center_crop_ratio <= 1.0:
        raise ValueError("center_crop_ratio must be in (0, 1]")
    if cfg.min_area_ratio <= 0 or cfg.max_area_ratio <= 0:
        raise ValueError("area ratio bounds must be > 0")
    if cfg.min_aspect_ratio_change <= 0 or cfg.max_aspect_ratio_change <= 0:
        raise ValueError("aspect ratio bounds must be > 0")
    if cfg.max_center_jump_norm <= 0:
        raise ValueError("max_center_jump_norm must be > 0")
    if not 0.0 <= cfg.min_score_margin <= 1.0:
        raise ValueError("min_score_margin must be in [0, 1]")


__all__ = [
    "RGBDTrackingConfig",
    "validate_rgbd_config",
]
