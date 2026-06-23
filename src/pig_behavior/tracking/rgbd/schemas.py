# ruff: noqa
"""Data structures for RGB-D tracking: detections, track state and audit."""

# ruff: noqa

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Detection schemas
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Detection2D:
    """Adapter over a raw YOLO detection for RGB-D projection.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    mask: np.ndarray | None = None
    hist: np.ndarray | None = None
    raw_id: int | None = None


@dataclass(slots=True)
class Detection3D:
    """A 2-D detection enriched with 3-D world position from depth data."""

    detection_2d: Detection2D
    camera_xyz: np.ndarray | None = None
    world_xyz: np.ndarray | None = None
    bev_xy: np.ndarray | None = None
    depth_m: float | None = None
    depth_valid: bool = False
    depth_ambiguous: bool = False
    depth_iqr_m: float | None = None
    valid_depth_pixel_count: int = 0
    invalid_reason: str | None = None


# ---------------------------------------------------------------------------
# BEV track state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BEVTrackState:
    """Kalman Filter state for one tracked pig in Bird's-Eye-View space."""

    fixed_id: int
    kf: object  # BEVKalmanFilter instance (duck-typed)
    bev_position: np.ndarray
    bev_velocity: np.ndarray
    last_depth_m: float | None = None
    missed: int = 0
    hits: int = 0
    age: int = 0
    occluded_age: int = 0
    state: str = "tentative"  # tentative | confirmed | lost


# ---------------------------------------------------------------------------
# Association audit trail
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AssociationDecision:
    """One-row audit record for a single track ↔ detection pairing."""

    frame_index: int
    track_id: int
    detection_index: int | None = None
    bev_distance_m: float | None = None
    cost: float | None = None
    best_score: float | None = None
    second_best_score: float | None = None
    score_margin: float | None = None
    accepted: bool = False
    reject_reason: str | None = None
    depth_valid: bool = True
    depth_ambiguous: bool = False
    is_occluded: bool = False


# ---------------------------------------------------------------------------
# Per-frame tracking result row (for CSV export)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameTrackRow:
    """One row per (frame, track) pair for CSV export."""

    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    world_x: float | None = None
    world_y: float | None = None
    world_z: float | None = None
    depth_m: float | None = None
    state: str = "tentative"
    confidence: float = 0.0
    is_occluded: bool = False
    is_predict_only: bool = False
    is_review: bool = False
    depth_valid: bool = True
    depth_ambiguous: bool = False
    association_distance_m: float | None = None
    reject_reason: str | None = None


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


@dataclass
class RGBDQualityMetrics:
    """Aggregate quality counters for the full tracking run."""

    total_frames: int = 0
    total_tracks: int = 0
    confirmed_tracks: int = 0
    lost_tracks: int = 0
    depth_invalid_count: int = 0
    depth_ambiguous_count: int = 0
    occlusion_frame_count: int = 0
    predict_only_frame_count: int = 0
    fallback_2d_count: int = 0
    ambiguous_match_count: int = 0
    rejected_update_count: int = 0
    rejected_by_invalid_depth: int = 0
    rejected_by_depth_ambiguous: int = 0
    rejected_by_bev_distance: int = 0
    rejected_by_center_jump: int = 0
    rejected_by_area_ratio: int = 0
    rejected_by_aspect_ratio: int = 0
    rejected_by_score_margin: int = 0
    bbox_jump_count: int = 0
    association_distances: list[float] = field(default_factory=list)

    @property
    def mean_association_distance_m(self) -> float:
        if not self.association_distances:
            return 0.0
        return float(np.mean(self.association_distances))

    @property
    def max_association_distance_m(self) -> float:
        if not self.association_distances:
            return 0.0
        return float(np.max(self.association_distances))


__all__ = [
    "AssociationDecision",
    "BEVTrackState",
    "Detection2D",
    "Detection3D",
    "FrameTrackRow",
    "RGBDQualityMetrics",
]
