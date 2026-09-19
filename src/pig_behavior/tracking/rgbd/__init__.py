# ruff: noqa
"""RGB-D Bird's-Eye-View tracking sub-package.

This package provides an *optional* tracking pipeline that projects 2-D
YOLO detections into 3-D world space using depth data, then performs
data association in Bird's-Eye-View using a Kalman filter with Euclidean
distance as the primary cost metric.

The existing 2-D pipeline in ``pig_behavior.tracking.runner`` is untouched.
"""

# ruff: noqa

from pig_behavior.tracking.rgbd.association_bev import match_bev_tracks
from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig, validate_rgbd_config
from pig_behavior.tracking.rgbd.depth import (
    DepthExtractionResult,
    RGBDCalibration,
    compute_depth_confidence,
    depth_frame_to_meters,
    extract_depth_for_bbox,
    load_calibration,
)
from pig_behavior.tracking.rgbd.kalman import (
    BEVKalmanFilter,
    bev_position,
    bev_velocity,
    create_bev_kalman,
    predict_bev,
    update_bev,
)
from pig_behavior.tracking.rgbd.occlusion import (
    infer_occlusions,
    track_is_occluded,
    update_occlusion_age,
)
from pig_behavior.tracking.rgbd.projector import RGBDProjector
from pig_behavior.tracking.rgbd.reporting import (
    write_association_log_csv,
    write_quality_report_csv,
    write_quality_report_json,
    write_tracking_csv,
)
from pig_behavior.tracking.rgbd.runner_rgbd import run_rgbd_tracking
from pig_behavior.tracking.rgbd.sanity import (
    validate_rgbd_update,
    validate_rgbd_update_with_frame_size,
)
from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    BEVTrackState,
    Detection2D,
    Detection3D,
    FrameTrackRow,
    RGBDQualityMetrics,
)
from pig_behavior.tracking.rgbd.sync import RGBDFrameSynchronizer, SyncStats

__all__ = [
    "AssociationDecision",
    "BEVKalmanFilter",
    "BEVTrackState",
    "DepthExtractionResult",
    "Detection2D",
    "Detection3D",
    "FrameTrackRow",
    "RGBDCalibration",
    "RGBDFrameSynchronizer",
    "RGBDProjector",
    "RGBDQualityMetrics",
    "RGBDTrackingConfig",
    "SyncStats",
    "bev_position",
    "bev_velocity",
    "compute_depth_confidence",
    "create_bev_kalman",
    "depth_frame_to_meters",
    "extract_depth_for_bbox",
    "infer_occlusions",
    "load_calibration",
    "match_bev_tracks",
    "predict_bev",
    "run_rgbd_tracking",
    "track_is_occluded",
    "update_bev",
    "update_occlusion_age",
    "validate_rgbd_config",
    "validate_rgbd_update",
    "validate_rgbd_update_with_frame_size",
    "write_association_log_csv",
    "write_quality_report_csv",
    "write_quality_report_json",
    "write_tracking_csv",
]
