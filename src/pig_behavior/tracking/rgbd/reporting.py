# ruff: noqa
"""CSV and JSON reporting for RGB-D tracking results."""

# ruff: noqa

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    FrameTrackRow,
    RGBDQualityMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "world_x",
    "world_y",
    "world_z",
    "depth_m",
    "state",
    "confidence",
    "is_occluded",
    "is_predict_only",
    "is_review",
    "depth_valid",
    "depth_ambiguous",
    "association_distance_m",
    "reject_reason",
]


def write_tracking_csv(
    path: Path,
    rows: list[FrameTrackRow],
) -> None:
    """Write per-frame-per-track results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row.frame,
                    "track_id": row.track_id,
                    "x1": f"{row.x1:.2f}",
                    "y1": f"{row.y1:.2f}",
                    "x2": f"{row.x2:.2f}",
                    "y2": f"{row.y2:.2f}",
                    "world_x": f"{row.world_x:.4f}" if row.world_x is not None else "",
                    "world_y": f"{row.world_y:.4f}" if row.world_y is not None else "",
                    "world_z": f"{row.world_z:.4f}" if row.world_z is not None else "",
                    "depth_m": f"{row.depth_m:.4f}" if row.depth_m is not None else "",
                    "state": row.state,
                    "confidence": f"{row.confidence:.4f}",
                    "is_occluded": row.is_occluded,
                    "is_predict_only": row.is_predict_only,
                    "is_review": row.is_review,
                    "depth_valid": row.depth_valid,
                    "depth_ambiguous": row.depth_ambiguous,
                    "association_distance_m": (
                        f"{row.association_distance_m:.4f}"
                        if row.association_distance_m is not None
                        else ""
                    ),
                    "reject_reason": row.reject_reason or "",
                }
            )
    logger.info("[OK] tracking CSV: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# JSON quality report
# ---------------------------------------------------------------------------


def _quality_to_dict(metrics: RGBDQualityMetrics) -> dict[str, Any]:
    """Serialise quality metrics to a JSON-friendly dict."""
    d: dict[str, Any] = {
        "total_frames": metrics.total_frames,
        "total_tracks": metrics.total_tracks,
        "confirmed_tracks": metrics.confirmed_tracks,
        "lost_tracks": metrics.lost_tracks,
        "depth_invalid_count": metrics.depth_invalid_count,
        "depth_ambiguous_count": metrics.depth_ambiguous_count,
        "occlusion_frame_count": metrics.occlusion_frame_count,
        "predict_only_frame_count": metrics.predict_only_frame_count,
        "fallback_2d_count": metrics.fallback_2d_count,
        "ambiguous_match_count": metrics.ambiguous_match_count,
        "rejected_update_count": metrics.rejected_update_count,
        "rejected_by_invalid_depth": metrics.rejected_by_invalid_depth,
        "rejected_by_depth_ambiguous": metrics.rejected_by_depth_ambiguous,
        "rejected_by_bev_distance": metrics.rejected_by_bev_distance,
        "rejected_by_center_jump": metrics.rejected_by_center_jump,
        "rejected_by_area_ratio": metrics.rejected_by_area_ratio,
        "rejected_by_aspect_ratio": metrics.rejected_by_aspect_ratio,
        "rejected_by_score_margin": metrics.rejected_by_score_margin,
        "bbox_jump_count": metrics.bbox_jump_count,
        "mean_association_distance_m": round(metrics.mean_association_distance_m, 6),
        "max_association_distance_m": round(metrics.max_association_distance_m, 6),
    }
    return d


def write_quality_report_json(
    path: Path,
    metrics: RGBDQualityMetrics,
) -> None:
    """Write the aggregate quality report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = _quality_to_dict(metrics)
    path.write_text(
        json.dumps(d, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[OK] quality report JSON: %s", path)


def write_quality_report_csv(
    path: Path,
    metrics: RGBDQualityMetrics,
) -> None:
    """Write the aggregate quality report as a single-row CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = _quality_to_dict(metrics)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(d.keys()))
        writer.writeheader()
        writer.writerow(d)
    logger.info("[OK] quality report CSV: %s", path)


# ---------------------------------------------------------------------------
# Association decisions log
# ---------------------------------------------------------------------------


def write_association_log_csv(
    path: Path,
    decisions: list[AssociationDecision],
) -> None:
    """Write the full association audit trail to CSV (debug mode)."""
    if not decisions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "frame_index",
        "track_id",
        "detection_index",
        "bev_distance_m",
        "cost",
        "best_score",
        "second_best_score",
        "score_margin",
        "accepted",
        "reject_reason",
        "depth_valid",
        "depth_ambiguous",
        "is_occluded",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            writer.writerow(
                {
                    "frame_index": d.frame_index,
                    "track_id": d.track_id,
                    "detection_index": d.detection_index if d.detection_index is not None else "",
                    "bev_distance_m": (
                        f"{d.bev_distance_m:.4f}" if d.bev_distance_m is not None else ""
                    ),
                    "cost": f"{d.cost:.4f}" if d.cost is not None else "",
                    "best_score": f"{d.best_score:.4f}" if d.best_score is not None else "",
                    "second_best_score": (
                        f"{d.second_best_score:.4f}" if d.second_best_score is not None else ""
                    ),
                    "score_margin": (
                        f"{d.score_margin:.4f}" if d.score_margin is not None else ""
                    ),
                    "accepted": d.accepted,
                    "reject_reason": d.reject_reason or "",
                    "depth_valid": d.depth_valid,
                    "depth_ambiguous": d.depth_ambiguous,
                    "is_occluded": d.is_occluded,
                }
            )
    logger.info("[OK] association log CSV: %s (%d rows)", path, len(decisions))


__all__ = [
    "write_association_log_csv",
    "write_quality_report_csv",
    "write_quality_report_json",
    "write_tracking_csv",
]
