"""Diagnostics and CSV report writing for the stable tracking pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrameDiagnosticRow:
    frame: int
    track_id: int
    tracklet_id: int
    bbox: tuple[float, float, float, float]
    center_jump_norm: float
    area_ratio_prev: float
    depth_valid: bool
    bev_valid: bool
    matched_by: str  # "bev_3d" | "2d_fallback" | "predict_only" | "gap"
    cost_bev: float | None
    cost_iou_2d: float
    cost_area: float
    cost_hist: float
    final_cost: float | None
    assignment_margin: float | None
    is_ambiguous: bool
    reject_reason: str | None


@dataclass
class StitchingReportRow:
    parent_tracklet_id: int
    child_tracklet_id: int
    gap_frames: int
    cost_iou_2d: float
    cost_center: float
    cost_area: float
    cost_hist: float
    cost_bev: float | None
    final_score: float
    is_stitched: bool


@dataclass
class SwapCandidateRow:
    track_id_a: int
    track_id_b: int
    frame_start: int
    frame_end: int
    crossing_frame: int
    swap_confidence: float
    is_fixed: bool
    distance_norm: float


def write_frame_diagnostics(rows: list[FrameDiagnosticRow], output_path: Path | str) -> None:
    """Writes frame-level diagnostic records to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "frame",
        "track_id",
        "tracklet_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_jump_norm",
        "area_ratio_prev",
        "depth_valid",
        "bev_valid",
        "matched_by",
        "cost_bev",
        "cost_iou_2d",
        "cost_area",
        "cost_hist",
        "final_cost",
        "assignment_margin",
        "is_ambiguous",
        "reject_reason",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            # Flatten the bbox tuple into individual coordinates
            x1, y1, x2, y2 = row.bbox
            writer.writerow(
                [
                    row.frame,
                    row.track_id,
                    row.tracklet_id,
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    f"{row.center_jump_norm:.4f}",
                    f"{row.area_ratio_prev:.4f}",
                    int(row.depth_valid),
                    int(row.bev_valid),
                    row.matched_by,
                    f"{row.cost_bev:.4f}" if row.cost_bev is not None else "",
                    f"{row.cost_iou_2d:.4f}",
                    f"{row.cost_area:.4f}",
                    f"{row.cost_hist:.4f}",
                    f"{row.final_cost:.4f}" if row.final_cost is not None else "",
                    f"{row.assignment_margin:.4f}" if row.assignment_margin is not None else "",
                    int(row.is_ambiguous),
                    row.reject_reason or "",
                ]
            )


def write_stitching_report(rows: list[StitchingReportRow], output_path: Path | str) -> None:
    """Writes stitching candidate metrics and decisions to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "parent_tracklet_id",
        "child_tracklet_id",
        "gap_frames",
        "cost_iou_2d",
        "cost_center",
        "cost_area",
        "cost_hist",
        "cost_bev",
        "final_score",
        "is_stitched",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row.parent_tracklet_id,
                    row.child_tracklet_id,
                    row.gap_frames,
                    f"{row.cost_iou_2d:.4f}",
                    f"{row.cost_center:.4f}",
                    f"{row.cost_area:.4f}",
                    f"{row.cost_hist:.4f}",
                    f"{row.cost_bev:.4f}" if row.cost_bev is not None else "",
                    f"{row.final_score:.4f}",
                    int(row.is_stitched),
                ]
            )


def write_swap_candidates(rows: list[SwapCandidateRow], output_path: Path | str) -> None:
    """Writes potential ID swap events to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "track_id_a",
        "track_id_b",
        "frame_start",
        "frame_end",
        "crossing_frame",
        "swap_confidence",
        "is_fixed",
        "distance_norm",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row.track_id_a,
                    row.track_id_b,
                    row.frame_start,
                    row.frame_end,
                    row.crossing_frame,
                    f"{row.swap_confidence:.4f}",
                    int(row.is_fixed),
                    f"{row.distance_norm:.4f}",
                ]
            )


def write_stability_summary(summary: dict[str, Any], output_path: Path | str) -> None:
    """Writes stability summary report to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            if isinstance(value, float):
                writer.writerow([metric, f"{value:.4f}"])
            else:
                writer.writerow([metric, str(value)])
