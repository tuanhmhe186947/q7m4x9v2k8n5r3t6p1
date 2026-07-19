"""Diagnostic logic for tracking ID events and continuity gaps."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .assets import TrackingPair
from .cvat_io import parse_cvat_video_xml
from .matching import match_frame
from .metrics import (
    best_id_mapping,
    matched_identity_counts,
    remap_prediction_ids,
)


def identity_events_for_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    remap_ids: bool = False,
    evaluation_start_frame: int | None = None,
    evaluation_end_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Return frame-level ID mismatches and switches for one GT/prediction pair."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    parse_kwargs = {
        "include_hidden": include_hidden,
        "start_frame": evaluation_start_frame,
        "end_frame": evaluation_end_frame,
    }
    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, **parse_kwargs)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, **parse_kwargs)
    id_mapping: dict[str, str] = {}
    if remap_ids:
        pred_by_frame, id_mapping, _mapped_matches, _coverage = remap_prediction_ids(
            gt_by_frame,
            pred_by_frame,
            iou_threshold=iou_threshold,
        )
    previous_pred_for_gt: dict[str, str] = {}
    events: list[dict[str, Any]] = []

    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        matches = match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        )
        for gt_idx, pred_idx, iou in matches:
            gt_id = gt_objects[gt_idx].obj_id
            pred_id = pred_objects[pred_idx].obj_id
            previous_pred_id = previous_pred_for_gt.get(gt_id)
            is_switch = previous_pred_id is not None and previous_pred_id != pred_id
            is_mismatch = gt_id != pred_id
            if is_switch or is_mismatch:
                event_type = (
                    "id_switch_and_mismatch"
                    if is_switch and is_mismatch
                    else "id_switch"
                    if is_switch
                    else "id_mismatch"
                )
                events.append(
                    {
                        "video_stem": pair.video_stem,
                        "frame": int(frame),
                        "gt_id": gt_id,
                        "pred_id": pred_id,
                        "previous_pred_id": previous_pred_id or "",
                        "event": event_type,
                        "iou": round(float(iou), 4),
                        "gt_source_track_id": gt_objects[gt_idx].source_track_id,
                        "pred_source_track_id": pred_objects[pred_idx].source_track_id,
                        "remapped": remap_ids,
                        "id_mapping": (
                            f"{pred_id}->{id_mapping[pred_id]}"
                            if remap_ids and pred_id in id_mapping
                            else ""
                        ),
                        "gt_xml": str(pair.gt_xml),
                        "pred_xml": str(pair.pred_xml),
                    }
                )
            previous_pred_for_gt[gt_id] = pred_id

    return events


def identity_events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert identity diagnostics to a stable dataframe shape."""
    columns = [
        "video_stem",
        "frame",
        "gt_id",
        "pred_id",
        "previous_pred_id",
        "event",
        "iou",
        "gt_source_track_id",
        "pred_source_track_id",
        "remapped",
        "id_mapping",
        "gt_xml",
        "pred_xml",
    ]
    return pd.DataFrame(events, columns=columns)


def identity_mapping_for_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    evaluation_start_frame: int | None = None,
    evaluation_end_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed prediction->GT ID remapping used for paper metrics."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    parse_kwargs = {
        "include_hidden": include_hidden,
        "start_frame": evaluation_start_frame,
        "end_frame": evaluation_end_frame,
    }
    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, **parse_kwargs)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, **parse_kwargs)
    pair_counts = matched_identity_counts(
        gt_by_frame,
        pred_by_frame,
        iou_threshold=iou_threshold,
    )
    mapping, mapped_matches, total_matches = best_id_mapping(pair_counts)
    rows = []
    for pred_id, gt_id in sorted(mapping.items()):
        rows.append(
            {
                "video_stem": pair.video_stem,
                "pred_id": pred_id,
                "mapped_gt_id": gt_id,
                "matched_frames": pair_counts[(gt_id, pred_id)],
                "total_matched_frames": total_matches,
                "mapping_coverage": (
                    mapped_matches / total_matches if total_matches else 0.0
                ),
                "gt_xml": str(pair.gt_xml),
                "pred_xml": str(pair.pred_xml),
            }
        )
    return rows


def identity_mapping_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert fixed ID remapping rows to a stable dataframe shape."""
    columns = [
        "video_stem",
        "pred_id",
        "mapped_gt_id",
        "matched_frames",
        "total_matched_frames",
        "mapping_coverage",
        "gt_xml",
        "pred_xml",
    ]
    return pd.DataFrame(rows, columns=columns)


def continuity_gaps_for_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    gap_tolerance_frames: int = 15,
    remap_ids: bool = True,
    evaluation_start_frame: int | None = None,
    evaluation_end_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Return matched-track gaps used to diagnose fragmentation."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    tolerance = max(0, int(gap_tolerance_frames))
    parse_kwargs = {
        "include_hidden": include_hidden,
        "start_frame": evaluation_start_frame,
        "end_frame": evaluation_end_frame,
    }
    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, **parse_kwargs)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, **parse_kwargs)
    if remap_ids:
        pred_by_frame, _mapping, _mapped_matches, _coverage = remap_prediction_ids(
            gt_by_frame,
            pred_by_frame,
            iou_threshold=iou_threshold,
        )

    matched_by_gt: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        for gt_idx, pred_idx, iou in match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        ):
            gt_id = gt_objects[gt_idx].obj_id
            pred_id = pred_objects[pred_idx].obj_id
            matched_by_gt[gt_id].append((int(frame), pred_id, float(iou)))

    rows: list[dict[str, Any]] = []
    for gt_id, matches in sorted(matched_by_gt.items()):
        ordered = sorted(matches, key=lambda item: item[0])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_frame, previous_pred_id, previous_iou = previous
            next_frame, next_pred_id, next_iou = current
            gap_frames = next_frame - previous_frame - 1
            if gap_frames <= 0:
                continue

            tolerated = gap_frames <= tolerance
            rows.append(
                {
                    "video_stem": pair.video_stem,
                    "gt_id": gt_id,
                    "previous_matched_frame": previous_frame,
                    "next_matched_frame": next_frame,
                    "gap_frames": gap_frames,
                    "tolerated": tolerated,
                    "event": "tolerated_gap" if tolerated else "fragment_gap",
                    "previous_pred_id": previous_pred_id,
                    "next_pred_id": next_pred_id,
                    "id_changed": previous_pred_id != next_pred_id,
                    "previous_iou": round(previous_iou, 4),
                    "next_iou": round(next_iou, 4),
                    "gap_tolerance_frames": tolerance,
                    "remapped": remap_ids,
                    "gt_xml": str(pair.gt_xml),
                    "pred_xml": str(pair.pred_xml),
                }
            )

    return rows


def continuity_gaps_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert continuity gap diagnostics to a stable dataframe shape."""
    columns = [
        "video_stem",
        "gt_id",
        "previous_matched_frame",
        "next_matched_frame",
        "gap_frames",
        "tolerated",
        "event",
        "previous_pred_id",
        "next_pred_id",
        "id_changed",
        "previous_iou",
        "next_iou",
        "gap_tolerance_frames",
        "remapped",
        "gt_xml",
        "pred_xml",
    ]
    return pd.DataFrame(rows, columns=columns)
