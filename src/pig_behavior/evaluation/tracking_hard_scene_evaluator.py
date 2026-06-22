"""Hard-Scene Identity Evaluator for pig tracking.

Standalone evaluator that diagnoses identity errors in hard/occluded scenes.
Produces per-frame CSV, swap event CSV, hard frame summary CSV, summary
metrics JSON, and optional overlay video.  Does NOT modify or depend on the
existing HOTA/MOTA evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from pig_behavior.evaluation.tracking.cvat_io import (
    TrackingObject,
    parse_cvat_video_xml,
    read_task_name,
)
from pig_behavior.evaluation.tracking.matching import iou_xyxy, match_frame
from pig_behavior.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §2  Data Model & Config
# ---------------------------------------------------------------------------


@dataclass
class HardSceneEvalConfig:
    """Configuration for hard-scene identity evaluation."""

    gt_xml: Path = field(default_factory=lambda: Path("gt.xml"))
    pred_xml: Path = field(default_factory=lambda: Path("pred.xml"))
    video_path: Path | None = None
    output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "evaluation" / "hard_scene_output"
    )
    match_iou_threshold: float = 0.30
    stable_match_iou_threshold: float = 0.50
    hard_threshold: float = 0.60
    critical_threshold: float = 0.80
    long_swap_threshold: int = 15
    event_clip_padding_frames: int = 30
    include_hidden: bool = False
    top_n_overlay_events: int = 10


# ---------------------------------------------------------------------------
# §3  Confidence helper
# ---------------------------------------------------------------------------


def _read_box_confidence(box_el: ET.Element) -> float | None:
    """Read confidence/score from a CVAT ``<box>`` element.

    Checks the ``score`` XML attribute first, then looks for a child
    ``<attribute name="score">``.  Returns ``None`` when no score is
    present — never fabricates a value.
    """
    raw = box_el.attrib.get("score")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    for attr in box_el.findall("attribute"):
        if (attr.attrib.get("name") or "").lower() == "score" and attr.text:
            try:
                return float(attr.text.strip())
            except (ValueError, TypeError):
                pass
    return None


def _parse_with_confidence(
    xml_path: Path,
    *,
    include_hidden: bool = False,
) -> tuple[dict[int, list[TrackingObject]], dict[tuple[int, str], float | None]]:
    """Parse CVAT XML and also extract per-box confidence values.

    Returns the normal frame-indexed dict **and** a mapping
    ``(frame, obj_id) -> confidence | None``.
    """
    by_frame = parse_cvat_video_xml(xml_path, include_hidden=include_hidden)

    # Second pass for confidence only
    tree = ET.parse(xml_path)
    root = tree.getroot()
    conf_map: dict[tuple[int, str], float | None] = {}
    for track_el in root.findall("track"):
        for box_el in track_el.findall("box"):
            outside = str(box_el.attrib.get("outside", "0")).lower()
            if outside in {"1", "true", "yes"}:
                continue
            frame = int(box_el.attrib["frame"])
            # Match obj_id logic from cvat_io to correlate
            for obj in by_frame.get(frame, []):
                bbox = (
                    float(box_el.attrib["xtl"]),
                    float(box_el.attrib["ytl"]),
                    float(box_el.attrib["xbr"]),
                    float(box_el.attrib["ybr"]),
                )
                if obj.bbox == bbox:
                    conf_map[(frame, obj.obj_id)] = _read_box_confidence(box_el)
    return by_frame, conf_map


# ---------------------------------------------------------------------------
# §3  Frame Matching with intermediate structure
# ---------------------------------------------------------------------------


@dataclass
class FrameMatchResult:
    """Intermediate per-frame matching result."""

    frame: int
    gt_objects: list[TrackingObject]
    pred_objects: list[TrackingObject]
    iou_matrix: np.ndarray  # shape (n_gt, n_pred)
    matches: list[tuple[int, int, float]]  # (gt_idx, pred_idx, iou)
    unmatched_gt: list[int]
    unmatched_pred: list[int]
    competing_matches: int  # count of ambiguous IoU situations


def _compute_competing(
    iou_matrix: np.ndarray,
    matches: list[tuple[int, int, float]],
    iou_threshold: float,
) -> int:
    """Count ambiguous matches: a GT/pred with multiple IoU near threshold."""
    competing = 0
    near_margin = 0.10
    for gt_idx, _pred_idx, _iou in matches:
        candidates = int(
            np.sum(iou_matrix[gt_idx, :] >= iou_threshold - near_margin)
        )
        if candidates > 1:
            competing += 1
    for _gt_idx, pred_idx, _iou in matches:
        candidates = int(
            np.sum(iou_matrix[:, pred_idx] >= iou_threshold - near_margin)
        )
        if candidates > 1:
            competing += 1
    return competing


def match_all_frames(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    iou_threshold: float,
) -> dict[int, FrameMatchResult]:
    """Match GT and pred for every frame, returning intermediate results."""
    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    results: dict[int, FrameMatchResult] = {}
    for frame in all_frames:
        gt_objs = gt_by_frame.get(frame, [])
        pred_objs = pred_by_frame.get(frame, [])

        n_gt = len(gt_objs)
        n_pred = len(pred_objs)
        iou_matrix = np.zeros((n_gt, n_pred), dtype=float)
        for i, gt in enumerate(gt_objs):
            for j, pred in enumerate(pred_objs):
                iou_matrix[i, j] = iou_xyxy(gt.bbox, pred.bbox)

        matches = match_frame(gt_objs, pred_objs, iou_threshold=iou_threshold)
        matched_gt = {gt_idx for gt_idx, _, _ in matches}
        matched_pred = {pred_idx for _, pred_idx, _ in matches}
        unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
        unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]
        competing = _compute_competing(iou_matrix, matches, iou_threshold)

        results[frame] = FrameMatchResult(
            frame=frame,
            gt_objects=gt_objs,
            pred_objects=pred_objs,
            iou_matrix=iou_matrix,
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            competing_matches=competing,
        )
    return results


# ---------------------------------------------------------------------------
# §5  Hardness & Occlusion scoring helpers
# ---------------------------------------------------------------------------


def _max_pair_iou(objects: list[TrackingObject]) -> float:
    """Max IoU between any two bboxes in the same frame."""
    max_iou = 0.0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            max_iou = max(max_iou, iou_xyxy(objects[i].bbox, objects[j].bbox))
    return max_iou


def _min_center_distance(objects: list[TrackingObject]) -> float:
    """Min center-to-center distance normalized by avg bbox diagonal.

    Returns 1.0 (max / "far apart") when fewer than 2 objects.
    """
    if len(objects) < 2:
        return 1.0
    centers = []
    diags = []
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        diags.append(math.hypot(x2 - x1, y2 - y1))
    avg_diag = sum(diags) / len(diags) if diags else 1.0
    if avg_diag <= 0:
        avg_diag = 1.0
    min_dist = float("inf")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            min_dist = min(min_dist, math.hypot(dx, dy) / avg_diag)
    return min(min_dist, 1.0)


def _detection_ambiguity(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> float:
    """Fraction of GT/pred rows with multiple near-threshold candidates."""
    if iou_matrix.size == 0:
        return 0.0
    near = 0.10
    ambiguous = 0
    total = iou_matrix.shape[0] + iou_matrix.shape[1]
    if total == 0:
        return 0.0
    for i in range(iou_matrix.shape[0]):
        if int(np.sum(iou_matrix[i, :] >= iou_threshold - near)) > 1:
            ambiguous += 1
    for j in range(iou_matrix.shape[1]):
        if int(np.sum(iou_matrix[:, j] >= iou_threshold - near)) > 1:
            ambiguous += 1
    return min(ambiguous / total, 1.0)


def _area_of_bbox(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def _area_drop_ratio(
    current_area: float,
    recent_areas: list[float],
) -> float:
    """Ratio of area change vs rolling median.  Returns 0 when stable."""
    if not recent_areas or current_area <= 0:
        return 0.0
    median_area = statistics.median(recent_areas)
    if median_area <= 0:
        return 0.0
    ratio = abs(current_area - median_area) / median_area
    return min(ratio, 1.0)


def _num_nearby(
    target: TrackingObject,
    others: list[TrackingObject],
    scale: float = 2.0,
) -> int:
    """Count objects within ``scale`` × target bbox diagonal."""
    x1, y1, x2, y2 = target.bbox
    diag = math.hypot(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    threshold = diag * scale
    count = 0
    for obj in others:
        if obj is target:
            continue
        ox1, oy1, ox2, oy2 = obj.bbox
        ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        if math.hypot(cx - ocx, cy - ocy) <= threshold:
            count += 1
    return count


def compute_frame_scores(
    fmr: FrameMatchResult,
    iou_threshold: float,
    gt_area_history: dict[str, list[float]],
) -> tuple[float, float, dict[str, Any]]:
    """Compute hardness and occlusion scores for one frame.

    Returns ``(hardness, occlusion, components_dict)``.
    """
    all_objects = fmr.gt_objects + fmr.pred_objects
    max_gt_iou = _max_pair_iou(fmr.gt_objects)
    max_pred_iou = _max_pair_iou(fmr.pred_objects)
    pair_iou = max(max_gt_iou, max_pred_iou)

    raw_dist = _min_center_distance(all_objects)
    center_closeness = 1.0 - raw_dist  # closer → higher

    ambiguity = _detection_ambiguity(fmr.iou_matrix, iou_threshold)

    # Area instability: max drop ratio across GT objects in this frame
    area_instability = 0.0
    for gt_obj in fmr.gt_objects:
        area = _area_of_bbox(gt_obj.bbox)
        recent = gt_area_history.get(gt_obj.obj_id, [])
        drop = _area_drop_ratio(area, recent)
        area_instability = max(area_instability, drop)
        # Update history (keep last 5)
        gt_area_history.setdefault(gt_obj.obj_id, []).append(area)
        if len(gt_area_history[gt_obj.obj_id]) > 5:
            gt_area_history[gt_obj.obj_id] = gt_area_history[gt_obj.obj_id][-5:]

    # Missing / low confidence fraction
    n_gt = len(fmr.gt_objects)
    missing_frac = len(fmr.unmatched_gt) / n_gt if n_gt > 0 else 0.0

    # Nearby tracks count (max across GT objects)
    num_nearby = 0
    for gt_obj in fmr.gt_objects:
        num_nearby = max(num_nearby, _num_nearby(gt_obj, all_objects))

    hardness = (
        0.30 * pair_iou
        + 0.20 * center_closeness
        + 0.20 * ambiguity
        + 0.15 * area_instability
        + 0.15 * missing_frac
    )

    # Missing nearby: unmatched GT that were nearby other GT last frame
    missing_nearby_frac = missing_frac  # simplified proxy

    occlusion = (
        0.35 * pair_iou
        + 0.25 * center_closeness
        + 0.25 * area_instability
        + 0.15 * missing_nearby_frac
    )

    components = {
        "pair_overlap": round(pair_iou, 4),
        "center_distance": round(raw_dist, 4),
        "detection_ambiguity": round(ambiguity, 4),
        "area_instability": round(area_instability, 4),
        "missing_frac": round(missing_frac, 4),
        "num_nearby": num_nearby,
    }
    return round(hardness, 4), round(occlusion, 4), components


def classify_difficulty(
    hardness: float,
    hard_threshold: float = 0.60,
    critical_threshold: float = 0.80,
) -> str:
    if hardness < 0.30:
        return "easy"
    if hardness < hard_threshold:
        return "medium"
    if hardness < critical_threshold:
        return "hard"
    return "critical"


# ---------------------------------------------------------------------------
# §4  Canonical Identity Mapping
# ---------------------------------------------------------------------------


def build_canonical_mapping(
    frame_results: dict[int, FrameMatchResult],
    frame_hardness: dict[int, float],
    config: HardSceneEvalConfig,
) -> dict[str, tuple[str, str]]:
    """Build ``pred_source_track_id -> (canonical_gt_id, confidence)``.

    Uses stable frames (high IoU, low difficulty) for majority vote.
    Falls back to all matched frames when stable sample is too small.
    """
    # Collect all (pred_source_track_id, gt_id) pairs per frame
    all_pairs: dict[str, Counter[str]] = defaultdict(Counter)
    stable_pairs: dict[str, Counter[str]] = defaultdict(Counter)

    for frame, fmr in sorted(frame_results.items()):
        difficulty = classify_difficulty(
            frame_hardness.get(frame, 0.0),
            config.hard_threshold,
            config.critical_threshold,
        )
        for gt_idx, pred_idx, iou in fmr.matches:
            gt_id = fmr.gt_objects[gt_idx].obj_id
            pred_track = fmr.pred_objects[pred_idx].source_track_id
            all_pairs[pred_track][gt_id] += 1

            is_stable = (
                iou >= config.stable_match_iou_threshold
                and difficulty not in ("hard", "critical")
                and fmr.competing_matches <= 1
            )
            if is_stable:
                stable_pairs[pred_track][gt_id] += 1

    mapping: dict[str, tuple[str, str]] = {}
    for pred_track in all_pairs:
        sp = stable_pairs.get(pred_track, Counter())
        total_stable = sum(sp.values())
        if total_stable >= 5:
            best_gt = sp.most_common(1)[0][0]
            mapping[pred_track] = (best_gt, "high")
        else:
            ap = all_pairs[pred_track]
            best_gt = ap.most_common(1)[0][0]
            mapping[pred_track] = (best_gt, "low")

    return mapping


# ---------------------------------------------------------------------------
# §6  Per-Frame Identity CSV rows
# ---------------------------------------------------------------------------


def build_per_frame_rows(
    frame_results: dict[int, FrameMatchResult],
    canonical_mapping: dict[str, tuple[str, str]],
    frame_scores: dict[int, tuple[float, float, dict[str, Any]]],
    pred_conf: dict[tuple[int, str], float | None],
    video_name: str,
) -> list[dict[str, Any]]:
    """Build per-frame identity analysis rows (one per GT object per frame)."""
    rows: list[dict[str, Any]] = []

    for frame in sorted(frame_results):
        fmr = frame_results[frame]
        hardness, occlusion, components = frame_scores.get(
            frame, (0.0, 0.0, {})
        )

        # Build quick lookup: gt_idx -> (pred_idx, iou)
        gt_to_pred: dict[int, tuple[int, float]] = {}
        for gt_idx, pred_idx, iou in fmr.matches:
            gt_to_pred[gt_idx] = (pred_idx, iou)

        for gt_idx, gt_obj in enumerate(fmr.gt_objects):
            match = gt_to_pred.get(gt_idx)
            is_matched = match is not None

            if is_matched:
                pred_idx, iou = match
                pred_obj = fmr.pred_objects[pred_idx]
                pred_track = pred_obj.source_track_id
                canonical_gt, _conf = canonical_mapping.get(
                    pred_track, ("", "")
                )
                pred_bbox_str = (
                    f"{pred_obj.bbox[0]:.1f},{pred_obj.bbox[1]:.1f},"
                    f"{pred_obj.bbox[2]:.1f},{pred_obj.bbox[3]:.1f}"
                )
                matched_pred_track_id = pred_track
                matched_iou = round(iou, 4)
            else:
                pred_bbox_str = ""
                matched_pred_track_id = ""
                canonical_gt = ""
                matched_iou = float("nan")

            is_id_correct = is_matched and canonical_gt == gt_obj.obj_id
            is_id_wrong = is_matched and canonical_gt != gt_obj.obj_id
            is_missing = not is_matched

            conf = pred_conf.get((frame, gt_obj.obj_id))

            gt_bbox_str = (
                f"{gt_obj.bbox[0]:.1f},{gt_obj.bbox[1]:.1f},"
                f"{gt_obj.bbox[2]:.1f},{gt_obj.bbox[3]:.1f}"
            )

            rows.append(
                {
                    "video_name": video_name,
                    "frame_idx": frame,
                    "gt_id": gt_obj.obj_id,
                    "gt_bbox": gt_bbox_str,
                    "matched_pred_track_id": matched_pred_track_id,
                    "pred_bbox": pred_bbox_str,
                    "matched_iou": matched_iou,
                    "canonical_gt_of_pred_track": canonical_gt,
                    "is_matched": is_matched,
                    "is_id_correct": is_id_correct,
                    "is_id_wrong": is_id_wrong,
                    "is_missing": is_missing,
                    "is_swap": False,  # back-filled in §7
                    "swap_with_gt_id": "",  # back-filled in §7
                    "hardness_score": hardness,
                    "occlusion_score": occlusion,
                    "pair_overlap": components.get("pair_overlap", 0.0),
                    "center_distance": components.get("center_distance", 1.0),
                    "area_drop_ratio": components.get("area_instability", 0.0),
                    "num_nearby_tracks": components.get("num_nearby", 0),
                    "num_competing_tracks": fmr.competing_matches,
                    "detection_confidence": conf,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# §7  Swap Detection
# ---------------------------------------------------------------------------


def detect_swap_events(
    rows: list[dict[str, Any]],
    canonical_mapping: dict[str, tuple[str, str]],
    long_swap_threshold: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect two-way identity swap events and back-fill rows.

    Returns ``(swap_events, updated_rows)``.
    """
    # Index rows by (frame, gt_id) for fast lookup
    by_frame_gt: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        by_frame_gt[(row["frame_idx"], row["gt_id"])] = row

    # Collect per-frame swap pairs
    frames = sorted({r["frame_idx"] for r in rows})
    swap_frames: dict[tuple[str, str], list[int]] = defaultdict(list)

    for frame in frames:
        # Get all matched rows for this frame
        frame_rows = [r for r in rows if r["frame_idx"] == frame and r["is_matched"]]
        # Check for two-way swaps
        for i, row_a in enumerate(frame_rows):
            for row_b in frame_rows[i + 1 :]:
                gt_a = row_a["gt_id"]
                gt_b = row_b["gt_id"]
                canonical_a = row_a["canonical_gt_of_pred_track"]
                canonical_b = row_b["canonical_gt_of_pred_track"]
                # Two-way swap: A matched to B's track and B matched to A's track
                if canonical_a == gt_b and canonical_b == gt_a:
                    pair = tuple(sorted([gt_a, gt_b]))
                    swap_frames[pair].append(frame)

    # Group consecutive frames into events
    events: list[dict[str, Any]] = []
    event_id = 0
    for pair, swap_frame_list in sorted(swap_frames.items()):
        sorted_frames = sorted(set(swap_frame_list))
        groups: list[list[int]] = []
        current_group: list[int] = [sorted_frames[0]]
        for f in sorted_frames[1:]:
            if f - current_group[-1] <= 2:  # allow 1-frame gap
                current_group.append(f)
            else:
                groups.append(current_group)
                current_group = [f]
        groups.append(current_group)

        gt_a, gt_b = pair
        for group in groups:
            start_frame = group[0]
            end_frame = group[-1]
            duration = end_frame - start_frame + 1

            # Collect hardness/occlusion over event frames
            event_hardness = []
            event_occlusion = []
            for f in group:
                row_a = by_frame_gt.get((f, gt_a))
                row_b = by_frame_gt.get((f, gt_b))
                if row_a:
                    event_hardness.append(row_a["hardness_score"])
                    event_occlusion.append(row_a["occlusion_score"])
                if row_b:
                    event_hardness.append(row_b["hardness_score"])
                    event_occlusion.append(row_b["occlusion_score"])

            # Determine baseline and wrong pred tracks
            baseline_a = ""
            baseline_b = ""
            wrong_a = ""
            wrong_b = ""
            row_a_start = by_frame_gt.get((start_frame, gt_a))
            row_b_start = by_frame_gt.get((start_frame, gt_b))
            if row_a_start:
                wrong_a = row_a_start["matched_pred_track_id"]
                # Canonical of wrong_a should be gt_b
                can_gt, _ = canonical_mapping.get(wrong_a, ("", ""))
                if can_gt == gt_b:
                    baseline_b = wrong_a  # this track normally tracks gt_b
            if row_b_start:
                wrong_b = row_b_start["matched_pred_track_id"]
                can_gt, _ = canonical_mapping.get(wrong_b, ("", ""))
                if can_gt == gt_a:
                    baseline_a = wrong_b  # this track normally tracks gt_a

            # Recovery detection
            recovered = False
            recovery_latency = -1
            all_sorted_frames = sorted(
                {r["frame_idx"] for r in rows if r["frame_idx"] > end_frame}
            )
            for f in all_sorted_frames:
                row_a_post = by_frame_gt.get((f, gt_a))
                row_b_post = by_frame_gt.get((f, gt_b))
                a_ok = row_a_post and row_a_post.get("is_id_correct", False)
                b_ok = row_b_post and row_b_post.get("is_id_correct", False)
                if a_ok and b_ok:
                    recovered = True
                    recovery_latency = f - end_frame
                    break

            events.append(
                {
                    "event_id": event_id,
                    "gt_id_a": gt_a,
                    "gt_id_b": gt_b,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration,
                    "hardness_mean": round(
                        statistics.mean(event_hardness), 4
                    )
                    if event_hardness
                    else 0.0,
                    "hardness_max": round(max(event_hardness), 4)
                    if event_hardness
                    else 0.0,
                    "occlusion_mean": round(
                        statistics.mean(event_occlusion), 4
                    )
                    if event_occlusion
                    else 0.0,
                    "occlusion_max": round(max(event_occlusion), 4)
                    if event_occlusion
                    else 0.0,
                    "baseline_pred_track_a": baseline_a,
                    "baseline_pred_track_b": baseline_b,
                    "wrong_pred_track_a": wrong_a,
                    "wrong_pred_track_b": wrong_b,
                    "recovered_after_event": recovered,
                    "recovery_latency_frames": recovery_latency,
                }
            )
            event_id += 1

    # Back-fill is_swap / swap_with_gt_id in rows
    swap_lookup: dict[tuple[int, str], str] = {}
    for event in events:
        gt_a = event["gt_id_a"]
        gt_b = event["gt_id_b"]
        for f in range(event["start_frame"], event["end_frame"] + 1):
            swap_lookup[(f, gt_a)] = gt_b
            swap_lookup[(f, gt_b)] = gt_a

    for row in rows:
        key = (row["frame_idx"], row["gt_id"])
        if key in swap_lookup:
            row["is_swap"] = True
            row["swap_with_gt_id"] = swap_lookup[key]

    return events, rows


# ---------------------------------------------------------------------------
# §8  Summary Metrics
# ---------------------------------------------------------------------------


def compute_summary_metrics(
    rows: list[dict[str, Any]],
    swap_events: list[dict[str, Any]],
    canonical_mapping: dict[str, tuple[str, str]],
    config: HardSceneEvalConfig,
    predicted_frame_count: int,
) -> dict[str, Any]:
    """Compute aggregate hard-scene identity metrics."""
    total_frames = len({r["frame_idx"] for r in rows})
    total_gt_instances = len(rows)

    matched_rows = [r for r in rows if r["is_matched"]]
    matched_instance_count = len(matched_rows)
    correct_rows = [r for r in matched_rows if r["is_id_correct"]]
    wrong_rows = [r for r in matched_rows if r["is_id_wrong"]]
    missing_rows = [r for r in rows if r["is_missing"]]

    global_id_accuracy = (
        len(correct_rows) / matched_instance_count if matched_instance_count else 0.0
    )
    prediction_frame_coverage = (
        predicted_frame_count / total_frames if total_frames else 0.0
    )
    matched_instance_ratio = (
        matched_instance_count / total_gt_instances if total_gt_instances else 0.0
    )
    invalid_reasons = []
    if prediction_frame_coverage < 0.5:
        invalid_reasons.append(
            "prediction_frame_coverage_below_0.5"
        )
    if matched_instance_ratio < 0.5:
        invalid_reasons.append("matched_instance_ratio_below_0.5")
    evaluation_valid = not invalid_reasons

    # Per-difficulty accuracy (on matched instances only)
    difficulty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in matched_rows:
        diff = classify_difficulty(
            r["hardness_score"],
            config.hard_threshold,
            config.critical_threshold,
        )
        difficulty_groups[diff].append(r)

    def _acc(group: list[dict[str, Any]]) -> float:
        if not group:
            return 0.0
        return sum(1 for r in group if r["is_id_correct"]) / len(group)

    # Swap statistics
    total_swap_events = len(swap_events)
    long_swaps = [
        e for e in swap_events if e["duration_frames"] >= config.long_swap_threshold
    ]
    durations = [e["duration_frames"] for e in swap_events]
    recovered_events = [e for e in swap_events if e["recovered_after_event"]]
    recovery_latencies = [
        e["recovery_latency_frames"]
        for e in recovered_events
        if e["recovery_latency_frames"] >= 0
    ]

    # Hard / critical frame counts
    frame_hardness: dict[int, float] = {}
    for r in rows:
        frame_hardness.setdefault(r["frame_idx"], r["hardness_score"])
    hard_frames = [
        f
        for f, h in frame_hardness.items()
        if classify_difficulty(h, config.hard_threshold, config.critical_threshold)
        == "hard"
    ]
    critical_frames = [
        f
        for f, h in frame_hardness.items()
        if classify_difficulty(h, config.hard_threshold, config.critical_threshold)
        == "critical"
    ]

    # Error rates for hard/critical frames
    hard_matched = [
        r
        for r in matched_rows
        if classify_difficulty(
            r["hardness_score"], config.hard_threshold, config.critical_threshold
        )
        == "hard"
    ]
    critical_matched = [
        r
        for r in matched_rows
        if classify_difficulty(
            r["hardness_score"], config.hard_threshold, config.critical_threshold
        )
        == "critical"
    ]
    hard_errors = sum(1 for r in hard_matched if r["is_id_wrong"])
    critical_errors = sum(1 for r in critical_matched if r["is_id_wrong"])

    # Canonical mapping for JSON export
    mapping_export = {
        pred_track: {"canonical_gt_id": gt_id, "confidence": conf}
        for pred_track, (gt_id, conf) in canonical_mapping.items()
    }

    return {
        "total_frames": total_frames,
        "total_gt_instances": total_gt_instances,
        "predicted_frame_count": predicted_frame_count,
        "prediction_frame_coverage": round(prediction_frame_coverage, 4),
        "matched_instance_count": matched_instance_count,
        "matched_instance_ratio": round(matched_instance_ratio, 4),
        "evaluation_valid": evaluation_valid,
        "invalid_reason": (
            "invalid_or_incomplete_prediction: " + ", ".join(invalid_reasons)
            if invalid_reasons
            else ""
        ),
        "global_id_accuracy": round(global_id_accuracy, 4),
        "id_accuracy_easy": round(_acc(difficulty_groups.get("easy", [])), 4),
        "id_accuracy_medium": round(
            _acc(difficulty_groups.get("medium", [])), 4
        ),
        "id_accuracy_hard": round(_acc(difficulty_groups.get("hard", [])), 4),
        "id_accuracy_critical": round(
            _acc(difficulty_groups.get("critical", [])), 4
        ),
        "total_swap_events": total_swap_events,
        "long_term_swap_count": len(long_swaps),
        "max_swap_duration": max(durations) if durations else 0,
        "mean_swap_duration": round(statistics.mean(durations), 2)
        if durations
        else 0.0,
        "total_wrong_id_frames": len(wrong_rows),
        "total_missing_instances": len(missing_rows),
        "recovery_rate": round(
            len(recovered_events) / total_swap_events, 4
        )
        if total_swap_events
        else 0.0,
        "mean_reid_latency": round(statistics.mean(recovery_latencies), 2)
        if recovery_latencies
        else 0.0,
        "max_reid_latency": max(recovery_latencies)
        if recovery_latencies
        else 0,
        "hard_frame_count": len(hard_frames),
        "critical_frame_count": len(critical_frames),
        "hard_error_rate": round(hard_errors / len(hard_matched), 4)
        if hard_matched
        else 0.0,
        "critical_error_rate": round(
            critical_errors / len(critical_matched), 4
        )
        if critical_matched
        else 0.0,
        "canonical_mapping": mapping_export,
        "config": asdict(config),
    }


# ---------------------------------------------------------------------------
# §9  Hard Frame Summary CSV
# ---------------------------------------------------------------------------


def build_hard_frame_summary(
    frame_results: dict[int, FrameMatchResult],
    frame_scores: dict[int, tuple[float, float, dict[str, Any]]],
    per_frame_rows: list[dict[str, Any]],
    config: HardSceneEvalConfig,
    video_name: str,
) -> list[dict[str, Any]]:
    """Build one row per frame for the hard frame summary CSV."""
    # Index per-frame rows by frame for quick aggregation
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in per_frame_rows:
        rows_by_frame[r["frame_idx"]].append(r)

    summary_rows: list[dict[str, Any]] = []
    for frame in sorted(frame_results):
        fmr = frame_results[frame]
        hardness, occlusion, components = frame_scores.get(
            frame, (0.0, 0.0, {})
        )
        difficulty = classify_difficulty(
            hardness, config.hard_threshold, config.critical_threshold
        )
        frame_rows = rows_by_frame.get(frame, [])
        num_wrong = sum(1 for r in frame_rows if r.get("is_id_wrong"))
        num_swaps = sum(1 for r in frame_rows if r.get("is_swap"))

        summary_rows.append(
            {
                "video_name": video_name,
                "frame_idx": frame,
                "difficulty": difficulty,
                "hardness_score": hardness,
                "occlusion_score": occlusion,
                "num_gt": len(fmr.gt_objects),
                "num_pred": len(fmr.pred_objects),
                "num_missing": len(fmr.unmatched_gt),
                "num_wrong_id": num_wrong,
                "num_swaps": num_swaps,
                "max_pair_iou": components.get("pair_overlap", 0.0),
                "min_center_distance": components.get("center_distance", 1.0),
                "num_competing_tracks": fmr.competing_matches,
            }
        )
    return summary_rows


# ---------------------------------------------------------------------------
# §10  Overlay Video
# ---------------------------------------------------------------------------


def _try_import_cv2():  # type: ignore[no-untyped-def]
    """Attempt to import OpenCV.  Returns module or None."""
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def _overlay_color(row: dict[str, Any]) -> tuple[int, int, int]:
    """Pick BGR color for a GT object overlay."""
    if row.get("is_swap"):
        return (128, 0, 128)  # purple
    if row.get("is_id_wrong"):
        return (0, 0, 255)  # red
    if row.get("is_missing"):
        return (0, 165, 255)  # orange
    return (0, 200, 0)  # green


def _draw_frame_overlay(
    frame_img: np.ndarray,
    frame_rows: list[dict[str, Any]],
    hardness: float,
    difficulty: str,
    cv2: Any,
) -> np.ndarray:
    """Draw bounding boxes and labels on a single video frame."""
    img = frame_img.copy()
    h, w = img.shape[:2]

    # Banner for hard/critical frames
    if difficulty in ("hard", "critical"):
        banner_color = (0, 0, 180) if difficulty == "critical" else (0, 100, 200)
        cv2.rectangle(img, (0, 0), (w, 30), banner_color, -1)
        cv2.putText(
            img,
            f"{difficulty.upper()} | hardness={hardness:.2f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for row in frame_rows:
        color = _overlay_color(row)
        # Parse bbox
        if row.get("gt_bbox"):
            parts = row["gt_bbox"].split(",")
            if len(parts) == 4:
                x1, y1, x2, y2 = (int(float(p)) for p in parts)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Build label
                status = "OK"
                if row.get("is_swap"):
                    status = "SWAP"
                elif row.get("is_id_wrong"):
                    status = "WRONG"
                elif row.get("is_missing"):
                    status = "MISSING"

                canonical = row.get("canonical_gt_of_pred_track", "")
                pred_track = row.get("matched_pred_track_id", "")
                label = (
                    f"GT:{row['gt_id']} | Pred:T{pred_track} | "
                    f"Can:{canonical} | {status} | "
                    f"hard={row.get('hardness_score', 0):.2f} | "
                    f"occ={row.get('occlusion_score', 0):.2f}"
                )
                font_scale = 0.35
                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return img


def render_overlay_video(
    config: HardSceneEvalConfig,
    per_frame_rows: list[dict[str, Any]],
    frame_scores: dict[int, tuple[float, float, dict[str, Any]]],
    swap_events: list[dict[str, Any]],
) -> None:
    """Render overlay video and optional event clips.

    Silently skips if OpenCV is unavailable or no video path given.
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        logger.warning("OpenCV not available — skipping overlay video.")
        return
    if config.video_path is None or not config.video_path.exists():
        logger.info("No video path — skipping overlay rendering.")
        return

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", config.video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Index rows by frame
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in per_frame_rows:
        rows_by_frame[r["frame_idx"]].append(r)

    # Full overlay video
    out_path = config.output_dir / "hard_scene_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame_img = cap.read()
        if not ret:
            break
        hardness, occlusion, comps = frame_scores.get(
            frame_idx, (0.0, 0.0, {})
        )
        difficulty = classify_difficulty(
            hardness, config.hard_threshold, config.critical_threshold
        )
        frame_rows = rows_by_frame.get(frame_idx, [])
        overlaid = _draw_frame_overlay(
            frame_img, frame_rows, hardness, difficulty, cv2
        )
        writer.write(overlaid)
        frame_idx += 1

    writer.release()
    cap.release()
    logger.info("Overlay video saved: %s", out_path)

    # Event clips (top N by duration)
    sorted_events = sorted(
        swap_events, key=lambda e: e["duration_frames"], reverse=True
    )
    for event in sorted_events[: config.top_n_overlay_events]:
        clip_start = max(0, event["start_frame"] - config.event_clip_padding_frames)
        clip_end = event["end_frame"] + config.event_clip_padding_frames
        clip_name = (
            f"hard_event_{event['event_id']}"
            f"_frames_{event['start_frame']}_{event['end_frame']}.mp4"
        )
        clip_path = config.output_dir / clip_name
        cap2 = cv2.VideoCapture(str(config.video_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        clip_writer = cv2.VideoWriter(
            str(clip_path), fourcc, fps, (width, height)
        )
        for f in range(clip_start, clip_end + 1):
            ret, frame_img = cap2.read()
            if not ret:
                break
            hardness_f, _occ, _c = frame_scores.get(f, (0.0, 0.0, {}))
            diff_f = classify_difficulty(
                hardness_f, config.hard_threshold, config.critical_threshold
            )
            clip_rows = rows_by_frame.get(f, [])
            overlaid = _draw_frame_overlay(
                frame_img, clip_rows, hardness_f, diff_f, cv2
            )
            clip_writer.write(overlaid)
        clip_writer.release()
        cap2.release()
        logger.info("Event clip saved: %s", clip_path)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def make_win_long_path(path: Path | None) -> Path | None:
    r"""Resolve and prefix path with \\?\ on Windows to support MAX_PATH > 260."""
    if path is None:
        return None
    import platform
    resolved = path.resolve()
    if platform.system() == "Windows" and not str(resolved).startswith("\\\\?\\"):
        return Path(f"\\\\?\\{resolved}")
    return resolved


def run_hard_scene_evaluation(
    config: HardSceneEvalConfig,
) -> dict[str, Any]:
    """Run the full hard-scene identity evaluation pipeline.

    Returns the summary metrics dict.  All CSV/JSON files are written to
    ``config.output_dir``.
    """
    config.gt_xml = make_win_long_path(config.gt_xml)
    config.pred_xml = make_win_long_path(config.pred_xml)
    config.output_dir = make_win_long_path(config.output_dir)
    if config.video_path:
        config.video_path = make_win_long_path(config.video_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    video_name = read_task_name(config.gt_xml) or config.gt_xml.stem


    # Parse GT and prediction
    gt_by_frame = parse_cvat_video_xml(
        config.gt_xml, include_hidden=config.include_hidden
    )
    pred_by_frame, pred_conf = _parse_with_confidence(
        config.pred_xml, include_hidden=config.include_hidden
    )
    predicted_frame_count = sum(1 for objs in pred_by_frame.values() if objs)

    # §3 Frame matching
    frame_results = match_all_frames(
        gt_by_frame, pred_by_frame, config.match_iou_threshold
    )

    # §5 Hardness scoring (first pass — needed for canonical mapping)
    gt_area_history: dict[str, list[float]] = {}
    frame_scores: dict[int, tuple[float, float, dict[str, Any]]] = {}
    for frame in sorted(frame_results):
        h, o, c = compute_frame_scores(
            frame_results[frame], config.match_iou_threshold, gt_area_history
        )
        frame_scores[frame] = (h, o, c)

    frame_hardness = {f: h for f, (h, _, _) in frame_scores.items()}

    # §4 Canonical mapping
    canonical_mapping = build_canonical_mapping(
        frame_results, frame_hardness, config
    )

    # §6 Per-frame CSV rows
    per_frame_rows = build_per_frame_rows(
        frame_results, canonical_mapping, frame_scores, pred_conf, video_name
    )

    # §7 Swap detection (also back-fills is_swap in rows)
    swap_events, per_frame_rows = detect_swap_events(
        per_frame_rows, canonical_mapping, config.long_swap_threshold
    )

    # §8 Summary metrics
    metrics = compute_summary_metrics(
        per_frame_rows,
        swap_events,
        canonical_mapping,
        config,
        predicted_frame_count,
    )
    if not metrics["evaluation_valid"]:
        logger.warning(
            "Hard-scene evaluation has incomplete prediction coverage: %s",
            metrics["invalid_reason"],
        )

    # Serialize config paths to strings for JSON export
    metrics["config"] = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in metrics["config"].items()
    }

    # §9 Hard frame summary (after swap back-fill)
    hard_frame_rows = build_hard_frame_summary(
        frame_results, frame_scores, per_frame_rows, config, video_name
    )

    # Write outputs
    pd.DataFrame(per_frame_rows).to_csv(
        config.output_dir / "per_frame_identity_analysis.csv", index=False
    )
    pd.DataFrame(swap_events).to_csv(
        config.output_dir / "swap_events.csv", index=False
    )
    pd.DataFrame(hard_frame_rows).to_csv(
        config.output_dir / "hard_frame_summary.csv", index=False
    )
    with (config.output_dir / "hard_scene_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    # §10 Overlay video
    render_overlay_video(config, per_frame_rows, frame_scores, swap_events)

    logger.info("Hard-scene evaluation complete → %s", config.output_dir)
    return metrics


# ---------------------------------------------------------------------------
# §11  CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from pig_behavior.evaluation.tracking.assets import (
        TRACKING_GT_DIR,
        VIDEO_DIR,
        PREDICTION_ROOT,
    )
    parser = argparse.ArgumentParser(
        description="Hard-Scene Identity Evaluator for pig tracking.",
    )
    parser.add_argument("--gt-xml", type=Path, required=False, default=None)
    parser.add_argument("--pred-xml", type=Path, required=False, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "hard_scene_output",
    )
    parser.add_argument("--gt-dir", type=Path, default=TRACKING_GT_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional prediction run directory to use for auto-mapping.",
    )
    parser.add_argument("--match-iou-threshold", type=float, default=0.30)
    parser.add_argument("--stable-match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--hard-threshold", type=float, default=0.60)
    parser.add_argument("--critical-threshold", type=float, default=0.80)
    parser.add_argument("--long-swap-threshold", type=int, default=15)
    parser.add_argument("--event-clip-padding-frames", type=int, default=30)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--top-n-overlay-events", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``pig-tracking-hard-eval``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    from pig_behavior.evaluation.tracking.assets import list_tracking_pairs

    pairs_to_evaluate = []

    if args.gt_xml and args.pred_xml:
        # Use explicitly provided paths
        pairs_to_evaluate.append((args.gt_xml, args.pred_xml, args.video, args.output_dir))
    else:
        # Auto-map using list_tracking_pairs
        prediction_root = args.run_dir or args.prediction_root
        pairs = list_tracking_pairs(
            tracking_gt_dir=args.gt_dir,
            video_dir=args.video_dir,
            prediction_root=prediction_root,
        )
        if args.video:
            target_video = Path(args.video)
            matched_pair = None
            for pair in pairs:
                if (pair.video_path and pair.video_path.resolve() == target_video.resolve()) or \
                   pair.video_stem.lower() == target_video.stem.lower() or \
                   target_video.name.lower() in pair.video_path.name.lower():
                    matched_pair = pair
                    break
            if matched_pair is None:
                raise FileNotFoundError(f"Could not find matching ground truth/video pair for --video {args.video}")
            if matched_pair.pred_xml is None:
                raise FileNotFoundError(f"No prediction XML found for video {args.video} under prediction-root {prediction_root}")
            pairs_to_evaluate.append((matched_pair.gt_xml, matched_pair.pred_xml, matched_pair.video_path, args.output_dir))
        else:
            # Run on all pairs that have predictions
            valid_pairs = [p for p in pairs if p.pred_xml is not None]
            if not valid_pairs:
                raise FileNotFoundError(f"No pairs with prediction XMLs found in {prediction_root}")
            for pair in valid_pairs:
                # Store in subdirectories under output_dir to prevent collision
                sub_out_dir = args.output_dir / pair.video_stem
                pairs_to_evaluate.append((pair.gt_xml, pair.pred_xml, pair.video_path, sub_out_dir))

    for gt_xml, pred_xml, video_path, out_dir in pairs_to_evaluate:
        config = HardSceneEvalConfig(
            gt_xml=gt_xml,
            pred_xml=pred_xml,
            video_path=video_path,
            output_dir=out_dir,
            match_iou_threshold=args.match_iou_threshold,
            stable_match_iou_threshold=args.stable_match_iou_threshold,
            hard_threshold=args.hard_threshold,
            critical_threshold=args.critical_threshold,
            long_swap_threshold=args.long_swap_threshold,
            event_clip_padding_frames=args.event_clip_padding_frames,
            include_hidden=args.include_hidden,
            top_n_overlay_events=args.top_n_overlay_events,
        )
        metrics = run_hard_scene_evaluation(config)
        print(f"\n--- Hard-Scene Metrics for {gt_xml.stem} ---")
        print(json.dumps(metrics, indent=2, default=str))

    return 0


def _build_compare_parser() -> argparse.ArgumentParser:
    from pig_behavior.evaluation.tracking.assets import (
        TRACKING_GT_DIR,
        VIDEO_DIR,
        PREDICTION_ROOT,
    )
    parser = argparse.ArgumentParser(
        description="Compare multiple tracking configs via hard-scene evaluation.",
    )
    parser.add_argument("--gt-xml", type=Path, required=False, default=None)
    parser.add_argument(
        "--pred",
        action="append",
        required=False,
        default=None,
        metavar="NAME=PATH",
        help="Prediction config as name=path (may be repeated).",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "hard_scene_compare",
    )
    parser.add_argument("--gt-dir", type=Path, default=TRACKING_GT_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--match-iou-threshold", type=float, default=0.30)
    parser.add_argument("--stable-match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--hard-threshold", type=float, default=0.60)
    parser.add_argument("--critical-threshold", type=float, default=0.80)
    parser.add_argument("--long-swap-threshold", type=int, default=15)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Optional benchmark summary CSV to merge HOTA/MOTA/IDF1 columns.",
    )
    return parser


def main_compare(argv: list[str] | None = None) -> int:
    """CLI entry point for ``pig-tracking-hard-eval-compare``."""
    parser = _build_compare_parser()
    args = parser.parse_args(argv)

    import re
    from pig_behavior.evaluation.tracking.assets import list_tracking_pairs

    # 1. Resolve run_dir if we are auto-discovering predictions
    run_dir = args.run_dir
    if not args.pred and run_dir is None:
        # Look up latest run directory under prediction-root
        benchmark_dirs = [
            args.prediction_root / "tracking_detector_benchmark",
            args.prediction_root / "tracking_rule_benchmark",
        ]
        runs = []
        for b_dir in benchmark_dirs:
            if not b_dir.exists():
                continue
            for run_path in b_dir.iterdir():
                if run_path.is_dir() and re.match(r"^\d{8}_\d{6}$", run_path.name):
                    runs.append(run_path)
        if not runs:
            raise FileNotFoundError(
                f"No benchmark runs found under {args.prediction_root}. Please specify --pred or --run-dir."
            )
        runs.sort(key=lambda p: p.name, reverse=True)
        run_dir = runs[0]
        print(f"Auto-discovered latest benchmark run directory: {run_dir}")

    # 2. Determine which videos to evaluate
    eval_targets = []  # list of tuples: (video_stem, output_dir, pred_configs)

    if args.pred:
        # Explicit predictions provided
        pred_configs = {}
        for spec in args.pred:
            if "=" not in spec:
                parser.error(f"--pred must be NAME=PATH, got: {spec}")
            name, path_str = spec.split("=", 1)
            pred_configs[name.strip()] = Path(path_str.strip())
        video_stem = Path(args.video).stem if args.video else None
        eval_targets.append((video_stem, args.output_dir, pred_configs))
    else:
        # Auto-discover from run_dir
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist or is not a directory: {run_dir}")

        video_stems = []
        if args.video:
            video_stems.append(Path(args.video).stem)
        else:
            # Discover video stems using valid project video stems
            pairs = list_tracking_pairs(
                tracking_gt_dir=args.gt_dir,
                video_dir=args.video_dir,
                prediction_root=args.prediction_root,
            )
            valid_stems = {p.video_stem.lower() for p in pairs}
            discovered_stems = set()
            for xml_path in run_dir.rglob("*.xml"):
                for p in xml_path.parts:
                    if p.lower() in valid_stems:
                        for pair in pairs:
                            if pair.video_stem.lower() == p.lower():
                                discovered_stems.add(pair.video_stem)
                                break
            if not discovered_stems:
                raise FileNotFoundError(f"Could not discover any valid video stem directories in {run_dir}")
            video_stems = sorted(discovered_stems)
            print(f"Auto-discovered {len(video_stems)} videos in benchmark run: {video_stems}")

        # Build mapping for each video stem
        for v_stem in video_stems:
            v_pred_configs = {}
            for xml_path in run_dir.rglob("*.xml"):
                is_match = False
                if v_stem.lower() in xml_path.name.lower():
                    is_match = True
                else:
                    for p in xml_path.parts:
                        if p.lower() == v_stem.lower():
                            is_match = True
                            break
                if is_match:
                    rel_path = xml_path.relative_to(run_dir)
                    config_parts = []
                    for p in rel_path.parent.parts:
                        # Exclude video stem
                        if p.lower() == v_stem.lower():
                            continue
                        # Filter out internal benchmark framework folders & timestamps
                        if p.lower() in ("tracking_rule_benchmark", "tracking_detector_benchmark"):
                            continue
                        if re.match(r"^\d{8}_\d{6}$", p):
                            continue
                        config_parts.append(p)
                    config_name = "_".join(config_parts)
                    if not config_name:
                        config_name = "default"
                    v_pred_configs[config_name] = xml_path

            if not v_pred_configs:
                print(f"[Warning] No prediction XMLs matching video '{v_stem}' found under {run_dir}. Skipping.")
                continue

            v_out_dir = args.output_dir / v_stem if len(video_stems) > 1 else args.output_dir
            eval_targets.append((v_stem, v_out_dir, v_pred_configs))

        if not eval_targets:
            raise FileNotFoundError(f"No prediction XMLs found under run directory {run_dir} for video stem(s): {video_stems}")

    # 3. Perform evaluation on all targets
    for v_stem, v_out_dir, v_pred_configs in eval_targets:
        print(f"\n==================================================")
        print(f"Running comparison evaluation for video: {v_stem}")
        print(f"Output folder: {v_out_dir}")
        print(f"==================================================")

        try:
            # Resolve GT XML path
            v_gt_xml = args.gt_xml
            if v_gt_xml is None:
                if v_stem is None:
                    raise ValueError("Either --gt-xml, --video, or --pred must be specified to determine target evaluation video.")
                pairs = list_tracking_pairs(
                    tracking_gt_dir=args.gt_dir,
                    video_dir=args.video_dir,
                    prediction_root=args.prediction_root,
                )
                matched_pair = None
                for pair in pairs:
                    if pair.video_stem.lower() == v_stem.lower() or \
                       (args.video and pair.video_path.resolve() == Path(args.video).resolve()):
                        matched_pair = pair
                        break
                if matched_pair is None:
                    raise FileNotFoundError(f"Could not resolve ground truth XML for video '{v_stem}' from {args.gt_dir}")
                v_gt_xml = matched_pair.gt_xml
                print(f"Auto-resolved ground truth XML for video '{v_stem}': {v_gt_xml}")

            v_out_dir.mkdir(parents=True, exist_ok=True)
            comparison_rows = []

            for name, pred_path in sorted(v_pred_configs.items()):
                sub_dir = v_out_dir / name
                config = HardSceneEvalConfig(
                    gt_xml=v_gt_xml,
                    pred_xml=pred_path,
                    video_path=args.video if len(eval_targets) == 1 else None,  # skip rendering overlay if running multiple
                    output_dir=sub_dir,
                    match_iou_threshold=args.match_iou_threshold,
                    stable_match_iou_threshold=args.stable_match_iou_threshold,
                    hard_threshold=args.hard_threshold,
                    critical_threshold=args.critical_threshold,
                    long_swap_threshold=args.long_swap_threshold,
                    include_hidden=args.include_hidden,
                )
                metrics = run_hard_scene_evaluation(config)
                row = {"config_name": name}
                for key in (
                    "global_id_accuracy",
                    "evaluation_valid",
                    "invalid_reason",
                    "predicted_frame_count",
                    "prediction_frame_coverage",
                    "matched_instance_count",
                    "matched_instance_ratio",
                    "id_accuracy_easy",
                    "id_accuracy_medium",
                    "id_accuracy_hard",
                    "id_accuracy_critical",
                    "total_swap_events",
                    "long_term_swap_count",
                    "max_swap_duration",
                    "mean_swap_duration",
                    "total_wrong_id_frames",
                    "total_missing_instances",
                    "recovery_rate",
                    "mean_reid_latency",
                    "max_reid_latency",
                    "hard_frame_count",
                    "critical_frame_count",
                    "hard_error_rate",
                    "critical_error_rate",
                ):
                    row[key] = metrics.get(key, "")
                comparison_rows.append(row)

            comparison_df = pd.DataFrame(comparison_rows)

            # Optionally merge benchmark CSV
            if args.benchmark_csv and args.benchmark_csv.exists():
                try:
                    bench_df = pd.read_csv(args.benchmark_csv)
                    merge_cols = [
                        c
                        for c in ["hota", "mota", "idf1", "assa", "idsw", "fragments"]
                        if c in bench_df.columns
                    ]
                    if "config_name" in bench_df.columns and merge_cols:
                        comparison_df = comparison_df.merge(
                            bench_df[["config_name", *merge_cols]],
                            on="config_name",
                            how="left",
                        )
                except Exception:
                    logger.warning("Could not merge benchmark CSV.", exc_info=True)

            comparison_df.to_csv(
                v_out_dir / "hard_scene_config_comparison.csv", index=False
            )
            print(comparison_df.to_string(index=False))

        except Exception as e:
            print(f"[Error] Failed comparison evaluation for video '{v_stem}': {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
