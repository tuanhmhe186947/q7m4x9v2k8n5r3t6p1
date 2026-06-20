"""Evaluate pig ID tracking metrics against CVAT video XML ground truth.

Ground truth XML files are expected in ``data/annotations/tracking`` and are
matched to videos in ``data/videos`` by video stem, for example:

``Tracking_annotation_Pigs291119_000263_30fps.xml`` ->
``Pigs291119_000263_30fps.mp4``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing project metadata."""
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
TRACKING_GT_DIR = DATA_DIR / "annotations" / "tracking"
VIDEO_DIR = DATA_DIR / "videos"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "id_tracking"
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "tracking_metrics"
DETECTOR_WEIGHTS = PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"

__all__ = [
    "DATA_DIR",
    "DETECTOR_WEIGHTS",
    "EVAL_OUTPUT_ROOT",
    "PREDICTION_ROOT",
    "PROJECT_ROOT",
    "TRACKING_GT_DIR",
    "VIDEO_DIR",
    "TrackingMetrics",
    "TrackingObject",
    "TrackingPair",
    "aggregate_metrics",
    "compute_association_accuracy",
    "compute_id_metrics",
    "evaluate_dataset",
    "evaluate_pair",
    "evaluate_tracking",
    "find_prediction_xml",
    "identity_events_for_pair",
    "identity_events_to_dataframe",
    "identity_mapping_for_pair",
    "identity_mapping_to_dataframe",
    "continuity_gaps_for_pair",
    "continuity_gaps_to_dataframe",
    "find_project_root",
    "iou_xyxy",
    "list_tracking_pairs",
    "match_frame",
    "metrics_to_dataframe",
    "normalize_key",
    "pairs_to_dataframe",
    "parse_cvat_video_xml",
    "read_cvat_task_size",
    "read_task_name",
    "remap_prediction_ids",
    "resolve_mask_path",
    "run_tracker_for_pair",
    "video_metadata",
]


@dataclass(slots=True)
class TrackingObject:
    """One box in a frame."""

    frame: int
    obj_id: str
    bbox: tuple[float, float, float, float]
    hidden: bool = False
    source_track_id: str = ""
    label: str = ""


@dataclass(slots=True)
class TrackingPair:
    """Matched ground-truth/prediction assets for one video."""

    video_stem: str
    video_path: Path
    gt_xml: Path
    pred_xml: Path | None = None


@dataclass(slots=True)
class TrackingMetrics:
    """Tracking metrics for one video or aggregate."""

    video_stem: str
    gt_detections: int
    pred_detections: int
    matches: int
    fp: int
    fn: int
    idsw: int
    fragments: int
    tracklets: int
    avg_tracklet_length_frames: float
    gap_tolerance_frames: int
    gap_tolerant_fragments: int
    gap_tolerant_tracklets: int
    gap_tolerant_avg_tracklet_length_frames: float
    gap_tolerant_suppressed_fragments: int
    mota: float
    motp_iou: float
    precision: float
    recall: float
    idf1: float
    idtp: int
    idfp: int
    idfn: int
    deta: float
    assa: float
    hota: float
    evaluated_frames: int
    gt_ids: int
    pred_ids: int
    gt_xml: str = ""
    pred_xml: str = ""
    video_path: str = ""
    remapped_idsw: int = 0
    remapped_fragments: int = 0
    remapped_tracklets: int = 0
    remapped_avg_tracklet_length_frames: float = 0.0
    remapped_gap_tolerant_fragments: int = 0
    remapped_gap_tolerant_tracklets: int = 0
    remapped_gap_tolerant_avg_tracklet_length_frames: float = 0.0
    remapped_gap_tolerant_suppressed_fragments: int = 0
    remapped_mota: float = 0.0
    remapped_idf1: float = 0.0
    remapped_assa: float = 0.0
    remapped_hota: float = 0.0
    remapped_idtp: int = 0
    remapped_idfp: int = 0
    remapped_idfn: int = 0
    idmap_matched_detections: int = 0
    idmap_coverage: float = 0.0


def normalize_key(text: str) -> str:
    """Normalize file names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def id_from_label(label: str, fallback: str) -> str:
    """Convert CVAT labels like Pig_3 to stable IDs like ID_3."""
    match = re.search(r"(?:pig|id)[_\-\s]*(\d+)", label, flags=re.IGNORECASE)
    if match:
        return f"ID_{int(match.group(1))}"
    return fallback


def box_hidden(box_el: ET.Element) -> bool:
    """Return whether a CVAT box has Hidden=Yes."""
    for attr in box_el.findall("attribute"):
        if attr.attrib.get("name") == "Hidden":
            return (attr.text or "").strip().lower() == "yes"
    return False


def box_id(box_el: ET.Element, track_label: str, track_id: str) -> str:
    """Read object identity from box attribute, track label, or track id."""
    for attr in box_el.findall("attribute"):
        if attr.attrib.get("name") == "ID" and attr.text:
            return attr.text.strip()
    return id_from_label(track_label, fallback=f"track_{track_id}")


def is_outside(box_el: ET.Element) -> bool:
    """Return whether a CVAT box is marked outside."""
    return str(box_el.attrib.get("outside", "0")).lower() in {"1", "true", "yes"}


def parse_cvat_video_xml(
    xml_path: Path,
    *,
    include_hidden: bool = False,
) -> dict[int, list[TrackingObject]]:
    """Parse CVAT for video 1.1 XML into frame-indexed boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    by_frame: dict[int, list[TrackingObject]] = defaultdict(list)

    for track_el in root.findall("track"):
        track_id = str(track_el.attrib.get("id", ""))
        label = str(track_el.attrib.get("label", ""))
        for box_el in track_el.findall("box"):
            if is_outside(box_el):
                continue
            hidden = box_hidden(box_el)
            if hidden and not include_hidden:
                continue
            frame = int(box_el.attrib["frame"])
            bbox = (
                float(box_el.attrib["xtl"]),
                float(box_el.attrib["ytl"]),
                float(box_el.attrib["xbr"]),
                float(box_el.attrib["ybr"]),
            )
            obj = TrackingObject(
                frame=frame,
                obj_id=box_id(box_el, label, track_id),
                bbox=bbox,
                hidden=hidden,
                source_track_id=track_id,
                label=label,
            )
            by_frame[frame].append(obj)

    return dict(sorted(by_frame.items()))


def read_cvat_task_size(xml_path: Path) -> int | None:
    """Read task size from CVAT XML metadata."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    size_el = root.find("./meta/task/size")
    if size_el is None or size_el.text is None:
        return None
    try:
        return int(size_el.text)
    except ValueError:
        return None


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_frame(
    gt_objects: list[TrackingObject],
    pred_objects: list[TrackingObject],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match one frame with Hungarian assignment maximizing IoU."""
    if not gt_objects or not pred_objects:
        return []

    ious = np.zeros((len(gt_objects), len(pred_objects)), dtype=float)
    for i, gt in enumerate(gt_objects):
        for j, pred in enumerate(pred_objects):
            ious[i, j] = iou_xyxy(gt.bbox, pred.bbox)

    row_ind, col_ind = linear_sum_assignment(-ious)
    matches = []
    for row, col in zip(row_ind, col_ind, strict=False):
        iou = float(ious[row, col])
        if iou >= iou_threshold:
            matches.append((int(row), int(col), iou))
    return matches


def evaluate_tracking(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float = 0.5,
    video_stem: str = "",
    gap_tolerance_frames: int = 15,
) -> TrackingMetrics:
    """Compute CLEAR MOT, IDF1, and HOTA-style summary metrics."""
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    total_gt = sum(len(items) for items in gt_by_frame.values())
    total_pred = sum(len(items) for items in pred_by_frame.values())

    matches_count = 0
    fp = 0
    fn = 0
    idsw = 0
    fragments = 0
    iou_sum = 0.0

    last_match_for_gt: dict[str, str] = {}
    last_tracked_state: dict[str, bool] = defaultdict(bool)
    active_tracklet_lengths: dict[str, int] = defaultdict(int)
    tracklet_lengths: list[int] = []
    ever_tracked: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    gt_id_counts: Counter[str] = Counter()
    pred_id_counts: Counter[str] = Counter()
    matched_frames_by_gt: dict[str, list[int]] = defaultdict(list)

    for frame in frames:
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        current_gt_ids = {obj.obj_id for obj in gt_objects}
        for gt_id, is_tracked in list(last_tracked_state.items()):
            if is_tracked and gt_id not in current_gt_ids:
                tracklet_lengths.append(active_tracklet_lengths[gt_id])
                active_tracklet_lengths[gt_id] = 0
                last_tracked_state[gt_id] = False

        gt_id_counts.update(obj.obj_id for obj in gt_objects)
        pred_id_counts.update(obj.obj_id for obj in pred_objects)

        matches = match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        )
        matched_gt = {gt_idx for gt_idx, _pred_idx, _iou in matches}
        matched_pred = {pred_idx for _gt_idx, pred_idx, _iou in matches}

        matches_count += len(matches)
        fp += len(pred_objects) - len(matched_pred)
        fn += len(gt_objects) - len(matched_gt)
        iou_sum += sum(iou for _gt_idx, _pred_idx, iou in matches)

        matched_gt_ids_this_frame: set[str] = set()
        for gt_idx, pred_idx, _iou in matches:
            gt_id = gt_objects[gt_idx].obj_id
            pred_id = pred_objects[pred_idx].obj_id
            pair_counts[(gt_id, pred_id)] += 1
            matched_gt_ids_this_frame.add(gt_id)
            matched_frames_by_gt[gt_id].append(frame)

            previous_pred_id = last_match_for_gt.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                idsw += 1
            last_match_for_gt[gt_id] = pred_id

            if gt_id in ever_tracked and not last_tracked_state[gt_id]:
                fragments += 1
            ever_tracked.add(gt_id)
            last_tracked_state[gt_id] = True

        for gt in gt_objects:
            if gt.obj_id not in matched_gt_ids_this_frame:
                if last_tracked_state[gt.obj_id]:
                    tracklet_lengths.append(active_tracklet_lengths[gt.obj_id])
                    active_tracklet_lengths[gt.obj_id] = 0
                last_tracked_state[gt.obj_id] = False

        for gt_id in matched_gt_ids_this_frame:
            active_tracklet_lengths[gt_id] += 1

    for gt_id, is_tracked in last_tracked_state.items():
        if is_tracked:
            tracklet_lengths.append(active_tracklet_lengths[gt_id])

    mota = 1.0 - ((fn + fp + idsw) / total_gt) if total_gt else 0.0
    motp = iou_sum / matches_count if matches_count else 0.0
    precision = matches_count / total_pred if total_pred else 0.0
    recall = matches_count / total_gt if total_gt else 0.0

    idtp, idfp, idfn, idf1 = compute_id_metrics(
        pair_counts,
        gt_id_counts,
        pred_id_counts,
    )
    deta = matches_count / (matches_count + fp + fn) if matches_count + fp + fn else 0
    assa = compute_association_accuracy(pair_counts, gt_id_counts, pred_id_counts)
    hota = math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0
    tracklets = len(tracklet_lengths)
    avg_tracklet_length = (
        float(sum(tracklet_lengths) / tracklets) if tracklets else 0.0
    )
    (
        gap_tolerant_fragments,
        gap_tolerant_tracklets,
        gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments,
    ) = continuity_stats_from_matches(
        matched_frames_by_gt,
        gap_tolerance_frames=gap_tolerance_frames,
    )

    return TrackingMetrics(
        video_stem=video_stem,
        gt_detections=total_gt,
        pred_detections=total_pred,
        matches=matches_count,
        fp=fp,
        fn=fn,
        idsw=idsw,
        fragments=fragments,
        tracklets=tracklets,
        avg_tracklet_length_frames=avg_tracklet_length,
        gap_tolerance_frames=max(0, int(gap_tolerance_frames)),
        gap_tolerant_fragments=gap_tolerant_fragments,
        gap_tolerant_tracklets=gap_tolerant_tracklets,
        gap_tolerant_avg_tracklet_length_frames=gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments=gap_tolerant_suppressed_fragments,
        mota=mota,
        motp_iou=motp,
        precision=precision,
        recall=recall,
        idf1=idf1,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        deta=deta,
        assa=assa,
        hota=hota,
        evaluated_frames=len(frames),
        gt_ids=len(gt_id_counts),
        pred_ids=len(pred_id_counts),
    )


def compute_id_metrics(
    pair_counts: Counter[tuple[str, str]],
    gt_id_counts: Counter[str],
    pred_id_counts: Counter[str],
) -> tuple[int, int, int, float]:
    """Compute IDTP/IDFP/IDFN/IDF1 via global identity assignment."""
    gt_ids = sorted(gt_id_counts)
    pred_ids = sorted(pred_id_counts)
    if not gt_ids or not pred_ids:
        idtp = 0
    else:
        counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
        gt_index = {obj_id: idx for idx, obj_id in enumerate(gt_ids)}
        pred_index = {obj_id: idx for idx, obj_id in enumerate(pred_ids)}
        for (gt_id, pred_id), count in pair_counts.items():
            counts[gt_index[gt_id], pred_index[pred_id]] = count
        row_ind, col_ind = linear_sum_assignment(-counts)
        idtp = int(counts[row_ind, col_ind].sum())

    total_gt = int(sum(gt_id_counts.values()))
    total_pred = int(sum(pred_id_counts.values()))
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    denom = (2 * idtp) + idfp + idfn
    idf1 = (2 * idtp / denom) if denom else 0.0
    return idtp, idfp, idfn, idf1


def compute_association_accuracy(
    pair_counts: Counter[tuple[str, str]],
    gt_id_counts: Counter[str],
    pred_id_counts: Counter[str],
) -> float:
    """Compute HOTA-style association accuracy over matched detections."""
    total_matches = sum(pair_counts.values())
    if not total_matches:
        return 0.0
    weighted_sum = 0.0
    for (gt_id, pred_id), count in pair_counts.items():
        union = gt_id_counts[gt_id] + pred_id_counts[pred_id] - count
        if union > 0:
            weighted_sum += count * (count / union)
    return float(weighted_sum / total_matches)


def matched_identity_counts(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float,
) -> Counter[tuple[str, str]]:
    """Count matched GT/prediction ID pairs over the whole video."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        for gt_idx, pred_idx, _iou in match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        ):
            pair_counts[
                (gt_objects[gt_idx].obj_id, pred_objects[pred_idx].obj_id)
            ] += 1
    return pair_counts


def continuity_stats_from_matches(
    matched_frames_by_gt: dict[str, list[int]],
    *,
    gap_tolerance_frames: int,
) -> tuple[int, int, float, int]:
    """Summarize matched track continuity with short-gap tolerance."""
    tolerance = max(0, int(gap_tolerance_frames))
    tracklet_lengths: list[int] = []
    fragments = 0
    suppressed_fragments = 0

    for frames in matched_frames_by_gt.values():
        ordered_frames = sorted(set(frames))
        if not ordered_frames:
            continue

        current_length = 1
        previous_frame = ordered_frames[0]
        for frame in ordered_frames[1:]:
            gap = frame - previous_frame - 1
            if gap <= tolerance:
                if gap > 0:
                    suppressed_fragments += 1
                current_length += 1
            else:
                tracklet_lengths.append(current_length)
                fragments += 1
                current_length = 1
            previous_frame = frame
        tracklet_lengths.append(current_length)

    tracklets = len(tracklet_lengths)
    avg_length = float(sum(tracklet_lengths) / tracklets) if tracklets else 0.0
    return fragments, tracklets, avg_length, suppressed_fragments


def best_id_mapping(
    pair_counts: Counter[tuple[str, str]],
) -> tuple[dict[str, str], int, int]:
    """Map prediction IDs to GT IDs once, maximizing matched detections."""
    gt_ids = sorted({gt_id for gt_id, _pred_id in pair_counts})
    pred_ids = sorted({pred_id for _gt_id, pred_id in pair_counts})
    if not gt_ids or not pred_ids:
        return {}, 0, 0

    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
    gt_index = {obj_id: idx for idx, obj_id in enumerate(gt_ids)}
    pred_index = {obj_id: idx for idx, obj_id in enumerate(pred_ids)}
    for (gt_id, pred_id), count in pair_counts.items():
        counts[gt_index[gt_id], pred_index[pred_id]] = count

    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping: dict[str, str] = {}
    matched = 0
    for row, col in zip(row_ind, col_ind, strict=False):
        count = int(counts[row, col])
        if count <= 0:
            continue
        mapping[pred_ids[col]] = gt_ids[row]
        matched += count
    return mapping, matched, int(sum(pair_counts.values()))


def remap_prediction_ids(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float,
) -> tuple[dict[int, list[TrackingObject]], dict[str, str], int, float]:
    """Apply one fixed prediction->GT ID mapping for permutation-safe scoring."""
    pair_counts = matched_identity_counts(
        gt_by_frame,
        pred_by_frame,
        iou_threshold=iou_threshold,
    )
    mapping, mapped_matches, total_matches = best_id_mapping(pair_counts)
    remapped: dict[int, list[TrackingObject]] = {}
    for frame, objects in pred_by_frame.items():
        remapped[frame] = [
            TrackingObject(
                frame=obj.frame,
                obj_id=mapping.get(obj.obj_id, obj.obj_id),
                bbox=obj.bbox,
                hidden=obj.hidden,
                source_track_id=obj.source_track_id,
                label=obj.label,
            )
            for obj in objects
        ]
    coverage = mapped_matches / total_matches if total_matches else 0.0
    return remapped, mapping, mapped_matches, coverage


def attach_remapped_metrics(
    metrics: TrackingMetrics,
    remapped: TrackingMetrics,
    *,
    mapped_matches: int,
    coverage: float,
) -> TrackingMetrics:
    """Copy permutation-safe identity fields onto the raw metric row."""
    metrics.remapped_idsw = remapped.idsw
    metrics.remapped_fragments = remapped.fragments
    metrics.remapped_tracklets = remapped.tracklets
    metrics.remapped_avg_tracklet_length_frames = (
        remapped.avg_tracklet_length_frames
    )
    metrics.remapped_gap_tolerant_fragments = remapped.gap_tolerant_fragments
    metrics.remapped_gap_tolerant_tracklets = remapped.gap_tolerant_tracklets
    metrics.remapped_gap_tolerant_avg_tracklet_length_frames = (
        remapped.gap_tolerant_avg_tracklet_length_frames
    )
    metrics.remapped_gap_tolerant_suppressed_fragments = (
        remapped.gap_tolerant_suppressed_fragments
    )
    metrics.remapped_mota = remapped.mota
    metrics.remapped_idf1 = remapped.idf1
    metrics.remapped_assa = remapped.assa
    metrics.remapped_hota = remapped.hota
    metrics.remapped_idtp = remapped.idtp
    metrics.remapped_idfp = remapped.idfp
    metrics.remapped_idfn = remapped.idfn
    metrics.idmap_matched_detections = mapped_matches
    metrics.idmap_coverage = coverage
    return metrics


def video_metadata(video_path: Path) -> dict[str, Any]:
    """Read optional video metadata with OpenCV if available."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    metadata = {
        "video_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "video_fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "video_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "video_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    capture.release()
    return metadata


def list_tracking_pairs(
    *,
    tracking_gt_dir: Path = TRACKING_GT_DIR,
    video_dir: Path = VIDEO_DIR,
    prediction_root: Path = PREDICTION_ROOT,
) -> list[TrackingPair]:
    """Match GT XML files to videos and prediction XML files."""
    videos = [p for p in video_dir.glob("*") if p.suffix.lower() in {".mp4", ".avi"}]
    video_by_key = {normalize_key(p.stem): p for p in videos}
    pairs = []

    for gt_xml in sorted(tracking_gt_dir.glob("*.xml")):
        gt_text = gt_xml.stem
        matched_video = None
        for key, video in video_by_key.items():
            if key in normalize_key(gt_text):
                matched_video = video
                break
        if matched_video is None:
            task_name = read_task_name(gt_xml)
            for key, video in video_by_key.items():
                if key in normalize_key(task_name):
                    matched_video = video
                    break
        if matched_video is None:
            continue

        pred_xml = find_prediction_xml(matched_video.stem, prediction_root)
        pairs.append(
            TrackingPair(
                video_stem=matched_video.stem,
                video_path=matched_video,
                gt_xml=gt_xml,
                pred_xml=pred_xml,
            )
        )
    return pairs


def read_task_name(xml_path: Path) -> str:
    """Read CVAT task name from XML."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return ""
    name_el = root.find("./meta/task/name")
    return name_el.text or "" if name_el is not None else ""


def find_prediction_xml(video_stem: str, prediction_root: Path) -> Path | None:
    """Find a prediction CVAT video XML for a video stem."""
    preferred = (
        prediction_root
        / video_stem
        / f"{video_stem}_annotations_cvat_video_1_1.xml"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path in prediction_root.rglob("*.xml")
        if video_stem.lower() in path.name.lower()
        and "cvat_video" in path.name.lower()
        and ".bak_" not in path.name.lower()
    )
    return candidates[0] if candidates else None


def evaluate_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    gap_tolerance_frames: int = 15,
) -> TrackingMetrics | None:
    """Evaluate one pair when a prediction XML is available."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return None

    gt = parse_cvat_video_xml(pair.gt_xml, include_hidden=include_hidden)
    pred = parse_cvat_video_xml(pair.pred_xml, include_hidden=include_hidden)
    metrics = evaluate_tracking(
        gt,
        pred,
        iou_threshold=iou_threshold,
        video_stem=pair.video_stem,
        gap_tolerance_frames=gap_tolerance_frames,
    )
    remapped_pred, _mapping, mapped_matches, coverage = remap_prediction_ids(
        gt,
        pred,
        iou_threshold=iou_threshold,
    )
    remapped_metrics = evaluate_tracking(
        gt,
        remapped_pred,
        iou_threshold=iou_threshold,
        video_stem=pair.video_stem,
        gap_tolerance_frames=gap_tolerance_frames,
    )
    attach_remapped_metrics(
        metrics,
        remapped_metrics,
        mapped_matches=mapped_matches,
        coverage=coverage,
    )
    metrics.gt_xml = str(pair.gt_xml)
    metrics.pred_xml = str(pair.pred_xml)
    metrics.video_path = str(pair.video_path)
    return metrics


def identity_events_for_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    remap_ids: bool = False,
) -> list[dict[str, Any]]:
    """Return frame-level ID mismatches and switches for one GT/prediction pair."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, include_hidden=include_hidden)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, include_hidden=include_hidden)
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
) -> list[dict[str, Any]]:
    """Return the fixed prediction->GT ID remapping used for paper metrics."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, include_hidden=include_hidden)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, include_hidden=include_hidden)
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
) -> list[dict[str, Any]]:
    """Return matched-track gaps used to diagnose fragmentation."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return []

    tolerance = max(0, int(gap_tolerance_frames))
    gt_by_frame = parse_cvat_video_xml(pair.gt_xml, include_hidden=include_hidden)
    pred_by_frame = parse_cvat_video_xml(pair.pred_xml, include_hidden=include_hidden)
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


def aggregate_metrics(metrics: list[TrackingMetrics]) -> TrackingMetrics:
    """Aggregate metric rows by summing counts and recomputing ratios."""
    if not metrics:
        return TrackingMetrics(
            video_stem="ALL",
            gt_detections=0,
            pred_detections=0,
            matches=0,
            fp=0,
            fn=0,
            idsw=0,
            fragments=0,
            mota=0.0,
            motp_iou=0.0,
            precision=0.0,
            recall=0.0,
            idf1=0.0,
            idtp=0,
            idfp=0,
            idfn=0,
            deta=0.0,
            assa=0.0,
            hota=0.0,
            evaluated_frames=0,
            gt_ids=0,
            pred_ids=0,
            tracklets=0,
            avg_tracklet_length_frames=0.0,
            gap_tolerance_frames=0,
            gap_tolerant_fragments=0,
            gap_tolerant_tracklets=0,
            gap_tolerant_avg_tracklet_length_frames=0.0,
            gap_tolerant_suppressed_fragments=0,
        )

    gt_total = sum(m.gt_detections for m in metrics)
    pred_total = sum(m.pred_detections for m in metrics)
    matches_total = sum(m.matches for m in metrics)
    fp_total = sum(m.fp for m in metrics)
    fn_total = sum(m.fn for m in metrics)
    idsw_total = sum(m.idsw for m in metrics)
    remapped_idsw_total = sum(m.remapped_idsw for m in metrics)
    remapped_fragments_total = sum(m.remapped_fragments for m in metrics)
    tracklets_total = sum(m.tracklets for m in metrics)
    remapped_tracklets_total = sum(m.remapped_tracklets for m in metrics)
    gap_tolerance_frames = max(m.gap_tolerance_frames for m in metrics)
    gap_tolerant_fragments_total = sum(m.gap_tolerant_fragments for m in metrics)
    gap_tolerant_tracklets_total = sum(m.gap_tolerant_tracklets for m in metrics)
    gap_tolerant_suppressed_total = sum(
        m.gap_tolerant_suppressed_fragments for m in metrics
    )
    remapped_gap_tolerant_fragments_total = sum(
        m.remapped_gap_tolerant_fragments for m in metrics
    )
    remapped_gap_tolerant_tracklets_total = sum(
        m.remapped_gap_tolerant_tracklets for m in metrics
    )
    remapped_gap_tolerant_suppressed_total = sum(
        m.remapped_gap_tolerant_suppressed_fragments for m in metrics
    )
    idtp_total = sum(m.idtp for m in metrics)
    idfp_total = sum(m.idfp for m in metrics)
    idfn_total = sum(m.idfn for m in metrics)
    remapped_idtp_total = sum(m.remapped_idtp for m in metrics)
    remapped_idfp_total = sum(m.remapped_idfp for m in metrics)
    remapped_idfn_total = sum(m.remapped_idfn for m in metrics)
    motp_num = sum(m.motp_iou * m.matches for m in metrics)
    deta = (
        matches_total / (matches_total + fp_total + fn_total)
        if matches_total + fp_total + fn_total
        else 0.0
    )
    idf1_denom = (2 * idtp_total) + idfp_total + idfn_total
    idf1 = (2 * idtp_total / idf1_denom) if idf1_denom else 0.0
    assa_num = sum(m.assa * m.matches for m in metrics)
    assa = assa_num / matches_total if matches_total else 0.0
    remapped_idf1_denom = (
        (2 * remapped_idtp_total) + remapped_idfp_total + remapped_idfn_total
    )
    remapped_idf1 = (
        (2 * remapped_idtp_total / remapped_idf1_denom)
        if remapped_idf1_denom
        else 0.0
    )
    remapped_mota = (
        1.0 - ((fn_total + fp_total + remapped_idsw_total) / gt_total)
        if gt_total
        else 0.0
    )
    remapped_assa_num = sum(m.remapped_assa * m.matches for m in metrics)
    remapped_assa = remapped_assa_num / matches_total if matches_total else 0.0
    remapped_hota = (
        math.sqrt(deta * remapped_assa) if deta > 0 and remapped_assa > 0 else 0.0
    )
    idmap_matched_detections = sum(m.idmap_matched_detections for m in metrics)
    idmap_coverage = (
        idmap_matched_detections / matches_total if matches_total else 0.0
    )
    avg_tracklet_length = matches_total / tracklets_total if tracklets_total else 0.0
    remapped_avg_tracklet_length = (
        matches_total / remapped_tracklets_total if remapped_tracklets_total else 0.0
    )
    gap_tolerant_avg_tracklet_length = (
        matches_total / gap_tolerant_tracklets_total
        if gap_tolerant_tracklets_total
        else 0.0
    )
    remapped_gap_tolerant_avg_tracklet_length = (
        matches_total / remapped_gap_tolerant_tracklets_total
        if remapped_gap_tolerant_tracklets_total
        else 0.0
    )

    return TrackingMetrics(
        video_stem="ALL",
        gt_detections=gt_total,
        pred_detections=pred_total,
        matches=matches_total,
        fp=fp_total,
        fn=fn_total,
        idsw=idsw_total,
        fragments=sum(m.fragments for m in metrics),
        mota=1.0 - ((fn_total + fp_total + idsw_total) / gt_total)
        if gt_total
        else 0.0,
        motp_iou=motp_num / matches_total if matches_total else 0.0,
        precision=matches_total / pred_total if pred_total else 0.0,
        recall=matches_total / gt_total if gt_total else 0.0,
        idf1=idf1,
        idtp=idtp_total,
        idfp=idfp_total,
        idfn=idfn_total,
        deta=deta,
        assa=assa,
        hota=math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0,
        evaluated_frames=sum(m.evaluated_frames for m in metrics),
        gt_ids=sum(m.gt_ids for m in metrics),
        pred_ids=sum(m.pred_ids for m in metrics),
        tracklets=tracklets_total,
        avg_tracklet_length_frames=avg_tracklet_length,
        gap_tolerance_frames=gap_tolerance_frames,
        gap_tolerant_fragments=gap_tolerant_fragments_total,
        gap_tolerant_tracklets=gap_tolerant_tracklets_total,
        gap_tolerant_avg_tracklet_length_frames=gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments=gap_tolerant_suppressed_total,
        remapped_idsw=remapped_idsw_total,
        remapped_fragments=remapped_fragments_total,
        remapped_tracklets=remapped_tracklets_total,
        remapped_avg_tracklet_length_frames=remapped_avg_tracklet_length,
        remapped_gap_tolerant_fragments=remapped_gap_tolerant_fragments_total,
        remapped_gap_tolerant_tracklets=remapped_gap_tolerant_tracklets_total,
        remapped_gap_tolerant_avg_tracklet_length_frames=(
            remapped_gap_tolerant_avg_tracklet_length
        ),
        remapped_gap_tolerant_suppressed_fragments=(
            remapped_gap_tolerant_suppressed_total
        ),
        remapped_mota=remapped_mota,
        remapped_idf1=remapped_idf1,
        remapped_assa=remapped_assa,
        remapped_hota=remapped_hota,
        remapped_idtp=remapped_idtp_total,
        remapped_idfp=remapped_idfp_total,
        remapped_idfn=remapped_idfn_total,
        idmap_matched_detections=idmap_matched_detections,
        idmap_coverage=idmap_coverage,
    )


def metrics_to_dataframe(metrics: list[TrackingMetrics]) -> pd.DataFrame:
    """Convert metrics to a dataframe with percent columns."""
    rows = [asdict(metric) for metric in metrics]
    df = pd.DataFrame(rows)
    percent_cols = [
        "mota",
        "motp_iou",
        "precision",
        "recall",
        "idf1",
        "deta",
        "assa",
        "hota",
        "remapped_mota",
        "remapped_idf1",
        "remapped_assa",
        "remapped_hota",
        "idmap_coverage",
    ]
    for col in percent_cols:
        if col in df.columns:
            df[f"{col}_pct"] = (df[col] * 100).round(2)
    return df


def pairs_to_dataframe(pairs: list[TrackingPair]) -> pd.DataFrame:
    """Convert matched assets to a dataframe."""
    rows = []
    for pair in pairs:
        gt_size = read_cvat_task_size(pair.gt_xml)
        metadata = {
            "video_frame_count": None,
            "video_fps": None,
            "video_width": None,
            "video_height": None,
            **video_metadata(pair.video_path),
        }
        rows.append(
            {
                "video_stem": pair.video_stem,
                "video_path": str(pair.video_path),
                "gt_xml": str(pair.gt_xml),
                "pred_xml": str(pair.pred_xml) if pair.pred_xml else "",
                "has_prediction": pair.pred_xml is not None and pair.pred_xml.exists(),
                "gt_task_size": gt_size,
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def evaluate_dataset(
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    gap_tolerance_frames: int = 15,
    tracking_gt_dir: Path = TRACKING_GT_DIR,
    video_dir: Path = VIDEO_DIR,
    prediction_root: Path = PREDICTION_ROOT,
    output_root: Path = EVAL_OUTPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Evaluate all matched GT/prediction pairs and save reports."""
    pairs = list_tracking_pairs(
        tracking_gt_dir=tracking_gt_dir,
        video_dir=video_dir,
        prediction_root=prediction_root,
    )
    asset_df = pairs_to_dataframe(pairs)

    metrics = []
    identity_events = []
    remapped_identity_events = []
    identity_mapping_rows = []
    continuity_gap_rows = []
    for pair in pairs:
        result = evaluate_pair(
            pair,
            iou_threshold=iou_threshold,
            include_hidden=include_hidden,
            gap_tolerance_frames=gap_tolerance_frames,
        )
        if result is not None:
            metrics.append(result)
            identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                )
            )
            remapped_identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                    remap_ids=True,
                )
            )
            identity_mapping_rows.extend(
                identity_mapping_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                )
            )
            continuity_gap_rows.extend(
                continuity_gaps_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                    gap_tolerance_frames=gap_tolerance_frames,
                    remap_ids=True,
                )
            )

    all_rows = metrics + ([aggregate_metrics(metrics)] if metrics else [])
    metrics_df = metrics_to_dataframe(all_rows)

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    asset_df.to_csv(run_dir / "tracking_eval_assets.csv", index=False)
    metrics_df.to_csv(run_dir / "tracking_metrics.csv", index=False)
    identity_events_to_dataframe(identity_events).to_csv(
        run_dir / "tracking_identity_events.csv",
        index=False,
    )
    identity_events_to_dataframe(remapped_identity_events).to_csv(
        run_dir / "tracking_remapped_identity_events.csv",
        index=False,
    )
    identity_mapping_to_dataframe(identity_mapping_rows).to_csv(
        run_dir / "tracking_id_mapping.csv",
        index=False,
    )
    continuity_gaps_to_dataframe(continuity_gap_rows).to_csv(
        run_dir / "tracking_continuity_gaps.csv",
        index=False,
    )
    with (run_dir / "tracking_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iou_threshold": iou_threshold,
                "include_hidden": include_hidden,
                "gap_tolerance_frames": gap_tolerance_frames,
                "tracking_gt_dir": str(tracking_gt_dir),
                "video_dir": str(video_dir),
                "prediction_root": str(prediction_root),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return asset_df, metrics_df, run_dir


def resolve_mask_path() -> Path | None:
    """Find the pen mask after annotations were split into subfolders."""
    candidates = [
        DATA_DIR / "annotations" / "scene" / "mask.png",
        DATA_DIR / "annotations" / "mask.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_tracker_for_pair(
    pair: TrackingPair,
    *,
    weights_path: Path = DETECTOR_WEIGHTS,
    mask_path: Path | None = None,
    output_root: Path = (
        PROJECT_ROOT / "outputs" / "evaluation" / "tracking_predictions"
    ),
    max_frames: int | None = None,
) -> Path:
    """Run the project tracker for one pair and return generated prediction XML."""
    from pig_behavior.data_preparation.tracking_annotation import (
        TrackingConfig,
        run_tracking,
    )

    mask_path = mask_path or resolve_mask_path()
    output_dir = output_root / pair.video_stem
    cfg = TrackingConfig(
        video_path=pair.video_path,
        weights_path=weights_path,
        mask_path=mask_path,
        output_dir=output_dir,
        max_frames=max_frames,
        display_inline=False,
        show=False,
    )
    summary = run_tracking(cfg)
    return Path(summary.cvat_video_xml)


if __name__ == "__main__":
    assets, metrics, output_dir = evaluate_dataset()
    print("[assets]")
    print(assets.to_string(index=False))
    print("[metrics]")
    print(metrics.to_string(index=False))
    print("[output]", output_dir)
