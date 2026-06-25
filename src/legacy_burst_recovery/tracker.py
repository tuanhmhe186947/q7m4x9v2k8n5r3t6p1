from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .detector import Detection

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedBox:
    frame_index: int
    bbox: BBox
    bbox_source: str
    det_confidence: float | None
    track_confidence: float
    is_anchor_frame: bool
    is_gt_support_frame: bool
    is_interpolated: bool
    tracking_status: str
    qa_status: str
    qa_notes: str
    legacy_gt_bbox_available: bool = False
    detector_best_iou_with_legacy_gt: float | None = None
    detector_disagrees_with_legacy_gt: bool = False
    segment_start_gt_frame: int | None = None
    segment_end_gt_frame: int | None = None
    segment_tracking_status: str = ""
    id_switch_risk_score: float = 0.0


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1) * max(1.0, y2 - y1)


def association_score(previous: BBox, detection: Detection) -> float:
    iou = bbox_iou(previous, detection.bbox)
    pcx, pcy = bbox_center(previous)
    dcx, dcy = bbox_center(detection.bbox)
    prev_diag = max(1.0, (bbox_area(previous) ** 0.5))
    distance_penalty = min(1.0, hypot(pcx - dcx, pcy - dcy) / (prev_diag * 2.0))
    size_ratio = bbox_area(detection.bbox) / bbox_area(previous)
    size_penalty = min(1.0, abs(1.0 - min(size_ratio, 1.0 / max(size_ratio, 1e-6))))
    return 0.55 * iou + 0.25 * (1.0 - distance_penalty) + 0.15 * detection.confidence + 0.05 * (1.0 - size_penalty)


def choose_detection(previous: BBox, detections: list[Detection]) -> tuple[Detection | None, float]:
    if not detections:
        return None, 0.0
    scored = sorted(
        ((association_score(previous, det), det) for det in detections),
        key=lambda item: item[0],
        reverse=True,
    )
    return scored[0][1], scored[0][0]


def interpolate_bbox(start: BBox, end: BBox, fraction: float) -> BBox:
    return tuple(float(s + (e - s) * fraction) for s, e in zip(start, end, strict=True))  # type: ignore[return-value]


def _best_iou_with_bbox(bbox: BBox, detections: list[Detection]) -> tuple[Detection | None, float]:
    if not detections:
        return None, 0.0
    scored = sorted(
        ((bbox_iou(bbox, detection.bbox), detection) for detection in detections),
        key=lambda item: item[0],
        reverse=True,
    )
    return scored[0][1], float(scored[0][0])


def _gt_segment(frame_index: int, gt_by_frame: dict[int, dict[str, object]]) -> tuple[int | None, int | None]:
    frames = sorted(gt_by_frame)
    start = max((frame for frame in frames if frame <= frame_index), default=None)
    end = min((frame for frame in frames if frame >= frame_index), default=None)
    return start, end


def _track_dense_range_multi_anchor(
    dense_frames: list[int],
    anchor_bbox: BBox,
    detections_by_frame: dict[int, list[Detection]],
    legacy_gt_by_frame: dict[int, dict[str, object]],
    no_detection_mode: bool,
) -> list[TrackedBox]:
    tracked: list[TrackedBox] = []
    if not legacy_gt_by_frame and dense_frames:
        legacy_gt_by_frame = {int(dense_frames[0]): {"bbox": anchor_bbox}}

    previous = anchor_bbox
    for frame_index in dense_frames:
        detections = detections_by_frame.get(frame_index, [])
        start_gt_frame, end_gt_frame = _gt_segment(frame_index, legacy_gt_by_frame)
        gt_record = legacy_gt_by_frame.get(frame_index)
        is_anchor = frame_index == dense_frames[0]

        if gt_record is not None:
            gt_bbox = tuple(float(v) for v in gt_record["bbox"])  # type: ignore[index]
            best, best_iou = _best_iou_with_bbox(gt_bbox, detections)
            disagrees = bool(best is not None and best_iou < 0.3)
            notes = "detector_disagrees_with_legacy_gt" if disagrees else ""
            status = "corrected_by_gt" if disagrees else "ok_gt"
            tracked_box = TrackedBox(
                frame_index=frame_index,
                bbox=gt_bbox,  # type: ignore[arg-type]
                bbox_source="gt_legacy",
                det_confidence=best.confidence if best else None,
                track_confidence=1.0,
                is_anchor_frame=is_anchor,
                is_gt_support_frame=True,
                is_interpolated=False,
                tracking_status=status,
                qa_status="ok",
                qa_notes=notes,
                legacy_gt_bbox_available=True,
                detector_best_iou_with_legacy_gt=best_iou if best else None,
                detector_disagrees_with_legacy_gt=disagrees,
                segment_start_gt_frame=frame_index,
                segment_end_gt_frame=frame_index,
                segment_tracking_status="gt",
                id_switch_risk_score=0.0,
            )
            previous = tracked_box.bbox
            tracked.append(tracked_box)
            continue

        expected_bbox: BBox | None = None
        if start_gt_frame is not None and end_gt_frame is not None and start_gt_frame != end_gt_frame:
            start_bbox = tuple(float(v) for v in legacy_gt_by_frame[start_gt_frame]["bbox"])  # type: ignore[index]
            end_bbox = tuple(float(v) for v in legacy_gt_by_frame[end_gt_frame]["bbox"])  # type: ignore[index]
            fraction = (frame_index - start_gt_frame) / max(1, end_gt_frame - start_gt_frame)
            expected_bbox = interpolate_bbox(start_bbox, end_bbox, fraction)
        elif start_gt_frame is not None:
            expected_bbox = tuple(float(v) for v in legacy_gt_by_frame[start_gt_frame]["bbox"])  # type: ignore[index]
        else:
            expected_bbox = previous

        if no_detection_mode:
            tracked_box = TrackedBox(
                frame_index=frame_index,
                bbox=expected_bbox,
                bbox_source=(
                    "interpolated_between_gt"
                    if start_gt_frame is not None and end_gt_frame is not None
                    else "tracker"
                ),
                det_confidence=None,
                track_confidence=0.65 if start_gt_frame is not None and end_gt_frame is not None else 0.0,
                is_anchor_frame=is_anchor,
                is_gt_support_frame=False,
                is_interpolated=start_gt_frame is not None and end_gt_frame is not None,
                tracking_status="ok" if start_gt_frame is not None and end_gt_frame is not None else "failed",
                qa_status="ok" if start_gt_frame is not None and end_gt_frame is not None else "needs_review",
                qa_notes=(
                    ""
                    if start_gt_frame is not None and end_gt_frame is not None
                    else "detector_disabled_manifest_only"
                ),
                segment_start_gt_frame=start_gt_frame,
                segment_end_gt_frame=end_gt_frame,
                segment_tracking_status="interpolated_between_gt",
            )
            previous = tracked_box.bbox
            tracked.append(tracked_box)
            continue

        best_path_detection, path_iou = _best_iou_with_bbox(expected_bbox, detections)
        best_previous_detection, _assoc_score = choose_detection(previous, detections)
        best = best_path_detection if best_path_detection is not None else best_previous_detection
        det_confidence = best.confidence if best else None

        high_conf_wrong_pig = bool(best is not None and best.confidence >= 0.5 and path_iou < 0.3)
        if best_path_detection is not None and path_iou >= 0.45:
            previous = best_path_detection.bbox
            tracked.append(
                TrackedBox(
                    frame_index=frame_index,
                    bbox=best_path_detection.bbox,
                    bbox_source="detector",
                    det_confidence=best_path_detection.confidence,
                    track_confidence=max(0.5, 0.6 * path_iou + 0.4 * best_path_detection.confidence),
                    is_anchor_frame=is_anchor,
                    is_gt_support_frame=False,
                    is_interpolated=False,
                    tracking_status="ok",
                    qa_status="ok",
                    qa_notes="",
                    segment_start_gt_frame=start_gt_frame,
                    segment_end_gt_frame=end_gt_frame,
                    segment_tracking_status="constrained_detector",
                    id_switch_risk_score=max(0.0, 1.0 - path_iou),
                )
            )
        else:
            risk = 1.0 if high_conf_wrong_pig else max(0.0, 1.0 - path_iou if best is not None else 0.2)
            status = "low_confidence" if high_conf_wrong_pig else "ok"
            qa_status = "review" if high_conf_wrong_pig else "ok"
            notes = "id_switch_risk_preferred_gt_interpolation" if high_conf_wrong_pig else ""
            previous = expected_bbox
            tracked.append(
                TrackedBox(
                    frame_index=frame_index,
                    bbox=expected_bbox,
                    bbox_source=(
                        "interpolated_between_gt"
                        if start_gt_frame is not None and end_gt_frame is not None
                        else "interpolated"
                    ),
                    det_confidence=det_confidence,
                    track_confidence=0.45 if high_conf_wrong_pig else 0.65,
                    is_anchor_frame=is_anchor,
                    is_gt_support_frame=False,
                    is_interpolated=True,
                    tracking_status=status,
                    qa_status=qa_status,
                    qa_notes=notes,
                    segment_start_gt_frame=start_gt_frame,
                    segment_end_gt_frame=end_gt_frame,
                    segment_tracking_status="id_switch_risk" if high_conf_wrong_pig else "interpolated_between_gt",
                    id_switch_risk_score=risk,
                )
            )
    return tracked


def track_dense_range(
    dense_frames: list[int],
    anchor_bbox: BBox,
    detections_by_frame: dict[int, list[Detection]],
    gt_support_frames: list[int],
    no_detection_mode: bool = False,
    legacy_gt_by_frame: dict[int, dict[str, object]] | None = None,
    legacy_gt_mode: str = "single_anchor",
) -> list[TrackedBox]:
    if legacy_gt_mode == "multi_anchor":
        return _track_dense_range_multi_anchor(
            dense_frames,
            anchor_bbox,
            detections_by_frame,
            legacy_gt_by_frame or {},
            no_detection_mode,
        )

    tracked: list[TrackedBox] = []
    previous = anchor_bbox
    support = set(gt_support_frames)
    anchor = dense_frames[0]
    for frame_index in dense_frames:
        is_anchor = frame_index == anchor
        is_support = frame_index in support
        detections = detections_by_frame.get(frame_index, [])
        if is_anchor:
            best, _score = choose_detection(anchor_bbox, detections)
            notes = ""
            status = "ok"
            qa_status = "ok"
            det_confidence = best.confidence if best else None
            if best and bbox_iou(anchor_bbox, best.bbox) < 0.3:
                notes = "detector_disagrees_with_gt_anchor"
                status = "corrected_by_gt"
                qa_status = "review"
            tracked_box = TrackedBox(
                frame_index=frame_index,
                bbox=anchor_bbox,
                bbox_source="gt_anchor",
                det_confidence=det_confidence,
                track_confidence=1.0 if not no_detection_mode else 0.0,
                is_anchor_frame=True,
                is_gt_support_frame=True,
                is_interpolated=False,
                tracking_status=status if not no_detection_mode else "failed",
                qa_status=qa_status if not no_detection_mode else "needs_review",
                qa_notes=notes if not no_detection_mode else "detector_disabled_manifest_only",
            )
            previous = tracked_box.bbox
            tracked.append(tracked_box)
            continue

        if no_detection_mode:
            tracked.append(
                TrackedBox(
                    frame_index=frame_index,
                    bbox=previous,
                    bbox_source="tracker",
                    det_confidence=None,
                    track_confidence=0.0,
                    is_anchor_frame=False,
                    is_gt_support_frame=is_support,
                    is_interpolated=False,
                    tracking_status="failed",
                    qa_status="needs_review",
                    qa_notes="detector_disabled_manifest_only",
                )
            )
            continue

        best, score = choose_detection(previous, detections)
        if best and score >= 0.35:
            previous = best.bbox
            tracked.append(
                TrackedBox(
                    frame_index=frame_index,
                    bbox=best.bbox,
                    bbox_source="detector" if is_support else "tracker",
                    det_confidence=best.confidence,
                    track_confidence=score,
                    is_anchor_frame=False,
                    is_gt_support_frame=is_support,
                    is_interpolated=False,
                    tracking_status="ok" if score >= 0.5 else "low_confidence",
                    qa_status="ok" if score >= 0.5 else "review",
                    qa_notes="" if score >= 0.5 else "weak_association_score",
                )
            )
        else:
            tracked.append(
                TrackedBox(
                    frame_index=frame_index,
                    bbox=previous,
                    bbox_source="interpolated",
                    det_confidence=None,
                    track_confidence=0.2,
                    is_anchor_frame=False,
                    is_gt_support_frame=is_support,
                    is_interpolated=True,
                    tracking_status="interpolated",
                    qa_status="review",
                    qa_notes="missing_or_unreliable_detection_short_gap",
                )
            )
    return tracked
