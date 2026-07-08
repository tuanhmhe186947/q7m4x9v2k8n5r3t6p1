"""Temporal box refinement and identity-swap correction for tracking shapes."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.geometry import (
    area_log_ratio,
    bbox_iom,
    bbox_iou,
    bbox_size,
    center_distance_norm,
    clip_box,
)

logger = logging.getLogger(__name__)


def _shape_attributes_dict(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        str(attribute["name"]): attribute.get("value")
        for attribute in shape.get("attributes", [])
    }


def _set_shape_id_value(shape: dict[str, Any], id_value: str) -> None:
    for attribute in shape.get("attributes", []):
        if attribute.get("name") == "ID":
            attribute["value"] = id_value
            return
    shape.setdefault("attributes", []).append({"value": id_value, "name": "ID"})


def shape_is_clean_for_training(shape: dict[str, Any], cfg: TrackingConfig) -> bool:
    attributes = _shape_attributes_dict(shape)
    return (
        shape.get("_track_source") == "detected"
        and attributes.get("Hidden", "No") == "No"
        and float(shape.get("score", 0.0)) >= cfg.review_conf
    )


def clean_training_shapes(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    return [shape for shape in shapes if shape_is_clean_for_training(shape, cfg)]


def shape_box(shape: dict[str, Any]) -> np.ndarray:
    return np.asarray(shape["points"], dtype=np.float32)


def set_shape_box(
    shape: dict[str, Any],
    box: np.ndarray,
    width: int,
    height: int,
) -> None:
    clipped = clip_box(box, width, height)
    shape["points"] = [round(float(value), 2) for value in clipped]


def shape_hidden_value(shape: dict[str, Any]) -> str:
    return str(_shape_attributes_dict(shape).get("Hidden", "No"))


def set_shape_hidden(shape: dict[str, Any], hidden: bool) -> None:
    value = "Yes" if hidden else "No"
    for attribute in shape.get("attributes", []):
        if attribute.get("name") == "Hidden":
            attribute["value"] = value
            break
    else:
        shape.setdefault("attributes", []).append({"value": value, "name": "Hidden"})
    shape["occluded"] = bool(hidden)
    if hidden:
        shape["_needs_review"] = True


def suppress_overlapped_small_low_confidence_boxes(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Hide small low-confidence boxes when they overlap a much larger box."""
    if not cfg.overlap_small_box_suppression:
        return shapes

    min_iou = float(cfg.overlap_small_box_min_iou)
    max_area_ratio = float(cfg.overlap_small_box_max_area_ratio)
    max_score = float(cfg.overlap_small_box_max_score)
    if min_iou <= 0.0 or max_area_ratio <= 0.0:
        return shapes

    suppressed_shapes = [shape.copy() for shape in shapes]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in suppressed_shapes:
        if shape.get("outside", False):
            continue
        if shape_hidden_value(shape) == "Yes":
            continue
        by_frame.setdefault(int(shape["frame"]), []).append(shape)

    suppressed_count = 0
    for frame_shapes in by_frame.values():
        indexes_to_hide: set[int] = set()
        for first_idx, first in enumerate(frame_shapes):
            first_box = shape_box(first)
            first_area = float(np.prod(np.maximum(first_box[2:] - first_box[:2], 0.0)))
            if first_area <= 0.0:
                continue
            for second_idx in range(first_idx + 1, len(frame_shapes)):
                second = frame_shapes[second_idx]
                second_box = shape_box(second)
                second_area = float(
                    np.prod(np.maximum(second_box[2:] - second_box[:2], 0.0))
                )
                if second_area <= 0.0:
                    continue
                if bbox_iou(first_box, second_box) < min_iou:
                    continue

                if first_area <= second_area:
                    small_idx, small_area, large_area = (
                        first_idx,
                        first_area,
                        second_area,
                    )
                else:
                    small_idx, small_area, large_area = (
                        second_idx,
                        second_area,
                        first_area,
                    )
                if small_area / max(large_area, 1e-6) > max_area_ratio:
                    continue
                small_shape = frame_shapes[small_idx]
                if float(small_shape.get("score", 1.0)) > max_score:
                    continue
                indexes_to_hide.add(small_idx)

        for idx in indexes_to_hide:
            shape = frame_shapes[idx]
            set_shape_hidden(shape, True)
            shape["_overlap_small_box_suppressed"] = True
            suppressed_count += 1

    if suppressed_count:
        logger.debug("suppressed %d small overlapped low-confidence boxes", suppressed_count)
    return suppressed_shapes


def _contiguous_runs(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append((start, previous))
        start = previous = frame
    runs.append((start, previous))
    return runs


def _median_score(shapes: list[dict[str, Any]]) -> float:
    if not shapes:
        return 0.0
    return float(np.median(np.asarray([float(shape.get("score", 0.0)) for shape in shapes])))


def repair_hidden_suffix_id_swaps(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Swap ID attributes after a low-confidence hidden track crosses a visible one."""
    if not cfg.hidden_suffix_id_swap_repair:
        return shapes

    repaired_shapes = [shape.copy() for shape in shapes]
    by_id_frame: dict[int, dict[int, dict[str, Any]]] = {}
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in repaired_shapes:
        if shape.get("outside", False):
            continue
        fixed_id = shape_fixed_id(shape)
        frame = int(shape["frame"])
        by_id_frame.setdefault(fixed_id, {})[frame] = shape
        by_frame.setdefault(frame, []).append(shape)

    repaired_pairs: set[tuple[int, int]] = set()
    for hidden_id, hidden_frames_by_frame in sorted(by_id_frame.items()):
        hidden_frames = sorted(
            frame
            for frame, shape in hidden_frames_by_frame.items()
            if shape_hidden_value(shape) == "Yes"
        )
        for run_start, run_end in _contiguous_runs(hidden_frames):
            run_length = run_end - run_start + 1
            if run_length < cfg.hidden_suffix_id_swap_min_hidden_frames:
                continue
            if (
                cfg.hidden_suffix_id_swap_max_hidden_frames > 0
                and run_length > cfg.hidden_suffix_id_swap_max_hidden_frames
            ):
                continue
            if run_end + 1 not in hidden_frames_by_frame:
                continue
            if shape_hidden_value(hidden_frames_by_frame[run_end + 1]) == "Yes":
                continue

            hidden_run_shapes = [
                hidden_frames_by_frame[frame]
                for frame in range(run_start, run_end + 1)
                if frame in hidden_frames_by_frame
            ]
            if (
                _median_score(hidden_run_shapes)
                > cfg.hidden_suffix_id_swap_max_hidden_median_score
            ):
                continue

            partner_overlaps: dict[int, list[float]] = {}
            for hidden_shape in hidden_run_shapes:
                frame = int(hidden_shape["frame"])
                for other in by_frame.get(frame, []):
                    partner_id = shape_fixed_id(other)
                    if partner_id == hidden_id:
                        continue
                    if shape_hidden_value(other) == "Yes":
                        continue
                    partner_overlaps.setdefault(partner_id, []).append(
                        bbox_iou(shape_box(hidden_shape), shape_box(other))
                    )
            if not partner_overlaps:
                continue

            partner_id, overlaps = max(
                partner_overlaps.items(),
                key=lambda item: max(item[1]) if item[1] else 0.0,
            )
            if max(overlaps) < cfg.hidden_suffix_id_swap_min_overlap_iou:
                continue
            pair_key = tuple(sorted((hidden_id, partner_id)))
            if pair_key in repaired_pairs:
                continue

            swap_start = max(
                run_start,
                run_end - cfg.hidden_suffix_id_swap_start_back_frames,
            )
            common_suffix_frames = sorted(
                frame
                for frame in set(hidden_frames_by_frame) & set(by_id_frame[partner_id])
                if frame >= swap_start
            )
            if len(common_suffix_frames) < cfg.hidden_suffix_id_swap_min_suffix_frames:
                continue

            hidden_id_value = f"ID_{hidden_id}"
            partner_id_value = f"ID_{partner_id}"
            for frame in common_suffix_frames:
                hidden_shape = hidden_frames_by_frame[frame]
                partner_shape = by_id_frame[partner_id][frame]
                _set_shape_id_value(hidden_shape, partner_id_value)
                _set_shape_id_value(partner_shape, hidden_id_value)
                hidden_shape["_hidden_suffix_id_swap_repair"] = True
                partner_shape["_hidden_suffix_id_swap_repair"] = True
                hidden_shape["_hidden_suffix_id_swap_with"] = partner_id
                partner_shape["_hidden_suffix_id_swap_with"] = hidden_id
                hidden_shape["_hidden_suffix_id_swap_start"] = swap_start
                partner_shape["_hidden_suffix_id_swap_start"] = swap_start
                hidden_shape["_needs_review"] = True
                partner_shape["_needs_review"] = True
            repaired_pairs.add(pair_key)

    if repaired_pairs:
        logger.debug("hidden suffix ID swap repaired %d pairs", len(repaired_pairs))
    return repaired_shapes


def stabilize_overlap_hidden_islands(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Keep short visible islands hidden when they sit inside heavy overlap."""
    threshold = float(cfg.hidden_overlap_iou_threshold)
    window = int(cfg.hidden_overlap_window_frames)
    if threshold <= 0.0 or window < 1:
        return shapes

    stabilized_shapes = [shape.copy() for shape in shapes]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    by_id: dict[int, dict[int, dict[str, Any]]] = {}
    for shape in stabilized_shapes:
        frame = int(shape["frame"])
        fixed_id = shape_fixed_id(shape)
        by_frame.setdefault(frame, []).append(shape)
        by_id.setdefault(fixed_id, {})[frame] = shape

    def nearby_hidden_count(fixed_id: int, frame: int) -> int:
        track = by_id.get(fixed_id, {})
        hidden_count = 0
        for offset in range(1, window + 1):
            for neighbor_frame in (frame - offset, frame + offset):
                neighbor = track.get(neighbor_frame)
                if neighbor is not None and shape_hidden_value(neighbor) == "Yes":
                    hidden_count += 1
        return hidden_count

    for frame_shapes in by_frame.values():
        candidate_shapes = [
            shape for shape in frame_shapes if not shape.get("outside", False)
        ]
        for idx, first in enumerate(candidate_shapes):
            for second in candidate_shapes[idx + 1 :]:
                if bbox_iou(shape_box(first), shape_box(second)) < threshold:
                    continue
                first_hidden_history = nearby_hidden_count(shape_fixed_id(first), int(first["frame"]))
                second_hidden_history = nearby_hidden_count(
                    shape_fixed_id(second),
                    int(second["frame"]),
                )
                if first_hidden_history == second_hidden_history:
                    continue
                hidden_shape = first if first_hidden_history > second_hidden_history else second
                visible_shape = second if hidden_shape is first else first
                set_shape_hidden(hidden_shape, True)
                set_shape_hidden(visible_shape, False)
                hidden_shape["_hidden_overlap_stabilized"] = True
                visible_shape["_hidden_overlap_stabilized"] = True

    return stabilized_shapes


def shape_is_stable_anchor(shape: dict[str, Any], cfg: TrackingConfig) -> bool:
    return (
        shape.get("_track_source") == "detected"
        and shape_hidden_value(shape) == "No"
        and float(shape.get("score", 0.0)) >= cfg.review_conf
    )


def interpolate_box(
    previous_shape: dict[str, Any],
    next_shape: dict[str, Any],
    target_frame: int,
) -> np.ndarray:
    previous_frame = int(previous_shape["frame"])
    next_frame = int(next_shape["frame"])
    if next_frame <= previous_frame:
        return shape_box(previous_shape)
    ratio = (target_frame - previous_frame) / float(next_frame - previous_frame)
    return (1.0 - ratio) * shape_box(previous_shape) + ratio * shape_box(next_shape)


def size_jump_ratio(box: np.ndarray, expected: np.ndarray) -> float:
    width, height = bbox_size(box)
    expected_width, expected_height = bbox_size(expected)
    width_ratio = abs(width / max(expected_width, 1e-6) - 1.0)
    height_ratio = abs(height / max(expected_height, 1e-6) - 1.0)
    return max(width_ratio, height_ratio)


def nearby_anchor_indices(
    track_shapes: list[dict[str, Any]],
    stable_indices: list[int],
    current_index: int,
    cfg: TrackingConfig,
) -> tuple[int | None, int | None]:
    frame = int(track_shapes[current_index]["frame"])
    previous_idx = None
    next_idx = None
    for idx in reversed(stable_indices):
        if idx >= current_index:
            continue
        if frame - int(track_shapes[idx]["frame"]) <= cfg.refine_max_gap_frames:
            previous_idx = idx
        break
    for idx in stable_indices:
        if idx <= current_index:
            continue
        if int(track_shapes[idx]["frame"]) - frame <= cfg.refine_max_gap_frames:
            next_idx = idx
        break
    return previous_idx, next_idx


def refine_original_weight(shape: dict[str, Any], cfg: TrackingConfig) -> float:
    if shape.get("_track_source") != "detected":
        return 0.15
    if float(shape.get("score", 0.0)) < cfg.review_conf:
        return 0.35
    return 0.65


def refine_shapes_temporally(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Use before/after stable boxes to reduce one-frame bbox jumps."""
    if not cfg.refine_boxes:
        return shapes

    refined_shapes = [shape.copy() for shape in shapes]
    for shape in refined_shapes:
        shape["_refined"] = False
        shape["_refine_reason"] = ""

    for fixed_id in range(1, cfg.expected_pigs + 1):
        track_shapes = sorted(
            [
                shape
                for shape in refined_shapes
                if str(shape["label"]) == f"Pig_{fixed_id}"
            ],
            key=lambda item: int(item["frame"]),
        )
        stable_indices = [
            idx
            for idx, shape in enumerate(track_shapes)
            if shape_is_stable_anchor(shape, cfg)
        ]
        if not stable_indices:
            continue

        for idx, shape in enumerate(track_shapes):
            frame = int(shape["frame"])
            previous_idx, next_idx = nearby_anchor_indices(
                track_shapes,
                stable_indices,
                idx,
                cfg,
            )
            if previous_idx is None and next_idx is None:
                continue

            original = shape_box(shape)
            expected = None
            if previous_idx is not None and next_idx is not None:
                expected = interpolate_box(
                    track_shapes[previous_idx],
                    track_shapes[next_idx],
                    frame,
                )
            elif not shape_is_stable_anchor(shape, cfg):
                anchor_idx = previous_idx if previous_idx is not None else next_idx
                if anchor_idx is not None:
                    expected = shape_box(track_shapes[anchor_idx])
            if expected is None:
                continue

            source = str(shape.get("_track_source", "unknown"))
            unstable_detection = (
                source != "detected"
                or shape_hidden_value(shape) == "Yes"
                or float(shape.get("score", 0.0)) < cfg.review_conf
            )
            size_jump = size_jump_ratio(original, expected)
            size_outlier = size_jump > cfg.refine_size_jump_threshold
            if not unstable_detection and not size_outlier:
                continue

            original_weight = refine_original_weight(shape, cfg)
            if size_outlier:
                original_weight = min(original_weight, 0.35)
            if size_outlier and not unstable_detection:
                reason = f"size_jump>{cfg.refine_size_jump_threshold:.2f}"
            elif source != "detected":
                reason = source
            else:
                reason = "low_score_or_hidden"
            refined = original_weight * original + (1.0 - original_weight) * expected
            shape["_original_points"] = [round(float(value), 2) for value in original]
            shape["_refined"] = True
            shape["_refine_reason"] = reason
            shape["_refine_size_jump"] = round(float(size_jump), 4)
            set_shape_box(shape, refined, width, height)

    return refined_shapes


def shape_fixed_id(shape: dict[str, Any]) -> int:
    return int(str(shape["label"]).removeprefix("Pig_"))


def transition_cost(
    previous_box: np.ndarray,
    current_box: np.ndarray,
    width: int,
    height: int,
) -> float:
    center_cost = center_distance_norm(previous_box, current_box, width, height)
    size_cost = min(area_log_ratio(previous_box, current_box), 2.0) / 2.0
    return float(center_cost + 0.10 * size_cost)


def identity_swap_reason(
    previous_first: dict[str, Any],
    previous_second: dict[str, Any],
    current_first: dict[str, Any],
    current_second: dict[str, Any],
    gain: float,
    cfg: TrackingConfig,
) -> str | None:
    previous_iom = bbox_iom(shape_box(previous_first), shape_box(previous_second))
    current_iom = bbox_iom(shape_box(current_first), shape_box(current_second))
    if max(previous_iom, current_iom) >= cfg.identity_swap_iom_threshold:
        return "overlap_continuity"

    uncertain = any(
        shape.get("_needs_review")
        or shape.get("_ambiguous_occlusion")
        or shape.get("_occlusion_hold")
        or shape.get("_track_source") != "detected"
        for shape in (current_first, current_second)
    )
    if uncertain:
        return "review_continuity"
    if gain >= cfg.identity_swap_min_gain * 2.0:
        return "large_continuity_gain"
    return None


def _non_id_attribute_values(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        str(attribute["name"]): attribute.get("value")
        for attribute in shape.get("attributes", [])
        if str(attribute.get("name")) != "ID"
    }


def _apply_non_id_attribute_values(
    shape: dict[str, Any],
    values: dict[str, Any],
) -> None:
    for attribute in shape.get("attributes", []):
        name = str(attribute.get("name"))
        if name != "ID" and name in values:
            attribute["value"] = values[name]


def _shape_payload(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in shape.items()
        if key not in {"label", "attributes", "elements"}
    }


def _apply_shape_payload(
    shape: dict[str, Any],
    payload: dict[str, Any],
    attributes: dict[str, Any],
) -> None:
    for key, value in payload.items():
        shape[key] = value
    _apply_non_id_attribute_values(shape, attributes)


def swap_shape_identity_payloads(
    first: dict[str, Any],
    second: dict[str, Any],
    reason: str,
) -> None:
    first_id = shape_fixed_id(first)
    second_id = shape_fixed_id(second)
    first_payload = _shape_payload(first)
    second_payload = _shape_payload(second)
    first_attrs = _non_id_attribute_values(first)
    second_attrs = _non_id_attribute_values(second)

    _apply_shape_payload(first, second_payload, second_attrs)
    _apply_shape_payload(second, first_payload, first_attrs)
    for shape, other_id in ((first, second_id), (second, first_id)):
        shape["_identity_swap_guard"] = True
        shape["_identity_swap_with"] = int(other_id)
        shape["_identity_swap_reason"] = reason
        shape["_needs_review"] = True


def apply_identity_swap_guard(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Swap per-frame geometry back when a pair assignment breaks continuity."""
    if not cfg.identity_swap_guard:
        return shapes

    guarded_shapes = [shape.copy() for shape in shapes]
    frames = sorted({int(shape["frame"]) for shape in guarded_shapes})
    previous_by_id: dict[int, dict[str, Any]] | None = None

    for frame in frames:
        current_by_id = {
            shape_fixed_id(shape): shape
            for shape in guarded_shapes
            if int(shape["frame"]) == frame
        }
        if previous_by_id is None:
            previous_by_id = {
                fixed_id: shape.copy()
                for fixed_id, shape in current_by_id.items()
            }
            continue

        changed = True
        while changed:
            changed = False
            best_pair: tuple[int, int] | None = None
            best_gain = cfg.identity_swap_min_gain
            best_reason: str | None = None
            ids = sorted(set(previous_by_id).intersection(current_by_id))

            for idx, first_id in enumerate(ids):
                for second_id in ids[idx + 1 :]:
                    prev_first = previous_by_id[first_id]
                    prev_second = previous_by_id[second_id]
                    cur_first = current_by_id[first_id]
                    cur_second = current_by_id[second_id]
                    own_cost = transition_cost(
                        shape_box(prev_first),
                        shape_box(cur_first),
                        width,
                        height,
                    ) + transition_cost(
                        shape_box(prev_second),
                        shape_box(cur_second),
                        width,
                        height,
                    )
                    swapped_cost = transition_cost(
                        shape_box(prev_first),
                        shape_box(cur_second),
                        width,
                        height,
                    ) + transition_cost(
                        shape_box(prev_second),
                        shape_box(cur_first),
                        width,
                        height,
                    )
                    gain = own_cost - swapped_cost
                    if gain <= best_gain:
                        continue
                    reason = identity_swap_reason(
                        prev_first,
                        prev_second,
                        cur_first,
                        cur_second,
                        gain,
                        cfg,
                    )
                    if reason is None:
                        continue
                    best_pair = (first_id, second_id)
                    best_gain = gain
                    best_reason = reason

            if best_pair is not None and best_reason is not None:
                first_id, second_id = best_pair
                swap_shape_identity_payloads(
                    current_by_id[first_id],
                    current_by_id[second_id],
                    best_reason,
                )
                changed = True

        previous_by_id = {
            fixed_id: shape.copy()
            for fixed_id, shape in current_by_id.items()
        }

    return guarded_shapes


def shape_is_visible_for_local_swap(shape: dict[str, Any]) -> bool:
    if shape.get("outside"):
        return False
    attributes = shape.get("attributes") or {}
    if isinstance(attributes, list):
        for attribute in attributes:
            if str(attribute.get("name")) == "Hidden":
                return attribute.get("value", "No") != "Yes"
        return True
    return attributes.get("Hidden", "No") != "Yes"


def shape_center_xy(shape: dict[str, Any]) -> tuple[float, float]:
    box = shape_box(shape)
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def shape_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_box = shape_box(first)
    second_box = shape_box(second)
    ax1, ay1, ax2, ay2 = map(float, first_box)
    bx1, by1, bx2, by2 = map(float, second_box)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    first_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    second_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = first_area + second_area - inter
    return inter / union if union > 0.0 else 0.0


def local_swap_motion_cost(
    first_prev: dict[str, Any],
    second_prev: dict[str, Any],
    first_now: dict[str, Any],
    second_now: dict[str, Any],
    width: int,
    height: int,
    *,
    swapped: bool,
) -> float:
    diagonal = max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)
    first_prev_center = shape_center_xy(first_prev)
    second_prev_center = shape_center_xy(second_prev)
    first_now_center = shape_center_xy(first_now)
    second_now_center = shape_center_xy(second_now)
    if swapped:
        first_now_center, second_now_center = second_now_center, first_now_center
    first_cost = (
        (first_prev_center[0] - first_now_center[0]) ** 2
        + (first_prev_center[1] - first_now_center[1]) ** 2
    ) ** 0.5
    second_cost = (
        (second_prev_center[0] - second_now_center[0]) ** 2
        + (second_prev_center[1] - second_now_center[1]) ** 2
    ) ** 0.5
    return (first_cost + second_cost) / diagonal


def repair_local_pair_swaps(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    if not cfg.local_pair_swap_repair:
        return shapes

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes:
        if shape_is_visible_for_local_swap(shape):
            by_frame.setdefault(int(shape["frame"]), []).append(shape)

    previous_by_id: dict[int, dict[str, Any]] = {}
    previous_frame_by_id: dict[int, int] = {}
    max_allowed_gap = min(
        cfg.local_pair_swap_max_gap_frames,
        cfg.local_pair_swap_window_frames,
    )
    repaired = 0

    for frame in sorted(by_frame):
        current_by_id = {
            shape_fixed_id(shape): shape
            for shape in by_frame[frame]
            if shape_fixed_id(shape) is not None
        }
        ids = sorted(current_by_id)
        swapped_ids: set[int] = set()
        for index, first_id in enumerate(ids):
            if first_id in swapped_ids or first_id not in previous_by_id:
                continue
            for second_id in ids[index + 1 :]:
                if second_id in swapped_ids or second_id not in previous_by_id:
                    continue
                if (
                    frame - previous_frame_by_id.get(first_id, -10_000)
                    > max_allowed_gap
                ):
                    continue
                if (
                    frame - previous_frame_by_id.get(second_id, -10_000)
                    > max_allowed_gap
                ):
                    continue

                first_now = current_by_id[first_id]
                second_now = current_by_id[second_id]
                first_prev = previous_by_id[first_id]
                second_prev = previous_by_id[second_id]
                if (
                    max(shape_iou(first_now, second_now), shape_iou(first_prev, second_prev))
                    < cfg.local_pair_swap_min_overlap_iou
                ):
                    continue

                keep_cost = local_swap_motion_cost(
                    first_prev, second_prev, first_now, second_now, width, height, swapped=False
                )
                swap_cost = local_swap_motion_cost(
                    first_prev, second_prev, first_now, second_now, width, height, swapped=True
                )
                if keep_cost - swap_cost < cfg.local_pair_swap_min_motion_gain:
                    continue

                swap_shape_identity_payloads(first_now, second_now, "local_pair_swap_repair")
                first_now["_local_pair_swap_repair"] = True
                second_now["_local_pair_swap_repair"] = True
                first_now["_local_pair_swap_with"] = second_id
                second_now["_local_pair_swap_with"] = first_id
                swapped_ids.update({first_id, second_id})
                repaired += 1
                break

        for fixed_id, shape in current_by_id.items():
            previous_by_id[fixed_id] = shape
            previous_frame_by_id[fixed_id] = frame

    if repaired:
        logger.debug("local pair swap repair adjusted %d frame-pairs", repaired)
    return shapes


def shape_id_value(shape: dict[str, Any]) -> str | None:
    value = _shape_attributes_dict(shape).get("ID")
    return str(value) if value else None


def realtime_motion_pair_candidates(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
    *,
    allowed_edges: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, float]]]:
    stabilized = [deepcopy(shape) for shape in shapes]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in stabilized:
        if shape_is_visible_for_local_swap(shape):
            by_frame.setdefault(int(shape["frame"]), []).append(shape)

    previous_center_by_id: dict[str, tuple[float, float]] = {}
    previous_frame_by_id: dict[str, int] = {}
    changed_edges: list[tuple[str, str, float]] = []

    for frame in sorted(by_frame):
        frame_shapes = by_frame[frame]
        active_previous = {
            id_value: center
            for id_value, center in previous_center_by_id.items()
            if frame - previous_frame_by_id.get(id_value, frame)
            <= cfg.realtime_motion_pair_memory_frames
        }
        if not active_previous:
            for shape in frame_shapes:
                id_value = shape_id_value(shape)
                if id_value is None:
                    continue
                previous_center_by_id[id_value] = shape_center_xy(shape)
                previous_frame_by_id[id_value] = frame
            continue

        candidates: list[tuple[float, str, int]] = []
        for index, shape in enumerate(frame_shapes):
            current_center = shape_center_xy(shape)
            for id_value, previous_center in active_previous.items():
                distance = normalized_center_distance(
                    current_center,
                    previous_center,
                    width,
                    height,
                )
                if distance <= cfg.realtime_motion_pair_max_jump:
                    candidates.append((distance, id_value, index))
        candidates.sort()

        used_ids: set[str] = set()
        used_indexes: set[int] = set()
        assigned: dict[int, str] = {}
        for _cost, id_value, index in candidates:
            if id_value in used_ids or index in used_indexes:
                continue
            used_ids.add(id_value)
            used_indexes.add(index)
            assigned[index] = id_value

        for index, proposed_id in assigned.items():
            shape = frame_shapes[index]
            current_id = shape_id_value(shape)
            if current_id is None or current_id == proposed_id:
                continue
            current_center = shape_center_xy(shape)
            if current_id in active_previous:
                keep_cost = normalized_center_distance(
                    current_center,
                    active_previous[current_id],
                    width,
                    height,
                )
            else:
                keep_cost = (
                    cfg.realtime_motion_pair_max_jump
                    + cfg.realtime_motion_pair_min_gain
                )
            proposed_cost = normalized_center_distance(
                current_center,
                active_previous[proposed_id],
                width,
                height,
            )
            gain = keep_cost - proposed_cost
            if gain < cfg.realtime_motion_pair_min_gain:
                continue
            edge = tuple(sorted((current_id, proposed_id)))
            if allowed_edges is not None and edge not in allowed_edges:
                continue
            _set_shape_id_value(shape, proposed_id)
            shape["_realtime_motion_pair_stabilizer"] = True
            shape["_needs_review"] = True
            changed_edges.append((edge[0], edge[1], gain))

        for shape in frame_shapes:
            id_value = shape_id_value(shape)
            if id_value is None:
                continue
            previous_center_by_id[id_value] = shape_center_xy(shape)
            previous_frame_by_id[id_value] = frame

    return stabilized, changed_edges


def motion_pair_allowed_edges(
    edges: list[tuple[str, str, float]],
    cfg: TrackingConfig,
) -> set[tuple[str, str]]:
    if cfg.realtime_motion_pair_max_component_size <= 0:
        return {(first_id, second_id) for first_id, second_id, _gain in edges}

    neighbors: dict[str, set[str]] = {}
    edge_support: dict[tuple[str, str], int] = {}
    edge_gains: dict[tuple[str, str], list[float]] = {}
    for first_id, second_id, gain in edges:
        neighbors.setdefault(first_id, set()).add(second_id)
        neighbors.setdefault(second_id, set()).add(first_id)
        edge = tuple(sorted((first_id, second_id)))
        edge_support[edge] = edge_support.get(edge, 0) + 1
        edge_gains.setdefault(edge, []).append(float(gain))

    allowed: set[tuple[str, str]] = set()
    seen: set[str] = set()
    for start in neighbors:
        if start in seen:
            continue
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbors.get(node, set()) - component)
        seen.update(component)
        component_edges = {
            tuple(sorted((first_id, second_id)))
            for first_id in component
            for second_id in neighbors.get(first_id, set()) & component
        }
        component_too_large = len(component) > cfg.realtime_motion_pair_max_component_size
        component_too_dense = (
            cfg.realtime_motion_pair_max_component_edges > 0
            and len(component_edges) > cfg.realtime_motion_pair_max_component_edges
        )
        if component_too_large or component_too_dense:
            max_fallback_edges = int(cfg.realtime_motion_pair_dense_fallback_max_edges)
            support_ratio = float(cfg.realtime_motion_pair_dense_fallback_max_support_ratio)
            min_median_gain = float(
                cfg.realtime_motion_pair_dense_fallback_min_median_gain
            )
            min_edge_gain = float(cfg.realtime_motion_pair_dense_fallback_min_edge_gain)
            if max_fallback_edges <= 0 or support_ratio <= 0.0:
                continue
            max_support = max(edge_support[edge] for edge in component_edges)
            rare_edges = [
                edge
                for edge in component_edges
                if edge_support[edge] <= max_support * support_ratio
                and float(np.median(edge_gains.get(edge, [0.0]))) >= min_median_gain
                and min(edge_gains.get(edge, [0.0])) >= min_edge_gain
            ]
            rare_edges.sort(key=lambda edge: (edge_support[edge], edge))
            allowed.update(rare_edges[:max_fallback_edges])
            continue
        allowed.update(component_edges)
    return allowed


def stabilize_realtime_motion_pairs(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Relabel short-memory motion components for delayed realtime quality."""
    if not cfg.realtime_motion_pair_stabilizer:
        return shapes
    if cfg.mode != "realtime":
        return shapes

    _, planned_edges = realtime_motion_pair_candidates(shapes, width, height, cfg)
    if not planned_edges:
        return shapes
    allowed_edges = motion_pair_allowed_edges(planned_edges, cfg)
    if not allowed_edges:
        return shapes
    stabilized, changed_edges = realtime_motion_pair_candidates(
        shapes,
        width,
        height,
        cfg,
        allowed_edges=allowed_edges,
    )
    simple_min_gain = float(cfg.realtime_motion_pair_simple_min_gain)
    if 0.0 < simple_min_gain < cfg.realtime_motion_pair_min_gain:
        simple_cfg = TrackingConfig(
            mode=cfg.mode,
            realtime_motion_pair_stabilizer=True,
            realtime_motion_pair_max_jump=cfg.realtime_motion_pair_max_jump,
            realtime_motion_pair_min_gain=simple_min_gain,
            realtime_motion_pair_memory_frames=cfg.realtime_motion_pair_memory_frames,
            realtime_motion_pair_max_component_size=(
                cfg.realtime_motion_pair_simple_max_component_size
            ),
            realtime_motion_pair_max_component_edges=(
                max(0, cfg.realtime_motion_pair_simple_max_component_size)
            ),
            realtime_motion_pair_dense_fallback_max_edges=0,
            realtime_motion_pair_dense_fallback_max_support_ratio=0.0,
        )
        _, simple_planned_edges = realtime_motion_pair_candidates(
            stabilized,
            width,
            height,
            simple_cfg,
        )
        simple_allowed_edges = motion_pair_allowed_edges(simple_planned_edges, simple_cfg)
        if simple_allowed_edges:
            stabilized, simple_changed_edges = realtime_motion_pair_candidates(
                stabilized,
                width,
                height,
                simple_cfg,
                allowed_edges=simple_allowed_edges,
            )
            changed_edges.extend(simple_changed_edges)
    if changed_edges:
        logger.debug(
            "realtime motion pair stabilizer relabeled %d shapes",
            len(changed_edges),
        )
    return stabilized


def median_shape_center(shapes: list[dict[str, Any]]) -> tuple[float, float]:
    centers = np.asarray([shape_center_xy(shape) for shape in shapes], dtype=np.float32)
    median = np.median(centers, axis=0)
    return float(median[0]), float(median[1])


def normalized_center_distance(
    first: tuple[float, float],
    second: tuple[float, float],
    width: int,
    height: int,
) -> float:
    diagonal = max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)
    return (((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5) / diagonal


def episode_pair_swap_cost(
    first_before: dict[str, Any],
    second_before: dict[str, Any],
    first_after: dict[str, Any],
    second_after: dict[str, Any],
    first_episode_shapes: list[dict[str, Any]],
    second_episode_shapes: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    swapped: bool,
) -> float:
    first_before_center = shape_center_xy(first_before)
    second_before_center = shape_center_xy(second_before)
    first_after_center = shape_center_xy(first_after)
    second_after_center = shape_center_xy(second_after)
    first_episode_center = median_shape_center(first_episode_shapes)
    second_episode_center = median_shape_center(second_episode_shapes)

    if swapped:
        first_episode_center, second_episode_center = (
            second_episode_center,
            first_episode_center,
        )

    return (
        normalized_center_distance(first_before_center, first_episode_center, width, height)
        + normalized_center_distance(first_episode_center, first_after_center, width, height)
        + normalized_center_distance(second_before_center, second_episode_center, width, height)
        + normalized_center_distance(second_episode_center, second_after_center, width, height)
    )


def find_episode_anchor(
    by_id_frame: dict[int, dict[int, dict[str, Any]]],
    fixed_id: int,
    start_frame: int,
    end_frame: int,
    cfg: TrackingConfig,
    *,
    before: bool,
) -> dict[str, Any] | None:
    frames = by_id_frame.get(fixed_id, {})
    if before:
        candidates = range(start_frame - 1, start_frame - cfg.episode_pair_swap_anchor_window_frames - 1, -1)
    else:
        candidates = range(end_frame + 1, end_frame + cfg.episode_pair_swap_anchor_window_frames + 1)
    for frame in candidates:
        shape = frames.get(frame)
        if shape is not None and shape_is_visible_for_local_swap(shape):
            return shape
    return None


def episode_overlap_runs(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    cfg: TrackingConfig,
) -> list[tuple[int, int]]:
    overlap_frames = sorted(
        frame
        for frame in set(first_frames) & set(second_frames)
        if shape_iou(first_frames[frame], second_frames[frame])
        >= cfg.episode_pair_swap_min_overlap_iou
    )
    if not overlap_frames:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = overlap_frames[0]
    for frame in overlap_frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append((start, previous))
        start = previous = frame
    runs.append((start, previous))
    return [
        (start_frame, end_frame)
        for start_frame, end_frame in runs
        if end_frame - start_frame + 1 <= cfg.episode_pair_swap_max_frames
    ]


def repair_episode_pair_swaps(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    if not cfg.episode_pair_swap_repair:
        return shapes

    by_id_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for shape in shapes:
        if not shape_is_visible_for_local_swap(shape):
            continue
        fixed_id = shape_fixed_id(shape)
        by_id_frame.setdefault(fixed_id, {})[int(shape["frame"])] = shape

    repaired = 0
    ids = sorted(by_id_frame)
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1 :]:
            first_frames = by_id_frame[first_id]
            second_frames = by_id_frame[second_id]
            for start_frame, end_frame in episode_overlap_runs(first_frames, second_frames, cfg):
                first_before = find_episode_anchor(
                    by_id_frame, first_id, start_frame, end_frame, cfg, before=True
                )
                second_before = find_episode_anchor(
                    by_id_frame, second_id, start_frame, end_frame, cfg, before=True
                )
                first_after = find_episode_anchor(
                    by_id_frame, first_id, start_frame, end_frame, cfg, before=False
                )
                second_after = find_episode_anchor(
                    by_id_frame, second_id, start_frame, end_frame, cfg, before=False
                )
                if any(
                    anchor is None
                    for anchor in (
                        first_before,
                        second_before,
                        first_after,
                        second_after,
                    )
                ):
                    continue

                episode_frames = range(start_frame, end_frame + 1)
                first_episode_shapes = [first_frames[frame] for frame in episode_frames if frame in first_frames]
                second_episode_shapes = [second_frames[frame] for frame in episode_frames if frame in second_frames]
                if not first_episode_shapes or not second_episode_shapes:
                    continue

                keep_cost = episode_pair_swap_cost(
                    first_before,
                    second_before,
                    first_after,
                    second_after,
                    first_episode_shapes,
                    second_episode_shapes,
                    width,
                    height,
                    swapped=False,
                )
                swap_cost = episode_pair_swap_cost(
                    first_before,
                    second_before,
                    first_after,
                    second_after,
                    first_episode_shapes,
                    second_episode_shapes,
                    width,
                    height,
                    swapped=True,
                )
                if keep_cost - swap_cost < cfg.episode_pair_swap_min_motion_gain:
                    continue

                for frame in episode_frames:
                    first_shape = first_frames.get(frame)
                    second_shape = second_frames.get(frame)
                    if first_shape is None or second_shape is None:
                        continue
                    swap_shape_identity_payloads(
                        first_shape,
                        second_shape,
                        "episode_pair_swap_repair",
                    )
                    first_shape["_episode_pair_swap_repair"] = True
                    second_shape["_episode_pair_swap_repair"] = True
                    first_shape["_episode_pair_swap_with"] = second_id
                    second_shape["_episode_pair_swap_with"] = first_id
                repaired += 1

    if repaired:
        logger.debug("episode pair swap repair adjusted %d episodes", repaired)
    return shapes


def long_pair_swap_segment_end(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    start_frame: int,
    cfg: TrackingConfig,
) -> int | None:
    common_frames = sorted(
        frame
        for frame in set(first_frames) & set(second_frames)
        if frame >= start_frame
    )
    if not common_frames or common_frames[0] != start_frame:
        return None

    segment_frames = [common_frames[0]]
    previous = common_frames[0]
    for frame in common_frames[1:]:
        if frame - previous > cfg.long_pair_swap_max_gap_frames + 1:
            break
        segment_frames.append(frame)
        previous = frame

    if len(segment_frames) < cfg.long_pair_swap_min_frames:
        return None
    return segment_frames[-1]


def long_pair_swap_median_separation(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
) -> float:
    distances = [
        normalized_center_distance(
            shape_center_xy(first_frames[frame]),
            shape_center_xy(second_frames[frame]),
            width,
            height,
        )
        for frame in range(start_frame, end_frame + 1)
        if frame in first_frames and frame in second_frames
    ]
    if not distances:
        return 0.0
    return float(np.median(np.asarray(distances, dtype=np.float32)))


def repair_long_pair_swaps(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    if not cfg.long_pair_swap_repair:
        return shapes

    by_id_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for shape in shapes:
        if not shape_is_visible_for_local_swap(shape):
            continue
        fixed_id = shape_fixed_id(shape)
        by_id_frame.setdefault(fixed_id, {})[int(shape["frame"])] = shape

    repaired = 0
    consumed: set[tuple[int, int, int, int]] = set()
    ids = sorted(by_id_frame)
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1 :]:
            first_frames = by_id_frame[first_id]
            second_frames = by_id_frame[second_id]
            common_frames = sorted(set(first_frames) & set(second_frames))
            for previous_frame, start_frame in zip(
                common_frames,
                common_frames[1:],
                strict=False,
            ):
                if start_frame - previous_frame > cfg.long_pair_swap_max_gap_frames + 1:
                    continue

                first_prev = first_frames[previous_frame]
                second_prev = second_frames[previous_frame]
                first_now = first_frames[start_frame]
                second_now = second_frames[start_frame]
                keep_cost = local_swap_motion_cost(
                    first_prev,
                    second_prev,
                    first_now,
                    second_now,
                    width,
                    height,
                    swapped=False,
                )
                swap_cost = local_swap_motion_cost(
                    first_prev,
                    second_prev,
                    first_now,
                    second_now,
                    width,
                    height,
                    swapped=True,
                )
                start_gain = keep_cost - swap_cost
                if start_gain < cfg.long_pair_swap_min_start_gain:
                    continue

                end_frame = long_pair_swap_segment_end(
                    first_frames,
                    second_frames,
                    start_frame,
                    cfg,
                )
                if end_frame is None:
                    continue

                median_separation = long_pair_swap_median_separation(
                    first_frames,
                    second_frames,
                    start_frame,
                    end_frame,
                    width,
                    height,
                )
                if median_separation < cfg.long_pair_swap_min_median_separation:
                    continue

                segment_key = (first_id, second_id, start_frame, end_frame)
                if segment_key in consumed:
                    continue
                consumed.add(segment_key)

                for frame in range(start_frame, end_frame + 1):
                    first_shape = first_frames.get(frame)
                    second_shape = second_frames.get(frame)
                    if first_shape is None or second_shape is None:
                        continue
                    swap_shape_identity_payloads(
                        first_shape,
                        second_shape,
                        "long_pair_swap_repair",
                    )
                    first_shape["_long_pair_swap_repair"] = True
                    second_shape["_long_pair_swap_repair"] = True
                    first_shape["_long_pair_swap_with"] = second_id
                    second_shape["_long_pair_swap_with"] = first_id
                    first_shape["_long_pair_swap_start_gain"] = round(float(start_gain), 4)
                    second_shape["_long_pair_swap_start_gain"] = round(float(start_gain), 4)
                repaired += 1
                break

    if repaired:
        logger.debug("long pair swap repair adjusted %d segments", repaired)
    return shapes


def shape_has_suffix_swap_uncertainty(shape: dict[str, Any]) -> bool:
    return (
        bool(shape.get("_occlusion_hold"))
        or str(shape.get("_track_source", "")) != "detected"
        or int(shape.get("_missed_frames", 0)) > 0
        or str(shape.get("_track_state", "")) == "LOST"
    )


def shape_is_present_for_suffix_swap(shape: dict[str, Any]) -> bool:
    return not bool(shape.get("outside"))


def suffix_swap_overlap_runs(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    cfg: TrackingConfig,
) -> list[tuple[int, int]]:
    overlap_frames = sorted(
        frame
        for frame in set(first_frames) & set(second_frames)
        if shape_iou(first_frames[frame], second_frames[frame])
        >= cfg.suffix_pair_swap_min_overlap_iou
    )
    if not overlap_frames:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = overlap_frames[0]
    for frame in overlap_frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append((start, previous))
        start = previous = frame
    runs.append((start, previous))
    return [
        (start_frame, end_frame)
        for start_frame, end_frame in runs
        if end_frame - start_frame + 1 <= cfg.suffix_pair_swap_max_overlap_frames
    ]


def suffix_swap_start_frame(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    start_frame: int,
    end_frame: int,
) -> int | None:
    for frame in range(start_frame, end_frame + 1):
        first_shape = first_frames.get(frame)
        second_shape = second_frames.get(frame)
        if first_shape is None or second_shape is None:
            continue
        first_uncertain = shape_has_suffix_swap_uncertainty(first_shape)
        second_uncertain = shape_has_suffix_swap_uncertainty(second_shape)
        if first_uncertain != second_uncertain:
            return frame
    return None


def suffix_is_stable_after_overlap(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    start_frame: int,
    cfg: TrackingConfig,
) -> bool:
    stable_frames = 0
    for frame in sorted(set(first_frames) & set(second_frames)):
        if frame <= start_frame:
            continue
        if (
            shape_iou(first_frames[frame], second_frames[frame])
            > cfg.suffix_pair_swap_max_suffix_overlap_iou
        ):
            continue
        stable_frames += 1
        if stable_frames >= cfg.suffix_pair_swap_min_suffix_frames:
            return True
    return False


def suffix_swap_start_is_visible(
    first_frames: dict[int, dict[str, Any]],
    second_frames: dict[int, dict[str, Any]],
    start_frame: int,
) -> bool:
    first_shape = first_frames.get(start_frame)
    second_shape = second_frames.get(start_frame)
    if first_shape is None or second_shape is None:
        return False
    return shape_hidden_value(first_shape) == "No" and shape_hidden_value(second_shape) == "No"


def repair_suffix_pair_swaps(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    if not cfg.suffix_pair_swap_repair:
        return shapes
    _ = (width, height)

    by_id_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for shape in shapes:
        if not shape_is_present_for_suffix_swap(shape):
            continue
        fixed_id = shape_fixed_id(shape)
        by_id_frame.setdefault(fixed_id, {})[int(shape["frame"])] = shape

    repaired = 0
    consumed_ids: set[int] = set()
    ids = sorted(by_id_frame)
    for index, first_id in enumerate(ids):
        if first_id in consumed_ids:
            continue
        for second_id in ids[index + 1 :]:
            if second_id in consumed_ids:
                continue
            first_frames = by_id_frame[first_id]
            second_frames = by_id_frame[second_id]
            for run_start, run_end in suffix_swap_overlap_runs(
                first_frames,
                second_frames,
                cfg,
            ):
                swap_start = suffix_swap_start_frame(
                    first_frames,
                    second_frames,
                    run_start,
                    run_end,
                )
                if swap_start is None:
                    continue
                if not suffix_swap_start_is_visible(
                    first_frames,
                    second_frames,
                    swap_start,
                ):
                    continue
                if not suffix_is_stable_after_overlap(
                    first_frames,
                    second_frames,
                    swap_start,
                    cfg,
                ):
                    continue

                common_suffix_frames = sorted(
                    frame
                    for frame in set(first_frames) & set(second_frames)
                    if frame >= swap_start
                )
                for frame in common_suffix_frames:
                    swap_shape_identity_payloads(
                        first_frames[frame],
                        second_frames[frame],
                        "suffix_pair_swap_repair",
                    )
                    first_frames[frame]["_suffix_pair_swap_repair"] = True
                    second_frames[frame]["_suffix_pair_swap_repair"] = True
                    first_frames[frame]["_suffix_pair_swap_with"] = second_id
                    second_frames[frame]["_suffix_pair_swap_with"] = first_id
                    first_frames[frame]["_suffix_pair_swap_start"] = swap_start
                    second_frames[frame]["_suffix_pair_swap_start"] = swap_start
                consumed_ids.update({first_id, second_id})
                repaired += 1
                break
            if first_id in consumed_ids:
                break

    if repaired:
        logger.debug("suffix pair swap repair adjusted %d suffixes", repaired)
    return shapes


__all__ = [
    "_apply_non_id_attribute_values",
    "_apply_shape_payload",
    "_non_id_attribute_values",
    "_shape_attributes_dict",
    "_shape_payload",
    "apply_identity_swap_guard",
    "clean_training_shapes",
    "identity_swap_reason",
    "interpolate_box",
    "nearby_anchor_indices",
    "refine_original_weight",
    "refine_shapes_temporally",
    "repair_episode_pair_swaps",
    "repair_local_pair_swaps",
    "repair_long_pair_swaps",
    "repair_hidden_suffix_id_swaps",
    "repair_suffix_pair_swaps",
    "set_shape_box",
    "shape_box",
    "shape_fixed_id",
    "shape_hidden_value",
    "shape_is_clean_for_training",
    "shape_is_stable_anchor",
    "size_jump_ratio",
    "suppress_overlapped_small_low_confidence_boxes",
    "swap_shape_identity_payloads",
    "transition_cost",
]
