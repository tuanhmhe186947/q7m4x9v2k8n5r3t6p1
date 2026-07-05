"""Temporal box refinement and identity-swap correction for tracking shapes."""

from __future__ import annotations

import logging
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
    "set_shape_box",
    "shape_box",
    "shape_fixed_id",
    "shape_hidden_value",
    "shape_is_clean_for_training",
    "shape_is_stable_anchor",
    "size_jump_ratio",
    "swap_shape_identity_payloads",
    "transition_cost",
]
