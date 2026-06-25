"""Temporal box refinement and identity-swap correction for tracking shapes."""

from __future__ import annotations

from typing import Any

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.geometry import (
    area_log_ratio,
    bbox_iom,
    bbox_size,
    center_distance_norm,
    clip_box,
)


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
