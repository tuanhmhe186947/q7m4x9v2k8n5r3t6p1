"""Shared frozen offline-repair entry point and RF compatibility adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.profiles.hybrid_bytetrack import HYBRID_BEST_CONFIG
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
from pig_behavior.tracking.refinement import (
    apply_identity_swap_guard,
    refine_far_camera_hidden_geometry,
    refine_near_wall_hidden_geometry,
    refine_shapes_temporally,
    repair_episode_pair_swaps,
    repair_hidden_suffix_id_swaps,
    repair_local_pair_swaps,
    repair_long_pair_swaps,
    repair_suffix_pair_swaps,
    stabilize_overlap_hidden_islands,
    stabilize_realtime_motion_pairs,
    suppress_overlapped_small_low_confidence_boxes,
)

RF_HYBRID_PROFILE_NAME = "rf_hybrid_offline"
RF_RAW_OUTPUT_SCHEMA_VERSION = "tracking.rf_raw_track_output.v1"
RF_REPAIR_LEDGER_SCHEMA_VERSION = "tracking.rf_offline_repair_ledger.v1"
OFFLINE_REPAIR_SEMANTIC_SHA256 = (
    "e078b5b165dda82dee5b61e9465dc9844446e4cb576a02858c4ed7369828d758"
)

OFFLINE_REPAIR_SEMANTIC_KEYS = (
    "enable_offline_smoothing",
    "far_camera_hidden_geometry_max_center_shift",
    "far_camera_hidden_geometry_max_future_gap_frames",
    "far_camera_hidden_geometry_min_height_excess",
    "far_camera_hidden_geometry_min_overlap_reduction",
    "far_camera_hidden_geometry_min_visible_overlap_iou",
    "far_camera_hidden_geometry_original_weight",
    "far_camera_hidden_geometry_refine",
    "far_camera_hidden_geometry_x_threshold",
    "hidden_owner_guard",
    "hidden_owner_guard_cost_margin",
    "hidden_owner_guard_hold_assignment",
    "hidden_owner_guard_min_missed",
    "hidden_suffix_id_swap_max_hidden_frames",
    "hidden_suffix_id_swap_max_hidden_median_score",
    "hidden_suffix_id_swap_min_hidden_frames",
    "hidden_suffix_id_swap_min_overlap_iou",
    "hidden_suffix_id_swap_min_overlap_persistence_frames",
    "hidden_suffix_id_swap_min_suffix_frames",
    "hidden_suffix_id_swap_repair",
    "hidden_suffix_id_swap_start_back_frames",
    "hidden_suffix_id_swap_use_overlap_persistence",
    "identity_swap_guard",
    "identity_swap_guard_far_x_threshold",
    "identity_swap_guard_skip_mixed_occlusion_hold",
    "identity_swap_guard_skip_mixed_occlusion_hold_far_only",
    "near_wall_hidden_geometry_distance_bbox_scale",
    "near_wall_hidden_geometry_max_center_shift",
    "near_wall_hidden_geometry_max_gap_frames",
    "near_wall_hidden_geometry_min_width_excess",
    "near_wall_hidden_geometry_original_weight",
    "near_wall_hidden_geometry_refine",
    "occlusion_reid_bad_match_action",
    "occlusion_reid_bad_match_include_recent_visible",
    "occlusion_reid_bad_match_max_cost",
    "occlusion_reid_bad_match_max_missed",
    "occlusion_reid_bad_match_min_cost",
    "occlusion_reid_bad_match_min_missed",
    "occlusion_reid_bad_match_occlusion_hold_only",
    "occlusion_reid_bad_match_once_per_episode",
    "occlusion_reid_bad_match_raw_mismatch_only",
    "occlusion_reid_bad_match_same_raw_only",
    "occlusion_reid_bad_match_unowned_raw_only",
    "occlusion_reid_bad_match_visible_min_cost",
    "occlusion_reid_prefer_gap_over_bad_match",
    "overlap_small_box_suppression",
    "reentry_unowned_raw_mismatch_episode_action",
    "reentry_unowned_raw_mismatch_episode_max_cost",
    "reentry_unowned_raw_mismatch_episode_max_events",
    "reentry_unowned_raw_mismatch_episode_max_missed",
    "reentry_unowned_raw_mismatch_episode_min_cost",
    "reentry_unowned_raw_mismatch_episode_min_events",
    "reentry_unowned_raw_mismatch_episode_min_missed",
    "reentry_unowned_raw_mismatch_episode_phases",
    "reentry_unowned_raw_mismatch_episode_reject",
    "reentry_unowned_raw_mismatch_episode_window_frames",
    "refine_boxes",
    "smooth_boxes",
    "suffix_pair_swap_max_overlap_frames",
    "suffix_pair_swap_max_suffix_overlap_iou",
    "suffix_pair_swap_min_overlap_iou",
    "suffix_pair_swap_min_suffix_frames",
    "suffix_pair_swap_repair",
)

_REQUIRED_SHAPE_FIELDS = (
    "attributes",
    "frame",
    "label",
    "occluded",
    "outside",
    "points",
    "score",
    "type",
    "_ambiguous_occlusion",
    "_ever_detected",
    "_missed_frames",
    "_motion_state",
    "_needs_review",
    "_occlusion_hold",
    "_raw_track_id",
    "_state_reason",
    "_track_source",
    "_track_state",
)
_REQUIRED_ATTRIBUTE_NAMES = ("Behavior", "Hidden", "ID")
_TRACK_STATE_VALUES = {"LOST", "MISSING", "OCCLUDED", "VISIBLE"}
_TRACK_SOURCE_VALUES = {"detected", "occlusion_hold", "predicted"}
_ACTOR_LABEL_PATTERN = re.compile(r"^Pig_([1-9][0-9]*)$")
_IDENTITY_PATTERN = re.compile(r"^ID_([1-9][0-9]*)$")

_RF_FORBIDDEN_CORE_FEATURES = (
    "enable_offline_smoothing",
    "far_camera_hidden_geometry_refine",
    "hidden_owner_guard",
    "hidden_owner_guard_hold_assignment",
    "hidden_suffix_id_swap_repair",
    "identity_swap_guard",
    "near_wall_hidden_geometry_refine",
    "occlusion_reid_prefer_gap_over_bad_match",
    "overlap_small_box_suppression",
    "reentry_unowned_raw_mismatch_episode_reject",
    "realtime_motion_pair_stabilizer",
    "suffix_pair_swap_repair",
)


class RepairInputContractError(ValueError):
    """Raised when a raw shape stream cannot safely enter offline repair."""


@dataclass(frozen=True, slots=True)
class RepairLedgerContext:
    """Run authority required for deterministic repair events."""

    source_core: str
    video_key: str


@dataclass(slots=True)
class OfflineRepairResult:
    """Repaired shapes and deterministic, GT-free provenance."""

    shapes: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    input_authority_hash: str
    output_authority_hash: str
    repair_config_hash: str


@dataclass(frozen=True, slots=True)
class _StageSpec:
    name: str
    operation: str
    reason: str
    future_frames_used: bool


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_authority_hash(payload: object) -> str:
    """Return a canonical SHA-256 for JSON-compatible authority payloads."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def offline_repair_semantic_configuration(
    cfg: TrackingConfig,
) -> dict[str, object]:
    """Select the frozen semantic fields without changing their values."""

    return {
        key: getattr(cfg, key)
        for key in sorted(OFFLINE_REPAIR_SEMANTIC_KEYS)
    }


def offline_repair_semantic_hash(cfg: TrackingConfig) -> str:
    """Hash the frozen offline-repair semantic configuration."""

    return canonical_authority_hash(
        offline_repair_semantic_configuration(cfg)
    )


def build_frozen_offline_repair_config() -> TrackingConfig:
    """Build the promoted hybrid repair config without core execution."""

    overrides = deepcopy(HYBRID_BEST_CONFIG)
    cfg = TrackingConfig(
        mode="hybrid_bytetrack",
        overrides=set(overrides),
        **overrides,
    )
    actual_hash = offline_repair_semantic_hash(cfg)
    if actual_hash != OFFLINE_REPAIR_SEMANTIC_SHA256:
        raise RuntimeError(
            "Frozen offline repair semantic hash mismatch: "
            f"{actual_hash} != {OFFLINE_REPAIR_SEMANTIC_SHA256}"
        )
    return cfg


def validate_rf_hybrid_core_config(cfg: TrackingConfig) -> None:
    """Fail closed unless the raw core is exactly realtime_fast."""

    if not cfg.rf_hybrid_offline:
        raise ValueError("rf_hybrid_offline must be explicitly enabled")
    if cfg.mode != "realtime":
        raise ValueError("rf_hybrid_offline requires mode='realtime'")
    if cfg.write_output_video:
        raise ValueError("rf_hybrid_offline forbids MP4 output")
    mismatches = {
        key: (getattr(cfg, key), expected)
        for key, expected in REALTIME_FAST_CONFIG.items()
        if getattr(cfg, key) != expected
    }
    if mismatches:
        mismatch_names = ", ".join(sorted(mismatches))
        raise ValueError(
            "rf_hybrid_offline core must equal realtime_fast; "
            f"mismatched fields: {mismatch_names}"
        )
    forbidden_enabled = [
        key for key in _RF_FORBIDDEN_CORE_FEATURES if bool(getattr(cfg, key))
    ]
    if forbidden_enabled:
        names = ", ".join(sorted(forbidden_enabled))
        raise ValueError(
            "rf_hybrid_offline raw core enabled repair/experimental fields: "
            f"{names}"
        )


def _attributes_by_name(shape: Mapping[str, Any]) -> dict[str, Any]:
    raw_attributes = shape.get("attributes")
    if not isinstance(raw_attributes, list):
        raise RepairInputContractError("attributes must be a list")
    attributes: dict[str, Any] = {}
    for attribute in raw_attributes:
        if not isinstance(attribute, dict):
            raise RepairInputContractError("attribute entries must be objects")
        name = attribute.get("name")
        if not isinstance(name, str) or name in attributes:
            raise RepairInputContractError(
                "attribute names must be unique strings"
            )
        attributes[name] = attribute.get("value")
    missing = sorted(set(_REQUIRED_ATTRIBUTE_NAMES) - set(attributes))
    if missing:
        raise RepairInputContractError(
            f"missing required attributes: {', '.join(missing)}"
        )
    return attributes


def _actor_slot(shape: Mapping[str, Any]) -> int:
    label = shape.get("label")
    if not isinstance(label, str):
        raise RepairInputContractError("label must be a string")
    match = _ACTOR_LABEL_PATTERN.fullmatch(label)
    if match is None:
        raise RepairInputContractError(
            f"invalid fixed actor label: {label!r}"
        )
    return int(match.group(1))


def _validate_numeric(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairInputContractError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RepairInputContractError(f"{field_name} must be finite")
    return numeric


def _validate_shape(
    shape: Mapping[str, Any],
    *,
    index: int,
) -> tuple[int, int, float | None]:
    missing = [field for field in _REQUIRED_SHAPE_FIELDS if field not in shape]
    if missing:
        raise RepairInputContractError(
            f"shape[{index}] missing fields: {', '.join(missing)}"
        )
    frame = shape["frame"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise RepairInputContractError(
            f"shape[{index}].frame must be a non-negative integer"
        )
    if shape["type"] != "rectangle":
        raise RepairInputContractError(
            f"shape[{index}].type must be 'rectangle'"
        )
    slot = _actor_slot(shape)
    points = shape["points"]
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise RepairInputContractError(
            f"shape[{index}].points must contain four coordinates"
        )
    x1, y1, x2, y2 = (
        _validate_numeric(value, f"shape[{index}].points")
        for value in points
    )
    if x2 <= x1 or y2 <= y1:
        raise RepairInputContractError(
            f"shape[{index}].points must satisfy x2>x1 and y2>y1"
        )
    score = _validate_numeric(shape["score"], f"shape[{index}].score")
    if not 0.0 <= score <= 1.0:
        raise RepairInputContractError(
            f"shape[{index}].score must be within [0,1]"
        )
    for field_name in (
        "_ambiguous_occlusion",
        "_ever_detected",
        "_needs_review",
        "_occlusion_hold",
        "occluded",
        "outside",
    ):
        if not isinstance(shape[field_name], bool):
            raise RepairInputContractError(
                f"shape[{index}].{field_name} must be boolean"
            )
    missed = shape["_missed_frames"]
    if isinstance(missed, bool) or not isinstance(missed, int) or missed < 0:
        raise RepairInputContractError(
            f"shape[{index}]._missed_frames must be non-negative"
        )
    if shape["_track_state"] not in _TRACK_STATE_VALUES:
        raise RepairInputContractError(
            f"shape[{index}] has invalid _track_state"
        )
    if shape["_track_source"] not in _TRACK_SOURCE_VALUES:
        raise RepairInputContractError(
            f"shape[{index}] has invalid _track_source"
        )
    attributes = _attributes_by_name(shape)
    if _IDENTITY_PATTERN.fullmatch(str(attributes["ID"])) is None:
        raise RepairInputContractError(
            f"shape[{index}] has invalid ID attribute"
        )
    hidden = attributes["Hidden"]
    if hidden not in {"Yes", "No"}:
        raise RepairInputContractError(
            f"shape[{index}] has invalid Hidden attribute"
        )
    if bool(shape["occluded"]) != (hidden == "Yes"):
        raise RepairInputContractError(
            f"shape[{index}] has inconsistent Hidden/occluded values"
        )
    timestamp_raw = shape.get("_timestamp_seconds")
    timestamp = (
        None
        if timestamp_raw is None
        else _validate_numeric(
            timestamp_raw,
            f"shape[{index}]._timestamp_seconds",
        )
    )
    return int(frame), slot, timestamp


def adapt_rf_shapes_for_offline_repair(
    raw_shapes: Sequence[Mapping[str, Any]],
    *,
    expected_track_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Validate and deep-copy RF shapes without making repair decisions."""

    if not raw_shapes:
        raise RepairInputContractError("raw RF shape stream is empty")
    adapted = deepcopy(list(raw_shapes))
    seen_keys: set[tuple[int, int]] = set()
    slots_by_frame: dict[int, set[int]] = defaultdict(set)
    timestamps_by_frame: dict[int, set[float]] = defaultdict(set)
    timestamp_presence_by_frame: dict[int, int] = defaultdict(int)
    for index, shape in enumerate(adapted):
        if not isinstance(shape, dict):
            raise RepairInputContractError(
                f"shape[{index}] must be a dictionary"
            )
        frame, slot, timestamp = _validate_shape(shape, index=index)
        key = (frame, slot)
        if key in seen_keys:
            raise RepairInputContractError(
                "duplicate frame/actor observation: "
                f"frame={frame}, actor=Pig_{slot}"
            )
        seen_keys.add(key)
        slots_by_frame[frame].add(slot)
        if timestamp is not None:
            timestamps_by_frame[frame].add(timestamp)
            timestamp_presence_by_frame[frame] += 1
    if expected_track_ids is not None:
        expected = {int(value) for value in expected_track_ids}
        for frame, slots in sorted(slots_by_frame.items()):
            if slots != expected:
                raise RepairInputContractError(
                    f"frame={frame} actor slots {sorted(slots)} "
                    f"do not equal expected {sorted(expected)}"
                )
    previous_timestamp: float | None = None
    for frame in sorted(timestamps_by_frame):
        timestamp_values = timestamps_by_frame[frame]
        if len(timestamp_values) != 1:
            raise RepairInputContractError(
                f"frame={frame} has inconsistent timestamps"
            )
        if timestamp_presence_by_frame[frame] != len(slots_by_frame[frame]):
            raise RepairInputContractError(
                f"frame={frame} has partially missing timestamps"
            )
        timestamp = next(iter(timestamp_values))
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise RepairInputContractError(
                "timestamps must increase strictly with frame index"
            )
        previous_timestamp = timestamp
    adapted.sort(key=lambda shape: (int(shape["frame"]), _actor_slot(shape)))
    return adapted


def _shape_identity(shape: Mapping[str, Any]) -> str:
    return str(_attributes_by_name(shape)["ID"])


def _public_shape_payload(shape: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in shape.items() if not key.startswith("_")
    }


def _changed_shape_keys(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str]]:
    before_by_key = {
        (int(shape["frame"]), str(shape["label"])): shape for shape in before
    }
    after_by_key = {
        (int(shape["frame"]), str(shape["label"])): shape for shape in after
    }
    if set(before_by_key) != set(after_by_key):
        raise RuntimeError("offline repair changed the observation universe")
    return [
        key
        for key in sorted(before_by_key)
        if _public_shape_payload(before_by_key[key])
        != _public_shape_payload(after_by_key[key])
    ]


def _contiguous_frame_groups(frames: Sequence[int]) -> list[list[int]]:
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if frame == groups[-1][-1] + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _stage_ledger_events(
    spec: _StageSpec,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    context: RepairLedgerContext,
    repair_config_hash: str,
) -> list[dict[str, Any]]:
    changed_keys = _changed_shape_keys(before, after)
    if not changed_keys:
        return []
    before_by_key = {
        (int(shape["frame"]), str(shape["label"])): shape for shape in before
    }
    after_by_key = {
        (int(shape["frame"]), str(shape["label"])): shape for shape in after
    }
    grouped_frames: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key in changed_keys:
        input_id = _shape_identity(before_by_key[key])
        output_id = _shape_identity(after_by_key[key])
        grouped_frames[(input_id, output_id)].append(key[0])
    input_hash = canonical_authority_hash(list(before))
    output_hash = canonical_authority_hash(list(after))
    events: list[dict[str, Any]] = []
    for (input_id, output_id), raw_frames in sorted(grouped_frames.items()):
        unique_frames = sorted(set(raw_frames))
        for frame_group in _contiguous_frame_groups(unique_frames):
            event = {
                "source_core": context.source_core,
                "video_key": context.video_key,
                "repair_stage": spec.name,
                "start_frame": frame_group[0],
                "end_frame": frame_group[-1],
                "input_track_id": input_id,
                "output_track_id": output_id,
                "repair_operation": spec.operation,
                "repair_reason": spec.reason,
                "future_frames_used": spec.future_frames_used,
                "frames_modified": len(frame_group),
                "repair_config_hash": repair_config_hash,
                "input_authority_hash": input_hash,
                "output_authority_hash": output_hash,
            }
            event["repair_event_id"] = canonical_authority_hash(event)
            events.append(event)
    return events


def apply_offline_repair_stack(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    mask: Any,
    cfg: TrackingConfig,
    *,
    ledger_context: RepairLedgerContext | None = None,
) -> OfflineRepairResult:
    """Run the existing frozen stage order without an RF-specific branch."""

    repair_config_hash = (
        offline_repair_semantic_hash(cfg)
        if ledger_context is not None
        else ""
    )
    input_hash = (
        canonical_authority_hash(shapes)
        if ledger_context is not None
        else ""
    )
    ledger: list[dict[str, Any]] = []
    current = shapes

    def run_stage(
        spec: _StageSpec,
        operation: Any,
        *,
        enabled: bool = True,
    ) -> None:
        nonlocal current
        if not enabled:
            return
        before = deepcopy(current) if ledger_context is not None else ()
        current = operation(current)
        if ledger_context is not None:
            ledger.extend(
                _stage_ledger_events(
                    spec,
                    before,
                    current,
                    ledger_context,
                    repair_config_hash,
                )
            )

    run_stage(
        _StageSpec(
            "identity_swap_guard",
            "identity_payload_swap",
            "frozen_transition_cost_guard",
            False,
        ),
        lambda value: apply_identity_swap_guard(value, width, height, cfg),
        enabled=cfg.enable_offline_smoothing and cfg.identity_swap_guard,
    )
    smoothing_enabled = cfg.enable_offline_smoothing and (
        cfg.smooth_boxes or cfg.refine_boxes
    )
    run_stage(
        _StageSpec(
            "temporal_box_refinement",
            "bbox_refinement",
            "frozen_temporal_anchor_rule",
            True,
        ),
        lambda value: refine_shapes_temporally(value, width, height, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "overlap_hidden_island_stabilization",
            "visibility_stabilization",
            "frozen_overlap_window_rule",
            True,
        ),
        lambda value: stabilize_overlap_hidden_islands(value, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "local_pair_swap_repair",
            "identity_payload_swap",
            "frozen_local_motion_rule",
            False,
        ),
        lambda value: repair_local_pair_swaps(value, width, height, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "episode_pair_swap_repair",
            "identity_payload_swap",
            "frozen_episode_anchor_rule",
            True,
        ),
        lambda value: repair_episode_pair_swaps(value, width, height, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "long_pair_swap_repair",
            "identity_payload_swap",
            "frozen_long_segment_rule",
            True,
        ),
        lambda value: repair_long_pair_swaps(value, width, height, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "suffix_pair_swap_repair",
            "identity_payload_swap",
            "frozen_uncertain_overlap_suffix_rule",
            True,
        ),
        lambda value: repair_suffix_pair_swaps(value, width, height, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "overlap_small_box_suppression",
            "visibility_stabilization",
            "frozen_small_overlap_rule",
            False,
        ),
        lambda value: suppress_overlapped_small_low_confidence_boxes(
            value,
            cfg,
        ),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "hidden_suffix_id_swap_repair",
            "identity_attribute_change",
            "frozen_hidden_overlap_suffix_rule",
            True,
        ),
        lambda value: repair_hidden_suffix_id_swaps(value, cfg),
        enabled=smoothing_enabled,
    )
    run_stage(
        _StageSpec(
            "realtime_motion_pair_stabilizer",
            "identity_attribute_change",
            "frozen_motion_component_rule",
            True,
        ),
        lambda value: stabilize_realtime_motion_pairs(
            value,
            width,
            height,
            cfg,
        ),
    )
    run_stage(
        _StageSpec(
            "near_wall_hidden_geometry_refinement",
            "bbox_refinement",
            "frozen_near_wall_anchor_rule",
            True,
        ),
        lambda value: refine_near_wall_hidden_geometry(
            value,
            width,
            height,
            mask,
            cfg,
        ),
    )
    run_stage(
        _StageSpec(
            "far_camera_hidden_geometry_refinement",
            "bbox_refinement",
            "frozen_far_camera_future_anchor_rule",
            True,
        ),
        lambda value: refine_far_camera_hidden_geometry(
            value,
            width,
            height,
            cfg,
        ),
    )
    output_hash = (
        canonical_authority_hash(current)
        if ledger_context is not None
        else ""
    )
    ledger.sort(
        key=lambda event: (
            str(event["repair_stage"]),
            int(event["start_frame"]),
            str(event["input_track_id"]),
            str(event["output_track_id"]),
        )
    )
    return OfflineRepairResult(
        shapes=current,
        ledger=ledger,
        input_authority_hash=input_hash,
        output_authority_hash=output_hash,
        repair_config_hash=repair_config_hash,
    )


def write_rf_raw_output(
    path: Path,
    *,
    shapes: Sequence[Mapping[str, Any]],
    video_key: str,
    input_authority_hash: str,
) -> None:
    """Write the immutable pre-repair RF authority with internal provenance."""

    payload = {
        "schema_version": RF_RAW_OUTPUT_SCHEMA_VERSION,
        "source_core": "realtime_fast",
        "video_key": video_key,
        "input_authority_hash": input_authority_hash,
        "shapes": list(shapes),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_repair_ledger(
    path: Path,
    result: OfflineRepairResult,
    *,
    video_key: str,
) -> None:
    """Write deterministic GT-free repair provenance."""

    payload = {
        "schema_version": RF_REPAIR_LEDGER_SCHEMA_VERSION,
        "profile": RF_HYBRID_PROFILE_NAME,
        "source_core": "realtime_fast",
        "video_key": video_key,
        "repair_config_hash": result.repair_config_hash,
        "input_authority_hash": result.input_authority_hash,
        "output_authority_hash": result.output_authority_hash,
        "events": result.ledger,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "OFFLINE_REPAIR_SEMANTIC_KEYS",
    "OFFLINE_REPAIR_SEMANTIC_SHA256",
    "OfflineRepairResult",
    "RF_HYBRID_PROFILE_NAME",
    "RepairInputContractError",
    "RepairLedgerContext",
    "adapt_rf_shapes_for_offline_repair",
    "apply_offline_repair_stack",
    "build_frozen_offline_repair_config",
    "canonical_authority_hash",
    "offline_repair_semantic_configuration",
    "offline_repair_semantic_hash",
    "validate_rf_hybrid_core_config",
    "write_repair_ledger",
    "write_rf_raw_output",
]
