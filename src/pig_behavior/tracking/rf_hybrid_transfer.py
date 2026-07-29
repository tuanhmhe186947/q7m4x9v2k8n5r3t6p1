"""Predeclared transfer mechanisms applied to frozen realtime_fast tracklets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    shape_id_value,
    stabilize_overlap_hidden_islands,
    stabilize_realtime_motion_pairs,
    suppress_overlapped_small_low_confidence_boxes,
)

RF_HYBRID_TRANSFER_SCHEMA = "tracking.rf_hybrid_transfer.v1"
RF_HYBRID_LEDGER_SCHEMA = "tracking.rf_hybrid_change_ledger.v1"
RF_HYBRID_TRANSFER_STAGE_IDS = (
    "OFFLINE_IDENTITY_SWAP_GUARD",
    "TEMPORAL_BBOX_REFINEMENT",
    "OVERLAP_HIDDEN_ISLAND_STABILIZATION",
    "LOCAL_EPISODE_LONG_PAIR_REPAIRS",
    "SUFFIX_PAIR_SWAP_REPAIR",
    "OVERLAP_SMALL_BOX_SUPPRESSION",
    "H5B_HIDDEN_SUFFIX_OVERLAP_PERSISTENCE",
    "REALTIME_MOTION_PAIR_STABILIZER",
    "NEAR_WALL_HIDDEN_GEOMETRY",
    "FAR_CAMERA_GEOMETRY_DURING_H5B",
)

RF_HYBRID_TRANSFER_PARAMETER_KEYS = (
    "enable_offline_smoothing",
    "identity_swap_guard",
    "identity_swap_guard_skip_mixed_occlusion_hold",
    "identity_swap_guard_skip_mixed_occlusion_hold_far_only",
    "identity_swap_guard_far_x_threshold",
    "identity_swap_min_gain",
    "identity_swap_iom_threshold",
    "smooth_boxes",
    "refine_boxes",
    "refine_max_gap_frames",
    "refine_max_previous_gap_frames",
    "refine_size_jump_threshold",
    "max_box_scale_change_per_frame",
    "max_box_scale_change_after_gap",
    "high_conf_smooth_alpha",
    "mid_conf_smooth_alpha",
    "low_conf_smooth_alpha",
    "local_pair_swap_repair",
    "local_pair_swap_window_frames",
    "local_pair_swap_max_gap_frames",
    "local_pair_swap_min_overlap_iou",
    "local_pair_swap_min_motion_gain",
    "episode_pair_swap_repair",
    "episode_pair_swap_max_frames",
    "episode_pair_swap_anchor_window_frames",
    "episode_pair_swap_min_overlap_iou",
    "episode_pair_swap_min_motion_gain",
    "long_pair_swap_repair",
    "long_pair_swap_min_frames",
    "long_pair_swap_max_gap_frames",
    "long_pair_swap_min_start_gain",
    "long_pair_swap_min_median_separation",
    "suffix_pair_swap_repair",
    "suffix_pair_swap_min_overlap_iou",
    "suffix_pair_swap_max_overlap_frames",
    "suffix_pair_swap_min_suffix_frames",
    "suffix_pair_swap_max_suffix_overlap_iou",
    "overlap_small_box_suppression",
    "overlap_small_box_min_iou",
    "overlap_small_box_max_area_ratio",
    "overlap_small_box_max_score",
    "hidden_suffix_id_swap_repair",
    "hidden_suffix_id_swap_min_hidden_frames",
    "hidden_suffix_id_swap_max_hidden_frames",
    "hidden_suffix_id_swap_min_overlap_iou",
    "hidden_suffix_id_swap_max_hidden_median_score",
    "hidden_suffix_id_swap_start_back_frames",
    "hidden_suffix_id_swap_min_suffix_frames",
    "hidden_suffix_id_swap_use_overlap_persistence",
    "hidden_suffix_id_swap_min_overlap_persistence_frames",
    "realtime_motion_pair_stabilizer",
    "realtime_motion_pair_fixed_lag_frames",
    "realtime_motion_pair_max_jump",
    "realtime_motion_pair_min_gain",
    "realtime_motion_pair_memory_frames",
    "realtime_motion_pair_max_component_size",
    "realtime_motion_pair_max_component_edges",
    "realtime_motion_pair_dense_fallback_max_edges",
    "realtime_motion_pair_dense_fallback_max_support_ratio",
    "realtime_motion_pair_dense_fallback_min_median_gain",
    "realtime_motion_pair_dense_fallback_min_edge_gain",
    "realtime_motion_pair_simple_min_gain",
    "realtime_motion_pair_simple_max_component_size",
    "near_wall_hidden_geometry_refine",
    "near_wall_hidden_geometry_max_gap_frames",
    "near_wall_hidden_geometry_distance_bbox_scale",
    "near_wall_hidden_geometry_min_width_excess",
    "near_wall_hidden_geometry_max_center_shift",
    "near_wall_hidden_geometry_original_weight",
    "far_camera_hidden_geometry_refine",
    "far_camera_hidden_geometry_x_threshold",
    "far_camera_hidden_geometry_max_future_gap_frames",
    "far_camera_hidden_geometry_min_height_excess",
    "far_camera_hidden_geometry_min_visible_overlap_iou",
    "far_camera_hidden_geometry_min_overlap_reduction",
    "far_camera_hidden_geometry_max_center_shift",
    "far_camera_hidden_geometry_original_weight",
)

_FORBIDDEN_RF_CORE_FLAGS = (
    "enable_offline_smoothing",
    "identity_swap_guard",
    "hidden_owner_guard",
    "hidden_suffix_id_swap_repair",
    "near_wall_hidden_geometry_refine",
    "far_camera_hidden_geometry_refine",
    "realtime_motion_pair_stabilizer",
)


class RFHybridContractError(ValueError):
    """Raised when input is not a frozen generic realtime tracklet table."""


@dataclass(slots=True)
class RFHybridTransferResult:
    """Transferred shapes plus deterministic stage and change evidence."""

    shapes: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    stage_activation: list[dict[str, Any]]
    input_authority_hash: str
    output_authority_hash: str
    transfer_config_hash: str


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        _json_value(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_transfer_hash(payload: object) -> str:
    """Hash a JSON-compatible transfer artifact without mutating it."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_rf_hybrid_transfer_config() -> TrackingConfig:
    """Bind only predeclared portable parameters from the accepted lineage."""

    historical = TrackingConfig(
        mode="hybrid_bytetrack",
        **deepcopy(HYBRID_BEST_CONFIG),
    )
    values = {
        key: deepcopy(getattr(historical, key))
        for key in RF_HYBRID_TRANSFER_PARAMETER_KEYS
    }
    return TrackingConfig(mode="realtime", **values)


def rf_hybrid_transfer_configuration() -> dict[str, object]:
    """Return the complete frozen transfer parameter binding."""

    cfg = build_rf_hybrid_transfer_config()
    return {
        key: _json_value(getattr(cfg, key))
        for key in RF_HYBRID_TRANSFER_PARAMETER_KEYS
    }


def rf_hybrid_transfer_config_hash() -> str:
    """Return the frozen transfer configuration hash."""

    return canonical_transfer_hash(rf_hybrid_transfer_configuration())


def validate_rf_hybrid_core_config(cfg: TrackingConfig) -> None:
    """Fail closed unless the prediction producer is exactly realtime_fast."""

    if not cfg.rf_hybrid_transfer:
        raise RFHybridContractError("rf_hybrid_transfer must be enabled")
    if cfg.mode != "realtime":
        raise RFHybridContractError("rf_hybrid requires the realtime engine")
    if cfg.write_output_video:
        raise RFHybridContractError("rf_hybrid forbids MP4 output")
    mismatches = [
        key
        for key, expected in REALTIME_FAST_CONFIG.items()
        if getattr(cfg, key) != expected
    ]
    if mismatches:
        raise RFHybridContractError(
            "rf_hybrid core differs from realtime_fast: "
            + ", ".join(sorted(mismatches))
        )
    enabled = [key for key in _FORBIDDEN_RF_CORE_FLAGS if getattr(cfg, key)]
    if enabled:
        raise RFHybridContractError(
            "rf_hybrid core enabled transfer stages before freeze: "
            + ", ".join(sorted(enabled))
        )


def _row_key(shape: Mapping[str, Any]) -> tuple[int, str]:
    frame = shape.get("frame")
    label = shape.get("label")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise RFHybridContractError("shape frame must be a non-negative integer")
    if not isinstance(label, str) or not label:
        raise RFHybridContractError("shape label must be a non-empty string")
    points = shape.get("points")
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise RFHybridContractError("shape points must contain four coordinates")
    return frame, label


def validate_frozen_realtime_tracklets(
    shapes: Sequence[Mapping[str, Any]],
) -> None:
    """Validate only the generic RF tracklet fields read by transfer stages."""

    if not shapes:
        raise RFHybridContractError("frozen realtime_fast output is empty")
    keys = [_row_key(shape) for shape in shapes]
    if len(keys) != len(set(keys)):
        raise RFHybridContractError("duplicate frame/label tracklet row")
    for shape in shapes:
        attributes = shape.get("attributes")
        if not isinstance(attributes, list):
            raise RFHybridContractError("shape attributes must be a list")
        names = {
            attribute.get("name")
            for attribute in attributes
            if isinstance(attribute, dict)
        }
        if not {"ID", "Hidden"}.issubset(names):
            raise RFHybridContractError(
                "shape attributes must contain ID and Hidden"
            )


def _public_payload(shape: Mapping[str, Any] | None) -> object:
    if shape is None:
        return None
    return {
        key: _json_value(value)
        for key, value in shape.items()
        if not str(key).startswith("_")
    }


def _rows_by_key(
    shapes: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {_row_key(shape): shape for shape in shapes}


def _identity(shape: Mapping[str, Any] | None) -> str:
    if shape is None:
        return "NONE"
    value = shape_id_value(dict(shape))
    return str(value) if value is not None else "NONE"


def _bbox(shape: Mapping[str, Any] | None) -> list[float] | None:
    if shape is None:
        return None
    return [float(value) for value in shape["points"]]


def _contiguous_groups(frames: Sequence[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for frame in sorted(set(frames)):
        if not groups or frame != groups[-1][-1] + 1:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return groups


def _stage_changes(
    *,
    stage_id: str,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    video: str,
    future_frames_used: bool,
    transfer_config_hash: str,
) -> list[dict[str, Any]]:
    before_rows = _rows_by_key(before)
    after_rows = _rows_by_key(after)
    changed = [
        key
        for key in sorted(set(before_rows) | set(after_rows))
        if _public_payload(before_rows.get(key))
        != _public_payload(after_rows.get(key))
    ]
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for key in changed:
        grouped[
            (
                _identity(before_rows.get(key)),
                _identity(after_rows.get(key)),
            )
        ].append(key)

    input_hash = canonical_transfer_hash(before)
    output_hash = canonical_transfer_hash(after)
    events: list[dict[str, Any]] = []
    for (old_identity, new_identity), keys in sorted(grouped.items()):
        keys_by_frame = defaultdict(list)
        for key in keys:
            keys_by_frame[key[0]].append(key)
        for frame_group in _contiguous_groups(tuple(keys_by_frame)):
            episode_keys = [
                key
                for frame in frame_group
                for key in sorted(keys_by_frame[frame])
            ]
            event = {
                "video": video,
                "start_frame": frame_group[0],
                "end_frame": frame_group[-1],
                "old_identity": old_identity,
                "new_identity": new_identity,
                "old_bbox": [
                    {
                        "frame": key[0],
                        "label": key[1],
                        "bbox": _bbox(before_rows.get(key)),
                    }
                    for key in episode_keys
                ],
                "new_bbox": [
                    {
                        "frame": key[0],
                        "label": key[1],
                        "bbox": _bbox(after_rows.get(key)),
                    }
                    for key in episode_keys
                ],
                "mechanism": stage_id,
                "future_frames_used": future_frames_used,
                "changed_frames": frame_group,
                "changed_tracks": sorted(
                    {old_identity, new_identity} - {"NONE"}
                ),
                "decision_evidence": {
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "transfer_config_hash": transfer_config_hash,
                    "rule": "PREDECLARED_RF_HYBRID_V1_STAGE",
                },
            }
            event["episode_id"] = canonical_transfer_hash(event)
            events.append(event)
    return events


def apply_rf_hybrid_transfer(
    raw_shapes: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    mask: Any,
    *,
    video: str,
) -> RFHybridTransferResult:
    """Apply the frozen RF-transfer stage set to immutable realtime output."""

    validate_frozen_realtime_tracklets(raw_shapes)
    raw_snapshot = deepcopy(list(raw_shapes))
    input_hash = canonical_transfer_hash(raw_snapshot)
    cfg = build_rf_hybrid_transfer_config()
    config_hash = rf_hybrid_transfer_config_hash()
    current = deepcopy(raw_snapshot)
    changes: list[dict[str, Any]] = []
    activation: list[dict[str, Any]] = []

    def run_stage(
        stage_id: str,
        operation: Callable[
            [list[dict[str, Any]]],
            list[dict[str, Any]],
        ],
        *,
        future_frames_used: bool,
    ) -> None:
        nonlocal current
        before = deepcopy(current)
        after = operation(current)
        stage_events = _stage_changes(
            stage_id=stage_id,
            before=before,
            after=after,
            video=video,
            future_frames_used=future_frames_used,
            transfer_config_hash=config_hash,
        )
        activation.append(
            {
                "stage_id": stage_id,
                "rows_before": len(before),
                "rows_after": len(after),
                "input_hash": canonical_transfer_hash(before),
                "output_hash": canonical_transfer_hash(after),
                "change_episode_count": len(stage_events),
                "output_changed": bool(stage_events),
            }
        )
        changes.extend(stage_events)
        current = after

    run_stage(
        "OFFLINE_IDENTITY_SWAP_GUARD",
        lambda rows: apply_identity_swap_guard(rows, width, height, cfg),
        future_frames_used=True,
    )
    run_stage(
        "TEMPORAL_BBOX_REFINEMENT",
        lambda rows: refine_shapes_temporally(rows, width, height, cfg),
        future_frames_used=True,
    )
    run_stage(
        "OVERLAP_HIDDEN_ISLAND_STABILIZATION",
        lambda rows: stabilize_overlap_hidden_islands(rows, cfg),
        future_frames_used=True,
    )

    def pair_repairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = repair_local_pair_swaps(rows, width, height, cfg)
        rows = repair_episode_pair_swaps(rows, width, height, cfg)
        return repair_long_pair_swaps(rows, width, height, cfg)

    run_stage(
        "LOCAL_EPISODE_LONG_PAIR_REPAIRS",
        pair_repairs,
        future_frames_used=True,
    )
    run_stage(
        "SUFFIX_PAIR_SWAP_REPAIR",
        lambda rows: repair_suffix_pair_swaps(rows, width, height, cfg),
        future_frames_used=True,
    )
    run_stage(
        "OVERLAP_SMALL_BOX_SUPPRESSION",
        lambda rows: suppress_overlapped_small_low_confidence_boxes(rows, cfg),
        future_frames_used=True,
    )
    run_stage(
        "H5B_HIDDEN_SUFFIX_OVERLAP_PERSISTENCE",
        lambda rows: repair_hidden_suffix_id_swaps(rows, cfg),
        future_frames_used=True,
    )
    run_stage(
        "REALTIME_MOTION_PAIR_STABILIZER",
        lambda rows: stabilize_realtime_motion_pairs(
            rows,
            width,
            height,
            cfg,
        ),
        future_frames_used=True,
    )
    run_stage(
        "NEAR_WALL_HIDDEN_GEOMETRY",
        lambda rows: refine_near_wall_hidden_geometry(
            rows,
            width,
            height,
            mask,
            cfg,
        ),
        future_frames_used=True,
    )
    run_stage(
        "FAR_CAMERA_GEOMETRY_DURING_H5B",
        lambda rows: refine_far_camera_hidden_geometry(
            rows,
            width,
            height,
            cfg,
        ),
        future_frames_used=True,
    )

    if tuple(item["stage_id"] for item in activation) != (
        RF_HYBRID_TRANSFER_STAGE_IDS
    ):
        raise RuntimeError("RF hybrid stage activation order changed")
    if canonical_transfer_hash(raw_snapshot) != input_hash:
        raise RuntimeError("frozen realtime_fast output was mutated")
    return RFHybridTransferResult(
        shapes=current,
        changes=changes,
        stage_activation=activation,
        input_authority_hash=input_hash,
        output_authority_hash=canonical_transfer_hash(current),
        transfer_config_hash=config_hash,
    )


def write_rf_hybrid_artifacts(
    *,
    realtime_fast_path: Path,
    rf_hybrid_path: Path,
    ledger_path: Path,
    video: str,
    raw_shapes: Sequence[Mapping[str, Any]],
    result: RFHybridTransferResult,
) -> None:
    """Persist the frozen source, transferred output, and change ledger."""

    realtime_payload = {
        "schema_version": RF_HYBRID_TRANSFER_SCHEMA,
        "method_id": "realtime_fast",
        "video": video,
        "authority_hash": result.input_authority_hash,
        "shapes": _json_value(raw_shapes),
    }
    hybrid_payload = {
        "schema_version": RF_HYBRID_TRANSFER_SCHEMA,
        "method_id": "rf_hybrid",
        "video": video,
        "source_authority_hash": result.input_authority_hash,
        "authority_hash": result.output_authority_hash,
        "transfer_config_hash": result.transfer_config_hash,
        "shapes": _json_value(result.shapes),
    }
    ledger_payload = {
        "schema_version": RF_HYBRID_LEDGER_SCHEMA,
        "source_method_id": "realtime_fast",
        "method_id": "rf_hybrid",
        "video": video,
        "input_authority_hash": result.input_authority_hash,
        "output_authority_hash": result.output_authority_hash,
        "transfer_config_hash": result.transfer_config_hash,
        "stage_activation": _json_value(result.stage_activation),
        "changes": _json_value(result.changes),
    }
    for path, payload in (
        (realtime_fast_path, realtime_payload),
        (rf_hybrid_path, hybrid_payload),
        (ledger_path, ledger_payload),
    ):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )


__all__ = [
    "RF_HYBRID_LEDGER_SCHEMA",
    "RF_HYBRID_TRANSFER_PARAMETER_KEYS",
    "RF_HYBRID_TRANSFER_SCHEMA",
    "RF_HYBRID_TRANSFER_STAGE_IDS",
    "RFHybridContractError",
    "RFHybridTransferResult",
    "apply_rf_hybrid_transfer",
    "build_rf_hybrid_transfer_config",
    "canonical_transfer_hash",
    "rf_hybrid_transfer_config_hash",
    "rf_hybrid_transfer_configuration",
    "validate_frozen_realtime_tracklets",
    "validate_rf_hybrid_core_config",
    "write_rf_hybrid_artifacts",
]
