"""Leakage-safe Pig-STRENet and causal-history artifact builders.

This module exports reusable, model-agnostic artifacts.  It deliberately does
not modify the canonical feature tables or integrate a fusion model.  The
native event remains the evaluation unit; derived history/target pairs are
training views with event mass conserved within each native event.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.roi import (
    load_scene_rois_from_coco,
)

ROI_CLASSES = ("feeder", "drinker", "toy")
HISTORY_LENGTH = 6
TARGET_LENGTH = 6
CONTROL_IDS = ("T0", "T1", "H0", "HA", "HS", "HR", "HRev", "PM")

MODEL_X_FORBIDDEN_TOKENS = (
    "behavior",
    "label",
    "review",
    "path",
    "source",
    "dataset",
    "video",
    "frame_uid",
    "track",
    "pig_id",
    "fold",
    "window_id",
    "native_event_id",
    "temporal_unit_key",
)

MODEL_X_AUDIT_TOKENS = (
    "frame_count",
    "expected_frame_count",
    "available_ratio",
    "_complete",
    "gap_count",
    "duration_sec",
    "target_gap_sec",
)

PAIR_REQUIRED_COLUMNS = {
    "object_track_key",
    "temporal_unit_key",
    "source_type",
    "frame_index",
    "frame_uid",
}


@dataclass(frozen=True, slots=True)
class PigSTRENetArtifacts:
    """Long-form artifacts plus a machine-readable audit."""

    pair_manifest: pd.DataFrame
    slot_manifest: pd.DataFrame
    history_features: pd.DataFrame
    roi_dynamics: pd.DataFrame
    roi_visual_selection: pd.DataFrame
    social_nodes: pd.DataFrame
    social_edges: pd.DataFrame
    control_matrix: pd.DataFrame
    audit: dict[str, Any]


def build_pig_strenet_artifacts(
    frames: pd.DataFrame,
    *,
    history_length: int = HISTORY_LENGTH,
    target_length: int = TARGET_LENGTH,
    legacy_target_starts: tuple[int, ...] = (6,),
    top_k_neighbors: int = 3,
    target_unit_keys: set[str] | None = None,
    roi_coco_path: Path | None = None,
) -> PigSTRENetArtifacts:
    """Build history, ROI, social and control artifacts from frame features.

    CVAT targets use each declared six-frame label interval.  Legacy targets
    use an explicit relative target start, defaulting to frames ``6..11``;
    this is marked as ``legacy_derived_6f`` and never masquerades as a CVAT
    native unit.
    """

    _validate_frame_input(frames)
    if history_length <= 0 or target_length <= 0:
        raise ValueError("history_length and target_length must be positive")
    if top_k_neighbors <= 0:
        raise ValueError("top_k_neighbors must be positive")

    work = _normalize_frames(frames)
    pairs, slots = _build_pair_and_slot_manifests(
        work,
        history_length=history_length,
        target_length=target_length,
        legacy_target_starts=legacy_target_starts,
        target_unit_keys=target_unit_keys,
    )
    history_features = _build_history_features(work, pairs, slots)
    roi_dynamics = _build_roi_dynamics(work, slots)
    roi_visual_selection = _build_roi_visual_selection(
        work,
        slots,
        roi_coco_path=roi_coco_path,
    )
    social_nodes, social_edges = _build_social_graph(
        work,
        slots,
        top_k_neighbors=top_k_neighbors,
    )
    controls = build_history_control_matrix(pairs)
    audit = _build_artifact_audit(
        frames=work,
        pairs=pairs,
        slots=slots,
        history_features=history_features,
        roi_dynamics=roi_dynamics,
        roi_visual_selection=roi_visual_selection,
        social_nodes=social_nodes,
        social_edges=social_edges,
        controls=controls,
    )
    return PigSTRENetArtifacts(
        pair_manifest=pairs,
        slot_manifest=slots,
        history_features=history_features,
        roi_dynamics=roi_dynamics,
        roi_visual_selection=roi_visual_selection,
        social_nodes=social_nodes,
        social_edges=social_edges,
        control_matrix=controls,
        audit=audit,
    )


def build_history_control_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return the predeclared T0/T1/H0/HA/HS/HR/HRev/PM control matrix."""

    required = {"pair_id", "source_type", "native_event_id"}
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"pair manifest missing control columns={missing}")

    records: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        source = str(pair.source_type)
        pair_id = str(pair.pair_id)
        event_id = str(pair.native_event_id)
        applicable = {
            "T0": source == "legacy_recovered",
            "T1": True,
            "H0": True,
            "HA": True,
            "HS": True,
            "HR": True,
            "HRev": True,
            "PM": True,
        }
        for control_id in CONTROL_IDS:
            target_start = _optional_int(getattr(pair, "target_start_relative", np.nan))
            target_end = _optional_int(getattr(pair, "target_end_relative", np.nan))
            if source == "legacy_recovered" and target_start is not None:
                target_spec = f"legacy_relative[{target_start}:{target_end}]"
                history_spec = (
                    f"legacy_relative[{target_start - 6}:{target_start - 1}]"
                )
            else:
                target_spec = (
                    f"actual[{int(pair.target_window_start_frame)}:"
                    f"{int(pair.target_window_end_frame)}]"
                )
                history_spec = (
                    f"actual[{int(pair.history_window_start_frame)}:"
                    f"{int(pair.history_window_end_frame)}]"
                )
            records.append(
                {
                    "pair_id": pair_id,
                    "native_event_id": event_id,
                    "source_type": source,
                    "control_id": control_id,
                    "target_view": (
                        "legacy_old_c6_5_10"
                        if control_id == "T0"
                        and source == "legacy_recovered"
                        else (
                            "not_applicable_legacy_only"
                            if control_id == "T0"
                            else target_spec
                        )
                    ),
                    "history_view": {
                        "T0": "none",
                        "T1": "none",
                        "H0": "zero_mask_false",
                        "HA": "availability_only",
                        "HS": "real_temporal_shuffle",
                        "HR": f"real_history_{history_spec}",
                        "HRev": "real_history_reversed_diagnostic",
                        "PM": "none_parameter_matched",
                    }[control_id],
                    "applicable": bool(applicable[control_id]),
                    "target_window_spec": target_spec,
                    "history_window_spec": history_spec,
                    "event_weight": float(pair.event_weight),
                    "model_x_allowed": control_id in {"HA", "HS", "HR", "HRev"},
                    "selection_role": (
                        "bridge_only" if control_id in {"T0", "T1"} else "ablation"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def compute_stabilized_difference_maps(
    crops: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    max_shift_ratio: float = 0.10,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Compute bounded pairwise RGB differences for an ordered crop sequence.

    ``crops`` is ``[time, height, width, 3]``.  The previous crop is aligned
    to the current crop with a bounded translation.  Failed or invalid pairs
    remain masked instead of being silently replaced by valid evidence.
    """

    array = np.asarray(crops)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("crops must have shape [time,height,width,3]")
    if array.shape[0] < 2:
        raise ValueError("at least two crops are required")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    valid = (
        np.ones(array.shape[0], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != (array.shape[0],):
        raise ValueError("valid_mask must have one value per crop")
    height, width = array.shape[1:3]
    max_shift = max(1.0, max(height, width) * float(max_shift_ratio))
    maps = np.zeros((array.shape[0] - 1, height, width), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    pair_valid = np.zeros(array.shape[0] - 1, dtype=bool)
    for index in range(1, array.shape[0]):
        if not valid[index - 1] or not valid[index]:
            rows.append(_difference_row(index, False, 0.0, 0.0, maps[index - 1]))
            continue
        previous = cv2.cvtColor(array[index - 1], cv2.COLOR_RGB2GRAY)
        current = cv2.cvtColor(array[index], cv2.COLOR_RGB2GRAY)
        shift_x, shift_y = _bounded_phase_shift(previous, current, max_shift)
        matrix = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
        aligned = cv2.warpAffine(
            array[index - 1],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        diff = np.mean(
            np.abs(array[index].astype(np.float32) - aligned.astype(np.float32)),
            axis=2,
        ) / 255.0
        maps[index - 1] = diff.astype(np.float32)
        pair_valid[index - 1] = True
        rows.append(_difference_row(index, True, shift_x, shift_y, diff))
    return maps, pd.DataFrame.from_records(rows), pair_valid


def _build_pair_and_slot_manifests(
    frames: pd.DataFrame,
    *,
    history_length: int,
    target_length: int,
    legacy_target_starts: tuple[int, ...],
    target_unit_keys: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_track = {
        str(key): group.sort_values("frame_index", kind="mergesort")
        for key, group in frames.groupby("object_track_key", sort=False)
    }
    pair_records: list[dict[str, Any]] = []
    slot_records: list[dict[str, Any]] = []
    for unit_key, unit in frames.groupby("temporal_unit_key", sort=False):
        if target_unit_keys is not None and str(unit_key) not in target_unit_keys:
            continue
        unit = unit.sort_values("frame_index", kind="mergesort")
        source = str(unit.iloc[0]["source_type"])
        track_key = str(unit.iloc[0]["object_track_key"])
        if source == "legacy_recovered":
            rel = pd.to_numeric(unit["relative_frame_index"], errors="coerce")
            starts = [int(value) for value in legacy_target_starts]
            target_specs = [(start, start + target_length - 1) for start in starts]
        else:
            start = _first_number(unit, "label_window_start")
            end = _first_number(unit, "label_window_end")
            if start is None or end is None:
                start = int(unit["frame_index"].min())
                end = start + target_length - 1
            if int(end - start + 1) != target_length:
                raise ValueError(
                    f"target interval is not {target_length}f for unit={unit_key}"
                )
            rel = None
            target_specs = [(int(start), int(end))]
        track = by_track[track_key]
        for pair_index, (target_start, target_end) in enumerate(target_specs):
            if rel is not None:
                target = unit[rel.between(target_start, target_end)].copy()
                history_expected = list(range(target_start - history_length, target_start))
                target_expected = list(range(target_start, target_end + 1))
                history = unit[rel.isin(history_expected)].copy()
                observed_history = set(
                    pd.to_numeric(
                        history["relative_frame_index"], errors="coerce"
                    ).dropna().astype(int)
                )
                observed_target = set(
                    pd.to_numeric(
                        target["relative_frame_index"], errors="coerce"
                    ).dropna().astype(int)
                )
                coordinate_column = "relative_frame_index"
            else:
                target = track[track["frame_index"].between(target_start, target_end)].copy()
                history_expected = list(range(target_start - history_length, target_start))
                target_expected = list(range(target_start, target_end + 1))
                history = track[track["frame_index"].isin(history_expected)].copy()
                observed_history = set(
                    pd.to_numeric(history["frame_index"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                observed_target = set(
                    pd.to_numeric(target["frame_index"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                coordinate_column = "frame_index"
            if target.empty:
                continue
            event_id = str(unit_key)
            pair_id = f"{event_id}::history{history_length}_target{target_length}::p{pair_index}"
            history_complete = set(history_expected).issubset(observed_history)
            target_complete = set(target_expected).issubset(observed_target)
            source_rows = unit if rel is not None else track
            target_start_actual = _actual_frame_for_coordinate(
                source_rows, target_start, relative_coordinates=rel is not None
            )
            target_end_actual = _actual_frame_for_coordinate(
                source_rows, target_end, relative_coordinates=rel is not None
            )
            history_start_actual = _actual_frame_for_coordinate(
                source_rows,
                target_start - history_length,
                relative_coordinates=rel is not None,
            )
            history_end_actual = _actual_frame_for_coordinate(
                source_rows,
                target_start - 1,
                relative_coordinates=rel is not None,
            )
            pair_records.append(
                {
                    "pair_id": pair_id,
                    "native_event_id": event_id,
                    "temporal_unit_key": event_id,
                    "source_type": source,
                    "dataset_id": str(unit.iloc[0].get("dataset_id", "")),
                    "video_key": str(unit.iloc[0].get("video_key", "")),
                    "object_track_key": track_key,
                    "target_start_frame": target_start_actual,
                    "target_end_frame": target_end_actual,
                    "history_start_frame": history_start_actual,
                    "history_end_frame": history_end_actual,
                    "target_window_start_frame": target_start_actual,
                    "target_window_end_frame": target_end_actual,
                    "history_window_start_frame": history_start_actual,
                    "history_window_end_frame": history_end_actual,
                    "target_start_relative": (
                        int(target_start) if rel is not None else np.nan
                    ),
                    "target_end_relative": (
                        int(target_end) if rel is not None else np.nan
                    ),
                    "history_start_relative": (
                        int(target_start - history_length)
                        if rel is not None
                        else np.nan
                    ),
                    "history_end_relative": (
                        int(target_start - 1) if rel is not None else np.nan
                    ),
                    "history_expected_frame_count": history_length,
                    "target_expected_frame_count": target_length,
                    "history_frame_count": int(len(observed_history)),
                    "target_frame_count": int(len(observed_target)),
                    "history_available_ratio": float(len(observed_history) / history_length),
                    "target_available_ratio": float(len(observed_target) / target_length),
                    "history_complete": bool(history_complete),
                    "target_complete": bool(target_complete),
                    "history_same_track": True,
                    "history_gap_count": _gap_count(observed_history, history_expected),
                    "history_max_gap_sec": _max_gap_seconds(history, history_expected),
                    "history_duration_sec": _duration_seconds(history),
                    "target_duration_sec": _duration_seconds(target),
                    "history_target_gap_sec": _boundary_gap_seconds(history, target),
                    "derived_view": (
                        "legacy_derived_6f" if source == "legacy_recovered" else "cvat_target_6f"
                    ),
                    "label_propagation_policy": str(
                        unit.iloc[0].get("label_propagation_policy", "")
                    ),
                    "behavior_label_audit_only": str(
                        unit.iloc[0].get("behavior_label", unit.iloc[0].get("behavior", ""))
                    ),
                    "event_pair_count": 1,
                    "event_weight": 1.0,
                    "source_lineage_review_complete": _bool_value(
                        unit.iloc[0].get("human_review_complete", False)
                    ),
                }
            )
            for role, rows, expected in (
                ("history", history, history_expected),
                ("target", target, target_expected),
            ):
                by_frame = {
                    int(row[coordinate_column]): row
                    for _, row in rows.iterrows()
                }
                for slot_index, frame_index in enumerate(expected):
                    row = by_frame.get(frame_index)
                    actual_frame_index = (
                        int(row["frame_index"])
                        if row is not None
                        else _fallback_slot_frame_index(
                            source_rows,
                            frame_index,
                            relative_coordinates=rel is not None,
                        )
                    )
                    slot_records.append(
                        {
                            "pair_id": pair_id,
                            "native_event_id": event_id,
                            "source_type": source,
                            "object_track_key": track_key,
                            "slot_role": role,
                            "slot_index": slot_index,
                            "global_slot_index": (
                                slot_index if role == "history" else history_length + slot_index
                            ),
                            "frame_index": actual_frame_index,
                            "coordinate_index": int(frame_index),
                            "relative_frame_index": int(frame_index) if rel is not None else np.nan,
                            "frame_available": row is not None,
                            "frame_uid": "" if row is None else str(row.get("frame_uid", "")),
                            "scene_frame_uid": (
                                ""
                                if row is None
                                else str(row.get("scene_frame_uid", ""))
                            ),
                            "crop_path": "" if row is None else str(row.get("crop_path", "")),
                            "history_mask": role == "history" and row is not None,
                            "target_mask": role == "target" and row is not None,
                        }
                    )
    pairs = pd.DataFrame.from_records(pair_records)
    slots = pd.DataFrame.from_records(slot_records)
    if pairs.empty:
        raise ValueError("no causal history pairs were generated")
    counts = pairs.groupby("native_event_id", sort=False)["pair_id"].transform("count")
    pairs["event_pair_count"] = counts.astype(int)
    pairs["event_weight"] = 1.0 / counts.astype(float)
    _audit_event_mass(pairs)
    return pairs, slots


def _build_history_features(
    frames: pd.DataFrame,
    pairs: pd.DataFrame,
    slots: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_lookup = {
        str(value): row
        for value, row in frames.set_index("frame_uid").iterrows()
        if str(value).strip()
    }
    for pair in pairs.itertuples(index=False):
        history = _slot_rows(frames, slots, pair.pair_id, "history", frame_lookup)
        target = _slot_rows(frames, slots, pair.pair_id, "target", frame_lookup)
        history_summary = _segment_summary(
            history,
            "history",
            expected_count=int(pair.history_expected_frame_count),
        )
        target_summary = _segment_summary(
            target,
            "target",
            expected_count=int(pair.target_expected_frame_count),
        )
        result = {
            "pair_id": str(pair.pair_id),
            "native_event_id": str(pair.native_event_id),
            "source_type": str(pair.source_type),
            "event_weight": float(pair.event_weight),
            **history_summary,
        }
        result.update(_transition_features(history_summary, target_summary))
        rows.append(result)
    output = pd.DataFrame.from_records(rows)
    _validate_model_x_names(output)
    return output


def _build_roi_dynamics(frames: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(value): row for value, row in frames.set_index("frame_uid").iterrows()}
    records: list[dict[str, Any]] = []
    for pair_id, group in slots.groupby("pair_id", sort=False):
        ordered = group.sort_values("global_slot_index")
        for roi_class in ROI_CLASSES:
            contact_run = 0
            previous_contact = False
            for row in ordered.itertuples(index=False):
                source = lookup.get(str(row.frame_uid))
                available = source is not None and bool(
                    _bool_value(source.get(f"roi_{roi_class}_available", False))
                )
                contact = available and bool(
                    _bool_value(source.get(f"roi_{roi_class}_contact", False))
                )
                near = available and bool(_bool_value(source.get(f"roi_{roi_class}_near", False)))
                if contact:
                    contact_run += 1
                else:
                    contact_run = 0
                records.append(
                    {
                        "pair_id": str(pair_id),
                        "native_event_id": str(row.native_event_id),
                        "slot_index": int(row.global_slot_index),
                        "slot_role": str(row.slot_role),
                        "roi_class": roi_class,
                        "available": available,
                        "min_dist_n": _number_or_nan(source, f"roi_{roi_class}_min_dist_n"),
                        "overlap_ratio": _number_or_zero(
                            source, f"roi_{roi_class}_max_overlap_ratio"
                        ),
                        "iou": _number_or_zero(source, f"roi_{roi_class}_max_iou"),
                        "center_inside": (
                            available
                            and bool(
                                _bool_value(
                                    source.get(
                                        f"roi_{roi_class}_center_inside",
                                        False,
                                    )
                                )
                            )
                            if source is not None
                            else False
                        ),
                        "near": near,
                        "contact": contact,
                        "entry": contact and not previous_contact,
                        "exit": previous_contact and not contact,
                        "contact_run_length": contact_run,
                        "motion_inside": (
                            _number_or_zero(source, "speed_n_per_frame")
                            if contact
                            else 0.0
                        ),
                    }
                )
                previous_contact = contact
    return pd.DataFrame.from_records(records)


def _build_roi_visual_selection(
    frames: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    roi_coco_path: Path | None,
) -> pd.DataFrame:
    """Export all-class actor/ROI geometry without target-ROI routing."""

    rois = load_scene_rois_from_coco(roi_coco_path) if roi_coco_path else []
    lookup = {
        str(value): row for value, row in frames.set_index("frame_uid").iterrows()
    }
    records: list[dict[str, Any]] = []
    for slot in slots.itertuples(index=False):
        actor = lookup.get(str(slot.frame_uid))
        for roi_class in ROI_CLASSES:
            selected = _nearest_scene_roi(actor, rois, roi_class)
            actor_box = _box_from_row(actor)
            roi_box = selected["box"] if selected is not None else None
            valid = bool(slot.frame_available and actor_box is not None and roi_box is not None)
            intersection = _intersection_box(actor_box, roi_box) if valid else None
            union = _union_box(actor_box, roi_box) if valid else None
            records.append(
                {
                    "pair_id": str(slot.pair_id),
                    "native_event_id": str(slot.native_event_id),
                    "frame_uid": str(slot.frame_uid),
                    "slot_index": int(slot.global_slot_index),
                    "slot_role": str(slot.slot_role),
                    "roi_class": roi_class,
                    "roi_id": "" if selected is None else str(selected["roi_id"]),
                    "actor_roi_visual_available": valid,
                    "actor_x1": _box_value(actor_box, 0),
                    "actor_y1": _box_value(actor_box, 1),
                    "actor_x2": _box_value(actor_box, 2),
                    "actor_y2": _box_value(actor_box, 3),
                    "roi_x1": _box_value(roi_box, 0),
                    "roi_y1": _box_value(roi_box, 1),
                    "roi_x2": _box_value(roi_box, 2),
                    "roi_y2": _box_value(roi_box, 3),
                    "intersection_x1": _box_value(intersection, 0),
                    "intersection_y1": _box_value(intersection, 1),
                    "intersection_x2": _box_value(intersection, 2),
                    "intersection_y2": _box_value(intersection, 3),
                    "union_x1": _box_value(union, 0),
                    "union_y1": _box_value(union, 1),
                    "union_x2": _box_value(union, 2),
                    "union_y2": _box_value(union, 3),
                    "visual_context_id": (
                        f"{slot.pair_id}::slot={slot.global_slot_index}::roi={roi_class}"
                    ),
                    "target_roi_selected": False,
                }
            )
    return pd.DataFrame.from_records(records)


def _nearest_scene_roi(
    actor: pd.Series | None,
    rois: list[Any],
    roi_class: str,
) -> dict[str, Any] | None:
    if actor is None:
        return None
    actor_box = _box_from_row(actor)
    if actor_box is None:
        return None
    image_width = _scalar(actor.get("image_width"))
    image_height = _scalar(actor.get("image_height"))
    if not np.isfinite([image_width, image_height]).all():
        image_width, image_height = 1280.0, 720.0
    candidates: list[dict[str, Any]] = []
    ax = (actor_box[0] + actor_box[2]) / 2.0
    ay = (actor_box[1] + actor_box[3]) / 2.0
    for roi in rois:
        if str(roi.category) != roi_class:
            continue
        sx = image_width / float(roi.image_width)
        sy = image_height / float(roi.image_height)
        box = (
            float(roi.x1 * sx),
            float(roi.y1 * sy),
            float(roi.x2 * sx),
            float(roi.y2 * sy),
        )
        rx = (box[0] + box[2]) / 2.0
        ry = (box[1] + box[3]) / 2.0
        candidates.append(
            {"roi_id": roi.roi_id, "box": box, "distance": float(np.hypot(ax - rx, ay - ry))}
        )
    return (
        min(candidates, key=lambda value: (value["distance"], value["roi_id"]))
        if candidates
        else None
    )


def _box_from_row(row: pd.Series | None) -> tuple[float, float, float, float] | None:
    if row is None:
        return None
    values = [_scalar(row.get(column)) for column in ("x1", "y1", "x2", "y2")]
    return tuple(float(value) for value in values) if np.isfinite(values).all() else None


def _intersection_box(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if left is None or right is None:
        return None
    box = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return box if box[2] >= box[0] and box[3] >= box[1] else None


def _union_box(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if left is None or right is None:
        return None
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _box_value(box: tuple[float, float, float, float] | None, index: int) -> float:
    return float(box[index]) if box is not None else float("nan")


def _build_social_graph(
    frames: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    top_k_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_column = "scene_frame_uid" if "scene_frame_uid" in frames else "frame_index"
    frame_groups = {
        str(key): group
        for key, group in frames.groupby(scene_column, sort=False)
    }
    frame_lookup = {
        str(value): row for value, row in frames.set_index("frame_uid").iterrows()
    }
    node_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    for pair_id, pair_slots in slots.groupby("pair_id", sort=False):
        ordered = pair_slots.sort_values("global_slot_index")
        for slot in ordered.itertuples(index=False):
            actor = frame_lookup.get(str(slot.frame_uid))
            available = actor is not None and bool(slot.frame_available)
            node_records.append(
                {
                    "pair_id": str(pair_id),
                    "native_event_id": str(slot.native_event_id),
                    "slot_index": int(slot.global_slot_index),
                    "node_key": str(slot.object_track_key),
                    "node_available": available,
                    "partner_available_ratio": 0.0,
                    "partner_persistence_ratio": 0.0,
                    "partner_key_consistency": False,
                    "partner_switch_count": 0,
                    "pair_valid_ratio": 0.0,
                }
            )
            if not available:
                for rank in range(1, top_k_neighbors + 1):
                    edge_records.append(
                        _missing_edge_record(pair_id, slot, rank)
                    )
                continue
            scene_key = str(actor.get(scene_column, ""))
            candidates = frame_groups.get(scene_key, pd.DataFrame())
            candidates = candidates[
                candidates["object_track_key"].astype(str).ne(str(slot.object_track_key))
            ].copy()
            ranked = _rank_neighbors(actor, candidates).head(top_k_neighbors)
            for rank in range(1, top_k_neighbors + 1):
                if rank <= len(ranked):
                    neighbor = ranked.iloc[rank - 1]
                    edge_records.append(
                        _edge_record(actor, neighbor, pair_id, slot, rank)
                    )
                else:
                    edge_records.append(
                        _missing_edge_record(pair_id, slot, rank)
                    )
    nodes = pd.DataFrame.from_records(node_records)
    edges = pd.DataFrame.from_records(edge_records)
    edges["neighbor_rank"] = edges["neighbor_rank"].astype(int)
    edges["pair_motion_energy"] = edges["pair_motion_energy"].astype(float)
    _add_partner_persistence_columns(edges)
    _add_edge_temporal_features(edges, frames)
    _copy_partner_audit_to_nodes(nodes, edges)
    return nodes, edges


def _segment_summary(
    segment: pd.DataFrame,
    prefix: str,
    *,
    expected_count: int,
) -> dict[str, Any]:
    expected = max(0, int(expected_count))
    if segment.empty:
        return _empty_segment_summary(prefix, expected_count=expected)
    frame = pd.to_numeric(segment.get("frame_index"), errors="coerce").dropna().astype(int)
    speed = _values(segment, "speed_n_per_frame")
    speed = speed[np.isfinite(speed)]
    displacement = _values(segment, "displacement_n")
    accel = _values(segment, "abs_accel_n_per_frame2")
    direction = _values(segment, "abs_direction_change_rad")
    contact = {
        name: _bool_values(segment, f"roi_{name}_contact") for name in ROI_CLASSES
    }
    result: dict[str, Any] = {
        f"{prefix}_frame_count": int(len(frame)),
        f"{prefix}_expected_frame_count": expected,
        f"{prefix}_available_ratio": float(len(frame) / expected) if expected else 0.0,
        f"{prefix}_complete": bool(expected > 0 and len(frame) == expected),
        f"{prefix}_gap_count": max(0, expected - len(frame)),
        f"{prefix}_duration_sec": _duration_seconds(segment),
        f"{prefix}_speed_mean": _mean(speed),
        f"{prefix}_speed_max": _max(speed),
        f"{prefix}_speed_std": _std(speed),
        f"{prefix}_path_length_n": float(np.nansum(displacement)) if len(displacement) else 0.0,
        f"{prefix}_displacement_n": _endpoint_displacement(segment),
        f"{prefix}_stationary_ratio": _ratio(speed <= 0.002),
        f"{prefix}_acceleration_mean": _mean(accel),
        f"{prefix}_direction_change_sum": float(np.nansum(direction)) if len(direction) else 0.0,
        f"{prefix}_turn_count": (
            int(np.sum(direction >= math.radians(30.0)))
            if len(direction)
            else 0
        ),
        f"{prefix}_motion_burstiness": float(_std(speed) / (_mean(speed) + 1e-9)),
        f"{prefix}_area_mean": _mean(_values(segment, "area_n")),
        f"{prefix}_nearest_dist_mean": _mean(_values(segment, "nearest_dist_n")),
        f"{prefix}_nearest_dist_min": _min(_values(segment, "nearest_dist_n")),
        f"{prefix}_nearest_dist_slope": _slope(segment, "nearest_dist_n"),
        f"{prefix}_approach_ratio": _ratio(_values(segment, "approach_speed_n_per_frame") > 0),
        f"{prefix}_retreat_ratio": _ratio(_values(segment, "separation_speed_n_per_frame") > 0),
        f"{prefix}_contact_ratio": _ratio(_bool_values(segment, "pair_contact_with_nearest")),
        f"{prefix}_pair_iou_mean": _mean(_values(segment, "nearest_pair_iou")),
        f"{prefix}_pair_motion_energy": float(np.nansum(np.square(speed))) if len(speed) else 0.0,
        f"{prefix}_partner_persistence": _partner_persistence(segment),
        f"{prefix}_partner_switch_count": _partner_switch_count(segment),
    }
    for roi_class, values in contact.items():
        distances = _values(segment, f"roi_{roi_class}_min_dist_n")
        result.update(
            {
                f"{prefix}_{roi_class}_dist_slope": _slope(segment, f"roi_{roi_class}_min_dist_n"),
                f"{prefix}_{roi_class}_contact_ratio": _ratio(values),
                f"{prefix}_{roi_class}_dwell_sec": _duration_from_mask(segment, values),
            }
        )
        result[f"{prefix}_{roi_class}_available_ratio"] = _ratio(
            _bool_values(segment, f"roi_{roi_class}_available")
        )
        result[f"{prefix}_{roi_class}_near_ratio"] = _ratio(
            _bool_values(segment, f"roi_{roi_class}_near")
        )
        result[f"{prefix}_{roi_class}_distance_mean"] = _mean(distances)
    return result


def _empty_segment_summary(prefix: str, *, expected_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{prefix}_frame_count": 0,
        f"{prefix}_expected_frame_count": int(expected_count),
        f"{prefix}_available_ratio": 0.0,
        f"{prefix}_complete": False,
        f"{prefix}_gap_count": int(expected_count),
        f"{prefix}_duration_sec": 0.0,
        f"{prefix}_speed_mean": 0.0,
        f"{prefix}_speed_max": 0.0,
        f"{prefix}_speed_std": 0.0,
        f"{prefix}_path_length_n": 0.0,
        f"{prefix}_displacement_n": 0.0,
        f"{prefix}_stationary_ratio": 0.0,
        f"{prefix}_acceleration_mean": 0.0,
        f"{prefix}_direction_change_sum": 0.0,
        f"{prefix}_turn_count": 0,
        f"{prefix}_motion_burstiness": 0.0,
        f"{prefix}_area_mean": 0.0,
        f"{prefix}_nearest_dist_mean": 0.0,
        f"{prefix}_nearest_dist_min": 0.0,
        f"{prefix}_nearest_dist_slope": 0.0,
        f"{prefix}_approach_ratio": 0.0,
        f"{prefix}_retreat_ratio": 0.0,
        f"{prefix}_contact_ratio": 0.0,
        f"{prefix}_pair_iou_mean": 0.0,
        f"{prefix}_pair_motion_energy": 0.0,
        f"{prefix}_partner_persistence": 0.0,
        f"{prefix}_partner_switch_count": 0,
    }
    for roi_class in ROI_CLASSES:
        result.update(
            {
                f"{prefix}_{roi_class}_dist_slope": 0.0,
                f"{prefix}_{roi_class}_contact_ratio": 0.0,
                f"{prefix}_{roi_class}_dwell_sec": 0.0,
                f"{prefix}_{roi_class}_available_ratio": 0.0,
                f"{prefix}_{roi_class}_near_ratio": 0.0,
                f"{prefix}_{roi_class}_distance_mean": 0.0,
            }
        )
    return result


def _transition_features(history: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "activity_delta_history_to_target": (
            target["target_speed_mean"] - history["history_speed_mean"]
        ),
        "speed_delta_history_to_target": (
            target["target_speed_mean"] - history["history_speed_mean"]
        ),
        "stationary_to_motion_score": (
            history["history_stationary_ratio"]
            * (1.0 - target["target_stationary_ratio"])
        ),
        "motion_to_stationary_score": (
            (1.0 - history["history_stationary_ratio"])
            * target["target_stationary_ratio"]
        ),
        "distance_delta_history_to_target": (
            target["target_nearest_dist_mean"]
            - history["history_nearest_dist_mean"]
        ),
        "approach_to_contact_score": (
            history["history_approach_ratio"] * target["target_contact_ratio"]
        ),
        "contact_persistence_score": (
            history["history_contact_ratio"] * target["target_contact_ratio"]
        ),
        "contact_to_separation_score": (
            history["history_contact_ratio"] * target["target_retreat_ratio"]
        ),
        "partner_change_count": (
            target["target_partner_switch_count"]
            + history["history_partner_switch_count"]
        ),
        "shape_change_history_to_target": abs(
            target.get("target_area_mean", 0.0) - history.get("history_area_mean", 0.0)
        ),
    }
    for roi_class in ROI_CLASSES:
        result[f"{roi_class}_approach_to_engagement"] = (
            history[f"history_{roi_class}_dist_slope"] * -1.0
            * target[f"target_{roi_class}_contact_ratio"]
        )
        result[f"{roi_class}_engagement_to_departure"] = (
            history[f"history_{roi_class}_contact_ratio"]
            * max(0.0, target[f"target_{roi_class}_dist_slope"])
        )
    return result


def _rank_neighbors(actor: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ax = _scalar(actor.get("cx_n"))
    ay = _scalar(actor.get("cy_n"))
    work = candidates.copy()
    work["_distance"] = np.hypot(
        pd.to_numeric(work.get("cx_n"), errors="coerce") - ax,
        pd.to_numeric(work.get("cy_n"), errors="coerce") - ay,
    )
    return work.sort_values(["_distance", "object_track_key"], kind="mergesort")


def _edge_record(
    actor: pd.Series,
    neighbor: pd.Series,
    pair_id: object,
    slot: Any,
    rank: int,
) -> dict[str, Any]:
    dx = _scalar(neighbor.get("cx_n")) - _scalar(actor.get("cx_n"))
    dy = _scalar(neighbor.get("cy_n")) - _scalar(actor.get("cy_n"))
    distance = float(np.hypot(dx, dy))
    pair_iou = _box_iou(_box_from_row(actor), _box_from_row(neighbor))
    overlap = _box_overlap_ratio(_box_from_row(actor), _box_from_row(neighbor))
    actor_speed = _number_or_zero(actor, "speed_n_per_frame")
    neighbor_speed = _number_or_zero(neighbor, "speed_n_per_frame")
    raw_vx = _number_or_nan(actor, "vx_n")
    raw_vy = _number_or_nan(actor, "vy_n")
    if not np.isfinite(raw_vx):
        raw_vx = _number_or_nan(actor, "vx_n_per_frame")
    if not np.isfinite(raw_vy):
        raw_vy = _number_or_nan(actor, "vy_n_per_frame")
    actor_vx = 0.0 if not np.isfinite(raw_vx) else raw_vx
    actor_vy = 0.0 if not np.isfinite(raw_vy) else raw_vy
    dot = actor_vx * dx + actor_vy * dy
    cross = actor_vx * dy - actor_vy * dx
    relative_angle = float(np.arctan2(cross, dot))
    angle_available = bool(
        np.isfinite(raw_vx) and np.isfinite(raw_vy) and distance > 0.0
    )
    contact = bool(
        _bool_value(actor.get("pair_contact_with_nearest", False))
        if rank == 1
        else pair_iou > 0.0
    )
    motion_energy = actor_speed**2 + neighbor_speed**2
    return {
        "pair_id": str(pair_id),
        "native_event_id": str(slot.native_event_id),
        "slot_index": int(slot.global_slot_index),
        "actor_node_key": str(slot.object_track_key),
        "neighbor_node_key": str(neighbor["object_track_key"]),
        "neighbor_rank": int(rank),
        "edge_available": True,
        "distance_n": distance,
        "relative_dx_n": dx,
        "relative_dy_n": dy,
        "relative_speed_n": neighbor_speed - actor_speed,
        "relative_angle": relative_angle,
        "relative_angle_available": angle_available,
        "approach_speed_n": max(0.0, -_number_or_zero(actor, "nearest_dist_delta")),
        "separation_speed_n": max(0.0, _number_or_zero(actor, "nearest_dist_delta")),
        "pair_iou": pair_iou,
        "pair_overlap_ratio": overlap,
        "pair_contact": contact,
        "pair_contact_duration_frames": 0,
        "timestamp_sec": _number_or_nan(actor, "timestamp_sec"),
        "pair_motion_energy": motion_energy,
        "pair_contact_motion_intensity": motion_energy if contact else 0.0,
    }


def _missing_edge_record(pair_id: object, slot: Any, rank: int) -> dict[str, Any]:
    return {
        "pair_id": str(pair_id),
        "native_event_id": str(slot.native_event_id),
        "slot_index": int(slot.global_slot_index),
        "actor_node_key": str(slot.object_track_key),
        "neighbor_node_key": "",
        "neighbor_rank": int(rank),
        "edge_available": False,
        "distance_n": 0.0,
        "relative_dx_n": 0.0,
        "relative_dy_n": 0.0,
        "relative_speed_n": 0.0,
        "relative_angle": 0.0,
        "relative_angle_available": False,
        "approach_speed_n": 0.0,
        "separation_speed_n": 0.0,
        "pair_iou": 0.0,
        "pair_overlap_ratio": 0.0,
        "pair_contact": False,
        "pair_contact_duration_frames": 0,
        "timestamp_sec": float("nan"),
        "pair_motion_energy": 0.0,
        "pair_contact_motion_intensity": 0.0,
    }


def _add_partner_persistence_columns(edges: pd.DataFrame) -> None:
    """Add pair-local persistence audit without exposing partner IDs as X."""

    for _, group in edges.groupby(["pair_id", "actor_node_key"], sort=False):
        ordered = group.sort_values(["slot_index", "neighbor_rank"])
        rank_one = ordered[ordered["neighbor_rank"].eq(1)]
        valid = rank_one["edge_available"].astype(bool)
        keys = rank_one["neighbor_node_key"].astype(str)
        valid_keys = keys[valid & keys.ne("")]
        persistence = (
            float(valid_keys.value_counts().iloc[0] / len(rank_one))
            if len(valid_keys)
            else 0.0
        )
        ordered_valid = valid_keys.reset_index(drop=True)
        switches = (
            int(ordered_valid.ne(ordered_valid.shift()).iloc[1:].sum())
            if len(ordered_valid) > 1
            else 0
        )
        consistent = bool(len(valid_keys) > 0 and valid_keys.nunique() == 1)
        available_ratio = float(valid_keys.size / len(rank_one)) if len(rank_one) else 0.0
        index = group.index
        edges.loc[index, "partner_available_ratio"] = available_ratio
        edges.loc[index, "partner_persistence_ratio"] = persistence
        edges.loc[index, "partner_switch_count"] = switches
        edges.loc[index, "partner_key_consistency"] = consistent
        edges.loc[index, "pair_valid_ratio"] = available_ratio


def _add_edge_temporal_features(edges: pd.DataFrame, frames: pd.DataFrame) -> None:
    """Add contact runs and neutral motion/contact interactions per edge rank."""

    for _, group in edges.groupby(
        ["pair_id", "actor_node_key", "neighbor_rank"], sort=False
    ):
        ordered = group.sort_values("slot_index")
        run = 0
        start_time: float | None = None
        for index, row in ordered.iterrows():
            contact = bool(row["edge_available"] and row["pair_contact"])
            timestamp = _scalar(row.get("timestamp_sec"))
            if contact:
                run += 1
                if start_time is None:
                    start_time = timestamp if np.isfinite(timestamp) else None
            else:
                run = 0
                start_time = None
            if start_time is not None and np.isfinite(timestamp):
                duration_sec = max(0.0, timestamp - start_time + 1.0 / 6.0)
            else:
                duration_sec = run / 6.0
            edges.loc[index, "pair_contact_duration_frames"] = int(run)
            edges.loc[index, "pair_contact_duration_sec"] = float(duration_sec)

    if "pair_contact_duration_sec" not in edges:
        edges["pair_contact_duration_sec"] = 0.0


def _copy_partner_audit_to_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    summary = edges[edges["neighbor_rank"].eq(1)].groupby(
        ["pair_id", "actor_node_key"],
        sort=False,
    ).agg(
        partner_available_ratio=("edge_available", "mean"),
        partner_persistence_ratio=("partner_persistence_ratio", "first"),
        partner_switch_count=("partner_switch_count", "first"),
        partner_key_consistency=("partner_key_consistency", "first"),
        pair_valid_ratio=("edge_available", "mean"),
    ).reset_index()
    summary = summary.set_index(["pair_id", "actor_node_key"])
    for column in (
        "partner_available_ratio",
        "partner_persistence_ratio",
        "partner_switch_count",
        "partner_key_consistency",
        "pair_valid_ratio",
    ):
        values = [
            summary[column].get((str(row.pair_id), str(row.node_key)), np.nan)
            for row in nodes.itertuples(index=False)
        ]
        nodes[column] = pd.Series(values, index=nodes.index).fillna(nodes[column])


def _box_iou(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    intersection = _intersection_box(left, right)
    if intersection is None or left is None or right is None:
        return 0.0
    intersection_area = max(0.0, intersection[2] - intersection[0]) * max(
        0.0, intersection[3] - intersection[1]
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection_area
    return float(intersection_area / union) if union > 0 else 0.0


def _box_overlap_ratio(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    intersection = _intersection_box(left, right)
    if intersection is None or left is None:
        return 0.0
    intersection_area = max(0.0, intersection[2] - intersection[0]) * max(
        0.0, intersection[3] - intersection[1]
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    return float(intersection_area / left_area) if left_area > 0 else 0.0


def _finalize_social_node_audit(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    pair_id: str,
) -> None:
    pair_edges = [row for row in edges if row["pair_id"] == pair_id]
    by_actor: dict[str, list[str]] = {}
    for row in pair_edges:
        by_actor.setdefault(row["actor_node_key"], []).append(row["neighbor_node_key"])
    for node in nodes:
        if node["pair_id"] != pair_id:
            continue
        keys = by_actor.get(node["node_key"], [])
        node["partner_available_ratio"] = float(bool(keys))
        node["partner_key_consistency"] = bool(keys)
        node["pair_valid_ratio"] = float(bool(keys))


def _slot_rows(
    frames: pd.DataFrame,
    slots: pd.DataFrame,
    pair_id: object,
    role: str,
    frame_lookup: dict[str, pd.Series],
) -> pd.DataFrame:
    selected = slots[(slots["pair_id"] == pair_id) & slots["slot_role"].eq(role)]
    rows = [
        frame_lookup[str(value)]
        for value in selected["frame_uid"]
        if str(value) in frame_lookup
    ]
    return pd.DataFrame(rows) if rows else frames.iloc[0:0].copy()


def _build_artifact_audit(**tables: pd.DataFrame) -> dict[str, Any]:
    pairs = tables["pairs"]
    return {
        "schema_version": "classification_v2.pig_strenet_artifacts.v2",
        "status": "PASS_PIG_STRENET_ARTIFACT_SCHEMA",
        "native_event_count": int(pairs["native_event_id"].nunique()),
        "pair_count": int(len(pairs)),
        "source_counts": _counts(pairs, "source_type"),
        "derived_view_counts": _counts(pairs, "derived_view"),
        "event_mass_min": float(pairs.groupby("native_event_id")["event_weight"].sum().min()),
        "event_mass_max": float(pairs.groupby("native_event_id")["event_weight"].sum().max()),
        "slot_count": int(len(tables["slots"])),
        "roi_row_count": int(len(tables["roi_dynamics"])),
        "roi_visual_selection_count": int(len(tables["roi_visual_selection"])),
        "roi_visual_available_count": int(
            tables["roi_visual_selection"]["actor_roi_visual_available"].sum()
        ),
        "social_node_count": int(len(tables["social_nodes"])),
        "social_edge_count": int(len(tables["social_edges"])),
        "control_count": int(len(tables["controls"])),
        "model_x_columns": model_x_columns(tables["history_features"]),
        "target_selected_roi_used": False,
        "behavior_selected_partner_used": False,
        "future_frame_used": False,
        "errors": [],
        "warnings": [],
        "valid": True,
    }


def model_x_columns(
    features: pd.DataFrame,
    *,
    include_availability: bool = False,
) -> list[str]:
    """Return causal numeric features, excluding audit/mask fields by default."""

    excluded = {
        "pair_id",
        "native_event_id",
        "source_type",
        "event_weight",
    }
    columns: list[str] = []
    for column in features.columns:
        lower = column.lower()
        if column in excluded or any(token in lower for token in MODEL_X_FORBIDDEN_TOKENS):
            continue
        if not include_availability and any(
            token in lower for token in MODEL_X_AUDIT_TOKENS
        ):
            continue
        if (
            pd.api.types.is_numeric_dtype(features[column])
            or pd.api.types.is_bool_dtype(features[column])
        ):
            columns.append(column)
    return columns


def availability_columns(features: pd.DataFrame) -> list[str]:
    """Return audit/mask fields reserved for HA and missingness controls."""

    columns: list[str] = []
    for column in features.columns:
        lower = column.lower()
        if any(token in lower for token in MODEL_X_AUDIT_TOKENS) and (
            pd.api.types.is_numeric_dtype(features[column])
            or pd.api.types.is_bool_dtype(features[column])
        ):
            columns.append(column)
    return columns


def _validate_model_x_names(features: pd.DataFrame) -> None:
    forbidden = [
        name
        for name in model_x_columns(features)
        if any(token in name.lower() for token in ("target_roi", "label", "behavior"))
    ]
    if forbidden:
        raise ValueError(f"forbidden Pig-STRENet model-X columns={forbidden}")


def _validate_frame_input(frames: pd.DataFrame) -> None:
    missing = sorted(PAIR_REQUIRED_COLUMNS.difference(frames.columns))
    if missing:
        raise ValueError(f"Pig-STRENet frames missing columns={missing}")
    if frames.empty:
        raise ValueError("Pig-STRENet frames must not be empty")
    if frames["frame_uid"].astype(str).duplicated().any():
        raise ValueError("Pig-STRENet frame_uid must be unique")


def _normalize_frames(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    out["frame_index"] = pd.to_numeric(out["frame_index"], errors="raise").astype(int)
    for column in (
        "timestamp_sec",
        "relative_frame_index",
        "label_window_start",
        "label_window_end",
    ):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in out.columns:
        if column.endswith("_available") or column.endswith("_contact") or column.endswith("_near"):
            out[column] = _bool_series(out[column])
    return out.sort_values(
        ["object_track_key", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _actual_frame_for_coordinate(
    frame: pd.DataFrame,
    coordinate: int,
    *,
    relative_coordinates: bool,
) -> int:
    return _fallback_slot_frame_index(
        frame,
        int(coordinate),
        relative_coordinates=relative_coordinates,
    )


def _optional_int(value: Any) -> int | None:
    number = _scalar(value)
    return int(number) if np.isfinite(number) else None


def _audit_event_mass(pairs: pd.DataFrame) -> None:
    mass = pairs.groupby("native_event_id", sort=False)["event_weight"].sum()
    if not np.allclose(mass.to_numpy(dtype=float), 1.0, atol=1e-9):
        raise ValueError("derived pair event mass does not sum to one")


def _difference_row(
    index: int,
    valid: bool,
    shift_x: float,
    shift_y: float,
    diff: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(diff, dtype=float)
    height, width = values.shape
    inner = values[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)]
    boundary = values.copy()
    boundary[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)] = np.nan
    finite = values[np.isfinite(values)]
    return {
        "pair_slot_index": int(index),
        "pair_valid": bool(valid),
        "shift_x_px": float(shift_x),
        "shift_y_px": float(shift_y),
        "diff_mean": _mean(finite),
        "diff_std": _std(finite),
        "diff_p90": _quantile(finite, 0.90),
        "diff_p95": _quantile(finite, 0.95),
        "diff_active_pixel_ratio": _ratio(finite > 0.10),
        "diff_inner_mean": _mean(inner[np.isfinite(inner)]),
        "diff_boundary_mean": _mean(boundary[np.isfinite(boundary)]),
    }


def _bounded_phase_shift(
    previous: np.ndarray,
    current: np.ndarray,
    maximum: float,
) -> tuple[float, float]:
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            np.float32(previous),
            np.float32(current),
        )
    except cv2.error:
        return 0.0, 0.0
    if not np.isfinite(response) or response <= 0:
        return 0.0, 0.0
    return float(np.clip(shift_x, -maximum, maximum)), float(np.clip(shift_y, -maximum, maximum))


def _number_or_nan(row: Any, column: str) -> float:
    if row is None:
        return float("nan")
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _number_or_zero(row: Any, column: str) -> float:
    value = _number_or_nan(row, column)
    return 0.0 if not np.isfinite(value) else value


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.asarray([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _bool_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.zeros(len(frame), dtype=bool)
    return _bool_series(frame[column]).to_numpy(dtype=bool)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _scalar(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _first_number(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _frame_min(frame: pd.DataFrame) -> int:
    return int(pd.to_numeric(frame["frame_index"], errors="raise").min())


def _frame_max(frame: pd.DataFrame) -> int:
    return int(pd.to_numeric(frame["frame_index"], errors="raise").max())


def _fallback_slot_frame_index(
    unit: pd.DataFrame,
    coordinate: int,
    *,
    relative_coordinates: bool,
) -> int:
    if relative_coordinates and "relative_frame_index" in unit:
        relative = pd.to_numeric(unit["relative_frame_index"], errors="coerce")
        actual = pd.to_numeric(unit["frame_index"], errors="coerce")
        observed = actual[relative.eq(coordinate)].dropna()
        if not observed.empty:
            return int(observed.iloc[0])
        base = actual.min() - relative.min()
        return int(base + coordinate)
    return int(coordinate)


def _gap_count(observed: set[int], expected: Iterable[int]) -> int:
    expected_set = set(int(value) for value in expected)
    return int(len(expected_set.difference(observed)))


def _max_gap_seconds(frame: pd.DataFrame, expected: Iterable[int]) -> float:
    if frame.empty or "timestamp_sec" not in frame:
        return 0.0
    times = pd.to_numeric(frame["timestamp_sec"], errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return 0.0
    return float(times.diff().dropna().max())


def _duration_seconds(frame: pd.DataFrame) -> float:
    if frame.empty or "timestamp_sec" not in frame:
        return 0.0
    values = pd.to_numeric(frame["timestamp_sec"], errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    deltas = values.sort_values().diff().dropna()
    step = float(deltas.median()) if not deltas.empty else 0.0
    return float(max(0.0, values.max() - values.min() + max(step, 0.0)))


def _boundary_gap_seconds(history: pd.DataFrame, target: pd.DataFrame) -> float:
    if (
        history.empty
        or target.empty
        or "timestamp_sec" not in history
        or "timestamp_sec" not in target
    ):
        return 0.0
    previous = pd.to_numeric(history["timestamp_sec"], errors="coerce").dropna()
    current = pd.to_numeric(target["timestamp_sec"], errors="coerce").dropna()
    if previous.empty or current.empty:
        return 0.0
    return float(max(0.0, current.min() - previous.max()))


def _endpoint_displacement(frame: pd.DataFrame) -> float:
    if len(frame) < 2 or not {"cx_n", "cy_n"}.issubset(frame.columns):
        return 0.0
    ordered = frame.sort_values("frame_index")
    x = pd.to_numeric(ordered["cx_n"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ordered["cy_n"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite([x[0], y[0], x[-1], y[-1]]).all():
        return 0.0
    return float(np.hypot(x[-1] - x[0], y[-1] - y[0]))


def _duration_from_mask(frame: pd.DataFrame, mask: np.ndarray) -> float:
    if "timestamp_sec" not in frame:
        return float(np.sum(mask))
    timestamps = pd.to_numeric(frame["timestamp_sec"], errors="coerce").to_numpy(dtype=float)
    valid = mask & np.isfinite(timestamps)
    if valid.sum() < 2:
        return float(valid.sum())
    return float(np.sum(np.diff(timestamps)[valid[:-1]]))


def _slope(frame: pd.DataFrame, column: str) -> float:
    values = _values(frame.sort_values("frame_index"), column)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    return float(np.polyfit(x, y, 1)[0])


def _partner_persistence(frame: pd.DataFrame) -> float:
    column = "nearest_track_id" if "nearest_track_id" in frame else "nearest_pig_id"
    if column not in frame:
        return 0.0
    values = frame[column].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    return float(values.value_counts().iloc[0] / len(values)) if len(values) else 0.0


def _partner_switch_count(frame: pd.DataFrame) -> int:
    column = "nearest_track_id" if "nearest_track_id" in frame else "nearest_pig_id"
    if column not in frame:
        return 0
    values = frame.sort_values("frame_index")[column].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    return int(values.ne(values.shift()).iloc[1:].sum()) if len(values) > 1 else 0


def _mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else 0.0


def _min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if len(finite) else 0.0


def _std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite)) if len(finite) else 0.0


def _quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else 0.0


def _ratio(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.mean(array)) if len(array) else 0.0


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).items()
    }


__all__ = [
    "CONTROL_IDS",
    "HISTORY_LENGTH",
    "MODEL_X_AUDIT_TOKENS",
    "MODEL_X_FORBIDDEN_TOKENS",
    "PigSTRENetArtifacts",
    "TARGET_LENGTH",
    "build_history_control_matrix",
    "build_pig_strenet_artifacts",
    "compute_stabilized_difference_maps",
    "model_x_columns",
    "availability_columns",
]
