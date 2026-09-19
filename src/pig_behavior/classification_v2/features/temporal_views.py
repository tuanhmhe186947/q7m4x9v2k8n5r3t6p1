"""Audited temporal input views built after temporal harmonization.

The primary view consumes only exact ``T6_contiguous`` final windows. Legacy
and CVAT windows share this view when their selected source frames are dense;
``S6@16`` remains a separately identified legacy-only ablation. Every input
window stays in a selection ledger, including non-primary views.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import (
    validate_model_input_columns,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
    require_ordered_window_ids,
)

FIXED6_OBSERVED_TIME = "fixed6_observed_time"
FIXED6_NORMALIZED_PHASE = "fixed6_normalized_phase"
NATIVE6_16 = "native6_16"
TEMPORAL_VIEW_VERSION = "classification_v2_temporal_views_v2"

SOURCE_NATIVE_LENGTHS = {
    "legacy_recovered": 16,
    "cvat_tracking_xml": 6,
}

MODEL_TENSOR_COLUMNS = [
    "time_value",
    "time_delta",
    "length_mask",
    "observed_mask",
    "timing_valid_mask",
    "bbox_quality_mask",
    "spatiotemporal_quality_mask",
    "roi_feeder_available_mask",
    "roi_drinker_available_mask",
    "roi_toy_available_mask",
    "social_neighbor_available_mask",
    "actor_context_available_mask",
    "partner_context_available_mask",
]

AVAILABILITY_SOURCES = {
    "roi_feeder_available_mask": "roi_feeder_available",
    "roi_drinker_available_mask": "roi_drinker_available",
    "roi_toy_available_mask": "roi_toy_available",
    "actor_context_available_mask": "image_context_loadable",
    "partner_context_available_mask": "partner_context_available",
}

OUTPUT_FILENAMES = {
    "selection": "temporal_view_selection_manifest.csv",
    "observed": "fixed6_observed_time_manifest.csv",
    "phase": "fixed6_normalized_phase_manifest.csv",
    "native": "native6_16_manifest.csv",
    "contract": "temporal_view_contract.json",
    "audit": "temporal_view_audit.json",
}


@dataclass(slots=True)
class TemporalViewResult:
    """All temporal-view artifacts produced from one ordered input packet."""

    selection_manifest: pd.DataFrame
    fixed6_observed_time_manifest: pd.DataFrame
    fixed6_normalized_phase_manifest: pd.DataFrame
    native6_16_manifest: pd.DataFrame
    contract: dict[str, Any]
    audit: dict[str, Any]


def build_temporal_views(
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    fixed_length: int = 6,
) -> TemporalViewResult:
    """Build fixed-six and native-length views without dropping source rows."""

    if fixed_length != 6:
        raise ValueError("classification_v2 primary temporal view must have six slots")
    work_windows = _validate_windows(windows)
    work_frames = _validate_frames(frames)
    work_intervals = _validate_intervals(intervals)
    _validate_cross_artifact_contract(
        work_windows,
        work_frames,
        work_intervals,
    )

    selection = _build_selection_manifest(work_windows, fixed_length)
    fixed_windows = work_windows.loc[selection["fixed6_keep"]].copy()
    frame_lookup = _frame_lookup(work_frames)
    interval_lookup = work_intervals.set_index("temporal_unit_key")
    observed = _build_fixed_slots(
        fixed_windows,
        frame_lookup,
        interval_lookup,
        view_name=FIXED6_OBSERVED_TIME,
        fixed_length=fixed_length,
    )
    phase = _normalized_phase_view(observed)
    native = _build_native_slots(
        work_intervals,
        frame_lookup,
    )
    contract = _build_contract(work_frames, fixed_length)
    audit = _audit_views(
        work_windows,
        work_frames,
        work_intervals,
        selection,
        observed,
        phase,
        native,
        contract,
    )
    if audit["errors"]:
        raise ValueError(f"temporal view contract failed: {audit['errors']}")
    return TemporalViewResult(
        selection_manifest=selection,
        fixed6_observed_time_manifest=observed,
        fixed6_normalized_phase_manifest=phase,
        native6_16_manifest=native,
        contract=contract,
        audit=audit,
    )


def write_temporal_view_outputs(
    result: TemporalViewResult,
    output_dir: Path,
    *,
    overwrite: bool = False,
    input_artifacts: dict[str, Path] | None = None,
) -> dict[str, str]:
    """Persist one complete packet and refuse implicit derived-output overwrite."""

    paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing and not overwrite:
        raise FileExistsError(f"temporal view outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result.selection_manifest.to_csv(paths["selection"], index=False)
    result.fixed6_observed_time_manifest.to_csv(paths["observed"], index=False)
    result.fixed6_normalized_phase_manifest.to_csv(paths["phase"], index=False)
    result.native6_16_manifest.to_csv(paths["native"], index=False)

    contract = dict(result.contract)
    contract["artifact_contract"] = {
        "selection": _artifact_key_contract(
            result.selection_manifest,
            "window_id",
        ),
        "fixed6_observed_time": _artifact_key_contract(
            result.fixed6_observed_time_manifest,
            "slot_key",
        ),
        "fixed6_normalized_phase": _artifact_key_contract(
            result.fixed6_normalized_phase_manifest,
            "slot_key",
        ),
        "native6_16": _artifact_key_contract(
            result.native6_16_manifest,
            "slot_key",
        ),
    }
    contract["input_artifacts"] = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in sorted((input_artifacts or {}).items())
    }
    _write_json(paths["contract"], contract)
    audit = dict(result.audit)
    audit["artifacts"] = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in paths.items()
        if name != "audit"
    }
    _write_json(paths["audit"], audit)
    return {f"{name}_path": str(path) for name, path in paths.items()}


def _validate_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Validate ordered window identity, spans, and declared native-unit keys."""

    required = {
        "window_id",
        "source_type",
        "object_track_key",
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
        "temporal_unit_keys_json",
        "feature_computation_grain",
        "pair_scope_key",
        "view_type",
        "sampling_pattern",
        "selected_frame_indices",
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    }
    _require_columns(windows, required, "windows")
    out = windows.reset_index(drop=True).copy()
    require_ordered_window_ids("temporal_view_input", out["window_id"])
    for column in [
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ]:
        out[column] = _strict_integer_series(out[column], column)
    invalid = (
        out["window_length_frames"].le(0)
        | out["window_start_frame"].lt(0)
        | out["window_end_frame"].lt(out["window_start_frame"])
    )
    if invalid.any():
        raise ValueError(f"invalid window span rows={int(invalid.sum())}")
    unsupported = sorted(set(out["source_type"].astype(str)) - set(SOURCE_NATIVE_LENGTHS))
    if unsupported:
        raise ValueError(f"unsupported temporal-view window sources={unsupported}")
    blank_object = out["object_track_key"].fillna("").astype(str).str.strip().eq("")
    if blank_object.any():
        raise ValueError(f"blank object_track_key window rows={int(blank_object.sum())}")
    out["temporal_unit_keys_parsed"] = out["temporal_unit_keys_json"].map(
        _parse_unit_keys
    )
    out["selected_frame_indices_parsed"] = [
        _parse_selected_frame_indices(row)
        for row in out.itertuples(index=False)
    ]
    invalid_grain = out["feature_computation_grain"].astype(str).ne(
        "FINAL_VIEW_FEATURES"
    )
    invalid_scope = out["pair_scope_key"].astype(str).ne(
        out["window_id"].astype(str)
    )
    invalid_recompute = ~out["pair_recomputed_for_view"].map(_bool_scalar)
    invalid_recompute |= ~out["aggregate_recomputed_for_view"].map(
        _bool_scalar
    )
    if invalid_grain.any() or invalid_scope.any() or invalid_recompute.any():
        raise ValueError(
            "invalid final-view computation contract: "
            f"grain={int(invalid_grain.sum())}, "
            f"scope={int(invalid_scope.sum())}, "
            f"recompute={int(invalid_recompute.sum())}"
        )
    return out


def _validate_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Validate one object observation per stable frame and temporal-unit slot."""

    required = {
        "frame_uid",
        "source_type",
        "object_track_key",
        "temporal_unit_key",
        "frame_index",
        "timestamp_sec",
        "bbox_valid",
        "spatiotemporal_feature_valid",
    }
    _require_columns(frames, required, "frames")
    out = frames.reset_index(drop=True).copy()
    out["frame_index"] = _strict_integer_series(out["frame_index"], "frame_index")
    for column in ["frame_uid", "object_track_key", "temporal_unit_key"]:
        cleaned = out[column].fillna("").astype(str).str.strip()
        if cleaned.eq("").any():
            raise ValueError(f"blank {column} frame rows={int(cleaned.eq('').sum())}")
        out[column] = cleaned
    duplicate_uid = int(out["frame_uid"].duplicated(keep=False).sum())
    duplicate_track_frame = int(
        out.duplicated(["object_track_key", "frame_index"], keep=False).sum()
    )
    duplicate_unit_frame = int(
        out.duplicated(["temporal_unit_key", "frame_index"], keep=False).sum()
    )
    if duplicate_uid or duplicate_track_frame or duplicate_unit_frame:
        raise ValueError(
            "duplicate frame alignment: "
            f"frame_uid={duplicate_uid}, "
            f"object_frame={duplicate_track_frame}, "
            f"unit_frame={duplicate_unit_frame}"
        )
    out["timestamp_sec"] = pd.to_numeric(out["timestamp_sec"], errors="coerce")
    return out


def _validate_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Validate native temporal units against the settled 16/6 source contract."""

    required = {
        "temporal_unit_key",
        "source_type",
        "object_track_key",
        "label_window_start",
        "label_window_end",
        "label_frame_count",
    }
    _require_columns(intervals, required, "intervals")
    out = intervals.reset_index(drop=True).copy()
    unit_keys = out["temporal_unit_key"].fillna("").astype(str).str.strip()
    blank = int(unit_keys.eq("").sum())
    duplicate = int(unit_keys.duplicated(keep=False).sum())
    if blank or duplicate:
        raise ValueError(
            f"invalid temporal_unit_key: blank={blank}, duplicate_rows={duplicate}"
        )
    out["temporal_unit_key"] = unit_keys
    for column in ["label_window_start", "label_window_end", "label_frame_count"]:
        out[column] = _strict_integer_series(out[column], column)
    span = out["label_window_end"] - out["label_window_start"] + 1
    if span.ne(out["label_frame_count"]).any():
        count = int(span.ne(out["label_frame_count"]).sum())
        raise ValueError(f"native interval span mismatch rows={count}")
    unsupported = sorted(set(out["source_type"].astype(str)) - set(SOURCE_NATIVE_LENGTHS))
    if unsupported:
        raise ValueError(f"unsupported temporal-view sources={unsupported}")
    expected = out["source_type"].astype(str).map(SOURCE_NATIVE_LENGTHS).astype(int)
    wrong_length = out["label_frame_count"].ne(expected)
    if wrong_length.any():
        details = (
            out.loc[wrong_length, ["source_type", "label_frame_count"]]
            .value_counts()
            .to_dict()
        )
        raise ValueError(f"source native-length contract mismatch={details}")
    return out


def _validate_cross_artifact_contract(
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    """Prove source, object, and native-unit membership across all inputs."""

    interval_keys = set(intervals["temporal_unit_key"])
    frame_keys = set(frames["temporal_unit_key"])
    missing_frames = sorted(interval_keys - frame_keys)
    extra_frames = sorted(frame_keys - interval_keys)
    if missing_frames or extra_frames:
        raise ValueError(
            "frame/interval temporal-unit coverage mismatch: "
            f"missing={len(missing_frames)}, extra={len(extra_frames)}"
        )
    interval_meta = intervals.set_index("temporal_unit_key")[
        ["source_type", "object_track_key"]
    ]
    frame_meta = frames.groupby("temporal_unit_key", sort=False).agg(
        source_count=("source_type", "nunique"),
        object_count=("object_track_key", "nunique"),
        source_type=("source_type", "first"),
        object_track_key=("object_track_key", "first"),
    )
    conflict = frame_meta["source_count"].ne(1) | frame_meta["object_count"].ne(1)
    if conflict.any():
        raise ValueError(f"native-unit frame metadata conflicts={int(conflict.sum())}")
    aligned = frame_meta.join(
        interval_meta,
        rsuffix="_interval",
        how="left",
        validate="one_to_one",
    )
    mismatch = (
        aligned["source_type"].astype(str).ne(aligned["source_type_interval"].astype(str))
        | aligned["object_track_key"]
        .astype(str)
        .ne(aligned["object_track_key_interval"].astype(str))
    )
    if mismatch.any():
        raise ValueError(f"frame/interval metadata mismatch units={int(mismatch.sum())}")

    frame_bounds = frames.merge(
        intervals[
            ["temporal_unit_key", "label_window_start", "label_window_end"]
        ],
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
    )
    outside = (
        frame_bounds["frame_index"].lt(frame_bounds["label_window_start"])
        | frame_bounds["frame_index"].gt(frame_bounds["label_window_end"])
    )
    if outside.any():
        raise ValueError(f"frame rows outside native interval={int(outside.sum())}")

    declared_keys = {
        key
        for values in windows["temporal_unit_keys_parsed"]
        for key in values
    }
    unknown = sorted(declared_keys - interval_keys)
    if unknown:
        raise ValueError(f"window references unknown temporal units={len(unknown)}")
    for row in windows.itertuples(index=False):
        for unit_key in row.temporal_unit_keys_parsed:
            interval = interval_meta.loc[unit_key]
            if str(row.source_type) != str(interval["source_type"]):
                raise ValueError(f"window/native source mismatch={row.window_id}")
            if str(row.object_track_key) != str(interval["object_track_key"]):
                raise ValueError(f"window/native object mismatch={row.window_id}")


def _build_selection_manifest(
    windows: pd.DataFrame,
    fixed_length: int,
) -> pd.DataFrame:
    """Retain every input window and mark the deterministic primary subset."""

    selected = (
        windows["view_type"].astype(str).eq(f"T{fixed_length}_contiguous")
        & windows["sampling_pattern"].astype(str).eq("contiguous")
        & windows["window_length_frames"].eq(fixed_length)
    )
    unit_counts = windows["temporal_unit_keys_parsed"].map(len)
    invalid_selected = selected & unit_counts.lt(1)
    if invalid_selected.any():
        raise ValueError(
            "fixed-six windows must map to at least one native unit: "
            f"rows={int(invalid_selected.sum())}"
        )
    keep_columns = [
        "window_id",
        "source_type",
        "object_track_key",
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
        "temporal_unit_keys_json",
        "view_type",
        "sampling_pattern",
        "selected_frame_indices",
    ]
    for optional in ["behavior_window_label", "window_valid_for_main_train"]:
        if optional in windows.columns:
            keep_columns.append(optional)
    out = windows[keep_columns].copy()
    out.insert(0, "input_window_order", np.arange(len(out), dtype=np.int64))
    out["native_unit_count_in_window"] = unit_counts.astype(int)
    out["fixed6_keep"] = selected.astype(bool)
    out["fixed6_reason"] = "retained_nonprimary_length_for_audit"
    out.loc[selected, "fixed6_reason"] = "selected_exact_T6_contiguous_window"
    return out


def _frame_lookup(frames: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    """Index harmonized object rows without relying on input row order."""

    return {
        (str(row.object_track_key), int(row.frame_index)): row._asdict()
        for row in frames.itertuples(index=False)
    }


def _build_fixed_slots(
    windows: pd.DataFrame,
    frame_lookup: dict[tuple[str, int], dict[str, Any]],
    interval_lookup: pd.DataFrame,
    *,
    view_name: str,
    fixed_length: int,
) -> pd.DataFrame:
    """Expand selected windows into six deterministic, auditable frame slots."""

    records: list[dict[str, Any]] = []
    for item_order, row in enumerate(windows.itertuples(index=False)):
        unit_keys = list(row.temporal_unit_keys_parsed)
        source_type = str(row.source_type)
        wanted = list(row.selected_frame_indices_parsed)
        if len(wanted) != fixed_length:
            raise ValueError(f"fixed window slot mismatch={row.window_id}")
        for slot_index, frame_index in enumerate(wanted):
            frame = frame_lookup.get((str(row.object_track_key), frame_index))
            candidate_units = [
                unit_key
                for unit_key in unit_keys
                if int(interval_lookup.loc[unit_key, "label_window_start"])
                <= frame_index
                <= int(interval_lookup.loc[unit_key, "label_window_end"])
            ]
            if len(candidate_units) != 1:
                raise ValueError(
                    "fixed window frame lacks unique constituent authority="
                    f"{row.window_id}@{frame_index}"
                )
            unit_key = candidate_units[0]
            interval = interval_lookup.loc[unit_key]
            if source_type != str(interval["source_type"]):
                raise ValueError(f"fixed window source mismatch={row.window_id}")
            if str(row.object_track_key) != str(interval["object_track_key"]):
                raise ValueError(f"fixed window object mismatch={row.window_id}")
            if frame is not None and str(frame["temporal_unit_key"]) != unit_key:
                raise ValueError(
                    f"fixed window frame/unit mismatch={row.window_id}@{frame_index}"
                )
            records.append(
                _slot_record(
                    view_name=view_name,
                    view_item_id=str(row.window_id),
                    parent_window_id=str(row.window_id),
                    temporal_unit_key=unit_key,
                    source_type=source_type,
                    source_native_length=int(SOURCE_NATIVE_LENGTHS[source_type]),
                    item_order=item_order,
                    slot_index=slot_index,
                    declared_sequence_length=fixed_length,
                    frame_index=frame_index,
                    frame=frame,
                )
            )
    result = pd.DataFrame.from_records(records, columns=_slot_columns())
    return _observed_time_view(result)


def _build_native_slots(
    intervals: pd.DataFrame,
    frame_lookup: dict[tuple[str, int], dict[str, Any]],
) -> pd.DataFrame:
    """Expand every native unit to its expected 6/16 slots without padding."""

    records: list[dict[str, Any]] = []
    for item_order, interval in enumerate(intervals.itertuples(index=False)):
        source_type = str(interval.source_type)
        native_length = int(SOURCE_NATIVE_LENGTHS[source_type])
        wanted = range(
            int(interval.label_window_start),
            int(interval.label_window_end) + 1,
        )
        for slot_index, frame_index in enumerate(wanted):
            frame = frame_lookup.get((str(interval.object_track_key), frame_index))
            if frame is not None:
                observed_unit = str(frame["temporal_unit_key"])
                if observed_unit != str(interval.temporal_unit_key):
                    raise ValueError(
                        "native slot maps to another temporal unit: "
                        f"{interval.temporal_unit_key}@{frame_index}={observed_unit}"
                    )
            records.append(
                _slot_record(
                    view_name=NATIVE6_16,
                    view_item_id=str(interval.temporal_unit_key),
                    parent_window_id="",
                    temporal_unit_key=str(interval.temporal_unit_key),
                    source_type=source_type,
                    source_native_length=native_length,
                    item_order=item_order,
                    slot_index=slot_index,
                    declared_sequence_length=native_length,
                    frame_index=frame_index,
                    frame=frame,
                )
            )
    result = pd.DataFrame.from_records(records, columns=_slot_columns())
    return _observed_time_view(result)


def _slot_record(
    *,
    view_name: str,
    view_item_id: str,
    parent_window_id: str,
    temporal_unit_key: str,
    source_type: str,
    source_native_length: int,
    item_order: int,
    slot_index: int,
    declared_sequence_length: int,
    frame_index: int,
    frame: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create one expected slot and preserve missing observations with masks."""

    observed = frame is not None
    timestamp = _finite_float(frame.get("timestamp_sec")) if observed else None
    record: dict[str, Any] = {
        "temporal_view_name": view_name,
        "temporal_view_version": TEMPORAL_VIEW_VERSION,
        "view_item_id": view_item_id,
        "parent_window_id": parent_window_id,
        "temporal_unit_key": temporal_unit_key,
        "source_type": source_type,
        "source_native_length_audit": source_native_length,
        "item_order": item_order,
        "slot_index": slot_index,
        "slot_key": f"{view_item_id}|slot={slot_index}",
        "declared_sequence_length": declared_sequence_length,
        "frame_index_expected_audit": frame_index,
        "frame_uid_audit": str(frame.get("frame_uid", "")) if observed else "",
        "timestamp_sec_audit": timestamp,
        "observed_time_delta_sec_audit": None,
        "time_coordinate_kind": "observed_seconds",
        "time_value": None,
        "time_delta": None,
        "length_mask": True,
        "observed_mask": observed,
        "timing_valid_mask": timestamp is not None,
        "bbox_quality_mask": _frame_bool(frame, "bbox_valid"),
        "spatiotemporal_quality_mask": _frame_bool(
            frame,
            "spatiotemporal_feature_valid",
        ),
        "roi_feeder_available_mask": False,
        "roi_drinker_available_mask": False,
        "roi_toy_available_mask": False,
        "social_neighbor_available_mask": _social_available(frame),
        "actor_context_available_mask": False,
        "partner_context_available_mask": False,
        "padding_mask": False,
    }
    for output_column, source_column in AVAILABILITY_SOURCES.items():
        record[output_column] = _frame_bool(frame, source_column)
    return record


def _observed_time_view(slots: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative observed time while retaining invalid timing masks."""

    result = slots.copy()
    for _, indices in result.groupby("view_item_id", sort=False).groups.items():
        positions = list(indices)
        timestamps = pd.to_numeric(
            result.loc[positions, "timestamp_sec_audit"],
            errors="coerce",
        )
        valid_values = timestamps.dropna()
        if not valid_values.empty and not valid_values.is_monotonic_increasing:
            item = str(result.loc[positions[0], "view_item_id"])
            raise ValueError(f"nonmonotonic observed timestamps={item}")
        if valid_values.duplicated().any():
            item = str(result.loc[positions[0], "view_item_id"])
            raise ValueError(f"duplicate observed timestamps={item}")
        origin = float(valid_values.iloc[0]) if not valid_values.empty else np.nan
        previous = np.nan
        for position, timestamp in zip(positions, timestamps, strict=True):
            if pd.isna(timestamp):
                continue
            current = float(timestamp)
            result.at[position, "time_value"] = current - origin
            delta = 0.0 if pd.isna(previous) else current - previous
            result.at[position, "time_delta"] = delta
            result.at[position, "observed_time_delta_sec_audit"] = delta
            previous = current
    return result


def _normalized_phase_view(observed: pd.DataFrame) -> pd.DataFrame:
    """Use identical slots but replace absolute timing with normalized phase."""

    phase = observed.copy()
    phase["temporal_view_name"] = FIXED6_NORMALIZED_PHASE
    phase["time_coordinate_kind"] = "normalized_phase"
    denominator = (phase["declared_sequence_length"] - 1).clip(lower=1)
    phase["time_value"] = phase["slot_index"] / denominator
    phase["time_delta"] = phase.groupby("view_item_id", sort=False)[
        "time_value"
    ].diff()
    first_slot = phase["slot_index"].eq(0)
    phase.loc[first_slot, "time_delta"] = 0.0
    phase["timing_valid_mask"] = phase["length_mask"].astype(bool)
    return phase


def _build_contract(frames: pd.DataFrame, fixed_length: int) -> dict[str, Any]:
    """Describe tensor fields, audit metadata, and non-resampling semantics."""

    model_schema = validate_model_input_columns(MODEL_TENSOR_COLUMNS)
    source_fields = {
        output: {
            "source_column": source,
            "source_column_present": source in frames.columns,
        }
        for output, source in AVAILABILITY_SOURCES.items()
    }
    source_fields["social_neighbor_available_mask"] = {
        "source_column": "nearest_pig_id|nearest_track_id",
        "source_column_present": (
            "nearest_pig_id" in frames.columns or "nearest_track_id" in frames.columns
        ),
    }
    return {
        "schema_version": TEMPORAL_VIEW_VERSION,
        "primary_view": FIXED6_OBSERVED_TIME,
        "diagnostic_view": FIXED6_NORMALIZED_PHASE,
        "native_ablation_view": NATIVE6_16,
        "fixed_length": fixed_length,
        "source_native_lengths": SOURCE_NATIVE_LENGTHS,
        "selection_rule": "exact_final_T6_contiguous_only",
        "legacy_fixed6_policy": (
            "select six consecutive source frames; S6@16 remains a "
            "separate legacy-only ablation"
        ),
        "cvat_fixed6_policy": (
            "select six consecutive source frames; a window may cross "
            "reviewed interval boundaries only with complete constituent "
            "authority"
        ),
        "unused_window_policy": "retain every input window in the selection manifest",
        "native_unit_policy": "preserve every native unit and every expected slot",
        "fixed6_padding_policy": "six conceptual slots and zero padding for both sources",
        "model_tensor_columns": MODEL_TENSOR_COLUMNS,
        "model_input_schema_audit": model_schema,
        "availability_sources": source_fields,
        "audit_only_column_families": [
            "stable identifiers",
            "source metadata",
            "native lengths",
            "frame indices and timestamps",
            "behavior/review metadata when present",
        ],
        "forbidden_model_input_families": [
            "source_type",
            "source_native_length_audit",
            "window_id",
            "temporal_unit_key",
            "frame_uid_audit",
            "frame_index_expected_audit",
            "timestamp_sec_audit",
            "behavior",
            "manual_*",
            "review_*",
            "paths",
            "fold identifiers",
        ],
        "window_uid_created": False,
        "training_authorized": False,
        "authorization_note": (
            "A valid temporal-view contract is technical evidence only. "
            "Human Hidden and behavior review gates remain independent."
        ),
    }


def _audit_views(
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
    selection: pd.DataFrame,
    observed: pd.DataFrame,
    phase: pd.DataFrame,
    native: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Return structural evidence and fail-closed error details."""

    errors: list[str] = []
    warnings: list[str] = []
    selected_ids = selection.loc[selection["fixed6_keep"], "window_id"].reset_index(
        drop=True
    )
    observed_ids = _ordered_item_ids(observed)
    phase_ids = _ordered_item_ids(phase)
    try:
        alignment = require_ordered_window_ids(
            "fixed6_selection",
            selected_ids,
            {
                "fixed6_observed": observed_ids,
                "fixed6_phase": phase_ids,
            },
        )
    except ValueError as exc:
        errors.append(str(exc))
        alignment = {}
    if len(selection) != len(windows):
        errors.append(f"selection_row_loss={len(windows)}->{len(selection)}")
    if not selection["window_id"].reset_index(drop=True).equals(
        windows["window_id"].reset_index(drop=True)
    ):
        errors.append("selection_window_order_changed")

    for name, slots, expected_length in [
        (FIXED6_OBSERVED_TIME, observed, 6),
        (FIXED6_NORMALIZED_PHASE, phase, 6),
    ]:
        slot_errors = _slot_table_errors(slots, expected_length=expected_length)
        errors.extend(f"{name}:{error}" for error in slot_errors)
    errors.extend(
        f"{NATIVE6_16}:{error}"
        for error in _slot_table_errors(native, expected_length=None)
    )

    fixed_identity_columns = [
        "view_item_id",
        "parent_window_id",
        "temporal_unit_key",
        "source_type",
        "source_native_length_audit",
        "item_order",
        "slot_index",
        "slot_key",
        "declared_sequence_length",
        "frame_index_expected_audit",
        "frame_uid_audit",
        "observed_mask",
        "length_mask",
        "padding_mask",
    ]
    identity_equal = observed[fixed_identity_columns].equals(
        phase[fixed_identity_columns]
    )
    if not identity_equal:
        errors.append("fixed6_observed_phase_membership_or_order_mismatch")

    observed_units = set(observed["temporal_unit_key"].astype(str))
    expected_units = set(intervals["temporal_unit_key"].astype(str))
    missing_fixed_units = sorted(expected_units - observed_units)
    if missing_fixed_units:
        errors.append(f"native_units_missing_from_fixed6={len(missing_fixed_units)}")
    native_units = _ordered_item_ids(native)
    if set(native_units) != expected_units or len(native_units) != len(expected_units):
        errors.append("native_view_unit_coverage_mismatch")

    missing_fixed_slots = int((~observed["observed_mask"].astype(bool)).sum())
    missing_native_slots = int((~native["observed_mask"].astype(bool)).sum())
    if missing_fixed_slots:
        warnings.append(f"fixed6_missing_observed_slots={missing_fixed_slots}")
    if missing_native_slots:
        warnings.append(f"native_missing_observed_slots={missing_native_slots}")
    model_schema = contract["model_input_schema_audit"]
    if not model_schema["valid"]:
        errors.append(
            f"invalid_temporal_tensor_schema={model_schema['forbidden_columns']}"
        )
    if "window_uid" in set(selection.columns).union(observed.columns).union(native.columns):
        errors.append("forbidden_window_uid_created")

    audit = {
        "schema_version": "classification_v2_temporal_view_audit_v1",
        "input_window_rows": int(len(windows)),
        "selection_rows": int(len(selection)),
        "input_frame_rows": int(len(frames)),
        "input_native_units": int(len(intervals)),
        "fixed6_window_rows": int(len(selected_ids)),
        "fixed6_slot_rows": int(len(observed)),
        "native_slot_rows": int(len(native)),
        "source_window_counts": _group_count(selection, ["source_type"]),
        "source_fixed6_counts": _group_count(
            selection.loc[selection["fixed6_keep"]],
            ["source_type"],
        ),
        "source_native_unit_counts": _group_count(intervals, ["source_type"]),
        "source_native_length_counts": _group_count(
            intervals,
            ["source_type", "label_frame_count"],
        ),
        "fixed6_missing_observed_slots": missing_fixed_slots,
        "native_missing_observed_slots": missing_native_slots,
        "all_input_windows_retained": len(selection) == len(windows),
        "all_native_units_in_fixed6": not missing_fixed_units,
        "all_native_units_in_native_view": set(native_units) == expected_units,
        "fixed6_observed_phase_identity_equal": identity_equal,
        "fixed6_ordered_window_id_sha256": ordered_window_id_sha256(selected_ids),
        "fixed6_slot_key_sha256": _ordered_digest(observed["slot_key"]),
        "phase_slot_key_sha256": _ordered_digest(phase["slot_key"]),
        "native_slot_key_sha256": _ordered_digest(native["slot_key"]),
        "window_alignment": alignment,
        "model_input_schema_audit": model_schema,
        "rows_dropped": 0,
        "labels_changed": 0,
        "human_review_inferred": False,
        "training_authorized": False,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return audit


def _slot_table_errors(
    slots: pd.DataFrame,
    *,
    expected_length: int | None,
) -> list[str]:
    """Validate slot uniqueness, order, masks, and declared sequence lengths."""

    errors: list[str] = []
    if slots.empty:
        return ["empty_slot_manifest"]
    blank_keys = int(slots["slot_key"].fillna("").astype(str).str.strip().eq("").sum())
    duplicate_keys = int(slots["slot_key"].duplicated(keep=False).sum())
    if blank_keys:
        errors.append(f"blank_slot_keys={blank_keys}")
    if duplicate_keys:
        errors.append(f"duplicate_slot_key_rows={duplicate_keys}")
    for item_id, group in slots.groupby("view_item_id", sort=False):
        declared = group["declared_sequence_length"].unique()
        if len(declared) != 1:
            errors.append(f"declared_length_conflict={item_id}")
            continue
        length = int(declared[0])
        if expected_length is not None and length != expected_length:
            errors.append(f"unexpected_length={item_id}:{length}")
        if len(group) != length:
            errors.append(f"slot_count_mismatch={item_id}:{len(group)}!={length}")
        expected_slots = list(range(length))
        if group["slot_index"].astype(int).tolist() != expected_slots:
            errors.append(f"slot_order_mismatch={item_id}")
    if not slots["length_mask"].map(_bool_scalar).all():
        errors.append("expected_slot_has_false_length_mask")
    if slots["padding_mask"].map(_bool_scalar).any():
        errors.append("manifest_contains_padding_slot")
    return errors


def _ordered_item_ids(slots: pd.DataFrame) -> pd.Series:
    """Return first-occurrence item order from a long slot manifest."""

    return slots.loc[~slots["view_item_id"].duplicated(), "view_item_id"].reset_index(
        drop=True
    )


def _slot_columns() -> list[str]:
    """Return a stable CSV schema shared by all temporal views."""

    return [
        "temporal_view_name",
        "temporal_view_version",
        "view_item_id",
        "parent_window_id",
        "temporal_unit_key",
        "source_type",
        "source_native_length_audit",
        "item_order",
        "slot_index",
        "slot_key",
        "declared_sequence_length",
        "frame_index_expected_audit",
        "frame_uid_audit",
        "timestamp_sec_audit",
        "observed_time_delta_sec_audit",
        "time_coordinate_kind",
        *MODEL_TENSOR_COLUMNS,
        "padding_mask",
    ]


def _parse_unit_keys(value: object) -> tuple[str, ...]:
    """Parse the unambiguous JSON unit-key list emitted by sequence windows."""

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid temporal_unit_keys_json={value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError("temporal_unit_keys_json must encode a list")
    cleaned = tuple(str(item).strip() for item in parsed)
    if any(not item for item in cleaned):
        raise ValueError("temporal_unit_keys_json contains a blank key")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("temporal_unit_keys_json contains duplicate keys")
    return cleaned


def _parse_selected_frame_indices(row: Any) -> tuple[int, ...]:
    try:
        parsed = json.loads(str(row.selected_frame_indices))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "invalid selected_frame_indices JSON for window_id="
            f"{row.window_id}"
        ) from exc
    if not isinstance(parsed, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in parsed
    ):
        raise ValueError(
            "selected_frame_indices must be an integer list for window_id="
            f"{row.window_id}"
        )
    indices = tuple(int(value) for value in parsed)
    expected_count = int(row.window_length_frames)
    if (
        len(indices) != expected_count
        or not indices
        or indices[0] != int(row.window_start_frame)
        or indices[-1] != int(row.window_end_frame)
    ):
        raise ValueError(
            "selected frame/span contract failed for window_id="
            f"{row.window_id}"
        )
    deltas = [
        current - previous
        for previous, current in zip(indices, indices[1:], strict=False)
    ]
    view_type = str(row.view_type)
    sampling_pattern = str(row.sampling_pattern)
    if view_type == "S6@16":
        valid_identity = (
            indices
            == tuple(
                int(row.window_start_frame) + value
                for value in (0, 3, 6, 9, 12, 15)
            )
            and sampling_pattern
            == "uniform_sparse_offsets_0_3_6_9_12_15"
        )
    else:
        valid_identity = (
            view_type == f"T{expected_count}_contiguous"
            and sampling_pattern == "contiguous"
            and deltas == [1] * max(0, expected_count - 1)
        )
    if not valid_identity:
        raise ValueError(
            "temporal view identity/sampling mismatch for window_id="
            f"{row.window_id}"
        )
    return indices


def _strict_integer_series(series: pd.Series, name: str) -> pd.Series:
    """Reject null, fractional, or nonnumeric alignment coordinates."""

    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() | np.mod(numeric.fillna(0), 1).ne(0)
    if invalid.any():
        raise ValueError(f"invalid integer {name} rows={int(invalid.sum())}")
    return numeric.astype(np.int64)


def _frame_bool(frame: dict[str, Any] | None, column: str) -> bool:
    """Read an inference-time mask without treating missing fields as evidence."""

    if frame is None or column not in frame:
        return False
    return _bool_scalar(frame[column])


def _social_available(frame: dict[str, Any] | None) -> bool:
    """Derive neighbor availability from label-independent relation keys."""

    if frame is None:
        return False
    if "social_neighbor_available" in frame:
        return _bool_scalar(frame["social_neighbor_available"])
    for column in ["nearest_pig_id", "nearest_track_id"]:
        value = str(frame.get(column, "")).strip()
        if value and value.lower() not in {"nan", "none", "<na>"}:
            return True
    return False


def _bool_scalar(value: object) -> bool:
    """Normalize CSV-like boolean values without Python truthiness surprises."""

    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(float(value))
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _finite_float(value: object) -> float | None:
    """Return finite timing values and mask everything else as unavailable."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    """Reject incomplete schemas before any row selection or alignment."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


def _group_count(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Serialize deterministic group counts into audit-friendly keys."""

    if frame.empty:
        return {}
    counts = frame.groupby(columns, dropna=False, sort=True).size()
    return {
        "|".join(str(part) for part in key if str(part)): int(value)
        for raw_key, value in counts.items()
        for key in [raw_key if isinstance(raw_key, tuple) else (raw_key,)]
    }


def _ordered_digest(values: pd.Series) -> str:
    """Hash ordered stable keys for deterministic repeat audits."""

    payload = "\n".join(values.fillna("").astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_key_contract(frame: pd.DataFrame, key_column: str) -> dict[str, Any]:
    """Bind each persisted manifest to row count and ordered stable keys."""

    return {
        "rows": int(len(frame)),
        "key_column": key_column,
        "ordered_key_sha256": _ordered_digest(frame[key_column]),
        "blank_key_rows": int(
            frame[key_column].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "duplicate_key_rows": int(frame[key_column].duplicated(keep=False).sum()),
    }


def _file_sha256(path: Path) -> str:
    """Hash persisted artifacts in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON so NaN cannot silently enter lineage metadata."""

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
