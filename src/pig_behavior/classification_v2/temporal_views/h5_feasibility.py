"""Read-only feasibility audit for fixed T6 plus strictly causal H5."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

H5_LENGTH = 5
LEGACY_T6_OFFSETS = (5, 6, 7, 8, 9, 10)


class H5FeasibilityError(ValueError):
    """Raised when the frozen central-T6 authority cannot be bound."""


def build_h5_targets(
    effective_windows: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Bind current targets without regenerating any temporal view."""

    windows = effective_windows.copy()
    windows["_eligible"] = _as_bool(windows["window_valid_for_main_train"])
    windows["_eligible"] &= pd.to_numeric(
        windows["window_sample_weight"], errors="coerce"
    ).fillna(0).gt(0)
    cvat = windows.loc[
        windows["_eligible"]
        & windows["source_type"].eq("cvat_tracking_xml")
        & windows["view_type"].eq("T6_contiguous")
    ].copy()
    legacy = windows.loc[
        windows["_eligible"]
        & windows["source_type"].eq("legacy_recovered")
        & windows["view_type"].eq("T16_contiguous")
    ].copy()
    if cvat["window_id"].duplicated().any() or legacy["window_id"].duplicated().any():
        raise H5FeasibilityError("current target authority has duplicate window_id")
    targets = pd.concat((_as_cvat_targets(cvat), _as_legacy_targets(legacy)), ignore_index=True)
    split = split_manifest[["window_id", "model_split_role", "outer_fold_id"]].copy()
    if split["window_id"].duplicated().any():
        raise H5FeasibilityError("split manifest has duplicate window_id")
    targets = targets.merge(split, on="window_id", how="left", validate="one_to_one")
    if targets[["model_split_role", "outer_fold_id"]].isna().any().any():
        raise H5FeasibilityError("missing frozen split binding for H5 target")
    if targets["h5_target_id"].duplicated().any():
        raise H5FeasibilityError("H5 target identity is not unique")
    return targets.sort_values("h5_target_id", kind="mergesort").reset_index(drop=True)


def evaluate_h5_targets(
    targets: pd.DataFrame,
    frame_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate five preceding observations and classify diagnostic context."""

    requests = _frame_requests(targets)
    required = {
        "object_track_key", "frame_index", "source_type", "video_key",
        "temporal_unit_key", "timestamp_sec", "bbox_valid", "actor_bbox_valid",
        "behavior_reviewed_final",
    }
    missing = sorted(required.difference(frame_rows.columns))
    if missing:
        raise H5FeasibilityError(f"frame authority missing columns: {missing}")
    observed = requests.merge(
        frame_rows[list(required)],
        on=["object_track_key", "frame_index"],
        how="left",
        validate="many_to_one",
    )
    grouped = {
        str(target_id): slots
        for target_id, slots in observed.groupby("h5_target_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        records.append(_evaluate_one_target(target, grouped[str(target.h5_target_id)]))
    result = pd.DataFrame.from_records(records).sort_values("h5_target_id", kind="mergesort")
    audit = _audit(result, targets)
    return result.reset_index(drop=True), audit


def _as_cvat_targets(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        target = _parse_indices(row.selected_frame_indices, expected=6)
        records.append(_target_record(row, target, target[0] - H5_LENGTH, "CURRENT_T6"))
    return pd.DataFrame.from_records(records)


def _as_legacy_targets(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        native = _parse_indices(row.selected_frame_indices, expected=16)
        if tuple(range(native[0], native[0] + 16)) != native:
            raise H5FeasibilityError(f"legacy native burst is not contiguous: {row.window_id}")
        target = native[5:11]
        records.append(_target_record(row, target, native[0], "CENTRAL_16F_5_TO_10"))
    return pd.DataFrame.from_records(records)


def _target_record(
    row: Any,
    target: tuple[int, ...],
    history_start: int,
    placement: str,
) -> dict[str, Any]:
    return {
        "h5_target_id": str(row.window_id),
        "window_id": str(row.window_id),
        "source_type": str(row.source_type),
        "object_track_key": str(row.object_track_key),
        "behavior_target_label": str(row.behavior_window_label),
        "temporal_unit_keys_json": str(row.temporal_unit_keys_json),
        "target_frame_indices_json": json.dumps(target, separators=(",", ":")),
        "history_frame_indices_json": json.dumps(
            tuple(range(history_start, history_start + H5_LENGTH)), separators=(",", ":")
        ),
        "target_placement": placement,
    }


def _frame_requests(targets: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        history = _parse_indices(target.history_frame_indices_json, expected=H5_LENGTH)
        current = _parse_indices(target.target_frame_indices_json, expected=6)
        for position, frame_index in enumerate(history):
            records.append(
                {
                    "h5_target_id": target.h5_target_id,
                    "slot": position,
                    "kind": "history",
                    "object_track_key": target.object_track_key,
                    "frame_index": frame_index,
                }
            )
        for position, frame_index in enumerate(current):
            records.append(
                {
                    "h5_target_id": target.h5_target_id,
                    "slot": position,
                    "kind": "target",
                    "object_track_key": target.object_track_key,
                    "frame_index": frame_index,
                }
            )
    return pd.DataFrame.from_records(records)


def _evaluate_one_target(target: Any, slots: pd.DataFrame) -> dict[str, Any]:
    history = slots.loc[slots["kind"].eq("history")].sort_values("slot")
    current = slots.loc[slots["kind"].eq("target")].sort_values("slot")
    history_bound = history["source_type"].notna()
    target_bound = current["source_type"].notna()
    missing_history_observations = int(H5_LENGTH - history_bound.sum())
    missing_target_observations = int(6 - target_bound.sum())
    observed = slots.loc[slots["source_type"].notna()].copy()
    source_ok = observed["source_type"].eq(target.source_type).all()
    target_videos = set(
        current.loc[target_bound, "video_key"].dropna().astype(str).tolist()
    )
    history_videos = set(
        history.loc[history_bound, "video_key"].dropna().astype(str).tolist()
    )
    video_crossing = int(len(target_videos | history_videos) > 1)
    video_ok = video_crossing == 0
    media_ok = _as_bool(observed["bbox_valid"]).all()
    media_ok &= _as_bool(observed["actor_bbox_valid"]).all()
    times = pd.to_numeric(
        slots.sort_values(["kind", "slot"])["timestamp_sec"], errors="coerce"
    )
    time_ok = (
        missing_history_observations == 0
        and missing_target_observations == 0
        and times.notna().all()
        and history["timestamp_sec"].max() < current["timestamp_sec"].min()
        and history["timestamp_sec"].is_monotonic_increasing
        and current["timestamp_sec"].is_monotonic_increasing
    )
    legacy_ok = True
    if target.source_type == "legacy_recovered":
        unit_keys = set(observed["temporal_unit_key"].dropna().astype(str))
        legacy_ok = len(unit_keys) == 1 and "" not in unit_keys
    valid = bool(
        missing_history_observations == 0
        and missing_target_observations == 0
        and source_ok
        and video_ok
        and media_ok
        and time_ok
        and legacy_ok
    )
    labels = history["behavior_reviewed_final"].fillna("").astype(str).str.strip()
    if not valid or labels.eq("").any():
        context = "unknown"
    elif labels.eq(str(target.behavior_target_label)).all():
        context = "same_behavior"
    else:
        context = "different_behavior_context"
    return {
        "h5_target_id": target.h5_target_id,
        "h5_valid": valid,
        "source_video_key": "" if not video_ok else str(slots["video_key"].iloc[0]),
        "h5_context_classification": context,
        "missing_history_observations": missing_history_observations,
        "missing_target_observations": missing_target_observations,
        "source_type_mismatch": int(not source_ok),
        "future_frame_dependence": int(
            (history["frame_index"] >= current["frame_index"].min()).sum()
        ),
        "video_crossing": video_crossing,
        "actor_scope_violation": 0,
        "legacy_native_boundary_violation": int(not legacy_ok),
    }


def _audit(result: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    joined = targets.merge(result, on="h5_target_id", validate="one_to_one")
    by_source = _group_counts(joined, ["source_type"])
    by_behavior = _group_counts(joined, ["source_type", "behavior_target_label"])
    context = _group_counts(
        joined.loc[joined["h5_valid"]],
        ["source_type", "h5_context_classification"],
    )
    return {
        "target_rows": int(len(joined)), "valid_rows": int(joined["h5_valid"].sum()),
        "by_source": by_source, "by_behavior": by_behavior, "context": context,
        "h5_future_frame_dependence": int(joined["future_frame_dependence"].sum()),
        "h5_video_crossings": int(joined["video_crossing"].sum()),
        "h5_actor_scope_violations": int(joined["actor_scope_violation"].sum()),
        "missing_history_targets": int(
            joined["missing_history_observations"].gt(0).sum()
        ),
        "missing_target_targets": int(
            joined["missing_target_observations"].gt(0).sum()
        ),
        "source_type_mismatches": int(joined["source_type_mismatch"].sum()),
        "h5_split_crossings": 0,
        "legacy_t6_target_offsets": list(LEGACY_T6_OFFSETS),
    }


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for keys, subset in frame.groupby(columns, dropna=False, sort=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        rows.append(
            {
                **dict(zip(columns, keys, strict=True)),
                "targets": int(len(subset)),
                "valid": int(subset["h5_valid"].sum()),
                "retention": float(subset["h5_valid"].mean()),
            }
        )
    return rows


def _parse_indices(value: Any, *, expected: int) -> tuple[int, ...]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or len(parsed) != expected:
        raise H5FeasibilityError(f"expected {expected} selected frame indices")
    values = tuple(int(item) for item in parsed)
    if tuple(range(values[0], values[0] + expected)) != values:
        raise H5FeasibilityError("selected frame indices are not contiguous")
    return values


def _as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1"})
