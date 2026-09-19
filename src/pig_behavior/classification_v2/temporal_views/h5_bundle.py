"""Create exporter-ready H5 then fixed-T6 windows from a frozen cohort."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

H5_VIEW_TYPE = "T6_TARGET_PLUS_H5"
H5_SAMPLING_PATTERN = "causal_history_5_then_target_6"


class H5BundleError(ValueError):
    """Raised when a matched H5 target cannot form one causal input window."""


def build_h5_window_manifest(
    cohort: pd.DataFrame,
    frame_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return 11-slot H5-to-T6 windows with target labels excluded.

    ``cohort`` is the already-audited matched target authority.  This function
    only constructs exporter metadata and validates timestamps; it does not
    alter the cohort, labels, source authority, or split membership.
    """

    _require_columns(
        cohort,
        {
            "h5_target_id",
            "object_track_key",
            "history_frame_indices_json",
            "target_frame_indices_json",
            "h5_valid",
        },
        name="cohort",
    )
    _require_columns(
        frame_rows,
        {"object_track_key", "frame_index", "timestamp_sec"},
        name="frame_rows",
    )
    selected = cohort.loc[_as_bool(cohort["h5_valid"])].copy()
    if len(selected) != len(cohort):
        raise H5BundleError("H5 bundle input includes invalid matched targets")
    if selected["h5_target_id"].duplicated().any():
        raise H5BundleError("H5 matched cohort has duplicate target identifiers")
    timing = frame_rows[
        ["object_track_key", "frame_index", "timestamp_sec"]
    ].copy()
    timing["frame_index"] = pd.to_numeric(
        timing["frame_index"], errors="raise"
    ).astype("int64")
    if timing.duplicated(["object_track_key", "frame_index"]).any():
        raise H5BundleError("frame timing authority has duplicate actor-frame keys")
    output: list[dict[str, Any]] = []
    for target in selected.itertuples(index=False):
        history = _indices(target.history_frame_indices_json, expected=5)
        current = _indices(target.target_frame_indices_json, expected=6)
        all_indices = history + current
        if all_indices != tuple(range(all_indices[0], all_indices[0] + 11)):
            raise H5BundleError(
                f"H5 target is not one contiguous causal sequence: {target.h5_target_id}"
            )
        timestamps = _timestamps(
            timing,
            object_track_key=str(target.object_track_key),
            indices=all_indices,
        )
        if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
            raise H5BundleError(
                f"H5 target has non-monotonic source time: {target.h5_target_id}"
            )
        output.append(
            {
                "window_id": str(target.h5_target_id),
                "object_track_key": str(target.object_track_key),
                "window_start_frame": all_indices[0],
                "window_end_frame": all_indices[-1],
                "window_length_frames": 11,
                "feature_computation_grain": "FINAL_VIEW_FEATURES",
                "pair_scope_key": str(target.h5_target_id),
                "view_type": H5_VIEW_TYPE,
                "sampling_pattern": H5_SAMPLING_PATTERN,
                "history_length": 5,
                "target_length": 6,
                "selected_frame_offsets": _compact_json(list(range(11))),
                "selected_frame_indices": _compact_json(all_indices),
                "selected_timestamps_seconds": _compact_json(timestamps.tolist()),
                "pair_delta_frames": _compact_json([1] * 10),
                "pair_delta_seconds": _compact_json(np.diff(timestamps).tolist()),
                "pair_recomputed_for_view": True,
                "aggregate_recomputed_for_view": True,
            }
        )
    result = pd.DataFrame.from_records(output)
    if len(result) != len(selected) or result["window_id"].duplicated().any():
        raise H5BundleError("H5 bundle window identity is not one-to-one")
    return result.sort_values("window_id", kind="mergesort").reset_index(drop=True)


def _timestamps(
    timing: pd.DataFrame,
    *,
    object_track_key: str,
    indices: tuple[int, ...],
) -> np.ndarray:
    requested = pd.DataFrame(
        {"object_track_key": object_track_key, "frame_index": list(indices)}
    )
    joined = requested.merge(
        timing,
        on=["object_track_key", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    if joined["timestamp_sec"].isna().any():
        raise H5BundleError(
            f"H5 target has missing source timestamps: {object_track_key}"
        )
    return pd.to_numeric(joined["timestamp_sec"], errors="raise").to_numpy(
        dtype="float64"
    )


def _indices(value: Any, *, expected: int) -> tuple[int, ...]:
    raw = json.loads(str(value))
    if not isinstance(raw, list) or len(raw) != expected:
        raise H5BundleError(f"expected exactly {expected} H5 indices")
    return tuple(int(item) for item in raw)


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _require_columns(
    frame: pd.DataFrame,
    expected: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise H5BundleError(f"{name} is missing required columns: {missing}")


def _as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1"})


__all__ = [
    "H5_SAMPLING_PATTERN",
    "H5_VIEW_TYPE",
    "H5BundleError",
    "build_h5_window_manifest",
]
