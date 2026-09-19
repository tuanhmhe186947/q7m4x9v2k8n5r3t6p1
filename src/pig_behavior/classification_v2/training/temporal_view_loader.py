"""Strict fixed-length timing tensor loader for classification_v2 training."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_TEMPORAL_COLUMNS = (
    "temporal_view_name",
    "view_item_id",
    "parent_window_id",
    "item_order",
    "slot_index",
    "slot_key",
    "declared_sequence_length",
    "time_delta",
    "length_mask",
    "observed_mask",
    "timing_valid_mask",
)


@dataclass(frozen=True, slots=True)
class TemporalViewTensors:
    """Timing tensors aligned to the complete ordered training-window universe."""

    time_delta: np.ndarray
    timing_valid_mask: np.ndarray
    observed_mask: np.ndarray
    audit: dict[str, Any]


def load_temporal_view_tensors(
    path: Path,
    *,
    expected_window_ids: Sequence[object] | pd.Series,
    selected_mask: Sequence[bool] | np.ndarray | pd.Series,
    expected_view_name: str,
    expected_sequence_length: int = 6,
) -> TemporalViewTensors:
    """Load real slot timing without dropping unselected training-window rows."""

    if expected_sequence_length <= 0:
        raise ValueError("expected_sequence_length must be positive")
    window_ids = _expected_ids(expected_window_ids)
    selected = _selected_mask(selected_mask, len(window_ids))
    selected_ids = [
        window_id
        for window_id, keep in zip(window_ids, selected, strict=True)
        if keep
    ]
    frame = _read_manifest(path)
    _validate_manifest_identity(
        frame,
        selected_ids=selected_ids,
        expected_view_name=expected_view_name,
        expected_sequence_length=expected_sequence_length,
    )
    observed = _strict_bool_array(frame["observed_mask"], "observed_mask")
    timing_valid = _strict_bool_array(
        frame["timing_valid_mask"],
        "timing_valid_mask",
    )
    length = _strict_bool_array(frame["length_mask"], "length_mask")
    if not length.all():
        raise ValueError("fixed temporal manifest contains false length_mask")
    if np.any(timing_valid & ~observed):
        raise ValueError("timing_valid_mask is true outside observed_mask")

    delta = pd.to_numeric(frame["time_delta"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite = np.isfinite(delta)
    if np.any(timing_valid & ~finite):
        raise ValueError("timing-valid slots contain nonfinite time_delta")
    if np.any(~timing_valid & finite):
        raise ValueError("time_delta is present outside timing_valid_mask")
    if np.any(delta[timing_valid] < 0.0):
        raise ValueError("time_delta contains negative timing-valid values")
    _validate_item_deltas(
        delta,
        timing_valid,
        selected_ids=selected_ids,
        sequence_length=expected_sequence_length,
    )

    row_count = len(window_ids)
    output_delta = np.full(
        (row_count, expected_sequence_length),
        np.nan,
        dtype=np.float32,
    )
    output_timing = np.zeros(
        (row_count, expected_sequence_length),
        dtype=np.bool_,
    )
    output_observed = np.zeros_like(output_timing)
    selected_positions = np.flatnonzero(selected)
    if len(selected_positions):
        output_delta[selected_positions] = delta.reshape(
            len(selected_ids),
            expected_sequence_length,
        ).astype(np.float32)
        output_timing[selected_positions] = timing_valid.reshape(
            len(selected_ids),
            expected_sequence_length,
        )
        output_observed[selected_positions] = observed.reshape(
            len(selected_ids),
            expected_sequence_length,
        )

    audit = {
        "schema_version": "classification_v2.temporal_view_tensors.v1",
        "path": str(path),
        "sha256": _file_sha256(path),
        "temporal_view_name": expected_view_name,
        "sequence_length": expected_sequence_length,
        "window_universe_rows": row_count,
        "selected_window_rows": int(selected.sum()),
        "manifest_slot_rows": int(len(frame)),
        "timing_valid_slots": int(timing_valid.sum()),
        "observed_without_timing_slots": int((observed & ~timing_valid).sum()),
        "ordered_selected_window_id_sha256": _ordered_hash(selected_ids),
        "ordered_slot_key_sha256": _ordered_hash(
            frame["slot_key"].astype(str).tolist()
        ),
        "unselected_rows_preserved": int((~selected).sum()),
        "errors": [],
    }
    return TemporalViewTensors(
        time_delta=output_delta,
        timing_valid_mask=output_timing,
        observed_mask=output_observed,
        audit=audit,
    )


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"temporal view manifest not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(REQUIRED_TEMPORAL_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"temporal view manifest missing columns={missing}")
    return pd.read_csv(
        path,
        usecols=list(REQUIRED_TEMPORAL_COLUMNS),
        low_memory=False,
    )


def _validate_manifest_identity(
    frame: pd.DataFrame,
    *,
    selected_ids: list[str],
    expected_view_name: str,
    expected_sequence_length: int,
) -> None:
    expected_rows = len(selected_ids) * expected_sequence_length
    if len(frame) != expected_rows:
        raise ValueError(
            "temporal view slot row mismatch: "
            f"observed={len(frame)}, expected={expected_rows}"
        )
    if frame["parent_window_id"].isna().any():
        raise ValueError("temporal view manifest contains null parent_window_id")
    parent_ids = frame["parent_window_id"].astype(str).str.strip()
    if parent_ids.eq("").any():
        raise ValueError("temporal view manifest contains blank parent_window_id")
    expected_parents = np.repeat(selected_ids, expected_sequence_length).tolist()
    if parent_ids.tolist() != expected_parents:
        raise ValueError("temporal view parent-window order mismatch")
    view_names = frame["temporal_view_name"].fillna("").astype(str).unique()
    if view_names.tolist() != [expected_view_name]:
        raise ValueError(
            "temporal view name mismatch: "
            f"observed={view_names.tolist()}, expected={expected_view_name}"
        )
    slot_index = _strict_integer_array(frame["slot_index"], "slot_index")
    expected_slots = np.tile(
        np.arange(expected_sequence_length, dtype=np.int64),
        len(selected_ids),
    )
    if not np.array_equal(slot_index, expected_slots):
        raise ValueError("temporal view slot order mismatch")
    declared = _strict_integer_array(
        frame["declared_sequence_length"],
        "declared_sequence_length",
    )
    if np.any(declared != expected_sequence_length):
        raise ValueError("temporal view declared sequence length mismatch")
    item_order = _strict_integer_array(frame["item_order"], "item_order")
    expected_item_order = np.repeat(
        np.arange(len(selected_ids), dtype=np.int64),
        expected_sequence_length,
    )
    if not np.array_equal(item_order, expected_item_order):
        raise ValueError("temporal view item order mismatch")
    if frame["slot_key"].isna().any():
        raise ValueError("temporal view manifest contains null slot_key")
    slot_keys = frame["slot_key"].astype(str).str.strip()
    if slot_keys.eq("").any() or slot_keys.duplicated().any():
        raise ValueError("temporal view slot_key is blank or duplicated")
    view_items = frame["view_item_id"].fillna("").astype(str).str.strip()
    if view_items.eq("").any():
        raise ValueError("temporal view manifest contains blank view_item_id")
    item_counts = frame.assign(_view_item=view_items).groupby(
        "parent_window_id",
        sort=False,
    )["_view_item"].nunique()
    if not item_counts.eq(1).all():
        raise ValueError("one parent window maps to multiple view_item_id values")


def _validate_item_deltas(
    delta: np.ndarray,
    timing_valid: np.ndarray,
    *,
    selected_ids: list[str],
    sequence_length: int,
) -> None:
    for item_index, window_id in enumerate(selected_ids):
        start = item_index * sequence_length
        stop = start + sequence_length
        item_valid = timing_valid[start:stop]
        item_delta = delta[start:stop]
        valid_values = item_delta[item_valid]
        if not len(valid_values):
            continue
        if not np.isclose(valid_values[0], 0.0, atol=1e-9, rtol=0.0):
            raise ValueError(f"first timing-valid delta is not zero: {window_id}")
        if len(valid_values) > 1 and np.any(valid_values[1:] <= 0.0):
            raise ValueError(f"later timing-valid delta is not positive: {window_id}")


def _expected_ids(values: Sequence[object] | pd.Series) -> list[str]:
    series = pd.Series(list(values), dtype="object")
    if series.isna().any():
        raise ValueError("expected window IDs contain null values")
    result = series.astype(str).str.strip()
    if result.eq("").any() or result.duplicated().any():
        raise ValueError("expected window IDs are blank or duplicated")
    return result.tolist()


def _selected_mask(values: Sequence[bool] | np.ndarray | pd.Series, size: int) -> np.ndarray:
    selected = np.asarray(values)
    if selected.ndim != 1 or len(selected) != size:
        raise ValueError("temporal selection mask shape mismatch")
    if selected.dtype != np.bool_:
        raise ValueError("temporal selection mask must be boolean")
    return selected


def _strict_bool_array(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name} contains null values")
        return series.to_numpy(dtype=np.bool_)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(f"{name} contains invalid boolean values")
    return normalized.isin(true_values).to_numpy(dtype=np.bool_)


def _strict_integer_array(series: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() | np.mod(numeric.fillna(0), 1).ne(0)
    if invalid.any():
        raise ValueError(f"{name} contains invalid integer values")
    return numeric.to_numpy(dtype=np.int64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


__all__ = [
    "REQUIRED_TEMPORAL_COLUMNS",
    "TemporalViewTensors",
    "load_temporal_view_tensors",
]
