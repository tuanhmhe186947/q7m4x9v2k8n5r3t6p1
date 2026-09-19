"""Canonical source-frame timestamp provenance for classification_v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANONICAL_TIMESTAMP_SOURCE = "source_frame_index_over_fps"


def apply_source_frame_clock(
    frame: pd.DataFrame,
    *,
    source_fps: float,
    preserve_input_as_acquisition: bool,
) -> pd.DataFrame:
    """Bind motion time to decoded source-frame index and declared video FPS."""
    fps = _validated_fps(source_fps)
    if "frame_index" not in frame:
        raise ValueError("source-frame clock requires frame_index")

    out = frame.copy()
    source_frame = pd.to_numeric(out["frame_index"], errors="coerce")
    invalid = (
        source_frame.isna()
        | ~np.isfinite(source_frame)
        | source_frame.mod(1).ne(0)
        | source_frame.lt(0)
    )
    if invalid.any():
        raise ValueError(
            "source-frame clock requires finite nonnegative integer frames: "
            f"invalid_rows={int(invalid.sum())}"
        )

    if "source_frame_index" in out:
        declared = pd.to_numeric(out["source_frame_index"], errors="coerce")
        mismatch = declared.notna() & declared.ne(source_frame)
        if mismatch.any():
            raise ValueError(
                "source_frame_index disagrees with decoded frame_index: "
                f"rows={int(mismatch.sum())}"
            )

    if "source_fps" in out:
        declared_fps = pd.to_numeric(out["source_fps"], errors="coerce")
        mismatch = declared_fps.notna() & ~np.isclose(
            declared_fps,
            fps,
            rtol=0.0,
            atol=1e-9,
        )
        if mismatch.any():
            raise ValueError(
                "declared per-row source_fps disagrees with requested FPS: "
                f"rows={int(mismatch.sum())}"
            )

    if preserve_input_as_acquisition:
        input_timestamp = pd.to_numeric(
            out.get("timestamp_sec", pd.Series(pd.NA, index=out.index)),
            errors="coerce",
        )
        out["acquisition_timestamp_sec"] = input_timestamp
        out["acquisition_timestamp_source"] = out.get(
            "timestamp_source",
            pd.Series("input_timestamp", index=out.index),
        )
    elif "acquisition_timestamp_sec" not in out:
        out["acquisition_timestamp_sec"] = pd.NA
        out["acquisition_timestamp_source"] = "not_available"

    out["source_frame_index"] = source_frame.round().astype("Int64")
    out["source_fps"] = fps
    out["timestamp_sec"] = source_frame / fps
    out["timestamp_source"] = CANONICAL_TIMESTAMP_SOURCE
    return out


def audit_source_frame_clock(
    frame: pd.DataFrame,
    *,
    tolerance_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Fail-closed row audit for timestamp = source_frame_index / source_fps."""
    required = {
        "source_frame_index",
        "source_fps",
        "timestamp_sec",
        "timestamp_source",
    }
    missing = sorted(required.difference(frame.columns))
    errors: list[str] = []
    if missing:
        errors.append(f"missing_columns={missing}")
        return {"rows": int(len(frame)), "errors": errors}

    source_frame = pd.to_numeric(frame["source_frame_index"], errors="coerce")
    fps = pd.to_numeric(frame["source_fps"], errors="coerce")
    timestamp = pd.to_numeric(frame["timestamp_sec"], errors="coerce")
    invalid_frame = (
        source_frame.isna()
        | ~np.isfinite(source_frame)
        | source_frame.mod(1).ne(0)
        | source_frame.lt(0)
    )
    invalid_fps = fps.isna() | ~np.isfinite(fps) | fps.le(0)
    invalid_timestamp = timestamp.isna() | ~np.isfinite(timestamp)
    expected = source_frame / fps
    mismatch = ~np.isclose(
        timestamp,
        expected,
        rtol=0.0,
        atol=tolerance_seconds,
        equal_nan=False,
    )
    wrong_source = frame["timestamp_source"].astype(str).ne(
        CANONICAL_TIMESTAMP_SOURCE
    )
    if "frame_index" in frame:
        frame_index = pd.to_numeric(frame["frame_index"], errors="coerce")
        source_frame_mismatch = frame_index.ne(source_frame)
    else:
        source_frame_mismatch = pd.Series(False, index=frame.index)
    for name, mask in (
        ("invalid_source_frame_rows", invalid_frame),
        ("source_frame_index_mismatch_rows", source_frame_mismatch),
        ("invalid_source_fps_rows", invalid_fps),
        ("invalid_timestamp_rows", invalid_timestamp),
        ("timestamp_formula_mismatch_rows", mismatch),
        ("timestamp_source_mismatch_rows", wrong_source),
    ):
        count = int(mask.sum())
        if count:
            errors.append(f"{name}={count}")
    return {
        "rows": int(len(frame)),
        "source_fps_values": sorted(
            float(value) for value in fps.dropna().unique().tolist()
        ),
        "errors": errors,
    }


def _validated_fps(value: float) -> float:
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("source_fps must be finite and > 0")
    return fps
