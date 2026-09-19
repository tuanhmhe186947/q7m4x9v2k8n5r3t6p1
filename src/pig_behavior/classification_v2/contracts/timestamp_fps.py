"""Hash-bound timestamp and FPS authority for Classification V2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    build_video_index,
    resolve_video,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    CANONICAL_TIMESTAMP_SOURCE,
    audit_source_frame_clock,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

TIMESTAMP_FPS_SCHEMA_VERSION = "classification_v2.timestamp_fps_contract.v1"


def build_timestamp_fps_contract(
    frame_local: pd.DataFrame,
    *,
    lineage_id: str,
    code_authority_sha: str,
    source_lineage_artifacts: Mapping[str, Path],
    video_fps_authority: Mapping[str, Mapping[str, Any]],
    tolerance_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Build a deterministic contract and fail closed on any clock ambiguity."""

    errors = list(
        audit_source_frame_clock(
            frame_local,
            tolerance_seconds=tolerance_seconds,
        )["errors"]
    )
    required = {
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "source_frame_index",
        "source_fps",
        "timestamp_sec",
        "timestamp_source",
        "acquisition_timestamp_sec",
        "acquisition_timestamp_source",
    }
    missing = sorted(required.difference(frame_local.columns))
    if missing:
        errors.append(f"missing_timestamp_contract_columns={missing}")
        return _result(
            lineage_id=lineage_id,
            code_authority_sha=code_authority_sha,
            source_lineage_artifacts=source_lineage_artifacts,
            videos=[],
            errors=errors,
        )
    source_choices = sorted(
        frame_local["timestamp_source"].fillna("").astype(str).unique().tolist()
    )
    if source_choices != [CANONICAL_TIMESTAMP_SOURCE]:
        errors.append(f"source_specific_clock_choice={source_choices}")
    if any("time" in value.casefold() and value != CANONICAL_TIMESTAMP_SOURCE
           for value in source_choices):
        errors.append("times_txt_used_as_motion_clock")

    records: list[dict[str, Any]] = []
    for (source, video), group in frame_local.groupby(
        ["source_type", "video_key"],
        dropna=False,
        sort=True,
    ):
        video_key = str(video)
        declared_fps = pd.to_numeric(group["source_fps"], errors="coerce")
        fps_values = sorted(float(value) for value in declared_fps.dropna().unique())
        authority = dict(video_fps_authority.get(video_key, {}))
        authority_fps = pd.to_numeric(authority.get("fps"), errors="coerce")
        if len(fps_values) != 1:
            errors.append(f"per_video_source_fps_not_unique={video_key}:{fps_values}")
        if not np.isfinite(authority_fps) or float(authority_fps) <= 0:
            errors.append(f"missing_video_fps_authority={video_key}")
        elif fps_values and not np.isclose(
            fps_values[0],
            float(authority_fps),
            rtol=0.0,
            atol=1e-9,
        ):
            errors.append(
                f"metadata_fps_disagreement={video_key}:"
                f"rows:{fps_values[0]},metadata:{float(authority_fps)}"
            )
        continuity = _continuity_summary(group, tolerance_seconds)
        errors.extend(f"{video_key}:{error}" for error in continuity["errors"])
        records.append(
            {
                "source_type": str(source),
                "video_key": video_key,
                "fps": fps_values[0] if len(fps_values) == 1 else None,
                "fps_authority": authority,
                **{key: value for key, value in continuity.items()
                   if key != "errors"},
            }
        )
    return _result(
        lineage_id=lineage_id,
        code_authority_sha=code_authority_sha,
        source_lineage_artifacts=source_lineage_artifacts,
        videos=records,
        errors=errors,
    )


def _continuity_summary(
    frame: pd.DataFrame,
    tolerance_seconds: float,
) -> dict[str, Any]:
    errors: list[str] = []
    frame_deltas: list[float] = []
    time_deltas: list[float] = []
    mismatch = 0
    nonmonotonic = 0
    identity = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
    ]
    for _, group in frame.groupby(identity, dropna=False, sort=True):
        ordered = group.sort_values("source_frame_index", kind="mergesort")
        source_frame = pd.to_numeric(
            ordered["source_frame_index"],
            errors="coerce",
        )
        timestamp = pd.to_numeric(ordered["timestamp_sec"], errors="coerce")
        fps = pd.to_numeric(ordered["source_fps"], errors="coerce")
        delta_frame = source_frame.diff()
        delta_time = timestamp.diff()
        pair = delta_frame.notna()
        frame_deltas.extend(delta_frame[pair].astype(float).tolist())
        time_deltas.extend(delta_time[pair].astype(float).tolist())
        nonmonotonic += int((pair & (delta_frame.le(0) | delta_time.le(0))).sum())
        adjacent = delta_frame.eq(1)
        expected = 1.0 / fps
        mismatch += int(
            (
                adjacent
                & ~np.isclose(
                    delta_time,
                    expected,
                    rtol=0.0,
                    atol=tolerance_seconds,
                )
            ).sum()
        )
    if nonmonotonic:
        errors.append(f"non_monotonic_clock_pairs={nonmonotonic}")
    if mismatch:
        errors.append(f"adjacent_frame_timestamp_mismatch_pairs={mismatch}")
    return {
        "rows": int(len(frame)),
        "source_frame_stride_summary": _numeric_summary(frame_deltas),
        "timestamp_delta_summary": _numeric_summary(time_deltas),
        "non_monotonic_clock_pairs": nonmonotonic,
        "adjacent_frame_timestamp_mismatch_pairs": mismatch,
        "errors": errors,
    }


def _result(
    *,
    lineage_id: str,
    code_authority_sha: str,
    source_lineage_artifacts: Mapping[str, Path],
    videos: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    source_hashes: dict[str, dict[str, Any]] = {}
    for name, path in sorted(source_lineage_artifacts.items()):
        exists = Path(path).is_file()
        if not exists:
            errors.append(f"missing_source_lineage_artifact={name}:{path}")
        source_hashes[name] = {
            "path": str(path),
            "sha256": file_sha256(Path(path)) if exists else None,
        }
    return {
        "schema_version": TIMESTAMP_FPS_SCHEMA_VERSION,
        "lineage_id": str(lineage_id),
        "code_authority_sha": str(code_authority_sha).lower(),
        "canonical_timestamp_formula": (
            "timestamp_sec=source_frame_index/source_fps"
        ),
        "motion_clock": CANONICAL_TIMESTAMP_SOURCE,
        "acquisition_clock_policy": (
            "audit_only_never_motion_or_pair_delta_time"
        ),
        "source_lineage_artifacts": source_hashes,
        "videos": videos,
        "invalid_or_mismatch_count": len(errors),
        "errors": errors,
        "valid": not errors,
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(clean.size),
        "min": float(clean.min()),
        "median": float(np.median(clean)),
        "max": float(clean.max()),
    }


def inspect_video_fps_authority(
    frame: pd.DataFrame,
    video_root: Path,
) -> dict[str, dict[str, object]]:
    """Read container FPS and file identity for every referenced video."""

    index = build_video_index(video_root)
    result: dict[str, dict[str, object]] = {}
    for video_key, group in frame.groupby("video_key", sort=True):
        path = resolve_video(group.iloc[0], video_root, index)
        if path is None or not path.is_file():
            result[str(video_key)] = {
                "authority": "decoded_video_container_metadata",
                "path": None,
                "fps": None,
                "sha256": None,
            }
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        result[str(video_key)] = {
            "authority": "decoded_video_container_metadata",
            "path": str(path.resolve()),
            "fps": fps,
            "frame_count": frame_count,
            "sha256": file_sha256(path),
        }
    return result


__all__ = [
    "TIMESTAMP_FPS_SCHEMA_VERSION",
    "build_timestamp_fps_contract",
    "inspect_video_fps_authority",
]
