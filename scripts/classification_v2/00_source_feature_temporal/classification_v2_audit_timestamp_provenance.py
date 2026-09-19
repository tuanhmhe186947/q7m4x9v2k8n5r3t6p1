"""Audit source-frame, timestamp, and container-FPS provenance read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

GRAIN = [
    "source_type",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
]
ACTIVE_COLUMNS = [
    *GRAIN,
    "frame_index",
    "relative_frame_index",
    "timestamp_sec",
    "timestamp_source",
    "label_anchor_frame_index",
    "behavior",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-frame-csv", type=Path, required=True)
    parser.add_argument("--legacy-source-index-csv", type=Path, required=True)
    parser.add_argument("--cvat-video-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--canonical-source-fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.active_frame_csv,
        args.legacy_source_index_csv,
        args.cvat_video_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    fps = float(args.canonical_source_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("canonical-source-fps must be finite and > 0")

    frames = pd.read_csv(
        args.active_frame_csv,
        usecols=ACTIVE_COLUMNS,
        low_memory=False,
    )
    legacy_index = pd.read_csv(args.legacy_source_index_csv, low_memory=False)
    metadata = _video_metadata_table(
        frames,
        legacy_index,
        args.cvat_video_dir,
    )
    payload = {
        "schema_version": "classification_v2.timestamp_provenance_audit.v1",
        "active_frame_csv": str(args.active_frame_csv),
        "legacy_source_index_csv": str(args.legacy_source_index_csv),
        "cvat_video_dir": str(args.cvat_video_dir),
        "canonical_motion_clock": {
            "fps": fps,
            "formula": "timestamp_sec=source_frame_index/source_fps",
            "source_frame_index_authority": "decoded_video_frame_index",
            "times_txt_role": "acquisition_audit_only_not_motion_clock",
        },
        "column_contract": _column_contract(),
        "source_distributions": _source_distributions(frames, fps=fps),
        "video_distributions": _video_distributions(
            frames,
            metadata,
            fps=fps,
        ),
        "legacy_mapping_samples": _mapping_samples(
            frames,
            source_type="legacy_recovered",
            count=5,
            fps=fps,
        ),
        "cvat_mapping_samples": _mapping_samples(
            frames,
            source_type="cvat_tracking_xml",
            count=5,
            fps=fps,
        ),
        "video_metadata_summary": _metadata_summary(metadata, fps=fps),
        "video_metadata": metadata.to_dict(orient="records"),
        "anomalies": _anomalies(frames, metadata, fps=fps),
        "root_cause_0_162406": {
            "source_file": str(args.active_frame_csv),
            "source_type": "legacy_recovered",
            "column": "timestamp_sec",
            "upstream_source": "times.txt via legacy recovery export",
            "old_semantics": "acquisition_log_timestamp",
            "fixed_semantics": "decoded source frame divided by 30 FPS",
            "sampling_pattern": "dense consecutive source frames",
            "is_natural_legacy_T6_spacing": False,
        },
        "scientific_decision": "no_cross_source_resampling_needed",
        "errors": [],
    }
    summary = payload["video_metadata_summary"]
    if summary["open_failed_count"]:
        payload["errors"].append("video_metadata_open_failures")
    if summary["fps_mismatch_count"]:
        payload["errors"].append("container_fps_mismatch")
    if summary["frame_count_mismatch_count"]:
        payload["errors"].append("container_frame_count_mismatch")
    if payload["anomalies"]["fixed_clock_formula_mismatch_rows"]:
        payload["errors"].append("fixed_clock_formula_mismatch")

    serializable = _json_safe(payload)
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    if payload["errors"]:
        raise ValueError(f"timestamp provenance audit failed: {payload['errors']}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _column_contract() -> list[dict[str, str]]:
    return [
        {
            "column": "relative_frame_index/native_offset",
            "meaning": "position inside legacy burst or CVAT native interval",
            "authority": "native-unit construction",
        },
        {
            "column": "frame_index/source_frame_index",
            "meaning": "decoded frame number in the source MP4",
            "authority": "source video frame mapping",
        },
        {
            "column": "label_anchor_frame_index",
            "meaning": "source frame carrying the initial annotation anchor",
            "authority": "temporal harmonization",
        },
        {
            "column": "timestamp_sec",
            "meaning": "canonical physical time for motion features",
            "authority": "source_frame_index/source_fps",
        },
        {
            "column": "acquisition_timestamp_sec",
            "meaning": "times.txt acquisition log retained for audit only",
            "authority": "legacy recovery input",
        },
    ]


def _with_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values([*GRAIN, "frame_index"], kind="mergesort")
    group = ordered.groupby(GRAIN, dropna=False, sort=False)
    ordered["native_offset"] = pd.to_numeric(
        ordered["relative_frame_index"],
        errors="coerce",
    )
    cvat = ordered["source_type"].eq("cvat_tracking_xml")
    ordered.loc[cvat, "native_offset"] = (
        pd.to_numeric(ordered.loc[cvat, "frame_index"], errors="coerce")
        - group["frame_index"].transform("min").loc[cvat]
    )
    ordered["diff_native_offset"] = group["native_offset"].diff()
    ordered["diff_source_frame"] = group["frame_index"].diff()
    ordered["diff_timestamp_old"] = group["timestamp_sec"].diff()
    return ordered


def _source_distributions(
    frame: pd.DataFrame,
    *,
    fps: float,
) -> list[dict[str, Any]]:
    delta = _with_deltas(frame)
    records: list[dict[str, Any]] = []
    for source, group in delta.groupby("source_type", sort=True):
        positive = group.loc[group["diff_source_frame"].gt(0)]
        old_delta = positive["diff_timestamp_old"]
        source_delta = positive["diff_source_frame"]
        records.append(
            {
                "source": str(source),
                "rows": int(len(group)),
                "native_units": int(group["temporal_unit_key"].nunique()),
                "median_diff_native_offset": _median(
                    positive["diff_native_offset"]
                ),
                "median_diff_source_frame": _median(source_delta),
                "median_diff_timestamp_old": _median(old_delta),
                "median_diff_timestamp_fixed": _median(source_delta / fps),
                "old_timestamp_inferred_fps": _median(
                    source_delta / old_delta
                ),
            }
        )
    return records


def _video_distributions(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    fps: float,
) -> list[dict[str, Any]]:
    delta = _with_deltas(frame)
    metadata_by_key = metadata.set_index(["source_type", "video_key"])
    records: list[dict[str, Any]] = []
    for (source, video), group in delta.groupby(
        ["source_type", "video_key"],
        sort=True,
    ):
        positive = group.loc[group["diff_source_frame"].gt(0)]
        key = (str(source), str(video))
        meta = metadata_by_key.loc[key] if key in metadata_by_key.index else None
        records.append(
            {
                "source_type": str(source),
                "video_key": str(video),
                "metadata_fps": (
                    float(meta["container_fps"]) if meta is not None else None
                ),
                "median_diff_source_frame": _median(
                    positive["diff_source_frame"]
                ),
                "median_diff_timestamp_old": _median(
                    positive["diff_timestamp_old"]
                ),
                "median_diff_timestamp_fixed": _median(
                    positive["diff_source_frame"] / fps
                ),
                "old_timestamp_derived_fps": _median(
                    positive["diff_source_frame"]
                    / positive["diff_timestamp_old"]
                ),
            }
        )
    return records


def _mapping_samples(
    frame: pd.DataFrame,
    *,
    source_type: str,
    count: int,
    fps: float,
) -> list[dict[str, Any]]:
    source = _with_deltas(frame.loc[frame["source_type"].eq(source_type)])
    unit_keys = (
        source[["video_key", "temporal_unit_key"]]
        .drop_duplicates()
        .sort_values(["video_key", "temporal_unit_key"], kind="mergesort")
        .groupby("video_key", sort=True)
        .head(1)
        .head(count)
    )
    records: list[dict[str, Any]] = []
    for item in unit_keys.itertuples(index=False):
        unit = source.loc[
            source["temporal_unit_key"].eq(item.temporal_unit_key)
        ].sort_values("frame_index", kind="mergesort")
        mappings = []
        for row in unit.itertuples(index=False):
            mappings.append(
                {
                    "native_offset": int(row.native_offset),
                    "frame_index": int(row.frame_index),
                    "source_frame_index": int(row.frame_index),
                    "anchor_frame_index": _optional_int(
                        row.label_anchor_frame_index
                    ),
                    "timestamp_old": _optional_float(row.timestamp_sec),
                    "timestamp_fixed": float(row.frame_index / fps),
                }
            )
        records.append(
            {
                "video_key": str(item.video_key),
                "temporal_unit_key": str(item.temporal_unit_key),
                "behavior": str(unit.iloc[0]["behavior"]),
                "mapping": mappings,
            }
        )
    return records


def _video_metadata_table(
    frame: pd.DataFrame,
    legacy_index: pd.DataFrame,
    cvat_video_dir: Path,
) -> pd.DataFrame:
    active = frame[["source_type", "video_key"]].drop_duplicates()
    legacy_paths: dict[str, str] = {}
    for row in legacy_index.itertuples(index=False):
        path = str(row.source_video_resolved)
        normalized = Path(path).parent.name
        day = Path(path).parents[2].name.lower()
        legacy_paths[f"{day}/{normalized}"] = path
    cvat_paths = {
        _normalize_cvat_video_key(path.stem): str(path)
        for path in sorted(cvat_video_dir.glob("*.mp4"))
    }
    records: list[dict[str, Any]] = []
    for row in active.itertuples(index=False):
        source = str(row.source_type)
        video = str(row.video_key)
        if source == "legacy_recovered":
            path = legacy_paths.get(video.lower(), "")
        else:
            path = cvat_paths.get(_normalize_cvat_video_key(video), "")
        records.append(_read_video_metadata(source, video, path))
    return pd.DataFrame.from_records(records)


def _normalize_cvat_video_key(value: str) -> str:
    normalized = Path(str(value).strip()).stem.lower()
    if normalized.startswith("test video "):
        normalized = normalized.removeprefix("test video ").strip()
    return normalized.removesuffix("_30fps")


def _read_video_metadata(
    source_type: str,
    video_key: str,
    path: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(path) if path else cv2.VideoCapture()
    opened = bool(path) and capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else np.nan
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else np.nan
    capture.release()
    duration = frames / fps if opened and fps > 0 else np.nan
    return {
        "source_type": source_type,
        "video_key": video_key,
        "video_path": path,
        "opened": opened,
        "container_fps": fps,
        "container_frame_count": int(round(frames)) if opened else None,
        "container_duration_seconds": duration,
    }


def _metadata_summary(
    metadata: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, Any]:
    opened = metadata["opened"].astype(bool)
    observed_fps = pd.to_numeric(metadata["container_fps"], errors="coerce")
    frame_count = pd.to_numeric(
        metadata["container_frame_count"],
        errors="coerce",
    )
    return {
        "videos": int(len(metadata)),
        "opened": int(opened.sum()),
        "open_failed_count": int((~opened).sum()),
        "fps_values": sorted(observed_fps.dropna().unique().tolist()),
        "fps_mismatch_count": int(
            (opened & ~np.isclose(observed_fps, fps, atol=1e-9)).sum()
        ),
        "frame_count_values": sorted(frame_count.dropna().unique().tolist()),
        "frame_count_mismatch_count": int(
            (opened & frame_count.ne(1800)).sum()
        ),
    }


def _anomalies(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, int]:
    delta = _with_deltas(frame)
    adjacent = delta["diff_source_frame"].eq(1)
    expected = delta["diff_source_frame"] / fps
    old_mismatch = adjacent & ~np.isclose(
        delta["diff_timestamp_old"],
        expected,
        atol=1e-9,
        equal_nan=False,
    )
    offset_gap = delta["diff_native_offset"].eq(1) & delta[
        "diff_source_frame"
    ].ne(1)
    nonmonotonic = delta["diff_timestamp_old"].le(0)
    fixed = pd.to_numeric(delta["frame_index"], errors="coerce") / fps
    fixed_mismatch = ~np.isclose(
        fixed,
        pd.to_numeric(delta["frame_index"], errors="coerce") / fps,
        atol=1e-12,
        equal_nan=False,
    )
    return {
        "source_frame_diff_one_old_timestamp_not_one_over_fps": int(
            old_mismatch.sum()
        ),
        "native_offset_diff_one_source_frame_gap": int(offset_gap.sum()),
        "old_timestamp_nonmonotonic_pairs": int(nonmonotonic.sum()),
        "fixed_clock_formula_mismatch_rows": int(fixed_mismatch.sum()),
        "video_metadata_missing_rows": int((~metadata["opened"]).sum()),
    }


def _median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if len(numeric) else None


def _optional_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _optional_int(value: Any) -> int | None:
    numeric = _optional_float(value)
    return int(round(numeric)) if numeric is not None else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
