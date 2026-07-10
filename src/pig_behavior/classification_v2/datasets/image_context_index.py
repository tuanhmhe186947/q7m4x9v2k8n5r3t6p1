"""Image-context index artifacts for classification_v2 multimodal training.

The index is intentionally a manifest, not an extracted image cache. It records
where a trainer can load each actor crop from, plus enough context metadata to
audit whether interaction samples can be rendered with full-frame/partner
context. Labels and review decisions stay outside model input tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
IMAGE_CONTEXT_SEQUENCE_DELIMITER = ";;"

FRAME_CONTEXT_COLUMNS = [
    "frame_uid",
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_video_path",
    "object_track_key",
    "pig_id",
    "track_id",
    "frame_index",
    "temporal_unit_key",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_valid",
    "crop_path",
    "image_path",
    "frame_path",
    "interaction_partner_count",
    "interaction_partner_ids",
    "nearest_pig_id",
    "nearest_track_id",
    "nearest_dist_n",
    "social_density_near_count",
    "social_contact_count",
    "requires_partner_context",
    "review_include_in_training",
    "review_training_action",
]

WINDOW_CONTEXT_COLUMNS = [
    "window_id",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "pig_id",
    "track_id",
    "window_length_frames",
    "window_start_frame",
    "window_end_frame",
    "window_valid_for_main_train",
]


@dataclass(slots=True)
class ImageContextIndex:
    frame_manifest: pd.DataFrame
    window_manifest: pd.DataFrame
    audit: dict[str, Any]


def build_video_index(video_root: Path) -> dict[str, Path]:
    """Return aliases for videos under ``video_root``.

    Aliases cover exact filenames, stems, and common ``_30fps`` variants. Search
    is recursive so GUI/training loaders do not depend on a flat video folder.
    """
    index: dict[str, Path] = {}
    if not video_root.exists():
        return index

    def add(alias: object, path: Path) -> None:
        key = str(alias).replace("\\", "/").strip().lower()
        if not key:
            return
        index.setdefault(key, path)
        index.setdefault(Path(key).stem.lower(), path)

    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = path.stem
        add(path.name, path)
        add(stem, path)
        for suffix in ["_30fps", "-30fps", " 30fps"]:
            if stem.lower().endswith(suffix):
                base = stem[: -len(suffix)]
                add(base, path)
                add(f"{base}.mp4", path)
                add(f"{base}{path.suffix.lower()}", path)
    return index


def build_image_context_index(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    video_root: Path,
    legacy_crop_root: Path,
) -> ImageContextIndex:
    """Build frame-level and window-level image context manifests."""
    _require_columns(frames, ["source_type", "object_track_key", "frame_index"], "frames")
    _require_columns(
        windows,
        ["window_id", "object_track_key", "window_start_frame", "window_end_frame", "window_length_frames"],
        "windows",
    )

    video_index = build_video_index(video_root)
    frame_manifest = _build_frame_manifest(frames, video_root, legacy_crop_root, video_index)
    window_manifest = _build_window_manifest(windows, frame_manifest)

    loadable = _to_bool(frame_manifest["image_context_loadable"])
    audit = {
        "frame_rows": int(len(frame_manifest)),
        "window_rows": int(len(window_manifest)),
        "source_counts": frame_manifest["source_type"].value_counts(dropna=False).to_dict()
        if "source_type" in frame_manifest
        else {},
        "window_source_counts": window_manifest["source_type"].value_counts(dropna=False).to_dict()
        if "source_type" in window_manifest
        else {},
        "duplicate_frame_uid": int(frame_manifest["frame_uid"].duplicated().sum())
        if "frame_uid" in frame_manifest
        else None,
        "duplicate_image_context_id": int(frame_manifest["image_context_id"].duplicated().sum())
        if "image_context_id" in frame_manifest
        else None,
        "duplicate_window_id": int(window_manifest["window_id"].duplicated().sum())
        if "window_id" in window_manifest
        else None,
        "video_index_size": int(len(video_index)),
        "frame_loadable_count": int(loadable.sum()),
        "frame_unloadable_count": int((~loadable).sum()),
        "loadable_by_source": frame_manifest.groupby("source_type")["image_context_loadable"]
        .apply(lambda s: int(_to_bool(s).sum()))
        .to_dict()
        if "source_type" in frame_manifest
        else {},
        "unloadable_reasons": frame_manifest.loc[~loadable, "image_context_error"]
        .value_counts(dropna=False)
        .to_dict(),
        "window_image_context_complete": int(_to_bool(window_manifest["window_image_context_complete"]).sum()),
        "window_missing_context_slots": int(
            pd.to_numeric(window_manifest["missing_image_context_slots"], errors="coerce").fillna(0).sum()
        ),
        "interaction_rows_requiring_partner_context": int(
            _to_bool(frame_manifest.get("requires_partner_context", pd.Series(False, index=frame_manifest.index))).sum()
        ),
        "interaction_rows_with_partner_context": int(
            _to_bool(frame_manifest["partner_context_available"]).sum()
        ),
        "errors": [],
        "warnings": [],
    }
    if audit["duplicate_image_context_id"]:
        audit["errors"].append(f"duplicate_image_context_id={audit['duplicate_image_context_id']}")
    if audit["duplicate_window_id"]:
        audit["errors"].append(f"duplicate_window_id={audit['duplicate_window_id']}")
    if audit["duplicate_frame_uid"]:
        audit["warnings"].append(
            f"duplicate_frame_uid={audit['duplicate_frame_uid']}; using image_context_id as unique key"
        )
    if audit["frame_unloadable_count"]:
        audit["warnings"].append(f"frame_unloadable_count={audit['frame_unloadable_count']}")
    return ImageContextIndex(frame_manifest=frame_manifest, window_manifest=window_manifest, audit=audit)


def _build_frame_manifest(
    frames: pd.DataFrame,
    video_root: Path,
    legacy_crop_root: Path,
    video_index: dict[str, Path],
) -> pd.DataFrame:
    usecols = [c for c in FRAME_CONTEXT_COLUMNS if c in frames.columns]
    out = frames[usecols].copy()
    if "frame_uid" not in out.columns:
        out["frame_uid"] = [
            f"{row.source_type}|{row.object_track_key}|f{int(row.frame_index):06d}"
            for row in frames[["source_type", "object_track_key", "frame_index"]].itertuples(index=False)
        ]
    out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce")
    out["image_context_id"] = [
        f"{row.source_type}|{row.object_track_key}|f{int(row.frame_index):06d}"
        for row in out[["source_type", "object_track_key", "frame_index"]].itertuples(index=False)
    ]
    out["image_context_source"] = out["source_type"].map(_source_to_context_mode).fillna("unknown")
    out["resolved_media_path"] = ""
    out["resolved_media_exists"] = False
    out["bbox_context_valid"] = False
    out["full_frame_context_available"] = False
    out["partner_context_available"] = False
    out["image_context_loadable"] = False
    out["image_context_error"] = ""

    out["bbox_context_valid"] = _bbox_valid_frame(out)
    out["partner_context_available"] = _partner_context_available_frame(out)

    legacy_mask = out["source_type"].astype(str).eq("legacy_recovered")
    cvat_mask = out["source_type"].astype(str).eq("cvat_tracking_xml")
    unknown_mask = ~(legacy_mask | cvat_mask)

    if legacy_mask.any():
        legacy_paths = _resolve_legacy_paths_frame(out.loc[legacy_mask], legacy_crop_root)
        out.loc[legacy_mask, "resolved_media_path"] = legacy_paths.fillna("")
        legacy_exists = legacy_paths.notna()
        out.loc[legacy_mask, "resolved_media_exists"] = legacy_exists.to_numpy()
        out.loc[legacy_mask, "image_context_loadable"] = legacy_exists.to_numpy()
        out.loc[legacy_exists.index[~legacy_exists], "image_context_error"] = "missing_legacy_crop"

    if cvat_mask.any():
        cvat_paths = _resolve_video_paths_frame(out.loc[cvat_mask], video_root, video_index)
        out.loc[cvat_mask, "resolved_media_path"] = cvat_paths.fillna("")
        cvat_exists = cvat_paths.notna()
        cvat_bbox = _to_bool(out.loc[cvat_mask, "bbox_context_valid"])
        out.loc[cvat_mask, "resolved_media_exists"] = cvat_exists.to_numpy()
        out.loc[cvat_mask, "full_frame_context_available"] = cvat_exists.to_numpy()
        out.loc[cvat_mask, "image_context_loadable"] = (cvat_exists & cvat_bbox).to_numpy()
        out.loc[cvat_exists.index[~cvat_exists], "image_context_error"] = "missing_video"
        invalid_bbox_index = cvat_exists.index[cvat_exists & ~cvat_bbox]
        out.loc[invalid_bbox_index, "image_context_error"] = "invalid_bbox"

    out.loc[unknown_mask, "image_context_error"] = "unknown_source_type"

    sort_cols = [c for c in ["source_type", "video_key", "object_track_key", "frame_index", "frame_uid"] if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _build_window_manifest(windows: pd.DataFrame, frame_manifest: pd.DataFrame) -> pd.DataFrame:
    usecols = [c for c in WINDOW_CONTEXT_COLUMNS if c in windows.columns]
    out = windows[usecols].copy()
    out["window_start_frame"] = pd.to_numeric(out["window_start_frame"], errors="coerce")
    out["window_end_frame"] = pd.to_numeric(out["window_end_frame"], errors="coerce")
    out["window_length_frames"] = pd.to_numeric(out["window_length_frames"], errors="coerce")

    frame_work = frame_manifest[
        ["object_track_key", "frame_index", "frame_uid", "image_context_id", "image_context_loadable"]
    ].copy()
    frame_work["frame_index"] = pd.to_numeric(frame_work["frame_index"], errors="coerce")
    frame_lookup: dict[str, dict[int, tuple[str, str, bool]]] = {}
    for key, group in frame_work.dropna(subset=["frame_index"]).groupby("object_track_key", sort=False):
        group = group.sort_values("frame_index")
        frame_lookup[str(key)] = {
            int(record.frame_index): (
                str(record.frame_uid),
                str(record.image_context_id),
                bool(record.image_context_loadable),
            )
            for record in group.itertuples(index=False)
        }

    frame_uids: list[str] = []
    image_context_ids: list[str] = []
    frame_indices: list[str] = []
    observed_counts: list[int] = []
    loadable_counts: list[int] = []
    missing_slots: list[int] = []
    complete_flags: list[bool] = []

    for row in out.itertuples(index=False):
        start = row.window_start_frame
        end = row.window_end_frame
        object_key = str(row.object_track_key)
        if pd.isna(start) or pd.isna(end):
            wanted: list[int] = []
        else:
            wanted = list(range(int(start), int(end) + 1))
        frame_indices.append("|".join(str(v) for v in wanted))
        by_frame = frame_lookup.get(object_key, {})
        uid_values: list[str] = []
        context_id_values: list[str] = []
        observed = 0
        loadable = 0
        for frame_index in wanted:
            record = by_frame.get(frame_index)
            if record is None:
                uid_values.append("")
                context_id_values.append("")
                continue
            observed += 1
            frame_uid, image_context_id, is_loadable = record
            uid_values.append(frame_uid)
            context_id_values.append(image_context_id)
            if is_loadable:
                loadable += 1
        frame_uids.append("|".join(uid_values))
        image_context_ids.append(IMAGE_CONTEXT_SEQUENCE_DELIMITER.join(context_id_values))
        observed_counts.append(observed)
        loadable_counts.append(loadable)
        missing = len(wanted) - loadable
        missing_slots.append(missing)
        complete_flags.append(bool(wanted) and missing == 0)

    out["expected_frame_indices"] = frame_indices
    out["frame_uid_sequence"] = frame_uids
    out["image_context_id_sequence"] = image_context_ids
    out["observed_image_context_rows"] = observed_counts
    out["loadable_image_context_rows"] = loadable_counts
    out["missing_image_context_slots"] = missing_slots
    out["window_image_context_complete"] = complete_flags
    sort_cols = [c for c in ["source_type", "video_key", "object_track_key", "window_start_frame"] if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True)


def resolve_video(row: pd.Series, video_root: Path, video_index: dict[str, Path]) -> Path | None:
    source_path = row.get("source_video_path")
    if pd.notna(source_path):
        raw_path = Path(str(source_path).strip())
        for candidate in [raw_path, video_root / raw_path, video_root / raw_path.name]:
            if _is_video_file(candidate):
                return candidate
    for key in _candidate_video_keys(row):
        hit = video_index.get(key.strip().lower())
        if hit is not None:
            return hit
    return None


def _is_video_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.suffix.lower() in VIDEO_EXTS


def resolve_legacy_crop(row: pd.Series, crop_root: Path) -> Path | None:
    for col in ["crop_path", "image_path", "frame_path"]:
        value = row.get(col)
        if pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        path = Path(raw)
        for candidate in [path, crop_root / _legacy_relative_path(raw)]:
            if candidate.exists():
                return candidate
    return None


def _resolve_legacy_paths_frame(rows: pd.DataFrame, crop_root: Path) -> pd.Series:
    raw_values = pd.Series("", index=rows.index, dtype=object)
    for col in ["crop_path", "image_path", "frame_path"]:
        if col not in rows.columns:
            continue
        fill_mask = raw_values.astype(str).eq("")
        values = rows.loc[fill_mask, col]
        values = values.where(values.notna(), "").astype(str).str.strip()
        raw_values.loc[fill_mask] = values

    unique_values = sorted(v for v in raw_values.unique() if str(v).strip())
    resolved: dict[str, str | None] = {}
    for raw in unique_values:
        path = Path(raw)
        hit = None
        for candidate in [path, crop_root / _legacy_relative_path(raw)]:
            if candidate.exists():
                hit = str(candidate)
                break
        resolved[raw] = hit
    return raw_values.map(resolved)


def _resolve_video_paths_frame(rows: pd.DataFrame, video_root: Path, video_index: dict[str, Path]) -> pd.Series:
    keys = rows[[c for c in ["video_key", "source_video_key", "source_video_path"] if c in rows.columns]].copy()
    if keys.empty:
        return pd.Series(pd.NA, index=rows.index, dtype=object)
    keys = keys.fillna("").astype(str)
    tuple_keys = list(keys.itertuples(index=False, name=None))
    unique_keys = sorted(set(tuple_keys))
    resolved: dict[tuple[str, ...], str | None] = {}
    key_columns = list(keys.columns)
    for values in unique_keys:
        row = pd.Series(dict(zip(key_columns, values, strict=True)))
        hit = resolve_video(row, video_root, video_index)
        resolved[values] = str(hit) if hit else None
    return pd.Series([resolved[value] for value in tuple_keys], index=rows.index, dtype=object)


def _candidate_video_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    for col in ["video_key", "source_video_key"]:
        value = row.get(col)
        if pd.isna(value):
            continue
        raw = str(value).replace("\\", "/").strip()
        stem = Path(raw).stem
        stems = [stem]
        lower = stem.lower()
        for prefix in ["test video ", "tracking_annotation_", "tracking annotation "]:
            if lower.startswith(prefix):
                stems.append(stem[len(prefix) :])
        for candidate_stem in stems:
            keys.extend(
                [
                    raw,
                    candidate_stem,
                    f"{candidate_stem}.mp4",
                    f"{candidate_stem}_30fps",
                    f"{candidate_stem}_30fps.mp4",
                    f"{raw}.mp4",
                    f"{raw}_30fps",
                    f"{raw}_30fps.mp4",
                ]
            )
            if candidate_stem.lower().endswith("_30fps"):
                base = candidate_stem[: -len("_30fps")]
                keys.extend([base, f"{base}.mp4"])
    return keys


def _legacy_relative_path(path_text: str) -> Path:
    normalized = path_text.replace("/", "\\")
    markers = [
        "\\legacy_full_multigt_masked_nodup_16f\\crops\\",
        "\\legacy_full_multigt_masked_nodup_16f\\",
        "\\crops\\",
    ]
    for marker in markers:
        if marker in normalized:
            return Path(normalized.split(marker, 1)[1])
    return Path(Path(normalized).name)


def _source_to_context_mode(source_type: object) -> str:
    if str(source_type) == "legacy_recovered":
        return "legacy_crop"
    if str(source_type) == "cvat_tracking_xml":
        return "cvat_video_bbox"
    return "unknown"


def _bbox_valid(row: pd.Series) -> bool:
    values = [pd.to_numeric(row.get(col), errors="coerce") for col in ["x1", "y1", "x2", "y2"]]
    if any(pd.isna(v) for v in values):
        return False
    x1, y1, x2, y2 = [float(v) for v in values]
    return x2 > x1 and y2 > y1


def _bbox_valid_frame(rows: pd.DataFrame) -> pd.Series:
    values = {col: _numeric_col(rows, col) for col in ["x1", "y1", "x2", "y2"]}
    return (
        values["x1"].notna()
        & values["y1"].notna()
        & values["x2"].notna()
        & values["y2"].notna()
        & values["x2"].gt(values["x1"])
        & values["y2"].gt(values["y1"])
    )


def _partner_context_available(row: pd.Series) -> bool:
    partner_count = pd.to_numeric(row.get("interaction_partner_count"), errors="coerce")
    social_count = pd.to_numeric(row.get("social_contact_count"), errors="coerce")
    partner_ids = str(row.get("interaction_partner_ids", "")).strip()
    if pd.notna(partner_count) and partner_count > 0:
        return True
    if pd.notna(social_count) and social_count > 0:
        return True
    return partner_ids not in {"", "nan", "<NA>", "None"}


def _partner_context_available_frame(rows: pd.DataFrame) -> pd.Series:
    partner_count = _numeric_col(rows, "interaction_partner_count").fillna(0)
    social_count = _numeric_col(rows, "social_contact_count").fillna(0)
    if "interaction_partner_ids" in rows.columns:
        partner_ids = rows["interaction_partner_ids"].fillna("").astype(str).str.strip()
        has_partner_ids = ~partner_ids.isin(["", "nan", "<NA>", "None"])
    else:
        has_partner_ids = pd.Series(False, index=rows.index)
    return partner_count.gt(0) | social_count.gt(0) | has_partner_ids


def _numeric_col(rows: pd.DataFrame, col: str) -> pd.Series:
    if col not in rows.columns:
        return pd.Series(pd.NA, index=rows.index, dtype="Float64")
    return pd.to_numeric(rows[col], errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
