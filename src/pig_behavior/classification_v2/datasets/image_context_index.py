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

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
    audit_frame_object_identifiers,
    ensure_frame_object_identifiers,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
IMAGE_CONTEXT_SEQUENCE_DELIMITER = ";;"

MANDATORY_CVAT_VIDEO_KEY = "Pigs291119_000231"
MANDATORY_CVAT_PIG_ID = "ID_4"
MANDATORY_CVAT_FRAME_INDICES = tuple(range(678, 684))
MANDATORY_CVAT_MEDIA_BASENAME = "Pigs291119_000231_30fps.mp4"

FRAME_CONTEXT_COLUMNS = [
    "identifier_schema_version",
    "scene_frame_uid",
    "frame_uid",
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_video_path",
    "clip_id",
    "task_id",
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
    "lineage_scope",
    "human_review_complete",
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
    "lineage_scope",
    "human_review_complete",
]


@dataclass(slots=True)
class ImageContextIndex:
    frame_manifest: pd.DataFrame
    window_manifest: pd.DataFrame
    audit: dict[str, Any]


def audit_image_context_identifier_contract(
    frame_manifest: pd.DataFrame,
    window_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Audit v2 scene/object keys while allowing explicit old-manifest reads."""
    version = (
        frame_manifest.get(
            "identifier_schema_version",
            pd.Series("", index=frame_manifest.index),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    scene_values = (
        frame_manifest.get(
            "scene_frame_uid",
            pd.Series("", index=frame_manifest.index),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    has_partial_v2 = scene_values.ne("").any() or (
        "scene_frame_uid_sequence" in window_manifest.columns
    )
    if version.eq("").all() and has_partial_v2:
        return {
            "status": "invalid_v2",
            "version": "",
            "valid": False,
            "errors": ["partial_identifier_v2_without_version"],
            "warnings": [],
        }
    if version.eq("").all():
        return {
            "status": "legacy_compatible",
            "version": "",
            "valid": True,
            "errors": [],
            "warnings": ["identifier_v2_not_present_rebuild_required"],
        }

    errors: list[str] = []
    required_frame = {
        "identifier_schema_version",
        "scene_frame_uid",
        "frame_uid",
    }
    required_window = {
        "scene_frame_uid_sequence",
        "frame_uid_sequence",
    }
    missing_frame = sorted(required_frame.difference(frame_manifest.columns))
    missing_window = sorted(required_window.difference(window_manifest.columns))
    if missing_frame:
        errors.append(f"missing_identifier_frame_columns={missing_frame}")
    if missing_window:
        errors.append(f"missing_identifier_window_columns={missing_window}")
    invalid_version = int(version.ne(FRAME_OBJECT_IDENTIFIER_VERSION).sum())
    if invalid_version:
        errors.append(f"invalid_identifier_version_rows={invalid_version}")
    identifier_audit = audit_frame_object_identifiers(frame_manifest) if not missing_frame else {}
    errors.extend(identifier_audit.get("errors", []))
    return {
        "status": "v2" if not errors else "invalid_v2",
        "version": FRAME_OBJECT_IDENTIFIER_VERSION,
        "frame_identifier_audit": identifier_audit,
        "valid": not errors,
        "errors": errors,
        "warnings": [],
    }


def audit_mandatory_cvat_video_case(frames: pd.DataFrame) -> dict[str, Any]:
    """Validate the fixed CVAT resolver regression case without loading video pixels."""
    required = {
        "video_key",
        "pig_id",
        "frame_index",
        "resolved_media_path",
        "image_context_loadable",
    }
    missing_columns = sorted(required.difference(frames.columns))
    errors: list[str] = []
    if missing_columns:
        errors.append(f"missing_columns={missing_columns}")
        return _mandatory_cvat_case_result(
            rows=0,
            observed_frames=[],
            resolved_basenames=[],
            unloadable_rows=0,
            errors=errors,
        )

    frame_index = pd.to_numeric(frames["frame_index"], errors="coerce")
    selected = frames[
        frames["video_key"].astype(str).eq(MANDATORY_CVAT_VIDEO_KEY)
        & frames["pig_id"].astype(str).eq(MANDATORY_CVAT_PIG_ID)
        & frame_index.between(
            min(MANDATORY_CVAT_FRAME_INDICES),
            max(MANDATORY_CVAT_FRAME_INDICES),
        )
    ].copy()
    selected_frame_index = pd.to_numeric(selected["frame_index"], errors="coerce")
    observed_frames = sorted(selected_frame_index.dropna().astype(int).tolist())
    expected_frames = list(MANDATORY_CVAT_FRAME_INDICES)
    loadable = _to_bool(selected["image_context_loadable"])
    unloadable_rows = int((~loadable).sum())
    resolved_basenames = sorted(
        {
            _portable_basename(value)
            for value in selected["resolved_media_path"]
            if str(value).strip()
        }
    )

    if len(selected) != len(expected_frames):
        errors.append(
            f"row_count_mismatch=expected:{len(expected_frames)},observed:{len(selected)}"
        )
    if observed_frames != expected_frames:
        errors.append(f"frame_set_mismatch=expected:{expected_frames},observed:{observed_frames}")
    if unloadable_rows:
        errors.append(f"unloadable_rows={unloadable_rows}")
    if resolved_basenames != [MANDATORY_CVAT_MEDIA_BASENAME]:
        errors.append(
            "resolved_media_basename_mismatch="
            f"expected:{MANDATORY_CVAT_MEDIA_BASENAME},observed:{resolved_basenames}"
        )
    return _mandatory_cvat_case_result(
        rows=len(selected),
        observed_frames=observed_frames,
        resolved_basenames=resolved_basenames,
        unloadable_rows=unloadable_rows,
        errors=errors,
    )


def _mandatory_cvat_case_result(
    *,
    rows: int,
    observed_frames: list[int],
    resolved_basenames: list[str],
    unloadable_rows: int,
    errors: list[str],
) -> dict[str, Any]:
    """Build the stable machine-readable result for the mandatory resolver case."""
    return {
        "video_key": MANDATORY_CVAT_VIDEO_KEY,
        "pig_id": MANDATORY_CVAT_PIG_ID,
        "expected_frame_indices": list(MANDATORY_CVAT_FRAME_INDICES),
        "expected_media_basename": MANDATORY_CVAT_MEDIA_BASENAME,
        "rows": int(rows),
        "observed_frame_indices": observed_frames,
        "resolved_media_basenames": resolved_basenames,
        "unloadable_rows": int(unloadable_rows),
        "ok": not errors,
        "errors": errors,
    }


def _portable_basename(value: object) -> str:
    """Return a basename consistently for Windows and POSIX path strings."""
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else ""


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
        [
            "window_id",
            "object_track_key",
            "window_start_frame",
            "window_end_frame",
            "window_length_frames",
        ],
        "windows",
    )
    identifier_frames = ensure_frame_object_identifiers(
        frames,
        source_name="image_context_index",
    )
    _validate_frame_context_contract(identifier_frames)
    _validate_window_context_contract(windows)
    lineage_claim = _validate_optional_lineage_claims(
        identifier_frames,
        windows,
    )

    video_index = build_video_index(video_root)
    frame_manifest = _build_frame_manifest(
        identifier_frames,
        video_root,
        legacy_crop_root,
        video_index,
    )
    window_manifest = _build_window_manifest(windows, frame_manifest)
    window_alignment = require_ordered_window_ids(
        "input_windows",
        windows["window_id"],
        {"image_context_windows": window_manifest["window_id"]},
    )

    loadable = _to_bool(frame_manifest["image_context_loadable"])
    audit = {
        "input_frame_rows": int(len(frames)),
        "frame_rows": int(len(frame_manifest)),
        "input_window_rows": int(len(windows)),
        "window_rows": int(len(window_manifest)),
        "frame_row_count_preserved": bool(len(frames) == len(frame_manifest)),
        "window_row_count_preserved": bool(len(windows) == len(window_manifest)),
        "window_order_preserved": True,
        "window_alignment": window_alignment,
        "invalid_frame_alignment_rows": 0,
        "invalid_window_alignment_rows": 0,
        "duplicate_frame_alignment_rows": 0,
        "source_counts": frame_manifest["source_type"].value_counts(dropna=False).to_dict()
        if "source_type" in frame_manifest
        else {},
        "image_context_source_counts": frame_manifest["image_context_source"]
        .value_counts(dropna=False)
        .to_dict(),
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
        "window_image_context_complete": int(
            _to_bool(window_manifest["window_image_context_complete"]).sum()
        ),
        "window_missing_context_slots": int(
            pd.to_numeric(
                window_manifest["missing_image_context_slots"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "interaction_rows_requiring_partner_context": int(
            _to_bool(
                frame_manifest.get(
                    "requires_partner_context",
                    pd.Series(False, index=frame_manifest.index),
                )
            ).sum()
        ),
        "interaction_rows_with_partner_context": int(
            _to_bool(frame_manifest["partner_context_available"]).sum()
        ),
        "errors": [],
        "warnings": [],
        **lineage_claim,
    }
    if audit["duplicate_image_context_id"]:
        audit["errors"].append(f"duplicate_image_context_id={audit['duplicate_image_context_id']}")
    if audit["duplicate_window_id"]:
        audit["errors"].append(f"duplicate_window_id={audit['duplicate_window_id']}")
    if not audit["window_order_preserved"]:
        audit["errors"].append("window_order_not_preserved")
    if audit["duplicate_frame_uid"]:
        audit["errors"].append(f"duplicate_frame_uid={audit['duplicate_frame_uid']}")
    if audit["frame_unloadable_count"]:
        audit["warnings"].append(f"frame_unloadable_count={audit['frame_unloadable_count']}")
    return ImageContextIndex(
        frame_manifest=frame_manifest,
        window_manifest=window_manifest,
        audit=audit,
    )


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
            for row in frames[["source_type", "object_track_key", "frame_index"]].itertuples(
                index=False
            )
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
        fallback_index = legacy_exists.index[~legacy_exists]
        if len(fallback_index):
            video_paths = _resolve_video_paths_frame(
                out.loc[fallback_index],
                video_root,
                video_index,
            )
            video_exists = video_paths.notna()
            video_bbox = _to_bool(out.loc[fallback_index, "bbox_context_valid"])
            video_loadable = video_exists & video_bbox
            out.loc[fallback_index, "resolved_media_path"] = video_paths.fillna("")
            out.loc[fallback_index, "resolved_media_exists"] = video_exists.to_numpy()
            out.loc[fallback_index, "full_frame_context_available"] = (
                video_exists.to_numpy()
            )
            out.loc[fallback_index, "image_context_loadable"] = (
                video_loadable.to_numpy()
            )
            video_index_rows = video_exists.index[video_exists]
            out.loc[video_index_rows, "image_context_source"] = "legacy_video_bbox"
            missing_video_rows = video_exists.index[~video_exists]
            out.loc[missing_video_rows, "image_context_error"] = (
                "missing_legacy_crop_and_video"
            )
            invalid_bbox_rows = video_exists.index[video_exists & ~video_bbox]
            out.loc[invalid_bbox_rows, "image_context_error"] = "invalid_bbox"

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

    sort_cols = [
        column
        for column in [
            "source_type",
            "video_key",
            "scene_frame_uid",
            "object_track_key",
            "frame_index",
            "frame_uid",
        ]
        if column in out
    ]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _build_window_manifest(windows: pd.DataFrame, frame_manifest: pd.DataFrame) -> pd.DataFrame:
    usecols = [c for c in WINDOW_CONTEXT_COLUMNS if c in windows.columns]
    out = windows[usecols].copy()
    out["window_start_frame"] = pd.to_numeric(out["window_start_frame"], errors="coerce")
    out["window_end_frame"] = pd.to_numeric(out["window_end_frame"], errors="coerce")
    out["window_length_frames"] = pd.to_numeric(out["window_length_frames"], errors="coerce")

    frame_work = frame_manifest[
        [
            "object_track_key",
            "frame_index",
            "scene_frame_uid",
            "frame_uid",
            "image_context_id",
            "image_context_loadable",
        ]
    ].copy()
    frame_work["frame_index"] = pd.to_numeric(frame_work["frame_index"], errors="coerce")
    frame_lookup: dict[str, dict[int, tuple[str, str, str, bool]]] = {}
    for key, group in frame_work.groupby("object_track_key", sort=False):
        group = group.sort_values("frame_index")
        frame_lookup[str(key)] = {
            int(record.frame_index): (
                str(record.scene_frame_uid),
                str(record.frame_uid),
                str(record.image_context_id),
                bool(record.image_context_loadable),
            )
            for record in group.itertuples(index=False)
        }

    scene_frame_uids: list[str] = []
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
        scene_uid_values: list[str] = []
        uid_values: list[str] = []
        context_id_values: list[str] = []
        observed = 0
        loadable = 0
        for frame_index in wanted:
            record = by_frame.get(frame_index)
            if record is None:
                scene_uid_values.append("")
                uid_values.append("")
                context_id_values.append("")
                continue
            observed += 1
            scene_frame_uid, frame_uid, image_context_id, is_loadable = record
            scene_uid_values.append(scene_frame_uid)
            uid_values.append(frame_uid)
            context_id_values.append(image_context_id)
            if is_loadable:
                loadable += 1
        scene_frame_uids.append("|".join(scene_uid_values))
        frame_uids.append("|".join(uid_values))
        image_context_ids.append(IMAGE_CONTEXT_SEQUENCE_DELIMITER.join(context_id_values))
        observed_counts.append(observed)
        loadable_counts.append(loadable)
        missing = len(wanted) - loadable
        missing_slots.append(missing)
        complete_flags.append(bool(wanted) and missing == 0)

    out["expected_frame_indices"] = frame_indices
    out["scene_frame_uid_sequence"] = scene_frame_uids
    out["frame_uid_sequence"] = frame_uids
    out["image_context_id_sequence"] = image_context_ids
    out["observed_image_context_rows"] = observed_counts
    out["loadable_image_context_rows"] = loadable_counts
    out["missing_image_context_slots"] = missing_slots
    out["window_image_context_complete"] = complete_flags
    # Positional X/y/spatial arrays use the source window-manifest row order.
    # Re-sorting here silently pairs actor images with another window's target.
    return out.reset_index(drop=True)


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


def _resolve_video_paths_frame(
    rows: pd.DataFrame,
    video_root: Path,
    video_index: dict[str, Path],
) -> pd.Series:
    keys = rows[
        [
            column
            for column in [
                "video_key",
                "source_video_key",
                "source_video_path",
            ]
            if column in rows.columns
        ]
    ].copy()
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
        "\\legacy_16f_rebuild\\legacy_16f_rebuild_20260718_v2\\06_full_recovery\\crops\\",
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


def _strict_bool(series: pd.Series) -> tuple[pd.Series, int]:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool), int(series.isna().sum())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    invalid = int((~normalized.isin(truthy | falsy)).sum())
    return normalized.isin(truthy), invalid


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _validate_frame_context_contract(frames: pd.DataFrame) -> None:
    """Reject frame rows that would be lost or overwritten in context lookup."""
    key = frames["object_track_key"].fillna("").astype(str).str.strip()
    frame_index = pd.to_numeric(frames["frame_index"], errors="coerce")
    invalid = key.eq("") | frame_index.isna() | frame_index.mod(1).ne(0) | frame_index.lt(0)
    duplicate = pd.DataFrame(
        {
            "object_track_key": key,
            "frame_index": frame_index,
        }
    ).duplicated(keep=False)
    duplicate &= ~invalid
    if invalid.any() or duplicate.any():
        _raise_context_alignment_error(
            "Frame",
            frames,
            invalid,
            duplicate,
            duplicate_name="duplicate_frame_alignment_rows",
        )


def _validate_window_context_contract(windows: pd.DataFrame) -> None:
    """Reject malformed windows before frame-sequence context alignment."""
    key = windows["object_track_key"].fillna("").astype(str).str.strip()
    window_id = windows["window_id"].fillna("").astype(str).str.strip()
    start = pd.to_numeric(windows["window_start_frame"], errors="coerce")
    end = pd.to_numeric(windows["window_end_frame"], errors="coerce")
    length = pd.to_numeric(windows["window_length_frames"], errors="coerce")
    integer_fields = (
        start.notna()
        & end.notna()
        & length.notna()
        & start.mod(1).eq(0)
        & end.mod(1).eq(0)
        & length.mod(1).eq(0)
    )
    span_valid = start.ge(0) & end.ge(start) & length.eq(end - start + 1)
    invalid = key.eq("") | window_id.eq("") | ~integer_fields | ~span_valid
    duplicate = window_id.ne("") & window_id.duplicated(keep=False)
    if invalid.any() or duplicate.any():
        _raise_context_alignment_error(
            "Window",
            windows,
            invalid,
            duplicate,
            duplicate_name="duplicate_window_id_rows",
        )


def _validate_optional_lineage_claims(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    """Preserve an explicit profile claim only when both manifests agree."""

    claim_columns = {"lineage_scope", "human_review_complete"}
    frame_present = claim_columns.intersection(frames.columns)
    window_present = claim_columns.intersection(windows.columns)
    if not frame_present and not window_present:
        return {}
    if frame_present != claim_columns or window_present != claim_columns:
        raise ValueError(
            "image context lineage claim requires both columns on frames "
            "and windows"
        )
    frame_scopes = set(frames["lineage_scope"].fillna("").astype(str))
    window_scopes = set(windows["lineage_scope"].fillna("").astype(str))
    frame_reviewed, frame_invalid = _strict_bool(
        frames["human_review_complete"]
    )
    window_reviewed, window_invalid = _strict_bool(
        windows["human_review_complete"]
    )
    if (
        len(frame_scopes) != 1
        or "" in frame_scopes
        or frame_scopes != window_scopes
    ):
        raise ValueError(
            "image context lineage_scope mismatch: "
            f"frames={sorted(frame_scopes)} windows={sorted(window_scopes)}"
        )
    frame_values = set(frame_reviewed.astype(bool))
    window_values = set(window_reviewed.astype(bool))
    if (
        frame_invalid
        or window_invalid
        or len(frame_values) != 1
        or frame_values != window_values
    ):
        raise ValueError(
            "image context human_review_complete mismatch: "
            f"frame_invalid={frame_invalid} window_invalid={window_invalid}"
        )
    return {
        "lineage_scope": next(iter(frame_scopes)),
        "human_review_complete": next(iter(frame_values)),
    }


def _raise_context_alignment_error(
    kind: str,
    rows: pd.DataFrame,
    invalid: pd.Series,
    duplicate: pd.Series,
    *,
    duplicate_name: str,
) -> None:
    """Raise an evidence-rich context alignment error without writing output."""
    affected = invalid | duplicate
    sample = [str(value) for value in rows.index[affected].tolist()[:10]]
    raise ValueError(
        f"{kind} image-context contract failed: "
        f"invalid_rows={int(invalid.sum())}, "
        f"{duplicate_name}={int(duplicate.sum())}, "
        f"sample_source_indices={sample}"
    )
