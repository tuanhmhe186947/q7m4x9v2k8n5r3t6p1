"""Exact visual authority for Classification V2 behavior review units."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.pig_strenet_media import (
    FrameMediaResolver,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

MEDIA_AUTHORITY_SCHEMA_VERSION = (
    "classification_v2.behavior_review_media_authority.v1"
)
STOPPED_V3 = "c2v2_human_review_20260721_reviewer01_v3"


def build_behavior_review_media_authority(
    review_units: pd.DataFrame,
    native_frames: pd.DataFrame,
    *,
    video_root: Path,
    legacy_crop_root: Path,
    lineage_id: str = "",
    code_authority_sha: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve every review frame and bind exact media bytes and geometry."""

    errors: list[str] = []
    required_units = {
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "video_key",
        "unit_start_frame",
        "unit_end_frame",
    }
    required_frames = {
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "frame_index",
        "source_frame_index",
        "pig_id",
        "track_id",
        "object_track_key",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing_units = sorted(required_units.difference(review_units.columns))
    missing_frames = sorted(required_frames.difference(native_frames.columns))
    if missing_units:
        errors.append(f"missing_review_unit_columns={missing_units}")
    if missing_frames:
        errors.append(f"missing_native_frame_columns={missing_frames}")
    if errors:
        return pd.DataFrame(), _summary(
            [],
            {},
            errors,
            lineage_id=lineage_id,
            code_authority_sha=code_authority_sha,
        )

    records: list[dict[str, Any]] = []
    with FrameMediaResolver(
        video_root=video_root,
        legacy_crop_root=legacy_crop_root,
    ) as resolver:
        for unit in review_units.sort_values(
            "review_unit_id",
            kind="mergesort",
        ).itertuples(index=False):
            record, unit_errors = _resolve_unit(unit, native_frames, resolver)
            records.append(record)
            errors.extend(unit_errors)
        media_manifest = resolver.manifest()
    if not media_manifest.get("valid"):
        errors.append("resolved_media_manifest_invalid")
    if media_manifest.get("background_as_temporal_scene_used") is not False:
        errors.append("static_background_substituted_for_temporal_scene")
    rejected = media_manifest.get("rejected_static_scene_candidates", [])
    if rejected:
        errors.append(f"forbidden_static_scene_candidates={rejected}")
    index = pd.DataFrame.from_records(records)
    return index, _summary(
        records,
        media_manifest,
        errors,
        lineage_id=lineage_id,
        code_authority_sha=code_authority_sha,
    )


def finalize_media_authority_summary(
    summary: dict[str, Any],
    *,
    index_csv: Path,
    index_display_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the checked CSV bytes into the JSON consumed by authority gate."""

    out = dict(summary)
    out["index_csv"] = Path(index_display_path or index_csv).name
    out["index_csv_sha256"] = file_sha256(index_csv)
    core = {key: value for key, value in out.items() if key != "authority_sha256"}
    out["authority_sha256"] = _payload_sha256(core)
    out["valid"] = not out.get("errors")
    return out


def _resolve_unit(
    unit: Any,
    frames: pd.DataFrame,
    resolver: FrameMediaResolver,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    unit_id = str(unit.review_unit_id)
    selected = frames.loc[
        frames["temporal_unit_key"].fillna("").astype(str).eq(
            str(unit.temporal_unit_key)
        )
    ].copy()
    for column in (
        "source_type",
        "video_key",
        "object_track_key",
        "track_id",
        "pig_id",
    ):
        expected = str(getattr(unit, column, "") or "").strip()
        if expected and column in selected:
            selected = selected.loc[selected[column].astype(str).eq(expected)]
    selected = selected.sort_values("source_frame_index", kind="mergesort")
    if STOPPED_V3.casefold() in selected.to_csv(index=False).casefold():
        errors.append(f"stopped_v3_media_reference={unit_id}")
    expected_frames = _selected_frames(unit)
    declared_span = (
        int(unit.unit_start_frame),
        int(unit.unit_end_frame),
    )
    selected_span = (
        (min(expected_frames), max(expected_frames))
        if expected_frames
        else (None, None)
    )
    if selected_span != declared_span:
        errors.append(f"media_frame_span_mismatch={unit_id}:declared")
    observed_frames = pd.to_numeric(
        selected["source_frame_index"],
        errors="coerce",
    ).dropna().astype(int).tolist()
    if observed_frames != expected_frames:
        errors.append(f"media_frame_span_mismatch={unit_id}")
    if selected.empty:
        errors.append(f"media_unit_has_no_matching_actor_rows={unit_id}")

    bbox_sequence: list[list[float]] = []
    scene_records: list[dict[str, Any]] = []
    actor_records: list[dict[str, Any]] = []
    partner_available: list[bool] = []
    for _, row in selected.iterrows():
        source_index = int(row["source_frame_index"])
        frame_index = int(row["frame_index"])
        if source_index != frame_index:
            errors.append(f"media_source_frame_mapping_mismatch={unit_id}")
        bbox = [float(row[column]) for column in ("x1", "y1", "x2", "y2")]
        bbox_sequence.append(bbox)
        scene = resolver.read_scene(row)
        actor = resolver.read_actor(row, image_size=224)
        scene_record = scene.provenance()
        actor_record = actor.provenance()
        scene_records.append(_bind_media_file(scene_record))
        actor_records.append(_bind_media_file(actor_record))
        if not scene.available:
            errors.append(f"missing_scene_media={unit_id}:{source_index}:{scene.status}")
        if not actor.available:
            errors.append(f"missing_actor_media={unit_id}:{source_index}:{actor.status}")
        if scene.available and not _video_basename_matches(
            str(row["video_key"]),
            scene.media_path,
        ):
            errors.append(f"wrong_video_basename={unit_id}:{scene.media_path}")
        partner_available.append(
            bool(str(row.get("nearest_pig_id", "")).strip())
            and _truthy(row.get("social_context_valid", False))
        )
    record: dict[str, Any] = {
        "review_unit_id": unit_id,
        "temporal_unit_key": str(unit.temporal_unit_key),
        "source_type": str(unit.source_type),
        "dataset_id": str(getattr(unit, "dataset_id", "")),
        "video_key": str(unit.video_key),
        "pig_id": str(getattr(unit, "pig_id", "")),
        "track_id": str(getattr(unit, "track_id", "")),
        "object_track_key": str(getattr(unit, "object_track_key", "")),
        "frame_start": int(unit.unit_start_frame),
        "frame_end": int(unit.unit_end_frame),
        "selected_source_frames": _compact_json(observed_frames),
        "bbox_sequence": _compact_json(bbox_sequence),
        "scene_context": _compact_json(scene_records),
        "actor_context": _compact_json(actor_records),
        "partner_context_available": _compact_json(partner_available),
        "all_scene_media_available": bool(
            scene_records and all(item.get("pixel_status") == "ok"
                                  for item in scene_records)
        ),
        "all_actor_media_available": bool(
            actor_records and all(item.get("pixel_status") == "ok"
                                  for item in actor_records)
        ),
    }
    if STOPPED_V3.casefold() in _compact_json(record).casefold():
        errors.append(f"stopped_v3_media_reference={unit_id}")
    record["unit_visual_authority_digest"] = _payload_sha256(record)
    return record, errors


def _selected_frames(unit: Any) -> list[int]:
    raw = str(getattr(unit, "display_frame_indices", "") or "").strip()
    if raw:
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = [item.strip() for item in raw.split(",") if item.strip()]
        return [int(value) for value in values]
    return list(range(int(unit.unit_start_frame), int(unit.unit_end_frame) + 1))


def _bind_media_file(provenance: dict[str, Any]) -> dict[str, Any]:
    record = dict(provenance)
    raw_path = str(record.get("pixel_media_path", "")).strip()
    path = Path(raw_path) if raw_path else None
    record["file_sha256"] = (
        file_sha256(path) if path is not None and path.is_file() else None
    )
    return record


def _video_basename_matches(video_key: str, path_text: str) -> bool:
    if not path_text:
        return False
    tokens = re.findall(r"\d{6}", video_key)
    basename = Path(path_text).stem.casefold()
    if "000231" in video_key:
        return "000231" in basename
    normalized_path = str(Path(path_text)).replace("\\", "/").casefold()
    return all(token.casefold() in normalized_path for token in tokens)


def _summary(
    records: list[dict[str, Any]],
    media_manifest: dict[str, Any],
    errors: list[str],
    *,
    lineage_id: str,
    code_authority_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": MEDIA_AUTHORITY_SCHEMA_VERSION,
        "lineage_id": str(lineage_id),
        "code_authority_sha": str(code_authority_sha).lower(),
        "review_unit_count": len(records),
        "unit_visual_authority_digest_sha256": _payload_sha256(
            [record.get("unit_visual_authority_digest") for record in records]
        ),
        "resolved_media_manifest": media_manifest,
        "errors": errors,
        "valid": not errors,
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


__all__ = [
    "MEDIA_AUTHORITY_SCHEMA_VERSION",
    "build_behavior_review_media_authority",
    "finalize_media_authority_summary",
]
