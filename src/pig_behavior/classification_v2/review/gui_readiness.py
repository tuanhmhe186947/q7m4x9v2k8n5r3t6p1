"""Bounded-memory readiness audit for the Behavior Review GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_context_index import (
    resolve_legacy_crop,
)

READINESS_SCHEMA_VERSION = "classification_v2.behavior_gui_readiness.v1"
UNIT_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
    "review_relevant_evidence_available",
    "review_evidence_reason_auto",
)
NATIVE_COLUMNS = (
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "frame_index",
    "crop_path",
    "bbox_valid",
    "hidden",
    "hidden_after_review",
    "hidden_review_status",
    "hidden_is_trusted",
    "hidden_trust_status",
    "hidden_source",
)


def audit_behavior_gui_readiness(
    *,
    review_units_csv: Path,
    native_evidence_csv: Path,
    pig_strenet_artifact_dir: Path,
    hidden_apply_manifest: Path,
    legacy_crop_root: Path,
    expected_hidden_reviewed_rows: int = 5233,
) -> dict[str, Any]:
    """Audit GUI identity and media using already-published pixel evidence."""

    errors: list[str] = []
    warnings: list[str] = []
    units = _read_required(review_units_csv, UNIT_COLUMNS, "review_units")
    duplicate_review_keys = int(units["review_unit_id"].duplicated().sum())
    duplicate_temporal_keys = int(units["temporal_unit_key"].duplicated().sum())
    if duplicate_review_keys:
        errors.append(f"duplicate_review_keys={duplicate_review_keys}")
    if duplicate_temporal_keys:
        errors.append(f"duplicate_temporal_keys={duplicate_temporal_keys}")

    pairs = _read_required(
        pig_strenet_artifact_dir / "pair_manifest.csv",
        (
            "pair_id",
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "history_start_frame",
            "history_end_frame",
            "target_start_frame",
            "target_end_frame",
        ),
        "pair_manifest",
    )
    pair_errors = _audit_pair_alignment(units, pairs)
    errors.extend(pair_errors)

    expected = _expected_frame_rows(units)
    native = _read_required(native_evidence_csv, NATIVE_COLUMNS, "native_evidence")
    frame_audit = _audit_native_frames(
        expected,
        native,
        legacy_crop_root=legacy_crop_root,
    )
    errors.extend(frame_audit.pop("errors"))

    slots = _read_required(
        pig_strenet_artifact_dir / "slot_manifest.csv",
        (
            "pair_id",
            "object_track_key",
            "slot_role",
            "global_slot_index",
            "frame_index",
            "frame_available",
            "frame_uid",
        ),
        "slot_manifest",
    )
    pixel_audit = _audit_published_pixels(
        pairs,
        slots,
        difference_index_path=(
            pig_strenet_artifact_dir / "difference_pixel_index.csv"
        ),
        roi_index_path=(
            pig_strenet_artifact_dir / "roi_visual_union_patch_index.csv"
        ),
    )
    errors.extend(pixel_audit.pop("errors"))

    media_manifest = json.loads(
        (pig_strenet_artifact_dir / "media_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    media_audit = _audit_media_manifest(media_manifest)
    errors.extend(media_audit.pop("errors"))

    hidden_audit = _audit_hidden_authority(
        native,
        hidden_apply_manifest,
        expected_reviewed_rows=expected_hidden_reviewed_rows,
    )
    errors.extend(hidden_audit.pop("errors"))
    unavailable = ~_bool_series(units["review_relevant_evidence_available"])
    unavailable_reasons = (
        units.loc[unavailable, "review_evidence_reason_auto"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    if int(unavailable.sum()):
        warnings.append(
            "review_relevant_evidence_unavailable="
            f"{int(unavailable.sum())}:{unavailable_reasons}"
        )

    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "review_unit_count": int(len(units)),
        "duplicate_review_keys": duplicate_review_keys,
        "duplicate_temporal_keys": duplicate_temporal_keys,
        **frame_audit,
        **pixel_audit,
        **media_audit,
        **hidden_audit,
    }


def _read_required(
    path: Path,
    columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"{label}_missing_columns={missing}")
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def _audit_pair_alignment(
    units: pd.DataFrame,
    pairs: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if pairs["temporal_unit_key"].duplicated().any():
        errors.append(
            "duplicate_pig_temporal_keys="
            f"{int(pairs['temporal_unit_key'].duplicated().sum())}"
        )
        return errors
    joined = units.merge(
        pairs,
        on="temporal_unit_key",
        how="left",
        suffixes=("_review", "_pig"),
        indicator=True,
        validate="one_to_one",
    )
    missing = int(joined["_merge"].eq("left_only").sum())
    if missing:
        errors.append(f"pig_review_unit_alignment_mismatch={missing}")
        return errors
    for column in (
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
    ):
        mismatch = _text(joined[f"{column}_review"]).ne(
            _text(joined[f"{column}_pig"])
        )
        if mismatch.any():
            errors.append(f"pig_{column}_mismatch={int(mismatch.sum())}")
    return errors


def _expected_frame_rows(units: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in units.itertuples(index=False):
        frames = _parse_frames(
            row.display_frame_indices,
            start=int(row.unit_start_frame),
            end=int(row.unit_end_frame),
        )
        for frame_index in frames:
            records.append(
                {
                    "review_unit_id": str(row.review_unit_id),
                    "temporal_unit_key": str(row.temporal_unit_key),
                    "source_type": str(row.source_type),
                    "dataset_id": str(row.dataset_id),
                    "video_key": str(row.video_key),
                    "object_track_key": str(row.object_track_key),
                    "frame_index": int(frame_index),
                }
            )
    return pd.DataFrame.from_records(records)


def _audit_native_frames(
    expected: pd.DataFrame,
    native: pd.DataFrame,
    *,
    legacy_crop_root: Path,
) -> dict[str, Any]:
    key_columns = [
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "frame_index",
    ]
    duplicate_native = int(native.duplicated(key_columns).sum())
    observed = native.merge(
        expected[key_columns],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    missing = len(expected) - len(observed)
    errors: list[str] = []
    if duplicate_native:
        errors.append(f"duplicate_native_actor_frames={duplicate_native}")
    if missing:
        errors.append(f"missing_native_actor_frames={missing}")
    invalid_bbox = int((~_bool_series(observed["bbox_valid"])).sum())
    if invalid_bbox:
        errors.append(f"invalid_actor_bbox_frames={invalid_bbox}")

    legacy = observed.loc[
        observed["source_type"].astype(str).eq("legacy_recovered")
    ]
    missing_crop = 0
    broken_crop = 0
    seen: dict[str, bool] = {}
    for row in legacy.itertuples(index=False):
        resolved = resolve_legacy_crop(
            pd.Series({"crop_path": row.crop_path}),
            legacy_crop_root,
        )
        if resolved is None or not resolved.is_file():
            missing_crop += 1
            continue
        path_text = str(resolved.resolve())
        valid = seen.get(path_text)
        if valid is None:
            valid = _image_is_readable(resolved)
            seen[path_text] = valid
        if not valid:
            broken_crop += 1
    if missing_crop:
        errors.append(f"missing_crop_media={missing_crop}")
    if broken_crop:
        errors.append(f"broken_crop_paths={broken_crop}")
    return {
        "errors": errors,
        "expected_actor_frame_rows": int(len(expected)),
        "matched_actor_frame_rows": int(len(observed)),
        "wrong_actor_media": int(missing + duplicate_native + invalid_bbox),
        "missing_crop_media": int(missing_crop),
        "broken_crop_paths": int(broken_crop),
    }


def _audit_published_pixels(
    pairs: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    difference_index_path: Path,
    roi_index_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    available_slots = slots.loc[_bool_series(slots["frame_available"])].copy()
    pair_keys = pairs.set_index("pair_id")["object_track_key"].astype(str)
    expected_keys = available_slots["pair_id"].map(pair_keys)
    wrong_slot_actor = int(
        expected_keys.ne(available_slots["object_track_key"].astype(str)).sum()
    )
    if wrong_slot_actor:
        errors.append(f"wrong_slot_actor={wrong_slot_actor}")

    difference = _read_required(
        difference_index_path,
        (
            "pair_id",
            "global_slot_index",
            "frame_uid",
            "frame_available",
            "pixel_available",
            "pixel_status",
        ),
        "difference_pixel_index",
    )
    required_actor = difference.loc[_bool_series(difference["frame_available"])]
    missing_actor = int((~_bool_series(required_actor["pixel_available"])).sum())
    actor_duplicates = int(
        required_actor.duplicated(["pair_id", "global_slot_index"]).sum()
    )
    if missing_actor:
        errors.append(f"missing_required_actor_pixels={missing_actor}")
    if actor_duplicates:
        errors.append(f"duplicate_actor_pixel_slots={actor_duplicates}")

    roi = _read_required(
        roi_index_path,
        (
            "pair_id",
            "slot_index",
            "roi_class",
            "pixel_geometry_expected",
            "pixel_available",
            "pixel_status",
        ),
        "roi_visual_union_patch_index",
    )
    required_scene = roi.loc[_bool_series(roi["pixel_geometry_expected"])]
    missing_scene = int((~_bool_series(required_scene["pixel_available"])).sum())
    scene_duplicates = int(
        required_scene.duplicated(["pair_id", "slot_index", "roi_class"]).sum()
    )
    if missing_scene:
        errors.append(f"missing_required_scene_media={missing_scene}")
    if scene_duplicates:
        errors.append(f"duplicate_scene_pixel_slots={scene_duplicates}")
    return {
        "errors": errors,
        "published_available_frame_slots": int(len(available_slots)),
        "missing_required_actor_pixels": missing_actor,
        "missing_required_scene_media": missing_scene,
        "duplicate_actor_pixel_slots": actor_duplicates,
        "duplicate_scene_pixel_slots": scene_duplicates,
        "wrong_published_actor_slots": wrong_slot_actor,
    }


def _audit_media_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    broken_paths = 0
    stat_mismatches = 0
    for source in manifest.get("sources", []):
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            broken_paths += 1
            continue
        stat = path.stat()
        if int(source.get("size", -1)) != int(stat.st_size):
            stat_mismatches += 1
        recorded_mtime = source.get("mtime_ns")
        if recorded_mtime is not None:
            if int(recorded_mtime) != int(stat.st_mtime_ns):
                stat_mismatches += 1
        if not source.get("authority_valid"):
            errors.append(f"invalid_media_authority={path}")
    if not manifest.get("valid"):
        errors.append("pig_media_manifest_invalid")
    if manifest.get("background_as_temporal_scene_used") is not False:
        errors.append("static_background_used_as_scene")
    if manifest.get("rejected_static_scene_candidates"):
        errors.append("rejected_static_scene_candidates_present")
    if broken_paths:
        errors.append(f"broken_published_media_paths={broken_paths}")
    if stat_mismatches:
        errors.append(f"published_media_stat_mismatches={stat_mismatches}")
    return {
        "errors": errors,
        "published_media_source_files": int(len(manifest.get("sources", []))),
        "broken_published_media_paths": broken_paths,
        "published_media_stat_mismatches": stat_mismatches,
    }


def _audit_hidden_authority(
    native: pd.DataFrame,
    manifest_path: Path,
    *,
    expected_reviewed_rows: int,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_transaction_state") != "COMMITTED_VALIDATED":
        errors.append("hidden_apply_manifest_not_committed_validated")
    if manifest.get("authority_state") != "CANDIDATE_VALIDATED":
        errors.append("hidden_apply_manifest_not_candidate_validated")
    reviewed = native["hidden_review_status"].fillna("").astype(str).eq("reviewed")
    current = native["hidden_source"].fillna("").astype(str).eq(
        "current_human_review"
    )
    trusted = _bool_series(native["hidden_is_trusted"])
    trust_status = native["hidden_trust_status"].fillna("").astype(str).eq(
        "trusted_current_review"
    )
    reviewed_rows = int(reviewed.sum())
    authority_mismatch = int((reviewed ^ current).sum())
    authority_mismatch += int((reviewed ^ trusted).sum())
    authority_mismatch += int((reviewed ^ trust_status).sum())
    applied = native["hidden_after_review"].fillna("").astype(str).str.strip()
    hidden = native["hidden"].fillna("").astype(str).str.strip()
    decision_mismatch = int((reviewed & applied.ne(hidden)).sum())
    if reviewed_rows != expected_reviewed_rows:
        errors.append(
            "hidden_reviewed_rows="
            f"{reviewed_rows}:expected={expected_reviewed_rows}"
        )
    if authority_mismatch:
        errors.append(f"hidden_authority_mismatch={authority_mismatch}")
    if decision_mismatch:
        errors.append(f"hidden_applied_value_mismatch={decision_mismatch}")
    return {
        "errors": errors,
        "hidden_metadata_present": True,
        "hidden_metadata_source": "VALIDATED_CURRENT_CANONICAL_LEDGER",
        "hidden_reviewed_rows": reviewed_rows,
        "hidden_authority_mismatch": authority_mismatch,
        "hidden_applied_value_mismatch": decision_mismatch,
    }


def _parse_frames(value: object, *, start: int, end: int) -> list[int]:
    text = str(value).strip()
    values: list[int] = []
    for token in text.replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(float(token)))
    return list(dict.fromkeys(values)) or list(range(start, end + 1))


def _image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"1", "true", "yes", "y"})
    )


__all__ = ["READINESS_SCHEMA_VERSION", "audit_behavior_gui_readiness"]
