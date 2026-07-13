"""Versioned scene-frame and frame-object identifier contracts.

Older classification_v2 artifacts used ``frame_uid`` for a shared video frame.
The current contract preserves that value as provenance while exposing an
explicit scene key and a globally unique actor observation key.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import pandas as pd

FRAME_OBJECT_IDENTIFIER_VERSION = "classification_v2.frame_object.v2"


def ensure_frame_object_identifiers(
    rows: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    """Return rows with global scene and object keys, preserving order and count.

    Old inputs may provide only a scene-level ``frame_uid`` or ``image_key``.
    New inputs carrying the current version are validated and retained. A
    duplicate actor observation is rejected rather than disambiguated by row
    order because that would hide an annotation or join error.
    """
    out = rows.copy()
    input_index = out.index.copy()
    input_rows = len(out)
    if out.empty:
        return _ensure_empty_identifier_columns(out)

    current_version = _clean_series(out, "identifier_schema_version").eq(
        FRAME_OBJECT_IDENTIFIER_VERSION
    )
    if current_version.all():
        _validate_current_identifiers(out, source_name=source_name)
        return out

    raw_scene = _first_nonempty_series(
        out,
        ["scene_frame_uid", "image_key", "frame_uid"],
    )
    fallback_scene = _build_fallback_scene_key(out)
    raw_scene = raw_scene.where(raw_scene.ne(""), fallback_scene)
    missing_scene = raw_scene.eq("")
    if missing_scene.any():
        raise ValueError(
            _identifier_error(
                source_name,
                "missing_scene_frame_key",
                out,
                missing_scene,
            )
        )

    source_type = _required_key_series(out, "source_type", source_name)
    dataset_id = _required_key_series(out, "dataset_id", source_name)
    scene_prefix = source_type.map(_escape_key) + "::" + dataset_id.map(_escape_key)
    out["scene_frame_uid"] = scene_prefix + "::scene=" + raw_scene.map(_escape_key)

    actor_kind, actor_value = _actor_identity(out)
    missing_actor = actor_value.eq("")
    if missing_actor.any():
        raise ValueError(
            _identifier_error(
                source_name,
                "missing_actor_key",
                out,
                missing_actor,
            )
        )

    out["frame_uid"] = (
        out["scene_frame_uid"]
        + "::"
        + actor_kind
        + "="
        + actor_value.map(_escape_key)
    )
    out["identifier_schema_version"] = FRAME_OBJECT_IDENTIFIER_VERSION
    _validate_current_identifiers(out, source_name=source_name)
    if len(out) != input_rows or not out.index.equals(input_index):
        raise RuntimeError(
            f"{source_name} identifier migration changed row count or order"
        )
    return out


def audit_frame_object_identifiers(rows: pd.DataFrame) -> dict[str, Any]:
    """Return machine-readable uniqueness and completeness evidence."""
    scene = _clean_series(rows, "scene_frame_uid")
    objects = _clean_series(rows, "frame_uid")
    versions = _clean_series(rows, "identifier_schema_version")
    missing_scene = scene.eq("")
    missing_objects = objects.eq("")
    duplicate_objects = objects.ne("") & objects.duplicated(keep=False)
    errors: list[str] = []
    if missing_scene.any():
        errors.append(f"missing_scene_frame_uid={int(missing_scene.sum())}")
    if missing_objects.any():
        errors.append(f"missing_frame_uid={int(missing_objects.sum())}")
    if duplicate_objects.any():
        errors.append(f"duplicate_frame_uid={int(duplicate_objects.sum())}")
    invalid_versions = versions.ne(FRAME_OBJECT_IDENTIFIER_VERSION)
    if invalid_versions.any():
        errors.append(f"invalid_identifier_version={int(invalid_versions.sum())}")
    return {
        "rows": int(len(rows)),
        "scene_frames": int(scene[scene.ne("")].nunique()),
        "frame_objects": int(objects[objects.ne("")].nunique()),
        "missing_scene_frame_uid": int(missing_scene.sum()),
        "missing_frame_uid": int(missing_objects.sum()),
        "duplicate_frame_uid": int(duplicate_objects.sum()),
        "identifier_versions": versions.value_counts(dropna=False).to_dict(),
        "errors": errors,
        "valid": not errors,
    }


def scene_frame_key(rows: pd.DataFrame) -> pd.Series:
    """Return the explicit scene key, with read-only fallback for old artifacts."""
    scene = _clean_series(rows, "scene_frame_uid")
    if scene.ne("").all():
        return scene
    legacy = _clean_series(rows, "frame_uid")
    return scene.where(scene.ne(""), legacy)


def _validate_current_identifiers(rows: pd.DataFrame, *, source_name: str) -> None:
    """Reject incomplete or duplicate identifiers in current-version rows."""
    scene = _clean_series(rows, "scene_frame_uid")
    objects = _clean_series(rows, "frame_uid")
    missing = scene.eq("") | objects.eq("")
    duplicate = objects.ne("") & objects.duplicated(keep=False)
    if missing.any():
        raise ValueError(
            _identifier_error(source_name, "missing_current_identifier", rows, missing)
        )
    if duplicate.any():
        raise ValueError(
            _identifier_error(source_name, "duplicate_frame_uid", rows, duplicate)
        )


def _actor_identity(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Select a deterministic inference-time actor key without reading labels."""
    kind = pd.Series("", index=rows.index, dtype=object)
    value = pd.Series("", index=rows.index, dtype=object)
    for column, token in [
        ("object_track_key", "track"),
        ("track_id", "track"),
        ("pig_id", "pig"),
        ("object_id_in_image", "object"),
    ]:
        candidate = _clean_series(rows, column)
        use = value.eq("") & candidate.ne("")
        value.loc[use] = candidate.loc[use]
        kind.loc[use] = token
    return kind, value


def _build_fallback_scene_key(rows: pd.DataFrame) -> pd.Series:
    """Build scene provenance only when no legacy scene key is available."""
    video = _clean_series(rows, "video_key")
    clip = _clean_series(rows, "clip_id")
    frame = pd.to_numeric(
        rows.get("frame_index", pd.Series(pd.NA, index=rows.index)),
        errors="coerce",
    )
    valid = video.ne("") & frame.notna() & frame.mod(1).eq(0)
    frame_text = frame.fillna(-1).astype(int).astype(str).str.zfill(6)
    built = video + "::" + clip + "::f" + frame_text
    return built.where(valid, "")


def _required_key_series(
    rows: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.Series:
    values = _clean_series(rows, column)
    missing = values.eq("")
    if missing.any():
        raise ValueError(
            _identifier_error(source_name, f"missing_{column}", rows, missing)
        )
    return values


def _first_nonempty_series(rows: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series("", index=rows.index, dtype=object)
    for column in columns:
        candidate = _clean_series(rows, column)
        use = values.eq("") & candidate.ne("")
        values.loc[use] = candidate.loc[use]
    return values


def _clean_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series("", index=rows.index, dtype=object)
    values = rows[column].fillna("").astype(str).str.strip()
    return values.mask(values.isin({"nan", "None", "<NA>"}), "")


def _escape_key(value: object) -> str:
    return quote(str(value), safe="-_.~")


def _identifier_error(
    source_name: str,
    reason: str,
    rows: pd.DataFrame,
    affected: pd.Series,
) -> str:
    sample = [str(value) for value in rows.index[affected].tolist()[:10]]
    return (
        f"{source_name} frame-object identifier contract failed: "
        f"{reason}={int(affected.sum())}, sample_source_indices={sample}"
    )


def _ensure_empty_identifier_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for column in [
        "identifier_schema_version",
        "scene_frame_uid",
        "frame_uid",
    ]:
        if column not in out.columns:
            out[column] = pd.Series(dtype=object)
    return out
