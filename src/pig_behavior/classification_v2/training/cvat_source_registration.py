"""Apply the canonical CVAT video registration without changing observations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

AUTHORITY_SCHEMA = "classification_v2.cvat_source_registration.v1"
PATH_REALIZATION_COLUMNS = frozenset({"source_video_path", "runtime_media_path"})
REVIEW_FIELD_PATTERN = re.compile(
    r"behavior|label|review|hidden|exclude|include|harmon|corrected|lineage|"
    r"trainab|sample_weight",
    flags=re.IGNORECASE,
)


class CvatSourceRegistrationError(ValueError):
    """Raised when a CVAT scientific media key lacks one exact registration."""


def load_cvat_source_registration(path: Path) -> tuple[dict[str, str], str]:
    """Load one hash-bound, one-to-one CVAT source registration authority."""

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CvatSourceRegistrationError(f"invalid source registration={path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != AUTHORITY_SCHEMA:
        raise CvatSourceRegistrationError("unsupported CVAT source-registration schema")
    if payload.get("status") != "ACTIVE_PATH_ONLY_ENRICHMENT":
        raise CvatSourceRegistrationError("CVAT source registration is not active")
    registrations = payload.get("registrations")
    if not isinstance(registrations, list) or not registrations:
        raise CvatSourceRegistrationError("CVAT source registration has no entries")
    mapping: dict[str, str] = {}
    for entry in registrations:
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_video_key",
            "registered_relative_media_path",
            "source_provenance",
        }:
            raise CvatSourceRegistrationError("CVAT source registration entry is invalid")
        key = _text(entry["source_video_key"])
        media_path = _registered_path(entry["registered_relative_media_path"])
        provenance = entry["source_provenance"]
        if not key or not isinstance(provenance, Mapping) or not provenance:
            raise CvatSourceRegistrationError("CVAT source registration entry is incomplete")
        if key in mapping:
            raise CvatSourceRegistrationError(f"duplicate CVAT source registration={key}")
        mapping[key] = media_path
    if len(set(mapping.values())) != len(mapping):
        raise CvatSourceRegistrationError("CVAT source registration paths are not one-to-one")
    return mapping, _sha256_file(path)


def enrich_cvat_source_video_paths(
    frames: pd.DataFrame,
    *,
    registration_path: Path,
) -> tuple[pd.DataFrame, str]:
    """Return copied reviewed rows with only blank CVAT locators populated."""

    required = {"source_type", "source_video_key", "source_video_path"}
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise CvatSourceRegistrationError(f"CVAT registration fields missing={missing}")
    mapping, authority_sha256 = load_cvat_source_registration(registration_path)
    enriched = frames.copy()
    cvat = enriched["source_type"].astype(str).eq("cvat_tracking_xml")
    if not cvat.any():
        return enriched, authority_sha256
    keys = enriched.loc[cvat, "source_video_key"].map(_text)
    missing_keys = sorted(set(keys).difference(mapping))
    if missing_keys:
        raise CvatSourceRegistrationError(
            f"unregistered CVAT scientific media IDs={missing_keys[:5]}"
        )
    expected = keys.map(mapping)
    observed = enriched.loc[cvat, "source_video_path"].map(_text)
    conflicting = observed.ne("") & observed.ne(expected)
    if conflicting.any():
        raise CvatSourceRegistrationError("CVAT source-video registration conflicts")
    enriched.loc[cvat, "source_video_path"] = expected.to_numpy()
    if enriched.loc[cvat, "source_video_path"].map(_text).eq("").any():
        raise CvatSourceRegistrationError("CVAT source-video registration remains blank")
    return enriched, authority_sha256


def audit_cvat_source_path_enrichment(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict[str, Any]:
    """Prove a registration join changed only CVAT source locators."""

    required = {"image_context_id", "source_type", "source_video_key", "source_video_path"}
    for label, frame in (("before", before), ("after", after)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise CvatSourceRegistrationError(f"{label} audit fields missing={missing}")
        if frame["image_context_id"].map(_text).duplicated().any():
            raise CvatSourceRegistrationError(f"{label} image context IDs are not unique")
    if set(before.columns) != set(after.columns):
        raise CvatSourceRegistrationError("source-registration changed frame schema")

    before_rows = _sorted_records(before, columns=before.columns)
    after_rows = _sorted_records(after, columns=before.columns)
    before_cvat = before["source_type"].map(_text).eq("cvat_tracking_xml")
    after_cvat = after["source_type"].map(_text).eq("cvat_tracking_xml")
    if not before_cvat.equals(after_cvat):
        raise CvatSourceRegistrationError("source-registration changed CVAT population")

    scientific_columns = sorted(set(before.columns).difference(PATH_REALIZATION_COLUMNS))
    review_columns = sorted(
        {"image_context_id"}
        | {
            column
            for column in scientific_columns
            if REVIEW_FIELD_PATTERN.search(column)
        }
    )
    changed_columns = _changed_column_counts(before, after)
    non_path_changes = {
        column: count
        for column, count in changed_columns.items()
        if column not in PATH_REALIZATION_COLUMNS and count
    }
    if non_path_changes:
        raise CvatSourceRegistrationError(
            f"source-registration changed scientific fields={sorted(non_path_changes)}"
        )
    before_paths = before["source_video_path"].map(_text)
    after_paths = after["source_video_path"].map(_text)
    if not before_paths.loc[~before_cvat].equals(after_paths.loc[~after_cvat]):
        raise CvatSourceRegistrationError("source-registration changed legacy source paths")
    return {
        "valid": True,
        "row_count_before": len(before_rows),
        "row_count_after": len(after_rows),
        "cvat_frame_context_rows": int(before_cvat.sum()),
        "unique_cvat_source_video_keys": int(
            before.loc[before_cvat, "source_video_key"].map(_text).nunique()
        ),
        "source_video_path_blank_rows_before": int(before_paths.loc[before_cvat].eq("").sum()),
        "source_video_path_blank_rows_after": int(after_paths.loc[after_cvat].eq("").sum()),
        "scientific_projection_sha256_before": _projection_sha256(
            before, scientific_columns
        ),
        "scientific_projection_sha256_after": _projection_sha256(
            after, scientific_columns
        ),
        "review_projection_sha256_before": _projection_sha256(before, review_columns),
        "review_projection_sha256_after": _projection_sha256(after, review_columns),
        "review_row_change_count": sum(
            changed_columns.get(column, 0) for column in review_columns
        ),
        "label_change_count": sum(
            count
            for column, count in changed_columns.items()
            if "label" in column.lower() or "behavior" in column.lower()
        ),
        "trainability_change_count": sum(
            count
            for column, count in changed_columns.items()
            if "trainab" in column.lower() or "include" in column.lower()
        ),
        "exclusion_change_count": sum(
            count for column, count in changed_columns.items() if "exclude" in column.lower()
        ),
        "harmonization_change_count": sum(
            count for column, count in changed_columns.items() if "harmon" in column.lower()
        ),
        "legacy_row_count_before": int((~before_cvat).sum()),
        "legacy_row_count_after": int((~after_cvat).sum()),
        "changed_columns": changed_columns,
    }


def _registered_path(value: object) -> str:
    raw = _text(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        raw == ""
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() != ".mp4"
        or path.parts[:2] != ("data", "videos")
    ):
        raise CvatSourceRegistrationError("invalid CVAT registered media path")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_column_counts(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, int]:
    before_by_id = before.set_index("image_context_id", drop=False).sort_index()
    after_by_id = after.set_index("image_context_id", drop=False).sort_index()
    if not before_by_id.index.equals(after_by_id.index):
        raise CvatSourceRegistrationError("source-registration changed image-context IDs")
    return {
        column: sum(
            _text(left) != _text(right)
            for left, right in zip(
                before_by_id[column], after_by_id[column], strict=True
            )
        )
        for column in before.columns
    }


def _projection_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            _sorted_records(frame, columns=columns),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sorted_records(frame: pd.DataFrame, *, columns: object) -> list[dict[str, str]]:
    selected = frame.loc[:, list(columns)].copy()
    selected["image_context_id"] = selected["image_context_id"].map(_text)
    selected = selected.sort_values("image_context_id", kind="stable")
    return [
        {column: _text(value) for column, value in record.items()}
        for record in selected.to_dict(orient="records")
    ]


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()
