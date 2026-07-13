"""Stable, label-independent identifiers for Hidden review subjects."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import quote

import pandas as pd

HIDDEN_REVIEW_KEY_VERSION = "classification_v2.hidden_review_item.v2"


def attach_hidden_review_identifiers(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach versioned subject and item keys without changing row order.

    Hidden review identity deliberately excludes behavior, Hidden state, and
    frame UID schema. A human decision therefore remains mappable when labels
    are corrected or scene/object identifiers are upgraded.
    """

    out = rows.copy()
    input_index = out.index.copy()
    subjects = build_hidden_review_subject_keys(out)
    duplicate = subjects.duplicated(keep=False)
    if duplicate.any():
        sample = subjects.loc[duplicate].head(10).tolist()
        raise ValueError(
            "Hidden review subject key is not unique: "
            f"duplicate_rows={int(duplicate.sum())}, sample={sample}"
        )
    out["hidden_review_key_version"] = HIDDEN_REVIEW_KEY_VERSION
    out["hidden_review_subject_key"] = subjects
    out["hidden_review_item_id"] = subjects.map(_item_id)
    if len(out) != len(rows) or not out.index.equals(input_index):
        raise RuntimeError("Hidden review identifier attachment changed rows")
    return out


def build_hidden_review_subject_keys(rows: pd.DataFrame) -> pd.Series:
    """Build stable source/frame/actor locators available before inference."""

    source = _required_text(rows, "source_type")
    dataset = _required_text(rows, "dataset_id")
    video = _required_text(rows, "video_key")
    frame = _normalized_frame_index(rows)
    actor = _actor_locator(rows)
    return (
        "source="
        + source.map(_escape)
        + "|dataset="
        + dataset.map(_escape)
        + "|video="
        + video.map(_escape)
        + "|frame="
        + frame
        + "|actor="
        + actor.map(_escape)
    )


def audit_hidden_review_identifiers(rows: pd.DataFrame) -> dict[str, Any]:
    """Return completeness and uniqueness evidence for review identifiers."""

    subject = _clean(rows, "hidden_review_subject_key")
    item = _clean(rows, "hidden_review_item_id")
    version = _clean(rows, "hidden_review_key_version")
    errors: list[str] = []
    missing_subject = subject.eq("")
    missing_item = item.eq("")
    duplicate_subject = subject.ne("") & subject.duplicated(keep=False)
    duplicate_item = item.ne("") & item.duplicated(keep=False)
    invalid_version = version.ne(HIDDEN_REVIEW_KEY_VERSION)
    if missing_subject.any():
        errors.append(f"missing_hidden_review_subject_key={int(missing_subject.sum())}")
    if missing_item.any():
        errors.append(f"missing_hidden_review_item_id={int(missing_item.sum())}")
    if duplicate_subject.any():
        errors.append(
            f"duplicate_hidden_review_subject_key={int(duplicate_subject.sum())}"
        )
    if duplicate_item.any():
        errors.append(f"duplicate_hidden_review_item_id={int(duplicate_item.sum())}")
    if invalid_version.any():
        errors.append(f"invalid_hidden_review_key_version={int(invalid_version.sum())}")
    return {
        "rows": int(len(rows)),
        "unique_subject_keys": int(subject[subject.ne("")].nunique()),
        "unique_item_ids": int(item[item.ne("")].nunique()),
        "key_versions": version.value_counts(dropna=False).to_dict(),
        "errors": errors,
        "valid": not errors,
    }


def _actor_locator(rows: pd.DataFrame) -> pd.Series:
    """Prefer canonical track identity and use explicit annotation fallbacks."""

    object_track = _clean(rows, "object_track_key")
    track = _clean(rows, "track_id")
    pig = _clean(rows, "pig_id")
    object_id = _clean(rows, "object_id_in_image")
    fallback = (
        "track="
        + track.map(_escape)
        + "|pig="
        + pig.map(_escape)
        + "|object="
        + object_id.map(_escape)
    )
    actor = object_track.where(object_track.ne(""), fallback)
    missing = object_track.eq("") & track.eq("") & pig.eq("") & object_id.eq("")
    if missing.any():
        raise ValueError(
            "Hidden review subject lacks actor identity: "
            f"rows={int(missing.sum())}, sample_indices={_sample_indices(missing)}"
        )
    return actor


def _normalized_frame_index(rows: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(
        rows.get("frame_index", pd.Series(pd.NA, index=rows.index)),
        errors="coerce",
    )
    invalid = raw.isna() | raw.mod(1).ne(0)
    if invalid.any():
        raise ValueError(
            "Hidden review subject has invalid frame_index: "
            f"rows={int(invalid.sum())}, sample_indices={_sample_indices(invalid)}"
        )
    return raw.astype("int64").astype(str)


def _required_text(rows: pd.DataFrame, column: str) -> pd.Series:
    values = _clean(rows, column)
    missing = values.eq("")
    if missing.any():
        raise ValueError(
            f"Hidden review subject missing {column}: rows={int(missing.sum())}, "
            f"sample_indices={_sample_indices(missing)}"
        )
    return values


def _clean(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series("", index=rows.index, dtype=object)
    values = rows[column].fillna("").astype(str).str.strip()
    values = values.mask(values.isin({"nan", "None", "<NA>"}), "")
    return values.str.replace("\\", "/", regex=False).str.lower()


def _item_id(subject: str) -> str:
    payload = f"{HIDDEN_REVIEW_KEY_VERSION}|{subject}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"hidden_item_v2_{digest}"


def _escape(value: object) -> str:
    return quote(str(value), safe="-_.~")


def _sample_indices(mask: pd.Series) -> list[str]:
    return [str(value) for value in mask.index[mask].tolist()[:10]]
