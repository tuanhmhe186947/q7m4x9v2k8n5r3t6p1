"""Deterministic date-grouped split authority (metadata tables only).

Outer protocol
--------------
``FOUR_FOLD_DATE_GROUPED_OUTER_CV`` — deliberately **not** full
leave-one-date-out. Three dates hold most of the data and eleven small legacy
tokens are pooled:

======  ==========================================
FOLD_1  ``291119``
FOLD_2  ``301119``
FOLD_3  ``281119``
FOLD_4  every remaining (small legacy) calendar date
======  ==========================================

Calendar-date binding
---------------------
* ``101219a`` and ``101219b`` share calendar group ``101219``;
* all CVAT and legacy samples from ``281119`` share one outer group;
* all CVAT and legacy samples from ``291119`` share one outer group;
* no calendar date may appear in two outer folds.

Inner validation is group-safe on recording/video/burst authority. Assignment
uses a stable SHA-256 of the group id and the seed, so the result depends on the
metadata, seed and configuration — never on the Git SHA or row order.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

SPLIT_SCHEMA_VERSION = "classification_v2.date_grouped_split.v1"

OUTER_PROTOCOL_NAME = "FOUR_FOLD_DATE_GROUPED_OUTER_CV"

#: Dates that carry their own outer fold; everything else pools into FOLD_4.
OUTER_FOLD_DATE_TOKENS: dict[str, str] = {
    "291119": "FOLD_1",
    "301119": "FOLD_2",
    "281119": "FOLD_3",
}
POOLED_SMALL_DATE_FOLD = "FOLD_4"
OUTER_FOLD_NAMES: tuple[str, ...] = ("FOLD_1", "FOLD_2", "FOLD_3", "FOLD_4")

#: 14 date tokens collapse to 13 calendar groups once 101219a/b are bound.
CALENDAR_GROUP_COUNT_EXPECTED = 13

REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
    "native_unit_id",
    "source_type",
    "dataset_id",
    "video_key",
)

#: Columns produced for the trainer. None of them may ever enter model X.
SPLIT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "outer_date_group_id",
    "session_group_id",
    "inner_recording_group_id",
    "outer_fold_id",
    "inner_fold_id",
)

_DATE_TOKEN_RE = re.compile(r"pigs?[_-]?([0-3]\d[01]\d\d{2})([a-z]?)", re.IGNORECASE)
_BARE_DATE_RE = re.compile(r"(?<!\d)([0-3]\d[01]\d\d{2})([a-z]?)(?!\d)")
_CLIP_RE = re.compile(r"(?:^|[_\\/])(\d{6})(?:[_\\/]|$)")


class SplitAuthorityError(ValueError):
    """Raised when metadata cannot support a leakage-safe split."""


@dataclass(frozen=True, slots=True)
class SplitAuthorityConfig:
    """Validated split configuration."""

    seed: int = 20260726
    inner_folds: int = 5
    protocol: str = OUTER_PROTOCOL_NAME
    prohibit_object_track_key_spanning: bool = True
    expected_calendar_groups: int | None = CALENDAR_GROUP_COUNT_EXPECTED

    def __post_init__(self) -> None:
        if self.protocol != OUTER_PROTOCOL_NAME:
            raise SplitAuthorityError(
                f"only {OUTER_PROTOCOL_NAME} is implemented; requested "
                f"{self.protocol}. This protocol is NOT full leave-one-date-out."
            )
        if self.inner_folds < 2:
            raise SplitAuthorityError("inner_folds must be at least two")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "protocol": self.protocol,
            "protocol_is_full_leave_one_date_out": False,
            "seed": self.seed,
            "inner_folds": self.inner_folds,
            "prohibit_object_track_key_spanning": (
                self.prohibit_object_track_key_spanning
            ),
            "expected_calendar_groups": self.expected_calendar_groups,
            "outer_fold_date_tokens": dict(OUTER_FOLD_DATE_TOKENS),
            "pooled_small_date_fold": POOLED_SMALL_DATE_FOLD,
        }


@dataclass(slots=True)
class SplitAuthorityTables:
    """Split assignment plus the descriptive statistics that must accompany it."""

    assignment: pd.DataFrame
    calendar_statistics: pd.DataFrame
    contract: dict[str, Any]


def canonical_calendar_date(value: str) -> tuple[str, str]:
    """Return ``(calendar_token, session_suffix)`` for one metadata string.

    ``101219a`` and ``101219b`` both return calendar token ``101219`` with
    suffixes ``a`` and ``b``, which is what binds them to one outer group.
    """

    text = str(value or "").strip()
    if not text:
        return "unknown_date", ""
    match = _DATE_TOKEN_RE.search(text) or _BARE_DATE_RE.search(text)
    if match is None:
        return "unknown_date", ""
    return match.group(1), match.group(2).lower()


def _clip_token(value: str) -> str:
    match = _CLIP_RE.search(str(value or ""))
    return match.group(1) if match else ""


def build_split_authority(
    metadata: pd.DataFrame,
    *,
    config: SplitAuthorityConfig | None = None,
) -> SplitAuthorityTables:
    """Build the deterministic split authority from a metadata table."""

    settings = config or SplitAuthorityConfig()
    missing = [
        column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns
    ]
    if missing:
        raise SplitAuthorityError(f"split metadata missing columns={missing}")
    work = metadata.reset_index(drop=True).copy()
    for column in REQUIRED_METADATA_COLUMNS:
        work[column] = work[column].fillna("").astype(str).str.strip()
        if work[column].eq("").any():
            blank = int(work[column].eq("").sum())
            raise SplitAuthorityError(f"blank {column} rows={blank}")
    if work["native_unit_id"].duplicated().any():
        duplicates = int(work["native_unit_id"].duplicated(keep=False).sum())
        raise SplitAuthorityError(
            f"native_unit_id must be unique in split metadata; duplicate "
            f"rows={duplicates}"
        )

    tokens: list[str] = []
    suffixes: list[str] = []
    for row in work.itertuples(index=False):
        explicit = getattr(row, "calendar_date_token", "")
        source_text = "|".join(
            [
                str(explicit or ""),
                str(row.video_key),
                str(row.dataset_id),
                str(row.source_type),
            ]
        )
        token, suffix = canonical_calendar_date(source_text)
        tokens.append(token)
        suffixes.append(suffix)
    work["calendar_date_token"] = tokens
    work["date_session_suffix"] = suffixes
    unknown = work["calendar_date_token"].eq("unknown_date")
    if unknown.any():
        sample = work.loc[unknown, "video_key"].head(5).tolist()
        raise SplitAuthorityError(
            "date-grouped splitting requires a resolvable calendar date; "
            f"unresolved rows={int(unknown.sum())} sample={sample}"
        )

    work["outer_date_group_id"] = "date=" + work["calendar_date_token"]
    clips = [_clip_token(row.video_key) for row in work.itertuples(index=False)]
    work["session_group_id"] = [
        f"session={token}|{suffix or 'none'}|{clip or 'none'}"
        for token, suffix, clip in zip(tokens, suffixes, clips, strict=True)
    ]
    work["inner_recording_group_id"] = (
        "recording="
        + work["source_type"]
        + "|"
        + work["dataset_id"]
        + "|"
        + work["video_key"]
    )
    work["outer_fold_id"] = [
        OUTER_FOLD_DATE_TOKENS.get(token, POOLED_SMALL_DATE_FOLD) for token in tokens
    ]
    work["outer_fold_is_pooled_small_dates"] = work["outer_fold_id"].eq(
        POOLED_SMALL_DATE_FOLD
    )
    work["inner_fold_id"] = [
        _stable_bucket(group_id, settings.seed, settings.inner_folds)
        for group_id in work["inner_recording_group_id"]
    ]

    calendar_groups = sorted(set(work["calendar_date_token"]))
    if settings.expected_calendar_groups is not None and (
        len(calendar_groups) != settings.expected_calendar_groups
    ):
        raise SplitAuthorityError(
            "calendar-date group count does not match the declared expectation: "
            f"observed={len(calendar_groups)} "
            f"expected={settings.expected_calendar_groups}; groups="
            f"{calendar_groups}"
        )

    statistics = per_calendar_date_statistics(work)
    contract = {
        **settings.to_payload(),
        "output_columns": list(SPLIT_OUTPUT_COLUMNS),
        "calendar_groups": calendar_groups,
        "calendar_group_count": len(calendar_groups),
        "date_tokens_with_session_suffix": sorted(
            {
                f"{token}{suffix}"
                for token, suffix in zip(tokens, suffixes, strict=True)
                if suffix
            }
        ),
        "outer_fold_support": {
            str(fold): int(count)
            for fold, count in work["outer_fold_id"].value_counts().sort_index().items()
        },
        "pooled_small_date_members": sorted(
            set(
                work.loc[
                    work["outer_fold_id"].eq(POOLED_SMALL_DATE_FOLD),
                    "calendar_date_token",
                ]
            )
        ),
        "effective_outer_generalization_units": len(OUTER_FOLD_NAMES),
        "statistical_power_note": (
            "four outer folds give low statistical power; per-date descriptive "
            "statistics are reported for all calendar groups but the tiny dates "
            "are not high-power independent tests"
        ),
        "row_splitting_used": False,
        "window_random_split_used": False,
        "assignment_sha256": _assignment_digest(work),
    }
    columns = [
        "native_unit_id",
        "source_type",
        "dataset_id",
        "video_key",
        "calendar_date_token",
        "date_session_suffix",
        *SPLIT_OUTPUT_COLUMNS,
        "outer_fold_is_pooled_small_dates",
    ]
    if "object_track_key" in work.columns:
        columns.insert(4, "object_track_key")
    return SplitAuthorityTables(
        assignment=work[columns].copy(),
        calendar_statistics=statistics,
        contract=contract,
    )


def per_calendar_date_statistics(assignment: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for every calendar-date group."""

    grouped = assignment.groupby("calendar_date_token", sort=True)
    records: list[dict[str, Any]] = []
    total = int(len(assignment))
    for token, group in grouped:
        records.append(
            {
                "calendar_date_token": str(token),
                "outer_date_group_id": str(group["outer_date_group_id"].iloc[0]),
                "outer_fold_id": str(group["outer_fold_id"].iloc[0]),
                "native_units": int(len(group)),
                "share_of_native_units": float(len(group)) / float(max(1, total)),
                "sources": "|".join(sorted(set(group["source_type"].astype(str)))),
                "source_count": int(group["source_type"].nunique()),
                "videos": int(group["video_key"].nunique()),
                "session_groups": int(group["session_group_id"].nunique()),
                "recording_groups": int(group["inner_recording_group_id"].nunique()),
                "is_pooled_small_date": bool(
                    group["outer_fold_is_pooled_small_dates"].iloc[0]
                ),
                "high_power_independent_test": False,
            }
        )
    return pd.DataFrame.from_records(records)


def outer_fold_partition(
    assignment: pd.DataFrame,
    fold_id: str,
) -> tuple[pd.Index, pd.Index]:
    """Return ``(train_index, test_index)`` for one outer fold."""

    if fold_id not in OUTER_FOLD_NAMES:
        raise SplitAuthorityError(
            f"unknown outer fold={fold_id}; expected one of {list(OUTER_FOLD_NAMES)}"
        )
    held_out = assignment["outer_fold_id"].eq(fold_id)
    return assignment.index[~held_out], assignment.index[held_out]


def _stable_bucket(group_id: str, seed: int, buckets: int) -> int:
    payload = f"{seed}|{group_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % buckets


def _assignment_digest(assignment: pd.DataFrame) -> str:
    ordered = assignment.sort_values("native_unit_id", kind="stable")
    lines = [
        "|".join(
            [
                str(row.native_unit_id),
                str(row.outer_date_group_id),
                str(row.session_group_id),
                str(row.inner_recording_group_id),
                str(row.outer_fold_id),
                str(row.inner_fold_id),
            ]
        )
        for row in ordered.itertuples(index=False)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def forbidden_split_columns(columns: Sequence[str]) -> list[str]:
    """Return split-authority columns that leaked into a candidate X schema."""

    forbidden = set(SPLIT_OUTPUT_COLUMNS) | {
        "calendar_date_token",
        "date_session_suffix",
        "outer_fold_is_pooled_small_dates",
    }
    return sorted({str(name) for name in columns if str(name) in forbidden})


__all__ = [
    "CALENDAR_GROUP_COUNT_EXPECTED",
    "OUTER_FOLD_DATE_TOKENS",
    "OUTER_FOLD_NAMES",
    "OUTER_PROTOCOL_NAME",
    "POOLED_SMALL_DATE_FOLD",
    "REQUIRED_METADATA_COLUMNS",
    "SPLIT_OUTPUT_COLUMNS",
    "SPLIT_SCHEMA_VERSION",
    "SplitAuthorityConfig",
    "SplitAuthorityError",
    "SplitAuthorityTables",
    "build_split_authority",
    "canonical_calendar_date",
    "forbidden_split_columns",
    "outer_fold_partition",
    "per_calendar_date_statistics",
]
