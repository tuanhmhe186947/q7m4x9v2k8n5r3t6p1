"""Select a complete legacy 16-frame development source without review fiction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    INTERACTION_BEHAVIORS,
    VALID_BEHAVIOR_SET,
)

LINEAGE_SCOPE = "legacy-only-unreviewed-development"
SOURCE_TYPE = "legacy_recovered"
DATASET_ID = "legacy_recovered_16f"
EXPECTED_SEQUENCE_LENGTH = 16
REVIEW_STATUS = "operator_cvat_checked_pending_hidden_behavior_double_check"


@dataclass(frozen=True, slots=True)
class LegacyC6SourceSelection:
    """Selected frame/object rows and their fail-closed audit."""

    frames: pd.DataFrame
    audit: dict[str, Any]


def select_legacy_c6_screening_source(
    frames: pd.DataFrame,
    *,
    max_groups: int | None = None,
    selection_salt: str = "legacy_c6_complete_group_gate_v1",
) -> LegacyC6SourceSelection:
    """Select whole groups and attach explicit unreviewed-development claims."""

    required = {
        "source_type",
        "dataset_id",
        "video_key",
        "clip_id",
        "track_id",
        "pig_id",
        "frame_uid",
        "scene_frame_uid",
        "relative_frame_index",
        "behavior",
        "bbox_valid",
        "include_in_training",
        "use_for_main_eval",
        "crop_path",
    }
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise ValueError(f"legacy C6 source missing columns={missing}")
    if frames.empty:
        raise ValueError("legacy C6 source is empty")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive")

    work = frames.copy()
    _require_constant(work, "source_type", SOURCE_TYPE)
    _require_constant(work, "dataset_id", DATASET_ID)
    _require_nonblank(work, ("video_key", "clip_id", "track_id", "pig_id"))

    if max_groups is not None:
        group_keys = work[["video_key", "clip_id"]].drop_duplicates()
        group_keys["_rank"] = [
            _selection_score(selection_salt, video_key, clip_id)
            for video_key, clip_id in group_keys.itertuples(index=False)
        ]
        selected = group_keys.sort_values(
            ["_rank", "video_key", "clip_id"],
            kind="mergesort",
        ).head(max_groups)
        selected_keys = pd.MultiIndex.from_frame(
            selected[["video_key", "clip_id"]]
        )
        row_keys = pd.MultiIndex.from_frame(work[["video_key", "clip_id"]])
        work = work.loc[row_keys.isin(selected_keys)].copy()

    object_key = [
        "dataset_id",
        "video_key",
        "track_id",
        "pig_id",
    ]
    relative = pd.to_numeric(work["relative_frame_index"], errors="coerce")
    work["_relative_frame_index"] = relative
    work["_behavior_valid"] = work["behavior"].isin(VALID_BEHAVIOR_SET)
    selected_rows_before_filter = int(len(work))

    grouped = work.groupby(object_key, sort=True, dropna=False)
    actor_summary = grouped.agg(
        rows=("_relative_frame_index", "size"),
        frame_count=("_relative_frame_index", "nunique"),
        frame_min=("_relative_frame_index", "min"),
        frame_max=("_relative_frame_index", "max"),
        behavior_count=("behavior", "nunique"),
        behavior_all_valid=("_behavior_valid", "all"),
        group_count=("clip_id", "nunique"),
        bbox_all_valid=("bbox_valid", _all_bool),
        include_all=("include_in_training", _all_bool),
        main_eval_all=("use_for_main_eval", _all_bool),
        crop_path_count=("crop_path", _nonblank_count),
    ).reset_index()
    actor_complete = (
        actor_summary["rows"].eq(EXPECTED_SEQUENCE_LENGTH)
        & actor_summary["frame_count"].eq(EXPECTED_SEQUENCE_LENGTH)
        & actor_summary["frame_min"].eq(0)
        & actor_summary["frame_max"].eq(EXPECTED_SEQUENCE_LENGTH - 1)
        & actor_summary["behavior_count"].eq(1)
        & actor_summary["behavior_all_valid"]
        & actor_summary["group_count"].eq(1)
        & actor_summary["bbox_all_valid"]
        & actor_summary["include_all"]
        & actor_summary["main_eval_all"]
        & actor_summary["crop_path_count"].eq(EXPECTED_SEQUENCE_LENGTH)
    )
    excluded = actor_summary.loc[~actor_complete].copy()
    excluded["reasons"] = excluded.apply(_actor_exclusion_reasons, axis=1)
    excluded_actor_units = [
        {
            **{key: str(row[key]) for key in object_key},
            "reasons": list(row["reasons"]),
        }
        for _, row in excluded.iterrows()
    ]

    eligible_keys = pd.MultiIndex.from_frame(
        actor_summary.loc[actor_complete, object_key]
    )
    actor_keys = pd.MultiIndex.from_frame(work[object_key])
    work = work.loc[actor_keys.isin(eligible_keys)].copy()
    while True:
        scene_counts = work.groupby("scene_frame_uid", dropna=False)[
            "pig_id"
        ].transform("nunique")
        missing_partner = (
            work["behavior"].isin(INTERACTION_BEHAVIORS)
            & scene_counts.lt(2)
        )
        if not bool(missing_partner.any()):
            break
        cascade_keys = work.loc[missing_partner, object_key].drop_duplicates()
        excluded_actor_units.extend(
            {
                **{key: str(row[key]) for key in object_key},
                "reasons": ["missing_interaction_partner_after_source_filter"],
            }
            for _, row in cascade_keys.iterrows()
        )
        cascade_index = pd.MultiIndex.from_frame(cascade_keys)
        actor_keys = pd.MultiIndex.from_frame(work[object_key])
        work = work.loc[~actor_keys.isin(cascade_index)].copy()
    if work.empty:
        raise ValueError("legacy C6 source has no eligible complete actor units")

    relative = pd.to_numeric(work["_relative_frame_index"], errors="coerce")
    duplicate_rows = int(
        work.duplicated(object_key + ["_relative_frame_index"], keep=False).sum()
    )
    duplicate_frame_uid = int(work["frame_uid"].astype(str).duplicated().sum())
    invalid_relative = int(
        (
            relative.isna()
            | relative.mod(1).ne(0)
            | relative.lt(0)
            | relative.ge(EXPECTED_SEQUENCE_LENGTH)
        ).sum()
    )
    invalid_behavior = sorted(
        set(work["behavior"].fillna("").astype(str)).difference(
            VALID_BEHAVIOR_SET
        )
    )

    errors: list[str] = []
    for name, count in {
        "duplicate_actor_frame_rows": duplicate_rows,
        "duplicate_frame_uid_rows": duplicate_frame_uid,
        "invalid_relative_frame_rows": invalid_relative,
    }.items():
        if count:
            errors.append(f"{name}={count}")
    if invalid_behavior:
        errors.append(f"invalid_behaviors={invalid_behavior}")
    if errors:
        raise ValueError("legacy C6 source contract failed: " + "; ".join(errors))

    work = work.drop(columns=["_relative_frame_index", "_behavior_valid"])
    work["lineage_scope"] = LINEAGE_SCOPE
    work["human_review_complete"] = False
    work["review_status"] = REVIEW_STATUS
    work["development_use_authorized"] = True
    work = work.sort_values(
        ["video_key", "clip_id", "pig_id", "relative_frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    behavior_counts = (
        work.drop_duplicates(object_key)["behavior"].value_counts().sort_index()
    )
    audit = {
        "schema_version": "classification_v2.legacy_c6_source_selection.v1",
        "status": "PASS_LEGACY_C6_SOURCE_SELECTION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "review_status": REVIEW_STATUS,
        "selection_scope": "complete_groups" if max_groups is not None else "full",
        "selection_salt": selection_salt if max_groups is not None else "",
        "requested_max_groups": max_groups,
        "rows": int(len(work)),
        "scene_frames": int(work["scene_frame_uid"].astype(str).nunique()),
        "frame_objects": int(work["frame_uid"].astype(str).nunique()),
        "groups": int(work[["video_key", "clip_id"]].drop_duplicates().shape[0]),
        "actors": int(len(actor_summary) - len(excluded_actor_units)),
        "selected_actor_units_before_filter": int(len(actor_summary)),
        "excluded_actor_unit_count": int(len(excluded_actor_units)),
        "excluded_actor_units": excluded_actor_units,
        "behavior_actor_counts": {
            str(label): int(count) for label, count in behavior_counts.items()
        },
        "duplicate_actor_frame_rows": duplicate_rows,
        "duplicate_frame_uid_rows": duplicate_frame_uid,
        "invalid_relative_frame_rows": invalid_relative,
        "incomplete_actor_units": 0,
        "rows_dropped_inside_selected_groups": (
            selected_rows_before_filter - int(len(work))
        ),
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    return LegacyC6SourceSelection(frames=work, audit=audit)


def _selection_score(salt: str, video_key: object, clip_id: object) -> str:
    value = f"{salt}|{video_key}|{clip_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _actor_exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row["rows"] != EXPECTED_SEQUENCE_LENGTH:
        reasons.append("row_count_not_16")
    if row["frame_count"] != EXPECTED_SEQUENCE_LENGTH:
        reasons.append("relative_frame_count_not_16")
    if row["frame_min"] != 0 or row["frame_max"] != EXPECTED_SEQUENCE_LENGTH - 1:
        reasons.append("relative_frame_range_not_0_15")
    if row["behavior_count"] != 1:
        reasons.append("behavior_not_constant")
    if not bool(row["behavior_all_valid"]):
        reasons.append("invalid_behavior")
    if row["group_count"] != 1:
        reasons.append("actor_spans_multiple_clips")
    if not bool(row["bbox_all_valid"]):
        reasons.append("invalid_bbox")
    if not bool(row["include_all"]):
        reasons.append("source_excluded_from_training")
    if not bool(row["main_eval_all"]):
        reasons.append("source_excluded_from_main_eval")
    if row["crop_path_count"] != EXPECTED_SEQUENCE_LENGTH:
        reasons.append("missing_crop_path")
    return reasons


def _require_constant(frame: pd.DataFrame, column: str, expected: str) -> None:
    observed = set(frame[column].fillna("").astype(str).str.strip())
    if observed != {expected}:
        raise ValueError(f"legacy C6 {column}={sorted(observed)} expected={expected}")


def _require_nonblank(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"legacy C6 source contains blank {column}")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}
    if (~normalized.isin(allowed)).any():
        raise ValueError(f"invalid boolean values={sorted(set(normalized) - allowed)}")
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _all_bool(series: pd.Series) -> bool:
    return bool(_as_bool(series).all())


def _nonblank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


__all__ = [
    "DATASET_ID",
    "EXPECTED_SEQUENCE_LENGTH",
    "LINEAGE_SCOPE",
    "LegacyC6SourceSelection",
    "REVIEW_STATUS",
    "SOURCE_TYPE",
    "select_legacy_c6_screening_source",
]
