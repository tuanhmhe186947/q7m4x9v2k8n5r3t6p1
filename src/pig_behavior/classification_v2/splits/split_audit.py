"""Fail-closed audits for the date-grouped split authority."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.splits.date_grouped_split import (
    OUTER_FOLD_NAMES,
    SplitAuthorityConfig,
    SplitAuthorityError,
    SplitAuthorityTables,
    build_split_authority,
    forbidden_split_columns,
    outer_fold_partition,
)

SPLIT_AUDIT_SCHEMA_VERSION = "classification_v2.split_audit.v1"

SPLIT_AUDIT_CHECKS: tuple[str, ...] = (
    "NO_CALENDAR_DATE_SPANS_OUTER_FOLDS",
    "NO_NATIVE_UNIT_SPANS_TRAIN_TEST",
    "NO_OBJECT_TRACK_KEY_SPANS_TRAIN_TEST_WHEN_PROHIBITED",
    "CROSS_SOURCE_DATES_COLOCATED",
    "SMALL_DATE_POOL_MEMBERS_RECORDED",
    "FORBIDDEN_SPLIT_COLUMNS_EXCLUDED_FROM_X",
    "SPLIT_REPRODUCIBLE",
    "FOLD_SUPPORT_REPORTED",
)


def audit_split_authority(
    tables: SplitAuthorityTables,
    *,
    metadata: pd.DataFrame,
    config: SplitAuthorityConfig | None = None,
    candidate_x_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Run every declared split audit and return a structured result."""

    settings = config or SplitAuthorityConfig()
    assignment = tables.assignment
    results: dict[str, dict[str, Any]] = {}

    spanning_dates = (
        assignment.groupby("outer_date_group_id")["outer_fold_id"].nunique().gt(1)
    )
    offenders = sorted(spanning_dates.index[spanning_dates].astype(str))
    results["NO_CALENDAR_DATE_SPANS_OUTER_FOLDS"] = {
        "passed": not offenders,
        "offending_dates": offenders,
    }

    unit_span: list[str] = []
    track_span: list[str] = []
    for fold in OUTER_FOLD_NAMES:
        train_index, test_index = outer_fold_partition(assignment, fold)
        train = assignment.loc[train_index]
        test = assignment.loc[test_index]
        shared_units = set(train["native_unit_id"]) & set(test["native_unit_id"])
        unit_span.extend(f"{fold}:{unit}" for unit in sorted(shared_units))
        if settings.prohibit_object_track_key_spanning and (
            "object_track_key" in assignment.columns
        ):
            shared_tracks = set(train["object_track_key"]) & set(
                test["object_track_key"]
            )
            track_span.extend(f"{fold}:{track}" for track in sorted(shared_tracks))
    results["NO_NATIVE_UNIT_SPANS_TRAIN_TEST"] = {
        "passed": not unit_span,
        "offending_units": unit_span[:16],
    }
    results["NO_OBJECT_TRACK_KEY_SPANS_TRAIN_TEST_WHEN_PROHIBITED"] = {
        "passed": not track_span,
        "prohibited": settings.prohibit_object_track_key_spanning,
        "object_track_key_present": "object_track_key" in assignment.columns,
        "offending_tracks": track_span[:16],
    }

    cross_source = (
        assignment.groupby("calendar_date_token")
        .agg(sources=("source_type", "nunique"), folds=("outer_fold_id", "nunique"))
        .reset_index()
    )
    multi_source = cross_source[cross_source["sources"].gt(1)]
    split_multi_source = multi_source[multi_source["folds"].gt(1)]
    results["CROSS_SOURCE_DATES_COLOCATED"] = {
        "passed": bool(split_multi_source.empty),
        "cross_source_dates": sorted(
            multi_source["calendar_date_token"].astype(str).tolist()
        ),
        "cross_source_dates_split_across_folds": sorted(
            split_multi_source["calendar_date_token"].astype(str).tolist()
        ),
    }

    pooled_rows = assignment.loc[assignment["outer_fold_is_pooled_small_dates"]]
    pooled = sorted(set(pooled_rows["calendar_date_token"].astype(str)))
    pooled_tokens = sorted(
        {
            f"{token}{suffix}"
            for token, suffix in zip(
                pooled_rows["calendar_date_token"].astype(str),
                pooled_rows["date_session_suffix"].astype(str),
                strict=True,
            )
        }
    )
    results["SMALL_DATE_POOL_MEMBERS_RECORDED"] = {
        "passed": bool(pooled),
        "pooled_calendar_dates": pooled,
        "pooled_date_count": len(pooled),
        "pooled_date_tokens": pooled_tokens,
        "pooled_date_token_count": len(pooled_tokens),
        "claimed_high_power_independent_test": False,
    }

    leaked = forbidden_split_columns(candidate_x_columns)
    results["FORBIDDEN_SPLIT_COLUMNS_EXCLUDED_FROM_X"] = {
        "passed": not leaked,
        "leaked_columns": leaked,
        "candidate_x_columns_checked": int(len(candidate_x_columns)),
    }

    repeated = build_split_authority(metadata, config=settings)
    reproducible = (
        repeated.contract["assignment_sha256"] == tables.contract["assignment_sha256"]
    )
    results["SPLIT_REPRODUCIBLE"] = {
        "passed": bool(reproducible),
        "assignment_sha256": tables.contract["assignment_sha256"],
        "repeat_assignment_sha256": repeated.contract["assignment_sha256"],
    }

    support = {
        str(fold): int(count)
        for fold, count in assignment["outer_fold_id"]
        .value_counts()
        .sort_index()
        .items()
    }
    missing_folds = [fold for fold in OUTER_FOLD_NAMES if fold not in support]
    results["FOLD_SUPPORT_REPORTED"] = {
        "passed": not missing_folds,
        "outer_fold_support": support,
        "missing_folds": missing_folds,
        "inner_fold_support": {
            str(fold): int(count)
            for fold, count in assignment["inner_fold_id"]
            .value_counts()
            .sort_index()
            .items()
        },
    }

    failed = [name for name, payload in results.items() if not payload["passed"]]
    return {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "protocol": tables.contract["protocol"],
        "checks": results,
        "failed_checks": failed,
        "passed": not failed,
    }


def require_split_authority(
    tables: SplitAuthorityTables,
    *,
    metadata: pd.DataFrame,
    config: SplitAuthorityConfig | None = None,
    candidate_x_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a clean split audit or raise one precise error."""

    audit = audit_split_authority(
        tables,
        metadata=metadata,
        config=config,
        candidate_x_columns=candidate_x_columns,
    )
    if not audit["passed"]:
        details = {
            name: audit["checks"][name] for name in audit["failed_checks"]
        }
        raise SplitAuthorityError(f"split audit failed: {details}")
    return audit


__all__ = [
    "SPLIT_AUDIT_CHECKS",
    "SPLIT_AUDIT_SCHEMA_VERSION",
    "audit_split_authority",
    "require_split_authority",
]
