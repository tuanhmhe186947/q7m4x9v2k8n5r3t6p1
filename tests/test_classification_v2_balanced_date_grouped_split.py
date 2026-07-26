"""Deterministic date-grouped split authority tests (fixture metadata only)."""

from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.splits.date_grouped_split import (
    CALENDAR_GROUP_COUNT_EXPECTED,
    OUTER_FOLD_NAMES,
    OUTER_PROTOCOL_NAME,
    SPLIT_OUTPUT_COLUMNS,
    SplitAuthorityConfig,
    SplitAuthorityError,
    build_split_authority,
    canonical_calendar_date,
    forbidden_split_columns,
    outer_fold_partition,
)
from pig_behavior.classification_v2.splits.split_audit import (
    SPLIT_AUDIT_CHECKS,
    audit_split_authority,
    require_split_authority,
)

#: 14 date tokens that collapse to 13 calendar groups (101219a/b are bound).
FIXTURE_DATE_TOKENS: tuple[tuple[str, str, int], ...] = (
    ("291119", "cvat_tracking_xml", 40),
    ("291119", "legacy_recovered", 20),
    ("301119", "legacy_recovered", 35),
    ("281119", "cvat_tracking_xml", 10),
    ("281119", "legacy_recovered", 12),
    ("101219a", "legacy_recovered", 4),
    ("101219b", "legacy_recovered", 3),
    ("021219", "legacy_recovered", 3),
    ("031219", "legacy_recovered", 2),
    ("041219", "legacy_recovered", 2),
    ("051219", "legacy_recovered", 2),
    ("061219", "legacy_recovered", 2),
    ("091219", "legacy_recovered", 2),
    ("111219", "legacy_recovered", 2),
    ("121219", "legacy_recovered", 2),
    ("131219", "legacy_recovered", 2),
)


def _fixture_metadata() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for token, source, count in FIXTURE_DATE_TOKENS:
        for index in range(count):
            clip = f"{index:06d}"
            video_key = f"Pigs{token}_{clip}_30fps"
            rows.append(
                {
                    "native_unit_id": f"{source}|{token}|{clip}|unit{index}",
                    "source_type": source,
                    "dataset_id": f"dataset_{source}",
                    "video_key": video_key,
                    "object_track_key": f"{source}|{token}|{clip}|track{index}",
                }
            )
    return pd.DataFrame(rows)


def test_calendar_date_binding_merges_session_suffixes() -> None:
    assert canonical_calendar_date("Pigs101219a_000001_30fps") == ("101219", "a")
    assert canonical_calendar_date("Pigs101219b_000002_30fps") == ("101219", "b")
    assert canonical_calendar_date("Pigs291119_000226_30fps") == ("291119", "")


def test_thirteen_calendar_groups_and_four_outer_folds() -> None:
    metadata = _fixture_metadata()
    tables = build_split_authority(metadata)
    assert tables.contract["protocol"] == OUTER_PROTOCOL_NAME
    assert tables.contract["protocol_is_full_leave_one_date_out"] is False
    assert tables.contract["calendar_group_count"] == CALENDAR_GROUP_COUNT_EXPECTED
    assert sorted(set(tables.assignment["outer_fold_id"])) == sorted(OUTER_FOLD_NAMES)
    assert set(SPLIT_OUTPUT_COLUMNS).issubset(tables.assignment.columns)

    fold_of = dict(
        zip(
            tables.assignment["calendar_date_token"],
            tables.assignment["outer_fold_id"],
            strict=True,
        )
    )
    assert fold_of["291119"] == "FOLD_1"
    assert fold_of["301119"] == "FOLD_2"
    assert fold_of["281119"] == "FOLD_3"
    assert fold_of["101219"] == "FOLD_4"


def test_cross_source_dates_stay_in_one_outer_fold() -> None:
    tables = build_split_authority(_fixture_metadata())
    for token in ("281119", "291119"):
        rows = tables.assignment[tables.assignment["calendar_date_token"].eq(token)]
        assert rows["source_type"].nunique() == 2
        assert rows["outer_fold_id"].nunique() == 1


def test_no_calendar_date_spans_two_outer_folds() -> None:
    tables = build_split_authority(_fixture_metadata())
    spans = tables.assignment.groupby("outer_date_group_id")["outer_fold_id"].nunique()
    assert int(spans.max()) == 1


def test_all_declared_audits_run_and_pass() -> None:
    metadata = _fixture_metadata()
    tables = build_split_authority(metadata)
    audit = audit_split_authority(
        tables,
        metadata=metadata,
        candidate_x_columns=["cx_n", "cy_n", "speed_n_per_second"],
    )
    assert tuple(audit["checks"]) == SPLIT_AUDIT_CHECKS
    assert audit["passed"], audit["failed_checks"]
    pooled = audit["checks"]["SMALL_DATE_POOL_MEMBERS_RECORDED"]
    # 11 small legacy date *tokens* collapse to 10 pooled calendar groups,
    # because 101219a and 101219b share calendar date 101219.
    assert pooled["pooled_date_token_count"] == 11
    assert pooled["pooled_date_count"] == 10
    assert (
        audit["checks"]["SMALL_DATE_POOL_MEMBERS_RECORDED"][
            "claimed_high_power_independent_test"
        ]
        is False
    )
    require_split_authority(tables, metadata=metadata)


def test_forbidden_split_columns_are_detected_in_candidate_x() -> None:
    metadata = _fixture_metadata()
    tables = build_split_authority(metadata)
    audit = audit_split_authority(
        tables,
        metadata=metadata,
        candidate_x_columns=["cx_n", "outer_date_group_id", "inner_fold_id"],
    )
    check = audit["checks"]["FORBIDDEN_SPLIT_COLUMNS_EXCLUDED_FROM_X"]
    assert not check["passed"]
    assert check["leaked_columns"] == ["inner_fold_id", "outer_date_group_id"]
    assert forbidden_split_columns(["cx_n"]) == []


def test_split_is_reproducible_and_seed_sensitive_only_inside_folds() -> None:
    metadata = _fixture_metadata()
    first = build_split_authority(metadata)
    second = build_split_authority(metadata)
    assert first.contract["assignment_sha256"] == second.contract["assignment_sha256"]

    shuffled = metadata.sample(frac=1.0, random_state=5).reset_index(drop=True)
    reordered = build_split_authority(shuffled)
    assert reordered.contract["assignment_sha256"] == first.contract["assignment_sha256"]

    other_seed = build_split_authority(
        metadata,
        config=SplitAuthorityConfig(seed=999),
    )
    assert other_seed.contract["assignment_sha256"] != first.contract["assignment_sha256"]
    assert other_seed.assignment["outer_fold_id"].tolist() == (
        first.assignment["outer_fold_id"].tolist()
    )


def test_no_native_unit_or_track_spans_train_and_test() -> None:
    tables = build_split_authority(_fixture_metadata())
    for fold in OUTER_FOLD_NAMES:
        train_index, test_index = outer_fold_partition(tables.assignment, fold)
        train = tables.assignment.loc[train_index]
        test = tables.assignment.loc[test_index]
        assert not set(train["native_unit_id"]) & set(test["native_unit_id"])
        assert not set(train["object_track_key"]) & set(test["object_track_key"])


def test_per_date_statistics_cover_every_calendar_group() -> None:
    tables = build_split_authority(_fixture_metadata())
    statistics = tables.calendar_statistics
    assert len(statistics) == CALENDAR_GROUP_COUNT_EXPECTED
    assert set(statistics["high_power_independent_test"]) == {False}
    cross_source = statistics[statistics["source_count"].gt(1)]
    assert sorted(cross_source["calendar_date_token"]) == ["281119", "291119"]
    assert statistics["native_units"].sum() == len(tables.assignment)


def test_unresolvable_dates_and_duplicate_units_fail_closed() -> None:
    metadata = _fixture_metadata()
    broken = metadata.copy()
    broken.loc[0, "video_key"] = "no_date_here"
    with pytest.raises(SplitAuthorityError, match="calendar date"):
        build_split_authority(broken)

    duplicated = pd.concat([metadata, metadata.head(1)], ignore_index=True)
    with pytest.raises(SplitAuthorityError, match="unique"):
        build_split_authority(duplicated)


def test_unexpected_calendar_group_count_fails_closed() -> None:
    metadata = _fixture_metadata().head(40)
    with pytest.raises(SplitAuthorityError, match="calendar-date group count"):
        build_split_authority(metadata)
