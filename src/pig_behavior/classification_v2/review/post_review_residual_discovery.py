"""Review-informed residual discovery without automatic relabeling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.composite_review_authority import (
    SCOPE_OUTCOME_COLUMNS,
)

UNIVERSE_REQUIRED_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "track_id",
    "unit_start_frame",
    "unit_end_frame",
    "behavior_label",
)
DECISION_REQUIRED_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "behavior_label",
    "manual_review_decision",
    "manual_corrected_behavior",
)


class ResidualDiscoveryContractError(ValueError):
    """Raised when the residual-discovery contract is not provable."""


def build_review_informed_temporal_residuals(
    universe: pd.DataFrame,
    composite_decisions: pd.DataFrame,
    *,
    maximum_gap_run_units: int = 2,
    included_severities: tuple[str, ...] = ("HIGH", "MEDIUM"),
) -> dict[str, Any]:
    """Find short unreviewed gaps bounded by one effective reviewed label."""
    _require_columns(universe, UNIVERSE_REQUIRED_COLUMNS, "universe")
    _require_columns(
        composite_decisions,
        DECISION_REQUIRED_COLUMNS,
        "composite_decisions",
    )
    if maximum_gap_run_units < 1:
        raise ResidualDiscoveryContractError("maximum_gap_run_units_invalid")
    normalized_severities = tuple(
        str(severity).strip().upper() for severity in included_severities
    )
    if not normalized_severities:
        raise ResidualDiscoveryContractError("included_severities_empty")
    invalid_severities = sorted(
        set(normalized_severities).difference({"HIGH", "MEDIUM"})
    )
    if invalid_severities:
        raise ResidualDiscoveryContractError(
            "included_severities_invalid=" + ",".join(invalid_severities)
        )

    units = universe.copy()
    decisions = composite_decisions.copy()
    for column in UNIVERSE_REQUIRED_COLUMNS:
        units[column] = _normalized(units[column])
    for column in DECISION_REQUIRED_COLUMNS:
        decisions[column] = _normalized(decisions[column])
    if units["temporal_unit_key"].duplicated().any():
        raise ResidualDiscoveryContractError("universe_temporal_unit_duplicate")
    if decisions["temporal_unit_key"].duplicated().any():
        raise ResidualDiscoveryContractError("decision_temporal_unit_duplicate")

    unit_by_key = units.set_index("temporal_unit_key", drop=False)
    missing = sorted(set(decisions["temporal_unit_key"]) - set(unit_by_key.index))
    if missing:
        raise ResidualDiscoveryContractError(
            f"reviewed_keys_missing_from_universe={len(missing)}"
        )
    source_labels = decisions["temporal_unit_key"].map(
        unit_by_key["behavior_label"]
    )
    if not source_labels.eq(decisions["behavior_label"]).all():
        raise ResidualDiscoveryContractError("composite_source_label_mismatch")

    corrected = decisions["manual_review_decision"].eq("corrected")
    effective_by_key = dict(
        zip(
            decisions["temporal_unit_key"],
            decisions["behavior_label"].where(
                ~corrected,
                decisions["manual_corrected_behavior"],
            ),
            strict=True,
        )
    )
    reviewed_keys = set(decisions["temporal_unit_key"])
    corrected_keys = set(decisions.loc[corrected, "temporal_unit_key"])
    units["effective_behavior"] = units["temporal_unit_key"].map(
        effective_by_key
    ).fillna(units["behavior_label"])
    units["reviewed"] = units["temporal_unit_key"].isin(reviewed_keys)
    units["corrected_from_source"] = units["temporal_unit_key"].isin(
        corrected_keys
    )
    units["_start"] = _integer_series(units["unit_start_frame"], "unit_start_frame")
    units["_end"] = _integer_series(units["unit_end_frame"], "unit_end_frame")

    finding_rows: list[dict[str, Any]] = []
    group_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
    ]
    ordered = units.sort_values(
        [*group_columns, "_start", "_end", "temporal_unit_key"],
        kind="mergesort",
    )
    run_counter = 0
    for _, group in ordered.groupby(group_columns, sort=True, dropna=False):
        rows = group.reset_index(drop=True)
        runs = _contiguous_behavior_runs(rows)
        for run_index in range(1, len(runs) - 1):
            middle_start, middle_end = runs[run_index]
            if middle_end - middle_start + 1 > maximum_gap_run_units:
                continue
            left_start, left_end = runs[run_index - 1]
            right_start, right_end = runs[run_index + 1]
            left = rows.iloc[left_start : left_end + 1]
            middle = rows.iloc[middle_start : middle_end + 1]
            right = rows.iloc[right_start : right_end + 1]
            flank_behavior = _text(left.iloc[-1]["effective_behavior"])
            middle_behavior = _text(middle.iloc[0]["effective_behavior"])
            if flank_behavior != _text(right.iloc[0]["effective_behavior"]):
                continue
            if flank_behavior == middle_behavior:
                continue
            unreviewed_middle = middle.loc[~middle["reviewed"]]
            if unreviewed_middle.empty:
                continue

            review_informed = bool(
                left.iloc[-1]["corrected_from_source"]
                or middle["corrected_from_source"].any()
                or right.iloc[0]["corrected_from_source"]
            )
            severity = "HIGH" if flank_behavior == "fight" else "MEDIUM"
            if not review_informed:
                severity = "LOW"
            run_counter += 1
            run_id = f"residual_gap_{run_counter:07d}"
            related_keys = [
                _text(left.iloc[-1]["temporal_unit_key"]),
                *unreviewed_middle["temporal_unit_key"].tolist(),
                _text(right.iloc[0]["temporal_unit_key"]),
            ]
            for target in unreviewed_middle.to_dict(orient="records"):
                finding_rows.append(
                    {
                        "finding_id": run_id,
                        "temporal_unit_key": target["temporal_unit_key"],
                        "source_type": target["source_type"],
                        "dataset_id": target["dataset_id"],
                        "video_key": target["video_key"],
                        "object_track_key": target["object_track_key"],
                        "track_id": target["track_id"],
                        "unit_start_frame": int(target["_start"]),
                        "unit_end_frame": int(target["_end"]),
                        "source_behavior": target["behavior_label"],
                        "effective_behavior": target["effective_behavior"],
                        "bounded_flank_behavior": flank_behavior,
                        "gap_run_unit_count": int(len(middle)),
                        "review_informed": review_informed,
                        "severity": severity,
                        "finding_reason": (
                            "POST_REVIEW_NON_FIGHT_GAP_BETWEEN_FIGHT"
                            if flank_behavior == "fight"
                            else "POST_REVIEW_SHORT_LABEL_GAP"
                        ),
                        "related_temporal_unit_keys": "||".join(related_keys),
                        "suggested_review_hypothesis": flank_behavior,
                        "automatic_label_change": False,
                    }
                )

    findings = pd.DataFrame(finding_rows, columns=_finding_columns())
    if findings.empty:
        selected_findings = findings.copy()
    else:
        selected_findings = findings.loc[
            findings["review_informed"]
            & findings["severity"].isin(normalized_severities)
        ].copy()
        selected_findings = selected_findings.sort_values(
            [
                "severity",
                "video_key",
                "object_track_key",
                "unit_start_frame",
                "temporal_unit_key",
            ],
            key=lambda column: (
                column.map({"HIGH": 0, "MEDIUM": 1}).fillna(2)
                if column.name == "severity"
                else column
            ),
            kind="mergesort",
        ).reset_index(drop=True)

    selected_keys = selected_findings["temporal_unit_key"].tolist()
    if set(selected_keys) & reviewed_keys:
        raise ResidualDiscoveryContractError("selected_scope_overlaps_reviewed_keys")
    selected_scope = units.loc[
        units["temporal_unit_key"].isin(selected_keys)
    ].copy()
    selected_scope["_selection_order"] = selected_scope["temporal_unit_key"].map(
        {key: index for index, key in enumerate(selected_keys)}
    )
    selected_scope = selected_scope.sort_values("_selection_order")
    selected_scope = selected_scope.drop(
        columns=[
            "_selection_order",
            "_start",
            "_end",
            "effective_behavior",
            "reviewed",
            "corrected_from_source",
        ]
    )
    selected_scope = selected_scope.merge(
        selected_findings[
            [
                "temporal_unit_key",
                "finding_id",
                "finding_reason",
                "severity",
                "related_temporal_unit_keys",
                "suggested_review_hypothesis",
            ]
        ],
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    selected_scope["review_item_id"] = [
        f"post_review_residual_{index:07d}"
        for index in range(1, len(selected_scope) + 1)
    ]
    selected_scope["review_reason"] = selected_scope["finding_reason"]
    selected_scope["review_priority"] = selected_scope["severity"]
    selected_scope["consistency_review_order"] = range(1, len(selected_scope) + 1)
    selected_scope["consistency_group_ids"] = selected_scope["finding_id"]
    selected_scope["original_behavior"] = selected_scope["behavior_label"]
    selected_scope["final_scope_component"] = (
        "POST_REVIEW_REVIEW_INFORMED_RESIDUAL"
    )
    selected_scope = activate_post_review_scope_for_gui(
        selected_scope,
        cohort="POST_REVIEW_RESIDUAL_TARGET",
        reason_code="POST_REVIEW_REVIEW_INFORMED_TEMPORAL_GAP",
    )

    clean_columns = [
        column for column in universe.columns if column not in SCOPE_OUTCOME_COLUMNS
    ]
    control_population = universe.loc[:, clean_columns].copy()
    excluded_keys = reviewed_keys | set(selected_keys)
    control_exclusion = control_population.loc[
        control_population["temporal_unit_key"]
        .astype(str)
        .str.strip()
        .isin(excluded_keys)
    ].copy()
    audit = {
        "included_severities": list(normalized_severities),
        "universe_rows": int(len(universe)),
        "composite_reviewed_keys": int(len(reviewed_keys)),
        "unreviewed_rows_before_targeting": int(len(universe) - len(reviewed_keys)),
        "all_short_gap_finding_rows": int(len(findings)),
        "review_informed_target_rows": int(len(selected_findings)),
        "review_informed_target_runs": int(
            selected_findings["finding_id"].nunique()
        ),
        "review_informed_target_unique_keys": int(len(set(selected_keys))),
        "high_target_rows": int(selected_findings["severity"].eq("HIGH").sum()),
        "high_target_runs": int(
            selected_findings.loc[
                selected_findings["severity"].eq("HIGH"), "finding_id"
            ].nunique()
        ),
        "medium_target_rows": int(
            selected_findings["severity"].eq("MEDIUM").sum()
        ),
        "reviewed_target_overlap": 0,
        "automatic_label_changes": 0,
        "control_exclusion_rows": int(len(control_exclusion)),
    }
    return {
        "findings": findings,
        "selected_findings": selected_findings,
        "selected_scope": selected_scope,
        "control_population": control_population,
        "control_exclusion_scope": control_exclusion,
        "audit": audit,
    }


def activate_post_review_scope_for_gui(
    scope: pd.DataFrame,
    *,
    cohort: str,
    reason_code: str,
) -> pd.DataFrame:
    """Activate a derived scope while preserving parent auto-carry provenance."""
    cohort_text = _text(cohort)
    reason_text = _text(reason_code)
    if not cohort_text or cohort_text == "AUTO_CARRY_LOW_RISK":
        raise ResidualDiscoveryContractError("gui_cohort_invalid")
    if not reason_text:
        raise ResidualDiscoveryContractError("gui_reason_code_blank")
    out = scope.copy()
    parent_columns = (
        "candidate_tier",
        "include_in_review",
        "review_reason",
        "review_reason_codes",
        "review_selection_predicates",
        "auto_carry_behavior",
        "auto_carry_provenance",
    )
    for column in parent_columns:
        if column in out.columns:
            out[f"post_review_parent_{column}"] = out[column]
    version = "classification_v2.post_review_residual_selection.v1"
    config_payload = f"{version}|{cohort_text}|{reason_text}".encode()
    out["include_in_review"] = True
    out["candidate_tier"] = cohort_text
    out["review_reason"] = reason_text
    out["review_reason_codes"] = reason_text
    out["review_selection_predicates"] = cohort_text
    out["selection_predicate_version"] = version
    out["selection_config_hash"] = hashlib.sha256(config_payload).hexdigest()
    out["review_predicate_global_mandatory"] = False
    out["mandatory_behavior_review_unit"] = True
    out["auto_carry_behavior"] = ""
    out["auto_carry_provenance"] = ""
    out["human_decision_synthesized"] = False
    return out


def _contiguous_behavior_runs(rows: pd.DataFrame) -> list[tuple[int, int]]:
    if rows.empty:
        return []
    runs: list[tuple[int, int]] = []
    run_start = 0
    for index in range(1, len(rows) + 1):
        at_end = index == len(rows)
        if not at_end:
            behavior_changed = (
                rows.iloc[index]["effective_behavior"]
                != rows.iloc[run_start]["effective_behavior"]
            )
            noncontiguous = (
                int(rows.iloc[index]["_start"])
                != int(rows.iloc[index - 1]["_end"]) + 1
            )
        else:
            behavior_changed = True
            noncontiguous = True
        if behavior_changed or noncontiguous:
            runs.append((run_start, index - 1))
            run_start = index
    return runs


def _finding_columns() -> list[str]:
    return [
        "finding_id",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "source_behavior",
        "effective_behavior",
        "bounded_flank_behavior",
        "gap_run_unit_count",
        "review_informed",
        "severity",
        "finding_reason",
        "related_temporal_unit_keys",
        "suggested_review_hypothesis",
        "automatic_label_change",
    ]


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ResidualDiscoveryContractError(
            f"{label}_missing_columns={','.join(missing)}"
        )


def _normalized(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ResidualDiscoveryContractError(
            f"{label}_non_numeric={int(numeric.isna().sum())}"
        )
    return numeric.astype(int)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
