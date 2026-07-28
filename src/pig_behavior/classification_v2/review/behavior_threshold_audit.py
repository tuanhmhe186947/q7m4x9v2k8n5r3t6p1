"""Independent threshold comparison and sensitivity audits for Behavior Review."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or not all(
        isinstance(record, dict) for record in parsed
    ):
        raise ValueError("threshold trace must be a JSON list of objects")
    return parsed


def _compare(observed: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return observed < threshold
    if operator == "<=":
        return observed <= threshold
    if operator == ">":
        return observed > threshold
    if operator == ">=":
        return observed >= threshold
    if operator == "==":
        return observed == threshold
    if operator == "!=":
        return observed != threshold
    if operator == "SCALE_BY_REFERENCE":
        return threshold > 0
    raise ValueError(f"unknown comparison operator: {operator}")


def independent_threshold_candidate_audit(
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
    auto_carry: pd.DataFrame,
    registry_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Verify published comparisons without calling production predicates."""

    errors: list[str] = []
    thresholds = registry_snapshot.get("thresholds", [])
    active = {
        str(entry["threshold_id"]): entry
        for entry in thresholds
        if not bool(entry.get("deprecated", False))
    }
    expected_hash = _stable_hash(
        {
            "registry_id": registry_snapshot.get("registry_id"),
            "registry_version": registry_snapshot.get("registry_version"),
            "active_thresholds": [
                active[key] for key in sorted(active)
            ],
        }
    )
    if registry_snapshot.get("registry_hash") != expected_hash:
        errors.append("registry_snapshot_hash_mismatch")
    universe_keys = set(universe["review_key"].astype(str))
    candidate_keys = set(candidates["review_key"].astype(str))
    auto_keys = set(auto_carry["review_key"].astype(str))
    overlap = candidate_keys & auto_keys
    missing = universe_keys - candidate_keys - auto_keys
    extra = (candidate_keys | auto_keys) - universe_keys
    if overlap:
        errors.append(f"candidate_auto_carry_overlap={len(overlap)}")
    if missing:
        errors.append(f"missing_universe_keys={len(missing)}")
    if extra:
        errors.append(f"extra_partition_keys={len(extra)}")
    duplicate_keys = int(
        universe["review_key"].astype(str).duplicated().sum()
        + candidates["review_key"].astype(str).duplicated().sum()
        + auto_carry["review_key"].astype(str).duplicated().sum()
    )
    if duplicate_keys:
        errors.append(f"duplicate_review_keys={duplicate_keys}")

    checked = 0
    threshold_candidate_rows = 0
    missing_authority_rows = 0
    availability_only_rows = 0
    for _, row in candidates.iterrows():
        records = _records(row.get("threshold_binding_details", "[]"))
        reasons = {
            token
            for token in str(row.get("review_reason_codes", "")).split(";")
            if token
        }
        predicates = {
            token
            for token in str(
                row.get("review_selection_predicates", "")
            ).split(";")
            if token
        }
        non_threshold_only = bool(
            predicates
            and predicates.issubset(
                {
                    "rare_class_census",
                    "stratified_low_risk_audit",
                }
            )
        )
        if records:
            threshold_candidate_rows += 1
        elif not non_threshold_only:
            missing_authority_rows += 1
        if (
            reasons
            and reasons.issubset({"review_evidence_available"})
        ):
            availability_only_rows += 1
        for record in records:
            checked += 1
            threshold_id = str(record.get("threshold_id", ""))
            entry = active.get(threshold_id)
            if entry is None:
                errors.append(f"unknown_threshold_id={threshold_id}")
                continue
            for field in (
                "metric_id",
                "metric_version",
                "metric_units",
                "comparison_operator",
                "threshold_value",
                "authority_hash",
            ):
                if record.get(field) != entry.get(field):
                    errors.append(
                        f"{row.get('review_key')}:{threshold_id}:"
                        f"{field}_mismatch"
                    )
            try:
                result = _compare(
                    float(record["observed_feature_value"]),
                    str(record["comparison_operator"]),
                    float(record["threshold_value"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"{row.get('review_key')}:{threshold_id}:"
                    f"invalid_comparison={type(exc).__name__}"
                )
                continue
            if not result:
                errors.append(
                    f"{row.get('review_key')}:{threshold_id}:"
                    "comparison_false"
                )
    if missing_authority_rows:
        errors.append(
            "threshold_candidates_without_authority="
            f"{missing_authority_rows}"
        )
    if availability_only_rows:
        errors.append(
            f"evidence_availability_only_candidates={availability_only_rows}"
        )
    return {
        "schema_version": (
            "classification_v2.independent_behavior_threshold_audit.v1"
        ),
        "registry_hash": expected_hash,
        "universe_count": len(universe),
        "candidate_count": len(candidates),
        "auto_carry_count": len(auto_carry),
        "candidate_auto_carry_overlap": len(overlap),
        "missing_universe_keys": len(missing),
        "extra_partition_keys": len(extra),
        "duplicate_review_keys": duplicate_keys,
        "threshold_candidate_rows": threshold_candidate_rows,
        "checked_threshold_comparisons": checked,
        "threshold_candidates_without_authority": missing_authority_rows,
        "evidence_availability_only_candidates": availability_only_rows,
        "errors": errors,
        "valid": not errors,
    }


def threshold_sensitivity_analysis(
    units: pd.DataFrame,
    registry_snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Vary one frozen comparison at a time without changing the registry."""

    key_column = (
        "review_key" if "review_key" in units else "review_unit_id"
    )
    baseline_mask = units["include_in_review"].astype(bool)
    baseline_keys = set(units.loc[baseline_mask, key_column].astype(str))
    unit_index = units.set_index(key_column, drop=False)
    decision_map = {
        str(row[key_column]): _records(
            row.get("threshold_binding_details", "[]")
        )
        for _, row in units.iterrows()
    }
    evaluation_rows: list[tuple[str, dict[str, Any], str]] = []
    for _, row in units.iterrows():
        key = str(row[key_column])
        for record in _records(
            row.get("review_threshold_evaluations", "[]")
        ):
            evaluation_rows.append((key, record, str(row.get("behavior", ""))))

    rows: list[dict[str, Any]] = []
    for entry in registry_snapshot["thresholds"]:
        if bool(entry.get("deprecated", False)):
            continue
        threshold_id = str(entry["threshold_id"])
        frozen = float(entry["threshold_value"])
        tested_values = sorted({frozen * 0.9, frozen, frozen * 1.1})
        matching = [
            (key, record)
            for key, record, _behavior in evaluation_rows
            if record.get("threshold_id") == threshold_id
        ]
        fixed_keys = {
            key
            for key in baseline_keys
            if (
                not any(
                    record.get("threshold_id") == threshold_id
                    for record in decision_map.get(key, [])
                )
                or len(decision_map.get(key, [])) > 1
            )
        }
        for tested in tested_values:
            if entry["comparison_operator"] == "SCALE_BY_REFERENCE":
                variant_keys = set(baseline_keys)
                diagnostic_status = (
                    "REFERENCE_REQUIRES_FULL_RESCORING;"
                    "FROZEN_MEMBERSHIP_REPORTED_NO_AUTO_CHANGE"
                )
            else:
                true_keys = {
                    key
                    for key, record in matching
                    if _compare(
                        float(record["observed_feature_value"]),
                        str(record["comparison_operator"]),
                        tested,
                    )
                }
                variant_keys = fixed_keys | true_keys
                diagnostic_status = "DIRECT_COMPARISON_SENSITIVITY"
            added = sorted(variant_keys - baseline_keys)
            removed = sorted(baseline_keys - variant_keys)
            subset = unit_index.loc[
                unit_index.index.intersection(variant_keys)
            ]
            rows.append(
                {
                    "threshold_id": threshold_id,
                    "tested_value": tested,
                    "frozen_value": frozen,
                    "metric_id": entry["metric_id"],
                    "metric_version": entry["metric_version"],
                    "affected_predicate": entry["predicate_scope"],
                    "candidate_count": len(variant_keys),
                    "added_key_count": len(added),
                    "removed_key_count": len(removed),
                    "added_keys": json.dumps(added, separators=(",", ":")),
                    "removed_keys": json.dumps(
                        removed,
                        separators=(",", ":"),
                    ),
                    "behavior_distribution": json.dumps(
                        _counts(subset, "behavior"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "source_distribution": json.dumps(
                        _counts(subset, "source"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "calendar_date_distribution": json.dumps(
                        _counts(subset, "recording_date"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "diagnostic_status": diagnostic_status,
                    "registry_modified": False,
                }
            )
    return pd.DataFrame(rows)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    counts = frame[column].fillna("").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


__all__ = [
    "independent_threshold_candidate_audit",
    "threshold_sensitivity_analysis",
]
