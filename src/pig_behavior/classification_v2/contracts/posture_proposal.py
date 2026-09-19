"""Fail-closed validation policy for burst-posture model proposals.

The proposer may reduce review work, but its confidence is not label authority.
Only an explicitly audited stratum can become ``AUTO_VALIDATED``. This module
contains no RGB model fitting and cannot read active review data.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.behavior_posture import (
    POSTURE_LABEL_SET,
)

PROPOSAL_KEY = "native_temporal_unit_key"
AUTO_POSTURE = "upright"


@dataclass(frozen=True, slots=True)
class PostureAutoValidationPolicy:
    """Predeclared gate for one posture-proposal audit."""

    confidence_threshold: float
    minimum_audit_rows_per_stratum: int
    required_precision_lower_bound: float = 0.98
    one_sided_z: float = 1.6448536269514722

    def validate(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if self.minimum_audit_rows_per_stratum <= 0:
            raise ValueError("minimum_audit_rows_per_stratum must be positive")
        if not 0.0 < self.required_precision_lower_bound <= 1.0:
            raise ValueError("required_precision_lower_bound must be in (0, 1]")
        if self.one_sided_z <= 0.0:
            raise ValueError("one_sided_z must be positive")


@dataclass(frozen=True, slots=True)
class PostureReviewScopePolicy:
    """Predeclared deterministic sampling policy for posture review."""

    confidence_threshold: float
    upright_control_rows_per_stratum: int
    seed: int

    def validate(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if self.upright_control_rows_per_stratum <= 0:
            raise ValueError("upright_control_rows_per_stratum must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def build_posture_review_scope(
    proposals: pd.DataFrame,
    *,
    policy: PostureReviewScopePolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select mandatory cases and random controls without reading decisions."""

    policy.validate()
    required = [
        PROPOSAL_KEY,
        "proposal_stratum",
        "posture_proposed",
        "posture_confidence",
        "posture_temporal_consistent",
        "posture_transition_flag",
    ]
    _require_columns(proposals, required, table="proposals")
    work = proposals.copy()
    _normalize_keys(work, table="proposals")
    _require_unique(work, table="proposals")
    work = work.reset_index(drop=True)
    proposed = work["posture_proposed"].fillna("").astype(str).str.strip().str.lower()
    invalid_proposed = ~proposed.isin(POSTURE_LABEL_SET)
    if invalid_proposed.any():
        raise ValueError(
            "invalid posture proposals: "
            f"{sorted(proposed[invalid_proposed].unique())}"
        )
    confidence = pd.to_numeric(work["posture_confidence"], errors="coerce")
    invalid_confidence = confidence.isna() | confidence.lt(0.0) | confidence.gt(1.0)
    if invalid_confidence.any():
        raise ValueError(
            f"invalid posture confidence rows: {int(invalid_confidence.sum())}"
        )
    consistent = _strict_bool(
        work["posture_temporal_consistent"],
        name="posture_temporal_consistent",
    )
    transition = _strict_bool(
        work["posture_transition_flag"],
        name="posture_transition_flag",
    )
    strata = work["proposal_stratum"].fillna("").astype(str).str.strip()
    if strata.eq("").any():
        raise ValueError("proposal_stratum contains empty values")
    work["posture_proposed"] = proposed
    work["posture_confidence"] = confidence
    work["posture_temporal_consistent"] = consistent
    work["posture_transition_flag"] = transition
    work["proposal_stratum"] = strata

    mandatory = (
        proposed.ne(AUTO_POSTURE)
        | confidence.lt(policy.confidence_threshold)
        | ~consistent
        | transition
    )
    work["posture_review_scope_reason"] = ""
    work.loc[proposed.ne(AUTO_POSTURE), "posture_review_scope_reason"] = (
        "NON_UPRIGHT_REQUIRES_REVIEW"
    )
    work.loc[confidence.lt(policy.confidence_threshold), "posture_review_scope_reason"] = (
        "LOW_CONFIDENCE"
    )
    work.loc[~consistent, "posture_review_scope_reason"] = (
        "TEMPORAL_INCONSISTENCY"
    )
    work.loc[transition, "posture_review_scope_reason"] = "POSTURE_TRANSITION"
    work["posture_control_sampling_probability"] = 0.0
    work.loc[mandatory, "posture_control_sampling_probability"] = 1.0

    control_indices: list[int] = []
    control_population = work.loc[~mandatory].sort_values(
        ["proposal_stratum", PROPOSAL_KEY],
        kind="mergesort",
    )
    control_reports: list[dict[str, Any]] = []
    for stratum, rows in control_population.groupby("proposal_stratum", sort=True):
        selected_count = min(policy.upright_control_rows_per_stratum, len(rows))
        random_state = _stable_stratum_seed(policy.seed, str(stratum))
        selected = rows.sample(n=selected_count, random_state=random_state)
        control_indices.extend(selected.index.astype(int).tolist())
        probability = selected_count / len(rows)
        work.loc[
            selected.index,
            "posture_control_sampling_probability",
        ] = probability
        control_reports.append(
            {
                "proposal_stratum": str(stratum),
                "eligible_upright_rows": int(len(rows)),
                "selected_control_rows": int(selected_count),
                "sampling_probability": probability,
            }
        )

    selected_mask = mandatory.copy()
    if control_indices:
        selected_mask.loc[control_indices] = True
        work.loc[control_indices, "posture_review_scope_reason"] = (
            "AUTO_UPRIGHT_RANDOM_CONTROL"
        )
    scope = work.loc[selected_mask].copy()
    scope = scope.sort_values(
        ["proposal_stratum", "posture_review_scope_reason", PROPOSAL_KEY],
        kind="mergesort",
    ).reset_index(drop=True)
    scope.insert(
        0,
        "posture_review_item_id",
        [f"posture_review_{index:07d}" for index in range(1, len(scope) + 1)],
    )
    audit = {
        "schema_version": "classification_v2_posture_review_scope_audit.v1",
        "population_rows": int(len(work)),
        "mandatory_rows": int(mandatory.sum()),
        "upright_control_rows": int(len(control_indices)),
        "scope_rows": int(len(scope)),
        "seed": policy.seed,
        "confidence_threshold": policy.confidence_threshold,
        "upright_control_rows_per_stratum": (
            policy.upright_control_rows_per_stratum
        ),
        "control_strata": control_reports,
        "reason_counts": scope["posture_review_scope_reason"].value_counts().to_dict(),
        "errors": [],
    }
    return scope, audit


def evaluate_posture_auto_validation(
    proposals: pd.DataFrame,
    human_audit: pd.DataFrame,
    *,
    policy: PostureAutoValidationPolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gate high-confidence upright proposals by audited stratum precision."""

    policy.validate()
    required_proposal = [
        PROPOSAL_KEY,
        "proposal_stratum",
        "posture_proposed",
        "posture_confidence",
        "posture_temporal_consistent",
        "posture_transition_flag",
    ]
    required_audit = [
        PROPOSAL_KEY,
        "posture_target",
        "posture_valid_mask",
    ]
    _require_columns(proposals, required_proposal, table="proposals")
    _require_columns(human_audit, required_audit, table="human_audit")
    work = proposals[required_proposal].copy()
    audit = human_audit[required_audit].copy()
    _normalize_keys(work, table="proposals")
    _normalize_keys(audit, table="human_audit")
    _require_unique(work, table="proposals")
    _require_unique(audit, table="human_audit")
    extra_audit = sorted(set(audit[PROPOSAL_KEY]).difference(work[PROPOSAL_KEY]))
    if extra_audit:
        raise ValueError(f"human audit keys not present in proposals: {extra_audit[:5]}")

    proposed = work["posture_proposed"].fillna("").astype(str).str.strip().str.lower()
    invalid_proposed = ~proposed.isin(POSTURE_LABEL_SET)
    if invalid_proposed.any():
        raise ValueError(
            "invalid posture proposals: "
            f"{sorted(proposed[invalid_proposed].unique())}"
        )
    work["posture_proposed"] = proposed
    strata = work["proposal_stratum"].fillna("").astype(str).str.strip()
    if strata.eq("").any():
        raise ValueError("proposal_stratum contains empty values")
    work["proposal_stratum"] = strata
    confidence = pd.to_numeric(work["posture_confidence"], errors="coerce")
    invalid_confidence = confidence.isna() | confidence.lt(0.0) | confidence.gt(1.0)
    if invalid_confidence.any():
        raise ValueError(
            f"invalid posture confidence rows: {int(invalid_confidence.sum())}"
        )
    work["posture_confidence"] = confidence
    temporal_consistent = _strict_bool(
        work["posture_temporal_consistent"],
        name="posture_temporal_consistent",
    )
    transition = _strict_bool(
        work["posture_transition_flag"],
        name="posture_transition_flag",
    )
    work["posture_temporal_consistent"] = temporal_consistent
    work["posture_transition_flag"] = transition
    work["auto_candidate"] = (
        proposed.eq(AUTO_POSTURE)
        & confidence.ge(policy.confidence_threshold)
        & temporal_consistent
        & ~transition
    )

    audit_target = audit["posture_target"].fillna("").astype(str).str.strip().str.lower()
    audit_valid = _strict_bool(
        audit["posture_valid_mask"],
        name="posture_valid_mask",
    )
    invalid_audit = audit_valid & ~audit_target.isin(POSTURE_LABEL_SET)
    target_while_invalid = ~audit_valid & audit_target.ne("")
    if invalid_audit.any() or target_while_invalid.any():
        raise ValueError("human posture audit target/mask contract is invalid")
    audit["audited_posture_target"] = audit_target
    audit["audited_posture_valid"] = audit_valid
    audit = audit.drop(columns=["posture_target", "posture_valid_mask"])
    work = work.merge(audit, on=PROPOSAL_KEY, how="left", validate="one_to_one")
    work["audited_posture_valid"] = work["audited_posture_valid"].fillna(False)
    work["proposal_correct"] = (
        work["audited_posture_valid"]
        & work["posture_proposed"].eq(work["audited_posture_target"])
    )

    stratum_reports: list[dict[str, Any]] = []
    passing_strata: set[str] = set()
    for stratum, rows in work.loc[work["auto_candidate"]].groupby(
        "proposal_stratum",
        sort=True,
    ):
        audited = rows.loc[rows["audited_posture_valid"]]
        count = int(len(audited))
        correct = int(audited["proposal_correct"].sum())
        lower = wilson_lower_bound(
            correct,
            count,
            z=policy.one_sided_z,
        )
        passed = (
            count >= policy.minimum_audit_rows_per_stratum
            and lower >= policy.required_precision_lower_bound
        )
        if passed:
            passing_strata.add(str(stratum))
        stratum_reports.append(
            {
                "proposal_stratum": str(stratum),
                "auto_candidate_rows": int(len(rows)),
                "audited_rows": count,
                "correct_rows": correct,
                "observed_precision": (correct / count if count else None),
                "precision_lower_bound": lower,
                "passed": passed,
            }
        )

    passed_stratum = work["proposal_stratum"].astype(str).isin(passing_strata)
    auto_validated = work["auto_candidate"] & passed_stratum
    work["posture_target"] = work["posture_proposed"].where(auto_validated, "")
    work["posture_valid_mask"] = auto_validated
    work["posture_authority"] = auto_validated.map(
        {True: "AUTO_VALIDATED", False: "UNRESOLVED"}
    )
    work["posture_review_required"] = ~auto_validated
    work["posture_review_reason"] = _review_reason(
        work,
        passed_stratum,
        confidence_threshold=policy.confidence_threshold,
    )

    report = {
        "schema_version": "classification_v2_posture_auto_validation_audit.v1",
        "rows": int(len(work)),
        "auto_candidate_rows": int(work["auto_candidate"].sum()),
        "auto_validated_rows": int(auto_validated.sum()),
        "review_required_rows": int((~auto_validated).sum()),
        "policy": {
            "confidence_threshold": policy.confidence_threshold,
            "minimum_audit_rows_per_stratum": (
                policy.minimum_audit_rows_per_stratum
            ),
            "required_precision_lower_bound": (
                policy.required_precision_lower_bound
            ),
            "one_sided_z": policy.one_sided_z,
            "auto_eligible_posture": AUTO_POSTURE,
        },
        "strata": stratum_reports,
        "errors": [],
    }
    return work, report


def wilson_lower_bound(correct: int, total: int, *, z: float) -> float | None:
    """Return a one-sided Wilson lower bound for a binomial proportion."""

    if correct < 0 or total < 0 or correct > total:
        raise ValueError("invalid binomial counts")
    if z <= 0.0:
        raise ValueError("z must be positive")
    if total == 0:
        return None
    proportion = correct / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _stable_stratum_seed(seed: int, stratum: str) -> int:
    """Derive a repeatable pandas-compatible random seed per stratum."""
    payload = f"{seed}\0{stratum}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _review_reason(
    work: pd.DataFrame,
    passed_stratum: pd.Series,
    *,
    confidence_threshold: float,
) -> pd.Series:
    reason = pd.Series("AUDIT_STRATUM_NOT_PASSED", index=work.index, dtype=object)
    reason.loc[work["posture_proposed"].ne(AUTO_POSTURE)] = (
        "NON_UPRIGHT_REQUIRES_REVIEW"
    )
    reason.loc[~work["posture_temporal_consistent"]] = "TEMPORAL_INCONSISTENCY"
    reason.loc[work["posture_transition_flag"]] = "POSTURE_TRANSITION"
    reason.loc[work["posture_confidence"].lt(confidence_threshold)] = (
        "LOW_CONFIDENCE"
    )
    reason.loc[work["posture_valid_mask"]] = "AUTO_VALIDATED"
    reason.loc[work["auto_candidate"] & passed_stratum] = "AUTO_VALIDATED"
    return reason


def _normalize_keys(df: pd.DataFrame, *, table: str) -> None:
    keys = df[PROPOSAL_KEY].fillna("").astype(str).str.strip()
    if keys.eq("").any():
        raise ValueError(f"{table} contains empty {PROPOSAL_KEY}")
    df[PROPOSAL_KEY] = keys


def _require_unique(df: pd.DataFrame, *, table: str) -> None:
    duplicate_count = int(df[PROPOSAL_KEY].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{table} duplicate {PROPOSAL_KEY}: count={duplicate_count}"
        )


def _require_columns(df: pd.DataFrame, columns: list[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{table} missing columns: {missing}")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name} contains null values")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(
            f"{name} contains invalid values: {sorted(normalized[invalid].unique())}"
        )
    return normalized.isin(true_values)


__all__ = [
    "AUTO_POSTURE",
    "PROPOSAL_KEY",
    "PostureAutoValidationPolicy",
    "PostureReviewScopePolicy",
    "build_posture_review_scope",
    "evaluate_posture_auto_validation",
    "wilson_lower_bound",
]
