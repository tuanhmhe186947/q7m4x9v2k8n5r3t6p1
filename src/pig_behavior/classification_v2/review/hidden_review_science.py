"""Scientific uncertainty and promotion gate for two-sided Hidden review.

Population prevalence is estimated only from the probability-sampled
``hidden_no_random_audit`` cohort. The targeted high-risk cohort is reported as
an enrichment yield and can never be interpreted as population prevalence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    audit_hidden_decision_coverage,
)

POLICY_SCHEMA_VERSION = "classification_v2.hidden_scientific_policy.v1"
DESIGN_SCHEMA_VERSION = "classification_v2.hidden_scientific_design.v1"
GATE_SCHEMA_VERSION = "classification_v2.hidden_scientific_gate.v2"

RANDOM_COHORT = "hidden_no_random_audit"
HIGH_RISK_COHORT = "hidden_no_high_risk"
TARGET_FIELD_TOKENS = (
    "behavior",
    "label",
    "target",
    "manual_",
    "review_status",
    "review_decision",
    "after_review",
)


@dataclass(frozen=True)
class HiddenScientificPolicy:
    """Predeclared support, uncertainty, and quality thresholds."""

    confidence_level: float = 0.95
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260714
    random_false_negative_upper_threshold: float = 0.05
    high_risk_yield_upper_threshold: float = 0.10
    min_random_reviewed_items: int = 100
    min_random_native_clusters: int = 50
    min_random_recording_clusters: int = 5
    min_high_risk_reviewed_items: int = 100
    min_high_risk_native_clusters: int = 50
    min_high_risk_recording_clusters: int = 5

    def validate(self) -> None:
        """Reject weak or malformed policy values before review starts."""

        if not 0.80 <= self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in [0.80, 1.0)")
        if self.bootstrap_iterations < 200:
            raise ValueError("bootstrap_iterations must be >= 200")
        for name in (
            "random_false_negative_upper_threshold",
            "high_risk_yield_upper_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "min_random_reviewed_items",
            "min_random_native_clusters",
            "min_random_recording_clusters",
            "min_high_risk_reviewed_items",
            "min_high_risk_native_clusters",
            "min_high_risk_recording_clusters",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")

    def to_payload(self) -> dict[str, Any]:
        """Serialize policy fields with an explicit schema identifier."""

        return {"schema_version": POLICY_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HiddenScientificPolicy:
        """Load only declared policy fields and reject schema ambiguity."""

        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("invalid Hidden scientific policy schema")
        field_names = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload).difference(field_names | {"schema_version"}))
        if unknown:
            raise ValueError(f"unknown Hidden scientific policy fields: {unknown}")
        values = {name: payload[name] for name in field_names if name in payload}
        policy = cls(**values)
        policy.validate()
        return policy


def load_hidden_scientific_policy(
    path: Path,
) -> tuple[HiddenScientificPolicy, dict[str, Any], str]:
    """Load and hash the immutable policy used to build a review design."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = HiddenScientificPolicy.from_payload(payload)
    return policy, payload, sha256_file(path)


def build_hidden_scientific_design(
    manifest: pd.DataFrame,
    *,
    manifest_sha256: str,
    policy_payload: dict[str, Any],
    policy_sha256: str,
    selection_contract: dict[str, Any],
    require_final_support: bool = True,
) -> dict[str, Any]:
    """Bind a target-independent manifest to thresholds before review."""

    policy = HiddenScientificPolicy.from_payload(policy_payload)
    _require_columns(
        manifest,
        [
            "hidden_review_item_id",
            "hidden_review_status",
            "hidden_after_review",
            "hidden_review_cohort",
            "hidden_sampling_stratum",
            "temporal_unit_key",
        ],
        "manifest",
    )
    if manifest["hidden_review_item_id"].duplicated().any():
        raise ValueError("manifest has duplicate Hidden review item IDs")
    status = manifest["hidden_review_status"].fillna("").astype(str).str.lower()
    after = manifest["hidden_after_review"].fillna("").astype(str).str.strip()
    resolved = status.isin(["reviewed", "resolved", "complete"])
    if resolved.any() or after.ne("").any():
        raise ValueError("scientific design must be written before decisions")
    selection_errors = _target_independence_errors(
        manifest,
        selection_contract,
    )
    if selection_errors:
        raise ValueError(
            "Hidden scientific design is target-derived: "
            f"{selection_errors}"
        )
    planned_support = _planned_support(manifest, policy)
    if require_final_support and planned_support["failures"]:
        raise ValueError(
            "Hidden review design has insufficient planned support: "
            f"{planned_support['failures']}"
        )

    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_status": "PREDECLARED_WITH_PENDING_MANIFEST",
        "manifest_sha256": manifest_sha256,
        "manifest_rows": int(len(manifest)),
        "manifest_unique_items": int(manifest["hidden_review_item_id"].nunique()),
        "resolved_items_at_declaration": int(resolved.sum()),
        "all_items_pending_at_declaration": True,
        "design_scope": "full" if require_final_support else "smoke",
        "planned_support": planned_support,
        "planned_support_meets_final_gate": not planned_support["failures"],
        "policy_sha256": policy_sha256,
        "policy": policy.to_payload(),
        "selection_contract": selection_contract,
        "prevalence_estimator": (
            "hajek_inverse_probability_weighted_random_hidden_no"
        ),
        "uncertainty_method": (
            "source_stratified_recording_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
        "recording_cluster_fields": [
            "source_type",
            "hidden_review_stratum_key",
        ],
        "native_cluster_field": "temporal_unit_key",
        "high_risk_estimand": "targeted_enrichment_correction_yield",
        "high_risk_is_population_prevalence": False,
    }


def evaluate_hidden_scientific_gate(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    design: dict[str, Any],
    *,
    manifest_sha256: str,
    design_sha256: str,
) -> dict[str, Any]:
    """Evaluate coverage, clustered uncertainty, and locked thresholds."""

    errors = _design_errors(manifest, design, manifest_sha256)
    try:
        policy = HiddenScientificPolicy.from_payload(design.get("policy", {}))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid_scientific_policy:{exc}")
        policy = HiddenScientificPolicy()

    selection_errors = _target_independence_errors(
        manifest,
        design.get("selection_contract", {}),
    )
    errors.extend(selection_errors)
    decision_contract = audit_hidden_decision_coverage(
        manifest,
        decisions,
        require_resolved=True,
    )
    errors.extend(_scientific_decision_contract_errors(decision_contract))
    joined, coverage = _join_review_decisions(manifest, decisions)
    errors.extend(coverage["errors"])
    blockers = list(coverage["blockers"])
    if design.get("design_scope") != "full":
        blockers.append("hidden_scientific_design_scope_not_full")
    if design.get("planned_support_meets_final_gate") is not True:
        blockers.append("hidden_scientific_planned_support_not_met")

    random_rows = joined.loc[
        joined["hidden_review_cohort"].eq(RANDOM_COHORT)
        & joined["hidden_before_review"].eq("No")
        & joined["decision_resolved"]
    ].copy()
    high_risk_rows = joined.loc[
        joined["hidden_review_cohort"].eq(HIGH_RISK_COHORT)
        & joined["hidden_before_review"].eq("No")
        & joined["decision_resolved"]
    ].copy()

    random_stats, random_errors = _cohort_statistics(
        random_rows,
        policy,
        use_sampling_weights=True,
    )
    high_risk_stats, high_risk_errors = _cohort_statistics(
        high_risk_rows,
        policy,
        use_sampling_weights=False,
    )
    errors.extend(f"random_cohort:{error}" for error in random_errors)
    errors.extend(f"high_risk_cohort:{error}" for error in high_risk_errors)

    support_failures = _support_failures(
        random_stats,
        high_risk_stats,
        policy,
    )
    if support_failures:
        blockers.extend(support_failures)

    threshold_failures: list[str] = []
    if not errors and not blockers:
        random_upper = random_stats["conservative_interval"][1]
        high_risk_upper = high_risk_stats["conservative_interval"][1]
        if random_upper > policy.random_false_negative_upper_threshold:
            threshold_failures.append(
                "random_false_negative_upper_exceeds_threshold="
                f"{random_upper:.12g}>"
                f"{policy.random_false_negative_upper_threshold:.12g}"
            )
        if high_risk_upper > policy.high_risk_yield_upper_threshold:
            threshold_failures.append(
                "high_risk_yield_upper_exceeds_threshold="
                f"{high_risk_upper:.12g}>"
                f"{policy.high_risk_yield_upper_threshold:.12g}"
            )

    if errors:
        status = "FAIL_CONTRACT"
    elif blockers:
        status = "BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW"
    elif threshold_failures:
        status = "FAIL_QUALITY_THRESHOLD"
    else:
        status = "PASS"

    final_estimate = status in {"PASS", "FAIL_QUALITY_THRESHOLD"}
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": status,
        "valid": status == "PASS",
        "training_snapshot_allowed": status == "PASS",
        "target_independent_selection": not selection_errors,
        "complete_review_coverage": (
            coverage["complete"] and not decision_contract["errors"]
        ),
        "manifest_sha256": manifest_sha256,
        "design_sha256": design_sha256,
        "policy_sha256": design.get("policy_sha256"),
        "policy": policy.to_payload(),
        "coverage": {
            **coverage,
            "decision_contract": decision_contract,
            "decision_metadata_drift_counts": decision_contract[
                "decision_metadata_drift_counts"
            ],
            "decision_metadata_drift_unique_items": decision_contract[
                "decision_metadata_drift_unique_items"
            ],
            "metadata_drift_policy": decision_contract[
                "metadata_drift_policy"
            ],
            "metadata_drift_policy_version": decision_contract[
                "metadata_drift_policy_version"
            ],
        },
        "decision_metadata_drift_counts": decision_contract[
            "decision_metadata_drift_counts"
        ],
        "decision_metadata_drift_unique_items": decision_contract[
            "decision_metadata_drift_unique_items"
        ],
        "random_hidden_no_prevalence": {
            **random_stats,
            "final_estimate": final_estimate,
            "is_population_prevalence": True,
            "upper_threshold": policy.random_false_negative_upper_threshold,
        },
        "high_risk_correction_yield": {
            **high_risk_stats,
            "final_estimate": final_estimate,
            "is_population_prevalence": False,
            "upper_threshold": policy.high_risk_yield_upper_threshold,
        },
        "errors": errors,
        "blockers": blockers,
        "threshold_failures": threshold_failures,
        "warnings": [
            "High-risk correction yield is enrichment evidence, not prevalence.",
            *decision_contract["warnings"],
        ],
    }


def _scientific_decision_contract_errors(
    decision_contract: dict[str, Any],
) -> list[str]:
    """Preserve incomplete-review blockers while failing contract drift."""
    blocker_prefixes = (
        "missing_decision_items=",
        "unclear_decision_items=",
        "pending_decision_items=",
    )
    return [
        error
        for error in decision_contract["errors"]
        if not str(error).startswith(blocker_prefixes)
    ]


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _design_errors(
    manifest: pd.DataFrame,
    design: dict[str, Any],
    manifest_sha256: str,
) -> list[str]:
    """Validate that review results use the exact predeclared design."""

    errors: list[str] = []
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        errors.append("invalid_hidden_scientific_design_schema")
    if design.get("design_status") != "PREDECLARED_WITH_PENDING_MANIFEST":
        errors.append("hidden_scientific_design_not_predeclared")
    if design.get("manifest_sha256") != manifest_sha256:
        errors.append("hidden_manifest_hash_drift")
    if int(design.get("manifest_rows", -1)) != len(manifest):
        errors.append("hidden_manifest_row_count_drift")
    unique_items = manifest.get(
        "hidden_review_item_id",
        pd.Series(dtype="object"),
    ).nunique()
    if int(design.get("manifest_unique_items", -1)) != unique_items:
        errors.append("hidden_manifest_unique_item_count_drift")
    if design.get("resolved_items_at_declaration") != 0:
        errors.append("hidden_design_declared_after_resolved_decisions")
    if design.get("all_items_pending_at_declaration") is not True:
        errors.append("hidden_design_did_not_bind_pending_manifest")
    if design.get("high_risk_is_population_prevalence") is not False:
        errors.append("high_risk_cohort_misdeclared_as_prevalence")
    return errors


def _target_independence_errors(
    manifest: pd.DataFrame,
    selection_contract: dict[str, Any],
) -> list[str]:
    """Reject target-derived fields or target markers in sampling metadata."""

    errors: list[str] = []
    if selection_contract.get("target_independent") is not True:
        errors.append("hidden_selection_contract_not_target_independent")
    declared_fields = [
        *selection_contract.get("risk_input_columns", []),
        *selection_contract.get("stratum_columns", []),
    ]
    target_fields = sorted(
        field
        for field in declared_fields
        if any(token in str(field).lower() for token in TARGET_FIELD_TOKENS)
    )
    if target_fields:
        errors.append(f"target_derived_hidden_selection_fields={target_fields}")
    required = [
        "hidden_sampling_stratum",
        "hidden_false_negative_risk_reasons",
        "hidden_false_negative_risk_band",
    ]
    missing = sorted(set(required).difference(manifest.columns))
    if missing:
        errors.append(f"missing_hidden_selection_metadata={missing}")
        return errors
    strata = manifest["hidden_sampling_stratum"].fillna("").astype(str).str.lower()
    target_rows = strata.map(
        lambda value: any(f"{token}=" in value for token in TARGET_FIELD_TOKENS)
    )
    if target_rows.any():
        errors.append(f"target_markers_in_hidden_strata={int(target_rows.sum())}")
    reasons = manifest[
        "hidden_false_negative_risk_reasons"
    ].fillna("").astype(str)
    informed = reasons.str.contains("interaction_scene", regex=False)
    if informed.any():
        errors.append(f"target_informed_hidden_risk_rows={int(informed.sum())}")
    return errors


def _join_review_decisions(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one exact manifest-to-decision row and audit complete coverage."""

    required_manifest = [
        "hidden_review_item_id",
        "hidden_before_review",
        "hidden_review_cohort",
        "source_type",
        "hidden_review_stratum_key",
        "temporal_unit_key",
        "hidden_sampling_probability",
        "hidden_sampling_weight",
    ]
    required_decisions = [
        "hidden_review_item_id",
        "hidden_before_review",
        "hidden_after_review",
        "hidden_review_status",
    ]
    _require_columns(manifest, required_manifest, "manifest")
    _require_columns(decisions, required_decisions, "decisions")
    errors: list[str] = []
    blockers: list[str] = []

    left = manifest[required_manifest].copy()
    right = decisions[required_decisions].copy()
    left["hidden_review_item_id"] = left["hidden_review_item_id"].astype(str)
    right["hidden_review_item_id"] = right["hidden_review_item_id"].astype(str)
    if left["hidden_review_item_id"].duplicated().any():
        errors.append("duplicate_manifest_items")
    if right["hidden_review_item_id"].duplicated().any():
        errors.append("duplicate_decision_items")

    joined = left.merge(
        right,
        on="hidden_review_item_id",
        how="outer",
        indicator=True,
        suffixes=("", "_decision"),
        validate="one_to_one" if not errors else None,
    )
    missing = int(joined["_merge"].eq("left_only").sum())
    unknown = int(joined["_merge"].eq("right_only").sum())
    if unknown:
        errors.append(f"unknown_decision_items={unknown}")
    if missing:
        blockers.append(f"missing_decision_items={missing}")

    status = joined.get(
        "hidden_review_status",
        pd.Series("", index=joined.index),
    ).fillna("").astype(str).str.strip().str.lower()
    status = status.replace({"complete": "reviewed", "resolved": "reviewed"})
    after = joined.get(
        "hidden_after_review",
        pd.Series("", index=joined.index),
    ).fillna("").astype(str).str.strip().str.lower()
    after = after.replace({"yes": "Yes", "no": "No"})
    joined["hidden_before_review"] = joined[
        "hidden_before_review"
    ].map(_normalize_hidden)
    decision_before = joined.get(
        "hidden_before_review_decision",
        pd.Series("", index=joined.index),
    ).map(_normalize_hidden)
    matched = joined["_merge"].eq("both")
    before_mismatch = matched & decision_before.ne(joined["hidden_before_review"])
    if before_mismatch.any():
        errors.append(
            "decision_hidden_before_mismatch="
            f"{int(before_mismatch.sum())}"
        )
    resolved = matched & status.eq("reviewed")
    invalid_after = resolved & ~after.isin(["Yes", "No"])
    if invalid_after.any():
        errors.append(f"invalid_resolved_hidden_values={int(invalid_after.sum())}")
    unresolved = matched & ~status.eq("reviewed")
    if unresolved.any():
        blockers.append(f"unresolved_decision_items={int(unresolved.sum())}")
    joined["hidden_after_review"] = after
    joined["decision_resolved"] = resolved & ~invalid_after
    joined["hidden_false_negative"] = (
        joined["decision_resolved"]
        & joined["hidden_before_review"].eq("No")
        & joined["hidden_after_review"].eq("Yes")
    )

    coverage = {
        "manifest_items": int(len(left)),
        "decision_items": int(len(right)),
        "matched_items": int(matched.sum()),
        "missing_items": missing,
        "unknown_items": unknown,
        "resolved_items": int(joined["decision_resolved"].sum()),
        "unresolved_items": int(unresolved.sum()),
        "complete": not errors and not blockers,
        "errors": errors,
        "blockers": blockers,
    }
    return joined, coverage


def _cohort_statistics(
    rows: pd.DataFrame,
    policy: HiddenScientificPolicy,
    *,
    use_sampling_weights: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Compute a weighted rate with two explicit correlation safeguards."""

    errors: list[str] = []
    if rows.empty:
        return _empty_statistics(policy), errors
    recording_key = (
        rows["source_type"].fillna("").astype(str)
        + "|"
        + rows["hidden_review_stratum_key"].fillna("").astype(str)
    )
    source_key = rows["source_type"].fillna("").astype(str).str.strip()
    native_key = rows["temporal_unit_key"].fillna("").astype(str).str.strip()
    blank_source = source_key.eq("")
    blank_recording = recording_key.str.replace("|", "", regex=False).eq("")
    if blank_source.any():
        errors.append(f"blank_source_type={int(blank_source.sum())}")
    if blank_recording.any():
        errors.append(f"blank_recording_cluster={int(blank_recording.sum())}")
    if native_key.eq("").any():
        errors.append(f"blank_native_cluster={int(native_key.eq('').sum())}")

    outcome = rows["hidden_false_negative"].astype(float)
    if use_sampling_weights:
        probability = pd.to_numeric(
            rows["hidden_sampling_probability"],
            errors="coerce",
        )
        weight = pd.to_numeric(
            rows["hidden_sampling_weight"],
            errors="coerce",
        )
        invalid_probability = probability.isna() | ~probability.between(
            0.0,
            1.0,
            inclusive="right",
        )
        invalid_weight = weight.isna() | weight.le(0.0)
        reciprocal_error = weight.mul(probability).sub(1.0).abs().gt(1e-8)
        if invalid_probability.any():
            errors.append(
                "invalid_sampling_probability="
                f"{int(invalid_probability.sum())}"
            )
        if invalid_weight.any():
            errors.append(f"invalid_sampling_weight={int(invalid_weight.sum())}")
        if reciprocal_error.any():
            errors.append(
                "sampling_probability_weight_mismatch="
                f"{int(reciprocal_error.sum())}"
            )
    else:
        weight = pd.Series(1.0, index=rows.index)

    valid = (
        weight.notna()
        & weight.gt(0.0)
        & ~blank_source
        & ~blank_recording
        & native_key.ne("")
    )
    outcome = outcome.loc[valid]
    weight = weight.loc[valid]
    recording_key = recording_key.loc[valid]
    native_key = native_key.loc[valid]
    source_key = source_key.loc[valid]
    denominator = float(weight.sum())
    numerator = float(weight.mul(outcome).sum())
    rate = numerator / denominator if denominator > 0 else None
    unweighted_rate = float(outcome.mean()) if len(outcome) else None

    bootstrap_interval = _recording_cluster_bootstrap_interval(
        outcome,
        weight,
        recording_key,
        source_key,
        policy,
    )
    native_effective_n = _native_cluster_effective_n(weight, native_key)
    wilson_interval = _weighted_wilson_interval(
        rate,
        native_effective_n,
        policy.confidence_level,
    )
    conservative = _interval_envelope(
        bootstrap_interval,
        wilson_interval,
    )
    return {
        "reviewed_items": int(len(outcome)),
        "corrected_to_hidden_yes": int(outcome.sum()),
        "unweighted_rate": unweighted_rate,
        "weighted_rate": rate,
        "weight_sum": denominator,
        "native_cluster_count": int(native_key.nunique()),
        "recording_cluster_count": int(recording_key.nunique()),
        "native_cluster_kish_effective_n": native_effective_n,
        "recording_cluster_bootstrap_interval": bootstrap_interval,
        "native_cluster_kish_wilson_interval": wilson_interval,
        "conservative_interval": conservative,
        "confidence_level": policy.confidence_level,
        "bootstrap_iterations": policy.bootstrap_iterations,
        "bootstrap_seed": policy.bootstrap_seed,
        "uncertainty_method": (
            "source_stratified_recording_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
    }, errors


def _recording_cluster_bootstrap_interval(
    outcome: pd.Series,
    weight: pd.Series,
    cluster: pd.Series,
    source: pd.Series,
    policy: HiddenScientificPolicy,
) -> list[float | None]:
    """Resample whole recordings within source-design strata."""

    frame = pd.DataFrame(
        {
            "source": source.astype(str),
            "cluster": cluster.astype(str),
            "numerator": weight.mul(outcome),
            "denominator": weight,
        }
    )
    totals = frame.groupby(["source", "cluster"], sort=True)[
        ["numerator", "denominator"]
    ].sum()
    if len(totals) < 2:
        return [None, None]
    source_totals = {
        source_name: group.to_numpy(dtype=float)
        for source_name, group in totals.groupby(level="source", sort=True)
    }
    rng = np.random.default_rng(policy.bootstrap_seed)
    estimates = np.empty(policy.bootstrap_iterations, dtype=float)
    for index in range(policy.bootstrap_iterations):
        sums = np.zeros(2, dtype=float)
        for values in source_totals.values():
            sampled = rng.integers(0, len(values), size=len(values))
            sums += values[sampled].sum(axis=0)
        estimates[index] = sums[0] / sums[1]
    alpha = (1.0 - policy.confidence_level) / 2.0
    return [
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    ]


def _native_cluster_effective_n(
    weight: pd.Series,
    native_key: pd.Series,
) -> float | None:
    """Use native-unit weight mass to avoid frame-level pseudo-replication."""

    totals = pd.DataFrame(
        {"native_key": native_key.astype(str), "weight": weight}
    ).groupby("native_key", sort=False)["weight"].sum()
    squared = float(totals.pow(2).sum())
    if squared <= 0:
        return None
    return float(totals.sum() ** 2 / squared)


def _weighted_wilson_interval(
    rate: float | None,
    effective_n: float | None,
    confidence_level: float,
) -> list[float | None]:
    """Provide non-degenerate uncertainty when no correction is observed."""

    if rate is None or effective_n is None or effective_n <= 0:
        return [None, None]
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z**2 / effective_n
    center = (rate + z**2 / (2.0 * effective_n)) / denominator
    margin = (
        z
        * np.sqrt(
            rate * (1.0 - rate) / effective_n
            + z**2 / (4.0 * effective_n**2)
        )
        / denominator
    )
    return [max(0.0, float(center - margin)), min(1.0, float(center + margin))]


def _interval_envelope(
    first: list[float | None],
    second: list[float | None],
) -> list[float | None]:
    """Use the conservative envelope of recording and native uncertainty."""

    lows = [value for value in (first[0], second[0]) if value is not None]
    highs = [value for value in (first[1], second[1]) if value is not None]
    return [min(lows) if lows else None, max(highs) if highs else None]


def _support_failures(
    random_stats: dict[str, Any],
    high_risk_stats: dict[str, Any],
    policy: HiddenScientificPolicy,
) -> list[str]:
    """Apply predeclared minimum support before reading quality thresholds."""

    failures: list[str] = []
    requirements = (
        (
            "random_reviewed_items",
            random_stats["reviewed_items"],
            policy.min_random_reviewed_items,
        ),
        (
            "random_native_clusters",
            random_stats["native_cluster_count"],
            policy.min_random_native_clusters,
        ),
        (
            "random_recording_clusters",
            random_stats["recording_cluster_count"],
            policy.min_random_recording_clusters,
        ),
        (
            "high_risk_reviewed_items",
            high_risk_stats["reviewed_items"],
            policy.min_high_risk_reviewed_items,
        ),
        (
            "high_risk_native_clusters",
            high_risk_stats["native_cluster_count"],
            policy.min_high_risk_native_clusters,
        ),
        (
            "high_risk_recording_clusters",
            high_risk_stats["recording_cluster_count"],
            policy.min_high_risk_recording_clusters,
        ),
    )
    for name, actual, minimum in requirements:
        if actual < minimum:
            failures.append(f"insufficient_{name}={actual}<{minimum}")
    return failures


def _planned_support(
    manifest: pd.DataFrame,
    policy: HiddenScientificPolicy,
) -> dict[str, Any]:
    """Prove that the selected workload can reach final minimum support."""

    def summarize(cohort: str) -> dict[str, int]:
        rows = manifest.loc[
            manifest["hidden_review_cohort"].eq(cohort)
            & manifest["hidden_before_review"].map(_normalize_hidden).eq("No")
        ]
        recording = (
            rows["source_type"].fillna("").astype(str)
            + "|"
            + rows["hidden_review_stratum_key"].fillna("").astype(str)
        )
        native = rows["temporal_unit_key"].fillna("").astype(str).str.strip()
        return {
            "reviewed_items": int(len(rows)),
            "native_cluster_count": int(native.loc[native.ne("")].nunique()),
            "recording_cluster_count": int(recording.nunique()),
        }

    random_stats = summarize(RANDOM_COHORT)
    high_risk_stats = summarize(HIGH_RISK_COHORT)
    return {
        "random_hidden_no": random_stats,
        "high_risk_hidden_no": high_risk_stats,
        "failures": _support_failures(
            random_stats,
            high_risk_stats,
            policy,
        ),
    }


def _empty_statistics(policy: HiddenScientificPolicy) -> dict[str, Any]:
    return {
        "reviewed_items": 0,
        "corrected_to_hidden_yes": 0,
        "unweighted_rate": None,
        "weighted_rate": None,
        "weight_sum": 0.0,
        "native_cluster_count": 0,
        "recording_cluster_count": 0,
        "native_cluster_kish_effective_n": None,
        "recording_cluster_bootstrap_interval": [None, None],
        "native_cluster_kish_wilson_interval": [None, None],
        "conservative_interval": [None, None],
        "confidence_level": policy.confidence_level,
        "bootstrap_iterations": policy.bootstrap_iterations,
        "bootstrap_seed": policy.bootstrap_seed,
        "uncertainty_method": (
            "source_stratified_recording_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
    }


def _normalize_hidden(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in {"yes", "true", "1", "hidden"}:
        return "Yes"
    if text in {"no", "false", "0", "visible"}:
        return "No"
    return ""


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
