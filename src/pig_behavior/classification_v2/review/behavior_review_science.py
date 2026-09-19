"""Scientific design and gate for the behavior-review sampling queue.

The behavior queue is intentionally target-conditioned: the existing behavior
label and Pig-STRENet review evidence can prioritize human review. Therefore
the random cohort estimates only the residual intervention rate after the
declared mandatory and high-risk waves. It is not a prevalence estimate for
the unreviewed population, and the clean-control cohort is never evidence that
unreviewed units are clean.
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

RANDOM_COHORT = "behavior_random_audit"
HIGH_RISK_COHORT = "behavior_high_risk"
CLEAN_CONTROL_COHORT = "behavior_clean_control"
POLICY_SCHEMA_VERSION = "classification_v2.behavior_scientific_policy.v1"
DESIGN_SCHEMA_VERSION = "classification_v2.behavior_scientific_design.v1"
GATE_SCHEMA_VERSION = "classification_v2.behavior_scientific_gate.v1"
RESOLVED_DECISIONS = frozenset({"accept", "corrected", "exclude"})
INTERVENTION_DECISIONS = frozenset({"corrected", "exclude"})


@dataclass(frozen=True, slots=True)
class BehaviorScientificPolicy:
    """Predeclared review support, uncertainty, and quality thresholds."""

    confidence_level: float = 0.95
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260720
    random_intervention_upper_threshold: float = 0.05
    min_random_reviewed_items: int = 100
    min_random_native_clusters: int = 50
    min_random_video_clusters: int = 5
    min_random_source_clusters: int = 1
    min_high_risk_reviewed_items: int = 100
    min_high_risk_native_clusters: int = 50
    min_high_risk_video_clusters: int = 5
    min_high_risk_source_clusters: int = 1

    def validate(self) -> None:
        if not 0.80 <= self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in [0.80, 1.0)")
        if self.bootstrap_iterations < 200:
            raise ValueError("bootstrap_iterations must be >= 200")
        for name in ("random_intervention_upper_threshold",):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "min_random_reviewed_items",
            "min_random_native_clusters",
            "min_random_video_clusters",
            "min_random_source_clusters",
            "min_high_risk_reviewed_items",
            "min_high_risk_native_clusters",
            "min_high_risk_video_clusters",
            "min_high_risk_source_clusters",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": POLICY_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BehaviorScientificPolicy:
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("invalid behavior scientific policy schema")
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload).difference(fields | {"schema_version"}))
        if unknown:
            raise ValueError(f"unknown behavior scientific policy fields: {unknown}")
        policy = cls(**{name: payload[name] for name in fields if name in payload})
        policy.validate()
        return policy


def load_behavior_scientific_policy(
    path: Path,
) -> tuple[BehaviorScientificPolicy, dict[str, Any], str]:
    """Load and hash an immutable policy file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = BehaviorScientificPolicy.from_payload(payload)
    return policy, payload, sha256_file(path)


def build_behavior_scientific_design(
    manifest: pd.DataFrame,
    *,
    manifest_sha256: str,
    policy_payload: dict[str, Any],
    policy_sha256: str,
    selection_contract: dict[str, Any],
    require_final_support: bool = True,
) -> dict[str, Any]:
    """Bind the exact pending queue to a predeclared estimand and policy."""

    policy = BehaviorScientificPolicy.from_payload(policy_payload)
    selected = _selected_manifest(manifest)
    _require_columns(
        selected,
        [
            "review_unit_id",
            "temporal_unit_key",
            "behavior_review_cohort",
            "source_type",
            "video_key",
        ],
        "manifest",
    )
    _validate_native_units(selected)
    if _manifest_has_decisions(selected):
        raise ValueError("scientific design must be written before decisions")
    selection_errors = _selection_contract_errors(selection_contract)
    if selection_errors:
        raise ValueError(
            "behavior scientific design has invalid selection contract: "
            f"{selection_errors}"
        )
    planned_support = _planned_support(selected, policy)
    if require_final_support and planned_support["failures"]:
        raise ValueError(
            "behavior review design has insufficient planned support: "
            f"{planned_support['failures']}"
        )
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_status": "PREDECLARED_WITH_PENDING_MANIFEST",
        "manifest_sha256": manifest_sha256,
        "manifest_rows": int(len(selected)),
        "manifest_unique_items": int(selected["review_unit_id"].nunique()),
        "resolved_items_at_declaration": 0,
        "all_items_pending_at_declaration": True,
        "design_scope": "full" if require_final_support else "smoke",
        "planned_support": planned_support,
        "planned_support_meets_final_gate": not planned_support["failures"],
        "policy_sha256": policy_sha256,
        "policy": policy.to_payload(),
        "selection_contract": selection_contract,
        "estimand": (
            "intervention_rate_in_post_mandatory_high_risk_residual_pool"
        ),
        "random_estimator": "hajek_inverse_probability_weighted_random_residual",
        "uncertainty_method": (
            "source_stratified_source_video_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
        "source_cluster_fields": ["source_type"],
        "video_cluster_fields": ["source_type", "video_key"],
        "native_cluster_field": "temporal_unit_key",
        "high_risk_estimand": "targeted_enrichment_intervention_yield",
        "high_risk_is_population_prevalence": False,
        "clean_control_is_population_prevalence": False,
    }


def evaluate_behavior_scientific_gate(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    design: dict[str, Any],
    *,
    manifest_sha256: str,
    design_sha256: str,
) -> dict[str, Any]:
    """Evaluate exact coverage, weighted residual yield, and uncertainty."""

    errors = _design_errors(manifest, design, manifest_sha256)
    try:
        policy = BehaviorScientificPolicy.from_payload(design.get("policy", {}))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid_scientific_policy:{exc}")
        policy = BehaviorScientificPolicy()

    errors.extend(
        _selection_contract_errors(design.get("selection_contract", {}))
    )
    selected = _selected_manifest(manifest)
    joined, coverage = _join_decisions(selected, decisions)
    errors.extend(coverage["errors"])
    blockers = list(coverage["blockers"])
    if design.get("design_scope") != "full":
        blockers.append("behavior_scientific_design_scope_not_full")
    if design.get("planned_support_meets_final_gate") is not True:
        blockers.append("behavior_scientific_planned_support_not_met")

    random_rows = joined.loc[
        joined["behavior_review_cohort"].eq(RANDOM_COHORT)
        & joined["decision_resolved"]
    ]
    high_rows = joined.loc[
        joined["behavior_review_cohort"].eq(HIGH_RISK_COHORT)
        & joined["decision_resolved"]
    ]
    clean_rows = joined.loc[
        joined["behavior_review_cohort"].eq(CLEAN_CONTROL_COHORT)
        & joined["decision_resolved"]
    ]
    random_stats, random_errors = _cohort_statistics(
        random_rows,
        policy,
        use_sampling_weights=True,
    )
    high_stats, high_errors = _cohort_statistics(
        high_rows,
        policy,
        use_sampling_weights=False,
    )
    clean_stats, clean_errors = _cohort_statistics(
        clean_rows,
        policy,
        use_sampling_weights=False,
    )
    errors.extend(f"random_cohort:{error}" for error in random_errors)
    errors.extend(f"high_risk_cohort:{error}" for error in high_errors)
    errors.extend(f"clean_control:{error}" for error in clean_errors)

    blockers.extend(_support_failures(random_stats, high_stats, policy))
    threshold_failures: list[str] = []
    if not errors and not blockers:
        random_upper = random_stats["conservative_interval"][1]
        if random_upper > policy.random_intervention_upper_threshold:
            threshold_failures.append(
                "random_intervention_upper_exceeds_threshold="
                f"{random_upper:.12g}>{policy.random_intervention_upper_threshold:.12g}"
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
        "complete_review_coverage": coverage["complete"],
        "manifest_sha256": manifest_sha256,
        "design_sha256": design_sha256,
        "policy_sha256": design.get("policy_sha256"),
        "policy": policy.to_payload(),
        "coverage": coverage,
        "random_residual_intervention_rate": {
            **random_stats,
            "final_estimate": final_estimate,
            "is_population_prevalence": False,
            "estimand": (
                "post_mandatory_high_risk_residual_intervention_rate"
            ),
            "upper_threshold": policy.random_intervention_upper_threshold,
        },
        "high_risk_intervention_yield": {
            **high_stats,
            "final_estimate": final_estimate,
            "is_population_prevalence": False,
            "gate_role": "diagnostic_enrichment_yield",
        },
        "clean_control_audit_only": {
            **clean_stats,
            "final_estimate": False,
            "is_population_prevalence": False,
            "interpretation": (
                "control evidence only; does not certify unreviewed units clean"
            ),
        },
        "errors": errors,
        "blockers": blockers,
        "threshold_failures": threshold_failures,
        "warnings": [
            "Behavior random rate is residual, not population prevalence.",
            "High-risk yield is enrichment evidence, not prevalence.",
            "Clean controls do not certify not-selected units as clean.",
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if "include_in_review" in manifest.columns:
        selected = manifest.loc[manifest["include_in_review"].astype(bool)]
    else:
        selected = manifest.loc[
            manifest["behavior_review_cohort"].astype(str).ne("behavior_not_selected")
        ]
    return selected.copy()


def _manifest_has_decisions(manifest: pd.DataFrame) -> bool:
    if "manual_review_decision" not in manifest.columns:
        return False
    values = manifest["manual_review_decision"].fillna("").astype(str).str.strip()
    return (values.ne("") & ~values.eq("pending")).any()


def _validate_native_units(manifest: pd.DataFrame) -> None:
    if manifest["review_unit_id"].astype(str).duplicated().any():
        raise ValueError("behavior manifest has duplicate review_unit_id")
    temporal = manifest["temporal_unit_key"].fillna("").astype(str).str.strip()
    if temporal.eq("").any():
        raise ValueError("behavior manifest has blank temporal_unit_key")
    if temporal.duplicated().any():
        raise ValueError("behavior manifest has duplicate native temporal units")


def _selection_contract_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("target_conditioned_selection") is not True:
        errors.append("behavior_selection_contract_not_target_conditioned")
    estimand = str(contract.get("random_residual_estimand", "")).strip()
    if not estimand:
        errors.append("behavior_random_residual_estimand_missing")
    if contract.get("errors"):
        errors.append("behavior_selection_contract_has_errors")
    return errors


def _design_errors(
    manifest: pd.DataFrame,
    design: dict[str, Any],
    manifest_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        errors.append("invalid_behavior_scientific_design_schema")
    if design.get("design_status") != "PREDECLARED_WITH_PENDING_MANIFEST":
        errors.append("behavior_scientific_design_not_predeclared")
    if design.get("manifest_sha256") != manifest_sha256:
        errors.append("behavior_manifest_hash_drift")
    selected = _selected_manifest(manifest)
    if int(design.get("manifest_rows", -1)) != len(selected):
        errors.append("behavior_manifest_row_count_drift")
    unique_items = selected.get(
        "review_unit_id",
        pd.Series(dtype="object"),
    ).nunique()
    if int(design.get("manifest_unique_items", -1)) != unique_items:
        errors.append("behavior_manifest_unique_item_count_drift")
    if design.get("resolved_items_at_declaration") != 0:
        errors.append("behavior_design_declared_after_resolved_decisions")
    if design.get("all_items_pending_at_declaration") is not True:
        errors.append("behavior_design_did_not_bind_pending_manifest")
    if design.get("high_risk_is_population_prevalence") is not False:
        errors.append("high_risk_cohort_misdeclared_as_prevalence")
    if design.get("clean_control_is_population_prevalence") is not False:
        errors.append("clean_control_misdeclared_as_prevalence")
    return errors


def _join_decisions(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    blockers: list[str] = []
    _require_columns(manifest, ["review_unit_id"], "manifest")
    _require_columns(decisions, ["review_unit_id"], "decisions")
    if decisions["review_unit_id"].astype(str).duplicated().any():
        errors.append("duplicate_behavior_decision_rows")
    expected_ids = manifest["review_unit_id"].astype(str)
    observed_ids = decisions["review_unit_id"].astype(str)
    missing = sorted(set(expected_ids).difference(observed_ids))
    unexpected = sorted(set(observed_ids).difference(expected_ids))
    if missing:
        blockers.append(f"missing_behavior_decision_items={len(missing)}")
    if unexpected:
        errors.append(f"unexpected_behavior_decision_items={len(unexpected)}")
    decision = decisions.copy()
    decision["review_unit_id"] = observed_ids
    decision = decision.drop_duplicates("review_unit_id", keep=False)
    for column, default in (
        ("manual_review_decision", ""),
        ("manual_corrected_behavior", ""),
    ):
        if column not in decision.columns:
            decision[column] = default
    decision["manual_review_decision"] = decision[
        "manual_review_decision"
    ].fillna("").astype(str).str.strip().str.lower()
    decision["manual_corrected_behavior"] = decision[
        "manual_corrected_behavior"
    ].fillna("").astype(str).str.strip().str.lower()
    merged = manifest.merge(
        decision[
            [
                "review_unit_id",
                "manual_review_decision",
                "manual_corrected_behavior",
            ]
        ],
        on="review_unit_id",
        how="left",
        validate="one_to_one",
    )
    for column in (
        "manual_review_decision",
        "manual_corrected_behavior",
    ):
        merged[column] = merged[column].fillna("").astype(str).str.strip().str.lower()
    merged["decision_resolved"] = merged["manual_review_decision"].isin(
        RESOLVED_DECISIONS
    )
    invalid = (
        merged["manual_review_decision"].ne("")
        & ~merged["manual_review_decision"].isin(
            RESOLVED_DECISIONS | {"pending", "review_later"}
        )
    )
    if invalid.any():
        errors.append(f"invalid_behavior_decisions={int(invalid.sum())}")
    unresolved = ~merged["decision_resolved"]
    if unresolved.any():
        blockers.append(f"unresolved_behavior_decision_items={int(unresolved.sum())}")
    intervention = merged["manual_review_decision"].isin(
        INTERVENTION_DECISIONS
    ) | merged["manual_corrected_behavior"].ne("")
    merged["behavior_intervention"] = intervention
    coverage = {
        "selected_review_units": int(len(manifest)),
        "decision_rows": int(len(decisions)),
        "covered_review_units": int(len(set(expected_ids) & set(observed_ids))),
        "missing_review_unit_count": int(len(missing)),
        "unexpected_review_unit_count": int(len(unexpected)),
        "duplicate_decision_rows": int(
            decisions["review_unit_id"].astype(str).duplicated().sum()
        ),
        "unresolved_review_unit_count": int(unresolved.sum()),
        "complete": not errors and not blockers,
        "errors": sorted(set(errors)),
        "blockers": sorted(set(blockers)),
    }
    return merged, coverage


def _cohort_statistics(
    rows: pd.DataFrame,
    policy: BehaviorScientificPolicy,
    *,
    use_sampling_weights: bool,
) -> tuple[dict[str, Any], list[str]]:
    if rows.empty:
        return _empty_statistics(policy), []
    required = [
        "behavior_intervention",
        "source_type",
        "video_key",
        "temporal_unit_key",
    ]
    if use_sampling_weights:
        required.extend(
            [
                "behavior_sampling_probability",
                "behavior_sampling_weight",
            ]
        )
    missing = sorted(set(required).difference(rows.columns))
    if missing:
        return _empty_statistics(policy), [f"missing_columns={missing}"]
    errors: list[str] = []
    source = rows["source_type"].fillna("").astype(str)
    video = rows["video_key"].fillna("").astype(str)
    native = rows["temporal_unit_key"].fillna("").astype(str)
    blank = source.eq("") | video.eq("") | native.eq("")
    if blank.any():
        errors.append(f"blank_cluster_keys={int(blank.sum())}")
    outcome = rows["behavior_intervention"].astype(bool)
    if use_sampling_weights:
        probability = pd.to_numeric(
            rows.get("behavior_sampling_probability"),
            errors="coerce",
        )
        weight = pd.to_numeric(
            rows.get("behavior_sampling_weight"),
            errors="coerce",
        )
        invalid_probability = probability.isna() | ~probability.between(
            0.0,
            1.0,
            inclusive="right",
        )
        invalid_weight = weight.isna() | weight.le(0.0)
        mismatch = weight.mul(probability).sub(1.0).abs().gt(1e-8)
        if invalid_probability.any():
            errors.append(
                f"invalid_sampling_probability={int(invalid_probability.sum())}"
            )
        if invalid_weight.any():
            errors.append(f"invalid_sampling_weight={int(invalid_weight.sum())}")
        if mismatch.any():
            errors.append(
                f"sampling_probability_weight_mismatch={int(mismatch.sum())}"
            )
    else:
        weight = pd.Series(1.0, index=rows.index)
    valid = weight.notna() & weight.gt(0.0) & ~blank
    outcome = outcome.loc[valid]
    weight = weight.loc[valid]
    source = source.loc[valid]
    video = video.loc[valid]
    native = native.loc[valid]
    numerator = float(weight.mul(outcome).sum())
    denominator = float(weight.sum())
    rate = numerator / denominator if denominator else None
    unweighted = float(outcome.mean()) if len(outcome) else None
    bootstrap = _video_cluster_bootstrap_interval(
        outcome,
        weight,
        source,
        video,
        policy,
    )
    effective_n = _native_effective_n(weight, native)
    wilson = _weighted_wilson_interval(
        rate,
        effective_n,
        policy.confidence_level,
    )
    return {
        "reviewed_items": int(len(outcome)),
        "intervention_items": int(outcome.sum()),
        "unweighted_rate": unweighted,
        "weighted_rate": rate,
        "weight_sum": denominator,
        "source_cluster_count": int(source.nunique()),
        "video_cluster_count": int((source + "|" + video).nunique()),
        "native_cluster_count": int(native.nunique()),
        "native_cluster_kish_effective_n": effective_n,
        "source_video_cluster_bootstrap_interval": bootstrap,
        "native_cluster_kish_wilson_interval": wilson,
        "conservative_interval": _interval_envelope(bootstrap, wilson),
        "confidence_level": policy.confidence_level,
        "bootstrap_iterations": policy.bootstrap_iterations,
        "bootstrap_seed": policy.bootstrap_seed,
        "uncertainty_method": (
            "source_stratified_source_video_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
    }, errors


def _video_cluster_bootstrap_interval(
    outcome: pd.Series,
    weight: pd.Series,
    source: pd.Series,
    video: pd.Series,
    policy: BehaviorScientificPolicy,
) -> list[float | None]:
    frame = pd.DataFrame(
        {
            "source": source.astype(str),
            "video": video.astype(str),
            "numerator": weight.mul(outcome),
            "denominator": weight,
        }
    )
    totals = frame.groupby(["source", "video"], sort=True)[
        ["numerator", "denominator"]
    ].sum()
    if len(totals) < 2:
        return [None, None]
    source_totals = {
        name: group.to_numpy(dtype=float)
        for name, group in totals.groupby(level="source", sort=True)
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


def _native_effective_n(weight: pd.Series, native: pd.Series) -> float | None:
    totals = pd.DataFrame(
        {"native": native.astype(str), "weight": weight}
    ).groupby("native", sort=False)["weight"].sum()
    squared = float(totals.pow(2).sum())
    if squared <= 0:
        return None
    return float(totals.sum() ** 2 / squared)


def _weighted_wilson_interval(
    rate: float | None,
    effective_n: float | None,
    confidence_level: float,
) -> list[float | None]:
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
    return [
        max(0.0, float(center - margin)),
        min(1.0, float(center + margin)),
    ]


def _interval_envelope(
    first: list[float | None],
    second: list[float | None],
) -> list[float | None]:
    lows = [value for value in (first[0], second[0]) if value is not None]
    highs = [value for value in (first[1], second[1]) if value is not None]
    return [min(lows) if lows else None, max(highs) if highs else None]


def _support_failures(
    random_stats: dict[str, Any],
    high_stats: dict[str, Any],
    policy: BehaviorScientificPolicy,
) -> list[str]:
    requirements = (
        ("random_reviewed_items", random_stats, "reviewed_items",
         policy.min_random_reviewed_items),
        ("random_native_clusters", random_stats, "native_cluster_count",
         policy.min_random_native_clusters),
        ("random_video_clusters", random_stats, "video_cluster_count",
         policy.min_random_video_clusters),
        ("random_source_clusters", random_stats, "source_cluster_count",
         policy.min_random_source_clusters),
        ("high_risk_reviewed_items", high_stats, "reviewed_items",
         policy.min_high_risk_reviewed_items),
        ("high_risk_native_clusters", high_stats, "native_cluster_count",
         policy.min_high_risk_native_clusters),
        ("high_risk_video_clusters", high_stats, "video_cluster_count",
         policy.min_high_risk_video_clusters),
        ("high_risk_source_clusters", high_stats, "source_cluster_count",
         policy.min_high_risk_source_clusters),
    )
    failures = []
    for name, stats, field, minimum in requirements:
        actual = stats[field]
        if actual < minimum:
            failures.append(f"insufficient_{name}={actual}<{minimum}")
    return failures


def _planned_support(
    manifest: pd.DataFrame,
    policy: BehaviorScientificPolicy,
) -> dict[str, Any]:
    def summarize(cohort: str) -> dict[str, int]:
        rows = manifest.loc[
            manifest["behavior_review_cohort"].eq(cohort)
        ]
        source = rows["source_type"].astype(str)
        video = rows["video_key"].astype(str)
        native = rows["temporal_unit_key"].astype(str)
        return {
            "reviewed_items": int(len(rows)),
            "native_cluster_count": int(native.nunique()),
            "video_cluster_count": int(
                (source + "|" + video).nunique()
            ),
            "source_cluster_count": int(source.nunique()),
        }

    random_stats = summarize(RANDOM_COHORT)
    high_stats = summarize(HIGH_RISK_COHORT)
    return {
        "random_behavior_residual": random_stats,
        "high_risk_behavior": high_stats,
        "failures": _support_failures(random_stats, high_stats, policy),
    }


def _empty_statistics(policy: BehaviorScientificPolicy) -> dict[str, Any]:
    return {
        "reviewed_items": 0,
        "intervention_items": 0,
        "unweighted_rate": None,
        "weighted_rate": None,
        "weight_sum": 0.0,
        "source_cluster_count": 0,
        "video_cluster_count": 0,
        "native_cluster_count": 0,
        "native_cluster_kish_effective_n": None,
        "source_video_cluster_bootstrap_interval": [None, None],
        "native_cluster_kish_wilson_interval": [None, None],
        "conservative_interval": [None, None],
        "confidence_level": policy.confidence_level,
        "bootstrap_iterations": policy.bootstrap_iterations,
        "bootstrap_seed": policy.bootstrap_seed,
        "uncertainty_method": (
            "source_stratified_source_video_cluster_bootstrap_enveloped_by_"
            "native_cluster_kish_wilson"
        ),
    }


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


__all__ = [
    "BehaviorScientificPolicy",
    "build_behavior_scientific_design",
    "evaluate_behavior_scientific_gate",
    "load_behavior_scientific_policy",
    "sha256_file",
]
