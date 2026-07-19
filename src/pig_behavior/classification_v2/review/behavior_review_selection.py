"""Deterministic, auditable selection cohorts for behavior review."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

BEHAVIOR_REVIEW_COHORTS: tuple[str, ...] = (
    "behavior_mandatory_census",
    "behavior_high_risk",
    "behavior_random_audit",
    "behavior_clean_control",
    "behavior_not_selected",
)

MANDATORY_BEHAVIORS = frozenset({"fight", "social-nose", "playwithtoy"})
RARE_CENSUS_BEHAVIORS = frozenset({"drink", "eat", "playwithtoy"})


@dataclass(frozen=True, slots=True)
class BehaviorReviewSelectionConfig:
    """Predeclared queue design; selection never changes labels or weights."""

    random_seed: int = 20260720
    random_per_stratum: int = 5
    clean_control_per_stratum: int = 1
    calibrated_high_risk_fraction: float = 0.10
    calibrated_high_risk_max_per_stratum: int = 32
    calibrated_high_risk_min_pool: int = 20
    rare_census_max_per_source_behavior: int = 64
    sampling_sources: tuple[str, ...] = ("cvat_tracking_xml",)
    stratum_columns: tuple[str, ...] = (
        "source_type",
        "video_key",
        "behavior_label",
    )

    def validate(self) -> None:
        if self.random_per_stratum < 0:
            raise ValueError("random_per_stratum must be >= 0")
        if self.clean_control_per_stratum < 0:
            raise ValueError("clean_control_per_stratum must be >= 0")
        if not 0.0 <= self.calibrated_high_risk_fraction <= 1.0:
            raise ValueError("calibrated_high_risk_fraction must be in [0, 1]")
        if self.calibrated_high_risk_max_per_stratum < 0:
            raise ValueError("calibrated high-risk cap must be >= 0")
        if self.calibrated_high_risk_min_pool < 1:
            raise ValueError("calibrated high-risk minimum pool must be positive")
        if self.rare_census_max_per_source_behavior < 0:
            raise ValueError("rare census maximum must be >= 0")


def assign_behavior_review_cohorts(
    units: pd.DataFrame,
    *,
    config: BehaviorReviewSelectionConfig | None = None,
    include_all_retained_legacy_units: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign disjoint cohorts and exact random inclusion probabilities."""

    cfg = config or BehaviorReviewSelectionConfig()
    cfg.validate()
    required = [
        "review_unit_id",
        "source_type",
        "video_key",
        "behavior_label",
        "temporal_consistency_status",
        "review_reason",
        "review_priority",
        "review_unit_type",
    ]
    _require_columns(units, required)
    if units["review_unit_id"].astype(str).duplicated().any():
        raise ValueError("behavior review units contain duplicate review_unit_id")

    out = units.copy()
    out["behavior_review_cohort"] = "behavior_not_selected"
    out["behavior_sampling_design"] = "not_sampled"
    out["behavior_sampling_stratum"] = _strata(out, cfg.stratum_columns)
    out["behavior_sampling_population"] = 0
    out["behavior_sampling_selected"] = 0
    out["behavior_sampling_probability"] = 0.0
    out["behavior_sampling_weight"] = 0.0
    out["behavior_review_residual_estimand"] = False

    source = out["source_type"].fillna("").astype(str)
    behavior = out["behavior_label"].fillna("").astype(str)
    temporal_bad = ~out["temporal_consistency_status"].fillna("").astype(str).eq(
        "stable"
    )
    legacy = out["review_unit_type"].fillna("").astype(str).eq(
        "legacy_burst_16"
    )
    rare = _rare_census_mask(out, cfg)
    mandatory = temporal_bad | behavior.isin(MANDATORY_BEHAVIORS) | rare
    if include_all_retained_legacy_units:
        mandatory |= legacy
    _assign(
        out,
        mandatory,
        cohort="behavior_mandatory_census",
        design="mandatory_census",
        priority_bonus=200.0,
        reason="mandatory_behavior_review",
    )

    inherited_high_risk = (
        out["review_reason"].fillna("").astype(str).ne("")
        & out["behavior_review_cohort"].eq("behavior_not_selected")
    )
    _assign(
        out,
        inherited_high_risk,
        cohort="behavior_high_risk",
        design="rule_based_high_risk",
        priority_bonus=100.0,
        reason="behavior_evidence_high_risk",
    )

    eligible_source = source.isin(cfg.sampling_sources)
    residual = out["behavior_review_cohort"].eq("behavior_not_selected")
    calibrated = _calibrated_high_risk(out, residual & eligible_source, cfg)
    _assign(
        out,
        calibrated,
        cohort="behavior_high_risk",
        design="source_behavior_calibrated_high_risk",
        priority_bonus=100.0,
        reason="source_calibrated_top_risk",
    )

    residual = out["behavior_review_cohort"].eq("behavior_not_selected")
    random_rows = _sample_each_stratum(
        out[residual & eligible_source],
        cfg.stratum_columns,
        cfg.random_per_stratum,
        cfg.random_seed,
        "behavior_random_audit",
    )
    random_mask = out["review_unit_id"].astype(str).isin(
        set(random_rows["review_unit_id"].astype(str))
    )
    _assign_random_metadata(
        out,
        population=out[residual & eligible_source],
        selected_mask=random_mask,
        config=cfg,
    )
    _assign(
        out,
        random_mask,
        cohort="behavior_random_audit",
        design="stratified_random_residual_low_risk",
        priority_bonus=50.0,
        reason="stratified_random_behavior_audit",
        preserve_sampling=True,
    )
    out.loc[random_mask, "behavior_review_residual_estimand"] = True

    residual = out["behavior_review_cohort"].eq("behavior_not_selected")
    clean_rows = _clean_controls(
        out[residual & eligible_source],
        cfg,
    )
    clean_mask = out["review_unit_id"].astype(str).isin(
        set(clean_rows["review_unit_id"].astype(str))
    )
    _assign(
        out,
        clean_mask,
        cohort="behavior_clean_control",
        design="low_risk_clean_control",
        priority_bonus=10.0,
        reason="low_risk_behavior_clean_control",
    )

    out["include_in_review"] = ~out["behavior_review_cohort"].eq(
        "behavior_not_selected"
    )
    audit = audit_behavior_review_selection(out, cfg)
    if audit["errors"]:
        raise ValueError(
            "behavior review selection contract failed: "
            + "; ".join(audit["errors"])
        )
    return out, audit


def audit_behavior_review_selection(
    units: pd.DataFrame,
    config: BehaviorReviewSelectionConfig,
) -> dict[str, Any]:
    """Audit disjointness, random weights, and ten-class queue support."""

    errors: list[str] = []
    warnings: list[str] = []
    invalid = sorted(
        set(units["behavior_review_cohort"].astype(str)).difference(
            BEHAVIOR_REVIEW_COHORTS
        )
    )
    if invalid:
        errors.append(f"invalid_behavior_review_cohorts={invalid}")
    expected_include = ~units["behavior_review_cohort"].eq(
        "behavior_not_selected"
    )
    actual_include = units["include_in_review"].astype(bool)
    if not expected_include.equals(actual_include):
        errors.append("behavior_review_include_cohort_mismatch")
    random_rows = units[
        units["behavior_review_cohort"].eq("behavior_random_audit")
    ]
    if not random_rows.empty:
        probability = pd.to_numeric(
            random_rows["behavior_sampling_probability"],
            errors="coerce",
        )
        weight = pd.to_numeric(
            random_rows["behavior_sampling_weight"],
            errors="coerce",
        )
        if (~probability.between(0.0, 1.0, inclusive="right")).any():
            errors.append("invalid_behavior_random_probability")
        if weight.le(0.0).any():
            errors.append("invalid_behavior_random_weight")
        if weight.mul(probability).sub(1.0).abs().gt(1e-8).any():
            errors.append("behavior_random_probability_weight_mismatch")
    expected_behaviors = {
        "drink",
        "eat",
        "fight",
        "social-nose",
        "explore",
        "lying",
        "stand",
        "move",
        "sitting",
        "playwithtoy",
    }
    present = set(units["behavior_label"].fillna("").astype(str))
    absent = sorted(expected_behaviors.difference(present))
    if absent:
        warnings.append(f"behaviors_absent_from_input={absent}")
    return {
        "schema_version": "classification_v2.behavior_review_selection.v1",
        "rows": int(len(units)),
        "selected_rows": int(actual_include.sum()),
        "cohort_counts": _counts(units, "behavior_review_cohort"),
        "behavior_counts": _counts(units, "behavior_label"),
        "selected_behavior_counts": _counts(
            units[actual_include],
            "behavior_label",
        ),
        "source_counts": _counts(units, "source_type"),
        "random_residual_estimand": (
            "human_intervention_rate_in_post_high-risk_residual_pool"
        ),
        "target_conditioned_selection": True,
        "config": {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        },
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _rare_census_mask(
    units: pd.DataFrame,
    config: BehaviorReviewSelectionConfig,
) -> pd.Series:
    if config.rare_census_max_per_source_behavior <= 0:
        return pd.Series(False, index=units.index)
    source = units["source_type"].fillna("").astype(str)
    behavior = units["behavior_label"].fillna("").astype(str)
    counts = units.groupby([source, behavior], dropna=False)[
        "review_unit_id"
    ].transform("size")
    return (
        source.isin(config.sampling_sources)
        & behavior.isin(RARE_CENSUS_BEHAVIORS)
        & counts.le(config.rare_census_max_per_source_behavior)
    )


def _calibrated_high_risk(
    units: pd.DataFrame,
    eligible: pd.Series,
    config: BehaviorReviewSelectionConfig,
) -> pd.Series:
    selected = pd.Series(False, index=units.index)
    pool = units[eligible].copy()
    if pool.empty or config.calibrated_high_risk_fraction <= 0:
        return selected
    for _, group in pool.groupby(
        list(config.stratum_columns),
        dropna=False,
        sort=True,
    ):
        if len(group) < config.calibrated_high_risk_min_pool:
            continue
        count = max(1, math.ceil(len(group) * config.calibrated_high_risk_fraction))
        count = min(count, config.calibrated_high_risk_max_per_stratum)
        ranked = _stable_rank(group, config.random_seed, "calibrated_high_risk")
        selected.loc[ranked.head(count).index] = True
    return selected


def _sample_each_stratum(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
    count: int,
    seed: int,
    salt: str,
) -> pd.DataFrame:
    if rows.empty or count <= 0:
        return rows.iloc[0:0].copy()
    parts = []
    for key, group in rows.groupby(list(columns), dropna=False, sort=True):
        ranked = _stable_sample(group, seed, f"{salt}|{key}")
        parts.append(ranked.head(count))
    return pd.concat(parts, ignore_index=False) if parts else rows.iloc[0:0]


def _clean_controls(
    rows: pd.DataFrame,
    config: BehaviorReviewSelectionConfig,
) -> pd.DataFrame:
    if rows.empty or config.clean_control_per_stratum <= 0:
        return rows.iloc[0:0].copy()
    parts = []
    for key, group in rows.groupby(
        list(config.stratum_columns),
        dropna=False,
        sort=True,
    ):
        priority = pd.to_numeric(group["review_priority"], errors="coerce").fillna(0.0)
        cutoff = float(priority.quantile(0.25))
        low = group[priority.le(cutoff)]
        ranked = _stable_sample(low, config.random_seed, f"clean|{key}")
        parts.append(ranked.head(config.clean_control_per_stratum))
    return pd.concat(parts, ignore_index=False) if parts else rows.iloc[0:0]


def _stable_rank(rows: pd.DataFrame, seed: int, salt: str) -> pd.DataFrame:
    out = rows.copy()
    out["_selection_hash"] = out["review_unit_id"].astype(str).map(
        lambda value: hashlib.sha256(
            f"{seed}|{salt}|{value}".encode()
        ).hexdigest()
    )
    out["_selection_priority"] = pd.to_numeric(
        out["review_priority"],
        errors="coerce",
    ).fillna(0.0)
    return out.sort_values(
        ["_selection_priority", "_selection_hash"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["_selection_hash", "_selection_priority"])


def _stable_sample(rows: pd.DataFrame, seed: int, salt: str) -> pd.DataFrame:
    out = rows.copy()
    out["_selection_hash"] = out["review_unit_id"].astype(str).map(
        lambda value: hashlib.sha256(
            f"{seed}|{salt}|{value}".encode()
        ).hexdigest()
    )
    return out.sort_values(
        "_selection_hash",
        kind="mergesort",
    ).drop(columns="_selection_hash")


def _assign_random_metadata(
    out: pd.DataFrame,
    *,
    population: pd.DataFrame,
    selected_mask: pd.Series,
    config: BehaviorReviewSelectionConfig,
) -> None:
    if population.empty:
        return
    population_strata = _strata(population, config.stratum_columns)
    selected_strata = out.loc[selected_mask, "behavior_sampling_stratum"]
    population_counts = population_strata.value_counts()
    selected_counts = selected_strata.value_counts()
    for index in out.index[selected_mask]:
        stratum = str(out.at[index, "behavior_sampling_stratum"])
        population_count = int(population_counts.get(stratum, 0))
        selected_count = int(selected_counts.get(stratum, 0))
        probability = selected_count / population_count if population_count else 0.0
        out.at[index, "behavior_sampling_population"] = population_count
        out.at[index, "behavior_sampling_selected"] = selected_count
        out.at[index, "behavior_sampling_probability"] = probability
        out.at[index, "behavior_sampling_weight"] = (
            1.0 / probability if probability > 0 else 0.0
        )


def _assign(
    out: pd.DataFrame,
    mask: pd.Series,
    *,
    cohort: str,
    design: str,
    priority_bonus: float,
    reason: str,
    preserve_sampling: bool = False,
) -> None:
    active = mask.reindex(out.index, fill_value=False).astype(bool)
    out.loc[active, "behavior_review_cohort"] = cohort
    out.loc[active, "behavior_sampling_design"] = design
    if not preserve_sampling:
        out.loc[active, "behavior_sampling_probability"] = 1.0
        out.loc[active, "behavior_sampling_weight"] = 1.0
    out.loc[active, "review_priority"] = (
        pd.to_numeric(out.loc[active, "review_priority"], errors="coerce")
        .fillna(0.0)
        .add(priority_bonus)
    )
    out.loc[active, "review_reason"] = [
        _combine_reason(value, reason)
        for value in out.loc[active, "review_reason"]
    ]


def _strata(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    parts = [
        f"{column}=" + rows[column].fillna("").astype(str)
        for column in columns
    ]
    result = parts[0]
    for part in parts[1:]:
        result = result + "|" + part
    return result


def _combine_reason(current: Any, new: str) -> str:
    values = [
        token.strip()
        for token in f"{current};{new}".split(";")
        if token.strip() and token.strip().lower() != "nan"
    ]
    return ";".join(dict.fromkeys(values))


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("").astype(str).value_counts().items()
    }


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"behavior review selection missing columns={missing}")


__all__ = [
    "BEHAVIOR_REVIEW_COHORTS",
    "BehaviorReviewSelectionConfig",
    "assign_behavior_review_cohorts",
    "audit_behavior_review_selection",
]
