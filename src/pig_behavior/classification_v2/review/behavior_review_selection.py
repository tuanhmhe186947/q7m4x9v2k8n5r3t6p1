"""Deterministic, auditable selection cohorts for behavior review."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

BEHAVIOR_REVIEW_COHORTS: tuple[str, ...] = (
    "behavior_mandatory_census",
    "behavior_high_risk",
    "behavior_random_audit",
    "behavior_not_selected",
)

SELECTION_PREDICATE_VERSION = (
    "classification_v2.behavior_review_candidate_selection.v2"
)
RARE_CENSUS_BEHAVIORS = frozenset({"playwithtoy"})

PREDICATE_COLUMNS: tuple[str, ...] = (
    "review_predicate_global_mandatory",
    "review_predicate_roi_contradiction",
    "review_predicate_roi_possible_false_negative",
    "review_predicate_interaction_contradiction",
    "review_predicate_partner_context_insufficient",
    "review_predicate_motion_contradiction",
    "review_predicate_posture_contradiction",
    "review_predicate_temporal_contradiction",
    "review_predicate_media_or_actor_authority_risk",
    "review_predicate_evidence_insufficiency",
    "review_predicate_rare_class_census",
    "review_predicate_risk_triggered",
    "review_predicate_stratified_low_risk_audit",
)

_ROI_CONTRADICTION_REASONS = frozenset(
    {
        "roi_label_without_persistent_target_support",
        "different_roi_has_stronger_support",
    }
)
_ROI_FALSE_NEGATIVE_REASONS = frozenset(
    {"explore_with_stationary_persistent_roi_contact"}
)
_INTERACTION_CONTRADICTION_REASONS = frozenset(
    {
        "fight_without_persistent_contact_or_aggression",
        "social_nose_without_persistent_partner_contact",
        "social_nose_with_fight_like_motion",
    }
)
_MOTION_CONTRADICTION_REASONS = frozenset(
    {
        "move_with_weak_motion_evidence",
        "stand_with_strong_motion_evidence",
        "explore_with_move_like_motion",
    }
)
_POSTURE_CONTRADICTION_REASONS = frozenset(
    {
        "posture_label_during_strong_shape_transition",
        "posture_label_with_strong_pixel_motion",
    }
)
_GENERIC_HIGH_RISK_REASONS = frozenset({"behavior_evidence_conflict"})
_MEDIA_AUTHORITY_REASONS = frozenset({"high_hidden_ratio_interval"})
_NONSELECTING_POLICY_REASONS = frozenset(
    {
        "interaction_requires_partner_context",
        "mandatory_behavior_review",
        "full_legacy_native_unit_review",
    }
)


@dataclass(frozen=True, slots=True)
class BehaviorReviewSelectionConfig:
    """Predeclared queue design; selection never changes labels or weights."""

    random_seed: int = 20260720
    random_per_stratum: int = 5
    clean_control_per_stratum: int = 0
    calibrated_high_risk_fraction: float = 0.0
    calibrated_high_risk_max_per_stratum: int = 0
    calibrated_high_risk_min_pool: int = 20
    rare_census_max_per_source_behavior: int = 0
    sampling_sources: tuple[str, ...] = (
        "legacy_recovered",
        "cvat_tracking_xml",
    )
    stratum_columns: tuple[str, ...] = (
        "behavior_label",
        "source_type",
        "recording_date",
        "evidence_quality_stratum",
    )

    def validate(self) -> None:
        if self.random_per_stratum < 0:
            raise ValueError("random_per_stratum must be >= 0")
        if self.clean_control_per_stratum < 0:
            raise ValueError("clean_control_per_stratum must be >= 0")
        if self.clean_control_per_stratum:
            raise ValueError(
                "clean controls are not a separate GUI cohort; use the "
                "stratified low-risk audit"
            )
        if self.calibrated_high_risk_fraction:
            raise ValueError(
                "calibrated top-fraction selection is disabled because a "
                "quota is not evidence of risk"
            )
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
    include_all_retained_native_units: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select review candidates from concrete predicates and a QC sample."""

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
    out["review_key"] = out["review_unit_id"].astype(str)
    out["behavior"] = out["behavior_label"].fillna("").astype(str)
    out["source"] = out["source_type"].fillna("").astype(str)
    out["recording_date"] = out["video_key"].map(_recording_date)
    out["evidence_quality_stratum"] = _evidence_quality_stratum(out)
    _attach_candidate_predicates(out)

    out["behavior_review_cohort"] = "behavior_not_selected"
    out["behavior_sampling_design"] = "not_sampled"
    out["behavior_sampling_stratum"] = _strata(out, cfg.stratum_columns)
    out["behavior_sampling_population"] = 0
    out["behavior_sampling_selected"] = 0
    out["behavior_sampling_probability"] = 0.0
    out["behavior_sampling_weight"] = 0.0
    out["behavior_review_residual_estimand"] = False

    source = out["source_type"].fillna("").astype(str)
    legacy = out["review_unit_type"].fillna("").astype(str).eq(
        "legacy_burst_16"
    )

    hard_predicates = (
        "review_predicate_roi_contradiction",
        "review_predicate_roi_possible_false_negative",
        "review_predicate_interaction_contradiction",
        "review_predicate_partner_context_insufficient",
        "review_predicate_motion_contradiction",
        "review_predicate_posture_contradiction",
        "review_predicate_temporal_contradiction",
        "review_predicate_media_or_actor_authority_risk",
        "review_predicate_evidence_insufficiency",
        "review_predicate_rare_class_census",
    )
    mandatory = out[list(hard_predicates)].any(axis=1)
    if include_all_retained_legacy_units:
        mandatory |= legacy
    if include_all_retained_native_units:
        out["review_predicate_global_mandatory"] = True
        mandatory |= out["review_predicate_global_mandatory"]
    _assign(
        out,
        mandatory,
        cohort="behavior_mandatory_census",
        design="concrete_predicate_or_rare_class_census",
        priority_bonus=200.0,
        reason="",
    )

    inherited_high_risk = (
        out["review_predicate_risk_triggered"]
        & out["behavior_review_cohort"].eq("behavior_not_selected")
    )
    _assign(
        out,
        inherited_high_risk,
        cohort="behavior_high_risk",
        design="rule_based_high_risk",
        priority_bonus=100.0,
        reason="",
    )

    eligible_source = source.isin(cfg.sampling_sources)
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
        reason="",
        preserve_sampling=True,
    )
    out.loc[random_mask, "behavior_review_residual_estimand"] = True
    out["review_predicate_stratified_low_risk_audit"] = random_mask

    out["include_in_review"] = ~out["behavior_review_cohort"].eq(
        "behavior_not_selected"
    )
    out["mandatory_behavior_review_unit"] = out[
        "behavior_review_cohort"
    ].eq("behavior_mandatory_census")
    out["candidate_tier"] = "AUTO_CARRY_LOW_RISK"
    out.loc[
        out["behavior_review_cohort"].eq("behavior_mandatory_census"),
        "candidate_tier",
    ] = "TIER_1_HARD_MANDATORY"
    out.loc[
        out["behavior_review_cohort"].eq("behavior_high_risk"),
        "candidate_tier",
    ] = "TIER_2_HIGH_RISK"
    out.loc[
        out["behavior_review_cohort"].eq("behavior_random_audit"),
        "candidate_tier",
    ] = "TIER_3_STRATIFIED_AUDIT"
    out["selection_predicate_version"] = SELECTION_PREDICATE_VERSION
    out["selection_config_hash"] = _selection_config_hash(cfg)
    out["review_selection_predicates"] = out.apply(
        _active_predicate_names,
        axis=1,
    )
    out["review_reason_codes"] = out.apply(_selection_reason_codes, axis=1)
    out["review_reason"] = out["review_reason_codes"]
    out["risk_score"] = _risk_score(out)
    out["risk_components"] = out.apply(_risk_components, axis=1)
    out["evidence_values_used"] = out.apply(_evidence_values_used, axis=1)
    out["evidence_availability"] = out.apply(
        _evidence_availability,
        axis=1,
    )
    out["review_risk_bucket"] = _risk_bucket(out)
    out["auto_carry_behavior"] = out["behavior_label"].astype(str)
    out["auto_carry_provenance"] = (
        "provisional_source_derived_behavior_unchanged"
    )
    out["human_decision_synthesized"] = False

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
    duplicate_keys = int(
        units["review_unit_id"].fillna("").astype(str).duplicated().sum()
    )
    if duplicate_keys:
        errors.append(f"duplicate_review_keys={duplicate_keys}")
    candidate_keys = set(
        units.loc[actual_include, "review_unit_id"].astype(str)
    )
    auto_keys = set(
        units.loc[~actual_include, "review_unit_id"].astype(str)
    )
    overlap = candidate_keys.intersection(auto_keys)
    missing = set(units["review_unit_id"].astype(str)).difference(
        candidate_keys | auto_keys
    )
    if overlap:
        errors.append(f"candidate_auto_carry_overlap={len(overlap)}")
    if missing:
        errors.append(f"missing_universe_keys={len(missing)}")
    concrete = units[
        [
            column
            for column in PREDICATE_COLUMNS
            if column != "review_predicate_global_mandatory"
        ]
    ].any(axis=1)
    unexplained = actual_include & ~concrete
    if unexplained.any():
        errors.append(f"unexplained_candidates={int(unexplained.sum())}")
    availability_only = (
        actual_include
        & _bool_column(units, "review_evidence_available")
        & ~concrete
    )
    if availability_only.any():
        errors.append(
            "evidence_availability_only_candidates="
            f"{int(availability_only.sum())}"
        )
    global_mandatory = _bool_column(
        units,
        "review_predicate_global_mandatory",
    )
    if global_mandatory.any():
        errors.append(
            f"global_mandatory_selection_forbidden={int(global_mandatory.sum())}"
        )
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
    predicate_audit = {
        column: _predicate_audit(units, column)
        for column in PREDICATE_COLUMNS
    }
    return {
        "schema_version": "classification_v2.behavior_review_selection.v2",
        "rows": int(len(units)),
        "selected_rows": int(actual_include.sum()),
        "auto_carry_rows": int((~actual_include).sum()),
        "candidate_plus_auto_carry_rows": int(len(units)),
        "candidate_auto_carry_overlap": int(len(overlap)),
        "missing_universe_keys": int(len(missing)),
        "duplicate_review_keys": duplicate_keys,
        "unexplained_candidates": int(unexplained.sum()),
        "evidence_availability_only_candidates": int(
            availability_only.sum()
        ),
        "cohort_counts": _counts(units, "behavior_review_cohort"),
        "candidate_tier_counts": _counts(units, "candidate_tier"),
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
        "selection_predicate_version": SELECTION_PREDICATE_VERSION,
        "selection_config_hash": _selection_config_hash(config),
        "field_provenance": _field_provenance(),
        "predicate_audit": predicate_audit,
        "predicate_overlap": _predicate_overlap(units),
        "config": {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        },
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _field_provenance() -> list[dict[str, Any]]:
    evidence_file = (
        "src/pig_behavior/classification_v2/review/behavior_evidence.py"
    )
    builder_file = (
        "src/pig_behavior/classification_v2/review/review_unit_builder.py"
    )
    selection_file = (
        "src/pig_behavior/classification_v2/review/"
        "behavior_review_selection.py"
    )
    availability = [
        "review_evidence_available",
        "review_relevant_evidence_available",
        "review_motion_evidence_available",
        "review_roi_evidence_available",
        "review_social_evidence_available",
        "review_posture_evidence_available",
    ]
    records = [
        {
            "field_name": "review_template",
            "producer_file": builder_file,
            "producer_function": "_finalize_unit_review_fields",
            "input_columns": ["behavior_label", "temporal_consistency_status"],
            "exact_predicate": (
                "label membership chooses interaction/roi/motion/posture; "
                "unstable temporal status chooses temporal_consistency"
            ),
            "default_value": "general",
            "missing_value_behavior": "general template",
            "used_for_template_only": True,
            "used_for_candidate_selection": False,
        },
        {
            "field_name": "review_reason",
            "producer_file": selection_file,
            "producer_function": "_selection_reason_codes",
            "input_columns": [
                "review_evidence_reason_auto",
                "interval_review_reason",
                *PREDICATE_COLUMNS,
            ],
            "exact_predicate": (
                "specific active contradiction/authority/rare/audit reason "
                "codes only; generic policy metadata is removed"
            ),
            "default_value": "",
            "missing_value_behavior": "not a candidate unless sampled",
            "used_for_template_only": False,
            "used_for_candidate_selection": False,
        },
        {
            "field_name": "review_priority",
            "producer_file": builder_file,
            "producer_function": "_finalize_unit_review_fields",
            "input_columns": [
                "review_priority_window_max",
                "review_evidence_priority_auto",
                "candidate_tier",
            ],
            "exact_predicate": "base evidence priority plus tier ordering bonus",
            "default_value": 0.0,
            "missing_value_behavior": "coerced to zero",
            "used_for_template_only": False,
            "used_for_candidate_selection": False,
        },
        {
            "field_name": "include_in_review",
            "producer_file": selection_file,
            "producer_function": "assign_behavior_review_cohorts",
            "input_columns": list(PREDICATE_COLUMNS),
            "exact_predicate": (
                "concrete hard predicate OR rule-based high risk OR "
                "deterministic stratified low-risk audit"
            ),
            "default_value": False,
            "missing_value_behavior": "AUTO_CARRY_LOW_RISK",
            "used_for_template_only": False,
            "used_for_candidate_selection": True,
        },
        {
            "field_name": "mandatory_behavior_review_unit",
            "producer_file": selection_file,
            "producer_function": "assign_behavior_review_cohorts",
            "input_columns": list(PREDICATE_COLUMNS[1:-2]),
            "exact_predicate": "candidate_tier == TIER_1_HARD_MANDATORY",
            "default_value": False,
            "missing_value_behavior": "false",
            "used_for_template_only": False,
            "used_for_candidate_selection": True,
        },
    ]
    for field_name in availability:
        records.append(
            {
                "field_name": field_name,
                "producer_file": evidence_file,
                "producer_function": "_score_behavior_row",
                "input_columns": [
                    "native evidence validity columns",
                    "Pig-STRENet availability columns",
                ],
                "exact_predicate": (
                    "declared modality validity/availability only; never a "
                    "contradiction by itself"
                ),
                "default_value": False,
                "missing_value_behavior": (
                    "unavailable; relevant missing modality may create a "
                    "separate evidence-insufficiency predicate"
                ),
                "used_for_template_only": False,
                "used_for_candidate_selection": False,
            }
        )
    return records


def _attach_candidate_predicates(out: pd.DataFrame) -> None:
    evidence_tokens = out.get(
        "review_evidence_reason_auto",
        pd.Series("", index=out.index),
    ).fillna("").astype(str).map(_reason_tokens)
    interval_tokens = out.get(
        "interval_review_reason",
        pd.Series("", index=out.index),
    ).fillna("").astype(str).map(_reason_tokens)
    behavior = out["behavior_label"].fillna("").astype(str)
    status = out["temporal_consistency_status"].fillna("").astype(str)

    out["review_predicate_global_mandatory"] = False
    out["review_predicate_roi_contradiction"] = evidence_tokens.map(
        lambda values: bool(values & _ROI_CONTRADICTION_REASONS)
    )
    out["review_predicate_roi_possible_false_negative"] = (
        evidence_tokens.map(
            lambda values: bool(values & _ROI_FALSE_NEGATIVE_REASONS)
        )
    )
    out["review_predicate_interaction_contradiction"] = (
        evidence_tokens.map(
            lambda values: bool(values & _INTERACTION_CONTRADICTION_REASONS)
        )
    )
    out["review_predicate_partner_context_insufficient"] = (
        behavior.isin({"fight", "social-nose"})
        & evidence_tokens.map(lambda values: "social_evidence_unavailable" in values)
    )
    out["review_predicate_motion_contradiction"] = evidence_tokens.map(
        lambda values: bool(values & _MOTION_CONTRADICTION_REASONS)
    )
    out["review_predicate_posture_contradiction"] = evidence_tokens.map(
        lambda values: bool(values & _POSTURE_CONTRADICTION_REASONS)
    )
    out["review_predicate_temporal_contradiction"] = ~status.eq("stable")
    out["review_predicate_media_or_actor_authority_risk"] = (
        interval_tokens.map(
            lambda values: bool(values & _MEDIA_AUTHORITY_REASONS)
        )
        | ~_bool_column(out, "temporal_interval_complete", default=True)
        | pd.to_numeric(
            out.get(
                "bbox_valid_ratio_interval",
                pd.Series(1.0, index=out.index),
            ),
            errors="coerce",
        ).fillna(0.0).le(0.0)
    )
    out["review_predicate_evidence_insufficiency"] = (
        out.get(
            "review_evidence_status_auto",
            pd.Series("", index=out.index),
        )
        .fillna("")
        .astype(str)
        .eq("missing_relevant_modality")
    )
    out["review_predicate_rare_class_census"] = behavior.isin(
        RARE_CENSUS_BEHAVIORS
    )
    explicit_high_risk = evidence_tokens.map(
        lambda values: bool(values & _GENERIC_HIGH_RISK_REASONS)
    )
    inherited_tokens = out.get(
        "review_reason",
        pd.Series("", index=out.index),
    ).fillna("").astype(str).map(_reason_tokens)
    inherited_concrete = inherited_tokens.map(
        lambda values: bool(values - _NONSELECTING_POLICY_REASONS)
    )
    out["review_predicate_risk_triggered"] = (
        explicit_high_risk | inherited_concrete
    )
    out["review_predicate_stratified_low_risk_audit"] = False


def _recording_date(value: Any) -> str:
    match = re.search(r"(?i)pigs(\d{2})(\d{2})(\d{2})", str(value))
    if not match:
        return "UNKNOWN"
    day, month, year = match.groups()
    return f"20{year}-{month}-{day}"


def _evidence_quality_stratum(rows: pd.DataFrame) -> pd.Series:
    relevant = _bool_column(rows, "review_relevant_evidence_available")
    pig_available = _bool_column(rows, "review_pig_evidence_available")
    history = pd.to_numeric(
        rows.get(
            "review_pig_history_available_ratio",
            pd.Series(0.0, index=rows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    result = pd.Series("native_only", index=rows.index, dtype="object")
    result.loc[~relevant] = "insufficient_relevant_modality"
    result.loc[relevant & pig_available & history.lt(1.0)] = (
        "target_complete_partial_history"
    )
    result.loc[relevant & pig_available & history.ge(1.0)] = (
        "target_and_history_complete"
    )
    result.loc[
        relevant
        & ~pig_available
        & ~_bool_column(rows, "review_evidence_available")
    ] = "evidence_unavailable"
    return result


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


def _reason_tokens(value: Any) -> set[str]:
    return {
        token.strip()
        for token in str(value).split(";")
        if token.strip() and token.strip().lower() != "nan"
    }


def _bool_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return (
        values.fillna(str(default))
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _selection_config_hash(config: BehaviorReviewSelectionConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _active_predicate_names(row: pd.Series) -> str:
    return ";".join(
        column.removeprefix("review_predicate_")
        for column in PREDICATE_COLUMNS
        if bool(row.get(column, False))
    )


def _selection_reason_codes(row: pd.Series) -> str:
    reasons = _reason_tokens(row.get("review_evidence_reason_auto", ""))
    reasons |= (
        _reason_tokens(row.get("review_reason", ""))
        - _NONSELECTING_POLICY_REASONS
    )
    interval_reasons = (
        _reason_tokens(row.get("interval_review_reason", ""))
        - _NONSELECTING_POLICY_REASONS
    )
    if bool(row.get("review_predicate_media_or_actor_authority_risk", False)):
        reasons |= interval_reasons or {"MEDIA_OR_ACTOR_AUTHORITY_RISK"}
    if bool(row.get("review_predicate_temporal_contradiction", False)):
        reasons.add("TEMPORAL_CONTRADICTION")
    if bool(row.get("review_predicate_rare_class_census", False)):
        reasons.add("RARE_CLASS_CENSUS_PLAYWITHTOY")
    if bool(row.get("review_predicate_stratified_low_risk_audit", False)):
        reasons.add("STRATIFIED_LOW_RISK_AUDIT")
    if bool(row.get("review_predicate_global_mandatory", False)):
        reasons.add("GLOBAL_MANDATORY_FORBIDDEN")
    if not bool(row.get("include_in_review", False)):
        return ""
    return ";".join(sorted(reasons))


def _risk_score(rows: pd.DataFrame) -> pd.Series:
    conflict = pd.to_numeric(
        rows.get(
            "review_evidence_conflict_score",
            pd.Series(0.0, index=rows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    insufficiency = pd.to_numeric(
        rows.get(
            "review_evidence_insufficiency_score",
            pd.Series(0.0, index=rows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    hard = rows["candidate_tier"].eq("TIER_1_HARD_MANDATORY")
    high = rows["candidate_tier"].eq("TIER_2_HIGH_RISK")
    return pd.concat(
        [
            conflict.clip(0.0, 1.0),
            insufficiency.clip(0.0, 1.0),
            hard.astype(float),
            high.astype(float) * 0.75,
        ],
        axis=1,
    ).max(axis=1)


def _risk_components(row: pd.Series) -> str:
    components = {
        "evidence_conflict": _finite_number(
            row.get("review_evidence_conflict_score", 0.0)
        ),
        "evidence_insufficiency": _finite_number(
            row.get("review_evidence_insufficiency_score", 0.0)
        ),
        "pig_strenet_conflict": _finite_number(
            row.get("review_pig_strenet_conflict_score", 0.0)
        ),
        "active_predicates": row.get("review_selection_predicates", ""),
    }
    return json.dumps(components, sort_keys=True, separators=(",", ":"))


def _evidence_values_used(row: pd.Series) -> str:
    values = {
        "motion_support": _finite_number(
            row.get("review_motion_support_score", 0.0)
        ),
        "roi_support": _finite_number(
            row.get("review_roi_support_score", 0.0)
        ),
        "social_support": _finite_number(
            row.get("review_social_support_score", 0.0)
        ),
        "posture_transition": _finite_number(
            row.get("review_posture_transition_score", 0.0)
        ),
        "temporal_status": str(
            row.get("temporal_consistency_status", "")
        ),
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _evidence_availability(row: pd.Series) -> str:
    availability = {
        "evidence": _truth_value(
            row.get("review_evidence_available", False)
        ),
        "relevant": _truth_value(
            row.get("review_relevant_evidence_available", False)
        ),
        "motion": _truth_value(
            row.get("review_motion_evidence_available", False)
        ),
        "roi": _truth_value(
            row.get("review_roi_evidence_available", False)
        ),
        "social": _truth_value(
            row.get("review_social_evidence_available", False)
        ),
        "posture": _truth_value(
            row.get("review_posture_evidence_available", False)
        ),
        "pig_strenet": _truth_value(
            row.get("review_pig_evidence_available", False)
        ),
    }
    return json.dumps(availability, sort_keys=True, separators=(",", ":"))


def _finite_number(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(number) else float(number)


def _truth_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _risk_bucket(rows: pd.DataFrame) -> pd.Series:
    result = pd.Series("LOW_RISK", index=rows.index, dtype="object")
    result.loc[rows["candidate_tier"].eq("TIER_3_STRATIFIED_AUDIT")] = (
        "LOW_RISK_AUDIT"
    )
    result.loc[rows["candidate_tier"].eq("TIER_2_HIGH_RISK")] = "HIGH_RISK"
    result.loc[rows["candidate_tier"].eq("TIER_1_HARD_MANDATORY")] = (
        "HARD_MANDATORY"
    )
    return result


def _predicate_audit(
    units: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    mask = _bool_column(units, column)
    selected = units.loc[mask].copy()
    other_columns = [
        value for value in PREDICATE_COLUMNS if value != column
    ]
    other = (
        units[other_columns].fillna(False).astype(bool).any(axis=1)
        if other_columns
        else pd.Series(False, index=units.index)
    )
    example_columns = [
        value
        for value in (
            "review_unit_id",
            "behavior_label",
            "source_type",
            "recording_date",
            "review_template",
            "review_evidence_conflict_score",
            "review_evidence_reason_auto",
        )
        if value in selected.columns
    ]
    example_frame = selected.sort_values(
        "review_unit_id",
        kind="mergesort",
    ).head(10)[example_columns]
    examples = example_frame.where(
        pd.notna(example_frame),
        None,
    ).to_dict(orient="records")
    return {
        "total_true": int(mask.sum()),
        "percent_universe": (
            float(mask.mean() * 100.0) if len(mask) else 0.0
        ),
        "count_by_behavior": _counts(selected, "behavior_label"),
        "count_by_source": _counts(selected, "source_type"),
        "count_by_calendar_date": _counts(selected, "recording_date"),
        "count_by_review_template": _counts(selected, "review_template"),
        "count_by_evidence_quality_bucket": _counts(
            selected,
            "evidence_quality_stratum",
        ),
        "exclusive_count": int((mask & ~other).sum()),
        "example_review_keys_and_features": examples,
    }


def _predicate_overlap(units: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(PREDICATE_COLUMNS):
        left_mask = _bool_column(units, left)
        for right in PREDICATE_COLUMNS[left_index:]:
            overlap = int(
                (left_mask & _bool_column(units, right)).sum()
            )
            rows.append(
                {
                    "predicate_a": left,
                    "predicate_b": right,
                    "overlap_count": overlap,
                }
            )
    return rows


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
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
    "PREDICATE_COLUMNS",
    "SELECTION_PREDICATE_VERSION",
    "BehaviorReviewSelectionConfig",
    "assign_behavior_review_cohorts",
    "audit_behavior_review_selection",
]
