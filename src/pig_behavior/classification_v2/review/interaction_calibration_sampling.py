"""Frozen grouped sampling for blinded interaction-label calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

CALIBRATION_SAMPLE_SCHEMA_VERSION = (
    "classification_v2.interaction_blind_calibration_sample.v1"
)
CALIBRATION_SAMPLE_STATUS = "PRE_REVIEW_CALIBRATION_INFRASTRUCTURE"
DEFAULT_SAMPLE_SEED = 2026072901
INTERACTION_BEHAVIORS = frozenset({"fight", "social-nose"})

_TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
_PUBLIC_FORBIDDEN_COLUMN_TOKENS = (
    "behavior",
    "candidate",
    "contact",
    "crowd",
    "date",
    "label",
    "partner",
    "reason",
    "score",
    "selector",
    "source",
    "stratum",
    "video",
)
_UNIVERSE_REQUIRED = frozenset(
    {
        "review_unit_id",
        "temporal_unit_key",
        "behavior_label",
        "source_type",
        "recording_date",
        "video_key",
        "dataset_id",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_pig_history_display_frame_indices",
        "review_pig_history_available_ratio",
        "include_in_review",
        "review_predicate_media_or_actor_authority_risk",
        "review_social_evidence_available",
        "evidence_quality_stratum",
    }
)
_DIAGNOSTIC_REQUIRED = frozenset(
    {
        "review_unit_id",
        "old_include_in_review",
        "new_include_in_review",
        "neighborhood_evidence_available",
        "frames_with_valid_neighbors",
        "neighbor_valid_ratio",
        "any_contact_proxy_ratio",
        "max_concurrent_contact_proxy_count",
        "min_edge_distance_over_unit",
        "median_min_edge_distance",
        "overlap_present_ratio",
        "crowding_ratio",
        "center_edge_top1_agreement_ratio",
        "previous_selector_evidence_invalid",
    }
)


class InteractionCalibrationSamplingError(ValueError):
    """Raised when a frozen blinded sample cannot be built safely."""


@dataclass(frozen=True)
class InteractionCalibrationSamplingConfig:
    """Predeclared recommended sample configuration."""

    development_count: int = 300
    confirmation_count: int = 180
    confirmation_fraction_by_group: float = 0.375
    seed: int = DEFAULT_SAMPLE_SEED
    schema_version: str = CALIBRATION_SAMPLE_SCHEMA_VERSION


@dataclass(frozen=True)
class InteractionCalibrationSampleResult:
    """Internal, blinded, media, group, and audit outputs."""

    group_split: pd.DataFrame
    internal_trace: pd.DataFrame
    blinded_manifest: pd.DataFrame
    media_authority: pd.DataFrame
    audit: dict[str, Any]


def _truth(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in _TRUE_VALUES


def _stable_hex(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}|{namespace}|{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_sampling_config_hash(
    config: InteractionCalibrationSamplingConfig,
) -> str:
    """Return the semantic hash of the frozen sampling configuration."""

    return _canonical_hash(asdict(config))


def _validate_inputs(
    universe: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    missing_universe = sorted(_UNIVERSE_REQUIRED.difference(universe.columns))
    missing_diagnostic = sorted(
        _DIAGNOSTIC_REQUIRED.difference(diagnostic.columns)
    )
    if missing_universe:
        raise InteractionCalibrationSamplingError(
            f"review universe missing columns: {missing_universe}"
        )
    if missing_diagnostic:
        raise InteractionCalibrationSamplingError(
            f"diagnostic evidence missing columns: {missing_diagnostic}"
        )
    for name, rows in (("universe", universe), ("diagnostic", diagnostic)):
        ids = rows["review_unit_id"].fillna("").astype(str).str.strip()
        if ids.eq("").any():
            raise InteractionCalibrationSamplingError(
                f"{name} contains blank review keys"
            )
        if ids.duplicated().any():
            raise InteractionCalibrationSamplingError(
                f"{name} contains duplicate review keys"
            )


def _prepare_population(
    universe: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    _validate_inputs(universe, diagnostic)
    behavior = universe["behavior_label"].fillna("").astype(str)
    interaction = universe.loc[behavior.isin(INTERACTION_BEHAVIORS)].copy()
    merged = interaction.merge(
        diagnostic,
        on="review_unit_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_diagnostic"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        count = int(merged["_merge"].ne("both").sum())
        raise InteractionCalibrationSamplingError(
            f"interaction units missing diagnostic evidence: {count}"
        )
    merged = merged.drop(columns=["_merge"])
    merged["current_interaction_candidate"] = merged[
        "old_include_in_review"
    ].map(_truth)
    merged["current_interaction_non_candidate"] = ~merged[
        "current_interaction_candidate"
    ]
    merged["static_set_95_diagnostic"] = merged[
        "new_include_in_review"
    ].map(_truth)
    merged["removed_by_static_diagnostic"] = (
        merged["current_interaction_candidate"]
        & ~merged["static_set_95_diagnostic"]
    )
    agreement = pd.to_numeric(
        merged["center_edge_top1_agreement_ratio"],
        errors="coerce",
    )
    merged["center_edge_disagreement"] = agreement.lt(1.0 - 1e-12)
    merged["contact_proxy_present"] = pd.to_numeric(
        merged["any_contact_proxy_ratio"],
        errors="coerce",
    ).gt(0.0)
    merged["contact_proxy_absent"] = (
        merged["neighborhood_evidence_available"].map(_truth)
        & ~merged["contact_proxy_present"]
    )
    concurrent = pd.to_numeric(
        merged["max_concurrent_contact_proxy_count"],
        errors="coerce",
    )
    lower_cut = float(concurrent.quantile(0.25))
    upper_cut = float(concurrent.quantile(0.75))
    merged["lower_crowding"] = concurrent.le(lower_cut)
    merged["high_crowding"] = concurrent.ge(upper_cut)
    merged["social_evidence_available"] = merged[
        "review_social_evidence_available"
    ].map(_truth)
    merged["social_evidence_unavailable_or_low_quality"] = (
        ~merged["social_evidence_available"]
        | merged["evidence_quality_stratum"]
        .fillna("")
        .astype(str)
        .str.contains("insufficient|partial", case=False, regex=True)
    )
    merged["authority_risk_control"] = merged[
        "review_predicate_media_or_actor_authority_risk"
    ].map(_truth) | merged["previous_selector_evidence_invalid"].map(_truth)
    merged["visually_clean_control"] = (
        ~merged["authority_risk_control"]
        & merged["neighborhood_evidence_available"].map(_truth)
        & pd.to_numeric(
            merged["neighbor_valid_ratio"],
            errors="coerce",
        ).ge(1.0)
    )
    merged["calibration_group_key"] = (
        merged["source_type"].astype(str)
        + "|"
        + merged["recording_date"].astype(str)
        + "|"
        + merged["video_key"].astype(str)
    )
    return merged


def _assign_group_split(
    population: pd.DataFrame,
    config: InteractionCalibrationSamplingConfig,
) -> pd.DataFrame:
    grouped = (
        population.groupby(
            [
                "calibration_group_key",
                "source_type",
                "recording_date",
                "video_key",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            row_count=("review_unit_id", "size"),
            fight_count=(
                "behavior_label",
                lambda values: int((values == "fight").sum()),
            ),
            social_nose_count=(
                "behavior_label",
                lambda values: int((values == "social-nose").sum()),
            ),
        )
        .reset_index()
    )
    grouped["_split_hash"] = grouped["calibration_group_key"].map(
        lambda value: _stable_hex(config.seed, "group_split", str(value))
    )
    assignments: list[pd.DataFrame] = []
    for _, source_groups in grouped.groupby("source_type", sort=True):
        source_groups = source_groups.sort_values("_split_hash").copy()
        target = (
            int(source_groups["row_count"].sum())
            * config.confirmation_fraction_by_group
        )
        confirmation_rows = 0
        subsets: list[str] = []
        remaining = len(source_groups)
        for record in source_groups.itertuples(index=False):
            remaining -= 1
            add_distance = abs(
                confirmation_rows + int(record.row_count) - target
            )
            keep_distance = abs(confirmation_rows - target)
            choose_confirmation = add_distance < keep_distance
            if not subsets:
                choose_confirmation = True
            if remaining == 0 and all(
                value == "BLINDED_CONFIRMATION_SET" for value in subsets
            ):
                choose_confirmation = False
            subset = (
                "BLINDED_CONFIRMATION_SET"
                if choose_confirmation
                else "CALIBRATION_DEVELOPMENT_SET"
            )
            subsets.append(subset)
            if choose_confirmation:
                confirmation_rows += int(record.row_count)
        source_groups["frozen_subset"] = subsets
        assignments.append(source_groups)
    result = pd.concat(assignments, ignore_index=True)
    return result.drop(columns=["_split_hash"]).sort_values(
        ["frozen_subset", "source_type", "recording_date", "video_key"]
    )


def _stratum_columns() -> tuple[str, ...]:
    return (
        "current_interaction_candidate",
        "current_interaction_non_candidate",
        "center_edge_disagreement",
        "static_set_95_diagnostic",
        "removed_by_static_diagnostic",
        "high_crowding",
        "lower_crowding",
        "contact_proxy_present",
        "contact_proxy_absent",
        "social_evidence_available",
        "social_evidence_unavailable_or_low_quality",
        "visually_clean_control",
        "authority_risk_control",
    )


def _quota(
    name: str,
    target: int,
    available: int,
) -> int:
    fractions = {
        "current_interaction_candidate": 0.25,
        "current_interaction_non_candidate": 0.40,
        "center_edge_disagreement": 0.20,
        "static_set_95_diagnostic": 0.06,
        "removed_by_static_diagnostic": 0.18,
        "high_crowding": 0.20,
        "lower_crowding": 0.20,
        "contact_proxy_present": 0.35,
        "contact_proxy_absent": 0.08,
        "social_evidence_available": 0.35,
        "social_evidence_unavailable_or_low_quality": 0.08,
        "visually_clean_control": 0.20,
        "authority_risk_control": 0.08,
        "behavior=fight": 0.35,
        "behavior=social-nose": 0.35,
        "source=cvat_tracking_xml": 0.35,
        "source=legacy_recovered": 0.18,
    }
    desired = max(1, int(math.ceil(target * fractions[name])))
    return min(desired, available)


def _sample_subset(
    pool: pd.DataFrame,
    *,
    subset: str,
    target: int,
    config: InteractionCalibrationSamplingConfig,
) -> pd.DataFrame:
    if len(pool) < target:
        raise InteractionCalibrationSamplingError(
            f"{subset} has only {len(pool)} rows for target={target}"
        )
    pool = pool.copy().reset_index(drop=True)
    pool["_sample_hash"] = pool["review_unit_id"].map(
        lambda value: _stable_hex(config.seed, f"sample:{subset}", str(value))
    )
    masks: dict[str, pd.Series] = {
        name: pool[name].map(_truth) for name in _stratum_columns()
    }
    masks.update(
        {
            "behavior=fight": pool["behavior_label"].eq("fight"),
            "behavior=social-nose": pool["behavior_label"].eq("social-nose"),
            "source=cvat_tracking_xml": pool["source_type"].eq(
                "cvat_tracking_xml"
            ),
            "source=legacy_recovered": pool["source_type"].eq(
                "legacy_recovered"
            ),
        }
    )
    quotas = {
        name: _quota(name, target, int(mask.sum()))
        for name, mask in masks.items()
    }
    names = list(masks)
    matrix = np.column_stack(
        [masks[name].to_numpy(dtype=bool) for name in names]
    )
    available = matrix.sum(axis=0).clip(min=1)
    quota_values = np.asarray([quotas[name] for name in names], dtype=int)
    counts = np.zeros(len(names), dtype=int)
    remaining = np.ones(len(pool), dtype=bool)
    hash_order = (
        pool["_sample_hash"].rank(method="first").to_numpy(dtype=float)
    )
    selected: list[int] = []

    while len(selected) < target:
        unmet = counts < quota_values
        weights = unmet.astype(float) * (1.0 + len(pool) / available)
        scores = matrix.astype(float) @ weights
        scores[~remaining] = -1.0
        best_score = float(scores.max())
        candidates = np.flatnonzero(scores == best_score)
        if not len(candidates):
            raise InteractionCalibrationSamplingError(
                f"unable to fill deterministic sample for {subset}"
            )
        best_index = int(candidates[np.argmin(hash_order[candidates])])
        selected.append(best_index)
        remaining[best_index] = False
        counts += matrix[best_index].astype(int)

    result = pool.loc[selected].copy()
    result["frozen_subset"] = subset
    result["_presentation_hash"] = result["review_unit_id"].map(
        lambda value: _stable_hex(
            config.seed,
            f"presentation_order:{subset}",
            str(value),
        )
    )
    result = result.sort_values("_presentation_hash").reset_index(drop=True)
    result["presentation_order"] = range(1, len(result) + 1)
    return result.drop(columns=["_sample_hash", "_presentation_hash"])


def _machine_strata(row: pd.Series) -> str:
    names = [
        name for name in _stratum_columns() if _truth(row.get(name, False))
    ]
    names.append(f"behavior={row['behavior_label']}")
    names.append(f"source={row['source_type']}")
    return ";".join(sorted(names))


def _frame_count(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([token for token in str(value).split(",") if token.strip()])


def build_interaction_calibration_sample(
    universe: pd.DataFrame,
    diagnostic: pd.DataFrame,
    *,
    producer_sha: str,
    input_hashes: dict[str, str],
    presentation_version: str,
    presentation_semantic_hash: str,
    config: InteractionCalibrationSamplingConfig | None = None,
) -> InteractionCalibrationSampleResult:
    """Build a frozen group-separated development and confirmation sample."""

    config = config or InteractionCalibrationSamplingConfig()
    if len(producer_sha) != 40:
        raise InteractionCalibrationSamplingError(
            "producer_sha must be a full Git SHA"
        )
    if not input_hashes or any(len(value) != 64 for value in input_hashes.values()):
        raise InteractionCalibrationSamplingError(
            "every declared input hash must be SHA-256"
        )
    if len(presentation_semantic_hash) != 64:
        raise InteractionCalibrationSamplingError(
            "presentation_semantic_hash must be SHA-256"
        )

    population = _prepare_population(universe, diagnostic)
    groups = _assign_group_split(population, config)
    population = population.merge(
        groups[["calibration_group_key", "frozen_subset"]],
        on="calibration_group_key",
        how="left",
        validate="many_to_one",
    )
    development = _sample_subset(
        population.loc[
            population["frozen_subset"].eq("CALIBRATION_DEVELOPMENT_SET")
        ],
        subset="CALIBRATION_DEVELOPMENT_SET",
        target=config.development_count,
        config=config,
    )
    confirmation = _sample_subset(
        population.loc[
            population["frozen_subset"].eq("BLINDED_CONFIRMATION_SET")
        ],
        subset="BLINDED_CONFIRMATION_SET",
        target=config.confirmation_count,
        config=config,
    )
    selected = pd.concat([development, confirmation], ignore_index=True)
    selected["machine_evidence_stratum"] = selected.apply(
        _machine_strata,
        axis=1,
    )
    selected["safe_view_eligibility"] = False
    selected["same_neighbor_continuity_valid"] = False
    selected["same_neighbor_continuity_status"] = (
        "NOT_CONSTRUCTED_FRAME_LOCAL_V1"
    )
    selected["center_selector"] = "CURRENT_NORMALIZED_CENTER"
    selected["edge_selector"] = "DIAGNOSTIC_MIN_AABB_EDGE"
    selected["calibration_sample_membership"] = True

    selected["_order_key"] = (
        selected["frozen_subset"].astype(str)
        + "|"
        + selected["presentation_order"].astype(str).str.zfill(6)
    )
    selected = selected.sort_values("_order_key").drop(columns=["_order_key"])
    selected = selected.reset_index(drop=True)
    selected["calibration_item_id"] = [
        f"calibration_item_{index:06d}"
        for index in range(1, len(selected) + 1)
    ]
    selected["media_authority_key"] = selected["review_unit_id"].map(
        lambda value: "media_" + _stable_hex(
            config.seed,
            "media_authority",
            str(value),
        )[:24]
    )
    config_hash = calibration_sampling_config_hash(config)
    input_hashes_json = json.dumps(
        dict(sorted(input_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    )

    blinded = selected[
        [
            "calibration_item_id",
            "media_authority_key",
            "frozen_subset",
            "presentation_order",
        ]
    ].copy()
    blinded["presentation_version"] = presentation_version
    blinded["presentation_semantic_hash"] = presentation_semantic_hash
    blinded["sampling_config_hash"] = config_hash
    blinded["semantic_status"] = CALIBRATION_SAMPLE_STATUS
    blinded["producer_sha"] = producer_sha
    blinded["input_hashes_json"] = input_hashes_json

    media_columns = [
        "calibration_item_id",
        "media_authority_key",
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "recording_date",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_pig_history_display_frame_indices",
        "review_pig_history_available_ratio",
    ]
    media = selected[media_columns].copy()
    media["target_frame_count"] = media["display_frame_indices"].map(
        _frame_count
    )
    media["history_frame_count"] = media[
        "review_pig_history_display_frame_indices"
    ].map(_frame_count)
    media["presentation_version"] = presentation_version
    media["presentation_semantic_hash"] = presentation_semantic_hash
    media["semantic_status"] = CALIBRATION_SAMPLE_STATUS
    media["producer_sha"] = producer_sha
    media["input_hashes_json"] = input_hashes_json

    trace_columns = [
        "calibration_item_id",
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "recording_date",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_pig_history_display_frame_indices",
        "neighborhood_evidence_available",
        "frames_with_valid_neighbors",
        "neighbor_valid_ratio",
        "max_concurrent_contact_proxy_count",
        "crowding_ratio",
        "center_selector",
        "edge_selector",
        "center_edge_top1_agreement_ratio",
        "min_edge_distance_over_unit",
        "median_min_edge_distance",
        "any_contact_proxy_ratio",
        "overlap_present_ratio",
        "same_neighbor_continuity_valid",
        "same_neighbor_continuity_status",
        "machine_evidence_stratum",
        "safe_view_eligibility",
        "frozen_subset",
        "presentation_order",
        "calibration_sample_membership",
        *_stratum_columns(),
        "behavior_label",
    ]
    internal = selected[trace_columns].copy()

    checker = audit_interaction_calibration_sample(
        population,
        groups,
        internal,
        blinded,
        media,
        config=config,
    )
    audit = {
        "schema_version": CALIBRATION_SAMPLE_SCHEMA_VERSION,
        "semantic_status": CALIBRATION_SAMPLE_STATUS,
        "authority_role": "DIAGNOSTIC_SAMPLE_NOT_REVIEW_AUTHORITY",
        "producer_sha": producer_sha,
        "sampling_config": asdict(config),
        "sampling_config_hash": config_hash,
        "input_hashes": dict(sorted(input_hashes.items())),
        "presentation_version": presentation_version,
        "presentation_semantic_hash": presentation_semantic_hash,
        "interaction_population_count": int(len(population)),
        "calibration_development_count": int(len(development)),
        "blinded_confirmation_count": int(len(confirmation)),
        "checker": checker,
        "valid": bool(checker["valid"]),
    }
    return InteractionCalibrationSampleResult(
        group_split=groups.reset_index(drop=True),
        internal_trace=internal.reset_index(drop=True),
        blinded_manifest=blinded.reset_index(drop=True),
        media_authority=media.reset_index(drop=True),
        audit=audit,
    )


def audit_interaction_calibration_sample(
    population: pd.DataFrame,
    groups: pd.DataFrame,
    internal: pd.DataFrame,
    blinded: pd.DataFrame,
    media: pd.DataFrame,
    *,
    config: InteractionCalibrationSamplingConfig,
) -> dict[str, Any]:
    """Independently check keys, group isolation, blinding, and coverage."""

    errors: list[str] = []
    internal_ids = internal["review_unit_id"].astype(str)
    if internal_ids.duplicated().any():
        errors.append("duplicate_review_keys")
    development = internal.loc[
        internal["frozen_subset"].eq("CALIBRATION_DEVELOPMENT_SET")
    ]
    confirmation = internal.loc[
        internal["frozen_subset"].eq("BLINDED_CONFIRMATION_SET")
    ]
    if len(development) != config.development_count:
        errors.append("development_count_mismatch")
    if len(confirmation) != config.confirmation_count:
        errors.append("confirmation_count_mismatch")

    group_lookup = groups.set_index("calibration_group_key")["frozen_subset"]
    population_subset = population["calibration_group_key"].map(group_lookup)
    group_counts = pd.DataFrame(
        {
            "group": population["calibration_group_key"],
            "subset": population_subset,
        }
    ).drop_duplicates()
    leakage = int(group_counts["group"].duplicated(keep=False).sum())
    if leakage:
        errors.append(f"group_leakage_count={leakage}")

    public_forbidden = sorted(
        column
        for column in blinded.columns
        if any(
            token in column.casefold()
            for token in _PUBLIC_FORBIDDEN_COLUMN_TOKENS
        )
    )
    if public_forbidden:
        errors.append(f"blinded_manifest_forbidden_columns={public_forbidden}")
    if len(blinded) != len(internal) or len(media) != len(internal):
        errors.append("manifest_row_count_mismatch")
    if set(blinded["calibration_item_id"]) != set(
        internal["calibration_item_id"]
    ):
        errors.append("blinded_internal_item_mismatch")
    if set(media["review_unit_id"]) != set(internal_ids):
        errors.append("media_internal_review_key_mismatch")

    combined_coverage = {
        name: int(internal[name].map(_truth).sum())
        for name in _stratum_columns()
    }
    combined_coverage.update(
        {
            "behavior=fight": int(
                internal["behavior_label"].eq("fight").sum()
            ),
            "behavior=social-nose": int(
                internal["behavior_label"].eq("social-nose").sum()
            ),
            "source=cvat_tracking_xml": int(
                internal["source_type"].eq("cvat_tracking_xml").sum()
            ),
            "source=legacy_recovered": int(
                internal["source_type"].eq("legacy_recovered").sum()
            ),
        }
    )
    available_coverage = {
        name: (
            int(population[name].map(_truth).sum())
            if name in population.columns
            else int(
                (
                    population[
                        "behavior_label"
                        if name.startswith("behavior=")
                        else "source_type"
                    ]
                    == name.split("=", maxsplit=1)[1]
                ).sum()
            )
        )
        for name in combined_coverage
    }
    missing_strata = sorted(
        name
        for name, count in combined_coverage.items()
        if available_coverage[name] > 0 and count == 0
    )
    if missing_strata:
        errors.append(f"required_strata_unrepresented={missing_strata}")
    return {
        "valid": not errors,
        "errors": errors,
        "duplicate_review_keys": int(internal_ids.duplicated().sum()),
        "group_leakage_count": leakage,
        "development_confirmation_overlap": int(
            len(
                set(development["review_unit_id"]).intersection(
                    confirmation["review_unit_id"]
                )
            )
        ),
        "blinded_manifest_forbidden_columns": public_forbidden,
        "combined_stratum_counts": combined_coverage,
        "available_stratum_counts": available_coverage,
        "missing_required_strata": missing_strata,
    }


def wilson_half_width(
    sample_size: int,
    *,
    proportion: float = 0.5,
    z_value: float = 1.959963984540054,
) -> float:
    """Wilson score half-width for a binomial proportion."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    denominator = 1.0 + z_value**2 / sample_size
    numerator = z_value * math.sqrt(
        proportion * (1.0 - proportion) / sample_size
        + z_value**2 / (4.0 * sample_size**2)
    )
    return numerator / denominator


def calibration_sample_size_options() -> pd.DataFrame:
    """Return predeclared budget options; the operator chooses later."""

    options = [
        (
            "MINIMUM_PILOT",
            192,
            120,
            72,
            48,
            "Cannot tightly estimate rare missed-error strata.",
        ),
        (
            "RECOMMENDED_CALIBRATION",
            480,
            300,
            180,
            120,
            "Small source/date cross-strata remain descriptive.",
        ),
        (
            "STRONG_CONFIRMATION_DESIGN",
            800,
            480,
            320,
            200,
            "Higher workload; still not powered for every cross-product.",
        ),
    ]
    records: list[dict[str, object]] = []
    for name, total, development, confirmation, per_stratum, limitation in options:
        records.append(
            {
                "OPTION": name,
                "TOTAL_ITEMS": total,
                "CALIBRATION_DEVELOPMENT_ITEMS": development,
                "BLINDED_CONFIRMATION_ITEMS": confirmation,
                "ITEMS_PER_MAJOR_STRATUM": per_stratum,
                "EXPECTED_REVIEW_TIME_HOURS_AT_3_MIN_PER_ITEM": total / 20.0,
                "EXPECTED_95_PERCENT_CI_HALF_WIDTH_WORST_CASE": (
                    wilson_half_width(per_stratum)
                ),
                "ABILITY_TO_ESTIMATE_CORRECTION_RATE": (
                    "AGGREGATED"
                    if per_stratum >= 96
                    else "PILOT_ONLY"
                ),
                "ABILITY_TO_ESTIMATE_MISSED_ERROR_RATE": (
                    "CONFIRMATORY_AGGREGATED"
                    if confirmation >= 180
                    else "INSUFFICIENT_FOR_SAFETY_CLAIM"
                ),
                "SOURCE_DATE_COVERAGE": "GROUPED_DESIGN_REQUIRED",
                "CROWDING_COVERAGE": "HIGH_AND_LOWER_STRATA_REQUIRED",
                "MAIN_LIMITATION": limitation,
                "STRATUM_ROLE": (
                    "PRIMARY_AGGREGATED;SECONDARY;DESCRIPTIVE_ONLY"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "CALIBRATION_SAMPLE_SCHEMA_VERSION",
    "CALIBRATION_SAMPLE_STATUS",
    "DEFAULT_SAMPLE_SEED",
    "InteractionCalibrationSampleResult",
    "InteractionCalibrationSamplingConfig",
    "InteractionCalibrationSamplingError",
    "audit_interaction_calibration_sample",
    "build_interaction_calibration_sample",
    "calibration_sample_size_options",
    "calibration_sampling_config_hash",
    "wilson_half_width",
]
