"""Reproducible development-only selector learned after frozen review.

The selector never consumes review metadata as model inputs.  It learns a
multiclass reviewed-behavior evidence model from explicitly masked spatial
features and scores a source label by evidence disagreement::

    suspicion = 1 - P(reviewed_behavior == original_behavior | spatial X)

The score is diagnostic only.  A fresh probability-sampled holdout is required
before a future selector can receive validation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_MASK_CONTRACT_VERSION,
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_SCHEMA_HASH,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    PostReviewContractError,
    validate_review_close_authority,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

SELECTOR_SCHEMA_VERSION = "classification_v2.post_review_selector_candidate.v1"
SELECTOR_FEATURE_CONTRACT_VERSION = (
    "classification_v2.post_review_selector_features.v1"
)
SELECTOR_STATUS = "DEVELOPMENT_DIAGNOSTIC_ONLY"
AGGREGATION_NAMES = (
    "masked_mean",
    "masked_std",
    "masked_p10",
    "masked_p50",
    "masked_p90",
    "valid_coverage",
    "valid_trend_per_frame",
)


class PostReviewSelectorContractError(PostReviewContractError):
    """Raised when selector evidence or grouping is not scientifically safe."""


@dataclass(frozen=True, slots=True)
class SelectorFeatureSpec:
    """One numeric feature and the masks required to observe it."""

    feature_name: str
    validity_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectorCandidateConfig:
    """Fixed modest selector configuration; no hyperparameter sweep."""

    seed: int = 20260802
    fold_count: int = 4
    regularization_c: float = 0.1
    max_iter: int = 4000
    review_budgets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

    def validate(self) -> None:
        if self.fold_count < 2:
            raise PostReviewSelectorContractError("selector_fold_count_below_two")
        if not math.isfinite(self.regularization_c) or self.regularization_c <= 0:
            raise PostReviewSelectorContractError("selector_regularization_invalid")
        if self.max_iter <= 0:
            raise PostReviewSelectorContractError("selector_max_iter_invalid")
        if not self.review_budgets:
            raise PostReviewSelectorContractError("selector_review_budgets_empty")
        if any(value <= 0 or value >= 1 for value in self.review_budgets):
            raise PostReviewSelectorContractError("selector_review_budget_invalid")


def canonical_selector_feature_specs() -> tuple[SelectorFeatureSpec, ...]:
    """Return the explicit 46D feature-to-validity mapping."""
    specs: list[SelectorFeatureSpec] = []
    for feature in SPATIAL_PREDICTIVE_FEATURES["bbox_xywh_n"]:
        specs.append(SelectorFeatureSpec(feature, ("geometry_feature_valid",)))
    for feature in SPATIAL_PREDICTIVE_FEATURES["bbox_shape_n"]:
        specs.append(SelectorFeatureSpec(feature, ("geometry_feature_valid",)))

    velocity = {"vx_n_per_second", "vy_n_per_second", "speed_n_per_second"}
    bbox_rate = {
        "bw_rate_n_per_second",
        "bh_rate_n_per_second",
        "area_rate_n_per_second",
        "aspect_ratio_rate_per_second",
    }
    tangential = {"tangential_acceleration_n_per_second2"}
    vector = {
        "ax_n_per_second2",
        "ay_n_per_second2",
        "acceleration_vector_magnitude_n_per_second2",
    }
    for feature in MOTION_FEATURE_NAMES:
        if feature in velocity:
            masks = ("velocity_valid",)
        elif feature in bbox_rate:
            masks = ("bbox_rate_valid",)
        elif feature == "direction_change_rad":
            masks = ("direction_change_valid",)
        elif feature in tangential:
            masks = ("tangential_acceleration_valid",)
        elif feature in vector:
            masks = ("vector_acceleration_valid",)
        else:  # pragma: no cover - schema additions must be classified explicitly
            raise PostReviewSelectorContractError(
                f"unclassified_motion_feature={feature}"
            )
        specs.append(SelectorFeatureSpec(feature, masks))

    for feature in SPATIAL_PREDICTIVE_FEATURES["roi_class_relation"]:
        roi_class = feature.split("_", 2)[1]
        specs.append(
            SelectorFeatureSpec(
                feature,
                ("roi_feature_valid", f"roi_{roi_class}_available"),
            )
        )

    social_masks = {
        "nearest_dist_n": ("social_context_valid", "nearest_distance_available"),
        "nearest_pair_iou": (
            "social_context_valid",
            "nearest_partner_available",
        ),
        "nearest_pair_overlap_ratio": (
            "social_context_valid",
            "nearest_partner_available",
        ),
        "social_density_near_count": (
            "social_context_valid",
            "social_density_available",
        ),
        "social_contact_count": (
            "social_context_valid",
            "social_density_available",
        ),
        "partner_distance_delta_n": ("relative_motion_available",),
        "approach_speed_n_per_second": ("relative_motion_available",),
        "retreat_speed_n_per_second": ("relative_motion_available",),
        "pair_contact_with_nearest": (
            "social_context_valid",
            "nearest_partner_available",
        ),
        "aggression_score_proxy_per_second": ("relative_motion_available",),
    }
    for feature in SPATIAL_PREDICTIVE_FEATURES["social_relation"]:
        masks = social_masks.get(feature)
        if masks is None:  # pragma: no cover - fail closed on schema expansion
            raise PostReviewSelectorContractError(
                f"unclassified_social_feature={feature}"
            )
        specs.append(SelectorFeatureSpec(feature, masks))

    expected = [
        feature
        for group in SPATIAL_PREDICTIVE_FEATURES.values()
        for feature in group
    ]
    observed = [spec.feature_name for spec in specs]
    if observed != expected or len(observed) != 46:
        raise PostReviewSelectorContractError("selector_46d_order_mismatch")
    return tuple(specs)


def selector_feature_contract() -> dict[str, Any]:
    """Return a hash-bound model-input contract for selector-only evidence."""
    specs = canonical_selector_feature_specs()
    payload: dict[str, Any] = {
        "schema_version": SELECTOR_FEATURE_CONTRACT_VERSION,
        "spatial_schema_hash": SPATIAL_SCHEMA_HASH,
        "spatial_mask_contract_version": SPATIAL_MASK_CONTRACT_VERSION,
        "ordered_features": [
            {
                "feature_name": spec.feature_name,
                "validity_columns": list(spec.validity_columns),
            }
            for spec in specs
        ],
        "aggregations": list(AGGREGATION_NAMES),
        "model_x_excludes": [
            "review outcomes",
            "review reasons and ranks",
            "source behavior labels",
            "source provenance",
            "video/date/identity/path fields",
            "sampling probabilities and weights",
        ],
        "score_context_only": ["original_behavior"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["contract_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def build_selector_outcomes(
    *,
    review_close_authority: Mapping[str, Any],
    primary_scope: pd.DataFrame,
    primary_quality: pd.DataFrame,
    control_scope: pd.DataFrame,
    control_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the weighted development population from frozen review files."""
    validate_review_close_authority(review_close_authority)
    expected_primary = int(
        review_close_authority["primary_review"]["scope_rows"]
    )
    expected_control = int(
        review_close_authority["control_review"]["scope_rows"]
    )
    if len(primary_scope) != expected_primary:
        raise PostReviewSelectorContractError("selector_primary_scope_count_drift")
    if len(control_scope) != expected_control:
        raise PostReviewSelectorContractError("selector_control_scope_count_drift")

    primary = _one_review_population(
        primary_scope,
        primary_quality,
        selection_group="SELECTED",
        sampling_weight=pd.Series(1.0, index=primary_scope.index),
    )
    _require_columns(
        control_scope,
        ["post_review_control_sampling_weight"],
        "control_scope",
    )
    control_weight = pd.to_numeric(
        control_scope["post_review_control_sampling_weight"],
        errors="coerce",
    )
    control = _one_review_population(
        control_scope,
        control_quality,
        selection_group="CONTROL",
        sampling_weight=control_weight,
    )
    outcomes = pd.concat([primary, control], ignore_index=True)
    if outcomes["review_unit_id"].duplicated().any():
        raise PostReviewSelectorContractError("selector_primary_control_overlap")
    if outcomes["temporal_unit_key"].duplicated().any():
        raise PostReviewSelectorContractError(
            "selector_duplicate_temporal_unit_key"
        )

    video_dates = outcomes.groupby("video_key")["recording_date"].nunique()
    if video_dates.gt(1).any():
        raise PostReviewSelectorContractError("selector_video_date_conflict")
    technical = outcomes["technical_exclusion"]
    analyzable = outcomes.loc[~technical].copy().reset_index(drop=True)
    if analyzable.empty:
        raise PostReviewSelectorContractError("selector_no_analyzable_outcomes")
    if analyzable["source_label_error"].nunique() != 2:
        raise PostReviewSelectorContractError("selector_binary_target_degenerate")
    if analyzable["reviewed_behavior"].nunique() < 2:
        raise PostReviewSelectorContractError(
            "selector_reviewed_behavior_degenerate"
        )

    audit = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": SELECTOR_STATUS,
        "primary_rows": int(len(primary)),
        "control_rows": int(len(control)),
        "technical_exclusions": int(technical.sum()),
        "analyzable_rows": int(len(analyzable)),
        "confirmed_source_errors": int(analyzable["source_label_error"].sum()),
        "primary_weighted_mass": float(
            analyzable.loc[
                analyzable["selection_group"].eq("SELECTED"),
                "sampling_weight",
            ].sum()
        ),
        "control_weighted_mass": float(
            analyzable.loc[
                analyzable["selection_group"].eq("CONTROL"),
                "sampling_weight",
            ].sum()
        ),
        "combined_weighted_mass": float(analyzable["sampling_weight"].sum()),
        "recording_date_groups": int(analyzable["recording_date"].nunique()),
        "video_groups": int(analyzable["video_key"].nunique()),
        "control_consumed_for_development": True,
        "control_validation_authority_for_candidate": False,
        "fresh_holdout_required": True,
        "weighting_interpretation": (
            "DEVELOPMENT_APPROXIMATION_FROM_TARGETED_CENSUS_PLUS_"
            "INVERSE_PROBABILITY_RESIDUAL_CONTROL"
        ),
    }
    return analyzable, audit


def aggregate_masked_selector_features(
    frame_features: pd.DataFrame,
    *,
    temporal_unit_keys: Sequence[str],
    specs: Sequence[SelectorFeatureSpec] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate 46D frame evidence without observing invalid placeholders."""
    selected_specs = tuple(specs or canonical_selector_feature_specs())
    unit_order = pd.Index([str(value).strip() for value in temporal_unit_keys])
    if unit_order.empty or (unit_order == "").any():
        raise PostReviewSelectorContractError("selector_temporal_keys_invalid")
    if unit_order.duplicated().any():
        raise PostReviewSelectorContractError("selector_temporal_keys_duplicate")

    required = {"temporal_unit_key", "frame_index"}
    for spec in selected_specs:
        required.add(spec.feature_name)
        required.update(spec.validity_columns)
    _require_columns(frame_features, sorted(required), "frame_features")

    work = frame_features.loc[
        frame_features["temporal_unit_key"].astype(str).str.strip().isin(unit_order),
        sorted(required),
    ].copy()
    work["temporal_unit_key"] = (
        work["temporal_unit_key"].astype(str).str.strip()
    )
    frame_number = pd.to_numeric(work["frame_index"], errors="coerce")
    if frame_number.isna().any() or (~np.isfinite(frame_number)).any():
        raise PostReviewSelectorContractError("selector_frame_index_invalid")
    if (frame_number % 1).ne(0).any():
        raise PostReviewSelectorContractError("selector_frame_index_non_integer")
    work["frame_index"] = frame_number.astype(np.int64)
    duplicate_frame = work.duplicated(["temporal_unit_key", "frame_index"])
    if duplicate_frame.any():
        raise PostReviewSelectorContractError(
            f"selector_duplicate_unit_frame={int(duplicate_frame.sum())}"
        )

    present = pd.Index(work["temporal_unit_key"].unique())
    missing_units = unit_order.difference(present)
    if len(missing_units):
        raise PostReviewSelectorContractError(
            f"selector_missing_temporal_units={len(missing_units)}"
        )
    work = work.sort_values(
        ["temporal_unit_key", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_sizes = work.groupby("temporal_unit_key", sort=False).size()

    result = pd.DataFrame(index=unit_order)
    feature_audit: list[dict[str, Any]] = []
    for spec in selected_specs:
        numeric = _numeric_feature(work[spec.feature_name], spec.feature_name)
        valid = np.isfinite(numeric.to_numpy(dtype=float))
        for mask_name in spec.validity_columns:
            valid &= _boolean_mask(work[mask_name], mask_name).to_numpy()
        masked = numeric.where(valid)
        table = pd.DataFrame(
            {
                "temporal_unit_key": work["temporal_unit_key"],
                "frame_index": work["frame_index"],
                "value": masked,
                "valid": valid.astype(np.int64),
            }
        )
        grouped = table.groupby("temporal_unit_key", sort=False)
        prefix = spec.feature_name
        result[f"{prefix}::masked_mean"] = grouped["value"].mean()
        result[f"{prefix}::masked_std"] = grouped["value"].std(ddof=0)
        quantiles = grouped["value"].quantile([0.1, 0.5, 0.9]).unstack()
        result[f"{prefix}::masked_p10"] = quantiles.get(0.1)
        result[f"{prefix}::masked_p50"] = quantiles.get(0.5)
        result[f"{prefix}::masked_p90"] = quantiles.get(0.9)
        valid_count = grouped["valid"].sum()
        result[f"{prefix}::valid_coverage"] = valid_count / group_sizes

        valid_rows = table.loc[table["valid"].eq(1)].copy()
        valid_group = valid_rows.groupby("temporal_unit_key", sort=False)
        first = valid_group[["frame_index", "value"]].first()
        last = valid_group[["frame_index", "value"]].last()
        denominator = last["frame_index"] - first["frame_index"]
        trend = (last["value"] - first["value"]) / denominator.where(
            denominator.gt(0)
        )
        result[f"{prefix}::valid_trend_per_frame"] = trend
        feature_audit.append(
            {
                "feature_name": spec.feature_name,
                "validity_columns": list(spec.validity_columns),
                "total_rows": int(len(table)),
                "valid_rows": int(valid.sum()),
                "invalid_zero_placeholders": int(
                    ((~valid) & numeric.fillna(np.nan).eq(0.0)).sum()
                ),
                "valid_zero_observations": int(
                    (valid & numeric.fillna(np.nan).eq(0.0)).sum()
                ),
            }
        )

    result = result.reindex(unit_order)
    if result.shape[1] != len(selected_specs) * len(AGGREGATION_NAMES):
        raise PostReviewSelectorContractError(
            "selector_aggregate_dimension_mismatch"
        )
    audit = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": SELECTOR_STATUS,
        "temporal_unit_count": int(len(result)),
        "frame_count": int(len(work)),
        "source_feature_count": int(len(selected_specs)),
        "aggregate_feature_count": int(result.shape[1]),
        "feature_contract_hash": selector_feature_contract()["contract_hash"],
        "features": feature_audit,
    }
    result.index.name = "temporal_unit_key"
    return result, audit


def run_post_review_selector_candidate(
    *,
    outcomes: pd.DataFrame,
    aggregates: pd.DataFrame,
    config: SelectorCandidateConfig | None = None,
) -> dict[str, Any]:
    """Cross-fit one fixed evidence model and return diagnostic artifacts."""
    cfg = config or SelectorCandidateConfig()
    cfg.validate()
    required = [
        "review_unit_id",
        "temporal_unit_key",
        "recording_date",
        "video_key",
        "source_type",
        "selection_group",
        "original_behavior",
        "reviewed_behavior",
        "source_label_error",
        "sampling_weight",
    ]
    _require_columns(outcomes, required, "selector_outcomes")
    if aggregates.index.name != "temporal_unit_key":
        raise PostReviewSelectorContractError("selector_aggregate_index_invalid")
    if aggregates.index.duplicated().any():
        raise PostReviewSelectorContractError("selector_aggregate_key_duplicate")

    data = outcomes.merge(
        aggregates.reset_index(),
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    feature_columns = list(aggregates.columns)
    if data[feature_columns].isna().all(axis=1).any():
        raise PostReviewSelectorContractError("selector_aggregate_row_missing")
    if set(data["original_behavior"]) - set(VALID_BEHAVIORS):
        raise PostReviewSelectorContractError("selector_original_behavior_invalid")
    if set(data["reviewed_behavior"]) - set(VALID_BEHAVIORS):
        raise PostReviewSelectorContractError("selector_reviewed_behavior_invalid")

    group_count = int(data["recording_date"].nunique())
    if group_count < cfg.fold_count:
        raise PostReviewSelectorContractError(
            f"selector_date_groups_below_folds={group_count}:{cfg.fold_count}"
        )
    stratify = data["reviewed_behavior"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=cfg.fold_count,
        shuffle=True,
        random_state=cfg.seed,
    )
    x = data[feature_columns].to_numpy(dtype=float)
    reviewed = data["reviewed_behavior"].astype(str).to_numpy()
    weights = data["sampling_weight"].to_numpy(dtype=float)
    if (~np.isfinite(weights)).any() or (weights <= 0).any():
        raise PostReviewSelectorContractError("selector_sampling_weight_invalid")

    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    all_classes = set(data["reviewed_behavior"])
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(x, stratify, groups=data["recording_date"])
    ):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        if set(train["recording_date"]) & set(test["recording_date"]):
            raise PostReviewSelectorContractError("selector_date_group_leakage")
        if set(train["video_key"]) & set(test["video_key"]):
            raise PostReviewSelectorContractError("selector_video_group_leakage")
        if set(train["reviewed_behavior"]) != all_classes:
            raise PostReviewSelectorContractError(
                f"selector_train_class_missing_fold={fold_index}"
            )

        train_weight = weights[train_index]
        preprocessor = _fit_weighted_preprocessor(x[train_index], train_weight)
        train_x = _apply_weighted_preprocessor(x[train_index], preprocessor)
        test_x = _apply_weighted_preprocessor(x[test_index], preprocessor)
        normalized_weight = train_weight / train_weight.mean()
        model = _new_model(cfg)
        model.fit(train_x, reviewed[train_index], sample_weight=normalized_weight)
        probabilities = model.predict_proba(test_x)
        suspicion = _source_label_disagreement(
            probabilities,
            model.classes_,
            test["original_behavior"],
        )
        prior = _weighted_class_prior(
            train["reviewed_behavior"],
            train_weight,
        )
        prior_suspicion = 1.0 - test["original_behavior"].map(prior).to_numpy()
        random_suspicion = np.array(
            [
                _stable_uniform(cfg.seed, review_unit_id)
                for review_unit_id in test["review_unit_id"]
            ],
            dtype=float,
        )
        predicted_behavior = model.classes_[np.argmax(probabilities, axis=1)]
        part = test[required].copy()
        part["fold_id"] = fold_index
        part["selector_suspicion_score"] = suspicion
        part["frequency_baseline_score"] = prior_suspicion
        part["random_baseline_score"] = random_suspicion
        part["evidence_predicted_behavior"] = predicted_behavior
        prediction_parts.append(part)

        fold_manifest = test[
            [
                "review_unit_id",
                "temporal_unit_key",
                "recording_date",
                "video_key",
                "source_type",
                "selection_group",
            ]
        ].copy()
        fold_manifest["fold_id"] = fold_index
        fold_parts.append(fold_manifest)
        fold_audits.append(
            {
                "fold_id": fold_index,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(test_index)),
                "train_date_groups": int(train["recording_date"].nunique()),
                "validation_date_groups": int(
                    test["recording_date"].nunique()
                ),
                "train_video_groups": int(train["video_key"].nunique()),
                "validation_video_groups": int(test["video_key"].nunique()),
                "date_group_overlap": 0,
                "video_group_overlap": 0,
                "preprocessor_hash": _preprocessor_hash(preprocessor),
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["fold_id", "review_unit_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_manifest = pd.concat(fold_parts, ignore_index=True)
    if predictions["review_unit_id"].duplicated().any():
        raise PostReviewSelectorContractError("selector_oof_prediction_duplicate")
    if len(predictions) != len(data):
        raise PostReviewSelectorContractError("selector_oof_prediction_count")

    metrics = _selector_metrics(predictions, cfg.review_budgets)
    full_preprocessor = _fit_weighted_preprocessor(x, weights)
    full_x = _apply_weighted_preprocessor(x, full_preprocessor)
    full_model = _new_model(cfg)
    full_model.fit(
        full_x,
        reviewed,
        sample_weight=weights / weights.mean(),
    )
    expanded_names = [
        *feature_columns,
        *(f"missing::{name}" for name in feature_columns),
    ]
    coefficients = _coefficient_table(full_model, expanded_names)
    formula = _formula_payload(
        full_model,
        full_preprocessor,
        feature_columns,
        cfg,
    )
    leakage_audit = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": "PASS",
        "model_x_feature_count": len(feature_columns),
        "review_fields_entering_model_x": 0,
        "source_label_entering_model_x": 0,
        "source_provenance_entering_model_x": 0,
        "metadata_fields_entering_model_x": 0,
        "sampling_weights_entering_model_x": 0,
        "original_behavior_usage": "POST_PREDICTION_SCORE_CONTEXT_ONLY",
        "date_group_overlap": 0,
        "video_group_overlap": 0,
        "feature_contract_hash": selector_feature_contract()["contract_hash"],
    }
    return {
        "predictions": predictions,
        "fold_manifest": fold_manifest,
        "fold_audits": fold_audits,
        "metrics": metrics,
        "coefficients": coefficients,
        "formula": formula,
        "leakage_audit": leakage_audit,
    }


def _one_review_population(
    scope: pd.DataFrame,
    quality: pd.DataFrame,
    *,
    selection_group: str,
    sampling_weight: pd.Series,
) -> pd.DataFrame:
    scope_columns = [
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "video_key",
        "recording_date",
        "behavior_label",
    ]
    quality_columns = [
        "review_unit_id",
        "original_behavior",
        "reviewed_behavior",
        "label_status",
        "source_label_error_confirmed",
        "error_pattern",
    ]
    _require_columns(scope, scope_columns, f"{selection_group}_scope")
    _require_columns(quality, quality_columns, f"{selection_group}_quality")
    if scope["review_unit_id"].duplicated().any():
        raise PostReviewSelectorContractError(
            f"selector_duplicate_scope_key={selection_group}"
        )
    if quality["review_unit_id"].duplicated().any():
        raise PostReviewSelectorContractError(
            f"selector_duplicate_quality_key={selection_group}"
        )
    if len(scope) != len(quality):
        raise PostReviewSelectorContractError(
            f"selector_scope_quality_count={selection_group}"
        )

    left = scope[scope_columns].copy()
    left["sampling_weight"] = sampling_weight.to_numpy()
    merged = left.merge(
        quality[quality_columns],
        on="review_unit_id",
        how="left",
        validate="one_to_one",
    )
    if merged[quality_columns[1:]].isna().all(axis=1).any():
        raise PostReviewSelectorContractError(
            f"selector_quality_missing={selection_group}"
        )
    if not merged["behavior_label"].astype(str).eq(
        merged["original_behavior"].astype(str)
    ).all():
        raise PostReviewSelectorContractError(
            f"selector_original_label_mismatch={selection_group}"
        )
    status = merged["source_label_error_confirmed"].astype(str).str.strip()
    technical = merged["label_status"].astype(str).eq("TECHNICAL_DEFECT")
    valid_status = status.isin({"YES", "NO"}) | (
        technical & status.eq("NOT_APPLICABLE")
    )
    if not valid_status.all():
        raise PostReviewSelectorContractError(
            f"selector_quality_semantics={selection_group}"
        )
    changed = merged["original_behavior"].astype(str).ne(
        merged["reviewed_behavior"].astype(str)
    )
    if (status.eq("YES") & ~changed).any():
        raise PostReviewSelectorContractError(
            f"selector_error_without_change={selection_group}"
        )
    if (status.eq("NO") & changed).any():
        raise PostReviewSelectorContractError(
            f"selector_change_without_error={selection_group}"
        )
    date = merged["recording_date"].fillna("").astype(str).str.strip()
    if date.eq("").any():
        raise PostReviewSelectorContractError(
            f"selector_recording_date_missing={selection_group}"
        )
    weight = pd.to_numeric(merged["sampling_weight"], errors="coerce")
    if weight.isna().any() or (~np.isfinite(weight)).any() or weight.le(0).any():
        raise PostReviewSelectorContractError(
            f"selector_sampling_weight_invalid={selection_group}"
        )
    return pd.DataFrame(
        {
            "review_unit_id": merged["review_unit_id"].astype(str),
            "temporal_unit_key": merged["temporal_unit_key"].astype(str),
            "recording_date": date,
            "video_key": merged["video_key"].astype(str),
            "source_type": merged["source_type"].astype(str),
            "selection_group": selection_group,
            "original_behavior": merged["original_behavior"].astype(str),
            "reviewed_behavior": merged["reviewed_behavior"].astype(str),
            "source_label_error": status.eq("YES"),
            "label_status": merged["label_status"].astype(str),
            "error_pattern": merged["error_pattern"].astype(str),
            "technical_exclusion": technical,
            "sampling_weight": weight.astype(float),
        }
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PostReviewSelectorContractError(
            f"selector_{name}_missing_columns={','.join(missing)}"
        )


def _numeric_feature(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.strip()
    invalid = text.ne("") & numeric.isna()
    if invalid.any():
        raise PostReviewSelectorContractError(
            f"selector_feature_non_numeric={name}:{int(invalid.sum())}"
        )
    return numeric.astype(float)


def _boolean_mask(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise PostReviewSelectorContractError(
                f"selector_mask_missing={name}:{int(series.isna().sum())}"
            )
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    truth = {"1", "1.0", "true", "yes"}
    falsehood = {"0", "0.0", "false", "no"}
    invalid = ~normalized.isin(truth | falsehood)
    if invalid.any():
        raise PostReviewSelectorContractError(
            f"selector_mask_invalid={name}:{int(invalid.sum())}"
        )
    return normalized.isin(truth)


def _fit_weighted_preprocessor(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    observed = np.isfinite(values)
    weighted_observed = observed * weights[:, None]
    denominator = weighted_observed.sum(axis=0)
    numerator = np.where(observed, values, 0.0) * weights[:, None]
    impute = np.divide(
        numerator.sum(axis=0),
        denominator,
        out=np.zeros(values.shape[1], dtype=float),
        where=denominator > 0,
    )
    filled = np.where(observed, values, impute[None, :])
    expanded = np.concatenate([filled, (~observed).astype(float)], axis=1)
    weight_sum = float(weights.sum())
    center = (expanded * weights[:, None]).sum(axis=0) / weight_sum
    variance = (
        ((expanded - center[None, :]) ** 2) * weights[:, None]
    ).sum(axis=0) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    return {"impute": impute, "center": center, "scale": scale}


def _apply_weighted_preprocessor(
    values: np.ndarray,
    preprocessor: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    filled = np.where(observed, values, preprocessor["impute"][None, :])
    expanded = np.concatenate([filled, (~observed).astype(float)], axis=1)
    transformed = (
        expanded - preprocessor["center"][None, :]
    ) / preprocessor["scale"][None, :]
    if (~np.isfinite(transformed)).any():
        raise PostReviewSelectorContractError("selector_transform_non_finite")
    return transformed


def _new_model(config: SelectorCandidateConfig) -> LogisticRegression:
    return LogisticRegression(
        C=float(config.regularization_c),
        max_iter=int(config.max_iter),
        solver="lbfgs",
        random_state=int(config.seed),
    )


def _source_label_disagreement(
    probabilities: np.ndarray,
    classes: Sequence[str],
    original_behavior: pd.Series,
) -> np.ndarray:
    lookup = {str(label): index for index, label in enumerate(classes)}
    indices: list[int] = []
    for label in original_behavior.astype(str):
        if label not in lookup:
            raise PostReviewSelectorContractError(
                f"selector_source_class_absent_from_train={label}"
            )
        indices.append(lookup[label])
    row = np.arange(len(indices), dtype=np.int64)
    score = 1.0 - probabilities[row, np.asarray(indices, dtype=np.int64)]
    return np.clip(score, 0.0, 1.0)


def _weighted_class_prior(
    labels: pd.Series,
    weights: np.ndarray,
) -> dict[str, float]:
    table = pd.DataFrame(
        {"label": labels.astype(str).to_numpy(), "weight": weights}
    )
    totals = table.groupby("label")["weight"].sum()
    return (totals / totals.sum()).to_dict()


def _stable_uniform(seed: int, key: Any) -> float:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)


def _selector_metrics(
    predictions: pd.DataFrame,
    review_budgets: Sequence[float],
) -> dict[str, Any]:
    score_columns = {
        "candidate": "selector_suspicion_score",
        "frequency_baseline": "frequency_baseline_score",
        "random_baseline": "random_baseline_score",
    }
    global_metrics = {
        name: _binary_score_metrics(
            predictions,
            score_column=column,
            review_budgets=review_budgets,
        )
        for name, column in score_columns.items()
    }
    strata: dict[str, list[dict[str, Any]]] = {}
    for column in (
        "fold_id",
        "selection_group",
        "source_type",
        "original_behavior",
        "recording_date",
    ):
        rows: list[dict[str, Any]] = []
        for value, group in predictions.groupby(column, sort=True, dropna=False):
            rows.append(
                {
                    "stratum": str(value),
                    **_binary_score_metrics(
                        group,
                        score_column="selector_suspicion_score",
                        review_budgets=review_budgets,
                    ),
                }
            )
        strata[column] = rows
    weights = predictions["sampling_weight"].to_numpy(dtype=float)
    behavior_correct = predictions["evidence_predicted_behavior"].eq(
        predictions["reviewed_behavior"]
    )
    behavior_accuracy = float(
        np.average(behavior_correct.to_numpy(dtype=float), weights=weights)
    )
    return {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": SELECTOR_STATUS,
        "scientific_authority": "NONE",
        "control_validation_authority_for_candidate": False,
        "fresh_holdout_required": True,
        "rows": int(len(predictions)),
        "weighted_population_mass": float(weights.sum()),
        "weighted_source_error_prevalence": float(
            np.average(
                predictions["source_label_error"].to_numpy(dtype=float),
                weights=weights,
            )
        ),
        "global": global_metrics,
        "stratified_candidate": strata,
        "reviewed_behavior_evidence_accuracy": behavior_accuracy,
        "interpretation_limits": [
            "All metrics are grouped OOF development diagnostics.",
            "The frozen control is consumed for development and cannot validate this candidate.",
            "A fresh probability-sampled holdout is required before selector promotion.",
            "No behavior-model or paper metric authority follows from this run.",
        ],
    }


def _binary_score_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    review_budgets: Sequence[float],
) -> dict[str, Any]:
    y = frame["source_label_error"].to_numpy(dtype=np.int64)
    score = frame[score_column].to_numpy(dtype=float)
    weight = frame["sampling_weight"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "weighted_mass": float(weight.sum()),
        "weighted_prevalence": float(np.average(y, weights=weight)),
    }
    if len(np.unique(y)) < 2:
        result.update(
            {
                "roc_auc": None,
                "average_precision": None,
                "brier": None,
                "log_loss": None,
                "ece_10": None,
                "threshold_0_5": None,
                "review_budgets": [],
            }
        )
        return result
    result.update(
        {
            "roc_auc": float(roc_auc_score(y, score, sample_weight=weight)),
            "average_precision": float(
                average_precision_score(y, score, sample_weight=weight)
            ),
            "brier": float(brier_score_loss(y, score, sample_weight=weight)),
            "log_loss": float(
                log_loss(
                    y,
                    np.column_stack([1.0 - score, score]),
                    labels=[0, 1],
                    sample_weight=weight,
                )
            ),
            "ece_10": _expected_calibration_error(y, score, weight),
            "threshold_0_5": _threshold_metrics(y, score, weight, 0.5),
            "review_budgets": _review_budget_metrics(
                frame,
                score_column,
                review_budgets,
            ),
        }
    )
    return result


def _threshold_metrics(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = score >= threshold
    positive = y == 1
    true_positive = float(weight[predicted & positive].sum())
    false_positive = float(weight[predicted & ~positive].sum())
    false_negative = float(weight[~predicted & positive].sum())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _review_budget_metrics(
    frame: pd.DataFrame,
    score_column: str,
    review_budgets: Sequence[float],
) -> list[dict[str, Any]]:
    ordered = frame.sort_values(
        [score_column, "review_unit_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    total_positive_mass = float(
        ordered.loc[
            ordered["source_label_error"],
            "sampling_weight",
        ].sum()
    )
    rows: list[dict[str, Any]] = []
    for budget in review_budgets:
        selected_count = max(1, int(math.ceil(len(ordered) * budget)))
        selected = ordered.iloc[:selected_count]
        selected_mass = float(selected["sampling_weight"].sum())
        captured_mass = float(
            selected.loc[
                selected["source_label_error"],
                "sampling_weight",
            ].sum()
        )
        rows.append(
            {
                "item_budget_fraction": float(budget),
                "selected_items": selected_count,
                "weighted_precision": _safe_ratio(captured_mass, selected_mass),
                "weighted_error_recall": _safe_ratio(
                    captured_mass,
                    total_positive_mass,
                ),
            }
        )
    return rows


def _expected_calibration_error(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(score, edges[1:-1], right=True), 0, 9)
    total = float(weight.sum())
    error = 0.0
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        mass = float(weight[mask].sum())
        confidence = float(np.average(score[mask], weights=weight[mask]))
        observed = float(np.average(y[mask], weights=weight[mask]))
        error += mass / total * abs(confidence - observed)
    return float(error)


def _coefficient_table(
    model: LogisticRegression,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_index, behavior in enumerate(model.classes_):
        for feature_index, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "reviewed_behavior_class": str(behavior),
                    "transformed_feature": feature_name,
                    "coefficient": float(model.coef_[class_index, feature_index]),
                }
            )
    return pd.DataFrame(rows)


def _formula_payload(
    model: LogisticRegression,
    preprocessor: Mapping[str, np.ndarray],
    raw_feature_names: Sequence[str],
    config: SelectorCandidateConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": SELECTOR_STATUS,
        "equations": {
            "masked_aggregate": (
                "A[u,j]=aggregate({x[u,t,j]:m[u,t,j]=1}); "
                "invalid numeric placeholders are unobserved"
            ),
            "evidence_model": (
                "P(Y=k|X)=softmax(intercept[k]+coef[k]^T Z)"
            ),
            "selector_score": "S(u)=1-P(Y=original_behavior[u]|X[u])",
            "control_weight": "w(u)=1/inclusion_probability(u)",
        },
        "config": {
            "seed": config.seed,
            "fold_count": config.fold_count,
            "regularization_c": config.regularization_c,
            "max_iter": config.max_iter,
        },
        "raw_aggregate_feature_names": list(raw_feature_names),
        "transformed_feature_names": [
            *raw_feature_names,
            *(f"missing::{name}" for name in raw_feature_names),
        ],
        "imputation_values": preprocessor["impute"].tolist(),
        "standardization_center": preprocessor["center"].tolist(),
        "standardization_scale": preprocessor["scale"].tolist(),
        "classes": [str(value) for value in model.classes_],
        "intercepts": model.intercept_.tolist(),
        "coefficients": model.coef_.tolist(),
        "source_label_is_model_input": False,
        "source_label_is_score_context": True,
        "review_outcomes_are_model_input": False,
        "fresh_holdout_required": True,
    }


def _preprocessor_hash(preprocessor: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("impute", "center", "scale"):
        array = np.ascontiguousarray(preprocessor[name], dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


__all__ = [
    "AGGREGATION_NAMES",
    "PostReviewSelectorContractError",
    "SELECTOR_FEATURE_CONTRACT_VERSION",
    "SELECTOR_SCHEMA_VERSION",
    "SELECTOR_STATUS",
    "SelectorCandidateConfig",
    "SelectorFeatureSpec",
    "aggregate_masked_selector_features",
    "build_selector_outcomes",
    "canonical_selector_feature_specs",
    "run_post_review_selector_candidate",
    "selector_feature_contract",
]
