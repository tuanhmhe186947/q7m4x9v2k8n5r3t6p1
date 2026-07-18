"""Paired native-unit evaluation for legacy-16f temporal sampling views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    CLASS_GROUPS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development.temporal_sampling_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development.temporal_sampling_decision.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
FULL_SCOPE = "full_development_confirmation"
VIEW_IDS = (
    "c6_contiguous_centered",
    "c8_contiguous_centered",
    "s6_uniform_span16",
)
PAIR_SPECS = {
    "primary_s6_vs_c6": (
        "s6_uniform_span16",
        "c6_contiguous_centered",
        "temporal_sampling_span_at_fixed_six_frames",
    ),
    "secondary_c8_vs_c6": (
        "c8_contiguous_centered",
        "c6_contiguous_centered",
        "sequence_length_at_contiguous_centered_sampling",
    ),
}
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33
METADATA_COLUMNS = (
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "target_index",
)


def evaluate_temporal_sampling_predictions(
    predictions: dict[str, pd.DataFrame],
    *,
    iterations: int,
    seed: int,
    maximum_rare_macro_f1_drop: float,
    enforce_project_counts: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate C6, C8, and S6 on one exact native-unit universe."""

    if iterations < 1000:
        raise ValueError("temporal sampling bootstrap requires at least 1000 draws")
    if maximum_rare_macro_f1_drop < 0.0:
        raise ValueError("maximum rare macro-F1 drop must be nonnegative")
    ordered = _validate_prediction_universe(
        predictions,
        enforce_project_counts=enforce_project_counts,
    )
    metrics = {
        view_id: _single_view_metrics(frame)
        for view_id, frame in ordered.items()
    }
    comparisons: dict[str, dict[str, Any]] = {}
    per_class_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    for pair_index, (pair_id, spec) in enumerate(PAIR_SPECS.items()):
        candidate_id, baseline_id, changed_factor = spec
        comparison, per_class, groups = _compare_pair(
            ordered[candidate_id],
            ordered[baseline_id],
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            pair_id=pair_id,
            changed_factor=changed_factor,
            iterations=iterations,
            seed=seed + pair_index,
        )
        comparisons[pair_id] = comparison
        per_class_parts.append(per_class)
        group_parts.append(groups)
    per_class_frame = pd.concat(per_class_parts, ignore_index=True)
    group_frame = pd.concat(group_parts, ignore_index=True)
    confusion_frame = _confusion_table(ordered)
    ranking = _rank_views(metrics)
    decision = _make_decision(
        ranking,
        comparisons,
        maximum_rare_macro_f1_drop=maximum_rare_macro_f1_drop,
    )
    universe = ordered[VIEW_IDS[0]][list(METADATA_COLUMNS)]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_LEGACY_TEMPORAL_SAMPLING_PAIRED_DECISION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "common_native_universe": {
            "native_units": int(len(universe)),
            "video_clusters": int(universe["video_key"].astype(str).nunique()),
            "native_unit_sha256": _mapping_hash(
                universe,
                ["temporal_unit_key"],
            ),
            "metadata_mapping_sha256": _mapping_hash(
                universe,
                list(METADATA_COLUMNS),
            ),
            "one_sequence_per_native_unit": True,
            "outer_holdout_rows": 0,
        },
        "view_metrics": metrics,
        "ranking": ranking,
        "paired_comparisons": comparisons,
        "decision": decision,
        "interpretation_boundary": [
            "legacy_16f_unreviewed_development_only",
            "one_development_validation_split_not_full_oof",
            "native_unit_is_the_complete_16_frame_burst",
            "model_inputs_are_c6_c8_or_s6_not_contiguous_t16",
            "historical_sliding_t6_is_context_only_due_four_windows_per_native",
            "rare_class_effects_require_support_aware_interpretation",
            "repeat_on_frozen_merged_reviewed_data_before_q2_claims",
        ],
        "errors": [],
        "valid": True,
    }
    return result, per_class_frame, group_frame, confusion_frame


def write_temporal_sampling_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Load hash-bound runs, evaluate them, and write exclusive evidence."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    _verify_file_spec(root, config["implementation"], "implementation")
    full_config_path = _verify_file_spec(
        root,
        config["full_training_config"],
        "full training config",
    )
    full_config = _read_json(full_config_path)
    _validate_full_config(full_config)
    short_gate_path = _verify_file_spec(
        root,
        config["short_matrix_gate"],
        "short matrix gate",
    )
    _validate_short_gate(_read_json(short_gate_path), full_config)
    predictions: dict[str, pd.DataFrame] = {}
    run_summaries: dict[str, dict[str, Any]] = {}
    for view_id in VIEW_IDS:
        frame, summary = _load_run_packet(
            root,
            view_id=view_id,
            spec=config["runs"][view_id],
            full_config_sha256=file_sha256(full_config_path),
        )
        predictions[view_id] = frame
        run_summaries[view_id] = summary
    _validate_equal_optimizer_exposure(run_summaries)
    contract = config["analysis_contract"]
    result, per_class, groups, confusion = evaluate_temporal_sampling_predictions(
        predictions,
        iterations=int(contract["bootstrap_iterations"]),
        seed=int(contract["bootstrap_seed"]),
        maximum_rare_macro_f1_drop=float(
            contract["maximum_rare_macro_f1_drop"]
        ),
    )
    output_paths = {
        name: _resolve_inside(root, value)
        for name, value in config["output"].items()
    }
    _write_csv_exclusive(output_paths["per_class_csv"], per_class)
    _write_csv_exclusive(output_paths["group_metrics_csv"], groups)
    _write_csv_exclusive(output_paths["confusion_csv"], confusion)
    result.update(
        {
            "config_path": str(resolved_config),
            "config_sha256": file_sha256(resolved_config),
            "full_training_config": _bound_file(full_config_path),
            "short_matrix_gate": _bound_file(short_gate_path),
            "runs": run_summaries,
            "analysis_contract": dict(contract),
            "artifacts": {
                "per_class_csv": _bound_table(
                    output_paths["per_class_csv"],
                    per_class,
                ),
                "group_metrics_csv": _bound_table(
                    output_paths["group_metrics_csv"],
                    groups,
                ),
                "confusion_csv": _bound_table(
                    output_paths["confusion_csv"],
                    confusion,
                ),
            },
            "optimizer_steps_executed_by_evaluator": 0,
            "source_media_reads": 0,
            "outer_holdout_rows_loaded": 0,
        }
    )
    _write_json_exclusive(output_paths["summary_json"], result)
    return output_paths["summary_json"], result


def _validate_prediction_universe(
    predictions: dict[str, pd.DataFrame],
    *,
    enforce_project_counts: bool,
) -> dict[str, pd.DataFrame]:
    if set(predictions) != set(VIEW_IDS):
        raise ValueError("temporal sampling prediction view set differs")
    required = set(METADATA_COLUMNS) | {
        "predicted_index",
        "predicted_label",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "temporal_sampling_view_id",
        *(_probability_column(label) for label in VALID_BEHAVIORS),
    }
    ordered: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for view_id in VIEW_IDS:
        frame = predictions[view_id].copy(deep=True)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{view_id} prediction columns missing={missing}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{view_id} contains duplicate native units")
        frame = frame.sort_values(
            "temporal_unit_key",
            kind="mergesort",
        ).reset_index(drop=True)
        if enforce_project_counts and len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{view_id} native units={len(frame)}")
        clusters = frame["video_key"].astype(str).nunique()
        if enforce_project_counts and clusters != EXPECTED_VIDEO_CLUSTERS:
            raise ValueError(f"{view_id} video clusters={clusters}")
        _validate_prediction_rows(frame, view_id)
        metadata = frame[list(METADATA_COLUMNS)].astype(str)
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"{view_id} native metadata universe differs")
        ordered[view_id] = frame
    return ordered


def _validate_prediction_rows(frame: pd.DataFrame, view_id: str) -> None:
    labels = list(VALID_BEHAVIORS)
    if set(frame["behavior_label"].astype(str)) - set(labels):
        raise ValueError(f"{view_id} contains unsupported true labels")
    if set(frame["predicted_label"].astype(str)) - set(labels):
        raise ValueError(f"{view_id} contains unsupported predicted labels")
    if not frame["training_scope"].astype(str).eq(FULL_SCOPE).all():
        raise ValueError(f"{view_id} training scope differs")
    if not frame["lineage_scope"].astype(str).eq(LINEAGE_SCOPE).all():
        raise ValueError(f"{view_id} lineage scope differs")
    reviewed = frame["human_review_complete"].astype(str).str.lower()
    if not reviewed.isin({"false", "0"}).all():
        raise ValueError(f"{view_id} incorrectly claims completed review")
    if not frame["temporal_sampling_view_id"].astype(str).eq(view_id).all():
        raise ValueError(f"{view_id} prediction view metadata differs")
    expected_target = frame["behavior_label"].astype(str).map(
        {label: index for index, label in enumerate(labels)}
    )
    if not np.array_equal(
        frame["target_index"].to_numpy(dtype=np.int64),
        expected_target.to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"{view_id} target index differs from label order")
    probabilities = frame[
        [_probability_column(label) for label in labels]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{view_id} contains nonfinite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{view_id} probability rows do not sum to one")
    argmax = probabilities.argmax(axis=1)
    if not np.array_equal(
        frame["predicted_index"].to_numpy(dtype=np.int64),
        argmax,
    ):
        raise ValueError(f"{view_id} predicted index differs from probabilities")
    expected_predicted = np.asarray(labels, dtype=object)[argmax]
    if not np.array_equal(
        frame["predicted_label"].astype(str).to_numpy(),
        expected_predicted,
    ):
        raise ValueError(f"{view_id} predicted label differs from probabilities")


def _single_view_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labels = list(VALID_BEHAVIORS)
    y_true = frame["behavior_label"].astype(str).to_numpy()
    y_pred = frame["predicted_label"].astype(str).to_numpy()
    per_class = {
        label: _hard_class_metrics(y_true, y_pred, label)
        for label in labels
    }
    supports = np.asarray(
        [per_class[label]["support"] for label in labels],
        dtype=np.float64,
    )
    f1_values = np.asarray(
        [per_class[label]["f1"] for label in labels],
        dtype=np.float64,
    )
    recall_values = np.asarray(
        [per_class[label]["recall"] for label in labels],
        dtype=np.float64,
    )
    true_probabilities = _true_probabilities(frame)
    groups = {
        group: float(np.mean([per_class[label]["f1"] for label in members]))
        for group, members in CLASS_GROUPS.items()
    }
    return {
        "native_units": int(len(frame)),
        "video_clusters": int(frame["video_key"].astype(str).nunique()),
        "macro_f1_global_10_class": float(f1_values.mean()),
        "weighted_f1": float(np.average(f1_values, weights=supports)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_recall_global_10_class": float(recall_values.mean()),
        "nll": float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()),
        "group_macro_f1": groups,
        "per_class": per_class,
    }


def _compare_pair(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    candidate_id: str,
    baseline_id: str,
    pair_id: str,
    changed_factor: str,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidate_metrics = _single_view_metrics(candidate)
    baseline_metrics = _single_view_metrics(baseline)
    bootstrap = _paired_video_bootstrap(
        candidate,
        baseline,
        iterations=iterations,
        seed=seed,
    )
    per_class_rows: list[dict[str, Any]] = []
    for label in VALID_BEHAVIORS:
        left = candidate_metrics["per_class"][label]
        right = baseline_metrics["per_class"][label]
        true_mask = candidate["behavior_label"].astype(str).eq(label)
        candidate_correct = candidate["predicted_label"].astype(str).eq(label)
        baseline_correct = baseline["predicted_label"].astype(str).eq(label)
        class_bootstrap = bootstrap["per_class"][label]
        per_class_rows.append(
            {
                "pair_id": pair_id,
                "changed_factor": changed_factor,
                "candidate_view": candidate_id,
                "baseline_view": baseline_id,
                "behavior_label": label,
                "support": int(left["support"]),
                "reliability_tier": _reliability_tier(int(left["support"])),
                "candidate_precision": float(left["precision"]),
                "baseline_precision": float(right["precision"]),
                "precision_delta": float(left["precision"] - right["precision"]),
                "candidate_recall": float(left["recall"]),
                "baseline_recall": float(right["recall"]),
                "recall_delta": float(left["recall"] - right["recall"]),
                "candidate_f1": float(left["f1"]),
                "baseline_f1": float(right["f1"]),
                "f1_delta": float(left["f1"] - right["f1"]),
                "f1_delta_ci_low": class_bootstrap["f1_delta_ci_low"],
                "f1_delta_ci_high": class_bootstrap["f1_delta_ci_high"],
                "candidate_true_nll": _class_true_nll(candidate, label),
                "baseline_true_nll": _class_true_nll(baseline, label),
                "true_nll_delta": class_bootstrap["true_nll_delta"],
                "true_nll_delta_ci_low": class_bootstrap[
                    "true_nll_delta_ci_low"
                ],
                "true_nll_delta_ci_high": class_bootstrap[
                    "true_nll_delta_ci_high"
                ],
                "candidate_only_correct": int(
                    (true_mask & candidate_correct & ~baseline_correct).sum()
                ),
                "baseline_only_correct": int(
                    (true_mask & ~candidate_correct & baseline_correct).sum()
                ),
            }
        )
    group_rows = []
    for group, members in CLASS_GROUPS.items():
        candidate_value = float(candidate_metrics["group_macro_f1"][group])
        baseline_value = float(baseline_metrics["group_macro_f1"][group])
        interval = bootstrap["groups"][group]
        group_rows.append(
            {
                "pair_id": pair_id,
                "changed_factor": changed_factor,
                "candidate_view": candidate_id,
                "baseline_view": baseline_id,
                "group": group,
                "classes_json": json.dumps(list(members), separators=(",", ":")),
                "candidate_macro_f1": candidate_value,
                "baseline_macro_f1": baseline_value,
                "macro_f1_delta": candidate_value - baseline_value,
                "macro_f1_delta_ci_low": interval["macro_f1_delta_ci_low"],
                "macro_f1_delta_ci_high": interval["macro_f1_delta_ci_high"],
            }
        )
    global_delta = {
        "macro_f1_global_10_class": float(
            candidate_metrics["macro_f1_global_10_class"]
            - baseline_metrics["macro_f1_global_10_class"]
        ),
        "weighted_f1": float(
            candidate_metrics["weighted_f1"] - baseline_metrics["weighted_f1"]
        ),
        "accuracy": float(candidate_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "macro_recall_global_10_class": float(
            candidate_metrics["macro_recall_global_10_class"]
            - baseline_metrics["macro_recall_global_10_class"]
        ),
        "nll": float(candidate_metrics["nll"] - baseline_metrics["nll"]),
    }
    comparison = {
        "pair_id": pair_id,
        "changed_factor": changed_factor,
        "candidate_view": candidate_id,
        "baseline_view": baseline_id,
        "paired_native_units": int(len(candidate)),
        "paired_video_clusters": int(candidate["video_key"].astype(str).nunique()),
        "candidate_metrics": _global_metric_subset(candidate_metrics),
        "baseline_metrics": _global_metric_subset(baseline_metrics),
        "delta_candidate_minus_baseline": global_delta,
        "video_cluster_bootstrap": {
            key: value for key, value in bootstrap.items() if key != "per_class"
        },
        "paired_outcomes": _paired_outcomes(candidate, baseline),
        "positive_point_f1_classes": [
            row["behavior_label"]
            for row in per_class_rows
            if float(row["f1_delta"]) > 0.0
        ],
        "negative_point_f1_classes": [
            row["behavior_label"]
            for row in per_class_rows
            if float(row["f1_delta"]) < 0.0
        ],
        "per_class": bootstrap["per_class"],
    }
    return comparison, pd.DataFrame(per_class_rows), pd.DataFrame(group_rows)


def _paired_video_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    clusters = sorted(candidate["video_key"].astype(str).unique())
    cluster_indices = {
        cluster: np.flatnonzero(
            candidate["video_key"].astype(str).eq(cluster).to_numpy()
        )
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    global_samples = {
        "macro_f1": np.empty(iterations, dtype=np.float64),
        "accuracy": np.empty(iterations, dtype=np.float64),
        "nll": np.empty(iterations, dtype=np.float64),
    }
    group_samples = {
        group: np.empty(iterations, dtype=np.float64)
        for group in CLASS_GROUPS
    }
    class_f1_samples = {
        label: np.empty(iterations, dtype=np.float64)
        for label in VALID_BEHAVIORS
    }
    class_nll_samples = {
        label: np.full(iterations, np.nan, dtype=np.float64)
        for label in VALID_BEHAVIORS
    }
    for iteration in range(iterations):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate(
            [cluster_indices[str(cluster)] for cluster in sampled_clusters]
        )
        left = candidate.iloc[indices]
        right = baseline.iloc[indices]
        left_metrics = _single_view_metrics(left)
        right_metrics = _single_view_metrics(right)
        global_samples["macro_f1"][iteration] = (
            left_metrics["macro_f1_global_10_class"]
            - right_metrics["macro_f1_global_10_class"]
        )
        global_samples["accuracy"][iteration] = (
            left_metrics["accuracy"] - right_metrics["accuracy"]
        )
        global_samples["nll"][iteration] = left_metrics["nll"] - right_metrics["nll"]
        for group in CLASS_GROUPS:
            group_samples[group][iteration] = (
                left_metrics["group_macro_f1"][group]
                - right_metrics["group_macro_f1"][group]
            )
        for label in VALID_BEHAVIORS:
            class_f1_samples[label][iteration] = (
                left_metrics["per_class"][label]["f1"]
                - right_metrics["per_class"][label]["f1"]
            )
            if left_metrics["per_class"][label]["support"] > 0:
                class_nll_samples[label][iteration] = (
                    _class_true_nll(left, label) - _class_true_nll(right, label)
                )
    observed_left = _single_view_metrics(candidate)
    observed_right = _single_view_metrics(baseline)
    observed_macro_f1_delta = float(
        observed_left["macro_f1_global_10_class"]
        - observed_right["macro_f1_global_10_class"]
    )
    per_class = {}
    for label in VALID_BEHAVIORS:
        f1_values = class_f1_samples[label]
        nll_values = class_nll_samples[label]
        valid_nll = nll_values[np.isfinite(nll_values)]
        per_class[label] = {
            "support": int(observed_left["per_class"][label]["support"]),
            "f1_delta": float(
                observed_left["per_class"][label]["f1"]
                - observed_right["per_class"][label]["f1"]
            ),
            "f1_delta_ci_low": float(np.quantile(f1_values, 0.025)),
            "f1_delta_ci_high": float(np.quantile(f1_values, 0.975)),
            "true_nll_delta": float(
                _class_true_nll(candidate, label)
                - _class_true_nll(baseline, label)
            ),
            "true_nll_delta_ci_low": (
                float(np.quantile(valid_nll, 0.025)) if len(valid_nll) else None
            ),
            "true_nll_delta_ci_high": (
                float(np.quantile(valid_nll, 0.975)) if len(valid_nll) else None
            ),
            "valid_nll_bootstrap_draws": int(len(valid_nll)),
        }
    return {
        "method": "paired_video_cluster_bootstrap_percentile",
        "cluster_unit": "video_key",
        "cluster_count": len(clusters),
        "paired_native_units": int(len(candidate)),
        "iterations": iterations,
        "seed": seed,
        "outer_predictions_used_for_model_selection": False,
        "macro_f1_delta": observed_macro_f1_delta,
        "bootstrap_mean_macro_f1_delta": float(
            global_samples["macro_f1"].mean()
        ),
        "macro_f1_delta_ci_low": float(
            np.quantile(global_samples["macro_f1"], 0.025)
        ),
        "macro_f1_delta_ci_high": float(
            np.quantile(global_samples["macro_f1"], 0.975)
        ),
        "accuracy_delta_ci_low": float(
            np.quantile(global_samples["accuracy"], 0.025)
        ),
        "accuracy_delta_ci_high": float(
            np.quantile(global_samples["accuracy"], 0.975)
        ),
        "nll_delta_ci_low": float(np.quantile(global_samples["nll"], 0.025)),
        "nll_delta_ci_high": float(np.quantile(global_samples["nll"], 0.975)),
        "bootstrap_fraction_macro_f1_delta_le_zero": float(
            np.mean(global_samples["macro_f1"] <= 0.0)
        ),
        "groups": {
            group: {
                "macro_f1_delta_ci_low": float(np.quantile(values, 0.025)),
                "macro_f1_delta_ci_high": float(np.quantile(values, 0.975)),
            }
            for group, values in group_samples.items()
        },
        "per_class": per_class,
    }


def _make_decision(
    ranking: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    *,
    maximum_rare_macro_f1_drop: float,
) -> dict[str, Any]:
    promotion: dict[str, dict[str, Any]] = {}
    for pair_id, comparison in comparisons.items():
        global_delta = comparison["delta_candidate_minus_baseline"]
        bootstrap = comparison["video_cluster_bootstrap"]
        rare = bootstrap["groups"]["rare"]
        rare_delta = next(
            row["macro_f1_delta"]
            for row in _comparison_group_rows(comparison)
            if row["group"] == "rare"
        )
        criteria = {
            "point_macro_f1_delta_positive": (
                global_delta["macro_f1_global_10_class"] > 0.0
            ),
            "macro_f1_ci_low_positive": bootstrap["macro_f1_delta_ci_low"] > 0.0,
            "rare_macro_f1_drop_within_limit": (
                rare_delta >= -maximum_rare_macro_f1_drop
            ),
            "rare_macro_f1_interval_reported": (
                rare["macro_f1_delta_ci_low"] is not None
            ),
        }
        promotion[pair_id] = {
            "candidate_view": comparison["candidate_view"],
            "baseline_view": comparison["baseline_view"],
            "criteria": criteria,
            "promote": all(criteria.values()),
        }
    promoted = [
        value["candidate_view"]
        for value in promotion.values()
        if value["promote"]
    ]
    selected = ranking[0]["view_id"] if ranking[0]["view_id"] in promoted else VIEW_IDS[0]
    return {
        "decision": (
            "RETAIN_C6_CONTIGUOUS_CENTERED_AS_ONE_SEQUENCE_LEGACY_WORKING_VIEW"
            if selected == "c6_contiguous_centered"
            else "PROMOTE_CONTROLLED_TEMPORAL_SAMPLING_VIEW"
        ),
        "selected_working_view": selected,
        "promotion_gates": promotion,
        "maximum_rare_macro_f1_drop": maximum_rare_macro_f1_drop,
        "contiguous_t16_model_input_tested": False,
        "historical_t6_sliding_used_as_paired_baseline": False,
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
    }


def _comparison_group_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = comparison["candidate_metrics"]["group_macro_f1"]
    baseline = comparison["baseline_metrics"]["group_macro_f1"]
    return [
        {
            "group": group,
            "macro_f1_delta": float(candidate[group] - baseline[group]),
        }
        for group in CLASS_GROUPS
    ]


def _rank_views(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "view_id": view_id,
            "rank": 0,
            **_global_metric_subset(value),
        }
        for view_id, value in metrics.items()
    ]
    rows.sort(
        key=lambda row: (
            -float(row["macro_f1_global_10_class"]),
            float(row["nll"]),
            VIEW_IDS.index(str(row["view_id"])),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _global_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "native_units": metrics["native_units"],
        "video_clusters": metrics["video_clusters"],
        "macro_f1_global_10_class": metrics["macro_f1_global_10_class"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "macro_recall_global_10_class": metrics[
            "macro_recall_global_10_class"
        ],
        "nll": metrics["nll"],
        "group_macro_f1": dict(metrics["group_macro_f1"]),
    }


def _hard_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float | int]:
    true = y_true == label
    predicted = y_pred == label
    tp = int(np.sum(true & predicted))
    fp = int(np.sum(~true & predicted))
    fn = int(np.sum(true & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": int(true.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _true_probabilities(frame: pd.DataFrame) -> np.ndarray:
    labels = list(VALID_BEHAVIORS)
    probabilities = frame[
        [_probability_column(label) for label in labels]
    ].to_numpy(dtype=np.float64)
    targets = frame["target_index"].to_numpy(dtype=np.int64)
    return probabilities[np.arange(len(frame)), targets]


def _class_true_nll(frame: pd.DataFrame, label: str) -> float:
    selected = frame.loc[frame["behavior_label"].astype(str).eq(label)]
    if selected.empty:
        return float("nan")
    probabilities = selected[_probability_column(label)].to_numpy(dtype=np.float64)
    return float(-np.log(np.clip(probabilities, 1e-12, 1.0)).mean())


def _paired_outcomes(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, int]:
    true = candidate["behavior_label"].astype(str)
    candidate_correct = candidate["predicted_label"].astype(str).eq(true)
    baseline_correct = baseline["predicted_label"].astype(str).eq(true)
    return {
        "both_correct": int((candidate_correct & baseline_correct).sum()),
        "candidate_only_correct": int((candidate_correct & ~baseline_correct).sum()),
        "baseline_only_correct": int((~candidate_correct & baseline_correct).sum()),
        "both_incorrect": int((~candidate_correct & ~baseline_correct).sum()),
    }


def _confusion_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for view_id, frame in predictions.items():
        true = frame["behavior_label"].astype(str)
        predicted = frame["predicted_label"].astype(str)
        for true_label in VALID_BEHAVIORS:
            for predicted_label in VALID_BEHAVIORS:
                rows.append(
                    {
                        "view_id": view_id,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(
                            (true.eq(true_label) & predicted.eq(predicted_label)).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _load_run_packet(
    root: Path,
    *,
    view_id: str,
    spec: dict[str, Any],
    full_config_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_result_path = _verify_file_spec(root, spec["run_result"], f"{view_id} result")
    artifact_path = _verify_file_spec(
        root,
        spec["artifact_manifest"],
        f"{view_id} artifact manifest",
    )
    result = _read_json(run_result_path)
    if result.get("status") != "completed" or result.get("valid") is not True:
        raise ValueError(f"{view_id} run is not valid and completed")
    if result.get("errors") != [] or result.get("view_id") != view_id:
        raise ValueError(f"{view_id} run result metadata differs")
    if result.get("config_sha256") != full_config_sha256:
        raise ValueError(f"{view_id} full config hash differs")
    manifest = _read_json(artifact_path)
    artifacts = _verify_artifact_manifest(root, manifest, view_id=view_id)
    prediction_path = artifacts["validation_native_predictions.csv"]
    prediction_manifest = _read_json(artifacts["prediction_manifest.json"])
    if prediction_manifest.get("prediction_sha256") != file_sha256(prediction_path):
        raise ValueError(f"{view_id} prediction manifest hash differs")
    if prediction_manifest.get("native_unit_rows") != EXPECTED_NATIVE_UNITS:
        raise ValueError(f"{view_id} prediction manifest native count differs")
    if prediction_manifest.get("outer_holdout_rows") != 0:
        raise ValueError(f"{view_id} prediction manifest exposes outer holdout")
    if prediction_manifest.get("view_id") != view_id:
        raise ValueError(f"{view_id} prediction manifest view differs")
    epochs = pd.read_csv(artifacts["epoch_metrics.csv"])
    if epochs.empty or int(epochs.iloc[-1]["optimizer_steps_cumulative"]) <= 0:
        raise ValueError(f"{view_id} epoch metrics are incomplete")
    frame = pd.read_csv(prediction_path)
    summary = {
        "view_id": view_id,
        "run_result": _bound_file(run_result_path),
        "artifact_manifest": _bound_file(artifact_path),
        "predictions": _bound_file(prediction_path),
        "prediction_manifest": _bound_file(artifacts["prediction_manifest.json"]),
        "epoch_metrics": _bound_file(artifacts["epoch_metrics.csv"]),
        "optimizer_steps": int(epochs.iloc[-1]["optimizer_steps_cumulative"]),
        "train_native_units": int(epochs.iloc[-1]["train_native_units"]),
        "validation_native_units": int(epochs.iloc[-1]["validation_native_units"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "parameter_sha256": str(result["parameter_sha256"]),
        "selection_native_unit_sha256": str(
            result["selection_native_unit_sha256"]
        ),
        "errors": [],
        "valid": True,
    }
    return frame, summary


def _verify_artifact_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    view_id: str,
) -> dict[str, Path]:
    if manifest.get("valid") is not True or manifest.get("errors") != []:
        raise ValueError(f"{view_id} artifact manifest is invalid")
    if manifest.get("view_id") != view_id:
        raise ValueError(f"{view_id} artifact manifest view differs")
    paths: dict[str, Path] = {}
    for artifact in manifest.get("artifacts", []):
        name = str(artifact["name"])
        path = _resolve_inside(root, str(artifact["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"{view_id} artifact missing: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"{view_id} artifact hash differs: {name}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"{view_id} artifact size differs: {name}")
        if name in paths:
            raise ValueError(f"{view_id} duplicate artifact name: {name}")
        paths[name] = path
    required = {
        "validation_native_predictions.csv",
        "prediction_manifest.json",
        "epoch_metrics.csv",
    }
    if not required.issubset(paths):
        raise ValueError(f"{view_id} required artifacts missing={sorted(required - paths.keys())}")
    return paths


def _validate_equal_optimizer_exposure(runs: dict[str, dict[str, Any]]) -> None:
    for field in (
        "optimizer_steps",
        "train_native_units",
        "validation_native_units",
        "selection_native_unit_sha256",
    ):
        values = {str(run[field]) for run in runs.values()}
        if len(values) != 1:
            raise ValueError(f"temporal sampling run {field} differs={sorted(values)}")


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "experiment_contract",
        "implementation",
        "full_training_config",
        "short_matrix_gate",
        "runs",
        "analysis_contract",
        "output",
    }
    if set(config) != required:
        raise ValueError("temporal sampling decision config keys differ")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("temporal sampling decision config schema differs")
    if config["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("temporal sampling decision lineage differs")
    if set(config["runs"]) != set(VIEW_IDS):
        raise ValueError("temporal sampling decision run set differs")
    contract = config["experiment_contract"]
    if contract.get("primary_pair") != "s6_uniform_span16_vs_c6_contiguous_centered":
        raise ValueError("temporal sampling primary pair differs")
    if contract.get("secondary_pair") != "c8_contiguous_centered_vs_c6_contiguous_centered":
        raise ValueError("temporal sampling secondary pair differs")
    if contract.get("outer_predictions_used_for_model_selection") is not False:
        raise ValueError("outer predictions cannot select temporal sampling")
    analysis = config["analysis_contract"]
    if int(analysis.get("bootstrap_iterations", 0)) < 1000:
        raise ValueError("temporal sampling bootstrap contract is too small")
    expected_output = {
        "summary_json",
        "per_class_csv",
        "group_metrics_csv",
        "confusion_csv",
    }
    if set(config["output"]) != expected_output:
        raise ValueError("temporal sampling decision output keys differ")
    for name in ("implementation", "full_training_config", "short_matrix_gate"):
        _validate_file_spec(config[name], name)
    for view_id, run in config["runs"].items():
        if set(run) != {"run_result", "artifact_manifest"}:
            raise ValueError(f"{view_id} decision run keys differ")
        _validate_file_spec(run["run_result"], f"{view_id} run result")
        _validate_file_spec(run["artifact_manifest"], f"{view_id} artifacts")


def _validate_full_config(config: dict[str, Any]) -> None:
    if config.get("training_scope") != FULL_SCOPE:
        raise ValueError("temporal sampling full config scope differs")
    if config.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("temporal sampling full config lineage differs")
    if set(config.get("views", {})) != set(VIEW_IDS):
        raise ValueError("temporal sampling full config view set differs")


def _validate_short_gate(gate: dict[str, Any], full_config: dict[str, Any]) -> None:
    if gate.get("status") != "PASS_LEGACY_TEMPORAL_SAMPLING_SHORT_MATRIX":
        raise ValueError("temporal sampling short gate status differs")
    if gate.get("valid") is not True or gate.get("errors") != []:
        raise ValueError("temporal sampling short gate is invalid")
    if gate.get("full_confirmation_authorized") is not True:
        raise ValueError("temporal sampling short gate did not authorize full")
    expected = full_config["experiment_contract"]["short_config_sha256"]
    if gate.get("config_sha256") != expected:
        raise ValueError("temporal sampling short config lineage differs")


def _validate_file_spec(spec: dict[str, Any], name: str) -> None:
    if set(spec) != {"path", "sha256"} or not is_sha256(spec.get("sha256")):
        raise ValueError(f"invalid {name} file specification")


def _verify_file_spec(root: Path, spec: dict[str, Any], name: str) -> Path:
    _validate_file_spec(spec, name)
    path = _resolve_inside(root, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"{name} hash differs")
    return path


def _resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root: {resolved}") from error
    return resolved


def _probability_column(label: str) -> str:
    return f"prob_{label.replace('-', '_')}"


def _mapping_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame[columns].astype(str).sort_values(columns, kind="mergesort")
    payload = "\n".join("\x1f".join(row) for row in rows.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reliability_tier(support: int) -> str:
    if support == 0:
        return "no_support"
    if support >= 20:
        return "moderate_descriptive_support"
    if support >= 5:
        return "low_support"
    if support >= 2:
        return "very_low_support"
    return "single_unit_not_estimable"


def _bound_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _bound_table(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {**_bound_file(path), "rows": int(len(frame))}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x", lineterminator="\n")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "CONFIG_SCHEMA",
    "PAIR_SPECS",
    "VIEW_IDS",
    "evaluate_temporal_sampling_predictions",
    "write_temporal_sampling_decision",
]
