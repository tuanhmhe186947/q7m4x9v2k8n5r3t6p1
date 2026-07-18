"""Paired transfer decisions for legacy Stage A temporal-base screening."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation import (
    legacy_development_temporal_sampling_decision as metric_engine,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_temporal_base_selection import (
    FULL_SCOPE,
    LINEAGE_SCOPE,
    MODE_SPECS,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_decision.v1"
)
MODE_IDS = tuple(MODE_SPECS)
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33
PAIR_SPECS = {
    "multiple_frames": {
        "candidate": "M128",
        "baseline": "SF128",
        "changed_factor": "multiple_frames_at_exact_parameter_count",
        "target_groups": ("locomotion_context", "rare"),
    },
    "content_weighting": {
        "candidate": "A128",
        "baseline": "M128",
        "changed_factor": "content_attention_pooling",
        "target_groups": ("rare",),
    },
    "ordered_tcn": {
        "candidate": "TCN128",
        "baseline": "MW317",
        "changed_factor": "ordered_local_dynamics_at_matched_capacity",
        "target_groups": ("locomotion_context", "rare"),
    },
    "timed_transformer": {
        "candidate": "TR128",
        "baseline": "MW381",
        "changed_factor": "order_and_elapsed_time_at_matched_capacity",
        "target_groups": ("locomotion_context", "rare"),
    },
}
OPERATIONAL_PAIR_SPECS = {
    "operational_attention_vs_single": {
        "candidate": "A128",
        "baseline": "SF128",
        "changed_factor": "compound_attention_sequence_candidate",
        "target_groups": ("locomotion_context", "rare"),
    },
    "operational_tcn_vs_single": {
        "candidate": "TCN128",
        "baseline": "SF128",
        "changed_factor": "compound_tcn_sequence_candidate",
        "target_groups": ("locomotion_context", "rare"),
    },
    "operational_transformer_vs_single": {
        "candidate": "TR128",
        "baseline": "SF128",
        "changed_factor": "compound_transformer_sequence_candidate",
        "target_groups": ("locomotion_context", "rare"),
    },
}
METADATA_COLUMNS = (
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "target_index",
)


def evaluate_temporal_base_predictions(
    predictions: dict[str, pd.DataFrame],
    *,
    run_summaries: dict[str, dict[str, Any]] | None,
    iterations: int,
    seed: int,
    material_negative_ci_limit: float,
    maximum_group_macro_f1_drop: float,
    enforce_project_counts: bool = True,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build paired metrics and transfer actions on one native universe."""

    if iterations < 1_000:
        raise ValueError("temporal-base bootstrap requires at least 1000 draws")
    if material_negative_ci_limit < 0.0:
        raise ValueError("material negative CI limit must be nonnegative")
    if maximum_group_macro_f1_drop < 0.0:
        raise ValueError("maximum group macro-F1 drop must be nonnegative")
    ordered = _validate_prediction_universe(
        predictions,
        enforce_project_counts=enforce_project_counts,
    )
    summaries = _validate_run_summaries(run_summaries, ordered)
    metrics = {
        mode_id: metric_engine._single_view_metrics(frame)
        for mode_id, frame in ordered.items()
    }
    comparisons: dict[str, dict[str, Any]] = {}
    per_class_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    all_pairs = {**PAIR_SPECS, **OPERATIONAL_PAIR_SPECS}
    for pair_index, (pair_id, spec) in enumerate(all_pairs.items()):
        candidate = str(spec["candidate"])
        baseline = str(spec["baseline"])
        comparison, per_class, groups = metric_engine._compare_pair(
            ordered[candidate],
            ordered[baseline],
            candidate_id=candidate,
            baseline_id=baseline,
            pair_id=pair_id,
            changed_factor=str(spec["changed_factor"]),
            iterations=iterations,
            seed=seed + pair_index,
        )
        comparison["transfer_decision"] = _pair_transfer_decision(
            comparison,
            target_groups=tuple(spec["target_groups"]),
            material_negative_ci_limit=material_negative_ci_limit,
            maximum_group_macro_f1_drop=maximum_group_macro_f1_drop,
            operational_check=pair_id in OPERATIONAL_PAIR_SPECS,
        )
        comparison["evidence_role"] = (
            "controlled_mechanism_test"
            if pair_id in PAIR_SPECS
            else "compound_operational_candidate_check"
        )
        comparisons[pair_id] = comparison
        per_class_parts.append(per_class)
        group_parts.append(groups)
    per_class_frame = pd.concat(per_class_parts, ignore_index=True)
    group_frame = pd.concat(group_parts, ignore_index=True)
    confusion = metric_engine._confusion_table(ordered).rename(
        columns={"view_id": "mode_id"}
    )
    mode_summary = _mode_summary(metrics, summaries)
    packet = _full_data_candidate_packet(metrics, comparisons, mode_summary)
    universe = ordered[MODE_IDS[0]][list(METADATA_COLUMNS)]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_LEGACY_TEMPORAL_BASE_PAIRED_DECISION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "legacy_sets_final_full_data_base": False,
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
        "mode_metrics": metrics,
        "paired_comparisons": comparisons,
        "full_data_candidate_packet": packet,
        "interpretation_boundary": [
            "legacy_16f_unreviewed_development_screening_only",
            "one_development_validation_split_not_full_oof",
            "regular_legacy_timing_cannot_establish_irregular_time_utility",
            "legacy_class_support_cannot_replace_reviewed_mixed_source_support",
            "all_carried_modes_require_frozen_mixed_reviewed_confirmation",
            "pooled_full_data_gain_must_not_be_driven_only_by_legacy",
        ],
        "errors": [],
        "valid": True,
    }
    return result, per_class_frame, group_frame, confusion, mode_summary


def _validate_prediction_universe(
    predictions: dict[str, pd.DataFrame],
    *,
    enforce_project_counts: bool,
) -> dict[str, pd.DataFrame]:
    if set(predictions) != set(MODE_IDS):
        raise ValueError("temporal-base prediction mode set differs")
    required = set(METADATA_COLUMNS) | {
        "predicted_index",
        "predicted_label",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "temporal_base_mode_id",
        *(_probability_column(label) for label in VALID_BEHAVIORS),
    }
    ordered: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for mode_id in MODE_IDS:
        frame = predictions[mode_id].copy(deep=True)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{mode_id} prediction columns missing={missing}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{mode_id} contains duplicate native units")
        frame = frame.sort_values(
            "temporal_unit_key",
            kind="mergesort",
        ).reset_index(drop=True)
        if enforce_project_counts and len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{mode_id} native units={len(frame)}")
        clusters = frame["video_key"].astype(str).nunique()
        if enforce_project_counts and clusters != EXPECTED_VIDEO_CLUSTERS:
            raise ValueError(f"{mode_id} video clusters={clusters}")
        _validate_prediction_rows(frame, mode_id)
        metadata = frame[list(METADATA_COLUMNS)].astype(str)
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"{mode_id} native metadata universe differs")
        ordered[mode_id] = frame
    return ordered


def _validate_prediction_rows(frame: pd.DataFrame, mode_id: str) -> None:
    labels = list(VALID_BEHAVIORS)
    if not frame["training_scope"].astype(str).eq(FULL_SCOPE).all():
        raise ValueError(f"{mode_id} training scope differs")
    if not frame["lineage_scope"].astype(str).eq(LINEAGE_SCOPE).all():
        raise ValueError(f"{mode_id} lineage scope differs")
    reviewed = frame["human_review_complete"].astype(str).str.lower()
    if not reviewed.isin({"false", "0"}).all():
        raise ValueError(f"{mode_id} incorrectly claims completed review")
    if not frame["temporal_base_mode_id"].astype(str).eq(mode_id).all():
        raise ValueError(f"{mode_id} prediction mode metadata differs")
    expected_target = frame["behavior_label"].astype(str).map(
        {label: index for index, label in enumerate(labels)}
    )
    if expected_target.isna().any() or not np.array_equal(
        frame["target_index"].to_numpy(dtype=np.int64),
        expected_target.to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"{mode_id} target index differs from label order")
    probabilities = frame[
        [_probability_column(label) for label in labels]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{mode_id} contains nonfinite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{mode_id} probability rows do not sum to one")
    predicted = probabilities.argmax(axis=1)
    if not np.array_equal(
        frame["predicted_index"].to_numpy(dtype=np.int64),
        predicted,
    ):
        raise ValueError(f"{mode_id} predicted index differs from probabilities")
    expected_labels = np.asarray(labels, dtype=object)[predicted]
    if not np.array_equal(
        frame["predicted_label"].astype(str).to_numpy(),
        expected_labels,
    ):
        raise ValueError(f"{mode_id} predicted label differs from probabilities")


def _pair_transfer_decision(
    comparison: dict[str, Any],
    *,
    target_groups: tuple[str, ...],
    material_negative_ci_limit: float,
    maximum_group_macro_f1_drop: float,
    operational_check: bool,
) -> dict[str, Any]:
    delta = float(
        comparison["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )
    bootstrap = comparison["video_cluster_bootstrap"]
    ci_low = float(bootstrap["macro_f1_delta_ci_low"])
    candidate_groups = comparison["candidate_metrics"]["group_macro_f1"]
    baseline_groups = comparison["baseline_metrics"]["group_macro_f1"]
    group_deltas = {
        group: float(candidate_groups[group] - baseline_groups[group])
        for group in candidate_groups
    }
    target_regressions = {
        group: group_deltas[group]
        for group in target_groups
        if group_deltas[group] < -maximum_group_macro_f1_drop
    }
    reasons: list[str] = []
    if delta <= 0.0:
        action = "DROP"
        reasons.append("paired_macro_f1_delta_not_positive")
    elif target_regressions:
        action = "RETEST" if operational_check else "DROP"
        reasons.append("target_group_regression_exceeds_limit")
    elif ci_low < -material_negative_ci_limit:
        action = "RETEST"
        reasons.append("paired_interval_remains_materially_negative")
    else:
        action = "CARRY"
        reasons.append("controlled_pair_passes_legacy_screening_gate")
    timing_claim_action = (
        "RETEST_ON_MIXED_REVIEWED_OBSERVED_TIME"
        if comparison["pair_id"] == "timed_transformer"
        else "NOT_APPLICABLE"
    )
    return {
        "screening_action": action,
        "reasons": reasons,
        "macro_f1_delta": delta,
        "macro_f1_delta_ci_low": ci_low,
        "macro_f1_delta_ci_high": float(
            bootstrap["macro_f1_delta_ci_high"]
        ),
        "material_negative_ci_limit": material_negative_ci_limit,
        "target_groups": list(target_groups),
        "group_macro_f1_deltas": group_deltas,
        "target_group_regressions": target_regressions,
        "maximum_group_macro_f1_drop": maximum_group_macro_f1_drop,
        "timing_claim_action": timing_claim_action,
        "operational_check": operational_check,
        "legacy_is_final_base_evidence": False,
    }


def _mode_summary(
    metrics: dict[str, dict[str, Any]],
    run_summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for mode_id in MODE_IDS:
        metric = metrics[mode_id]
        run = run_summaries[mode_id]
        rows.append(
            {
                "mode_id": mode_id,
                "temporal_encoder_name": MODE_SPECS[mode_id][
                    "temporal_encoder_name"
                ],
                "hidden_dim": int(MODE_SPECS[mode_id]["hidden_dim"]),
                "native_frame_offsets_json": json.dumps(
                    MODE_SPECS[mode_id]["native_frame_offsets"],
                    separators=(",", ":"),
                ),
                "timing_contract": MODE_SPECS[mode_id]["timing_contract"],
                "parameter_count": int(run["parameter_count"]),
                "optimizer_steps": int(run["optimizer_steps"]),
                "runtime_seconds": float(run["runtime_seconds"]),
                "peak_memory_bytes": int(run["peak_memory_bytes"]),
                "macro_f1_global_10_class": float(
                    metric["macro_f1_global_10_class"]
                ),
                "weighted_f1": float(metric["weighted_f1"]),
                "accuracy": float(metric["accuracy"]),
                "macro_recall_global_10_class": float(
                    metric["macro_recall_global_10_class"]
                ),
                "nll": float(metric["nll"]),
            }
        )
    return pd.DataFrame(rows)


def _full_data_candidate_packet(
    metrics: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    mode_summary: pd.DataFrame,
) -> dict[str, Any]:
    multiple_action = comparisons["multiple_frames"]["transfer_decision"][
        "screening_action"
    ]
    simple_mode = "M128" if multiple_action == "CARRY" else "SF128"
    carried: list[dict[str, str]] = [
        {
            "slot": "F0",
            "mode_id": simple_mode,
            "role": "simplest_valid_visual_temporal_control",
        }
    ]
    candidate_pairs = {
        "content_weighting": "operational_attention_vs_single",
        "ordered_tcn": "operational_tcn_vs_single",
        "timed_transformer": "operational_transformer_vs_single",
    }
    eligible = []
    for pair_id, operational_pair_id in candidate_pairs.items():
        comparison = comparisons[pair_id]
        operational = comparisons[operational_pair_id]
        if (
            comparison["transfer_decision"]["screening_action"] == "CARRY"
            and operational["transfer_decision"]["screening_action"] == "CARRY"
        ):
            mode_id = str(comparison["candidate_view"])
            eligible.append(mode_id)
    if eligible:
        best = max(
            eligible,
            key=lambda mode: (
                metrics[mode]["macro_f1_global_10_class"],
                -metrics[mode]["nll"],
            ),
        )
        if best != simple_mode:
            carried.append(
                {
                    "slot": "F1",
                    "mode_id": best,
                    "role": "best_legacy_screened_temporal_candidate",
                }
            )
    retest = []
    for pair_id, operational_pair_id in candidate_pairs.items():
        controlled = comparisons[pair_id]
        operational = comparisons[operational_pair_id]
        controlled_action = controlled["transfer_decision"]["screening_action"]
        operational_action = operational["transfer_decision"][
            "screening_action"
        ]
        if (
            controlled_action in {"CARRY", "RETEST"}
            and operational_action in {"CARRY", "RETEST"}
            and not (
                controlled_action == "CARRY"
                and operational_action == "CARRY"
            )
        ):
            retest.append(
                {
                    "pair_id": pair_id,
                    "operational_pair_id": operational_pair_id,
                    "candidate_mode": str(controlled["candidate_view"]),
                    "controlled_action": controlled_action,
                    "operational_action": operational_action,
                    "controlled_reasons": controlled["transfer_decision"][
                        "reasons"
                    ],
                    "operational_reasons": operational[
                        "transfer_decision"
                    ]["reasons"],
                }
            )
    dropped = [
        {
            "pair_id": pair_id,
            "candidate_mode": str(comparison["candidate_view"]),
            "reason": comparison["transfer_decision"]["reasons"],
        }
        for pair_id, comparison in comparisons.items()
        if pair_id in PAIR_SPECS
        if comparison["transfer_decision"]["screening_action"] == "DROP"
    ]
    selected_ids = [item["mode_id"] for item in carried]
    selected_rows = mode_summary.loc[
        mode_summary["mode_id"].isin(selected_ids)
    ].to_dict(orient="records")
    return {
        "packet_status": "LEGACY_SCREENING_PACKET_NOT_FINAL_BASE_SELECTION",
        "candidate_limit": 3,
        "carried_finalists": carried,
        "carried_mode_summaries": selected_rows,
        "retest_on_mixed_reviewed": retest,
        "dropped_from_legacy_expansion": dropped,
        "reserved_slot": {
            "slot": "F2",
            "role": "best_individually_validated_spatiotemporal_candidate",
            "status": "NOT_EVALUATED_IN_STAGE_A",
        },
        "required_full_data_confirmation": {
            "dataset_profile": "mixed-reviewed",
            "same_grouped_fold_manifest_required": True,
            "same_paired_native_units_required": True,
            "required_strata": [
                "legacy_recovered",
                "cvat_tracking_xml",
                "context_available",
                "context_missing",
            ],
            "promotion_rule": (
                "pooled_gain_plus_no_material_source_or_missingness_regression"
            ),
            "irregular_observed_time_confirmation_required": True,
            "full_oof_launch_gate_still_required": True,
        },
        "unresolved_transfer_questions": [
            {
                "question": "irregular_observed_time_utility",
                "legacy_status": "NOT_ESTIMABLE_FROM_REGULAR_LEGACY_TIMING",
                "required_action": "RETEST_ON_MIXED_REVIEWED_OBSERVED_TIME",
            },
            {
                "question": "six_frame_cvat_transfer",
                "legacy_status": "CVAT_SOURCE_ABSENT",
                "required_action": "PAIRED_PER_SOURCE_CONFIRMATION",
            },
            {
                "question": "missing_modality_robustness",
                "legacy_status": "NOT_EXERCISED_BY_ACTOR_ONLY_STAGE_A",
                "required_action": "STRATIFIED_FULL_DATA_CONFIRMATION",
            },
        ],
        "legacy_can_set_final_base": False,
    }


def _validate_run_summaries(
    run_summaries: dict[str, dict[str, Any]] | None,
    predictions: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    if run_summaries is None:
        return {
            mode_id: {
                "parameter_count": int(
                    MODE_SPECS[mode_id]["expected_parameter_count"]
                ),
                "optimizer_steps": 0,
                "runtime_seconds": 0.0,
                "peak_memory_bytes": 0,
            }
            for mode_id in predictions
        }
    if set(run_summaries) != set(MODE_IDS):
        raise ValueError("temporal-base run summary mode set differs")
    for mode_id, summary in run_summaries.items():
        if int(summary["parameter_count"]) != int(
            MODE_SPECS[mode_id]["expected_parameter_count"]
        ):
            raise ValueError(f"{mode_id} run parameter count differs")
        if int(summary["optimizer_steps"]) <= 0:
            raise ValueError(f"{mode_id} run optimizer steps invalid")
    optimizer_steps = {
        int(summary["optimizer_steps"]) for summary in run_summaries.values()
    }
    if len(optimizer_steps) != 1:
        raise ValueError("temporal-base optimizer exposure differs")
    return run_summaries


def write_temporal_base_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Load hash-bound full runs and write one exclusive transfer packet."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    _verify_file_spec(root, config["implementation"], "implementation")
    metric_path = _verify_file_spec(
        root,
        config["metric_engine"],
        "metric engine",
    )
    full_config_path = _verify_file_spec(
        root,
        config["full_training_config"],
        "full training config",
    )
    full_config = _read_json(full_config_path)
    if full_config.get("training_scope") != FULL_SCOPE:
        raise ValueError("temporal-base full config scope differs")
    if set(full_config.get("modes", {})) != set(MODE_IDS):
        raise ValueError("temporal-base full config mode set differs")
    short_gate_path = _verify_file_spec(
        root,
        config["short_matrix_gate"],
        "short matrix gate",
    )
    short_gate = _read_json(short_gate_path)
    if short_gate.get("status") != "PASS_LEGACY_TEMPORAL_BASE_SHORT_MATRIX":
        raise ValueError("temporal-base short gate status differs")
    if short_gate.get("valid") is not True or short_gate.get("errors") != []:
        raise ValueError("temporal-base short gate is invalid")
    predictions: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for mode_id in MODE_IDS:
        frame, summary = _load_run_packet(
            root,
            mode_id=mode_id,
            spec=config["runs"][mode_id],
            full_config_sha256=file_sha256(full_config_path),
        )
        predictions[mode_id] = frame
        summaries[mode_id] = summary
    contract = config["analysis_contract"]
    result, per_class, groups, confusion, mode_summary = (
        evaluate_temporal_base_predictions(
            predictions,
            run_summaries=summaries,
            iterations=int(contract["bootstrap_iterations"]),
            seed=int(contract["bootstrap_seed"]),
            material_negative_ci_limit=float(
                contract["material_negative_ci_limit"]
            ),
            maximum_group_macro_f1_drop=float(
                contract["maximum_group_macro_f1_drop"]
            ),
        )
    )
    outputs = {
        name: _resolve_inside(root, value)
        for name, value in config["output"].items()
    }
    _write_csv_exclusive(outputs["per_class_csv"], per_class)
    _write_csv_exclusive(outputs["group_metrics_csv"], groups)
    _write_csv_exclusive(outputs["confusion_csv"], confusion)
    _write_csv_exclusive(outputs["mode_summary_csv"], mode_summary)
    result.update(
        {
            "config_path": str(resolved_config),
            "config_sha256": file_sha256(resolved_config),
            "full_training_config": _bound_file(full_config_path),
            "short_matrix_gate": _bound_file(short_gate_path),
            "metric_engine": _bound_file(metric_path),
            "runs": summaries,
            "analysis_contract": dict(contract),
            "artifacts": {
                "per_class_csv": _bound_table(
                    outputs["per_class_csv"],
                    per_class,
                ),
                "group_metrics_csv": _bound_table(
                    outputs["group_metrics_csv"],
                    groups,
                ),
                "confusion_csv": _bound_table(
                    outputs["confusion_csv"],
                    confusion,
                ),
                "mode_summary_csv": _bound_table(
                    outputs["mode_summary_csv"],
                    mode_summary,
                ),
            },
            "optimizer_steps_executed_by_evaluator": 0,
            "source_media_reads": 0,
            "outer_holdout_rows_loaded": 0,
        }
    )
    _write_json_exclusive(outputs["summary_json"], result)
    return outputs["summary_json"], result


def _load_run_packet(
    root: Path,
    *,
    mode_id: str,
    spec: dict[str, Any],
    full_config_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result_path = _verify_file_spec(root, spec["run_result"], f"{mode_id} result")
    artifact_path = _verify_file_spec(
        root,
        spec["artifact_manifest"],
        f"{mode_id} artifacts",
    )
    result = _read_json(result_path)
    if result.get("status") != "completed" or result.get("valid") is not True:
        raise ValueError(f"{mode_id} run is not valid and completed")
    if result.get("errors") != [] or result.get("mode_id") != mode_id:
        raise ValueError(f"{mode_id} run identity differs")
    if result.get("config_sha256") != full_config_sha256:
        raise ValueError(f"{mode_id} full config hash differs")
    artifact = _read_json(artifact_path)
    paths = _verify_artifact_manifest(root, artifact, mode_id=mode_id)
    if result.get("artifact_manifest_sha256") != file_sha256(artifact_path):
        raise ValueError(f"{mode_id} artifact manifest link differs")
    frame = pd.read_csv(paths["validation_native_predictions.csv"])
    epoch_metrics = pd.read_csv(paths["epoch_metrics.csv"])
    run_manifest = _read_json(paths["run_manifest.json"])
    if len(frame) != EXPECTED_NATIVE_UNITS:
        raise ValueError(f"{mode_id} prediction rows differ")
    if int(result["optimizer_steps"]) != int(
        epoch_metrics["optimizer_steps_cumulative"].iloc[-1]
    ):
        raise ValueError(f"{mode_id} optimizer steps differ")
    runtime = run_manifest.get("runtime_profile", {})
    summary = {
        "mode_id": mode_id,
        "run_result": _bound_file(result_path),
        "artifact_manifest": _bound_file(artifact_path),
        "checkpoint_sha256": _read_json(
            paths["checkpoint_manifest.json"]
        )["checkpoint_sha256"],
        "prediction_sha256": file_sha256(
            paths["validation_native_predictions.csv"]
        ),
        "parameter_count": int(result["parameter_count"]),
        "optimizer_steps": int(result["optimizer_steps"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "peak_memory_bytes": int(runtime.get("peak_memory_bytes", 0)),
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
    mode_id: str,
) -> dict[str, Path]:
    if manifest.get("valid") is not True or manifest.get("errors") != []:
        raise ValueError(f"{mode_id} artifact manifest is invalid")
    if manifest.get("mode_id") != mode_id:
        raise ValueError(f"{mode_id} artifact manifest mode differs")
    paths: dict[str, Path] = {}
    for artifact in manifest.get("artifacts", []):
        name = str(artifact["name"])
        path = _resolve_inside(root, str(artifact["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"{mode_id} artifact missing: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"{mode_id} artifact hash differs: {name}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"{mode_id} artifact size differs: {name}")
        if name in paths:
            raise ValueError(f"{mode_id} duplicate artifact name: {name}")
        paths[name] = path
    run_manifest_path = _resolve_inside(
        root,
        str(Path(manifest["artifacts"][0]["path"]).parent / "run_manifest.json"),
    )
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"{mode_id} run manifest missing")
    paths["run_manifest.json"] = run_manifest_path
    required = {
        "validation_native_predictions.csv",
        "epoch_metrics.csv",
        "checkpoint_manifest.json",
        "run_manifest.json",
    }
    if not required.issubset(paths):
        raise ValueError(
            f"{mode_id} required artifacts missing={sorted(required - paths.keys())}"
        )
    return paths


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "experiment_contract",
        "implementation",
        "metric_engine",
        "full_training_config",
        "short_matrix_gate",
        "runs",
        "analysis_contract",
        "output",
    }
    if set(config) != required:
        raise ValueError("temporal-base decision config keys differ")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("temporal-base decision config schema differs")
    if config["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("temporal-base decision lineage differs")
    if set(config["runs"]) != set(MODE_IDS):
        raise ValueError("temporal-base decision run set differs")
    contract = config["experiment_contract"]
    if contract.get("legacy_sets_final_full_data_base") is not False:
        raise ValueError("legacy decision cannot set final full-data base")
    if contract.get("outer_predictions_used_for_model_selection") is not False:
        raise ValueError("outer predictions cannot select temporal base")
    analysis = config["analysis_contract"]
    if int(analysis.get("bootstrap_iterations", 0)) < 1_000:
        raise ValueError("temporal-base bootstrap contract is too small")
    if float(analysis.get("material_negative_ci_limit", -1.0)) < 0.0:
        raise ValueError("temporal-base material CI limit invalid")
    if float(analysis.get("maximum_group_macro_f1_drop", -1.0)) < 0.0:
        raise ValueError("temporal-base group guardrail invalid")
    expected_output = {
        "summary_json",
        "per_class_csv",
        "group_metrics_csv",
        "confusion_csv",
        "mode_summary_csv",
    }
    if set(config["output"]) != expected_output:
        raise ValueError("temporal-base decision output keys differ")
    for name in (
        "implementation",
        "metric_engine",
        "full_training_config",
        "short_matrix_gate",
    ):
        _validate_file_spec(config[name], name)
    for mode_id, run in config["runs"].items():
        if set(run) != {"run_result", "artifact_manifest"}:
            raise ValueError(f"{mode_id} decision run keys differ")
        _validate_file_spec(run["run_result"], f"{mode_id} run result")
        _validate_file_spec(run["artifact_manifest"], f"{mode_id} artifacts")


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
    "MODE_IDS",
    "OPERATIONAL_PAIR_SPECS",
    "PAIR_SPECS",
    "evaluate_temporal_base_predictions",
    "write_temporal_base_decision",
]
