"""Lock the bounded legacy_16f development candidate and evidence registries."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.calibration import (
    probability_calibration_metrics,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    load_temporal_ladder_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder_runtime import (
    audit_temporal_ladder_run,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    CONFUSION_GROUPS,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l8.candidate_lock_config.v1"
)
LOCK_SCHEMA = "classification_v2.legacy_development_l8.candidate_lock.v1"
FINALIST_SCHEMA = "classification_v2.legacy_development_l8.finalist_lock.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"

CLAIM_BOUNDARY = {
    "lineage_scope": LINEAGE_SCOPE,
    "human_review_complete": False,
    "reviewed_or_final_claim_allowed": False,
    "q2_claim_allowed": False,
    "canonical_full_oof_authorized": False,
    "outer_holdout_predictions_authorized": False,
}

LOCKED_MODEL = {
    "architecture": "cached_frame_feature_temporal_classifier_v1",
    "feature_control_id": "V1",
    "backbone_name": "resnet18",
    "input_resolution": 224,
    "temporal_encoder_name": "masked_mean",
    "hidden_dim": 128,
    "dropout": 0.1,
    "parameter_count": 68_234,
    "native_probability_aggregation": "mean_window_probability_v1",
}

LOCKED_OPTIMIZATION = {
    "seed": 20260714,
    "epochs": 3,
    "batch_size": 32,
    "evaluation_batch_size": 64,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "loss": "event_mass_balanced_cross_entropy",
    "sampler": "deterministic_seeded_window_shuffle_after_native_selection",
    "checkpoint_selection": "native_global_10_class_macro_f1_then_nll",
    "precision": "float32",
    "autocast_enabled": False,
    "deterministic_algorithms": True,
}

OUTPUT_FILES = {
    "candidate_lock": "candidate_lock.json",
    "finalist_lock": "finalist_lock.json",
    "model_card": "model_card.md",
    "experiment_matrix": "experiment_matrix.csv",
    "ablation_registry": "ablation_registry.csv",
    "promotion_decisions": "promotion_decisions.json",
    "rejected_experiments": "rejected_experiments.json",
}


def lock_legacy_l8_candidate(
    config_path: Path,
    *,
    project_root: Path | None = None,
    enforce_git_guard: bool = True,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Audit the full retained run and emit the immutable L8 lock packet."""

    resolved_config = config_path.resolve()
    root = (project_root or Path.cwd()).resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _validate_bound_file(
        root,
        config["implementation_source"],
        "implementation_source",
    )
    full_config_path = _validate_bound_file(
        root,
        config["full_training_config"],
        "full_training_config",
    )
    result_path = _validate_bound_file(
        root,
        config["full_candidate_result"],
        "full_candidate_result",
    )
    l7_decision_path = _validate_bound_file(
        root,
        config["l7_decision"],
        "l7_decision",
    )
    git_guard = (
        _git_guard(root, config["execution_guard"])
        if enforce_git_guard
        else {
            "status": "SKIPPED_UNIT_TEST_ONLY",
            "errors": [],
            "valid": True,
        }
    )
    full_config = load_temporal_ladder_config(full_config_path)
    run_audit = audit_temporal_ladder_run(full_config, result_path=result_path)
    if not run_audit["valid"]:
        raise ValueError(f"L8 full candidate run audit failed={run_audit['errors']}")
    result = _read_json(result_path)
    _validate_full_candidate(full_config.payload, result)
    l7_decision = _read_json(l7_decision_path)
    _validate_l7_decision(l7_decision)
    decision_records = _load_decision_artifacts(root, config["decision_artifacts"])
    predictions_path = result_path.parent / "validation_native_predictions.csv"
    predictions = pd.read_csv(predictions_path, low_memory=False)
    evidence = _candidate_evidence(
        predictions,
        result,
        bootstrap_iterations=int(config["evidence_contract"]["bootstrap_iterations"]),
        bootstrap_seed=int(config["evidence_contract"]["bootstrap_seed"]),
    )
    output_root = _resolve_inside(root, config["output"]["root_relative_path"])
    artifacts = _build_outputs(
        config=config,
        result=result,
        result_path=result_path,
        run_audit=run_audit,
        l7_decision=l7_decision,
        l7_decision_path=l7_decision_path,
        decision_records=decision_records,
        evidence=evidence,
    )
    output_hashes: dict[str, Any] = {}
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_dataframe(
            output_root / OUTPUT_FILES["experiment_matrix"],
            artifacts["experiment_matrix"],
        )
        _write_dataframe(
            output_root / OUTPUT_FILES["ablation_registry"],
            artifacts["ablation_registry"],
        )
        _write_json(
            output_root / OUTPUT_FILES["promotion_decisions"],
            artifacts["promotion_decisions"],
        )
        _write_json(
            output_root / OUTPUT_FILES["rejected_experiments"],
            artifacts["rejected_experiments"],
        )
        _write_text(output_root / OUTPUT_FILES["model_card"], artifacts["model_card"])
        _write_json(output_root / OUTPUT_FILES["finalist_lock"], artifacts["finalist_lock"])
        for name in (
            "experiment_matrix",
            "ablation_registry",
            "promotion_decisions",
            "rejected_experiments",
            "model_card",
            "finalist_lock",
        ):
            path = output_root / OUTPUT_FILES[name]
            output_hashes[name] = _artifact_spec(path)
    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L8_CANDIDATE_LOCK",
        **CLAIM_BOUNDARY,
        "lock_id": config["lock_id"],
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(implementation),
        "implementation_source_sha256": file_sha256(implementation),
        "git_guard": git_guard,
        "selected_candidate": copy.deepcopy(artifacts["finalist_lock"]),
        "candidate_evidence": evidence,
        "l7_decision": {
            "path": str(l7_decision_path),
            "sha256": file_sha256(l7_decision_path),
            "decision_payload_sha256": l7_decision["decision_payload_sha256"],
        },
        "full_candidate_run_audit": {
            "result_path": str(result_path),
            "result_sha256": run_audit["result_sha256"],
            "run_manifest_sha256": run_audit["run_manifest_sha256"],
            "artifact_manifest_sha256": run_audit["artifact_manifest_sha256"],
            "verified_artifacts": run_audit["verified_artifacts"],
            "valid": run_audit["valid"],
        },
        "registry_artifacts": output_hashes,
        "interpretation_boundary": copy.deepcopy(
            config["interpretation_boundary"]
        ),
        "unresolved_risks": [
            "legacy_16f labels are explicitly unreviewed development evidence",
            "validation contains only two fight and one playwithtoy native units",
            "outer holdout was not read or predicted during candidate selection",
            "candidate evidence does not transfer to merged-reviewed data",
            "architecture family is not finalized for merged-reviewed training",
        ],
        "rollback": {
            "action": "remove L8 lock packet and resume from L7 decision",
            "parent_decision_sha256": file_sha256(l7_decision_path),
            "full_candidate_checkpoint_sha256": artifacts["finalist_lock"][
                "checkpoint_sha256"
            ],
        },
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    lock["lock_payload_sha256"] = payload_sha256(lock)
    if write_outputs:
        lock_path = output_root / OUTPUT_FILES["candidate_lock"]
        _write_json(lock_path, lock)
    return lock


def _candidate_evidence(
    frame: pd.DataFrame,
    result: dict[str, Any],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    probability_columns = [
        "prob_" + label.replace("-", "_") for label in VALID_BEHAVIORS
    ]
    required = {
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "behavior_label",
        "target_index",
        "predicted_index",
        "predicted_label",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        *probability_columns,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"L8 candidate predictions missing={missing}")
    if len(frame) != 245 or frame["video_key"].astype(str).nunique() != 33:
        raise ValueError("L8 candidate validation universe drift")
    if frame["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("L8 candidate duplicate native units")
    _validate_prediction_claims(frame)
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("L8 candidate has nonfinite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("L8 candidate probability mass drift")
    label_to_index = {
        label: index for index, label in enumerate(VALID_BEHAVIORS)
    }
    true_labels = frame["behavior_label"].astype(str)
    predicted_labels = frame["predicted_label"].astype(str)
    unknown_labels = sorted(
        (set(true_labels) | set(predicted_labels)) - set(VALID_BEHAVIORS)
    )
    if unknown_labels:
        raise ValueError(f"L8 candidate unknown labels={unknown_labels}")
    expected_targets = np.asarray(
        [label_to_index[label] for label in true_labels],
        dtype=np.int64,
    )
    expected_predictions = np.asarray(
        [label_to_index[label] for label in predicted_labels],
        dtype=np.int64,
    )
    observed_targets = pd.to_numeric(
        frame["target_index"],
        errors="raise",
    ).to_numpy()
    observed_predictions = pd.to_numeric(
        frame["predicted_index"],
        errors="raise",
    ).to_numpy()
    if not np.array_equal(observed_targets, expected_targets):
        raise ValueError("L8 candidate target label/index drift")
    if not np.array_equal(observed_predictions, expected_predictions):
        raise ValueError("L8 candidate predicted label/index drift")
    argmax = probabilities.argmax(axis=1)
    if not np.array_equal(argmax, expected_predictions):
        raise ValueError("L8 candidate prediction argmax drift")
    global_metrics = evaluate_predictions(
        frame,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )
    calibration = probability_calibration_metrics(
        probabilities,
        expected_targets,
        ece_bins=15,
    )
    groups = _group_report(frame, global_metrics["per_class"])
    recording = _recording_report(frame)
    uncertainty = _cluster_bootstrap(
        frame,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    _require_close(
        float(global_metrics["macro_f1"]),
        float(result["validation_metrics"]["macro_f1_global_10_class"]),
        "L8 candidate macro-F1",
    )
    _require_close(
        float(calibration["negative_log_likelihood"]),
        float(result["validation_metrics"]["nll"]),
        "L8 candidate NLL",
    )
    return {
        "native_units": int(len(frame)),
        "video_clusters": int(frame["video_key"].astype(str).nunique()),
        "global": {
            "macro_f1_global_10_class": float(global_metrics["macro_f1"]),
            "accuracy": float(global_metrics["accuracy"]),
            "nll": float(calibration["negative_log_likelihood"]),
            "multiclass_brier": float(calibration["multiclass_brier"]),
            "top_label_ece": float(calibration["top_label_ece"]),
        },
        "groups": groups,
        "per_class": global_metrics["per_class"],
        "recording": recording,
        "video_cluster_bootstrap": uncertainty,
        "runtime": {
            "runtime_seconds": float(result["runtime_seconds"]),
            "optimizer_steps": int(result["optimizer_steps"]),
            "best_epoch": int(result["best_epoch"]),
            "peak_allocated_bytes": int(
                result["execution"]["peak_allocated_bytes"]
            ),
            "peak_reserved_bytes": int(
                result["execution"]["peak_reserved_bytes"]
            ),
            "post_cleanup_allocated_bytes": int(
                result["execution"]["post_cleanup_allocated_bytes"]
            ),
            "post_cleanup_reserved_bytes": int(
                result["execution"]["post_cleanup_reserved_bytes"]
            ),
            "oom": bool(result["execution"]["oom"]),
            "oom_retry_count": int(result["execution"]["oom_retry_count"]),
        },
    }


def _group_report(
    frame: pd.DataFrame,
    per_class: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, labels in CONFUSION_GROUPS.items():
        selected = frame["behavior_label"].astype(str).isin(labels)
        support = int(selected.sum())
        correct = frame.loc[selected, "predicted_label"].astype(str).eq(
            frame.loc[selected, "behavior_label"].astype(str)
        )
        inside = frame.loc[selected, "predicted_label"].astype(str).isin(labels)
        report[name] = {
            "classes": list(labels),
            "support": support,
            "accuracy": float(correct.mean()) if support else 0.0,
            "macro_f1": float(
                np.mean([float(per_class[label]["f1"]) for label in labels])
            ),
            "predicted_inside_group_rate": (
                float(inside.mean()) if support else 0.0
            ),
        }
    return report


def _recording_report(frame: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for video_key, group in frame.groupby("video_key", sort=True):
        metrics = evaluate_predictions(
            group,
            y_true_col="behavior_label",
            y_pred_col="predicted_label",
            label_order=list(VALID_BEHAVIORS),
        )
        rows.append(
            {
                "video_key": str(video_key),
                "native_units": int(len(group)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1_global_10_class": float(metrics["macro_f1"]),
                "macro_f1_supported_classes": float(
                    metrics["macro_f1_supported"]
                ),
            }
        )
    values = np.asarray(
        [row["macro_f1_global_10_class"] for row in rows],
        dtype=np.float64,
    )
    return {
        "video_count": len(rows),
        "macro_f1_global_10_class_min": float(values.min()),
        "macro_f1_global_10_class_median": float(np.median(values)),
        "macro_f1_global_10_class_max": float(values.max()),
        "rows": rows,
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations < 1000:
        raise ValueError("L8 cluster bootstrap requires at least 1000 iterations")
    clusters = sorted(frame["video_key"].astype(str).unique())
    grouped = {
        key: frame.loc[frame["video_key"].astype(str).eq(key)]
        for key in clusters
    }
    rng = np.random.default_rng(seed)
    macro = np.empty(iterations, dtype=np.float64)
    accuracy = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        metrics = evaluate_predictions(
            sample,
            y_true_col="behavior_label",
            y_pred_col="predicted_label",
            label_order=list(VALID_BEHAVIORS),
        )
        macro[index] = float(metrics["macro_f1"])
        accuracy[index] = float(metrics["accuracy"])
    return {
        "method": "recording_cluster_bootstrap_percentile",
        "cluster_column": "video_key",
        "cluster_count": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "macro_f1_global_10_class_ci": [
            float(np.quantile(macro, 0.025)),
            float(np.quantile(macro, 0.975)),
        ],
        "accuracy_ci": [
            float(np.quantile(accuracy, 0.025)),
            float(np.quantile(accuracy, 0.975)),
        ],
        "p_value": None,
        "outer_predictions_used_for_model_selection": False,
    }


def _build_outputs(
    *,
    config: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    run_audit: dict[str, Any],
    l7_decision: dict[str, Any],
    l7_decision_path: Path,
    decision_records: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    experiment_rows = _experiment_rows(
        decision_records,
        l7_decision,
        l7_decision_path,
    )
    rejected = _rejected_experiments(experiment_rows)
    run_root = result_path.parent
    checkpoint_path = run_root / "best_validation_checkpoint.pt"
    native_predictions_path = run_root / "validation_native_predictions.csv"
    metrics_path = run_root / "validation_metrics.json"
    promotion = {
        "schema_version": "classification_v2.legacy_development_l8.promotion_decisions.v1",
        **CLAIM_BOUNDARY,
        "decisions": decision_records,
        "l7_selected_loss_policy": "event_balanced_ce",
        "l8_full_candidate_locked": True,
        "errors": [],
        "valid": True,
    }
    finalist = {
        "schema_version": FINALIST_SCHEMA,
        **CLAIM_BOUNDARY,
        "candidate_id": "legacy_16f_t6_sliding_event_balanced_v1",
        "candidate_role": "bounded_legacy_development_candidate",
        "canonical_source_name": "legacy_16f",
        "view_id": "t6_sliding",
        "sequence_length": 6,
        "windows_per_native_unit": 4,
        "temporal_encoder": LOCKED_MODEL["temporal_encoder_name"],
        "loss_policy": "event_balanced_ce",
        "training_engine_loss_name": LOCKED_OPTIMIZATION["loss"],
        "model_parameter_count": LOCKED_MODEL["parameter_count"],
        "full_training_native_units": int(result["train_native_units"]),
        "full_training_windows": int(result["train_windows"]),
        "validation_native_units": int(result["validation_native_units"]),
        "validation_windows": int(result["validation_windows"]),
        "full_training_config": copy.deepcopy(config["full_training_config"]),
        "checkpoint": _artifact_spec(checkpoint_path),
        "validation_native_predictions": _artifact_spec(
            native_predictions_path
        ),
        "validation_metrics": _artifact_spec(metrics_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "run_result_path": str(result_path),
        "run_result_sha256": run_audit["result_sha256"],
        "run_manifest_sha256": run_audit["run_manifest_sha256"],
        "artifact_manifest_sha256": run_audit["artifact_manifest_sha256"],
        "native_prediction_content_sha256": result[
            "native_prediction_content_sha256"
        ],
        "parameter_sha256": result["parameter_sha256"],
        "l7_decision_path": str(l7_decision_path),
        "l7_decision_sha256": file_sha256(l7_decision_path),
        "metrics": copy.deepcopy(evidence["global"]),
        "candidate_locked": True,
        "review_required_before_any_external_or_q2_use": True,
        "merged_reviewed_retraining_required": True,
        "errors": [],
        "valid": True,
    }
    return {
        "experiment_matrix": pd.DataFrame.from_records(experiment_rows),
        "ablation_registry": pd.DataFrame.from_records(
            _ablation_rows(experiment_rows)
        ),
        "promotion_decisions": promotion,
        "rejected_experiments": {
            "schema_version": (
                "classification_v2.legacy_development_l8.rejected_experiments.v1"
            ),
            **CLAIM_BOUNDARY,
            "experiments": rejected,
            "rejected_count": len(rejected),
            "merged_reviewed_reassessment_required": True,
            "errors": [],
            "valid": True,
        },
        "finalist_lock": finalist,
        "model_card": _model_card(finalist, evidence, rejected),
    }


def _experiment_rows(
    decision_records: list[dict[str, Any]],
    l7_decision: dict[str, Any],
    l7_decision_path: Path,
) -> list[dict[str, Any]]:
    rows = [
        {
            "experiment_id": record["experiment_id"],
            "stage": record["stage"],
            "principal_family": record["principal_family"],
            "decision": record["decision"],
            "disposition": record["disposition"],
            "selected_for_candidate": record["disposition"] == "retained",
            "decision_artifact_path": record["path"],
            "decision_artifact_sha256": record["sha256"],
            "lineage_scope": LINEAGE_SCOPE,
            "applies_to_merged_reviewed_data": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        }
        for record in decision_records
    ]
    for policy in ("effective_number_ce", "balanced_softmax"):
        comparison = l7_decision[
            "paired_comparisons_vs_event_balanced_ce"
        ][policy]
        rows.append(
            {
                "experiment_id": f"L7_{policy.upper()}",
                "stage": "L7",
                "principal_family": "imbalance_loss_policy",
                "decision": "REJECT_ALTERNATIVE_LOSS_POLICY",
                "disposition": "rejected",
                "selected_for_candidate": False,
                "decision_artifact_path": str(l7_decision_path),
                "decision_artifact_sha256": file_sha256(l7_decision_path),
                "lineage_scope": LINEAGE_SCOPE,
                "applies_to_merged_reviewed_data": False,
                "reviewed_or_final_claim_allowed": False,
                "q2_claim_allowed": False,
                "macro_f1_delta_vs_event_balanced": comparison[
                    "delta_candidate_minus_baseline"
                ]["macro_f1_global_10_class"],
                "cluster_ci_low": comparison["video_cluster_bootstrap"][
                    "ci_low"
                ],
                "cluster_ci_high": comparison["video_cluster_bootstrap"][
                    "ci_high"
                ],
            }
        )
    rows.append(
        {
            "experiment_id": "L7_EVENT_BALANCED_CE",
            "stage": "L7",
            "principal_family": "imbalance_loss_policy",
            "decision": "RETAIN_EVENT_BALANCED_CE",
            "disposition": "retained",
            "selected_for_candidate": True,
            "decision_artifact_path": str(l7_decision_path),
            "decision_artifact_sha256": file_sha256(l7_decision_path),
            "lineage_scope": LINEAGE_SCOPE,
            "applies_to_merged_reviewed_data": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        }
    )
    return rows


def _ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ablation_id": row["experiment_id"],
            "stage": row["stage"],
            "changed_family": row["principal_family"],
            "one_principal_family_only": True,
            "result": row["disposition"],
            "selected_for_candidate": row["selected_for_candidate"],
            "decision_artifact_sha256": row["decision_artifact_sha256"],
            "lineage_scope": LINEAGE_SCOPE,
            "merged_reviewed_reassessment_required": True,
        }
        for row in rows
    ]


def _rejected_experiments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": row["experiment_id"],
            "stage": row["stage"],
            "principal_family": row["principal_family"],
            "decision": row["decision"],
            "decision_artifact_path": row["decision_artifact_path"],
            "decision_artifact_sha256": row["decision_artifact_sha256"],
            "rejection_scope": LINEAGE_SCOPE,
            "applies_to_merged_reviewed_data": False,
            "reassess_on_merged_reviewed_data": True,
        }
        for row in rows
        if row["disposition"] == "rejected"
    ]


def _model_card(
    finalist: dict[str, Any],
    evidence: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> str:
    global_metrics = evidence["global"]
    groups = evidence["groups"]
    lines = [
        "# legacy_16f bounded development model card",
        "",
        "This is an unreviewed legacy-development candidate, not a reviewed or final model.",
        "It must not be used for Q2 claims or canonical full OOF reporting.",
        "",
        "## Locked candidate",
        "",
        f"- Candidate: `{finalist['candidate_id']}`",
        (
            "- Input: cached actor-only ResNet18 features, T6 sliding, "
            "four windows per burst"
        ),
        "- Temporal encoder: masked mean; loss: event-balanced cross-entropy",
        f"- Parameters: `{finalist['model_parameter_count']}`",
        (
            "- Full training bursts/windows: "
            f"`{finalist['full_training_native_units']}` / "
            f"`{finalist['full_training_windows']}`"
        ),
        "",
        "## Validation evidence",
        "",
        f"- Native macro-F1 (10 classes): `{global_metrics['macro_f1_global_10_class']:.10f}`",
        f"- Accuracy: `{global_metrics['accuracy']:.10f}`",
        f"- NLL: `{global_metrics['nll']:.10f}`",
        f"- Multiclass Brier: `{global_metrics['multiclass_brier']:.10f}`",
        f"- Top-label ECE: `{global_metrics['top_label_ece']:.10f}`",
        f"- Rare-group macro-F1: `{groups['rare']['macro_f1']:.10f}`",
        f"- Interaction macro-F1: `{groups['interaction']['macro_f1']:.10f}`",
        f"- Feeding macro-F1: `{groups['feeding']['macro_f1']:.10f}`",
        f"- Posture macro-F1: `{groups['posture']['macro_f1']:.10f}`",
        f"- Locomotion/exploration macro-F1: `{groups['locomotion_exploration']['macro_f1']:.10f}`",
        f"- Validation videos: `{evidence['video_clusters']}`",
        "",
        "## Runtime",
        "",
        f"- Runtime seconds: `{evidence['runtime']['runtime_seconds']:.4f}`",
        f"- Optimizer steps: `{evidence['runtime']['optimizer_steps']}`",
        f"- Peak reserved VRAM bytes: `{evidence['runtime']['peak_reserved_bytes']}`",
        "- OOM retries: `0`; post-cleanup allocated/reserved bytes: `0/0`",
        "",
        "## Scope and limitations",
        "",
        "- The validation support for fight and playwithtoy is extremely small.",
        (
            "- ROI, geometry, motion, social, union-crop, full-frame, T1, and "
            "alternative loss results are bounded legacy evidence only."
        ),
        f"- Preserved rejected experiments: `{len(rejected)}`.",
        "- Reassess every rejected family on frozen merged-reviewed data.",
        "- Local 4 GiB VRAM was a correctness host, not an architecture limit.",
        "- The outer holdout was not read or predicted for selection.",
        "",
    ]
    return "\n".join(lines)


def _validate_full_candidate(config: dict[str, Any], result: dict[str, Any]) -> None:
    expected_result = {
        **CLAIM_BOUNDARY,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_TRAINING",
        "view_id": "t6_sliding",
        "training_scope": "full_development_baseline",
        "train_native_units": 3652,
        "validation_native_units": 245,
        "train_windows": 14608,
        "validation_windows": 980,
        "optimizer_steps": 1371,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    _require_mapping(result, expected_result, "full candidate result")
    _require_mapping(config["model"], LOCKED_MODEL, "full candidate model")
    _require_mapping(
        config["optimization"],
        LOCKED_OPTIMIZATION,
        "full candidate optimization",
    )
    execution = result["execution"]
    _require_mapping(
        execution,
        {
            "oom": False,
            "oom_retry_count": 0,
            "post_cleanup_allocated_bytes": 0,
            "post_cleanup_reserved_bytes": 0,
            "errors": [],
            "valid": True,
        },
        "full candidate execution",
    )


def _validate_l7_decision(payload: dict[str, Any]) -> None:
    _require_mapping(
        payload,
        {
            **CLAIM_BOUNDARY,
            "status": "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_DECISION",
            "errors": [],
            "valid": True,
        },
        "L7 decision",
    )
    _require_mapping(
        payload["decision"],
        {
            "decision": "RETAIN_EVENT_BALANCED_CE_REJECT_L7_ALTERNATIVES",
            "selected_loss_policy": "event_balanced_ce",
            "full_confirmation_authorized": False,
            "l8_candidate_lock_authorized": True,
        },
        "L7 selection",
    )


def _load_decision_artifacts(
    root: Path,
    values: object,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("decision_artifacts must be a list")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        spec = _object(value, f"decision_artifacts[{index}]")
        path = _resolve_inside(root, spec["path"])
        _require_equal(file_sha256(path), spec["sha256"], f"decision {index} hash")
        payload = _read_json(path)
        decision = payload.get("decision")
        observed = decision.get("decision") if isinstance(decision, dict) else decision
        _require_equal(payload.get("status"), spec["status"], f"decision {index} status")
        _require_equal(observed, spec["decision"], f"decision {index} value")
        records.append(
            {
                "experiment_id": spec["experiment_id"],
                "stage": spec["stage"],
                "principal_family": spec["principal_family"],
                "path": str(path),
                "sha256": spec["sha256"],
                "status": spec["status"],
                "decision": spec["decision"],
                "disposition": spec["disposition"],
            }
        )
    return records


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lock_id",
        *CLAIM_BOUNDARY,
        "implementation_source",
        "full_training_config",
        "full_candidate_result",
        "l7_decision",
        "decision_artifacts",
        "execution_guard",
        "evidence_contract",
        "interpretation_boundary",
        "output",
        "model",
        "optimization",
    }
    _require_exact_keys(config, required, "L8 lock config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "config schema")
    _require_mapping(config, CLAIM_BOUNDARY, "config claim boundary")
    _require_equal(config["model"], LOCKED_MODEL, "locked model")
    _require_equal(
        config["optimization"],
        LOCKED_OPTIMIZATION,
        "locked optimization",
    )
    for name in (
        "implementation_source",
        "full_training_config",
        "full_candidate_result",
        "l7_decision",
    ):
        _validate_bound_spec(config[name], name)
    if not isinstance(config["decision_artifacts"], list):
        raise ValueError("decision_artifacts must be a list")
    for index, value in enumerate(config["decision_artifacts"]):
        spec = _object(value, f"decision_artifacts[{index}]")
        _require_exact_keys(
            spec,
            {
                "experiment_id",
                "stage",
                "principal_family",
                "path",
                "sha256",
                "status",
                "decision",
                "disposition",
            },
            f"decision_artifacts[{index}]",
        )
        _validate_hash(spec["sha256"], f"decision_artifacts[{index}]")
        if spec["disposition"] not in {"retained", "rejected"}:
            raise ValueError("invalid decision disposition")
    _require_equal(
        config["evidence_contract"],
        {
            "validation_native_units": 245,
            "validation_video_clusters": 33,
            "class_order": list(VALID_BEHAVIORS),
            "confusion_groups": {
                name: list(labels) for name, labels in CONFUSION_GROUPS.items()
            },
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260716,
        },
        "evidence contract",
    )
    _require_equal(
        config["interpretation_boundary"],
        {
            "decision_scope": LINEAGE_SCOPE,
            "candidate_role": "bounded_legacy_development_candidate",
            "legacy_dataset_is_legacy_16f_not_merged": True,
            "legacy_rare_support_generalizes_to_merged_data": False,
            "merged_data_has_materially_more_rare_behaviors": True,
            "merged_reviewed_reassessment_required": True,
            "local_vram_is_architecture_limit": False,
            "rented_gpu_allowed_after_target_environment_gate": True,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        },
        "interpretation boundary",
    )
    guard = _object(config["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(config["output"], "output")
    _require_exact_keys(output, {"root_relative_path"}, "output")


def _validate_prediction_claims(frame: pd.DataFrame) -> None:
    expected = {
        "training_scope": "full_development_baseline",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    for field, value in expected.items():
        observed = frame[field].map(_as_bool) if isinstance(value, bool) else frame[field]
        _require_equal(set(observed), {value}, f"candidate predictions {field}")


def _git_guard(root: Path, value: object) -> dict[str, Any]:
    guard = _object(value, "execution_guard")
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    observed = sorted(_status_path(line) for line in lines if line.strip())
    allowed = sorted(
        str(item).replace("\\", "/") for item in guard["allowed_dirty_paths"]
    )
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(item).replace("\\", "/")
        for item in guard["required_tracked_paths"]
    ]
    untracked = [
        path
        for path in required
        if subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        != 0
    ]
    errors = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    if errors:
        raise ValueError("L8 lock Git guard failed: " + "; ".join(errors))
    return {
        "status": "PASS_COMMITTED_INPUT_GUARD",
        "code_sha": _git(root, "rev-parse", "HEAD").strip(),
        "dirty_entries": lines,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": [],
        "valid": True,
    }


def _validate_bound_file(root: Path, value: object, name: str) -> Path:
    spec = _object(value, name)
    path = _resolve_inside(root, spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing={path}")
    _require_equal(file_sha256(path), spec["sha256"], f"{name} hash")
    return path


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_hash(spec["sha256"], name)


def _artifact_spec(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g")


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _resolve_inside(root: Path, value: object) -> Path:
    path = (root / str(value)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root={value}") from error
    return path


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git command failed={' '.join(arguments)}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"')


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object={path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_hash(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} is not a lowercase SHA-256")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} keys mismatch missing={sorted(expected - observed)},"
            f"extra={sorted(observed - expected)}"
        )


def _require_mapping(
    payload: dict[str, Any],
    expected: dict[str, Any],
    name: str,
) -> None:
    for field, value in expected.items():
        _require_equal(payload.get(field), value, f"{name}.{field}")


def _require_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{name} mismatch observed={observed!r},expected={expected!r}"
        )


def _require_close(observed: float, expected: float, name: str) -> None:
    if not np.isclose(observed, expected, atol=1e-8, rtol=1e-8):
        raise ValueError(f"{name} mismatch observed={observed},expected={expected}")


def _as_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value={value!r}")
    return normalized == "true"


__all__ = [
    "CONFIG_SCHEMA",
    "FINALIST_SCHEMA",
    "LOCK_SCHEMA",
    "OUTPUT_FILES",
    "lock_legacy_l8_candidate",
]
