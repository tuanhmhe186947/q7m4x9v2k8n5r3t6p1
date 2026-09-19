"""Fail-closed paired decisions for legacy L5 temporal controls."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    evaluate_predictions,
)
from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
    payload_sha256,
)

CONFIG_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.paired_decision_config.v1"
)
RESULT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.paired_decision.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
TRAINING_SCOPE = "full_development_baseline"
RARE_CLASSES = ("fight", "social-nose", "playwithtoy", "move")

ARTIFACT_FILES = {
    "environment": "environment.json",
    "preflight": "preflight.json",
    "selection_manifest": "training_selection_manifest.csv",
    "selection_audit": "training_selection_audit.json",
    "epoch_metrics": "epoch_metrics.csv",
    "validation_predictions": "validation_predictions.csv",
    "validation_metrics": "validation_metrics.json",
    "validation_per_class": "validation_per_class.csv",
    "validation_confusion": "validation_confusion.csv",
    "checkpoint": "best_validation_checkpoint.pt",
    "checkpoint_manifest": "checkpoint_manifest.json",
    "prediction_manifest": "prediction_manifest.json",
    "run_result": "run_result.json",
}

RUN_SPEC_KEYS = {
    "role",
    "run_root",
    "run_id",
    "training_config_path",
    "training_config_sha256",
    "artifact_manifest_sha256",
    "run_manifest_sha256",
    "expected_code_sha",
    "expected_training_source_sha256",
    "expected_temporal_encoder",
    "expected_parameter_count",
}

FIXED_RUN_MANIFEST_FIELDS = (
    "lineage_scope",
    "training_scope",
    "selection_content_sha256",
    "train_native_units",
    "validation_native_units",
    "outer_holdout_native_units_loaded",
    "cache_hash",
    "feature_index_hash",
    "fold_manifest_hash",
    "feature_whitelist_hash",
    "fold",
    "outer_holdout_fold",
    "control_id",
    "backbone_name",
    "pretrained_weight_enum",
    "pretrained_weight_sha256",
    "resolution",
    "normalization_name",
    "image_preprocessing",
    "temporal_view_name",
    "sequence_length",
    "seed",
    "epochs",
    "batch_size",
    "evaluation_batch_size",
    "maximum_optimizer_steps",
    "precision",
    "autocast_enabled",
    "oom_retry_allowed",
)


def evaluate_legacy_l5_paired_decision(
    config_path: Path,
    *,
    project_root: Path | None = None,
    enforce_git_guard: bool = True,
) -> dict[str, Any]:
    """Audit two immutable full runs and make one bounded paired decision."""

    resolved_config = config_path.resolve()
    root = (project_root or Path.cwd()).resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    source_spec = _object(config["implementation_source"], "implementation_source")
    source_path = _resolve_inside(root, source_spec["path"])
    source_hash = file_sha256(source_path)
    _require_equal(
        source_hash,
        source_spec["sha256"],
        "comparison implementation source hash",
    )
    git_guard = (
        _git_guard(root, _object(config["execution_guard"], "execution_guard"))
        if enforce_git_guard
        else {
            "status": "SKIPPED_UNIT_TEST_ONLY",
            "code_sha": None,
            "errors": [],
            "valid": True,
        }
    )
    candidate = _load_run_packet(
        root,
        _object(config["candidate"], "candidate"),
    )
    baseline = _load_run_packet(
        root,
        _object(config["baseline"], "baseline"),
    )
    _validate_semantic_pair(candidate, baseline)
    contract = _object(config["paired_contract"], "paired_contract")
    comparison = _compare_packets(candidate, baseline, contract)
    decision = _make_decision(
        comparison,
        _object(config["decision_contract"], "decision_contract"),
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_PAIRED_DECISION",
        "comparison_id": config["comparison_id"],
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(source_path),
        "implementation_source_sha256": source_hash,
        "git_guard": git_guard,
        "candidate_packet": _packet_summary(candidate),
        "baseline_packet": _packet_summary(baseline),
        "paired_comparison": comparison,
        "decision": decision,
        "interpretation_boundary": deepcopy(config["interpretation_boundary"]),
        "optimizer_steps": 0,
        "gpu_required": False,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "warnings": _comparison_warnings(comparison, contract),
        "errors": [],
        "valid": True,
    }
    result["decision_payload_sha256"] = payload_sha256(result)
    return result


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    """Resolve the immutable comparison artifact path from the config."""

    config = _read_json(config_path.resolve())
    _validate_config(config)
    output = _object(config["output"], "output")
    return _resolve_output_inside(project_root.resolve(), output["artifact_path"])


def _validate_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "comparison_id",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "implementation_source",
        "execution_guard",
        "candidate",
        "baseline",
        "paired_contract",
        "decision_contract",
        "interpretation_boundary",
        "output",
    }
    _require_exact_keys(config, expected_keys, "comparison config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA_VERSION, "config schema")
    _require_equal(config["lineage_scope"], LINEAGE_SCOPE, "lineage scope")
    for field in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    ):
        _require_equal(config[field], False, field)
    source = _object(config["implementation_source"], "implementation_source")
    _require_exact_keys(source, {"path", "sha256"}, "implementation_source")
    _require_sha(source["sha256"], "implementation source")
    guard = _object(config["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"require_committed_inputs", "allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    _require_equal(guard["require_committed_inputs"], True, "committed input guard")
    for role in ("candidate", "baseline"):
        spec = _object(config[role], role)
        _require_exact_keys(spec, RUN_SPEC_KEYS, role)
        for field in (
            "training_config_sha256",
            "artifact_manifest_sha256",
            "run_manifest_sha256",
            "expected_training_source_sha256",
        ):
            _require_sha(spec[field], f"{role} {field}")
        if int(spec["expected_parameter_count"]) <= 0:
            raise ValueError(f"{role} parameter count must be positive")
    _require_equal(config["candidate"]["role"], "candidate", "candidate role")
    _require_equal(config["baseline"]["role"], "baseline", "baseline role")
    paired = _object(config["paired_contract"], "paired_contract")
    paired_keys = {
        "unit_column",
        "cluster_column",
        "true_column",
        "predicted_column",
        "validation_fold_id",
        "expected_native_units",
        "expected_clusters",
        "bootstrap_iterations",
        "bootstrap_seed",
        "class_order",
        "rare_classes",
    }
    _require_exact_keys(paired, paired_keys, "paired_contract")
    expected_columns = {
        "unit_column": "temporal_unit_key",
        "cluster_column": "video_key",
        "true_column": "behavior_label",
        "predicted_column": "predicted_label",
    }
    for field, value in expected_columns.items():
        _require_equal(paired[field], value, f"paired {field}")
    _require_equal(paired["class_order"], list(VALID_BEHAVIORS), "class order")
    _require_equal(paired["rare_classes"], list(RARE_CLASSES), "rare classes")
    if int(paired["bootstrap_iterations"]) < 1000:
        raise ValueError("paired bootstrap requires at least 1000 iterations")
    if int(paired["expected_native_units"]) <= 0:
        raise ValueError("expected paired native units must be positive")
    if int(paired["expected_clusters"]) < 2:
        raise ValueError("expected paired clusters must be at least two")
    decision = _object(config["decision_contract"], "decision_contract")
    decision_keys = {
        "minimum_macro_f1_gain_to_override_simpler",
        "require_positive_cluster_ci_low",
        "maximum_rare_group_recall_drop",
        "maximum_runtime_ratio_to_parent",
        "transformer_requires_tcn_promotion",
    }
    _require_exact_keys(decision, decision_keys, "decision_contract")
    if float(decision["minimum_macro_f1_gain_to_override_simpler"]) < 0.0:
        raise ValueError("minimum macro-F1 gain cannot be negative")
    if not 0.0 <= float(decision["maximum_rare_group_recall_drop"]) <= 1.0:
        raise ValueError("rare-group recall drop must be in [0,1]")
    if float(decision["maximum_runtime_ratio_to_parent"]) < 1.0:
        raise ValueError("maximum runtime ratio must be at least one")
    _require_equal(decision["require_positive_cluster_ci_low"], True, "CI rule")
    _require_equal(
        decision["transformer_requires_tcn_promotion"],
        True,
        "Transformer gate",
    )
    boundary = _object(config["interpretation_boundary"], "interpretation_boundary")
    boundary_expected = {
        "decision_scope": LINEAGE_SCOPE,
        "legacy_rejection_applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
        "rented_gpu_allowed_after_target_environment_gate": True,
    }
    _require_equal(boundary, boundary_expected, "interpretation boundary")
    output = _object(config["output"], "output")
    _require_exact_keys(output, {"artifact_path"}, "output")


def _load_run_packet(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_root = _resolve_inside(root, spec["run_root"])
    training_config_path = _resolve_inside(root, spec["training_config_path"])
    _require_equal(
        file_sha256(training_config_path),
        spec["training_config_sha256"],
        f"{spec['role']} training config hash",
    )
    artifact_path = run_root / "artifact_manifest.json"
    run_manifest_path = run_root / "run_manifest.json"
    _require_equal(
        file_sha256(artifact_path),
        spec["artifact_manifest_sha256"],
        f"{spec['role']} artifact manifest hash",
    )
    _require_equal(
        file_sha256(run_manifest_path),
        spec["run_manifest_sha256"],
        f"{spec['role']} run manifest hash",
    )
    artifact_manifest = _read_json(artifact_path)
    artifact_audit = _audit_artifact_manifest(
        run_root,
        artifact_manifest,
        expected_run_id=str(spec["run_id"]),
    )
    training_config = _read_json(training_config_path)
    run_manifest = _read_json(run_manifest_path)
    run_result = _read_json(run_root / ARTIFACT_FILES["run_result"])
    preflight = _read_json(run_root / ARTIFACT_FILES["preflight"])
    metrics_payload = _read_json(run_root / ARTIFACT_FILES["validation_metrics"])
    predictions = pd.read_csv(
        run_root / ARTIFACT_FILES["validation_predictions"],
        low_memory=False,
    )
    per_class = pd.read_csv(
        run_root / ARTIFACT_FILES["validation_per_class"],
        low_memory=False,
    )
    _validate_run_payloads(
        spec,
        training_config,
        run_manifest,
        run_result,
        preflight,
    )
    recomputed = _validate_predictions(predictions, metrics_payload, per_class)
    return {
        "spec": deepcopy(spec),
        "run_root": run_root,
        "training_config": training_config,
        "run_manifest": run_manifest,
        "run_result": run_result,
        "preflight": preflight,
        "predictions": predictions,
        "recomputed_metrics": recomputed,
        "artifact_audit": artifact_audit,
    }


def _audit_artifact_manifest(
    run_root: Path,
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    _require_equal(manifest.get("run_id"), expected_run_id, "artifact run id")
    _require_equal(manifest.get("status"), "completed", "artifact status")
    _require_equal(manifest.get("training_scope"), TRAINING_SCOPE, "artifact scope")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("artifact manifest artifacts must be a list")
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    _require_equal(set(by_name), set(ARTIFACT_FILES), "artifact names")
    observed: dict[str, Any] = {}
    for name, filename in ARTIFACT_FILES.items():
        row = by_name[name]
        path = run_root / filename
        manifest_path = Path(str(row.get("path", ""))).resolve()
        _require_equal(manifest_path, path.resolve(), f"artifact path {name}")
        _require_equal(row.get("direction"), "output", f"artifact direction {name}")
        actual_hash = file_sha256(path)
        actual_size = path.stat().st_size
        _require_equal(actual_hash, row.get("sha256"), f"artifact hash {name}")
        _require_equal(actual_size, int(row.get("size_bytes", -1)), f"artifact size {name}")
        observed[name] = {
            "sha256": actual_hash,
            "size_bytes": actual_size,
        }
    return {
        "required_artifacts": len(ARTIFACT_FILES),
        "verified_artifacts": len(observed),
        "artifacts": observed,
        "valid": True,
    }


def _validate_run_payloads(
    spec: dict[str, Any],
    training_config: dict[str, Any],
    run_manifest: dict[str, Any],
    run_result: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    for name, payload in (
        ("training config", training_config),
        ("run manifest", run_manifest),
        ("run result", run_result),
        ("preflight", preflight),
    ):
        _validate_claim_boundary(payload, name)
    _require_equal(run_manifest.get("run_id"), spec["run_id"], "run manifest id")
    _require_equal(run_result.get("run_id"), spec["run_id"], "run result id")
    _require_equal(run_result.get("valid"), True, "run result valid")
    _require_equal(run_result.get("errors"), [], "run result errors")
    _require_equal(preflight.get("valid"), True, "preflight valid")
    _require_equal(preflight.get("errors"), [], "preflight errors")
    _require_equal(run_result.get("code_sha"), spec["expected_code_sha"], "code SHA")
    _require_equal(
        run_result.get("implementation_source_sha256"),
        spec["expected_training_source_sha256"],
        "run result training source SHA",
    )
    _require_equal(
        run_manifest.get("implementation_source_sha256"),
        spec["expected_training_source_sha256"],
        "run manifest training source SHA",
    )
    _require_equal(
        preflight.get("implementation_source_sha256"),
        spec["expected_training_source_sha256"],
        "preflight training source SHA",
    )
    encoder = _object(training_config.get("model"), "training model").get(
        "temporal_encoder_name"
    )
    _require_equal(encoder, spec["expected_temporal_encoder"], "training encoder")
    _require_equal(
        run_manifest.get("temporal_encoder_name"),
        spec["expected_temporal_encoder"],
        "run encoder",
    )
    _require_equal(
        int(preflight.get("model_parameter_count", -1)),
        int(spec["expected_parameter_count"]),
        "model parameter count",
    )
    _require_equal(run_result.get("outer_holdout_rows_loaded"), 0, "outer rows")
    _require_equal(
        run_result.get("outer_holdout_predictions_created"),
        0,
        "outer predictions",
    )
    _require_equal(run_result.get("source_media_reads"), 0, "source reads")
    execution = _object(run_result.get("execution"), "run execution")
    execution_expected = {
        "oom": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "errors": [],
        "valid": True,
    }
    for field, value in execution_expected.items():
        _require_equal(execution.get(field), value, f"execution {field}")


def _validate_claim_boundary(payload: dict[str, Any], name: str) -> None:
    expected = {
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, value in expected.items():
        _require_equal(payload.get(field), value, f"{name} {field}")


def _validate_predictions(
    predictions: pd.DataFrame,
    metrics_payload: dict[str, Any],
    per_class: pd.DataFrame,
) -> dict[str, Any]:
    probability_columns = [_probability_column(label) for label in VALID_BEHAVIORS]
    required = {
        "prediction_order",
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "target_index",
        "predicted_index",
        "predicted_label",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        *probability_columns,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"validation predictions missing columns: {missing}")
    if predictions["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("validation predictions contain duplicate native units")
    if predictions["window_id"].astype(str).duplicated().any():
        raise ValueError("validation predictions contain duplicate windows")
    for column in (
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "predicted_label",
    ):
        if predictions[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"validation predictions contain blank {column}")
    _require_equal(
        set(predictions["training_scope"].astype(str)),
        {TRAINING_SCOPE},
        "prediction training scope",
    )
    _require_equal(
        set(predictions["lineage_scope"].astype(str)),
        {LINEAGE_SCOPE},
        "prediction lineage scope",
    )
    if _strict_bool(predictions["human_review_complete"]).any():
        raise ValueError("validation predictions claim human review")
    label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    true_labels = predictions["behavior_label"].astype(str)
    predicted_labels = predictions["predicted_label"].astype(str)
    if not set(true_labels).issubset(label_to_index):
        raise ValueError("validation predictions contain invalid true labels")
    if not set(predicted_labels).issubset(label_to_index):
        raise ValueError("validation predictions contain invalid predicted labels")
    expected_targets = true_labels.map(label_to_index).to_numpy(dtype=int)
    expected_predictions = predicted_labels.map(label_to_index).to_numpy(dtype=int)
    if not np.array_equal(
        predictions["target_index"].to_numpy(dtype=int),
        expected_targets,
    ):
        raise ValueError("validation target index does not match behavior label")
    if not np.array_equal(
        predictions["predicted_index"].to_numpy(dtype=int),
        expected_predictions,
    ):
        raise ValueError("validation predicted index does not match label")
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("validation probabilities contain nonfinite values")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("validation probabilities are outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("validation probabilities do not sum to one")
    if not np.array_equal(probabilities.argmax(axis=1), expected_predictions):
        raise ValueError("validation probability argmax does not match prediction")
    metrics = evaluate_predictions(
        predictions,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )
    true_probability = probabilities[np.arange(len(predictions)), expected_targets]
    nll = float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean())
    reported = {
        "native_unit_rows": int(metrics_payload.get("native_unit_rows", -1)),
        "macro_f1": float(metrics_payload.get("macro_f1_global_10_class", -1.0)),
        "accuracy": float(metrics_payload.get("accuracy", -1.0)),
        "nll": float(metrics_payload.get("nll", -1.0)),
    }
    _require_equal(reported["native_unit_rows"], len(predictions), "metric rows")
    _require_close(reported["macro_f1"], metrics["macro_f1"], "macro-F1")
    _require_close(reported["accuracy"], metrics["accuracy"], "accuracy")
    _require_close(reported["nll"], nll, "NLL")
    _validate_per_class(per_class, metrics["per_class"])
    return {
        "native_units": int(len(predictions)),
        "video_clusters": int(predictions["video_key"].astype(str).nunique()),
        "macro_f1_global_10_class": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "nll": nll,
        "per_class": metrics["per_class"],
    }


def _validate_per_class(
    frame: pd.DataFrame,
    expected: dict[str, dict[str, float | int]],
) -> None:
    required = {"behavior_label", "support", "true_positive", "precision", "recall", "f1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"validation per-class missing columns: {missing}")
    labels = frame["behavior_label"].astype(str).tolist()
    _require_equal(labels, list(VALID_BEHAVIORS), "per-class label order")
    for row in frame.itertuples(index=False):
        metrics = expected[str(row.behavior_label)]
        _require_equal(int(row.support), int(metrics["support"]), "class support")
        _require_equal(int(row.true_positive), int(metrics["tp"]), "class TP")
        for field in ("precision", "recall", "f1"):
            _require_close(float(getattr(row, field)), float(metrics[field]), field)


def _validate_semantic_pair(candidate: dict[str, Any], baseline: dict[str, Any]) -> None:
    candidate_config = candidate["training_config"]
    baseline_config = baseline["training_config"]
    for field in (
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "base_config",
        "consumer_parent",
        "data",
        "optimization",
    ):
        _require_equal(candidate_config.get(field), baseline_config.get(field), f"fixed {field}")
    candidate_model = deepcopy(_object(candidate_config.get("model"), "candidate model"))
    baseline_model = deepcopy(_object(baseline_config.get("model"), "baseline model"))
    candidate_encoder = candidate_model.pop("temporal_encoder_name", None)
    baseline_encoder = baseline_model.pop("temporal_encoder_name", None)
    _require_equal(candidate_model, baseline_model, "fixed model fields")
    _require_equal(candidate_encoder, "masked_tcn", "candidate temporal encoder")
    _require_equal(baseline_encoder, "masked_mean", "baseline temporal encoder")
    for field in FIXED_RUN_MANIFEST_FIELDS:
        _require_equal(
            candidate["run_manifest"].get(field),
            baseline["run_manifest"].get(field),
            f"paired run field {field}",
        )
    ablation = _object(candidate_config.get("ablation_contract"), "candidate ablation")
    _require_equal(ablation.get("changed_variable"), "temporal_encoder_name", "ablation variable")
    _require_equal(ablation.get("reference_value"), "masked_mean", "ablation reference")
    _require_equal(ablation.get("candidate_value"), "masked_tcn", "ablation candidate")
    _require_equal(ablation.get("single_variable_only"), True, "single-variable ablation")
    _require_equal(
        ablation.get("reference_full_config_sha256"),
        baseline["spec"].get("training_config_sha256"),
        "ablation reference config hash",
    )


def _compare_packets(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidate_predictions = candidate["predictions"].copy()
    baseline_predictions = baseline["predictions"].copy()
    expected_units = int(contract["expected_native_units"])
    _require_equal(len(candidate_predictions), expected_units, "candidate native units")
    _require_equal(len(baseline_predictions), expected_units, "baseline native units")
    expected_clusters = int(contract["expected_clusters"])
    _require_equal(
        candidate_predictions["video_key"].astype(str).nunique(),
        expected_clusters,
        "candidate video clusters",
    )
    _require_equal(
        baseline_predictions["video_key"].astype(str).nunique(),
        expected_clusters,
        "baseline video clusters",
    )
    unit_col = str(contract["unit_column"])
    equality_columns = (
        "window_id",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "target_index",
    )
    paired = candidate_predictions[[unit_col, *equality_columns]].merge(
        baseline_predictions[[unit_col, *equality_columns]],
        on=unit_col,
        how="outer",
        suffixes=("_candidate", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"candidate and baseline native-unit sets differ: {counts}")
    for field in equality_columns:
        left = paired[f"{field}_candidate"].astype(str)
        right = paired[f"{field}_baseline"].astype(str)
        if not left.eq(right).all():
            raise ValueError(f"paired prediction field differs: {field}")
    baseline_predictions = (
        baseline_predictions.set_index(unit_col)
        .loc[candidate_predictions[unit_col].astype(str)]
        .reset_index()
    )
    fold_col = "development_validation_fold_id"
    for frame in (candidate_predictions, baseline_predictions):
        frame[fold_col] = str(contract["validation_fold_id"])
    bootstrap = paired_cluster_bootstrap(
        _bootstrap_frame(candidate_predictions, fold_col),
        _bootstrap_frame(baseline_predictions, fold_col),
        cluster_col=str(contract["cluster_column"]),
        unit_col=unit_col,
        fold_col=fold_col,
        true_col="true_label",
        pred_col="native_predicted_behavior",
        iterations=int(contract["bootstrap_iterations"]),
        seed=int(contract["bootstrap_seed"]),
        outer_predictions_used_for_model_selection=False,
    )
    candidate_metrics = candidate["recomputed_metrics"]
    baseline_metrics = baseline["recomputed_metrics"]
    per_class = _per_class_comparison(candidate_metrics, baseline_metrics)
    rare = _rare_group_comparison(
        candidate_predictions,
        baseline_predictions,
        tuple(str(value) for value in contract["rare_classes"]),
    )
    runtime = _resource_comparison(candidate, baseline)
    outcomes = _paired_outcomes(candidate_predictions, baseline_predictions)
    return {
        "paired_native_units": expected_units,
        "paired_video_clusters": expected_clusters,
        "validation_fold_id": str(contract["validation_fold_id"]),
        "candidate_metrics": _global_metrics(candidate_metrics),
        "baseline_metrics": _global_metrics(baseline_metrics),
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": float(
                candidate_metrics["macro_f1_global_10_class"]
                - baseline_metrics["macro_f1_global_10_class"]
            ),
            "accuracy": float(candidate_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "nll": float(candidate_metrics["nll"] - baseline_metrics["nll"]),
        },
        "video_cluster_bootstrap": bootstrap,
        "per_class": per_class,
        "rare_group": rare,
        "paired_outcomes": outcomes,
        "resource_comparison": runtime,
    }


def _bootstrap_frame(frame: pd.DataFrame, fold_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": frame["temporal_unit_key"].astype(str),
            "video_key": frame["video_key"].astype(str),
            fold_col: frame[fold_col].astype(str),
            "true_label": frame["behavior_label"].astype(str),
            "native_predicted_behavior": frame["predicted_label"].astype(str),
        }
    )


def _per_class_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in VALID_BEHAVIORS:
        left = candidate["per_class"][label]
        right = baseline["per_class"][label]
        _require_equal(left["support"], right["support"], f"paired support {label}")
        rows.append(
            {
                "behavior_label": label,
                "support": int(left["support"]),
                "candidate_precision": float(left["precision"]),
                "baseline_precision": float(right["precision"]),
                "precision_delta": float(left["precision"] - right["precision"]),
                "candidate_recall": float(left["recall"]),
                "baseline_recall": float(right["recall"]),
                "recall_delta": float(left["recall"] - right["recall"]),
                "candidate_f1": float(left["f1"]),
                "baseline_f1": float(right["f1"]),
                "f1_delta": float(left["f1"] - right["f1"]),
            }
        )
    return rows


def _rare_group_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    rare_classes: tuple[str, ...],
) -> dict[str, Any]:
    candidate_rare = candidate["behavior_label"].astype(str).isin(rare_classes)
    baseline_rare = baseline["behavior_label"].astype(str).isin(rare_classes)
    _require_equal(candidate_rare.tolist(), baseline_rare.tolist(), "rare paired rows")
    support = int(candidate_rare.sum())
    if support <= 0:
        raise ValueError("rare-group guardrail has zero support")
    candidate_recall = float(
        candidate.loc[candidate_rare, "behavior_label"]
        .astype(str)
        .eq(candidate.loc[candidate_rare, "predicted_label"].astype(str))
        .mean()
    )
    baseline_recall = float(
        baseline.loc[baseline_rare, "behavior_label"]
        .astype(str)
        .eq(baseline.loc[baseline_rare, "predicted_label"].astype(str))
        .mean()
    )
    return {
        "classes": list(rare_classes),
        "support": support,
        "candidate_micro_recall": candidate_recall,
        "baseline_micro_recall": baseline_recall,
        "recall_delta_candidate_minus_baseline": candidate_recall - baseline_recall,
        "recall_drop_candidate_vs_baseline": baseline_recall - candidate_recall,
        "merged_support_inference_allowed": False,
    }


def _resource_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    candidate_runtime = float(candidate["run_result"]["runtime_seconds"])
    baseline_runtime = float(baseline["run_result"]["runtime_seconds"])
    candidate_parameters = int(candidate["preflight"]["model_parameter_count"])
    baseline_parameters = int(baseline["preflight"]["model_parameter_count"])
    candidate_vram = int(candidate["run_result"]["execution"]["peak_reserved_bytes"])
    baseline_vram = int(baseline["run_result"]["execution"]["peak_reserved_bytes"])
    if baseline_runtime <= 0.0 or baseline_parameters <= 0 or baseline_vram <= 0:
        raise ValueError("baseline resource denominator must be positive")
    return {
        "candidate_runtime_seconds": candidate_runtime,
        "baseline_runtime_seconds": baseline_runtime,
        "runtime_ratio_candidate_to_baseline": candidate_runtime / baseline_runtime,
        "candidate_parameter_count": candidate_parameters,
        "baseline_parameter_count": baseline_parameters,
        "parameter_ratio_candidate_to_baseline": candidate_parameters / baseline_parameters,
        "candidate_peak_reserved_vram_bytes": candidate_vram,
        "baseline_peak_reserved_vram_bytes": baseline_vram,
        "peak_reserved_vram_ratio_candidate_to_baseline": candidate_vram / baseline_vram,
        "local_vram_is_architecture_rejection_reason": False,
    }


def _paired_outcomes(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, int]:
    candidate_correct = candidate["behavior_label"].astype(str).eq(
        candidate["predicted_label"].astype(str)
    )
    baseline_correct = baseline["behavior_label"].astype(str).eq(
        baseline["predicted_label"].astype(str)
    )
    return {
        "both_correct": int((candidate_correct & baseline_correct).sum()),
        "candidate_only_correct": int((candidate_correct & ~baseline_correct).sum()),
        "baseline_only_correct": int((~candidate_correct & baseline_correct).sum()),
        "both_wrong": int((~candidate_correct & ~baseline_correct).sum()),
    }


def _make_decision(
    comparison: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    delta = float(
        comparison["delta_candidate_minus_baseline"]["macro_f1_global_10_class"]
    )
    ci_low = float(comparison["video_cluster_bootstrap"]["ci_low"])
    rare_drop = float(comparison["rare_group"]["recall_drop_candidate_vs_baseline"])
    runtime_ratio = float(
        comparison["resource_comparison"]["runtime_ratio_candidate_to_baseline"]
    )
    criteria = {
        "macro_f1_gain_exceeds_simpler_margin": delta
        > float(contract["minimum_macro_f1_gain_to_override_simpler"]),
        "video_cluster_ci_low_is_positive": ci_low > 0.0,
        "rare_group_recall_drop_within_limit": rare_drop
        <= float(contract["maximum_rare_group_recall_drop"]),
        "runtime_ratio_within_limit": runtime_ratio
        <= float(contract["maximum_runtime_ratio_to_parent"]),
    }
    promoted = all(criteria.values())
    return {
        "candidate_promoted": promoted,
        "decision": (
            "PROMOTE_T1_FOR_LEGACY_TEMPORAL_LENGTH_SEARCH"
            if promoted
            else "RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH"
        ),
        "criteria": criteria,
        "thresholds": deepcopy(contract),
        "retained_temporal_length_control": "T1_masked_tcn" if promoted else "V1_masked_mean",
        "transformer_action": (
            "CONSIDER_EXACT_GATED_SMALL_TRANSFORMER"
            if promoted
            else "DEFER_TRANSFORMER_NO_SUPPORTING_TCN_EVIDENCE"
        ),
        "next_action": "run_t6_t8_t12_t16_two_protocol_ladder",
        "decision_strength": "bounded_legacy_development_evidence",
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    spec = packet["spec"]
    result = packet["run_result"]
    return {
        "role": spec["role"],
        "run_id": spec["run_id"],
        "run_root": str(packet["run_root"]),
        "training_config_sha256": spec["training_config_sha256"],
        "artifact_manifest_sha256": spec["artifact_manifest_sha256"],
        "run_manifest_sha256": spec["run_manifest_sha256"],
        "training_source_sha256": spec["expected_training_source_sha256"],
        "code_sha": result["code_sha"],
        "temporal_encoder_name": spec["expected_temporal_encoder"],
        "model_parameter_count": int(packet["preflight"]["model_parameter_count"]),
        "artifact_audit": packet["artifact_audit"],
        "recomputed_metrics": _global_metrics(packet["recomputed_metrics"]),
    }


def _comparison_warnings(
    comparison: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    support = int(comparison["rare_group"]["support"])
    return [
        f"legacy_rare_group_support_is_bounded={support}",
        "legacy_rare_class_results_do_not_estimate_merged_reviewed_support",
        "single_development_validation_fold_is_not_external_generalization_evidence",
        f"bootstrap_cluster_unit={contract['cluster_column']}",
    ]


def _global_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "native_units": int(metrics["native_units"]),
        "video_clusters": int(metrics["video_clusters"]),
        "macro_f1_global_10_class": float(metrics["macro_f1_global_10_class"]),
        "accuracy": float(metrics["accuracy"]),
        "nll": float(metrics["nll"]),
    }


def _git_guard(root: Path, guard: dict[str, Any]) -> dict[str, Any]:
    code_sha = _git(root, "rev-parse", "HEAD").strip()
    status_lines = [
        line
        for line in _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
        if line.strip()
    ]
    observed_paths = sorted(_status_path(line) for line in status_lines)
    allowed_paths = sorted(str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"])
    unexpected = sorted(set(observed_paths) - set(allowed_paths))
    required = [str(path).replace("\\", "/") for path in guard["required_tracked_paths"]]
    untracked_required: list[str] = []
    for path in required:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            untracked_required.append(path)
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked_required:
        errors.append(f"required_paths_untracked={untracked_required}")
    if errors:
        raise ValueError("comparison git guard failed: " + "; ".join(errors))
    return {
        "status": "PASS_COMMITTED_INPUT_GUARD",
        "code_sha": code_sha,
        "dirty_entries": status_lines,
        "allowed_dirty_paths": allowed_paths,
        "observed_dirty_paths": observed_paths,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked_required,
        "errors": [],
        "valid": True,
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"git command failed: {' '.join(arguments)}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"')


def _resolve_inside(root: Path, value: object) -> Path:
    path = Path(str(value))
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {resolved}") from exc
    if not resolved.exists():
        raise ValueError(f"required path does not exist: {resolved}")
    return resolved


def _resolve_output_inside(root: Path, value: object) -> Path:
    path = Path(str(value))
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path escapes project root: {resolved}") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - observed)},"
            f"extra={sorted(observed - expected)}"
        )


def _require_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: observed={observed!r},expected={expected!r}")


def _require_close(observed: float, expected: float, name: str) -> None:
    if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} mismatch: observed={observed},expected={expected}")


def _require_sha(value: object, name: str) -> None:
    if not is_sha256(str(value)):
        raise ValueError(f"{name} is not a lowercase SHA-256")


def _strict_bool(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    invalid = ~values.isin(allowed)
    if invalid.any():
        raise ValueError(f"invalid strict boolean rows={int(invalid.sum())}")
    return values.eq("true")


def _probability_column(label: str) -> str:
    return "prob_" + label.replace("-", "_")


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "configured_output_path",
    "evaluate_legacy_l5_paired_decision",
]
