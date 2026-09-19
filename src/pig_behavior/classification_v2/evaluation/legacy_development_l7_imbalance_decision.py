"""Paired short-matrix decision for legacy L7 imbalance policies."""

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
from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.imbalance_losses import LOSS_POLICIES
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance_config import (
    load_l7_imbalance_config,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance_runtime import (
    PASS_MATRIX_STATUS,
    PASS_REPEAT_STATUS,
    audit_l7_imbalance_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_decision.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
BASELINE_POLICY = "event_balanced_ce"
RARE_CLASSES = ("fight", "social-nose", "move", "playwithtoy")
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33

CLAIM_BOUNDARY = {
    "lineage_scope": LINEAGE_SCOPE,
    "human_review_complete": False,
    "reviewed_or_final_claim_allowed": False,
    "q2_claim_allowed": False,
    "canonical_full_oof_authorized": False,
    "outer_holdout_predictions_authorized": False,
}


def evaluate_l7_imbalance_decision(
    config_path: Path,
    *,
    project_root: Path | None = None,
    enforce_git_guard: bool = True,
) -> dict[str, Any]:
    """Audit the short matrix and select one bounded legacy loss policy."""

    resolved_config = config_path.resolve()
    root = (project_root or Path.cwd()).resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _validate_bound_file(
        root,
        config["implementation_source"],
        "implementation_source",
    )
    training_path = _validate_bound_file(
        root,
        config["short_training_config"],
        "short_training_config",
    )
    matrix_path = _validate_bound_file(
        root,
        config["short_matrix_gate"],
        "short_matrix_gate",
    )
    training_config = load_l7_imbalance_config(training_path)
    matrix = _read_json(matrix_path)
    _validate_matrix(matrix, training_config.sha256)
    git_guard = (
        _git_guard(root, config["execution_guard"])
        if enforce_git_guard
        else {
            "status": "SKIPPED_UNIT_TEST_ONLY",
            "errors": [],
            "valid": True,
        }
    )
    packets = {
        policy: _load_packet(
            root,
            training_config,
            policy,
            config["runs"][policy],
        )
        for policy in LOSS_POLICIES
    }
    baseline = packets[BASELINE_POLICY]
    contract = config["paired_contract"]
    decision_contract = config["decision_contract"]
    comparisons = {
        policy: _compare_packets(
            packets[policy],
            baseline,
            contract=contract,
            decision_contract=decision_contract,
            seed_offset=index,
        )
        for index, policy in enumerate(LOSS_POLICIES)
        if policy != BASELINE_POLICY
    }
    decision = _select_policy(packets, comparisons, decision_contract)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_DECISION",
        **CLAIM_BOUNDARY,
        "decision_id": config["decision_id"],
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(implementation),
        "implementation_source_sha256": file_sha256(implementation),
        "short_training_config_path": str(training_path),
        "short_training_config_sha256": training_config.sha256,
        "short_matrix_gate_path": str(matrix_path),
        "short_matrix_gate_sha256": file_sha256(matrix_path),
        "git_guard": git_guard,
        "policies": {
            policy: _packet_summary(packet)
            for policy, packet in packets.items()
        },
        "paired_comparisons_vs_event_balanced_ce": comparisons,
        "decision": decision,
        "interpretation_boundary": copy.deepcopy(
            config["interpretation_boundary"]
        ),
        "warnings": [
            "legacy_16f_rare_support_is_bounded",
            "legacy_16f_loss_decision_does_not_transfer_to_merged_reviewed_data",
            "merged_reviewed_data_has_materially_more_rare_behavior_evidence",
            "short_three_epoch_evidence_does_not_estimate_full_convergence",
            "local_4gb_vram_is_not_an_architecture_rejection_reason",
        ],
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    result["decision_payload_sha256"] = payload_sha256(result)
    return result


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    """Return the config-bound decision output path."""

    config = _read_json(config_path.resolve())
    _validate_config(config)
    return _resolve_inside(project_root.resolve(), config["output"]["artifact_path"])


def _load_packet(
    root: Path,
    training_config: Any,
    policy: str,
    value: object,
) -> dict[str, Any]:
    spec = _object(value, f"runs.{policy}")
    result_path = _bound_path(root, spec, "primary_result", f"{policy} result")
    repeat_path = _bound_path(root, spec, "repeat_gate", f"{policy} repeat")
    audit = audit_l7_imbalance_run(training_config, result_path=result_path)
    if not audit["valid"]:
        raise ValueError(f"L7 run audit failed for {policy}: {audit['errors']}")
    result = _read_json(result_path)
    repeat = _read_json(repeat_path)
    expected_repeat = {
        "status": PASS_REPEAT_STATUS,
        "loss_policy": policy,
        "short_config_sha256": training_config.sha256,
        "full_matrix_authorized": True,
        "valid": True,
        "errors": [],
    }
    _require_mapping(repeat, expected_repeat, f"{policy} repeat gate")
    _require_mapping(result, CLAIM_BOUNDARY, f"{policy} result")
    _require_equal(result["loss_policy"], policy, f"{policy} result policy")
    predictions_path = result_path.parent / "validation_native_predictions.csv"
    predictions = pd.read_csv(predictions_path, low_memory=False)
    metrics = _validate_and_measure_predictions(predictions, policy)
    _require_close(
        metrics["macro_f1_global_10_class"],
        float(result["validation_metrics"]["macro_f1_global_10_class"]),
        f"{policy} result macro-F1",
    )
    _require_close(
        metrics["nll"],
        float(result["validation_metrics"]["nll"]),
        f"{policy} result NLL",
    )
    return {
        "policy": policy,
        "spec": copy.deepcopy(spec),
        "result_path": result_path,
        "repeat_path": repeat_path,
        "run_audit": audit,
        "result": result,
        "repeat": repeat,
        "predictions": predictions,
        "metrics": metrics,
    }


def _validate_and_measure_predictions(
    frame: pd.DataFrame,
    policy: str,
) -> dict[str, Any]:
    probability_columns = [
        "prob_" + label.replace("-", "_") for label in VALID_BEHAVIORS
    ]
    required = {
        "temporal_unit_key",
        "video_key",
        "behavior_label",
        "target_index",
        "predicted_index",
        "predicted_label",
        "loss_policy",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        *probability_columns,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{policy} native predictions missing={missing}")
    if len(frame) != EXPECTED_NATIVE_UNITS:
        raise ValueError(f"{policy} native units={len(frame)}")
    if frame["video_key"].astype(str).nunique() != EXPECTED_VIDEO_CLUSTERS:
        raise ValueError(f"{policy} video cluster count drift")
    if frame["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError(f"{policy} duplicate native units")
    for column in ("temporal_unit_key", "video_key", "behavior_label"):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{policy} blank {column}")
    _require_equal(set(frame["loss_policy"].astype(str)), {policy}, "policy column")
    _validate_prediction_claims(frame, policy)
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{policy} nonfinite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{policy} probability mass drift")
    observed_argmax = probabilities.argmax(axis=1)
    if not np.array_equal(observed_argmax, frame["predicted_index"].to_numpy()):
        raise ValueError(f"{policy} predicted-index argmax drift")
    labels = np.asarray(VALID_BEHAVIORS, dtype=object)
    if not np.array_equal(labels[observed_argmax], frame["predicted_label"]):
        raise ValueError(f"{policy} predicted-label argmax drift")
    global_metrics = evaluate_predictions(
        frame,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )
    calibration = probability_calibration_metrics(
        probabilities,
        frame["target_index"].to_numpy(dtype=np.int64),
        ece_bins=15,
    )
    rare_f1 = float(
        np.mean(
            [
                float(global_metrics["per_class"][label]["f1"])
                for label in RARE_CLASSES
            ]
        )
    )
    predicted_share = frame["predicted_label"].astype(str).value_counts(
        normalize=True
    )
    return {
        "native_units": int(len(frame)),
        "video_clusters": int(frame["video_key"].astype(str).nunique()),
        "macro_f1_global_10_class": float(global_metrics["macro_f1"]),
        "accuracy": float(global_metrics["accuracy"]),
        "rare_group_classes": list(RARE_CLASSES),
        "rare_group_macro_f1": rare_f1,
        "nll": float(calibration["negative_log_likelihood"]),
        "multiclass_brier": float(calibration["multiclass_brier"]),
        "top_label_ece": float(calibration["top_label_ece"]),
        "maximum_predicted_class": str(predicted_share.index[0]),
        "maximum_predicted_class_share": float(predicted_share.iloc[0]),
        "per_class": global_metrics["per_class"],
    }


def _compare_packets(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    contract: dict[str, Any],
    decision_contract: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    left = candidate["predictions"].assign(
        outer_fold_id=contract["validation_fold_id"]
    )
    right = baseline["predictions"].assign(
        outer_fold_id=contract["validation_fold_id"]
    )
    bootstrap = paired_cluster_bootstrap(
        left,
        right,
        cluster_col=contract["cluster_column"],
        unit_col=contract["unit_column"],
        fold_col="outer_fold_id",
        true_col=contract["true_column"],
        pred_col=contract["predicted_column"],
        iterations=int(contract["bootstrap_iterations"]),
        seed=int(contract["bootstrap_seed"]) + seed_offset,
    )
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    deltas = {
        field: float(candidate_metrics[field]) - float(baseline_metrics[field])
        for field in (
            "macro_f1_global_10_class",
            "accuracy",
            "rare_group_macro_f1",
            "nll",
            "multiclass_brier",
            "top_label_ece",
            "maximum_predicted_class_share",
        )
    }
    criteria = {
        "macro_f1_gain_meets_margin": deltas["macro_f1_global_10_class"]
        >= float(decision_contract["minimum_macro_f1_gain"]),
        "video_cluster_ci_low_is_positive": bootstrap["ci_low"] > 0.0,
        "rare_group_drop_within_limit": deltas["rare_group_macro_f1"]
        >= -float(decision_contract["maximum_rare_group_macro_f1_drop"]),
        "majority_collapse_guard_passes": float(
            candidate_metrics["maximum_predicted_class_share"]
        )
        <= float(decision_contract["maximum_predicted_class_share"]),
        "nll_regression_within_limit": deltas["nll"]
        <= float(decision_contract["maximum_nll_increase"]),
        "ece_regression_within_limit": deltas["top_label_ece"]
        <= float(decision_contract["maximum_ece_increase"]),
        "repeat_gate_is_exact": bool(candidate["repeat"]["valid"]),
    }
    return {
        "candidate_policy": candidate["policy"],
        "baseline_policy": baseline["policy"],
        "candidate_metrics": copy.deepcopy(candidate_metrics),
        "baseline_metrics": copy.deepcopy(baseline_metrics),
        "delta_candidate_minus_baseline": deltas,
        "video_cluster_bootstrap": bootstrap,
        "criteria": criteria,
        "promotion_gate_passes": all(criteria.values()),
    }


def _select_policy(
    packets: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    promoted = [
        policy
        for policy, comparison in comparisons.items()
        if comparison["promotion_gate_passes"]
    ]
    if promoted:
        selected = min(
            promoted,
            key=lambda policy: (
                -packets[policy]["metrics"]["macro_f1_global_10_class"],
                packets[policy]["metrics"]["nll"],
                policy,
            ),
        )
    else:
        selected = BASELINE_POLICY
    selected_alternative = selected != BASELINE_POLICY
    return {
        "decision": (
            "PROMOTE_L7_ALTERNATIVE_TO_FULL_CONFIRMATION"
            if selected_alternative
            else "RETAIN_EVENT_BALANCED_CE_REJECT_L7_ALTERNATIVES"
        ),
        "selected_loss_policy": selected,
        "baseline_policy": BASELINE_POLICY,
        "promoted_alternatives": promoted,
        "full_confirmation_authorized": selected_alternative,
        "l8_candidate_lock_authorized": not selected_alternative,
        "thresholds": copy.deepcopy(contract),
        "decision_strength": "bounded_legacy_development_evidence",
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "next_action": (
            "run_exact_full_confirmation_for_selected_alternative"
            if selected_alternative
            else "start_l8_lock_from_retained_event_balanced_t6_base"
        ),
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    result = packet["result"]
    return {
        "loss_policy": packet["policy"],
        "run_id": result["run_id"],
        "result_path": str(packet["result_path"]),
        "run_result_sha256": file_sha256(packet["result_path"]),
        "repeat_gate_path": str(packet["repeat_path"]),
        "repeat_gate_sha256": file_sha256(packet["repeat_path"]),
        "verified_artifacts": packet["run_audit"]["verified_artifacts"],
        "metrics": copy.deepcopy(packet["metrics"]),
        "optimizer_steps": int(result["optimizer_steps"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "peak_reserved_bytes": int(result["execution"]["peak_reserved_bytes"]),
        "post_cleanup_allocated_bytes": int(
            result["execution"]["post_cleanup_allocated_bytes"]
        ),
        "post_cleanup_reserved_bytes": int(
            result["execution"]["post_cleanup_reserved_bytes"]
        ),
        "errors": [],
        "valid": True,
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "decision_id",
        *CLAIM_BOUNDARY,
        "implementation_source",
        "short_training_config",
        "short_matrix_gate",
        "execution_guard",
        "runs",
        "paired_contract",
        "decision_contract",
        "interpretation_boundary",
        "output",
    }
    _require_exact_keys(config, required, "L7 decision config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "config schema")
    _require_mapping(config, CLAIM_BOUNDARY, "config claim boundary")
    for name in (
        "implementation_source",
        "short_training_config",
        "short_matrix_gate",
    ):
        _validate_bound_spec(config[name], name)
    runs = _object(config["runs"], "runs")
    _require_equal(set(runs), set(LOSS_POLICIES), "run policy set")
    for policy, value in runs.items():
        spec = _object(value, f"runs.{policy}")
        _require_exact_keys(
            spec,
            {
                "primary_result_path",
                "primary_result_sha256",
                "repeat_gate_path",
                "repeat_gate_sha256",
            },
            f"runs.{policy}",
        )
        _validate_hash(spec["primary_result_sha256"], f"{policy} result")
        _validate_hash(spec["repeat_gate_sha256"], f"{policy} repeat")
    paired = {
        "unit_column": "temporal_unit_key",
        "cluster_column": "video_key",
        "true_column": "behavior_label",
        "predicted_column": "predicted_label",
        "validation_fold_id": "native_oof_006",
        "expected_native_units": EXPECTED_NATIVE_UNITS,
        "expected_clusters": EXPECTED_VIDEO_CLUSTERS,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260716,
        "class_order": list(VALID_BEHAVIORS),
        "rare_classes": list(RARE_CLASSES),
    }
    _require_equal(config["paired_contract"], paired, "paired contract")
    decision = {
        "minimum_macro_f1_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "maximum_predicted_class_share": 0.5,
        "maximum_nll_increase": 0.05,
        "maximum_ece_increase": 0.02,
    }
    _require_equal(config["decision_contract"], decision, "decision contract")
    boundary = {
        "decision_scope": LINEAGE_SCOPE,
        "legacy_dataset_is_legacy_16f_not_merged": True,
        "legacy_rare_support_generalizes_to_merged_data": False,
        "merged_data_has_materially_more_rare_behaviors": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
        "rented_gpu_allowed_after_target_environment_gate": True,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    _require_equal(config["interpretation_boundary"], boundary, "boundary")
    guard = _object(config["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(config["output"], "output")
    _require_exact_keys(output, {"artifact_path"}, "output")


def _validate_matrix(matrix: dict[str, Any], config_sha256: str) -> None:
    _require_mapping(
        matrix,
        {
            "status": PASS_MATRIX_STATUS,
            "lineage_scope": LINEAGE_SCOPE,
            "short_config_sha256": config_sha256,
            "all_repeat_gates_pass": True,
            "full_expansion_authorized": True,
            "canonical_full_oof_authorized": False,
            "errors": [],
            "valid": True,
        },
        "short matrix",
    )


def _validate_prediction_claims(frame: pd.DataFrame, policy: str) -> None:
    expected = {
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    for field, value in expected.items():
        observed = frame[field].map(_as_bool) if isinstance(value, bool) else frame[field]
        _require_equal(set(observed), {value}, f"{policy} predictions {field}")


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
        raise ValueError("L7 decision Git guard failed: " + "; ".join(errors))
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


def _validate_bound_file(
    root: Path,
    value: object,
    name: str,
) -> Path:
    spec = _object(value, name)
    path = _resolve_inside(root, spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing={path}")
    _require_equal(file_sha256(path), spec["sha256"], f"{name} hash")
    return path


def _bound_path(
    root: Path,
    spec: dict[str, Any],
    prefix: str,
    name: str,
) -> Path:
    path = _resolve_inside(root, spec[f"{prefix}_path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing={path}")
    _require_equal(file_sha256(path), spec[f"{prefix}_sha256"], f"{name} hash")
    return path


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_hash(spec["sha256"], name)


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
    if not np.isclose(observed, expected, atol=1e-9, rtol=1e-9):
        raise ValueError(f"{name} mismatch observed={observed},expected={expected}")


def _as_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value={value!r}")
    return normalized == "true"


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "configured_output_path",
    "evaluate_l7_imbalance_decision",
]
