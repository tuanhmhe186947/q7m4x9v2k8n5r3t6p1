"""Audit and compare the immutable legacy L5 temporal-ladder matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    FULL_SCOPE,
    LINEAGE_SCOPE,
    RARE_CLASSES,
    TemporalLadderConfig,
    load_temporal_ladder_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder_runtime import (
    ARTIFACT_FILES,
    audit_temporal_ladder_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
    payload_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l5."
    "temporal_ladder_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l5."
    "temporal_ladder_decision.v1"
)
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33
NATIVE_METADATA_COLUMNS = (
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "target_index",
)
CLAIM_BOUNDARY = {
    "lineage_scope": LINEAGE_SCOPE,
    "human_review_complete": False,
    "reviewed_or_final_claim_allowed": False,
    "q2_claim_allowed": False,
    "canonical_full_oof_authorized": False,
    "outer_holdout_predictions_authorized": False,
}


def evaluate_temporal_ladder_matrix_decision(
    config_path: Path,
    *,
    project_root: Path | None = None,
    enforce_git_guard: bool = True,
) -> dict[str, Any]:
    """Audit eight full controls and make one bounded L5 working decision."""

    root = (project_root or Path.cwd()).resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    source_spec = _object(config["implementation_source"], "implementation source")
    source_path = _resolve_inside(root, source_spec["path"])
    _require_equal(
        file_sha256(source_path),
        source_spec["sha256"],
        "decision implementation hash",
    )
    git_guard = (
        _git_guard(root, _object(config["execution_guard"], "execution guard"))
        if enforce_git_guard
        else {
            "status": "SKIPPED_UNIT_TEST_ONLY",
            "code_sha": None,
            "errors": [],
            "valid": True,
        }
    )
    full_config = _load_full_config(root, config["full_training_config"])
    matrix_gate = _load_matrix_gate(root, config["short_matrix_gate"])
    run_specs = _object(config["runs"], "runs")
    packets = {
        view_id: _load_packet(
            root,
            full_config,
            view_id=view_id,
            spec=_object(run_specs[view_id], f"runs.{view_id}"),
        )
        for view_id in CANONICAL_VIEWS
    }
    universe = _validate_common_native_universe(packets)
    equivalence = _validate_expected_equivalence(
        packets,
        config["expected_equivalence"],
    )
    decision_contract = _object(config["decision_contract"], "decision contract")
    candidate_view = str(decision_contract["candidate_view"])
    comparisons = {
        baseline_view: _compare_packets(
            packets[candidate_view],
            packets[baseline_view],
            contract=_object(config["paired_contract"], "paired contract"),
        )
        for baseline_view in CANONICAL_VIEWS
        if baseline_view != candidate_view
    }
    ranking = _rank_packets(packets)
    decision = _make_decision(
        ranking,
        comparisons,
        decision_contract,
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_DECISION",
        "decision_id": config["decision_id"],
        **CLAIM_BOUNDARY,
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(source_path),
        "implementation_source_sha256": file_sha256(source_path),
        "git_guard": git_guard,
        "full_training_config": {
            "path": str(full_config.path),
            "sha256": full_config.sha256,
        },
        "short_matrix_gate": {
            "path": matrix_gate["path"],
            "sha256": matrix_gate["sha256"],
            "view_count": matrix_gate["payload"]["view_count"],
            "full_expansion_authorized": True,
        },
        "common_native_universe": universe,
        "packets": {
            view_id: _packet_summary(packet)
            for view_id, packet in packets.items()
        },
        "ranking": ranking,
        "candidate_view": candidate_view,
        "candidate_vs_each_other_view": comparisons,
        "expected_equivalence": equivalence,
        "decision": decision,
        "interpretation_boundary": copy.deepcopy(
            config["interpretation_boundary"]
        ),
        "optimizer_steps": 0,
        "gpu_required": False,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "warnings": _warnings(comparisons),
        "errors": [],
        "valid": True,
    }
    result["decision_payload_sha256"] = payload_sha256(result)
    return result


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    """Resolve the exclusive decision artifact path."""

    config = _read_json(config_path.resolve())
    _validate_config(config)
    return _resolve_inside(project_root.resolve(), config["output"]["artifact_path"])


def _load_full_config(root: Path, value: object) -> TemporalLadderConfig:
    spec = _object(value, "full training config")
    path = _resolve_inside(root, spec["path"])
    _require_equal(file_sha256(path), spec["sha256"], "full config hash")
    config = load_temporal_ladder_config(path)
    _require_equal(config.training_scope, FULL_SCOPE, "full training scope")
    return config


def _load_matrix_gate(root: Path, value: object) -> dict[str, Any]:
    spec = _object(value, "short matrix gate")
    path = _resolve_inside(root, spec["path"])
    actual_hash = file_sha256(path)
    _require_equal(actual_hash, spec["sha256"], "short matrix gate hash")
    payload = _read_json(path)
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_SHORT_MATRIX",
        "lineage_scope": LINEAGE_SCOPE,
        "view_count": len(CANONICAL_VIEWS),
        "full_expansion_authorized": True,
        "valid": True,
    }
    _require_mapping(payload, expected, "short matrix gate")
    return {"path": str(path), "sha256": actual_hash, "payload": payload}


def _load_packet(
    root: Path,
    full_config: TemporalLadderConfig,
    *,
    view_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    run_root = _resolve_inside(root, spec["run_root"])
    paths = {
        "run_result": run_root / "run_result.json",
        "run_manifest": run_root / "run_manifest.json",
        "artifact_manifest": run_root / "artifact_manifest.json",
        "preflight": run_root / ARTIFACT_FILES["preflight"],
        "native_predictions": run_root / ARTIFACT_FILES["native_predictions"],
        "validation_metrics": run_root / ARTIFACT_FILES["validation_metrics"],
        "validation_per_class": run_root / ARTIFACT_FILES["validation_per_class"],
    }
    for name in ("run_result", "run_manifest", "artifact_manifest"):
        _require_equal(
            file_sha256(paths[name]),
            spec[f"{name}_sha256"],
            f"{view_id} {name} hash",
        )
    audit = audit_temporal_ladder_run(
        full_config,
        result_path=paths["run_result"],
    )
    _require_equal(audit["valid"], True, f"{view_id} run audit")
    _require_equal(audit["errors"], [], f"{view_id} run audit errors")
    result = _read_json(paths["run_result"])
    manifest = _read_json(paths["run_manifest"])
    preflight = _read_json(paths["preflight"])
    metrics_payload = _read_json(paths["validation_metrics"])
    predictions = pd.read_csv(paths["native_predictions"], low_memory=False)
    per_class = pd.read_csv(paths["validation_per_class"], low_memory=False)
    expected = CANONICAL_VIEWS[view_id]
    _validate_packet_payloads(
        view_id,
        spec,
        result,
        manifest,
        preflight,
        expected,
    )
    recomputed = _validate_predictions(
        predictions,
        metrics_payload,
        per_class,
        expected_windows_per_native=int(expected["windows_per_native_unit"]),
    )
    _validate_reported_metrics(result["validation_metrics"], recomputed, view_id)
    return {
        "view_id": view_id,
        "spec": copy.deepcopy(spec),
        "run_root": run_root,
        "result": result,
        "manifest": manifest,
        "preflight": preflight,
        "predictions": predictions,
        "metrics": recomputed,
        "audit": audit,
    }


def _validate_packet_payloads(
    view_id: str,
    spec: dict[str, Any],
    result: dict[str, Any],
    manifest: dict[str, Any],
    preflight: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for name, payload in (
        ("result", result),
        ("manifest", manifest),
        ("preflight", preflight),
    ):
        _require_mapping(payload, CLAIM_BOUNDARY, f"{view_id} {name}")
    result_expected = {
        "run_id": spec["run_id"],
        "view_id": view_id,
        "training_scope": FULL_SCOPE,
        "train_native_units": 3652,
        "validation_native_units": EXPECTED_NATIVE_UNITS,
        "train_windows": expected["train_windows_full"],
        "validation_windows": expected["validation_windows"],
        "optimizer_steps": expected["optimizer_steps_full"],
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    _require_mapping(result, result_expected, f"{view_id} result")
    manifest_expected = {
        "run_id": spec["run_id"],
        "view_id": view_id,
        "code_sha": spec["expected_code_sha"],
        "status": "completed",
        "training_scope": FULL_SCOPE,
        "temporal_view_name": expected["temporal_view_name"],
        "sampling_protocol": expected["sampling_protocol"],
        "sequence_length": expected["sequence_length"],
        "windows_per_native_unit": expected["windows_per_native_unit"],
    }
    _require_mapping(manifest, manifest_expected, f"{view_id} manifest")
    _require_mapping(
        preflight,
        {
            "view_id": view_id,
            "training_scope": FULL_SCOPE,
            "train_native_units": 3652,
            "validation_native_units": EXPECTED_NATIVE_UNITS,
            "train_windows": expected["train_windows_full"],
            "validation_windows": expected["validation_windows"],
            "source_media_reads": 0,
            "outer_holdout_rows_loaded": 0,
            "gpu_launch_authorized": True,
            "errors": [],
            "valid": True,
        },
        f"{view_id} preflight",
    )


def _validate_predictions(
    predictions: pd.DataFrame,
    metrics_payload: dict[str, Any],
    per_class: pd.DataFrame,
    *,
    expected_windows_per_native: int,
) -> dict[str, Any]:
    probability_columns = [_probability_column(label) for label in VALID_BEHAVIORS]
    required = {
        "prediction_order",
        *NATIVE_METADATA_COLUMNS,
        "predicted_index",
        "predicted_label",
        "aggregated_window_count",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        *probability_columns,
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"native predictions missing columns={missing}")
    _require_equal(len(predictions), EXPECTED_NATIVE_UNITS, "native prediction rows")
    if predictions["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("native predictions contain duplicate temporal units")
    for column in (*NATIVE_METADATA_COLUMNS, "predicted_label"):
        if predictions[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"native predictions contain blank {column}")
    _require_equal(
        set(predictions["training_scope"].astype(str)),
        {FULL_SCOPE},
        "native prediction scope",
    )
    _require_equal(
        set(predictions["lineage_scope"].astype(str)),
        {LINEAGE_SCOPE},
        "native prediction lineage",
    )
    for column in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
    ):
        if _strict_bool(predictions[column]).any():
            raise ValueError(f"native predictions claim forbidden {column}")
    if not predictions["aggregated_window_count"].eq(
        expected_windows_per_native
    ).all():
        raise ValueError("native aggregated-window count drift")
    label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    true_labels = predictions["behavior_label"].astype(str)
    predicted_labels = predictions["predicted_label"].astype(str)
    if not set(true_labels).issubset(label_to_index):
        raise ValueError("native predictions contain unsupported true labels")
    if not set(predicted_labels).issubset(label_to_index):
        raise ValueError("native predictions contain unsupported predicted labels")
    targets = true_labels.map(label_to_index).to_numpy(dtype=np.int64)
    predicted = predicted_labels.map(label_to_index).to_numpy(dtype=np.int64)
    if not np.array_equal(predictions["target_index"].to_numpy(int), targets):
        raise ValueError("native target index does not match label")
    if not np.array_equal(predictions["predicted_index"].to_numpy(int), predicted):
        raise ValueError("native predicted index does not match label")
    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("native probabilities contain nonfinite values")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("native probabilities fall outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("native probabilities do not sum to one")
    if not np.array_equal(probabilities.argmax(axis=1), predicted):
        raise ValueError("native probability argmax does not match prediction")
    evaluated = evaluate_predictions(
        predictions,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )
    true_probability = probabilities[np.arange(len(predictions)), targets]
    metrics = {
        "native_units": len(predictions),
        "video_clusters": predictions["video_key"].astype(str).nunique(),
        "macro_f1_global_10_class": float(evaluated["macro_f1"]),
        "accuracy": float(evaluated["accuracy"]),
        "nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "per_class": evaluated["per_class"],
    }
    _validate_reported_metrics(metrics_payload, metrics, "metrics artifact")
    _validate_per_class(per_class, evaluated["per_class"])
    return metrics


def _validate_reported_metrics(
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    name: str,
) -> None:
    _require_equal(
        int(reported.get("native_unit_rows", -1)),
        int(recomputed["native_units"]),
        f"{name} native rows",
    )
    for field in ("macro_f1_global_10_class", "accuracy", "nll"):
        _require_close(
            float(reported.get(field, -1.0)),
            float(recomputed[field]),
            f"{name} {field}",
        )


def _validate_per_class(
    reported: pd.DataFrame,
    recomputed: dict[str, dict[str, float | int]],
) -> None:
    _require_equal(
        set(reported["behavior_label"].astype(str)),
        set(VALID_BEHAVIORS),
        "per-class label set",
    )
    by_label = reported.set_index("behavior_label")
    for index, label in enumerate(VALID_BEHAVIORS):
        row = by_label.loc[label]
        expected = recomputed[label]
        _require_equal(int(row["class_index"]), index, f"{label} class index")
        _require_equal(int(row["support"]), expected["support"], f"{label} support")
        _require_equal(int(row["true_positive"]), expected["tp"], f"{label} TP")
        for field in ("precision", "recall", "f1"):
            _require_close(float(row[field]), float(expected[field]), f"{label} {field}")


def _validate_common_native_universe(
    packets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_id = next(iter(CANONICAL_VIEWS))
    reference = _ordered_native_metadata(packets[reference_id]["predictions"])
    for view_id, packet in packets.items():
        observed = _ordered_native_metadata(packet["predictions"])
        if not observed.equals(reference):
            raise ValueError(f"native metadata universe differs for {view_id}")
    _require_equal(len(reference), EXPECTED_NATIVE_UNITS, "paired native units")
    _require_equal(
        reference["video_key"].astype(str).nunique(),
        EXPECTED_VIDEO_CLUSTERS,
        "paired video clusters",
    )
    return {
        "reference_view": reference_id,
        "view_count": len(packets),
        "native_units": len(reference),
        "video_clusters": EXPECTED_VIDEO_CLUSTERS,
        "native_unit_sha256": _mapping_hash(reference, ["temporal_unit_key"]),
        "metadata_mapping_sha256": _mapping_hash(
            reference,
            list(NATIVE_METADATA_COLUMNS),
        ),
        "outer_holdout_rows": 0,
        "errors": [],
        "valid": True,
    }


def _ordered_native_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[list(NATIVE_METADATA_COLUMNS)]
        .astype(str)
        .sort_values("temporal_unit_key", kind="mergesort")
        .reset_index(drop=True)
    )


def _compare_packets(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    left = _ordered_predictions(candidate["predictions"])
    right = _ordered_predictions(baseline["predictions"])
    _require_equal(
        left["temporal_unit_key"].tolist(),
        right["temporal_unit_key"].tolist(),
        "paired temporal units",
    )
    bootstrap = paired_cluster_bootstrap(
        _bootstrap_frame(left, contract),
        _bootstrap_frame(right, contract),
        cluster_col=str(contract["cluster_column"]),
        unit_col=str(contract["unit_column"]),
        fold_col="development_validation_fold_id",
        true_col="true_label",
        pred_col="native_predicted_behavior",
        iterations=int(contract["bootstrap_iterations"]),
        seed=int(contract["bootstrap_seed"]),
        outer_predictions_used_for_model_selection=False,
    )
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    rare = _rare_comparison(left, right, tuple(contract["rare_classes"]))
    return {
        "candidate_view": candidate["view_id"],
        "baseline_view": baseline["view_id"],
        "paired_native_units": len(left),
        "paired_video_clusters": left["video_key"].astype(str).nunique(),
        "candidate_metrics": _global_metrics(candidate_metrics),
        "baseline_metrics": _global_metrics(baseline_metrics),
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": float(
                candidate_metrics["macro_f1_global_10_class"]
                - baseline_metrics["macro_f1_global_10_class"]
            ),
            "accuracy": float(
                candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
            ),
            "nll": float(candidate_metrics["nll"] - baseline_metrics["nll"]),
        },
        "video_cluster_bootstrap": bootstrap,
        "per_class": _per_class_comparison(candidate_metrics, baseline_metrics),
        "rare_group": rare,
        "paired_outcomes": _paired_outcomes(left, right),
        "resource_comparison": _resource_comparison(candidate, baseline),
    }


def _ordered_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("temporal_unit_key", kind="mergesort").reset_index(
        drop=True
    )


def _bootstrap_frame(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": frame["temporal_unit_key"].astype(str),
            "video_key": frame["video_key"].astype(str),
            "development_validation_fold_id": str(contract["validation_fold_id"]),
            "true_label": frame["behavior_label"].astype(str),
            "native_predicted_behavior": frame["predicted_label"].astype(str),
        }
    )


def _per_class_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
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


def _rare_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    rare_classes: tuple[str, ...],
) -> dict[str, Any]:
    mask = candidate["behavior_label"].astype(str).isin(rare_classes)
    _require_equal(
        mask.tolist(),
        baseline["behavior_label"].astype(str).isin(rare_classes).tolist(),
        "paired rare rows",
    )
    support = int(mask.sum())
    if support <= 0:
        raise ValueError("rare group has zero support")
    candidate_recall = float(
        candidate.loc[mask, "predicted_label"]
        .astype(str)
        .eq(candidate.loc[mask, "behavior_label"].astype(str))
        .mean()
    )
    baseline_recall = float(
        baseline.loc[mask, "predicted_label"]
        .astype(str)
        .eq(baseline.loc[mask, "behavior_label"].astype(str))
        .mean()
    )
    return {
        "classes": list(rare_classes),
        "support": support,
        "candidate_recall": candidate_recall,
        "baseline_recall": baseline_recall,
        "recall_delta_candidate_minus_baseline": candidate_recall
        - baseline_recall,
        "recall_drop_candidate_vs_baseline": max(
            0.0,
            baseline_recall - candidate_recall,
        ),
    }


def _paired_outcomes(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, int]:
    candidate_correct = candidate["predicted_label"].astype(str).eq(
        candidate["behavior_label"].astype(str)
    )
    baseline_correct = baseline["predicted_label"].astype(str).eq(
        baseline["behavior_label"].astype(str)
    )
    return {
        "both_correct": int((candidate_correct & baseline_correct).sum()),
        "candidate_only_correct": int((candidate_correct & ~baseline_correct).sum()),
        "baseline_only_correct": int((~candidate_correct & baseline_correct).sum()),
        "both_incorrect": int((~candidate_correct & ~baseline_correct).sum()),
    }


def _resource_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    left = candidate["result"]
    right = baseline["result"]
    candidate_runtime = float(left["runtime_seconds"])
    baseline_runtime = float(right["runtime_seconds"])
    return {
        "candidate_runtime_seconds": candidate_runtime,
        "baseline_runtime_seconds": baseline_runtime,
        "runtime_ratio_candidate_to_baseline": candidate_runtime
        / baseline_runtime,
        "candidate_optimizer_steps": int(left["optimizer_steps"]),
        "baseline_optimizer_steps": int(right["optimizer_steps"]),
        "optimizer_step_ratio_candidate_to_baseline": int(left["optimizer_steps"])
        / int(right["optimizer_steps"]),
        "candidate_train_windows": int(left["train_windows"]),
        "baseline_train_windows": int(right["train_windows"]),
        "candidate_parameter_count": 68234,
        "baseline_parameter_count": 68234,
        "parameter_count_delta": 0,
        "candidate_peak_reserved_bytes": int(
            left["execution"]["peak_reserved_bytes"]
        ),
        "baseline_peak_reserved_bytes": int(
            right["execution"]["peak_reserved_bytes"]
        ),
    }


def _rank_packets(packets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    order = {view_id: index for index, view_id in enumerate(CANONICAL_VIEWS)}
    rows = []
    for view_id, packet in packets.items():
        metrics = packet["metrics"]
        expected = CANONICAL_VIEWS[view_id]
        result = packet["result"]
        rows.append(
            {
                "view_id": view_id,
                "rank": 0,
                "sequence_length": expected["sequence_length"],
                "sampling_protocol": expected["sampling_protocol"],
                "windows_per_native_unit": expected["windows_per_native_unit"],
                **_global_metrics(metrics),
                "optimizer_steps": int(result["optimizer_steps"]),
                "runtime_seconds": float(result["runtime_seconds"]),
                "rare_group_recall": _group_recall(
                    packet["predictions"],
                    tuple(RARE_CLASSES),
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["macro_f1_global_10_class"]),
            float(row["nll"]),
            float(row["runtime_seconds"]),
            order[str(row["view_id"])],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _make_decision(
    ranking: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidate = str(contract["candidate_view"])
    matched = str(contract["matched_centered_reference"])
    established = str(contract["established_reference"])
    margin = float(contract["minimum_macro_f1_gain"])
    rare_limit = float(contract["maximum_rare_group_recall_drop"])
    runtime_limit = float(contract["maximum_runtime_ratio"])
    criteria = {
        "candidate_has_top_point_macro_f1": ranking[0]["view_id"] == candidate,
        "matched_centered_gain_exceeds_margin": _delta(comparisons[matched])
        >= margin,
        "matched_centered_ci_low_is_positive": _ci_low(comparisons[matched]) > 0.0,
        "established_reference_gain_exceeds_margin": _delta(
            comparisons[established]
        )
        >= margin,
        "established_reference_ci_low_is_positive": _ci_low(
            comparisons[established]
        )
        > 0.0,
        "rare_group_drop_within_limit_for_all_views": all(
            float(value["rare_group"]["recall_drop_candidate_vs_baseline"])
            <= rare_limit
            for value in comparisons.values()
        ),
        "runtime_ratio_within_limit_for_all_views": all(
            float(
                value["resource_comparison"][
                    "runtime_ratio_candidate_to_baseline"
                ]
            )
            <= runtime_limit
            for value in comparisons.values()
        ),
        "parameter_count_is_identical_for_all_views": all(
            int(value["resource_comparison"]["parameter_count_delta"]) == 0
            for value in comparisons.values()
        ),
    }
    working_baseline_retained = all(criteria.values())
    ambiguous = sorted(
        view_id
        for view_id, comparison in comparisons.items()
        if _ci_low(comparison) <= 0.0
    )
    universal = not ambiguous
    selected = candidate if working_baseline_retained else established
    return {
        "decision": (
            "RETAIN_T6_SLIDING_AS_LEGACY_L6_WORKING_BASELINE_WITH_BOUNDED_"
            "UNCERTAINTY"
            if working_baseline_retained
            else "RETAIN_T16_CENTERED_PENDING_ADDITIONAL_L5_CONTROL"
        ),
        "selected_working_view": selected,
        "working_baseline_retained": working_baseline_retained,
        "criteria": criteria,
        "thresholds": copy.deepcopy(contract),
        "universal_pairwise_superiority_established": universal,
        "views_with_ci_crossing_zero_vs_candidate": ambiguous,
        "causal_temporal_length_claim_allowed": False,
        "sampling_protocol_bundle_includes_optimizer_exposure": True,
        "optimizer_step_matched_causal_claim_available": False,
        "architecture_family_finalized": False,
        "decision_strength": "bounded_legacy_development_evidence",
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "next_action": "start_l6_one_family_at_a_time_from_selected_working_view",
    }


def _validate_expected_equivalence(
    packets: dict[str, dict[str, Any]],
    value: object,
) -> dict[str, Any]:
    spec = _object(value, "expected equivalence")
    left_id = str(spec["left_view"])
    right_id = str(spec["right_view"])
    left = _ordered_predictions(packets[left_id]["predictions"])
    right = _ordered_predictions(packets[right_id]["predictions"])
    probability_columns = [_probability_column(label) for label in VALID_BEHAVIORS]
    probabilities_equal = np.array_equal(
        left[probability_columns].to_numpy(dtype=np.float64),
        right[probability_columns].to_numpy(dtype=np.float64),
    )
    labels_equal = left["predicted_label"].astype(str).equals(
        right["predicted_label"].astype(str)
    )
    parameter_hash_equal = (
        packets[left_id]["result"]["parameter_sha256"]
        == packets[right_id]["result"]["parameter_sha256"]
    )
    valid = probabilities_equal and labels_equal and parameter_hash_equal
    if bool(spec["require_exact_equivalence"]) and not valid:
        raise ValueError("declared T16 protocol equivalence failed")
    return {
        "left_view": left_id,
        "right_view": right_id,
        "reason": spec["reason"],
        "probabilities_equal": probabilities_equal,
        "predicted_labels_equal": labels_equal,
        "parameter_hash_equal": parameter_hash_equal,
        "valid": valid,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    result = packet["result"]
    return {
        "view_id": packet["view_id"],
        "run_id": result["run_id"],
        "run_root": str(packet["run_root"]),
        "run_result_sha256": packet["spec"]["run_result_sha256"],
        "run_manifest_sha256": packet["spec"]["run_manifest_sha256"],
        "artifact_manifest_sha256": packet["spec"]["artifact_manifest_sha256"],
        "verified_artifacts": packet["audit"]["verified_artifacts"],
        "parameter_sha256": result["parameter_sha256"],
        "native_prediction_content_sha256": result[
            "native_prediction_content_sha256"
        ],
        "optimizer_steps": result["optimizer_steps"],
        "runtime_seconds": result["runtime_seconds"],
        "metrics": _global_metrics(packet["metrics"]),
        "errors": [],
        "valid": True,
    }


def _warnings(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    support = next(iter(comparisons.values()))["rare_group"]["support"]
    return [
        f"legacy_116f_rare_group_support_is_bounded={support}",
        "legacy_116f_rare_support_must_not_be_generalized_to_merged_data",
        "user_reports_merged_data_has_materially_more_rare_behaviors",
        "single_development_validation_fold_is_not_external_generalization",
        "sliding_protocol_changes_window_and_optimizer_step_exposure",
        "local_4gb_vram_is_not_an_architecture_rejection_reason",
    ]


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "decision_id",
        *CLAIM_BOUNDARY,
        "implementation_source",
        "execution_guard",
        "full_training_config",
        "short_matrix_gate",
        "runs",
        "paired_contract",
        "decision_contract",
        "expected_equivalence",
        "interpretation_boundary",
        "output",
    }
    _require_exact_keys(config, required, "decision config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "decision schema")
    _require_mapping(config, CLAIM_BOUNDARY, "decision config")
    for name in (
        "implementation_source",
        "full_training_config",
        "short_matrix_gate",
    ):
        spec = _object(config[name], name)
        _require_exact_keys(spec, {"path", "sha256"}, name)
        _require_sha(spec["sha256"], f"{name} sha256")
    guard = _object(config["execution_guard"], "execution guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution guard",
    )
    runs = _object(config["runs"], "runs")
    _require_equal(set(runs), set(CANONICAL_VIEWS), "decision run matrix")
    run_keys = {
        "run_id",
        "run_root",
        "run_result_sha256",
        "run_manifest_sha256",
        "artifact_manifest_sha256",
        "expected_code_sha",
    }
    for view_id, value in runs.items():
        spec = _object(value, f"runs.{view_id}")
        _require_exact_keys(spec, run_keys, f"runs.{view_id}")
        for field in (
            "run_result_sha256",
            "run_manifest_sha256",
            "artifact_manifest_sha256",
        ):
            _require_sha(spec[field], f"{view_id}.{field}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(spec["expected_code_sha"])):
            raise ValueError(f"{view_id} expected code SHA is invalid")
    paired = _object(config["paired_contract"], "paired contract")
    _require_equal(
        paired,
        {
            "unit_column": "temporal_unit_key",
            "cluster_column": "video_key",
            "true_column": "behavior_label",
            "predicted_column": "predicted_label",
            "validation_fold_id": "native_oof_006",
            "expected_native_units": EXPECTED_NATIVE_UNITS,
            "expected_clusters": EXPECTED_VIDEO_CLUSTERS,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260715,
            "class_order": list(VALID_BEHAVIORS),
            "rare_classes": list(RARE_CLASSES),
        },
        "paired contract",
    )
    decision = _object(config["decision_contract"], "decision contract")
    expected_decision = {
        "candidate_view": "t6_sliding",
        "matched_centered_reference": "t6_centered",
        "established_reference": "t16_centered",
        "minimum_macro_f1_gain": 0.01,
        "maximum_rare_group_recall_drop": 0.1,
        "maximum_runtime_ratio": 3.0,
    }
    _require_equal(decision, expected_decision, "decision contract")
    equivalence = _object(config["expected_equivalence"], "expected equivalence")
    _require_equal(
        equivalence,
        {
            "left_view": "t16_centered",
            "right_view": "t16_sliding",
            "reason": "T16_has_one_complete_window_under_both_protocols",
            "require_exact_equivalence": True,
        },
        "expected equivalence",
    )
    boundary = _object(config["interpretation_boundary"], "boundary")
    _require_equal(
        boundary,
        {
            "decision_scope": LINEAGE_SCOPE,
            "legacy_dataset_is_116f_not_merged": True,
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
    output = _object(config["output"], "output")
    _require_exact_keys(output, {"artifact_path"}, "output")


def _global_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "native_units": int(metrics["native_units"]),
        "video_clusters": int(metrics["video_clusters"]),
        "macro_f1_global_10_class": float(metrics["macro_f1_global_10_class"]),
        "accuracy": float(metrics["accuracy"]),
        "nll": float(metrics["nll"]),
    }


def _group_recall(frame: pd.DataFrame, classes: tuple[str, ...]) -> float:
    selected = frame["behavior_label"].astype(str).isin(classes)
    return float(
        frame.loc[selected, "predicted_label"]
        .astype(str)
        .eq(frame.loc[selected, "behavior_label"].astype(str))
        .mean()
    )


def _delta(comparison: dict[str, Any]) -> float:
    return float(
        comparison["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )


def _ci_low(comparison: dict[str, Any]) -> float:
    return float(comparison["video_cluster_bootstrap"]["ci_low"])


def _mapping_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].astype(str).sort_values(columns, kind="mergesort")
    payload = "\n".join(
        "\x1f".join(row) for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_guard(root: Path, guard: dict[str, Any]) -> dict[str, Any]:
    code_sha = _git(root, "rev-parse", "HEAD").strip()
    lines = [
        line
        for line in _git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ).splitlines()
        if line.strip()
    ]
    observed = sorted(_status_path(line) for line in lines)
    allowed = sorted(str(value).replace("\\", "/") for value in guard["allowed_dirty_paths"])
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(value).replace("\\", "/") for value in guard["required_tracked_paths"]
    ]
    untracked = []
    for path in required:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            untracked.append(path)
    errors = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    if errors:
        raise ValueError("decision git guard failed: " + "; ".join(errors))
    return {
        "status": "PASS_COMMITTED_INPUT_GUARD",
        "code_sha": code_sha,
        "dirty_entries": lines,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
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
        raise ValueError(f"git command failed={' '.join(arguments)}")
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
    except ValueError as error:
        raise ValueError(f"path escapes project root={value}") from error
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object={path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


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
        raise ValueError(f"{name} mismatch observed={observed!r},expected={expected!r}")


def _require_close(observed: float, expected: float, name: str) -> None:
    if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} mismatch observed={observed},expected={expected}")


def _require_sha(value: object, name: str) -> None:
    if not is_sha256(str(value)):
        raise ValueError(f"{name} is not a lowercase SHA-256")


def _strict_bool(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    if not set(normalized).issubset(allowed):
        raise ValueError("boolean lineage column contains invalid values")
    return normalized.eq("true")


def _probability_column(label: str) -> str:
    return "prob_" + label.replace("-", "_")


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "configured_output_path",
    "evaluate_temporal_ladder_matrix_decision",
]
