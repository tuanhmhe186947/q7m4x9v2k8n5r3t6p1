"""Paired native-unit decision for the legacy L6 pen-context experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_decision import (
    _compare_packets,
    _global_metrics,
    _object,
    _read_json,
    _require_equal,
    _require_exact_keys,
    _require_mapping,
    _resolve_inside,
    _validate_bound_file,
)
from pig_behavior.classification_v2.training.legacy_development_l6_pen_context import (
    EXPECTED_PARAMETER_COUNT,
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6PenContextConfig,
    load_pen_context_training_config,
    pen_context_training_git_guard,
)
from pig_behavior.classification_v2.training.legacy_development_l6_pen_context_runtime import (
    MATRIX_GATE_SCHEMA,
    PASS_MATRIX_STATUS,
    PASS_REPEAT_STATUS,
    REPEAT_GATE_SCHEMA,
    audit_pen_context_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_decision.v1"
)
PASS_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_SHORT_DECISION"
FAIL_STATUS = "FAIL_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_SHORT_DECISION"
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33
FOCUS_GROUP = ("stand", "move", "explore")


def evaluate_pen_context_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit three deterministic packets and apply the frozen promotion gate."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    _validate_bound_file(
        _resolve_inside(root, config["implementation"]["path"]),
        config["implementation"]["sha256"],
        "pen-context decision implementation",
    )
    training_path = _resolve_inside(
        root,
        config["short_training_config"]["path"],
    )
    _validate_bound_file(
        training_path,
        config["short_training_config"]["sha256"],
        "pen-context short training config",
    )
    training_config = load_pen_context_training_config(training_path)
    _require_equal(
        training_config.payload["promotion_contract"],
        config["decision_contract"],
        "training and decision promotion contract",
    )
    matrix = _load_matrix(root, config["short_matrix_gate"], training_config)
    packets = {
        mode: _load_packet(
            root,
            training_config,
            mode=mode,
            spec=_object(config["runs"][mode], f"runs.{mode}"),
        )
        for mode in MODES
    }
    universe = _validate_common_universe(packets)
    paired_contract = _object(config["paired_contract"], "paired_contract")
    comparisons = {
        "pen_context_vs_parameter_matched_zero": _compare_with_focus(
            packets["pen_context"],
            packets["parameter_matched_zero"],
            contract=paired_contract,
        ),
        "pen_context_vs_availability_only": _compare_with_focus(
            packets["pen_context"],
            packets["availability_only"],
            contract=paired_contract,
        ),
        "availability_only_vs_parameter_matched_zero": _compare_with_focus(
            packets["availability_only"],
            packets["parameter_matched_zero"],
            contract=paired_contract,
        ),
    }
    decision = make_pen_context_decision(
        comparisons,
        contract=_object(config["decision_contract"], "decision_contract"),
    )
    git_guard = pen_context_training_git_guard(training_config)
    errors = [str(value) for value in git_guard["errors"]]
    valid = not errors
    return {
        "schema_version": RESULT_SCHEMA,
        "status": PASS_STATUS if valid else FAIL_STATUS,
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "short_training_config_sha256": training_config.sha256,
        "short_matrix_gate": matrix,
        "common_native_universe": universe,
        "packets": {
            mode: _packet_summary(packet) for mode, packet in packets.items()
        },
        "comparisons": comparisons,
        "decision": decision,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "git_guard": git_guard,
        "errors": errors,
        "valid": valid,
    }


def make_pen_context_decision(
    comparisons: dict[str, dict[str, Any]],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply all declared gates; a scientific rejection is still a valid run."""

    zero = comparisons["pen_context_vs_parameter_matched_zero"]
    availability = comparisons["pen_context_vs_availability_only"]
    diagnostic = comparisons[
        "availability_only_vs_parameter_matched_zero"
    ]
    minimum_gain = float(contract["minimum_macro_f1_gain"])
    minimum_focus_gain = float(
        contract["minimum_focus_group_macro_f1_gain"]
    )
    maximum_availability = float(
        contract["maximum_absolute_availability_only_gain"]
    )
    maximum_rare_drop = float(contract["maximum_rare_group_macro_f1_drop"])
    criteria = {
        "pen_context_gain_vs_zero_meets_margin": (
            _macro_delta(zero) >= minimum_gain
        ),
        "pen_context_gain_vs_availability_meets_margin": (
            _macro_delta(availability) >= minimum_gain
        ),
        "stand_move_explore_gain_vs_zero_meets_margin": (
            _focus_delta(zero) >= minimum_focus_gain
        ),
        "pen_context_vs_zero_cluster_ci_low_positive": (
            _ci_requirement(zero, contract)
        ),
        "pen_context_vs_availability_cluster_ci_low_positive": (
            _ci_requirement(availability, contract)
        ),
        "pen_context_nll_improves_vs_zero": (
            _nll_requirement(zero, contract)
        ),
        "availability_only_is_bounded_diagnostic": (
            abs(_macro_delta(diagnostic)) <= maximum_availability
        ),
        "rare_group_drop_within_limit": (
            float(zero["confusion_groups"]["rare"]["macro_f1_delta"])
            >= -maximum_rare_drop
        ),
        "all_packets_cleanup_zero": True,
        "all_modes_parameter_matched": True,
    }
    authorized = all(criteria.values())
    return {
        "decision": (
            "RETAIN_PEN_CONTEXT_FOR_FULL_LEGACY_DEVELOPMENT"
            if authorized
            else "DO_NOT_EXPAND_PEN_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "criteria": criteria,
        "thresholds": dict(contract),
        "full_pen_context_expansion_authorized": authorized,
        "negative_result_is_valid_evidence": True,
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "availability_only_is_behavior_evidence": False,
        "source_probe_status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
        "next_action": (
            "prepare_hash_bound_full_pen_context_config"
            if authorized
            else "retain_parameter_matched_zero_and_stop_pen_context_expansion"
        ),
    }


def write_pen_context_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = evaluate_pen_context_short_decision(
        config_path,
        project_root=project_root,
    )
    output = configured_output_path(config_path, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    return output, payload


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    config = _read_json(config_path.resolve())
    _validate_config(config)
    return _resolve_inside(project_root.resolve(), config["output_path"])


def _load_matrix(
    root: Path,
    value: object,
    training_config: LegacyL6PenContextConfig,
) -> dict[str, Any]:
    spec = _object(value, "short_matrix_gate")
    path = _resolve_inside(root, spec["path"])
    _validate_bound_file(path, spec["sha256"], "pen-context matrix gate")
    matrix = _read_json(path)
    _require_mapping(
        matrix,
        {
            "schema_version": MATRIX_GATE_SCHEMA,
            "status": PASS_MATRIX_STATUS,
            "lineage_scope": LINEAGE_SCOPE,
            "training_scope": SHORT_SCOPE,
            "short_config_sha256": training_config.sha256,
            "modes": list(MODES),
            "all_process_ids_distinct": True,
            "all_mode_repeat_gates_pass": True,
            "full_expansion_authorized": True,
            "errors": [],
            "valid": True,
        },
        "pen-context matrix gate",
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": matrix["status"],
        "all_process_ids_distinct": True,
        "valid": True,
    }


def _load_packet(
    root: Path,
    training_config: LegacyL6PenContextConfig,
    *,
    mode: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    result_path = _resolve_inside(root, spec["result_path"])
    _validate_bound_file(result_path, spec["result_sha256"], f"{mode} result")
    audit = audit_pen_context_run(training_config, result_path=result_path)
    if not audit["valid"]:
        raise ValueError(f"pen-context run audit failed mode={mode}")
    _require_equal(audit["mode"], mode, f"{mode} audit mode")
    _require_equal(
        audit["run_manifest_sha256"],
        spec["run_manifest_sha256"],
        f"{mode} run-manifest hash",
    )
    _require_equal(
        audit["artifact_manifest_sha256"],
        spec["artifact_manifest_sha256"],
        f"{mode} artifact-manifest hash",
    )
    repeat_path = _resolve_inside(root, spec["repeat_gate_path"])
    _validate_bound_file(
        repeat_path,
        spec["repeat_gate_sha256"],
        f"{mode} repeat gate",
    )
    repeat = _read_json(repeat_path)
    _require_mapping(
        repeat,
        {
            "schema_version": REPEAT_GATE_SCHEMA,
            "status": PASS_REPEAT_STATUS,
            "mode": mode,
            "short_config_sha256": training_config.sha256,
            "full_mode_expansion_authorized": True,
            "errors": [],
            "valid": True,
        },
        f"{mode} repeat gate",
    )
    run_root = result_path.parent
    return {
        "mode": mode,
        "audit": audit,
        "result": _object(audit["result"], f"{mode} result"),
        "predictions": pd.read_csv(
            run_root / "validation_native_predictions.csv"
        ),
        "confusion_groups": pd.read_csv(
            run_root / "validation_confusion_groups.csv"
        ),
        "repeat_gate_path": str(repeat_path),
        "repeat_gate_sha256": file_sha256(repeat_path),
    }


def _validate_common_universe(
    packets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference: pd.DataFrame | None = None
    lineage_reference: tuple[object, ...] | None = None
    process_ids: set[int] = set()
    for mode in MODES:
        result = packets[mode]["result"]
        lineage = (
            result["config_sha256"],
            result["selection_content_sha256"],
            result["normalization_state_sha256"],
            result["optimizer_steps"],
        )
        if lineage_reference is None:
            lineage_reference = lineage
        elif lineage != lineage_reference:
            raise ValueError(f"paired training lineage differs mode={mode}")
        process_ids.add(int(result["process_id"]))
        frame = packets[mode]["predictions"]
        required = {
            "temporal_unit_key",
            "video_key",
            "behavior_label",
            "predicted_label",
            "pen_context_mode",
            "missing_modality",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{mode} native predictions missing={missing}")
        if len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{mode} native rows={len(frame)}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{mode} duplicate native units")
        if set(frame["pen_context_mode"].astype(str)) != {mode}:
            raise ValueError(f"{mode} prediction mode drift")
        if frame["missing_modality"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{mode} predictions marked missing")
        metadata = frame[
            ["temporal_unit_key", "video_key", "behavior_label"]
        ].astype(str).sort_values("temporal_unit_key", kind="mergesort")
        metadata = metadata.reset_index(drop=True)
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"paired native universe differs mode={mode}")
        execution = _object(result["execution"], f"{mode} execution")
        if (
            execution["oom"]
            or execution["post_cleanup_allocated_bytes"] != 0
            or execution["post_cleanup_reserved_bytes"] != 0
        ):
            raise ValueError(f"{mode} runtime cleanup drift")
    assert reference is not None
    assert lineage_reference is not None
    _require_equal(
        int(reference["video_key"].nunique()),
        EXPECTED_VIDEO_CLUSTERS,
        "video clusters",
    )
    _require_equal(len(process_ids), len(MODES), "primary process IDs")
    return {
        "native_units": len(reference),
        "video_clusters": EXPECTED_VIDEO_CLUSTERS,
        "modes": list(MODES),
        "exact_metadata_equality": True,
        "exact_training_lineage_equality": True,
        "selection_content_sha256": lineage_reference[1],
        "normalization_state_sha256": lineage_reference[2],
        "outer_holdout_rows": 0,
    }


def _compare_with_focus(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    comparison = _compare_packets(candidate, baseline, contract=contract)
    candidate_focus = _focus_group_metrics(candidate["predictions"])
    baseline_focus = _focus_group_metrics(baseline["predictions"])
    comparison["focus_group"] = {
        "labels": list(FOCUS_GROUP),
        "support": candidate_focus["support"],
        "candidate_macro_f1": candidate_focus["macro_f1"],
        "baseline_macro_f1": baseline_focus["macro_f1"],
        "macro_f1_delta": (
            candidate_focus["macro_f1"] - baseline_focus["macro_f1"]
        ),
        "candidate_per_class_f1": candidate_focus["per_class_f1"],
        "baseline_per_class_f1": baseline_focus["per_class_f1"],
    }
    return comparison


def _focus_group_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    true = frame["behavior_label"].astype(str)
    predicted = frame["predicted_label"].astype(str)
    per_class: dict[str, float] = {}
    for label in FOCUS_GROUP:
        true_label = true.eq(label)
        predicted_label = predicted.eq(label)
        true_positive = int((true_label & predicted_label).sum())
        false_positive = int((~true_label & predicted_label).sum())
        false_negative = int((true_label & ~predicted_label).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label] = (
            2 * true_positive / denominator if denominator else 0.0
        )
    return {
        "support": int(true.isin(FOCUS_GROUP).sum()),
        "macro_f1": float(sum(per_class.values()) / len(FOCUS_GROUP)),
        "per_class_f1": per_class,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    result = packet["result"]
    execution = _object(result["execution"], "execution")
    return {
        "mode": packet["mode"],
        "run_id": result["run_id"],
        "process_id": result["process_id"],
        "result_sha256": packet["audit"]["result_sha256"],
        "run_manifest_sha256": packet["audit"]["run_manifest_sha256"],
        "artifact_manifest_sha256": packet["audit"][
            "artifact_manifest_sha256"
        ],
        "validation_metrics": _global_metrics(result["validation_metrics"]),
        "focus_group": _focus_group_metrics(packet["predictions"]),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "optimizer_steps": result["optimizer_steps"],
        "runtime_seconds": result["runtime_seconds"],
        "peak_reserved_bytes": execution["peak_reserved_bytes"],
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "oom": False,
        "valid": True,
        "repeat_gate_path": packet["repeat_gate_path"],
        "repeat_gate_sha256": packet["repeat_gate_sha256"],
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "short_training_config",
        "short_matrix_gate",
        "runs",
        "paired_contract",
        "decision_contract",
        "interpretation_boundary",
        "implementation",
        "output_path",
    }
    _require_exact_keys(config, required, "pen-context decision config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "config schema")
    _require_equal(config["lineage_scope"], LINEAGE_SCOPE, "lineage scope")
    for field in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    ):
        _require_equal(config[field], False, field)
    for name in (
        "short_training_config",
        "short_matrix_gate",
        "implementation",
    ):
        _validate_bound_spec(config[name], name)
    runs = _object(config["runs"], "runs")
    _require_equal(set(runs), set(MODES), "run mode set")
    run_fields = {
        "result_path",
        "result_sha256",
        "run_manifest_sha256",
        "artifact_manifest_sha256",
        "repeat_gate_path",
        "repeat_gate_sha256",
    }
    for mode, value in runs.items():
        spec = _object(value, f"runs.{mode}")
        _require_exact_keys(spec, run_fields, f"runs.{mode}")
        for field in run_fields:
            if field.endswith("sha256"):
                _validate_sha(spec[field], f"runs.{mode}.{field}")
    _require_equal(
        _object(config["paired_contract"], "paired_contract"),
        {
            "unit_column": "temporal_unit_key",
            "cluster_column": "video_key",
            "validation_fold_id": "legacy_l6_pen_context_short_validation_v1",
            "expected_native_units": EXPECTED_NATIVE_UNITS,
            "expected_clusters": EXPECTED_VIDEO_CLUSTERS,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260717,
            "focus_group": list(FOCUS_GROUP),
        },
        "paired contract",
    )
    _require_equal(
        _object(config["decision_contract"], "decision_contract"),
        {
            "minimum_macro_f1_gain": 0.01,
            "minimum_focus_group_macro_f1_gain": 0.01,
            "maximum_absolute_availability_only_gain": 0.01,
            "maximum_rare_group_macro_f1_drop": 0.02,
            "require_positive_video_cluster_ci_low": True,
            "require_nll_improvement_vs_zero": True,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260717,
        },
        "decision contract",
    )
    _require_equal(
        _object(config["interpretation_boundary"], "interpretation_boundary"),
        {
            "legacy_only_decision": True,
            "architecture_family_finalized": False,
            "applies_to_merged_reviewed_data": False,
            "merged_reviewed_reassessment_required": True,
            "availability_only_is_behavior_evidence": False,
            "negative_result_is_valid_evidence": True,
        },
        "interpretation boundary",
    )


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_sha(spec["sha256"], f"{name}.sha256")


def _validate_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} is not lowercase SHA256")


def _macro_delta(comparison: dict[str, Any]) -> float:
    return float(
        comparison["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )


def _focus_delta(comparison: dict[str, Any]) -> float:
    return float(comparison["focus_group"]["macro_f1_delta"])


def _ci_requirement(
    comparison: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    if not bool(contract["require_positive_video_cluster_ci_low"]):
        return True
    return float(comparison["video_cluster_bootstrap"]["ci_low"]) > 0.0


def _nll_requirement(
    comparison: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    if not bool(contract["require_nll_improvement_vs_zero"]):
        return True
    return float(comparison["delta_candidate_minus_baseline"]["nll"]) < 0.0


__all__ = [
    "CONFIG_SCHEMA",
    "FOCUS_GROUP",
    "RESULT_SCHEMA",
    "evaluate_pen_context_short_decision",
    "make_pen_context_decision",
    "write_pen_context_short_decision",
]
