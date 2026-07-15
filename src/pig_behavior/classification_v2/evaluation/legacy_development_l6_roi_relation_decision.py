"""Paired native-unit/video-cluster decision for the legacy L6 ROI gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_cache_repeat import (
    _git_guard,
)
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
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation import (
    EXPECTED_PARAMETER_COUNT,
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6ROIRelationConfig,
    load_roi_relation_training_config,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_runtime import (
    MATRIX_GATE_SCHEMA,
    REPEAT_GATE_SCHEMA,
    audit_roi_relation_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_decision.v1"
)
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33


def evaluate_roi_relation_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit the three short packets and produce a paired ROI decision."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    _validate_bound_file(
        _resolve_inside(root, config["implementation"]["path"]),
        config["implementation"]["sha256"],
        "ROI decision implementation",
    )
    training_path = _resolve_inside(
        root,
        config["short_training_config"]["path"],
    )
    _validate_bound_file(
        training_path,
        config["short_training_config"]["sha256"],
        "ROI short training config",
    )
    training_config = load_roi_relation_training_config(training_path)
    if training_config.training_scope != SHORT_SCOPE:
        raise ValueError("L6 ROI decision requires short training scope")
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
    contract = _object(config["paired_contract"], "paired_contract")
    comparisons = {
        "roi_relation_vs_parameter_matched_zero": _compare_packets(
            packets["roi_relation"],
            packets["parameter_matched_zero"],
            contract=contract,
        ),
        "roi_relation_vs_availability_only": _compare_packets(
            packets["roi_relation"],
            packets["availability_only"],
            contract=contract,
        ),
        "availability_only_vs_parameter_matched_zero": _compare_packets(
            packets["availability_only"],
            packets["parameter_matched_zero"],
            contract=contract,
        ),
    }
    decision = make_roi_relation_decision(
        comparisons,
        contract=_object(config["decision_contract"], "decision_contract"),
    )
    git_guard = _git_guard(
        root,
        config,
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_DECISION"
            if not git_guard["errors"]
            else "FAIL_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_DECISION"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "canonical_source_name": training_config.payload[
            "canonical_source_name"
        ],
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
        "errors": git_guard["errors"],
        "valid": not git_guard["errors"],
    }


def make_roi_relation_decision(
    comparisons: dict[str, dict[str, Any]],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply predeclared ROI promotion criteria without hiding rejection evidence."""

    zero = comparisons["roi_relation_vs_parameter_matched_zero"]
    availability = comparisons["roi_relation_vs_availability_only"]
    diagnostic = comparisons[
        "availability_only_vs_parameter_matched_zero"
    ]
    minimum_gain = float(contract["minimum_macro_f1_gain"])
    maximum_availability = float(
        contract["maximum_absolute_availability_only_gain"]
    )
    maximum_rare_drop = float(contract["maximum_rare_group_macro_f1_drop"])
    rare_delta = float(
        zero["confusion_groups"]["rare"]["macro_f1_delta"]
    )
    zero_delta = float(
        zero["delta_candidate_minus_baseline"]["macro_f1_global_10_class"]
    )
    availability_delta = float(
        availability["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )
    diagnostic_delta = float(
        diagnostic["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )
    criteria = {
        "roi_gain_vs_zero_meets_margin": zero_delta >= minimum_gain,
        "roi_gain_vs_availability_meets_margin": (
            availability_delta >= minimum_gain
        ),
        "roi_vs_zero_cluster_ci_low_positive": (
            float(zero["video_cluster_bootstrap"]["ci_low"]) > 0.0
        ),
        "roi_vs_availability_cluster_ci_low_positive": (
            float(availability["video_cluster_bootstrap"]["ci_low"]) > 0.0
        ),
        "roi_nll_improves_vs_zero": (
            float(zero["delta_candidate_minus_baseline"]["nll"]) < 0.0
        ),
        "availability_only_is_bounded_diagnostic": (
            abs(diagnostic_delta) <= maximum_availability
        ),
        "rare_group_drop_within_limit": rare_delta >= -maximum_rare_drop,
        "all_packets_cleanup_zero": True,
        "all_modes_parameter_matched": True,
    }
    authorized = all(criteria.values())
    return {
        "decision": (
            "RETAIN_ROI_RELATION_FOR_FULL_LEGACY_DEVELOPMENT"
            if authorized
            else "DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "criteria": criteria,
        "thresholds": copy.deepcopy(contract),
        "full_roi_relation_expansion_authorized": authorized,
        "negative_result_is_valid_evidence": True,
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "availability_only_is_behavior_evidence": False,
        "source_probe_status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
        "next_action": (
            "prepare_hash_bound_full_roi_relation_config"
            if authorized
            else "retain_parameter_matched_zero_and_stop_roi_relation_expansion"
        ),
    }


def write_roi_relation_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = evaluate_roi_relation_short_decision(
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
    training_config: LegacyL6ROIRelationConfig,
) -> dict[str, Any]:
    spec = _object(value, "short_matrix_gate")
    path = _resolve_inside(root, spec["path"])
    _validate_bound_file(path, spec["sha256"], "short matrix gate")
    matrix = _read_json(path)
    expected = {
        "schema_version": MATRIX_GATE_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_MATRIX",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "short_config_sha256": training_config.sha256,
        "modes": list(MODES),
        "all_process_ids_distinct": True,
        "all_mode_repeat_gates_pass": True,
        "full_expansion_authorized": True,
        "errors": [],
        "valid": True,
    }
    _require_mapping(matrix, expected, "short matrix gate")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": matrix["status"],
        "all_process_ids_distinct": matrix["all_process_ids_distinct"],
        "valid": matrix["valid"],
    }


def _load_packet(
    root: Path,
    training_config: LegacyL6ROIRelationConfig,
    *,
    mode: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    result_path = _resolve_inside(root, spec["result_path"])
    _validate_bound_file(result_path, spec["result_sha256"], f"{mode} result")
    audit = audit_roi_relation_run(training_config, result_path=result_path)
    if not audit["valid"]:
        raise ValueError(f"L6 ROI relation run audit failed mode={mode}")
    _require_equal(audit["mode"], mode, f"{mode} audit mode")
    _require_equal(
        audit["run_manifest_sha256"],
        spec["run_manifest_sha256"],
        f"{mode} run manifest hash",
    )
    _require_equal(
        audit["artifact_manifest_sha256"],
        spec["artifact_manifest_sha256"],
        f"{mode} artifact manifest hash",
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
            "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_REPEAT",
            "mode": mode,
            "short_config_sha256": training_config.sha256,
            "full_mode_expansion_authorized": True,
            "errors": [],
            "valid": True,
        },
        f"{mode} repeat gate",
    )
    run_root = result_path.parent
    predictions = pd.read_csv(run_root / "validation_native_predictions.csv")
    groups = pd.read_csv(run_root / "validation_confusion_groups.csv")
    result = _object(audit["result"], f"{mode} result")
    return {
        "mode": mode,
        "audit": audit,
        "result": result,
        "predictions": predictions,
        "confusion_groups": groups,
        "repeat_gate_path": str(repeat_path),
        "repeat_gate_sha256": file_sha256(repeat_path),
    }


def _validate_common_universe(
    packets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference: pd.DataFrame | None = None
    lineage_reference: tuple[object, ...] | None = None
    for mode in MODES:
        result = _object(packets[mode]["result"], f"{mode} result")
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
        frame = packets[mode]["predictions"]
        required = {
            "temporal_unit_key",
            "video_key",
            "behavior_label",
            "predicted_label",
            "roi_relation_mode",
            "missing_modality",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{mode} native predictions missing={missing}")
        if len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{mode} native rows={len(frame)}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{mode} duplicate native units")
        if set(frame["roi_relation_mode"].astype(str)) != {mode}:
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
    clusters = int(reference["video_key"].nunique())
    _require_equal(clusters, EXPECTED_VIDEO_CLUSTERS, "video clusters")
    return {
        "native_units": len(reference),
        "video_clusters": clusters,
        "modes": list(MODES),
        "exact_metadata_equality": True,
        "exact_training_lineage_equality": True,
        "selection_content_sha256": lineage_reference[1],
        "normalization_state_sha256": lineage_reference[2],
        "outer_holdout_rows": 0,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    result = packet["result"]
    execution = _object(result["execution"], "execution")
    if (
        execution.get("post_cleanup_allocated_bytes") != 0
        or execution.get("post_cleanup_reserved_bytes") != 0
        or execution.get("oom") is not False
        or execution.get("valid") is not True
    ):
        raise ValueError(f"invalid runtime cleanup mode={packet['mode']}")
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
        "missing_validation_metrics": _global_metrics(
            result["missing_validation_metrics"]
        ),
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
        "execution_guard",
        "output_path",
    }
    _require_exact_keys(config, required, "ROI decision config")
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
    for name in ("short_training_config", "short_matrix_gate", "implementation"):
        _validate_bound_spec(config[name], name)
    runs = _object(config["runs"], "runs")
    _require_equal(set(runs), set(MODES), "run mode set")
    fields = {
        "result_path",
        "result_sha256",
        "run_manifest_sha256",
        "artifact_manifest_sha256",
        "repeat_gate_path",
        "repeat_gate_sha256",
    }
    for mode, value in runs.items():
        spec = _object(value, f"runs.{mode}")
        _require_exact_keys(spec, fields, f"runs.{mode}")
        for field in fields:
            if field.endswith("sha256"):
                _validate_sha(spec[field], f"runs.{mode}.{field}")
    _require_equal(
        _object(config["paired_contract"], "paired_contract"),
        {
            "unit_column": "temporal_unit_key",
            "cluster_column": "video_key",
            "validation_fold_id": "legacy_l6_short_validation_v1",
            "expected_native_units": EXPECTED_NATIVE_UNITS,
            "expected_clusters": EXPECTED_VIDEO_CLUSTERS,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260715,
        },
        "paired contract",
    )
    _require_equal(
        _object(config["decision_contract"], "decision_contract"),
        {
            "minimum_macro_f1_gain": 0.02,
            "maximum_absolute_availability_only_gain": 0.01,
            "maximum_rare_group_macro_f1_drop": 0.02,
            "require_positive_video_cluster_ci_low": True,
            "require_nll_improvement_vs_zero": True,
        },
        "decision contract",
    )
    _require_equal(
        _object(config["interpretation_boundary"], "interpretation boundary"),
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
    guard = _object(config["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_sha(spec["sha256"], f"{name}.sha256")


def _validate_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} is not lowercase SHA256")
