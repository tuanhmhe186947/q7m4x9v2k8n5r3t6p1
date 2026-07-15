"""Paired full-development decision for the legacy L6 ROI confirmation."""

from __future__ import annotations

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
    _resolve_inside,
    _validate_bound_file,
)
from pig_behavior.classification_v2.evaluation.legacy_development_l6_roi_relation_decision import (
    _validate_common_universe,
    make_roi_relation_decision,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation import (
    EXPECTED_PARAMETER_COUNT,
    FULL_SCOPE,
    LINEAGE_SCOPE,
    MODES,
    LegacyL6ROIRelationConfig,
    load_roi_relation_training_config,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_runtime import (
    audit_roi_relation_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_full_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_full_decision.v1"
)
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33


def evaluate_roi_relation_full_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit all full packets and produce the paired confirmation decision."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _object(config["implementation"], "implementation")
    _validate_bound_file(
        _resolve_inside(root, implementation["path"]),
        implementation["sha256"],
        "ROI full decision implementation",
    )
    training_spec = _object(
        config["full_training_config"],
        "full_training_config",
    )
    training_path = _resolve_inside(root, training_spec["path"])
    _validate_bound_file(
        training_path,
        training_spec["sha256"],
        "ROI full training config",
    )
    training_config = load_roi_relation_training_config(training_path)
    if training_config.training_scope != FULL_SCOPE:
        raise ValueError("L6 ROI full decision requires full training scope")
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
    decision["full_confirmation_complete"] = True
    decision["next_action"] = (
        "lock_roi_relation_as_legacy_development_candidate"
        if decision["full_roi_relation_expansion_authorized"]
        else "continue_l6_from_parameter_matched_zero_without_roi_values"
    )
    git_guard = _git_guard(root, config)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_FULL_DECISION"
            if not git_guard["errors"]
            else "FAIL_LEGACY_DEVELOPMENT_L6_ROI_RELATION_FULL_DECISION"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": FULL_SCOPE,
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
        "full_training_config_sha256": training_config.sha256,
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


def write_roi_relation_full_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = evaluate_roi_relation_full_decision(
        config_path,
        project_root=project_root,
    )
    output = _resolve_inside(project_root.resolve(), _read_json(config_path)["output_path"])
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
        raise ValueError(f"L6 ROI full run audit failed mode={mode}")
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
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    result = packet["result"]
    execution = _object(result["execution"], "execution")
    _require_equal(execution.get("oom"), False, f"{packet['mode']} oom")
    _require_equal(
        execution.get("post_cleanup_allocated_bytes"),
        0,
        f"{packet['mode']} allocated cleanup",
    )
    _require_equal(
        execution.get("post_cleanup_reserved_bytes"),
        0,
        f"{packet['mode']} reserved cleanup",
    )
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
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "training_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "full_training_config",
        "runs",
        "paired_contract",
        "decision_contract",
        "interpretation_boundary",
        "implementation",
        "execution_guard",
        "output_path",
    }
    _require_exact_keys(config, required, "ROI full decision config")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "config schema")
    _require_equal(config["lineage_scope"], LINEAGE_SCOPE, "lineage scope")
    _require_equal(config["training_scope"], FULL_SCOPE, "training scope")
    for field in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    ):
        _require_equal(config[field], False, field)
    for name in ("full_training_config", "implementation"):
        spec = _object(config[name], name)
        _require_exact_keys(spec, {"path", "sha256"}, name)
        _validate_sha(spec["sha256"], f"{name}.sha256")
    runs = _object(config["runs"], "runs")
    _require_equal(set(runs), set(MODES), "run mode set")
    fields = {
        "result_path",
        "result_sha256",
        "run_manifest_sha256",
        "artifact_manifest_sha256",
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
            "validation_fold_id": "legacy_l6_full_validation_v1",
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


def _validate_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} is not lowercase SHA256")
