"""Prove completion of the bounded legacy_16f development goal."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development.goal_completion_config.v1"
)
AUDIT_SCHEMA = "classification_v2.legacy_development.goal_completion.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"

CLAIM_BOUNDARY = {
    "lineage_scope": LINEAGE_SCOPE,
    "human_review_complete": False,
    "reviewed_or_final_claim_allowed": False,
    "q2_claim_allowed": False,
    "canonical_full_oof_authorized": False,
    "outer_holdout_predictions_authorized": False,
}

TEMPORAL_VIEWS = {
    "t6_centered",
    "t8_centered",
    "t12_centered",
    "t16_centered",
    "t6_sliding",
    "t8_sliding",
    "t12_sliding",
    "t16_sliding",
}


def write_legacy_goal_completion_audit(
    config_path: Path,
    *,
    project_root: Path | None = None,
    enforce_git_guard: bool = True,
    write_output: bool = True,
) -> dict[str, Any]:
    """Validate L0-L8 evidence and optionally write the immutable handback."""

    root = (project_root or Path.cwd()).resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _bound_file(
        root,
        config["implementation_source"],
        "implementation_source",
    )
    goal_authority = _bound_file(
        root,
        config["goal_authority"],
        "goal_authority",
    )
    git_guard = (
        _git_guard(root, config["execution_guard"])
        if enforce_git_guard
        else {
            "status": "SKIPPED_UNIT_TEST_ONLY",
            "code_sha": _git(root, "rev-parse", "HEAD").strip(),
            "errors": [],
            "valid": True,
        }
    )
    alias_audit = _forbidden_alias_audit(root)
    frozen_inputs = _validate_frozen_inputs(root, config["frozen_inputs"])
    milestone_payloads = _load_milestones(root, config["milestones"])
    milestone_evidence = {
        "L1": _validate_l1(milestone_payloads["L1"]),
        "L2": _validate_l2(milestone_payloads["L2"]),
        "L3": _validate_l3(milestone_payloads["L3"]),
        "L4": _validate_l4(
            milestone_payloads["L4_short"],
            milestone_payloads["L4"],
        ),
        "L5": _validate_l5(milestone_payloads["L5"]),
        "L6": _validate_l6(
            milestone_payloads["L6"],
            config["milestones"]["L6"]["top_k_disposition"],
        ),
        "L7": _validate_l7(milestone_payloads["L7"]),
        "L8": _validate_l8(root, milestone_payloads["L8"]),
    }
    l8 = milestone_payloads["L8"]
    run_lineage = _validate_selected_run_lineage(root, l8, frozen_inputs)
    requirements = [
        _requirement(
            "L0",
            alias_audit["valid"] and git_guard["valid"],
            [
                _artifact_record(goal_authority),
                {
                    "git_code_sha": git_guard["code_sha"],
                    "forbidden_source_alias_occurrences": alias_audit[
                        "occurrences"
                    ],
                },
            ],
        ),
        *[
            _requirement(name, True, [milestone_evidence[name]])
            for name in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")
        ],
    ]
    all_pass = all(item["passed"] for item in requirements)
    if not all_pass:
        raise ValueError("legacy_16f goal completion requirements did not pass")
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS_LEGACY_16F_GOAL_COMPLETION",
        **CLAIM_BOUNDARY,
        "goal_complete": True,
        "parent_reviewed_all_source_goal_complete": False,
        "audit_id": config["audit_id"],
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(implementation),
        "implementation_source_sha256": file_sha256(implementation),
        "goal_authority": _artifact_record(goal_authority),
        "final_code_sha": git_guard["code_sha"],
        "git_guard": git_guard,
        "source_identity": frozen_inputs["source_identity"],
        "frozen_inputs": frozen_inputs,
        "milestones": milestone_evidence,
        "requirements": requirements,
        "selected_candidate": l8["selected_candidate"],
        "selected_run_lineage": run_lineage,
        "candidate_metrics": l8["candidate_evidence"],
        "hardware_policy": {
            "local_gpu_role": "correctness_and_bounded_development_host",
            "local_vram_is_architecture_limit": False,
            "rented_gpu_allowed_after_target_environment_gate": True,
            "observed_device": run_lineage["environment"]["device_name"],
            "observed_total_vram_bytes": run_lineage["environment"][
                "total_vram_bytes"
            ],
        },
        "interpretation_boundary": {
            "legacy_dataset_is_legacy_16f_not_merged": True,
            "legacy_rare_support_generalizes_to_merged_data": False,
            "merged_data_has_materially_more_rare_behaviors": True,
            "merged_reviewed_reassessment_required": True,
            "architecture_family_finalized": False,
        },
        "unresolved_risks": [
            "legacy_16f labels remain unreviewed development evidence",
            "rare-class validation support is too small for transfer claims",
            "merged-reviewed model selection and human review remain pending",
            "parent reviewed all-source P0-P8 goal remains incomplete",
        ],
        "rollback": {
            "action": (
                "remove this completion audit and the L8 lock packet, then "
                "resume from the L7 decision"
            ),
            "l8_candidate_lock_sha256": config["milestones"]["L8"][
                "sha256"
            ],
            "l7_decision_sha256": config["milestones"]["L7"]["sha256"],
            "code_rollback_commit": git_guard["code_sha"],
        },
        "next_action": (
            "resume the parent classification_v2 goal and re-audit canonical "
            "reviewed all-source P0-P8 blockers"
        ),
        "errors": [],
        "valid": True,
    }
    audit["audit_payload_sha256"] = payload_sha256(audit)
    if write_output:
        output = _resolve_inside(root, config["output"]["path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(audit, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    return audit


def _validate_frozen_inputs(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = _bound_file(root, config["raw_authority"], "raw_authority")
    _require_equal(
        _count_csv_rows(raw),
        int(config["raw_authority"]["expected_rows"]),
        "raw authority rows",
    )
    snapshot_path = _bound_file(root, config["snapshot"], "snapshot")
    snapshot = _read_json(snapshot_path)
    _require_mapping(
        snapshot,
        {
            "status": "FROZEN_LEGACY_DEVELOPMENT_INPUTS_PRE_L3_GATE",
            "lineage_scope": LINEAGE_SCOPE,
            "human_review_complete": False,
            "q2_claim_allowed": False,
            "canonical_full_oof_authorized": False,
            "errors": [],
            "valid": True,
        },
        "input snapshot",
    )
    expected_frozen_views = {
        "legacy_t6_all_sliding_observed_time",
        "legacy_t8_all_sliding_observed_time",
        "legacy_t12_all_sliding_observed_time",
        "legacy_t16_all_sliding_observed_time",
        "legacy_t6_centered_matched_observed_time",
        "legacy_t8_centered_matched_observed_time",
        "legacy_t12_centered_matched_observed_time",
        "legacy_t16_centered_matched_observed_time",
    }
    _require_equal(
        set(snapshot["frozen_contract"]["temporal_views"]),
        expected_frozen_views,
        "frozen temporal views",
    )
    row_bound: dict[str, Any] = {}
    for name in ("lineage_manifest", "native_folds", "window_folds"):
        path = _bound_file(root, config[name], name)
        rows = _count_csv_rows(path)
        _require_equal(rows, int(config[name]["expected_rows"]), f"{name} rows")
        row_bound[name] = {
            **_artifact_record(path),
            "rows": rows,
        }
    feature_contract = _bound_file(
        root,
        config["feature_contract"],
        "feature_contract",
    )
    whitelist_path = _bound_file(
        root,
        config["feature_whitelist"],
        "feature_whitelist",
    )
    whitelist = _read_json(whitelist_path)
    _require_mapping(
        whitelist,
        {
            "lineage_scope": LINEAGE_SCOPE,
            "feature_dim": 512,
            "feature_dtype": "float32",
            "selection_policy": "explicit_512_cached_frame_features_only",
        },
        "candidate feature whitelist",
    )
    _require_equal(len(whitelist["features"]), 512, "whitelist feature count")
    _require_equal(
        set(whitelist["features"]) & set(whitelist["routing_only_fields"]),
        set(),
        "whitelist routing overlap",
    )
    packed = _resolve_inside(root, config["packed_actor_cache"]["path"])
    if not packed.is_file():
        raise FileNotFoundError(f"packed actor cache missing={packed}")
    _require_equal(
        int(packed.stat().st_size),
        int(config["packed_actor_cache"]["size_bytes"]),
        "packed actor cache size",
    )
    return {
        "source_identity": {
            "canonical_short_name": "legacy_16f",
            "source_type": "legacy_recovered",
            "dataset_id": "legacy_recovered_16f",
            "raw_authority": _artifact_record(raw),
            "raw_rows": int(config["raw_authority"]["expected_rows"]),
            "merged_dataset": False,
        },
        "snapshot": _artifact_record(snapshot_path),
        **row_bound,
        "feature_contract": _artifact_record(feature_contract),
        "feature_whitelist": {
            **_artifact_record(whitelist_path),
            "feature_count": len(whitelist["features"]),
        },
        "packed_actor_cache": {
            "path": str(packed),
            "size_bytes": int(packed.stat().st_size),
            "sha256": config["packed_actor_cache"]["sha256"],
            "hash_verification": "BOUND_THROUGH_L3_ALL_ARTIFACT_AUDIT",
            "verification_audit_sha256": config["packed_actor_cache"][
                "verification_audit_sha256"
            ],
        },
    }


def _load_milestones(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name in ("L1", "L2", "L3", "L4_short", "L4", "L7", "L8"):
        loaded[name] = _bound_json(root, config[name], name)
    l5 = config["L5"]
    loaded["L5"] = {
        "foundation": [
            (spec, _bound_json(root, spec, f"L5 foundation {index}"))
            for index, spec in enumerate(l5["foundation"])
        ],
        "visual_runs": [
            (spec, _bound_json(root, spec, f"L5 visual run {index}"))
            for index, spec in enumerate(l5["visual_runs"])
        ],
        "tcn_run": (
            l5["tcn_run"],
            _bound_json(root, l5["tcn_run"], "L5 TCN run"),
        ),
        "t1_decision": _bound_json(
            root,
            l5["t1_decision"],
            "L5 T1 decision",
        ),
        "temporal_decision": _bound_json(
            root,
            l5["temporal_decision"],
            "L5 temporal decision",
        ),
    }
    loaded["L6"] = [
        (spec, _bound_json(root, spec, f"L6 decision {index}"))
        for index, spec in enumerate(config["L6"]["decisions"])
    ]
    return loaded


def _validate_l1(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(payload, "PASS_LEGACY_DEVELOPMENT_L1", "L1")
    relational = payload["relational"]
    _require_mapping(
        relational,
        {
            "image_frame_rows": 496,
            "image_window_rows": 310,
            "cache_manifest_rows": 496,
            "native_fold_rows": 31,
            "fold_count": 3,
            "missing_cache_slot_rows": 0,
            "missing_window_fold_rows": 0,
            "fold_inheritance_mismatches": 0,
        },
        "L1 relational",
    )
    _require_mapping(
        payload["loader_audit"]["image_load_audit"],
        {
            "packed_image_cache_hits": 2728,
            "disk_image_cache_misses": 0,
            "source_image_loads": 0,
        },
        "L1 cache loader",
    )
    _require_mapping(
        payload["tensor_audit"],
        {
            "tensor_shape": [496, 160, 160, 3],
            "tensor_dtype": "uint8",
            "packed_pixel_mismatches": 0,
            "errors": [],
            "valid": True,
        },
        "L1 tensor",
    )
    _require_equal(payload["repeat_hash_audit"]["valid"], True, "L1 repeat")
    return {
        "status": payload["status"],
        "frame_rows": 496,
        "native_units": 31,
        "window_rows": 310,
        "fold_count": 3,
        "source_media_reads": 0,
        "repeat_byte_identical": True,
    }


def _validate_l2(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(payload, "PASS_LEGACY_DEVELOPMENT_L2", "L2")
    expected = {
        "frame_rows": 72_864,
        "native_units": 4_554,
        "all_sliding_rows": 45_540,
        "centered_matched_rows": 18_216,
        "all_sliding_by_length": {
            "6": 18_216,
            "8": 13_662,
            "12": 9_108,
            "16": 4_554,
        },
    }
    _require_equal(payload["expected_counts"], expected, "L2 counts")
    for name in ("primary", "repeat", "repeat_hash_audit"):
        _require_equal(payload[name]["valid"], True, f"L2 {name}")
    return {
        "status": payload["status"],
        "frame_rows": 72_864,
        "native_units": 4_554,
        "sliding_windows": 45_540,
        "centered_windows": 18_216,
        "temporal_lengths": [6, 8, 12, 16],
        "repeat_byte_identical": True,
    }


def _validate_l3(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(payload, "PASS_LEGACY_DEVELOPMENT_L3", "L3")
    _require_mapping(
        payload["relational"],
        {
            "image_frame_rows": 72_864,
            "image_window_rows": 45_540,
            "native_fold_rows": 4_554,
            "fold_count": 12,
            "missing_cache_slot_rows": 0,
            "missing_window_fold_rows": 0,
            "fold_inheritance_mismatches": 0,
        },
        "L3 relational",
    )
    _require_mapping(
        payload["packed_pixel_and_loader_audit"],
        {
            "all_pixel_checked_rows": 72_864,
            "packed_pixel_mismatches": 0,
            "source_media_fallback_reads": 0,
            "tensor_shape": [72_864, 160, 160, 3],
            "tensor_dtype": "uint8",
            "errors": [],
            "valid": True,
        },
        "L3 packed cache",
    )
    feature = payload["feature_contract_audit"]
    _require_mapping(
        feature,
        {
            "unblocked_forbidden_probe_columns": [],
            "errors": [],
            "valid": True,
        },
        "L3 feature contract",
    )
    for name in (
        "repeat_input_hash_audit",
        "shortcut_audit",
        "snapshot_audit",
        "artifact_manifest_audit",
    ):
        _require_equal(payload[name]["valid"], True, f"L3 {name}")
    packed_record = payload["artifact_manifest_audit"][
        "verified_artifacts"
    ]["packed_tensor"]
    _require_mapping(
        packed_record,
        {
            "exists": True,
            "sha256_match": True,
            "size_match": True,
            "tensor_shape_match": True,
            "tensor_dtype_match": True,
        },
        "L3 packed artifact",
    )
    return {
        "status": payload["status"],
        "fold_count": 12,
        "packed_tensor_sha256": payload["packed_tensor_sha256"],
        "predictive_whitelist_sha256": feature[
            "predictive_whitelist_sha256"
        ],
        "blacklist_sha256": feature["blacklist_sha256"],
        "source_media_reads": 0,
        "all_artifacts_hash_verified": True,
    }


def _validate_l4(
    short: dict[str, Any],
    full: dict[str, Any],
) -> dict[str, Any]:
    _validate_common(
        short,
        "PASS_LEGACY_DEVELOPMENT_L4_SHORT",
        "L4 short",
    )
    _validate_common(full, "PASS_LEGACY_DEVELOPMENT_L4", "L4")
    for name in (
        "input_contract_audit",
        "one_batch_gradient_audit",
        "deterministic_repeat_audit",
        "checkpoint_resume_audit",
        "tiny_overfit_audit",
        "cache_only_audit",
        "memory_audit",
    ):
        _require_equal(short[name]["valid"], True, f"L4 short {name}")
    _require_equal(
        short["tiny_overfit_audit"]["memorization_accuracy"],
        1.0,
        "L4 tiny overfit",
    )
    for name in (
        "cache_only_audit",
        "memory_audit",
        "lineage_audit",
        "one_fold_one_epoch_audit",
        "optimizer_support_audit",
    ):
        _require_equal(full[name]["valid"], True, f"L4 {name}")
    _require_mapping(
        full,
        {
            "held_out_predictions_computed": False,
            "held_out_accuracy_f1_computed": False,
            "l5_controlled_baselines_authorized": True,
        },
        "L4 boundary",
    )
    return {
        "status": full["status"],
        "gradient_groups": sorted(
            short["one_batch_gradient_audit"]["gradient_groups"][
                "groups"
            ]
        ),
        "deterministic_repeat": True,
        "checkpoint_resume_equivalent": True,
        "tiny_overfit_accuracy": 1.0,
        "one_epoch_native_units": full["one_fold_one_epoch_audit"][
            "unique_rows_seen"
        ],
        "source_media_reads": 0,
        "peak_vram_bytes": full["memory_audit"]["peak_vram_bytes"],
    }


def _validate_l5(payloads: dict[str, Any]) -> dict[str, Any]:
    foundation: list[dict[str, Any]] = []
    for spec, payload in payloads["foundation"]:
        _validate_common(
            payload,
            spec["expected_status"],
            spec["name"],
        )
        foundation.append(
            {"name": spec["name"], "status": payload["status"]}
        )
    runs: list[dict[str, Any]] = []
    for spec, payload in payloads["visual_runs"]:
        runs.append(_validate_l5_run(spec, payload))
    tcn_spec, tcn_payload = payloads["tcn_run"]
    tcn = _validate_l5_run(tcn_spec, tcn_payload)
    t1 = payloads["t1_decision"]
    _validate_common(
        t1,
        "PASS_LEGACY_DEVELOPMENT_L5_PAIRED_DECISION",
        "L5 T1 decision",
    )
    _require_equal(
        t1["decision"]["decision"],
        "RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH",
        "L5 T1 decision value",
    )
    temporal = payloads["temporal_decision"]
    _validate_common(
        temporal,
        "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_DECISION",
        "L5 temporal decision",
    )
    _require_equal(set(temporal["packets"]), TEMPORAL_VIEWS, "L5 views")
    _require_mapping(
        temporal["decision"],
        {
            "selected_working_view": "t6_sliding",
            "working_baseline_retained": True,
            "causal_temporal_length_claim_allowed": False,
            "architecture_family_finalized": False,
            "applies_to_merged_reviewed_data": False,
            "merged_reviewed_reassessment_required": True,
        },
        "L5 temporal selection",
    )
    return {
        "status": "PASS_LEGACY_DEVELOPMENT_L5",
        "foundation_gates": foundation,
        "visual_controls": runs,
        "masked_tcn_control": tcn,
        "temporal_views": sorted(TEMPORAL_VIEWS),
        "selected_working_view": "t6_sliding",
        "causal_temporal_length_claim_allowed": False,
    }


def _validate_l5_run(
    spec: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_common(payload, spec["expected_status"], spec["name"])
    _require_mapping(
        payload,
        {
            "run_id": spec["expected_run_id"],
            "training_scope": "full_development_baseline",
            "train_native_units": 3_652,
            "validation_native_units": 245,
            "source_media_reads": 0,
            "outer_holdout_rows_loaded": 0,
            "outer_holdout_predictions_created": 0,
        },
        spec["name"],
    )
    _require_mapping(
        payload["execution"],
        {
            "oom": False,
            "oom_retry_count": 0,
            "post_cleanup_allocated_bytes": 0,
            "post_cleanup_reserved_bytes": 0,
            "errors": [],
            "valid": True,
        },
        f"{spec['name']} execution",
    )
    return {
        "name": spec["name"],
        "run_id": payload["run_id"],
        "macro_f1": payload["validation_metrics"][
            "macro_f1_global_10_class"
        ],
        "optimizer_steps": payload["optimizer_steps"],
        "peak_reserved_bytes": payload["execution"]["peak_reserved_bytes"],
    }


def _validate_l6(
    payloads: list[tuple[dict[str, Any], dict[str, Any]]],
    top_k: dict[str, Any],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for spec, payload in payloads:
        _validate_common(
            payload,
            spec["expected_status"],
            spec["name"],
        )
        _require_equal(
            payload["decision"]["decision"],
            spec["expected_decision"],
            f"{spec['name']} decision",
        )
        _require_mapping(
            payload["decision"],
            {
                "applies_to_merged_reviewed_data": False,
                "merged_reviewed_reassessment_required": True,
            },
            f"{spec['name']} boundary",
        )
        decisions.append(
            {
                "name": spec["name"],
                "decision": spec["expected_decision"],
            }
        )
    _require_equal(
        top_k,
        {
            "status": "DEFERRED_NOT_AUTHORIZED",
            "reason": "numeric_social_prerequisite_did_not_pass",
            "values_enter_candidate_x": False,
            "applies_to_merged_reviewed_data": False,
            "reassess_on_merged_reviewed_data": True,
        },
        "L6 top-K disposition",
    )
    return {
        "status": "PASS_LEGACY_DEVELOPMENT_L6",
        "decisions": decisions,
        "top_k_partner": top_k,
        "selected_base": "parameter_matched_t6_actor_only_zero",
    }


def _validate_l7(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(
        payload,
        "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_DECISION",
        "L7",
    )
    _require_mapping(
        payload["decision"],
        {
            "decision": "RETAIN_EVENT_BALANCED_CE_REJECT_L7_ALTERNATIVES",
            "selected_loss_policy": "event_balanced_ce",
            "full_confirmation_authorized": False,
            "l8_candidate_lock_authorized": True,
            "architecture_family_finalized": False,
            "applies_to_merged_reviewed_data": False,
            "merged_reviewed_reassessment_required": True,
        },
        "L7 decision",
    )
    return {
        "status": payload["status"],
        "policies": [
            "event_balanced_ce",
            "effective_number_ce",
            "balanced_softmax",
        ],
        "selected_loss_policy": "event_balanced_ce",
        "alternative_full_confirmation_authorized": False,
    }


def _validate_l8(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(
        payload,
        "PASS_LEGACY_DEVELOPMENT_L8_CANDIDATE_LOCK",
        "L8",
    )
    _require_mapping(
        payload,
        {
            "outer_holdout_rows_loaded": 0,
            "outer_holdout_predictions_created": 0,
        },
        "L8 holdout boundary",
    )
    finalist = payload["selected_candidate"]
    _require_mapping(
        finalist,
        {
            "candidate_id": "legacy_16f_t6_sliding_event_balanced_v1",
            "canonical_source_name": "legacy_16f",
            "view_id": "t6_sliding",
            "loss_policy": "event_balanced_ce",
            "candidate_locked": True,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
            "valid": True,
        },
        "L8 finalist",
    )
    for name, spec in payload["registry_artifacts"].items():
        _verify_absolute_artifact(spec, f"L8 registry {name}")
    for name in ("checkpoint", "validation_native_predictions", "validation_metrics"):
        _verify_absolute_artifact(finalist[name], f"L8 finalist {name}")
    matrix_path = Path(payload["registry_artifacts"]["experiment_matrix"]["path"])
    with matrix_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require_equal(len(rows), 11, "L8 experiment matrix rows")
    for index, row in enumerate(rows):
        path = Path(row["decision_artifact_path"])
        if not path.is_file():
            raise FileNotFoundError(f"L8 decision artifact missing={path}")
        _require_equal(
            file_sha256(path),
            row["decision_artifact_sha256"],
            f"L8 decision artifact {index}",
        )
    groups = payload["candidate_evidence"]["groups"]
    _require_equal(
        set(groups),
        {"rare", "interaction", "feeding", "posture", "locomotion_exploration"},
        "L8 evidence groups",
    )
    _require_equal(
        payload["candidate_evidence"]["recording"]["video_count"],
        33,
        "L8 video count",
    )
    _require_equal(
        set(payload["candidate_evidence"]["per_class"]),
        set(VALID_BEHAVIORS),
        "L8 per-class labels",
    )
    return {
        "status": payload["status"],
        "candidate_id": finalist["candidate_id"],
        "macro_f1": payload["candidate_evidence"]["global"][
            "macro_f1_global_10_class"
        ],
        "accuracy": payload["candidate_evidence"]["global"]["accuracy"],
        "nll": payload["candidate_evidence"]["global"]["nll"],
        "video_clusters": 33,
        "registry_artifact_count": len(payload["registry_artifacts"]),
        "experiment_rows": len(rows),
        "all_path_hash_links_verified": True,
    }


def _validate_selected_run_lineage(
    root: Path,
    l8: dict[str, Any],
    frozen_inputs: dict[str, Any],
) -> dict[str, Any]:
    finalist = l8["selected_candidate"]
    result_path = Path(l8["full_candidate_run_audit"]["result_path"])
    _require_equal(
        file_sha256(result_path),
        l8["full_candidate_run_audit"]["result_sha256"],
        "selected run result hash",
    )
    result = _read_json(result_path)
    run_root = result_path.parent
    run_manifest = run_root / "run_manifest.json"
    artifact_manifest = run_root / "artifact_manifest.json"
    _require_equal(
        file_sha256(run_manifest),
        finalist["run_manifest_sha256"],
        "selected run manifest hash",
    )
    _require_equal(
        file_sha256(artifact_manifest),
        finalist["artifact_manifest_sha256"],
        "selected artifact manifest hash",
    )
    run = _read_json(run_manifest)
    artifacts = _read_json(artifact_manifest)
    records = {
        record["name"]: record for record in artifacts["artifacts"]
    }
    required_artifacts = {
        "environment",
        "checkpoint",
        "checkpoint_manifest",
        "prediction_manifest",
        "native_predictions",
        "validation_metrics",
        "run_result",
    }
    _require_equal(
        required_artifacts - set(records),
        set(),
        "selected required run artifacts",
    )
    for name in sorted(required_artifacts):
        _verify_absolute_artifact(records[name], f"selected run {name}")
    _require_mapping(
        run,
        {
            "run_id": "tlf_t6s_f_fda8f43_v1",
            "view_id": "t6_sliding",
            "status": "completed",
            "lineage_scope": LINEAGE_SCOPE,
            "source_media_reads": 0,
            "outer_predictions_created": 0,
        },
        "selected run manifest",
    )
    execution = result["execution"]
    _require_mapping(
        execution,
        {
            "precision": "float32",
            "autocast_enabled": False,
            "oom": False,
            "oom_retry_count": 0,
            "post_cleanup_allocated_bytes": 0,
            "post_cleanup_reserved_bytes": 0,
            "errors": [],
            "valid": True,
        },
        "selected run environment",
    )
    full_config_path = _resolve_inside(
        root,
        finalist["full_training_config"]["path"],
    )
    _require_equal(
        file_sha256(full_config_path),
        finalist["full_training_config"]["sha256"],
        "selected full config hash",
    )
    full_config = _read_json(full_config_path)
    return {
        "execution_mode": "local_bounded_development",
        "run_id": run["run_id"],
        "fold": "native_oof_006_validation",
        "seed": 20260714,
        "config_hash": result["config_sha256"],
        "dataset_snapshot_hash": frozen_inputs["snapshot"]["sha256"],
        "cache_hash": full_config["feature_parent"]["feature_tensor_sha256"],
        "fold_manifest_hash": frozen_inputs["native_folds"]["sha256"],
        "feature_whitelist_hash": frozen_inputs["feature_whitelist"][
            "sha256"
        ],
        "run_manifest": _artifact_record(run_manifest),
        "artifact_manifest": _artifact_record(artifact_manifest),
        "checkpoint": finalist["checkpoint"],
        "predictions": finalist["validation_native_predictions"],
        "metrics": finalist["validation_metrics"],
        "environment_artifact": {
            "path": records["environment"]["path"],
            "sha256": records["environment"]["sha256"],
            "size_bytes": records["environment"]["size_bytes"],
        },
        "checkpoint_manifest": {
            "path": records["checkpoint_manifest"]["path"],
            "sha256": records["checkpoint_manifest"]["sha256"],
            "size_bytes": records["checkpoint_manifest"]["size_bytes"],
        },
        "prediction_manifest": {
            "path": records["prediction_manifest"]["path"],
            "sha256": records["prediction_manifest"]["sha256"],
            "size_bytes": records["prediction_manifest"]["size_bytes"],
        },
        "fold_outputs_isolated": True,
        "outer_holdout_predictions_created": 0,
        "environment": {
            "device": execution["device"],
            "device_name": execution["device_name"],
            "total_vram_bytes": execution["actual_total_vram_bytes"],
            "precision": execution["precision"],
            "peak_reserved_bytes": execution["peak_reserved_bytes"],
            "oom_retry_count": execution["oom_retry_count"],
        },
    }


def _forbidden_alias_audit(root: Path) -> dict[str, Any]:
    pattern = re.compile(r"\b" + "11" + "6f" + r"\b", re.IGNORECASE)
    extensions = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
    occurrences: list[str] = []
    for relative in _git(root, "ls-files").splitlines():
        path = _resolve_inside(root, relative)
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            occurrences.append(relative.replace("\\", "/"))
    if occurrences:
        raise ValueError(f"forbidden source alias files={occurrences}")
    return {"occurrences": [], "errors": [], "valid": True}


def _validate_common(
    payload: dict[str, Any],
    status: str,
    name: str,
) -> None:
    expected = {
        "status": status,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    _require_mapping(payload, expected, name)
    if "reviewed_or_final_claim_allowed" in payload:
        _require_equal(
            payload["reviewed_or_final_claim_allowed"],
            False,
            f"{name}.reviewed_or_final_claim_allowed",
        )
    if "outer_holdout_predictions_authorized" in payload:
        _require_equal(
            payload["outer_holdout_predictions_authorized"],
            False,
            f"{name}.outer_holdout_predictions_authorized",
        )


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "audit_id",
        *CLAIM_BOUNDARY,
        "implementation_source",
        "goal_authority",
        "frozen_inputs",
        "milestones",
        "execution_guard",
        "output",
    }
    _require_equal(set(config), required, "completion config keys")
    _require_equal(config["schema_version"], CONFIG_SCHEMA, "config schema")
    _require_mapping(config, CLAIM_BOUNDARY, "completion config boundary")
    for name in ("implementation_source", "goal_authority"):
        _validate_spec(config[name], name)
    frozen = config["frozen_inputs"]
    _require_equal(
        set(frozen),
        {
            "raw_authority",
            "snapshot",
            "lineage_manifest",
            "native_folds",
            "window_folds",
            "feature_contract",
            "feature_whitelist",
            "packed_actor_cache",
        },
        "frozen input keys",
    )
    for name in (
        "raw_authority",
        "snapshot",
        "lineage_manifest",
        "native_folds",
        "window_folds",
        "feature_contract",
        "feature_whitelist",
    ):
        _validate_spec(frozen[name], name)
    milestones = config["milestones"]
    _require_equal(
        set(milestones),
        {"L1", "L2", "L3", "L4_short", "L4", "L5", "L6", "L7", "L8"},
        "milestone keys",
    )
    for name in ("L1", "L2", "L3", "L4_short", "L4", "L7", "L8"):
        _validate_spec(milestones[name], name)
    for group in (milestones["L5"]["foundation"], milestones["L5"]["visual_runs"]):
        for index, spec in enumerate(group):
            _validate_spec(spec, f"L5 spec {index}")
    for name in ("tcn_run", "t1_decision", "temporal_decision"):
        _validate_spec(milestones["L5"][name], f"L5 {name}")
    for index, spec in enumerate(milestones["L6"]["decisions"]):
        _validate_spec(spec, f"L6 decision {index}")
    guard = milestones.get("execution_guard", config["execution_guard"])
    _require_equal(
        set(config["execution_guard"]),
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution guard keys",
    )
    if not isinstance(guard, dict):
        raise ValueError("execution guard must be an object")
    _require_equal(set(config["output"]), {"path"}, "output keys")


def _validate_spec(spec: object, name: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be an object")
    if not {"path", "sha256"}.issubset(spec):
        raise ValueError(f"{name} missing path or sha256")
    value = str(spec["sha256"])
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} invalid SHA-256")


def _git_guard(root: Path, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("execution_guard must be an object")
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    observed = sorted(_status_path(line) for line in lines if line.strip())
    allowed = sorted(str(item).replace("\\", "/") for item in value["allowed_dirty_paths"])
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(item).replace("\\", "/")
        for item in value["required_tracked_paths"]
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
    if unexpected or untracked:
        raise ValueError(
            f"completion Git guard unexpected={unexpected},untracked={untracked}"
        )
    return {
        "status": "PASS_COMMITTED_INPUT_GUARD",
        "code_sha": _git(root, "rev-parse", "HEAD").strip(),
        "dirty_entries": lines,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": [],
        "required_tracked_paths": required,
        "untracked_required_paths": [],
        "errors": [],
        "valid": True,
    }


def _bound_json(root: Path, spec: object, name: str) -> dict[str, Any]:
    return _read_json(_bound_file(root, spec, name))


def _bound_file(root: Path, spec: object, name: str) -> Path:
    _validate_spec(spec, name)
    assert isinstance(spec, dict)
    path = _resolve_inside(root, spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing={path}")
    _require_equal(file_sha256(path), spec["sha256"], f"{name} hash")
    return path


def _verify_absolute_artifact(spec: object, name: str) -> None:
    _validate_spec(spec, name)
    assert isinstance(spec, dict)
    path = Path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing={path}")
    _require_equal(file_sha256(path), spec["sha256"], f"{name} hash")
    if "size_bytes" in spec:
        _require_equal(path.stat().st_size, spec["size_bytes"], f"{name} size")


def _requirement(
    milestone: str,
    passed: bool,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "milestone": milestone,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "evidence": evidence,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
    return max(0, line_count - 1)


def _resolve_inside(root: Path, value: object) -> Path:
    path = (root / str(value)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root={value}") from error
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object={path}")
    return payload


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


__all__ = [
    "AUDIT_SCHEMA",
    "CONFIG_SCHEMA",
    "write_legacy_goal_completion_audit",
]
