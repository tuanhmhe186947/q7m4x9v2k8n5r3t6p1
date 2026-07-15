"""Paired native/video-cluster decision for the legacy L6 geometry short gate."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6GeometryConfig,
    load_geometry_training_config,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_runtime import (
    MATRIX_GATE_SCHEMA,
    REPEAT_GATE_SCHEMA,
    audit_geometry_run,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_decision_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_decision.v1"
)
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33


def evaluate_geometry_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit exact run packets and decide whether geometry may expand."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _resolve_inside(root, config["implementation"]["path"])
    _validate_bound_file(
        implementation,
        config["implementation"]["sha256"],
        "decision implementation",
    )
    training_config_path = _resolve_inside(
        root,
        config["short_training_config"]["path"],
    )
    _validate_bound_file(
        training_config_path,
        config["short_training_config"]["sha256"],
        "short training config",
    )
    training_config = load_geometry_training_config(training_config_path)
    if training_config.training_scope != SHORT_SCOPE:
        raise ValueError("L6 geometry decision requires short training scope")
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
        "geometry_vs_parameter_matched_zero": _compare_packets(
            packets["geometry"],
            packets["parameter_matched_zero"],
            contract=paired_contract,
        ),
        "geometry_vs_availability_only": _compare_packets(
            packets["geometry"],
            packets["availability_only"],
            contract=paired_contract,
        ),
        "availability_only_vs_parameter_matched_zero": _compare_packets(
            packets["availability_only"],
            packets["parameter_matched_zero"],
            contract=paired_contract,
        ),
    }
    decision = make_geometry_decision(
        comparisons,
        contract=_object(config["decision_contract"], "decision_contract"),
    )
    git_guard = _git_guard(
        root,
        _object(config["execution_guard"], "execution_guard"),
    )
    errors = [*git_guard["errors"]]
    if not decision["full_geometry_expansion_authorized"]:
        errors.append("paired_geometry_promotion_criteria_not_all_met")
    valid = not errors
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_DECISION"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_DECISION"
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
        "errors": errors,
        "valid": valid,
    }


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    config = _read_json(config_path.resolve())
    _validate_config(config)
    return _resolve_inside(project_root.resolve(), config["output_path"])


def write_geometry_short_decision(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = evaluate_geometry_short_decision(
        config_path,
        project_root=project_root,
    )
    output = configured_output_path(config_path, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
    return output, payload


def _load_matrix(
    root: Path,
    value: object,
    training_config: LegacyL6GeometryConfig,
) -> dict[str, Any]:
    spec = _object(value, "short_matrix_gate")
    path = _resolve_inside(root, spec["path"])
    _validate_bound_file(path, spec["sha256"], "short matrix gate")
    matrix = _read_json(path)
    expected = {
        "schema_version": MATRIX_GATE_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_MATRIX",
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
    training_config: LegacyL6GeometryConfig,
    *,
    mode: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    result_path = _resolve_inside(root, spec["result_path"])
    _validate_bound_file(result_path, spec["result_sha256"], f"{mode} result")
    audit = audit_geometry_run(training_config, result_path=result_path)
    if not audit["valid"]:
        raise ValueError(f"L6 geometry run audit failed mode={mode}")
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
            "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_REPEAT",
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
    for mode in MODES:
        frame = packets[mode]["predictions"]
        required = {
            "temporal_unit_key",
            "video_key",
            "behavior_label",
            "predicted_label",
            "geometry_mode",
            "missing_modality",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{mode} native predictions missing={missing}")
        if len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{mode} native rows={len(frame)}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{mode} duplicate native units")
        if set(frame["geometry_mode"].astype(str)) != {mode}:
            raise ValueError(f"{mode} prediction mode drift")
        if frame["missing_modality"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{mode} main predictions marked missing")
        metadata = frame[
            ["temporal_unit_key", "video_key", "behavior_label"]
        ].astype(str).sort_values("temporal_unit_key", kind="mergesort")
        metadata = metadata.reset_index(drop=True)
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"paired native universe differs mode={mode}")
    assert reference is not None
    clusters = int(reference["video_key"].nunique())
    _require_equal(clusters, EXPECTED_VIDEO_CLUSTERS, "video clusters")
    return {
        "native_units": len(reference),
        "video_clusters": clusters,
        "modes": list(MODES),
        "exact_metadata_equality": True,
        "outer_holdout_rows": 0,
    }


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
        cluster_col="video_key",
        unit_col="temporal_unit_key",
        fold_col="development_validation_fold_id",
        true_col="true_label",
        pred_col="native_predicted_behavior",
        iterations=int(contract["bootstrap_iterations"]),
        seed=int(contract["bootstrap_seed"]),
        outer_predictions_used_for_model_selection=False,
    )
    candidate_metrics = candidate["result"]["validation_metrics"]
    baseline_metrics = baseline["result"]["validation_metrics"]
    return {
        "candidate_mode": candidate["mode"],
        "baseline_mode": baseline["mode"],
        "paired_native_units": len(left),
        "paired_video_clusters": int(left["video_key"].nunique()),
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
        "confusion_groups": _compare_confusion_groups(candidate, baseline),
        "paired_outcomes": _paired_outcomes(left, right),
        "runtime": {
            "candidate_seconds": candidate["result"]["runtime_seconds"],
            "baseline_seconds": baseline["result"]["runtime_seconds"],
            "candidate_peak_reserved_bytes": candidate["result"]["execution"][
                "peak_reserved_bytes"
            ],
            "baseline_peak_reserved_bytes": baseline["result"]["execution"][
                "peak_reserved_bytes"
            ],
        },
    }


def make_geometry_decision(
    comparisons: dict[str, dict[str, Any]],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    zero = comparisons["geometry_vs_parameter_matched_zero"]
    availability = comparisons["geometry_vs_availability_only"]
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
    criteria = {
        "geometry_gain_vs_zero_meets_margin": _macro_delta(zero) >= minimum_gain,
        "geometry_gain_vs_availability_meets_margin": (
            _macro_delta(availability) >= minimum_gain
        ),
        "geometry_vs_zero_cluster_ci_low_positive": _ci_low(zero) > 0.0,
        "geometry_vs_availability_cluster_ci_low_positive": (
            _ci_low(availability) > 0.0
        ),
        "geometry_nll_improves_vs_zero": (
            zero["delta_candidate_minus_baseline"]["nll"] < 0.0
        ),
        "availability_only_is_bounded_diagnostic": (
            abs(_macro_delta(diagnostic)) <= maximum_availability
        ),
        "rare_group_drop_within_limit": rare_delta >= -maximum_rare_drop,
        "all_packets_cleanup_zero": True,
        "all_modes_parameter_matched": True,
    }
    authorized = all(criteria.values())
    return {
        "decision": (
            "RETAIN_GEOMETRY_FOR_FULL_LEGACY_DEVELOPMENT"
            if authorized
            else "DO_NOT_EXPAND_GEOMETRY_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "criteria": criteria,
        "thresholds": copy.deepcopy(contract),
        "full_geometry_expansion_authorized": authorized,
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "availability_only_is_behavior_evidence": False,
        "source_probe_status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
        "next_action": (
            "prepare_hash_bound_full_geometry_config"
            if authorized
            else "retain_zero_control_and_stop_geometry_expansion"
        ),
    }


def _compare_confusion_groups(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    left = candidate["confusion_groups"].set_index("confusion_group")
    right = baseline["confusion_groups"].set_index("confusion_group")
    _require_equal(set(left.index), set(right.index), "confusion group set")
    rows: dict[str, Any] = {}
    for group in sorted(left.index.astype(str)):
        _require_equal(
            int(left.loc[group, "support"]),
            int(right.loc[group, "support"]),
            f"confusion support {group}",
        )
        rows[group] = {
            "support": int(left.loc[group, "support"]),
            "candidate_macro_f1": float(left.loc[group, "macro_f1"]),
            "baseline_macro_f1": float(right.loc[group, "macro_f1"]),
            "macro_f1_delta": float(
                left.loc[group, "macro_f1"] - right.loc[group, "macro_f1"]
            ),
            "candidate_accuracy": float(left.loc[group, "accuracy"]),
            "baseline_accuracy": float(right.loc[group, "accuracy"]),
        }
    return rows


def _paired_outcomes(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, int]:
    target = candidate["behavior_label"].astype(str)
    candidate_correct = candidate["predicted_label"].astype(str).eq(target)
    baseline_correct = baseline["predicted_label"].astype(str).eq(target)
    return {
        "both_correct": int((candidate_correct & baseline_correct).sum()),
        "candidate_only_correct": int(
            (candidate_correct & ~baseline_correct).sum()
        ),
        "baseline_only_correct": int(
            (~candidate_correct & baseline_correct).sum()
        ),
        "both_wrong": int((~candidate_correct & ~baseline_correct).sum()),
    }


def _ordered_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("temporal_unit_key", kind="mergesort").reset_index(
        drop=True
    )


def _bootstrap_frame(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": frame["temporal_unit_key"].astype(str),
            "video_key": frame["video_key"].astype(str),
            "development_validation_fold_id": str(
                contract["validation_fold_id"]
            ),
            "true_label": frame["behavior_label"].astype(str),
            "native_predicted_behavior": frame["predicted_label"].astype(str),
        }
    )


def _global_metrics(value: object) -> dict[str, float]:
    metrics = _object(value, "validation metrics")
    return {
        "macro_f1_global_10_class": float(
            metrics["macro_f1_global_10_class"]
        ),
        "accuracy": float(metrics["accuracy"]),
        "nll": float(metrics["nll"]),
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
        "repeat_gate_path": packet["repeat_gate_path"],
        "repeat_gate_sha256": packet["repeat_gate_sha256"],
        "validation_metrics": _global_metrics(result["validation_metrics"]),
        "missing_validation_metrics": _global_metrics(
            result["missing_validation_metrics"]
        ),
        "parameter_count": 69_404,
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
    _require_exact_keys(config, required, "decision config")
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
                _require_sha(spec[field], f"runs.{mode}.{field}")
    paired = {
        "unit_column": "temporal_unit_key",
        "cluster_column": "video_key",
        "validation_fold_id": "legacy_l6_short_validation_v1",
        "expected_native_units": EXPECTED_NATIVE_UNITS,
        "expected_clusters": EXPECTED_VIDEO_CLUSTERS,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260715,
    }
    _require_equal(
        _object(config["paired_contract"], "paired_contract"),
        paired,
        "paired contract",
    )
    decision = {
        "minimum_macro_f1_gain": 0.02,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
    }
    _require_equal(
        _object(config["decision_contract"], "decision_contract"),
        decision,
        "decision contract",
    )
    boundary = {
        "legacy_only_decision": True,
        "architecture_family_finalized": False,
        "applies_to_merged_reviewed_data": False,
        "merged_reviewed_reassessment_required": True,
        "availability_only_is_behavior_evidence": False,
    }
    _require_equal(
        _object(config["interpretation_boundary"], "interpretation_boundary"),
        boundary,
        "interpretation boundary",
    )
    guard = _object(config["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )


def _git_guard(root: Path, guard: dict[str, Any]) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    entries = [line for line in status.splitlines() if line.strip()]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(str(value).replace("\\", "/") for value in guard[
        "allowed_dirty_paths"
    ])
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(value).replace("\\", "/")
        for value in guard["required_tracked_paths"]
    ]
    untracked: list[str] = []
    for path in required:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            untracked.append(path)
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    return {
        "code_sha": _git(root, "rev-parse", "HEAD").strip(),
        "observed_dirty_paths": observed,
        "allowed_dirty_paths": allowed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def _macro_delta(comparison: dict[str, Any]) -> float:
    return float(
        comparison["delta_candidate_minus_baseline"][
            "macro_f1_global_10_class"
        ]
    )


def _ci_low(comparison: dict[str, Any]) -> float:
    return float(comparison["video_cluster_bootstrap"]["ci_low"])


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _require_sha(spec["sha256"], f"{name}.sha256")


def _validate_bound_file(path: Path, expected_sha: object, name: str) -> None:
    _require_sha(expected_sha, f"{name}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    observed = file_sha256(path)
    if observed != str(expected_sha):
        raise ValueError(f"{name} SHA256={observed}!={expected_sha}")


def _resolve_inside(root: Path, value: object) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / str(value)).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"decision path escapes project root={value}") from error
    return path


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"decision git command failed: {message}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", maxsplit=1)[1]
    return value.strip('"')


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid decision JSON={path}") from error
    return _object(payload, str(path))


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
            f"{name} keys missing={sorted(expected - observed)} "
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
        raise ValueError(f"{name} drift observed={observed!r} expected={expected!r}")


def _require_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} is not lowercase SHA256")
