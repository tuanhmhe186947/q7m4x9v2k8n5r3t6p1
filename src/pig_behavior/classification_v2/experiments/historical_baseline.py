"""Fail-closed registration for scientifically invalid historical baselines.

Historical artifacts can retain compute, checkpoint, and orchestration value
without becoming model-quality controls. This module reproduces known ordered
window defects, hashes reusable artifacts, and makes every promotion flag
explicitly false.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    audit_ordered_window_ids,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

SCHEMA_VERSION = "classification_v2_historical_baseline_reconciliation_v1"
FULL_CONTROL_STATUS = "HISTORICAL_ONLY"
LEGACY_CONTROL_STATUS = "HISTORICAL_ARCHITECTURE_ONLY"
DISABLED_CLAIM_FLAGS = {
    "performance_claim_allowed": False,
    "paired_comparison_allowed": False,
    "promotion_allowed": False,
    "model_selection_allowed": False,
    "paper_claim_allowed": False,
}


@dataclass(frozen=True, slots=True)
class HistoricalFullOOFConfig:
    """Inputs that bind the invalid historical full-OOF artifact lineage."""

    split_manifest_csv: Path
    image_manifest_csv: Path
    interaction_manifest_csv: Path
    run_audit_json: Path
    metrics_json: Path
    prediction_schema_json: Path
    window_predictions_csv: Path
    native_predictions_csv: Path
    fold_artifact_dir: Path
    origin_git_commit: str
    alignment_fix_commit: str
    expected_manifest_rows: int
    expected_positional_mismatch_rows: int


@dataclass(frozen=True, slots=True)
class LegacySequenceCheckpointConfig:
    """Inputs for a checkpoint that lacks paired data and metric lineage."""

    checkpoint_path: Path
    expected_sha256: str


def build_historical_baseline_reconciliation(
    full_config: HistoricalFullOOFConfig,
    legacy_config: LegacySequenceCheckpointConfig | None = None,
) -> dict[str, Any]:
    """Build one audit packet without training or reading image tensors."""

    full_control = _build_full_oof_control(full_config)
    legacy_control = (
        _build_legacy_checkpoint_control(legacy_config)
        if legacy_config is not None
        else None
    )
    git_commit, git_dirty = _git_state()
    errors = [
        f"historical_full_oof:{error}"
        for error in full_control.get("errors", [])
    ]
    if legacy_control is not None:
        errors.extend(
            f"legacy_sequence_checkpoint:{error}"
            for error in legacy_control.get("errors", [])
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "registration_git_commit": git_commit,
        "registration_git_dirty": git_dirty,
        "historical_full_oof": full_control,
        "legacy_sequence_checkpoint": legacy_control,
        "io_scope": {
            "raw_data_read": False,
            "feature_tensor_read": False,
            "image_or_video_decode": False,
            "full_manifest_window_id_scan": True,
            "artifact_content_hash_scan": True,
        },
        "optimizer_steps": 0,
        "model_training_run": False,
        "errors": errors,
        "warnings": [
            "historical metrics are diagnostic metadata, not performance evidence",
            "the first performance baseline requires a reviewed aligned short run",
        ],
        "valid": not errors,
    }
    payload["errors"].extend(_semantic_errors(payload))
    payload["errors"] = sorted(set(payload["errors"]))
    payload["valid"] = not payload["errors"]
    return payload


def write_historical_baseline_reconciliation(
    payload: dict[str, Any],
    output_json: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a derived immutable-style audit with explicit replacement control."""

    require_output_paths_available([output_json], overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def check_historical_baseline_reconciliation(
    audit_json: Path,
) -> dict[str, Any]:
    """Recompute semantic and artifact-hash checks for a saved audit packet."""

    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    errors = list(payload.get("errors", []))
    errors.extend(_semantic_errors(payload))
    for control_name in ["historical_full_oof", "legacy_sequence_checkpoint"]:
        control = payload.get(control_name)
        if not control:
            continue
        errors.extend(
            f"{control_name}:{error}"
            for error in _artifact_drift_errors(control.get("artifacts", []))
        )
    errors = sorted(set(errors))
    return {
        "schema_version": "classification_v2_historical_baseline_check_v1",
        "audit_json": str(audit_json),
        "historical_full_oof_status": payload.get(
            "historical_full_oof", {}
        ).get("status"),
        "legacy_sequence_checkpoint_status": (
            payload.get("legacy_sequence_checkpoint") or {}
        ).get("status"),
        "performance_claim_allowed": False,
        "errors": errors,
        "warnings": payload.get("warnings", []),
        "valid": not errors,
    }


def _build_full_oof_control(
    config: HistoricalFullOOFConfig,
) -> dict[str, Any]:
    errors: list[str] = []
    alignment = _alignment_evidence(config, errors)
    run_audit = _read_json(config.run_audit_json, "run_audit", errors)
    metrics = _read_json(config.metrics_json, "metrics", errors)
    prediction_schema = _read_json(
        config.prediction_schema_json,
        "prediction_schema",
        errors,
    )
    _validate_historical_run_payloads(
        config,
        run_audit,
        metrics,
        prediction_schema,
        errors,
    )
    artifacts = _full_oof_artifacts(config, run_audit, errors)
    native_metrics = metrics.get("native_temporal_metrics", {})
    control = {
        "status": FULL_CONTROL_STATUS,
        "origin_git_commit": config.origin_git_commit,
        "alignment_fix_commit": config.alignment_fix_commit,
        "performance_evidence_valid": False,
        "claim_flags": dict(DISABLED_CLAIM_FLAGS),
        "known_defects": [
            {
                "code": "multimodal_window_positional_misalignment",
                "affected_rows": config.expected_positional_mismatch_rows,
                "manifest_rows": config.expected_manifest_rows,
                "defect_reproduced": alignment.get("defect_reproduced", False),
            }
        ],
        "known_lineage_gaps": [
            "origin_run_did_not_bind_input_artifact_hashes",
            "registration_hashes_do_not_prove_origin_time_input_bytes",
            "lineage_predates_current_hidden_and_behavior_review_gates",
        ],
        "artifact_hash_scope": "registration_time_integrity_only",
        "superseded_origin_claim_flags": {
            "paper_facing_result": run_audit.get("paper_facing_result"),
            "full_oof_training_verified": run_audit.get(
                "full_oof_training_verified"
            ),
            "superseded_by_known_alignment_defect": True,
        },
        "alignment_evidence": alignment,
        "diagnostic_metrics_snapshot": {
            "status": "INVALID_FOR_MODEL_QUALITY",
            "native_temporal_rows": native_metrics.get("rows"),
            "accuracy": native_metrics.get("accuracy"),
            "macro_f1": native_metrics.get("macro_f1"),
            "macro_recall": native_metrics.get("macro_recall"),
        },
        "reusable_evidence": [
            "cache_only_io_behavior",
            "checkpoint_loadability",
            "fold_orchestration_debugging",
            "runtime_and_vram_estimation",
        ],
        "forbidden_uses": [
            "architecture_promotion",
            "model_quality_baseline",
            "paired_performance_comparison",
            "paper_or_q2_claim",
            "threshold_or_model_selection",
        ],
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bytes_hashed": sum(
            int(artifact.get("size_bytes", 0)) for artifact in artifacts
        ),
        "errors": errors,
        "valid": not errors,
    }
    return control


def _alignment_evidence(
    config: HistoricalFullOOFConfig,
    errors: list[str],
) -> dict[str, Any]:
    try:
        split_ids = _read_window_ids(config.split_manifest_csv)
        image_ids = _read_window_ids(config.image_manifest_csv)
        interaction_ids = _read_window_ids(config.interaction_manifest_csv)
    except (OSError, ValueError) as exc:
        errors.append(f"alignment_input_error={exc}")
        return {"defect_reproduced": False}

    split_audit = audit_ordered_window_ids(
        "split_manifest",
        split_ids,
        {
            "image_window_manifest": image_ids,
            "interaction_window_manifest": interaction_ids,
        },
    )
    context_audit = audit_ordered_window_ids(
        "image_window_manifest",
        image_ids,
        {"interaction_window_manifest": interaction_ids},
    )
    image_comparison = split_audit["comparisons"]["image_window_manifest"]
    interaction_comparison = split_audit["comparisons"][
        "interaction_window_manifest"
    ]
    context_comparison = context_audit["comparisons"][
        "interaction_window_manifest"
    ]
    expected_rows = int(config.expected_manifest_rows)
    expected_mismatch = int(config.expected_positional_mismatch_rows)
    defect_reproduced = bool(
        len(split_ids) == expected_rows
        and len(image_ids) == expected_rows
        and len(interaction_ids) == expected_rows
        and image_comparison["missing_count"] == 0
        and image_comparison["extra_count"] == 0
        and interaction_comparison["missing_count"] == 0
        and interaction_comparison["extra_count"] == 0
        and image_comparison["order_mismatch_rows"] == expected_mismatch
        and interaction_comparison["order_mismatch_rows"] == expected_mismatch
        and context_comparison["order_mismatch_rows"] == 0
        and not context_audit["errors"]
    )
    if not defect_reproduced:
        errors.append("expected_historical_alignment_defect_not_reproduced")
    return {
        "columns_read": ["window_id"],
        "manifest_rows": expected_rows,
        "expected_positional_mismatch_rows": expected_mismatch,
        "split_to_context": split_audit,
        "image_to_interaction": context_audit,
        "defect_reproduced": defect_reproduced,
    }


def _validate_historical_run_payloads(
    config: HistoricalFullOOFConfig,
    run_audit: dict[str, Any],
    metrics: dict[str, Any],
    prediction_schema: dict[str, Any],
    errors: list[str],
) -> None:
    if run_audit.get("git_commit") != config.origin_git_commit:
        errors.append("origin_git_commit_mismatch")
    if run_audit.get("run_mode") != "full":
        errors.append("historical_run_mode_not_full")
    if run_audit.get("full_oof_training_verified") is not True:
        errors.append("historical_full_oof_not_verified")
    if run_audit.get("prediction_rows") != prediction_schema.get(
        "prediction_rows"
    ):
        errors.append("prediction_row_count_disagrees_with_schema_audit")
    native_rows = metrics.get("native_temporal_metrics", {}).get("rows")
    if run_audit.get("native_temporal_rows") != native_rows:
        errors.append("native_prediction_row_count_disagrees_with_metrics")
    if run_audit.get("load_audit", {}).get("row_counts", {}).get(
        "split"
    ) != config.expected_manifest_rows:
        errors.append("run_audit_manifest_row_count_mismatch")


def _full_oof_artifacts(
    config: HistoricalFullOOFConfig,
    run_audit: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    paths = [
        config.split_manifest_csv,
        config.image_manifest_csv,
        config.interaction_manifest_csv,
        config.run_audit_json,
        config.metrics_json,
        config.prediction_schema_json,
        config.window_predictions_csv,
        config.native_predictions_csv,
    ]
    checkpoints = sorted(
        config.fold_artifact_dir.glob("native_oof_*_work/trained_model.pt")
    )
    training_audits = sorted(
        config.fold_artifact_dir.glob("native_oof_*_work/training_audit.json")
    )
    expected_folds = len(run_audit.get("fold_audits", []))
    if expected_folds <= 0:
        errors.append("historical_run_has_no_fold_audits")
    if len(checkpoints) != expected_folds:
        errors.append(
            "historical_checkpoint_count_mismatch="
            f"expected:{expected_folds},actual:{len(checkpoints)}"
        )
    if len(training_audits) != expected_folds:
        errors.append(
            "historical_training_audit_count_mismatch="
            f"expected:{expected_folds},actual:{len(training_audits)}"
        )
    paths.extend(checkpoints)
    paths.extend(training_audits)
    records = [_artifact_record(path) for path in paths]
    errors.extend(
        f"missing_or_invalid_artifact={record['path']}"
        for record in records
        if not record.get("exists") or record.get("hash_status") != "ok"
    )
    return records


def _build_legacy_checkpoint_control(
    config: LegacySequenceCheckpointConfig,
) -> dict[str, Any]:
    errors: list[str] = []
    artifact = _artifact_record(config.checkpoint_path)
    if not artifact.get("exists"):
        errors.append("legacy_checkpoint_missing")
    elif artifact.get("sha256") != config.expected_sha256:
        errors.append("legacy_checkpoint_sha256_mismatch")

    spec: dict[str, Any] = {}
    state_keys = 0
    state_tensor_element_count = 0
    if not errors:
        try:
            import torch

            raw = torch.load(
                config.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            state = _checkpoint_state_dict(raw)
            spec = _infer_checkpoint_spec(state)
            state_keys = len(state)
            state_tensor_element_count = sum(
                int(value.numel())
                for value in state.values()
                if hasattr(value, "numel")
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"safe_checkpoint_inspection_failed={exc}")
    labels = _labels_for_class_count(int(spec.get("num_classes", 0))) if spec else []
    if spec and len(labels) != 10:
        errors.append("legacy_checkpoint_not_ten_class")
    return {
        "status": LEGACY_CONTROL_STATUS,
        "performance_evidence_valid": False,
        "claim_flags": dict(DISABLED_CLAIM_FLAGS),
        "safe_load_policy": "torch_load_weights_only_true",
        "model_spec": spec,
        "label_order": labels,
        "state_dict_keys": state_keys,
        "state_tensor_element_count": state_tensor_element_count,
        "known_lineage_gaps": [
            "no_frozen_training_dataset_hash",
            "no_verified_grouped_split_manifest",
            "no_paired_native_unit_predictions",
            "no_reproducible_training_config_or_seed_manifest",
        ],
        "reusable_evidence": [
            "architecture_reference",
            "checkpoint_integrity",
            "runtime_loader_compatibility",
        ],
        "forbidden_uses": [
            "model_quality_baseline",
            "paired_performance_comparison",
            "paper_or_q2_claim",
        ],
        "artifacts": [artifact],
        "errors": errors,
        "valid": not errors,
    }


def _semantic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("historical_reconciliation_schema_mismatch")
    full = payload.get("historical_full_oof") or {}
    if full.get("status") != FULL_CONTROL_STATUS:
        errors.append("historical_full_oof_status_not_locked")
    if full.get("performance_evidence_valid") is not False:
        errors.append("historical_full_oof_performance_must_be_invalid")
    if full.get("alignment_evidence", {}).get("defect_reproduced") is not True:
        errors.append("historical_alignment_defect_not_reproduced")
    errors.extend(_claim_flag_errors(full, "historical_full_oof"))
    legacy = payload.get("legacy_sequence_checkpoint")
    if legacy is not None:
        if legacy.get("status") != LEGACY_CONTROL_STATUS:
            errors.append("legacy_checkpoint_status_not_locked")
        if legacy.get("performance_evidence_valid") is not False:
            errors.append("legacy_checkpoint_performance_must_be_invalid")
        errors.extend(_claim_flag_errors(legacy, "legacy_sequence_checkpoint"))
    if payload.get("optimizer_steps") != 0 or payload.get("model_training_run"):
        errors.append("historical_reconciliation_must_not_train")
    return errors


def _claim_flag_errors(control: dict[str, Any], name: str) -> list[str]:
    flags = control.get("claim_flags", {})
    return [
        f"{name}_claim_flag_not_false={flag}"
        for flag in DISABLED_CLAIM_FLAGS
        if flags.get(flag) is not False
    ]


def _artifact_drift_errors(
    artifacts: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for recorded in artifacts:
        current = _artifact_record(Path(str(recorded.get("path", ""))))
        if not current.get("exists"):
            errors.append(f"artifact_missing={recorded.get('path')}")
            continue
        if current.get("size_bytes") != recorded.get("size_bytes"):
            errors.append(f"artifact_size_drift={recorded.get('path')}")
        if current.get("sha256") != recorded.get("sha256"):
            errors.append(f"artifact_sha256_drift={recorded.get('path')}")
    return errors


def _read_window_ids(path: Path) -> pd.Series:
    """Read only the stable key needed to reproduce positional alignment."""

    try:
        frame = pd.read_csv(
            path,
            usecols=["window_id"],
            dtype={"window_id": "string"},
        )
    except ValueError as exc:
        raise ValueError(f"missing_window_id_column={path}") from exc
    return frame["window_id"]


def _checkpoint_state_dict(raw: Any) -> dict[str, Any]:
    """Extract tensors without importing the runtime model package."""

    if isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    if not isinstance(state, dict):
        raise ValueError("unsupported legacy checkpoint format")
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
    }


def _infer_checkpoint_spec(state: dict[str, Any]) -> dict[str, Any]:
    """Infer only architecture fields proven by state-dict tensor shapes."""

    head_weight = state.get("head.3.weight")
    projection_weight = state.get("cnn_proj.weight")
    first_head_weight = state.get("head.0.weight")
    if head_weight is None or projection_weight is None or first_head_weight is None:
        raise ValueError("legacy checkpoint is missing required architecture tensors")
    d_model = int(first_head_weight.shape[0])
    base_dim = int(projection_weight.shape[0])
    extra_dim = d_model - base_dim
    if extra_dim <= 0:
        extra_dim = 10
    layer_indices = [
        int(key.split(".")[2])
        for key in state
        if key.startswith("transformer.layers.")
        and len(key.split(".")) > 2
        and key.split(".")[2].isdigit()
    ]
    backbone_name = (
        "resnet34"
        if any(
            key.startswith("cnn.6.5.") or key.startswith("cnn.7.2.")
            for key in state
        )
        else "resnet18"
    )
    return {
        "num_classes": int(head_weight.shape[0]),
        "d_model": d_model,
        "extra_dim": extra_dim,
        "num_layers": max(layer_indices) + 1 if layer_indices else 2,
        "backbone_name": backbone_name,
    }


def _labels_for_class_count(num_classes: int) -> list[str]:
    """Use the canonical order only when the checkpoint proves ten outputs."""

    if num_classes == len(VALID_BEHAVIORS):
        return list(VALID_BEHAVIORS)
    return [f"class_{index}" for index in range(num_classes)]


def _read_json(
    path: Path,
    name: str,
    errors: list[str],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_{name}_json={path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"invalid_{name}_payload_type={path}")
        return {}
    return payload


def _artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not record["exists"]:
        return record
    record["size_bytes"] = int(path.stat().st_size)
    record["sha256"] = _sha256(path)
    record["hash_status"] = "ok"
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, dirty
