"""Controlled temporal-base screening on the frozen legacy 16f universe."""

from __future__ import annotations

import copy
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as cached_engine,
)
from pig_behavior.classification_v2.training.legacy_c6_prepared_source import (
    LegacyC6PreparedSource,
    load_legacy_c6_prepared_source,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    git_state,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_training import (
    LegacyL5CachedShortSelection,
    LegacyL5CachedTrainingConfig,
    LegacyL5CachedTrainingOutcome,
    train_legacy_l5_cached_short_core,
)
from pig_behavior.classification_v2.training.legacy_development_temporal_sampling import (
    SHORT_SCOPE,
    TemporalSamplingConfig,
    TemporalSamplingSource,
    derive_temporal_sampling_view,
    load_temporal_sampling_source,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_selection_config.v1"
)
CONFIG_SCHEMA_V2 = (
    "classification_v2.legacy_development.temporal_base_selection_config.v2"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_selection_preflight.v1"
)
RUN_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_selection_run.v1"
)
MATRIX_SCHEMA = (
    "classification_v2.legacy_development.temporal_base_selection_short_matrix.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
FULL_SCOPE = "full_development_confirmation"
SOURCE_FULL_SCOPE = "full_development_baseline"
PARAMETER_MATCH_MAX_RELATIVE_DELTA = 0.005

MODE_SPECS: dict[str, dict[str, Any]] = {
    "SF128": {
        "native_frame_offsets": [7],
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 68_234,
        "timing_contract": "ignored",
    },
    "M128": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 68_234,
        "timing_contract": "ignored",
    },
    "A128": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "masked_attention",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 68_363,
        "timing_contract": "ignored",
    },
    "MW317": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 317,
        "transformer_layers": 1,
        "transformer_heads": 1,
        "expected_parameter_count": 167_459,
        "timing_contract": "ignored",
    },
    "TCN128": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "masked_tcn",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 167_435,
        "timing_contract": "ordered_slots_without_elapsed_time",
    },
    "MW381": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 381,
        "transformer_layers": 1,
        "transformer_heads": 1,
        "expected_parameter_count": 201_059,
        "timing_contract": "ignored",
    },
    "TR128": {
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_encoder_name": "small_transformer",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 200_843,
        "timing_contract": "ordered_slots_with_real_elapsed_time",
    },
}

CONTROLLED_PAIRS = {
    "multiple_frames": ("M128", "SF128"),
    "content_weighting": ("A128", "M128"),
    "ordered_tcn": ("TCN128", "MW317"),
    "timed_transformer": ("TR128", "MW381"),
}


@dataclass(frozen=True, slots=True)
class TemporalBaseSelectionConfig:
    """Hash-bound configuration for one Stage A execution scope."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    @property
    def training_scope(self) -> str:
        return str(self.payload["training_scope"])

    @property
    def source_config_path(self) -> Path:
        source_field = _source_spec_name(self.payload)
        value = str(self.payload[source_field]["path"])
        return self.repo_root / value

    @property
    def output_root(self) -> Path:
        return self.repo_root / str(self.payload["output"]["root"])


@dataclass(frozen=True, slots=True)
class DerivedTemporalBaseView:
    """One mode-specific view over an unchanged native-unit universe."""

    mode_id: str
    view: LegacyL5CachedFeatureView
    slot_manifest: pd.DataFrame
    audit: dict[str, Any]


def load_temporal_base_selection_config(
    path: Path,
) -> TemporalBaseSelectionConfig:
    """Load one exact Stage A config and reject semantic or hash drift."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_config(payload)
    config = TemporalBaseSelectionConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    source_field = _source_spec_name(payload)
    _verify_file_spec(config.repo_root, payload[source_field])
    _verify_file_spec(config.repo_root, payload["implementation"])
    if config.payload["schema_version"] == CONFIG_SCHEMA_V2:
        _verify_file_spec(config.repo_root, payload["model_implementation"])
    return config


def load_temporal_base_source(
    config: TemporalBaseSelectionConfig,
) -> TemporalSamplingSource | LegacyC6PreparedSource:
    """Reuse the frozen T16 source and its native-first grouped selection."""

    if config.payload["schema_version"] == CONFIG_SCHEMA_V2:
        execution = config.payload["execution"]
        if execution.get("data_run_authorized") is not True:
            raise PermissionError("rebuild temporal-base data run is fail-closed")
        return load_legacy_c6_prepared_source(
            config.source_config_path,
            repo_root=config.repo_root,
        )
    shim = TemporalSamplingConfig(
        path=config.path,
        payload={
            "training_scope": config.training_scope,
            "source_ladder_config": config.payload["source_ladder_config"],
            "output": config.payload["output"],
        },
        repo_root=config.repo_root,
    )
    source = load_temporal_sampling_source(shim)
    expected_source_scope = (
        SHORT_SCOPE if config.training_scope == SHORT_SCOPE else SOURCE_FULL_SCOPE
    )
    if source.parent_audit.get("training_scope") not in {
        expected_source_scope,
        None,
    }:
        raise ValueError("temporal-base source scope drift")
    return source


def derive_temporal_base_view(
    base_view: LegacyL5CachedFeatureView,
    mode_id: str,
) -> DerivedTemporalBaseView:
    """Derive an exact offset view without dropping or duplicating native units."""

    spec = _mode_spec(mode_id)
    c6 = derive_temporal_sampling_view(
        base_view,
        "c6_contiguous_centered",
    )
    offsets = np.asarray(spec["native_frame_offsets"], dtype=np.int64)
    selected_rows = base_view.feature_rows[:, offsets].copy()
    selected_mask = base_view.observed_mask[:, offsets].copy()
    base_elapsed = np.cumsum(base_view.time_delta, axis=1)
    selected_elapsed = base_elapsed[:, offsets]
    selected_delta = np.zeros_like(selected_elapsed, dtype=np.float32)
    selected_delta[:, 1:] = np.diff(selected_elapsed, axis=1).astype(np.float32)
    if not selected_mask.all():
        raise ValueError(f"{mode_id} contains unavailable selected slots")
    if not np.isfinite(selected_delta).all():
        raise ValueError(f"{mode_id} contains nonfinite timing")
    if selected_delta.shape[1] > 1 and (selected_delta[:, 1:] <= 0.0).any():
        raise ValueError(f"{mode_id} contains nonpositive elapsed deltas")

    windows = base_view.windows.copy(deep=True)
    windows["temporal_base_mode_id"] = mode_id
    windows["temporal_sampling_offsets_json"] = json.dumps(
        spec["native_frame_offsets"],
        separators=(",", ":"),
    )
    if windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("temporal-base view duplicates native units")
    view_name = f"legacy_{mode_id.lower()}_stage_a_v1"
    audit = {
        "schema_version": "classification_v2.temporal_base_view_audit.v1",
        "mode_id": mode_id,
        "temporal_view_name": view_name,
        "native_frame_offsets": list(spec["native_frame_offsets"]),
        "sequence_length": int(len(offsets)),
        "timing_contract": str(spec["timing_contract"]),
        "base_temporal_view_name": base_view.temporal_view_name,
        "native_units": int(len(windows)),
        "eligible_train_native_units": int(
            (windows["l5_role"] == "train").sum()
        ),
        "validation_native_units": int(
            (windows["l5_role"] == "validation").sum()
        ),
        "rows_dropped": 0,
        "labels_changed": 0,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "one_sequence_per_native_unit": True,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "errors": [],
        "valid": True,
    }
    inherited = copy.deepcopy(base_view.audit)
    inherited["derived_temporal_base"] = audit
    inherited["temporal_view_name"] = view_name
    inherited["sequence_length"] = len(offsets)
    view = replace(
        base_view,
        temporal_view_name=view_name,
        sequence_length=len(offsets),
        windows=windows,
        feature_rows=selected_rows,
        observed_mask=selected_mask,
        time_delta=selected_delta,
        sample_weights=np.ones(len(windows), dtype=np.float64),
        audit=inherited,
    )
    slot_manifest = _build_slot_manifest(view, mode_id, offsets)
    if len(c6.view.windows) != len(view.windows):
        raise ValueError("temporal-base derivation loses native units")
    return DerivedTemporalBaseView(
        mode_id=mode_id,
        view=view,
        slot_manifest=slot_manifest,
        audit=audit,
    )


def build_training_adapter(
    config: TemporalBaseSelectionConfig,
    source: TemporalSamplingSource | LegacyC6PreparedSource,
    derived: DerivedTemporalBaseView,
) -> tuple[LegacyL5CachedTrainingConfig, LegacyL5CachedShortSelection]:
    """Bind one mode to the canonical cached trainer and frozen selection."""

    spec = _mode_spec(derived.mode_id)
    train_count = len(source.selection.train_positions)
    validation_count = len(source.selection.validation_positions)
    optimization = copy.deepcopy(config.payload["optimization"])
    batch_size = int(optimization["batch_size"])
    steps_per_epoch = (train_count + batch_size - 1) // batch_size
    optimization["maximum_optimizer_steps"] = steps_per_epoch * int(
        optimization["epochs"]
    )
    model = {
        **copy.deepcopy(config.payload["model_common"]),
        "temporal_encoder_name": spec["temporal_encoder_name"],
        "hidden_dim": spec["hidden_dim"],
        "transformer_layers": spec["transformer_layers"],
        "transformer_heads": spec["transformer_heads"],
    }
    adapter_payload = {
        "schema_version": CONFIG_SCHEMA,
        "training_scope": config.training_scope,
        "data": {
            "control_id": derived.view.control_id,
            "temporal_view_name": derived.view.temporal_view_name,
            "sequence_length": derived.view.sequence_length,
            "feature_dim": FEATURE_DIM,
            "model_visible_roles": ["train", "validation"],
            "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
            "expected_train_native_units": train_count,
            "expected_validation_native_units": validation_count,
            "native_prediction_aggregation": "one_sequence_per_native_unit",
        },
        "model": model,
        "optimization": optimization,
    }
    adapter = LegacyL5CachedTrainingConfig(
        path=config.path,
        payload=adapter_payload,
        repo_root=config.repo_root,
    )
    manifest = source.selection.manifest.copy(deep=True)
    manifest["temporal_base_mode_id"] = derived.mode_id
    manifest["temporal_sampling_offsets_json"] = json.dumps(
        spec["native_frame_offsets"],
        separators=(",", ":"),
    )
    selection_hash = cached_engine._dataframe_sha256(manifest)
    selection_audit = copy.deepcopy(source.selection.audit)
    selection_audit.update(
        {
            "training_scope": config.training_scope,
            "mode_id": derived.mode_id,
            "selection_content_sha256": selection_hash,
            "temporal_sampling_offsets": list(spec["native_frame_offsets"]),
        }
    )
    selection = LegacyL5CachedShortSelection(
        manifest=manifest,
        train_positions=source.selection.train_positions.copy(),
        validation_positions=source.selection.validation_positions.copy(),
        audit=selection_audit,
    )
    _validate_adapter(config, derived, selection, adapter)
    return adapter, selection


def preflight_temporal_base_selection(
    config: TemporalBaseSelectionConfig,
) -> dict[str, Any]:
    """Check data, shapes, parameters, timing, pairing, and CPU isolation."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    modes: dict[str, dict[str, Any]] = {}
    native_keys: list[str] | None = None
    try:
        source = load_temporal_base_source(config)
        for mode_id in MODE_SPECS:
            derived = derive_temporal_base_view(source.base_view, mode_id)
            adapter, selection = build_training_adapter(
                config,
                source,
                derived,
            )
            sample = selection.train_positions[:64]
            batch = derived.view.load_sequences(sample)
            model = _build_model(adapter)
            model.eval()
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(batch),
                    torch.from_numpy(
                        derived.view.observed_mask[sample]
                    ).float(),
                    time_delta=torch.from_numpy(
                        derived.view.time_delta[sample]
                    ).float(),
                )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            expected = int(MODE_SPECS[mode_id]["expected_parameter_count"])
            if parameters != expected:
                errors.append(
                    f"{mode_id}:parameter_count={parameters},expected={expected}"
                )
            expected_shape = [len(sample), len(VALID_BEHAVIORS)]
            if list(logits.shape) != expected_shape:
                errors.append(f"{mode_id}:forward_shape={list(logits.shape)}")
            keys = derived.view.windows["temporal_unit_key"].astype(str).tolist()
            if native_keys is None:
                native_keys = keys
            elif keys != native_keys:
                errors.append(f"{mode_id}:native_unit_order_drift")
            modes[mode_id] = {
                **derived.audit,
                "slot_manifest_sha256": cached_engine._dataframe_sha256(
                    derived.slot_manifest
                ),
                "selection_sha256": selection.audit[
                    "selection_content_sha256"
                ],
                "parameter_count": parameters,
                "forward_shape": list(logits.shape),
                "selected_train_native_units": int(
                    len(selection.train_positions)
                ),
                "selected_validation_native_units": int(
                    len(selection.validation_positions)
                ),
                "optimizer_steps": int(
                    adapter.payload["optimization"]["maximum_optimizer_steps"]
                ),
            }
            del batch, logits, model
        errors.extend(_parameter_control_errors(modes))
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("CPU preflight initialized CUDA")
    valid = not errors and set(modes) == set(MODE_SPECS)
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_TEMPORAL_BASE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_TEMPORAL_BASE_PREFLIGHT"
        ),
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "selected_skills": list(config.payload["experiment_contract"]["skills"]),
        "modes": modes,
        "controlled_pairs": {
            name: {"candidate": pair[0], "baseline": pair[1]}
            for name, pair in CONTROLLED_PAIRS.items()
        },
        "native_units": len(native_keys or []),
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "short_run_authorized": valid and config.training_scope == SHORT_SCOPE,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "errors": errors,
        "valid": valid,
    }


def validate_full_launch_gate(
    config: TemporalBaseSelectionConfig,
) -> dict[str, Any] | None:
    """Require the exact short matrix before any full-development process."""

    if config.training_scope != FULL_SCOPE:
        return None
    contract = config.payload["experiment_contract"]
    required = {
        "short_gate_path",
        "short_gate_sha256",
        "short_gate_status",
        "short_config_sha256",
        "launch_script_path",
        "launch_script_sha256",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"full temporal-base gate fields missing={missing}")
    gate_path = config.repo_root / str(contract["short_gate_path"])
    if not gate_path.is_file():
        raise FileNotFoundError(f"temporal-base short gate missing: {gate_path}")
    if file_sha256(gate_path) != contract["short_gate_sha256"]:
        raise ValueError("temporal-base short gate hash drift")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != contract["short_gate_status"]:
        raise ValueError("temporal-base short gate status drift")
    if gate.get("config_sha256") != contract["short_config_sha256"]:
        raise ValueError("temporal-base short config hash drift")
    if gate.get("full_confirmation_authorized") is not True:
        raise ValueError("temporal-base short gate does not authorize full")
    if gate.get("valid") is not True or gate.get("errors") != []:
        raise ValueError("temporal-base short gate is invalid")
    if set(gate.get("modes", {})) != set(MODE_SPECS):
        raise ValueError("temporal-base short mode set drift")
    launch_path = config.repo_root / str(contract["launch_script_path"])
    if not launch_path.is_file():
        raise FileNotFoundError(f"temporal-base launcher missing: {launch_path}")
    if file_sha256(launch_path) != contract["launch_script_sha256"]:
        raise ValueError("temporal-base launcher hash drift")
    return {
        "path": str(gate_path.resolve()),
        "sha256": file_sha256(gate_path),
        "status": str(contract["short_gate_status"]),
        "short_config_sha256": str(contract["short_config_sha256"]),
        "launch_script_sha256": str(contract["launch_script_sha256"]),
        "full_confirmation_authorized": True,
        "valid": True,
    }


def execute_temporal_base_run(
    config: TemporalBaseSelectionConfig,
    mode_id: str,
    repeat_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Train one exact mode in an exclusive process-local directory."""

    _mode_spec(mode_id)
    if not repeat_id or any(character in repeat_id for character in "\\/:*?\"<>|"):
        raise ValueError("repeat_id is blank or unsafe")
    full_gate = validate_full_launch_gate(config)
    run_root = config.output_root / config.training_scope / mode_id / repeat_id
    run_root.mkdir(parents=True, exist_ok=False)
    started_wall = time.time()
    planned = {
        "schema_version": "classification_v2.temporal_base_planned_run.v1",
        "status": "planned",
        "mode_id": mode_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "started_at_utc": _utc_now(),
        "started_wall_time": started_wall,
        "git_state": git_state(),
    }
    _write_json_exclusive(run_root / "planned_run_manifest.json", planned)
    try:
        preflight = preflight_temporal_base_selection(config)
        if not preflight["valid"]:
            raise RuntimeError(f"preflight failed: {preflight['errors']}")
        source = load_temporal_base_source(config)
        derived = derive_temporal_base_view(source.base_view, mode_id)
        adapter, selection = build_training_adapter(config, source, derived)
        device = str(config.payload["execution"]["device"])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        outcome = train_legacy_l5_cached_short_core(
            derived.view,
            selection,
            adapter,
            device=device,
        )
        peak_memory = (
            int(torch.cuda.max_memory_reserved(torch.device(device)))
            if device.startswith("cuda")
            else 0
        )
        result = _write_success_artifacts(
            config,
            run_root=run_root,
            mode_id=mode_id,
            source=source,
            derived=derived,
            selection=selection,
            adapter=adapter,
            outcome=outcome,
            planned=planned,
            preflight=preflight,
            full_gate=full_gate,
            runtime_seconds=time.time() - started_wall,
            device=device,
            peak_memory_bytes=peak_memory,
        )
        return run_root, result
    except Exception as error:
        failure = {
            **planned,
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": time.time() - started_wall,
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
        }
        _write_json_exclusive(run_root / "failure_result.json", failure)
        raise


def audit_temporal_base_short_matrix(
    config: TemporalBaseSelectionConfig,
) -> tuple[Path, dict[str, Any]]:
    """Require deterministic isolated repeats for all predeclared Stage A modes."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("short matrix requires short_repeat_gate config")
    required_repeats = list(config.payload["execution"]["required_repeats"])
    errors: list[str] = []
    modes: dict[str, Any] = {}
    process_ids: list[int] = []
    intervals: list[tuple[float, float]] = []
    common_native_hash: str | None = None
    for mode_id in MODE_SPECS:
        packets: list[dict[str, Any]] = []
        for repeat_id in required_repeats:
            path = (
                config.output_root
                / config.training_scope
                / mode_id
                / str(repeat_id)
                / "run_result.json"
            )
            if not path.is_file():
                errors.append(f"missing_run_result={mode_id}:{repeat_id}")
                continue
            packet = json.loads(path.read_text(encoding="utf-8"))
            packets.append(packet)
            process_ids.append(int(packet["process_id"]))
            intervals.append(
                (float(packet["started_wall_time"]), float(packet["ended_wall_time"]))
            )
        if len(packets) != len(required_repeats):
            continue
        exact_fields = (
            "config_sha256",
            "implementation_sha256",
            "source_config_sha256",
            "slot_manifest_sha256",
            "selection_native_unit_sha256",
            "parameter_sha256",
            "prediction_content_sha256",
            "epoch_metrics_content_sha256",
            "parameter_count",
            "optimizer_steps",
        )
        mismatches = [
            field
            for field in exact_fields
            if len({str(packet[field]) for packet in packets}) != 1
        ]
        if mismatches:
            errors.append(f"{mode_id}:repeat_mismatch={mismatches}")
        native_hash = str(packets[0]["selection_native_unit_sha256"])
        if common_native_hash is None:
            common_native_hash = native_hash
        elif native_hash != common_native_hash:
            errors.append(f"{mode_id}:paired_native_universe_drift")
        modes[mode_id] = {
            "repeat_ids": required_repeats,
            "process_ids": [int(packet["process_id"]) for packet in packets],
            "repeat_mismatches": mismatches,
            "metrics": packets[0]["metrics"],
            "parameter_count": int(packets[0]["parameter_count"]),
            "optimizer_steps": int(packets[0]["optimizer_steps"]),
        }
    if len(process_ids) != len(set(process_ids)):
        errors.append("repeat_process_ids_are_not_distinct")
    ordered = sorted(intervals)
    if any(
        right[0] < left[1]
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        errors.append("repeat_execution_intervals_overlap")
    valid = not errors and set(modes) == set(MODE_SPECS)
    payload = {
        "schema_version": MATRIX_SCHEMA,
        "status": (
            "PASS_LEGACY_TEMPORAL_BASE_SHORT_MATRIX"
            if valid
            else "FAIL_LEGACY_TEMPORAL_BASE_SHORT_MATRIX"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "changed_scientific_family": "predeclared_temporal_base_matrix",
        "controlled_pairs": {
            name: {"candidate": pair[0], "baseline": pair[1]}
            for name, pair in CONTROLLED_PAIRS.items()
        },
        "modes": modes,
        "common_native_unit_sha256": common_native_hash,
        "full_confirmation_authorized": valid,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }
    output = config.output_root / "temporal_base_short_matrix.json"
    _write_json_exclusive(output, payload)
    return output, payload


def _write_success_artifacts(
    config: TemporalBaseSelectionConfig,
    *,
    run_root: Path,
    mode_id: str,
    source: TemporalSamplingSource | LegacyC6PreparedSource,
    derived: DerivedTemporalBaseView,
    selection: LegacyL5CachedShortSelection,
    adapter: LegacyL5CachedTrainingConfig,
    outcome: LegacyL5CachedTrainingOutcome,
    planned: dict[str, Any],
    preflight: dict[str, Any],
    full_gate: dict[str, Any] | None,
    runtime_seconds: float,
    device: str,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    spec = _mode_spec(mode_id)
    predictions = outcome.predictions.copy()
    predictions["temporal_base_mode_id"] = mode_id
    predictions["temporal_input_offsets_json"] = json.dumps(
        spec["native_frame_offsets"],
        separators=(",", ":"),
    )
    predictions["temporal_encoder_name"] = spec["temporal_encoder_name"]
    predictions["hidden_dim"] = int(spec["hidden_dim"])
    per_class = outcome.per_class_metrics.assign(temporal_base_mode_id=mode_id)
    confusion = outcome.confusion.assign(temporal_base_mode_id=mode_id)
    epoch_metrics = outcome.epoch_metrics.assign(temporal_base_mode_id=mode_id)
    _write_dataframe_exclusive(
        run_root / "validation_native_predictions.csv",
        predictions,
    )
    _write_dataframe_exclusive(run_root / "metrics_per_class.csv", per_class)
    _write_dataframe_exclusive(run_root / "confusion_matrix.csv", confusion)
    _write_dataframe_exclusive(run_root / "epoch_metrics.csv", epoch_metrics)
    _write_dataframe_exclusive(
        run_root / "derived_slot_manifest.csv",
        derived.slot_manifest,
    )
    parameters = sum(parameter.numel() for parameter in _build_model(adapter).parameters())
    metrics = {
        **outcome.metrics,
        "temporal_base_mode_id": mode_id,
        "native_frame_offsets": list(spec["native_frame_offsets"]),
        "temporal_encoder_name": str(spec["temporal_encoder_name"]),
        "hidden_dim": int(spec["hidden_dim"]),
        "parameter_count": parameters,
        "optimizer_steps": outcome.optimizer_steps,
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
    }
    _write_json_exclusive(run_root / "metrics_global.json", metrics)
    checkpoint = {
        "schema_version": "classification_v2.temporal_base_checkpoint.v1",
        "config_sha256": config.sha256,
        "mode_id": mode_id,
        "mode_spec": spec,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "best_epoch": outcome.best_epoch,
        "model_state": outcome.model_state,
        "optimizer_state": outcome.optimizer_state,
    }
    checkpoint_path = run_root / "checkpoint.pt"
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_manifest = {
        "schema_version": "classification_v2.temporal_base_checkpoint_manifest.v1",
        "mode_id": mode_id,
        "config_sha256": config.sha256,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "parameter_sha256": outcome.parameter_sha256,
        "best_epoch": outcome.best_epoch,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(
        run_root / "checkpoint_manifest.json",
        checkpoint_manifest,
    )
    prediction_path = run_root / "validation_native_predictions.csv"
    prediction_manifest = {
        "schema_version": "classification_v2.temporal_base_prediction_manifest.v1",
        "mode_id": mode_id,
        "config_sha256": config.sha256,
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": file_sha256(prediction_path),
        "prediction_rows": int(len(predictions)),
        "native_unit_rows": int(
            predictions["temporal_unit_key"].astype(str).nunique()
        ),
        "outer_holdout_rows": 0,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(
        run_root / "prediction_manifest.json",
        prediction_manifest,
    )
    environment = {
        "schema_version": "classification_v2.temporal_base_environment.v1",
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "peak_memory_bytes": peak_memory_bytes,
    }
    _write_json_exclusive(run_root / "environment.json", environment)
    artifact_names = (
        "validation_native_predictions.csv",
        "metrics_per_class.csv",
        "confusion_matrix.csv",
        "epoch_metrics.csv",
        "derived_slot_manifest.csv",
        "metrics_global.json",
        "checkpoint.pt",
        "checkpoint_manifest.json",
        "prediction_manifest.json",
        "environment.json",
    )
    artifact_manifest = {
        "schema_version": "classification_v2.temporal_base_artifacts.v1",
        "mode_id": mode_id,
        "artifacts": [
            {
                "name": name,
                "path": str((run_root / name).resolve()),
                "sha256": file_sha256(run_root / name),
                "size_bytes": int((run_root / name).stat().st_size),
            }
            for name in artifact_names
        ],
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "artifact_manifest.json", artifact_manifest)
    validation_keys = derived.view.windows.iloc[
        selection.validation_positions
    ]["temporal_unit_key"].astype(str)
    native_hash = _string_sequence_sha256(validation_keys.tolist())
    ended_wall = time.time()
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "mode_id": mode_id,
        "mode_spec": spec,
        "process_id": os.getpid(),
        "started_at_utc": planned["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_config_sha256": source.source_config_sha256,
        "git_state": planned["git_state"],
        "view_audit": derived.audit,
        "selection_audit": selection.audit,
        "selection_native_unit_sha256": native_hash,
        "parameter_count": parameters,
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "epoch_metrics_content_sha256": outcome.epoch_metrics_sha256,
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "runtime_profile": {
            "device": device,
            "peak_memory_bytes": peak_memory_bytes,
            "maximum_loaded_batch_bytes": outcome.maximum_loaded_batch_bytes,
        },
        "full_launch_gate": full_gate,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "run_manifest.json", run_manifest)
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "mode_id": mode_id,
        "repeat_id": planned["repeat_id"],
        "process_id": os.getpid(),
        "started_wall_time": float(planned["started_wall_time"]),
        "ended_wall_time": ended_wall,
        "runtime_seconds": runtime_seconds,
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_config_sha256": source.source_config_sha256,
        "slot_manifest_sha256": cached_engine._dataframe_sha256(
            derived.slot_manifest
        ),
        "selection_native_unit_sha256": native_hash,
        "parameter_count": parameters,
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "epoch_metrics_content_sha256": outcome.epoch_metrics_sha256,
        "optimizer_steps": outcome.optimizer_steps,
        "metrics": metrics,
        "preflight_status": preflight["status"],
        "run_manifest_sha256": file_sha256(run_root / "run_manifest.json"),
        "artifact_manifest_sha256": file_sha256(
            run_root / "artifact_manifest.json"
        ),
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "run_result.json", result)
    return result


def _build_slot_manifest(
    view: LegacyL5CachedFeatureView,
    mode_id: str,
    offsets: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, row in enumerate(view.windows.itertuples(index=False)):
        for slot_index, offset in enumerate(offsets):
            records.append(
                {
                    "temporal_base_mode_id": mode_id,
                    "window_id": str(row.window_id),
                    "temporal_unit_key": str(row.temporal_unit_key),
                    "l5_role": str(row.l5_role),
                    "slot_index": slot_index,
                    "native_frame_offset": int(offset),
                    "feature_row": int(view.feature_rows[position, slot_index]),
                    "observed_mask": bool(
                        view.observed_mask[position, slot_index]
                    ),
                    "time_delta": float(view.time_delta[position, slot_index]),
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != len(view.windows) * view.sequence_length:
        raise ValueError("temporal-base slot manifest row count drift")
    if frame[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("temporal-base slot manifest duplicates slots")
    return frame


def _validate_adapter(
    config: TemporalBaseSelectionConfig,
    derived: DerivedTemporalBaseView,
    selection: LegacyL5CachedShortSelection,
    adapter: LegacyL5CachedTrainingConfig,
) -> None:
    data = adapter.payload["data"]
    expected_train = int(data["expected_train_native_units"])
    expected_validation = int(data["expected_validation_native_units"])
    if len(selection.train_positions) != expected_train:
        raise ValueError("temporal-base train native count drift")
    if len(selection.validation_positions) != expected_validation:
        raise ValueError("temporal-base validation native count drift")
    if selection.audit.get("outer_holdout_rows") != 0:
        raise ValueError("temporal-base selection exposes outer holdout")
    if data["temporal_view_name"] != derived.view.temporal_view_name:
        raise ValueError("temporal-base adapter view name drift")
    if int(data["sequence_length"]) != derived.view.sequence_length:
        raise ValueError("temporal-base adapter sequence length drift")
    if len(derived.view.windows) != len(set(
        derived.view.windows["temporal_unit_key"].astype(str)
    )):
        raise ValueError("temporal-base adapter duplicates native units")


def _parameter_control_errors(modes: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for pair_id in ("ordered_tcn", "timed_transformer"):
        candidate, baseline = CONTROLLED_PAIRS[pair_id]
        left = int(modes[candidate]["parameter_count"])
        right = int(modes[baseline]["parameter_count"])
        relative = abs(left - right) / max(left, right)
        if relative > PARAMETER_MATCH_MAX_RELATIVE_DELTA:
            errors.append(
                f"{pair_id}:parameter_relative_delta={relative:.8f}"
            )
    return errors


def _build_model(
    adapter: LegacyL5CachedTrainingConfig,
) -> LegacyL5CachedFeatureClassifier:
    model = adapter.payload["model"]
    return LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )


def _mode_spec(mode_id: str) -> dict[str, Any]:
    try:
        return MODE_SPECS[mode_id]
    except KeyError as error:
        raise ValueError(f"unknown temporal-base mode={mode_id}") from error


def _validate_config(payload: dict[str, Any]) -> None:
    source_field = _source_spec_name(payload)
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "experiment_contract",
        source_field,
        "modes",
        "model_common",
        "optimization",
        "implementation",
        "execution",
        "output",
    }
    if payload.get("schema_version") == CONFIG_SCHEMA_V2:
        required.add("model_implementation")
    if set(payload) != required:
        raise ValueError("temporal-base config keys differ")
    if payload["schema_version"] not in {CONFIG_SCHEMA, CONFIG_SCHEMA_V2}:
        raise ValueError("temporal-base config schema drift")
    if payload["training_scope"] not in {SHORT_SCOPE, FULL_SCOPE}:
        raise ValueError("unsupported temporal-base training scope")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("temporal-base lineage drift")
    if payload["modes"] != MODE_SPECS:
        raise ValueError("temporal-base mode contract drift")
    contract = payload["experiment_contract"]
    if contract.get("changed_family") != "predeclared_temporal_base_matrix":
        raise ValueError("temporal-base changed-family drift")
    if contract.get("outer_predictions_used_for_model_selection") is not False:
        raise ValueError("outer predictions cannot select temporal base")
    if contract.get("legacy_sets_final_full_data_base") is not False:
        raise ValueError("legacy screening cannot set the full-data base")
    expected_skills = {
        "safe-refactor-test-guardian",
        "dataset-contract-leakage-guard",
        "experiment-lineage-reproducibility",
        "scientific-ablation-controller",
        "multimodal-sequence-model-builder",
        "grouped-cv-evaluation",
    }
    if set(contract.get("skills", [])) != expected_skills:
        raise ValueError("temporal-base selected-skill contract drift")
    common = payload["model_common"]
    if common.get("architecture") != "cached_frame_feature_temporal_classifier_v1":
        raise ValueError("temporal-base architecture drift")
    if common.get("feature_control_id") != "V1":
        raise ValueError("temporal-base feature control drift")
    if common.get("backbone_name") != "resnet18":
        raise ValueError("temporal-base backbone drift")
    if int(common.get("input_resolution", 0)) != 224:
        raise ValueError("temporal-base resolution drift")
    if not 0.0 <= float(common.get("dropout", -1.0)) < 1.0:
        raise ValueError("temporal-base dropout invalid")
    execution = payload["execution"]
    if execution.get("device") not in {"cpu", "cuda:0"}:
        raise ValueError("unsupported temporal-base device")
    if payload["training_scope"] == SHORT_SCOPE:
        if execution.get("required_repeats") != ["repeat01", "repeat02"]:
            raise ValueError("temporal-base short repeats drift")
    if payload["schema_version"] == CONFIG_SCHEMA_V2:
        if execution.get("data_run_authorized") is not True:
            raise ValueError("rebuild temporal-base run is not authorized")
        if not str(execution.get("clean_lineage_handoff_id", "")).strip():
            raise ValueError("rebuild temporal-base clean handoff is missing")
        if execution.get("full_oof_authorized") is not False:
            raise ValueError("rebuild temporal-base cannot authorize full OOF")
    fields = [source_field, "implementation"]
    if payload["schema_version"] == CONFIG_SCHEMA_V2:
        fields.append("model_implementation")
    for field in fields:
        spec = payload[field]
        if set(spec) != {"path", "sha256"} or not is_sha256(spec["sha256"]):
            raise ValueError(f"invalid temporal-base {field}")


def _source_spec_name(payload: dict[str, Any]) -> str:
    schema = payload.get("schema_version")
    if schema == CONFIG_SCHEMA:
        return "source_ladder_config"
    if schema == CONFIG_SCHEMA_V2:
        return "prepared_source"
    raise ValueError("temporal-base config schema drift")


def _verify_file_spec(root: Path, spec: dict[str, Any]) -> Path:
    path = (root / str(spec["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"temporal-base path escapes root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"temporal-base file hash drift: {path}")
    return path


def _string_sequence_sha256(values: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x", lineterminator="\n")


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLED_PAIRS",
    "DerivedTemporalBaseView",
    "FULL_SCOPE",
    "MODE_SPECS",
    "TemporalBaseSelectionConfig",
    "audit_temporal_base_short_matrix",
    "build_training_adapter",
    "derive_temporal_base_view",
    "execute_temporal_base_run",
    "load_temporal_base_selection_config",
    "load_temporal_base_source",
    "preflight_temporal_base_selection",
    "validate_full_launch_gate",
]
