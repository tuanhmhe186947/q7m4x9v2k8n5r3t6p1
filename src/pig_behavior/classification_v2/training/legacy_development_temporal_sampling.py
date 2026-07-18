"""Controlled legacy-16f temporal sampling over the frozen T16 feature view."""

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
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
    build_temporal_ladder_selection,
    load_temporal_ladder_config,
    load_temporal_ladder_view,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)

CONFIG_SCHEMA = (
    "classification_v2.legacy_development.temporal_sampling_config.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development.temporal_sampling_preflight.v1"
)
RUN_SCHEMA = "classification_v2.legacy_development.temporal_sampling_run.v1"
MATRIX_SCHEMA = (
    "classification_v2.legacy_development.temporal_sampling_short_matrix.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
SOURCE_VIEW_ID = "t16_centered"
SOURCE_SEQUENCE_LENGTH = 16
MODEL_PARAMETER_COUNT = 68_234
SHORT_SCOPE = "short_repeat_gate"
FULL_SCOPE = "full_development_confirmation"
SOURCE_FULL_SCOPE = "full_development_baseline"

VIEW_SPECS: dict[str, dict[str, Any]] = {
    "c6_contiguous_centered": {
        "temporal_view_name": "legacy_c6_contiguous_centered_span6_v1",
        "sampling_protocol": "one_contiguous_centered_sequence_per_native",
        "sequence_length": 6,
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "temporal_span_frames": 6,
        "historical_alignment": False,
    },
    "c8_contiguous_centered": {
        "temporal_view_name": "legacy_c8_contiguous_centered_span8_v1",
        "sampling_protocol": "one_contiguous_centered_sequence_per_native",
        "sequence_length": 8,
        "native_frame_offsets": [4, 5, 6, 7, 8, 9, 10, 11],
        "temporal_span_frames": 8,
        "historical_alignment": False,
    },
    "s6_uniform_span16": {
        "temporal_view_name": "legacy_s6_uniform_offsets_0_3_6_9_12_15_v1",
        "sampling_protocol": "one_uniform_span16_sequence_per_native",
        "sequence_length": 6,
        "native_frame_offsets": [0, 3, 6, 9, 12, 15],
        "temporal_span_frames": 16,
        "historical_alignment": True,
    },
}


@dataclass(frozen=True, slots=True)
class TemporalSamplingConfig:
    """Immutable one-sequence-per-native temporal sampling contract."""

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
        return self.repo_root / str(self.payload["source_ladder_config"]["path"])

    @property
    def output_root(self) -> Path:
        return self.repo_root / str(self.payload["output"]["root"])


@dataclass(frozen=True, slots=True)
class TemporalSamplingSource:
    """Frozen T16 parent plus one native-first selection."""

    base_view: LegacyL5CachedFeatureView
    selection: TemporalLadderSelection
    parent_audit: dict[str, Any]
    source_config_sha256: str


@dataclass(frozen=True, slots=True)
class DerivedTemporalSamplingView:
    """Offset-selected cached view with an explicit slot audit."""

    view_id: str
    view: LegacyL5CachedFeatureView
    slot_manifest: pd.DataFrame
    audit: dict[str, Any]


def load_temporal_sampling_config(path: Path) -> TemporalSamplingConfig:
    """Load and fail closed on any semantic or hash drift."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_config(payload)
    repo_root = resolved.parents[2]
    config = TemporalSamplingConfig(
        path=resolved,
        payload=payload,
        repo_root=repo_root,
    )
    source = config.source_config_path
    if not source.is_file():
        raise FileNotFoundError(f"source ladder config missing: {source}")
    if file_sha256(source) != payload["source_ladder_config"]["sha256"]:
        raise ValueError("source ladder config hash drift")
    implementation = payload["implementation"]
    implementation_path = repo_root / str(implementation["path"])
    if not implementation_path.is_file():
        raise FileNotFoundError(
            f"temporal sampling implementation missing: {implementation_path}"
        )
    if file_sha256(implementation_path) != implementation["sha256"]:
        raise ValueError("temporal sampling implementation hash drift")
    return config


def load_temporal_sampling_source(
    config: TemporalSamplingConfig,
) -> TemporalSamplingSource:
    """Load the already-audited T16 view without media reads."""

    source_config = load_temporal_ladder_config(config.source_config_path)
    expected_source_scope = (
        SHORT_SCOPE if config.training_scope == SHORT_SCOPE else SOURCE_FULL_SCOPE
    )
    if source_config.training_scope != expected_source_scope:
        raise ValueError("source and temporal-sampling training scopes differ")
    _, base_view, parent = load_temporal_ladder_view(
        source_config,
        SOURCE_VIEW_ID,
    )
    selection = build_temporal_ladder_selection(
        base_view,
        source_config,
        SOURCE_VIEW_ID,
    )
    _validate_source_view(base_view, selection, config)
    return TemporalSamplingSource(
        base_view=base_view,
        selection=selection,
        parent_audit=parent,
        source_config_sha256=source_config.sha256,
    )


def derive_temporal_sampling_view(
    base_view: LegacyL5CachedFeatureView,
    view_id: str,
) -> DerivedTemporalSamplingView:
    """Select native-frame offsets and recompute observed elapsed-time deltas."""

    if view_id not in VIEW_SPECS:
        raise ValueError(f"unknown temporal sampling view={view_id}")
    _validate_base_t16_view(base_view)
    spec = VIEW_SPECS[view_id]
    offsets = np.asarray(spec["native_frame_offsets"], dtype=np.int64)
    selected_rows = base_view.feature_rows[:, offsets].copy()
    selected_mask = base_view.observed_mask[:, offsets].copy()
    base_delta = np.asarray(base_view.time_delta, dtype=np.float64)
    elapsed = np.cumsum(base_delta, axis=1)
    selected_elapsed = elapsed[:, offsets]
    selected_delta = np.zeros_like(selected_elapsed, dtype=np.float32)
    selected_delta[:, 1:] = np.diff(selected_elapsed, axis=1).astype(np.float32)
    if not selected_mask.all():
        raise ValueError(f"{view_id} contains unavailable selected slots")
    if not np.isfinite(selected_delta).all():
        raise ValueError(f"{view_id} contains nonfinite selected timing")
    if (selected_delta[:, 1:] <= 0.0).any():
        raise ValueError(f"{view_id} contains nonpositive elapsed deltas")

    windows = base_view.windows.copy(deep=True)
    windows["temporal_sampling_view_id"] = view_id
    windows["temporal_sampling_offsets_json"] = json.dumps(
        spec["native_frame_offsets"],
        separators=(",", ":"),
    )
    if windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("derived temporal view is not one row per native unit")
    derived_audit = {
        "schema_version": "classification_v2.temporal_sampling_view_audit.v1",
        "view_id": view_id,
        "temporal_view_name": spec["temporal_view_name"],
        "sampling_protocol": spec["sampling_protocol"],
        "native_frame_offsets": list(spec["native_frame_offsets"]),
        "sequence_length": int(spec["sequence_length"]),
        "temporal_span_frames": int(spec["temporal_span_frames"]),
        "base_temporal_view_name": base_view.temporal_view_name,
        "base_sequence_length": base_view.sequence_length,
        "model_windows": int(len(windows)),
        "train_native_units": int((windows["l5_role"] == "train").sum()),
        "validation_native_units": int(
            (windows["l5_role"] == "validation").sum()
        ),
        "selected_slot_rows": int(selected_rows.size),
        "rows_dropped": 0,
        "labels_changed": 0,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "availability_is_behavior_evidence": False,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "errors": [],
        "valid": True,
    }
    audit = copy.deepcopy(base_view.audit)
    audit["derived_temporal_sampling"] = derived_audit
    audit["temporal_view_name"] = spec["temporal_view_name"]
    audit["sequence_length"] = int(spec["sequence_length"])
    view = replace(
        base_view,
        temporal_view_name=str(spec["temporal_view_name"]),
        sequence_length=int(spec["sequence_length"]),
        windows=windows,
        feature_rows=selected_rows,
        observed_mask=selected_mask,
        time_delta=selected_delta,
        sample_weights=np.ones(len(windows), dtype=np.float64),
        audit=audit,
    )
    slot_manifest = _build_slot_manifest(view, offsets)
    return DerivedTemporalSamplingView(
        view_id=view_id,
        view=view,
        slot_manifest=slot_manifest,
        audit=derived_audit,
    )


def build_training_adapter(
    config: TemporalSamplingConfig,
    source: TemporalSamplingSource,
    derived: DerivedTemporalSamplingView,
) -> tuple[LegacyL5CachedTrainingConfig, LegacyL5CachedShortSelection]:
    """Adapt the frozen trainer without changing its model or optimization."""

    spec = VIEW_SPECS[derived.view_id]
    train_count = len(source.selection.train_positions)
    validation_count = len(source.selection.validation_positions)
    optimization = copy.deepcopy(config.payload["optimization"])
    steps_per_epoch = (train_count + int(optimization["batch_size"]) - 1) // int(
        optimization["batch_size"]
    )
    optimization["maximum_optimizer_steps"] = steps_per_epoch * int(
        optimization["epochs"]
    )
    adapter_payload = {
        "schema_version": CONFIG_SCHEMA,
        "training_scope": config.training_scope,
        "data": {
            "control_id": derived.view.control_id,
            "temporal_view_name": spec["temporal_view_name"],
            "sampling_protocol": spec["sampling_protocol"],
            "sequence_length": spec["sequence_length"],
            "feature_dim": FEATURE_DIM,
            "model_visible_roles": ["train", "validation"],
            "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
            "train_selection_policy": "frozen_l5_native_selection_v1",
            "train_selection_salt": "inherited_from_source_ladder_config",
            "train_native_units_per_class": (
                8 if config.training_scope == SHORT_SCOPE else None
            ),
            "expected_train_native_units": train_count,
            "validation_selection_policy": "all_validation_native_units_v1",
            "expected_validation_native_units": validation_count,
            "native_prediction_aggregation": "one_sequence_per_native_unit",
        },
        "model": copy.deepcopy(config.payload["model"]),
        "optimization": optimization,
    }
    adapter = LegacyL5CachedTrainingConfig(
        path=config.path,
        payload=adapter_payload,
        repo_root=config.repo_root,
    )
    manifest = source.selection.manifest.copy(deep=True)
    manifest["temporal_sampling_view_id"] = derived.view_id
    manifest["temporal_sampling_offsets_json"] = json.dumps(
        spec["native_frame_offsets"],
        separators=(",", ":"),
    )
    selection_hash = cached_engine._dataframe_sha256(manifest)
    selection_audit = copy.deepcopy(source.selection.audit)
    selection_audit.update(
        {
            "training_scope": config.training_scope,
            "view_id": derived.view_id,
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
    _validate_adapter(derived, selection, adapter)
    return adapter, selection


def preflight_temporal_sampling(
    config: TemporalSamplingConfig,
) -> dict[str, Any]:
    """Run CPU-only parent, offset, timing, shape, and paired-universe gates."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    view_audits: dict[str, dict[str, Any]] = {}
    common_units: list[str] | None = None
    try:
        source = load_temporal_sampling_source(config)
        for view_id in VIEW_SPECS:
            derived = derive_temporal_sampling_view(source.base_view, view_id)
            adapter, selection = build_training_adapter(
                config,
                source,
                derived,
            )
            sample = selection.train_positions[:64]
            batch = derived.view.load_sequences(sample)
            model = _build_model(adapter)
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(batch),
                    torch.from_numpy(derived.view.observed_mask[sample]).float(),
                    time_delta=torch.from_numpy(
                        derived.view.time_delta[sample]
                    ).float(),
                )
            parameters = sum(parameter.numel() for parameter in model.parameters())
            if parameters != MODEL_PARAMETER_COUNT:
                errors.append(f"{view_id}:model_parameter_count={parameters}")
            if list(logits.shape) != [len(sample), len(VALID_BEHAVIORS)]:
                errors.append(f"{view_id}:forward_shape={list(logits.shape)}")
            units = derived.view.windows["temporal_unit_key"].astype(str).tolist()
            if common_units is None:
                common_units = units
            elif units != common_units:
                errors.append(f"{view_id}:native_unit_order_drift")
            view_audits[view_id] = {
                **derived.audit,
                "slot_manifest_sha256": cached_engine._dataframe_sha256(
                    derived.slot_manifest
                ),
                "selection_sha256": selection.audit[
                    "selection_content_sha256"
                ],
                "parameter_count": parameters,
                "forward_shape": list(logits.shape),
            }
            del logits, model, batch
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("CPU preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_TEMPORAL_SAMPLING_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_TEMPORAL_SAMPLING_PREFLIGHT"
        ),
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "source_ladder_config_sha256": config.payload[
            "source_ladder_config"
        ]["sha256"],
        "view_audits": view_audits,
        "native_units": len(common_units or []),
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "short_run_authorized": valid and config.training_scope == SHORT_SCOPE,
        "errors": errors,
        "valid": valid,
    }


def execute_temporal_sampling_run(
    config: TemporalSamplingConfig,
    view_id: str,
    repeat_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Execute one isolated deterministic process and write hash-linked evidence."""

    if view_id not in VIEW_SPECS:
        raise ValueError(f"unknown temporal sampling view={view_id}")
    if not repeat_id or any(character in repeat_id for character in "\\/:*?\"<>|"):
        raise ValueError("repeat_id is blank or unsafe")
    run_root = config.output_root / config.training_scope / view_id / repeat_id
    run_root.mkdir(parents=True, exist_ok=False)
    started_wall = time.time()
    started_utc = _utc_now()
    planned = {
        "schema_version": "classification_v2.temporal_sampling_planned_run.v1",
        "status": "planned",
        "view_id": view_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "started_at_utc": started_utc,
        "started_wall_time": started_wall,
        "git_state": git_state(),
    }
    _write_json_exclusive(run_root / "planned_run_manifest.json", planned)
    try:
        preflight = preflight_temporal_sampling(config)
        if not preflight["valid"]:
            raise RuntimeError(f"preflight failed: {preflight['errors']}")
        source = load_temporal_sampling_source(config)
        derived = derive_temporal_sampling_view(source.base_view, view_id)
        adapter, selection = build_training_adapter(config, source, derived)
        device = str(config.payload["execution"]["device"])
        outcome = train_legacy_l5_cached_short_core(
            derived.view,
            selection,
            adapter,
            device=device,
        )
        completed_utc = _utc_now()
        runtime_seconds = time.time() - started_wall
        result = _write_success_artifacts(
            config,
            run_root=run_root,
            view_id=view_id,
            repeat_id=repeat_id,
            source=source,
            derived=derived,
            selection=selection,
            outcome=outcome,
            planned=planned,
            preflight=preflight,
            started_utc=started_utc,
            completed_utc=completed_utc,
            runtime_seconds=runtime_seconds,
            device=device,
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


def audit_temporal_sampling_short_matrix(
    config: TemporalSamplingConfig,
) -> tuple[Path, dict[str, Any]]:
    """Require two exact, fresh, non-overlapping repeats for every view."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("short matrix requires short_repeat_gate config")
    required_repeats = list(config.payload["execution"]["required_repeats"])
    errors: list[str] = []
    views: dict[str, Any] = {}
    processes: list[int] = []
    intervals: list[tuple[float, float]] = []
    common_native_hash: str | None = None
    for view_id in VIEW_SPECS:
        packets = []
        for repeat_id in required_repeats:
            path = (
                config.output_root
                / config.training_scope
                / view_id
                / str(repeat_id)
                / "run_result.json"
            )
            if not path.is_file():
                errors.append(f"missing_run_result={view_id}:{repeat_id}")
                continue
            packet = json.loads(path.read_text(encoding="utf-8"))
            packets.append(packet)
            processes.append(int(packet["process_id"]))
            intervals.append(
                (float(packet["started_wall_time"]), float(packet["ended_wall_time"]))
            )
        if len(packets) != len(required_repeats):
            continue
        exact_fields = (
            "config_sha256",
            "implementation_sha256",
            "source_config_sha256",
            "view_slot_manifest_sha256",
            "selection_native_unit_sha256",
            "parameter_sha256",
            "prediction_content_sha256",
            "epoch_metrics_content_sha256",
        )
        mismatches = [
            field
            for field in exact_fields
            if len({str(packet[field]) for packet in packets}) != 1
        ]
        if mismatches:
            errors.append(f"{view_id}:repeat_mismatch={mismatches}")
        native_hash = str(packets[0]["selection_native_unit_sha256"])
        if common_native_hash is None:
            common_native_hash = native_hash
        elif native_hash != common_native_hash:
            errors.append(f"{view_id}:paired_native_universe_drift")
        views[view_id] = {
            "repeat_ids": required_repeats,
            "process_ids": [int(packet["process_id"]) for packet in packets],
            "exact_repeat_fields": list(exact_fields),
            "repeat_mismatches": mismatches,
            "metrics": packets[0]["metrics"],
            "result_sha256": file_sha256(
                config.output_root
                / config.training_scope
                / view_id
                / str(required_repeats[0])
                / "run_result.json"
            ),
        }
    if len(processes) != len(set(processes)):
        errors.append("repeat_process_ids_are_not_distinct")
    ordered = sorted(intervals)
    if any(
        right[0] < left[1]
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        errors.append("repeat_execution_intervals_overlap")
    valid = not errors and len(views) == len(VIEW_SPECS)
    payload = {
        "schema_version": MATRIX_SCHEMA,
        "status": (
            "PASS_LEGACY_TEMPORAL_SAMPLING_SHORT_MATRIX"
            if valid
            else "FAIL_LEGACY_TEMPORAL_SAMPLING_SHORT_MATRIX"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "changed_scientific_family": "temporal_sampling_pattern_and_length_matrix",
        "one_sequence_per_native_unit": True,
        "views": views,
        "common_native_unit_sha256": common_native_hash,
        "full_confirmation_authorized": valid,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }
    output = config.output_root / "temporal_sampling_short_matrix.json"
    _write_json_exclusive(output, payload)
    return output, payload


def _write_success_artifacts(
    config: TemporalSamplingConfig,
    *,
    run_root: Path,
    view_id: str,
    repeat_id: str,
    source: TemporalSamplingSource,
    derived: DerivedTemporalSamplingView,
    selection: LegacyL5CachedShortSelection,
    outcome: LegacyL5CachedTrainingOutcome,
    planned: dict[str, Any],
    preflight: dict[str, Any],
    started_utc: str,
    completed_utc: str,
    runtime_seconds: float,
    device: str,
) -> dict[str, Any]:
    predictions = outcome.predictions.copy()
    predictions["temporal_sampling_view_id"] = view_id
    per_class = outcome.per_class_metrics.copy()
    per_class["temporal_sampling_view_id"] = view_id
    confusion = outcome.confusion.copy()
    confusion["temporal_sampling_view_id"] = view_id
    epoch_metrics = outcome.epoch_metrics.copy()
    epoch_metrics["temporal_sampling_view_id"] = view_id
    _write_dataframe_exclusive(run_root / "validation_native_predictions.csv", predictions)
    _write_dataframe_exclusive(run_root / "metrics_per_class.csv", per_class)
    _write_dataframe_exclusive(run_root / "confusion_matrix.csv", confusion)
    _write_dataframe_exclusive(run_root / "epoch_metrics.csv", epoch_metrics)
    _write_dataframe_exclusive(run_root / "derived_slot_manifest.csv", derived.slot_manifest)
    metrics = {
        **outcome.metrics,
        "temporal_sampling_view_id": view_id,
        "temporal_view_name": derived.view.temporal_view_name,
        "native_frame_offsets": VIEW_SPECS[view_id]["native_frame_offsets"],
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
    }
    _write_json_exclusive(run_root / "metrics_global.json", metrics)
    checkpoint = {
        "schema_version": "classification_v2.temporal_sampling_checkpoint.v1",
        "config_sha256": config.sha256,
        "view_id": view_id,
        "view_spec": VIEW_SPECS[view_id],
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
        "schema_version": (
            "classification_v2.temporal_sampling_checkpoint_manifest.v1"
        ),
        "view_id": view_id,
        "repeat_id": repeat_id,
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
        "schema_version": (
            "classification_v2.temporal_sampling_prediction_manifest.v1"
        ),
        "view_id": view_id,
        "repeat_id": repeat_id,
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
        "schema_version": "classification_v2.temporal_sampling_environment.v1",
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
    artifact_rows = [
        {
            "name": name,
            "path": str((run_root / name).resolve()),
            "sha256": file_sha256(run_root / name),
            "size_bytes": int((run_root / name).stat().st_size),
        }
        for name in artifact_names
    ]
    artifact_manifest = {
        "schema_version": "classification_v2.temporal_sampling_artifacts.v1",
        "view_id": view_id,
        "repeat_id": repeat_id,
        "artifacts": artifact_rows,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "artifact_manifest.json", artifact_manifest)
    native_keys = derived.view.windows.iloc[
        selection.validation_positions
    ]["temporal_unit_key"].astype(str)
    native_hash = _string_sequence_sha256(native_keys.tolist())
    ended_wall = time.time()
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "view_id": view_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "started_at_utc": started_utc,
        "completed_at_utc": completed_utc,
        "runtime_seconds": runtime_seconds,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_config_sha256": source.source_config_sha256,
        "source_parent_audit": source.parent_audit,
        "git_state": planned["git_state"],
        "view_spec": VIEW_SPECS[view_id],
        "view_audit": derived.audit,
        "selection_audit": selection.audit,
        "selection_native_unit_sha256": native_hash,
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "epoch_metrics_content_sha256": outcome.epoch_metrics_sha256,
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "artifact_manifest_sha256": file_sha256(
            run_root / "artifact_manifest.json"
        ),
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "run_manifest.json", run_manifest)
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "view_id": view_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "started_at_utc": started_utc,
        "completed_at_utc": completed_utc,
        "started_wall_time": float(planned["started_wall_time"]),
        "ended_wall_time": ended_wall,
        "runtime_seconds": runtime_seconds,
        "config_sha256": config.sha256,
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_config_sha256": source.source_config_sha256,
        "view_slot_manifest_sha256": cached_engine._dataframe_sha256(
            derived.slot_manifest
        ),
        "selection_native_unit_sha256": native_hash,
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "epoch_metrics_content_sha256": outcome.epoch_metrics_sha256,
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
    offsets: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, row in enumerate(view.windows.itertuples(index=False)):
        for slot_index, offset in enumerate(offsets):
            records.append(
                {
                    "temporal_sampling_view_id": str(
                        row.temporal_sampling_view_id
                    ),
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
        raise ValueError("derived slot manifest row count drift")
    if frame[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("derived slot manifest contains duplicate slots")
    return frame


def _validate_source_view(
    base_view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalSamplingConfig,
) -> None:
    _validate_base_t16_view(base_view)
    expected_train = 80 if config.training_scope == SHORT_SCOPE else 3_652
    if len(selection.train_positions) != expected_train:
        raise ValueError("source train native count drift")
    if len(selection.validation_positions) != 245:
        raise ValueError("source validation native count drift")
    if selection.audit.get("outer_holdout_rows") != 0:
        raise ValueError("source selection exposes outer holdout")


def _validate_base_t16_view(base_view: LegacyL5CachedFeatureView) -> None:
    if base_view.sequence_length != SOURCE_SEQUENCE_LENGTH:
        raise ValueError("temporal sampling parent is not T16")
    if base_view.feature_rows.ndim != 2 or base_view.feature_rows.shape[1] != 16:
        raise ValueError("T16 feature-row shape drift")
    if base_view.observed_mask.shape != base_view.feature_rows.shape:
        raise ValueError("T16 observed-mask shape drift")
    if base_view.time_delta.shape != base_view.feature_rows.shape:
        raise ValueError("T16 timing shape drift")
    if not base_view.observed_mask.all():
        raise ValueError("T16 parent contains unavailable slots")
    if base_view.windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("T16 parent is not one row per native unit")


def _validate_adapter(
    derived: DerivedTemporalSamplingView,
    selection: LegacyL5CachedShortSelection,
    adapter: LegacyL5CachedTrainingConfig,
) -> None:
    data = adapter.payload["data"]
    if derived.view.temporal_view_name != data["temporal_view_name"]:
        raise ValueError("adapter temporal-view name drift")
    if derived.view.sequence_length != int(data["sequence_length"]):
        raise ValueError("adapter sequence length drift")
    if len(selection.train_positions) != int(data["expected_train_native_units"]):
        raise ValueError("adapter train count drift")
    if len(selection.validation_positions) != int(
        data["expected_validation_native_units"]
    ):
        raise ValueError("adapter validation count drift")


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


def _validate_config(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "experiment_contract",
        "source_ladder_config",
        "views",
        "model",
        "optimization",
        "implementation",
        "execution",
        "output",
    }
    if set(payload) != required:
        raise ValueError(
            f"temporal sampling config keys={sorted(payload)} expected={sorted(required)}"
        )
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("temporal sampling config schema drift")
    if payload["training_scope"] not in {SHORT_SCOPE, FULL_SCOPE}:
        raise ValueError("unsupported temporal sampling training scope")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("temporal sampling lineage drift")
    if payload["views"] != VIEW_SPECS:
        raise ValueError("temporal sampling view contract drift")
    for field in ("source_ladder_config", "implementation"):
        value = payload[field]
        if set(value) != {"path", "sha256"} or not is_sha256(value["sha256"]):
            raise ValueError(f"invalid temporal sampling {field}")
    experiment = payload["experiment_contract"]
    if experiment.get("changed_family") != "temporal_sampling_pattern_and_length":
        raise ValueError("temporal sampling changed-family drift")
    if experiment.get("native_evaluation_unit") != "complete_16_frame_burst":
        raise ValueError("temporal sampling evaluation grain drift")
    if experiment.get("outer_predictions_used_for_model_selection") is not False:
        raise ValueError("outer predictions cannot select temporal sampling")
    execution = payload["execution"]
    if execution.get("device") not in {"cpu", "cuda:0"}:
        raise ValueError("unsupported temporal sampling device")
    repeats = execution.get("required_repeats")
    if payload["training_scope"] == SHORT_SCOPE and repeats != ["repeat01", "repeat02"]:
        raise ValueError("short temporal sampling repeats drift")


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
    "DerivedTemporalSamplingView",
    "TemporalSamplingConfig",
    "TemporalSamplingSource",
    "VIEW_SPECS",
    "audit_temporal_sampling_short_matrix",
    "build_training_adapter",
    "derive_temporal_sampling_view",
    "execute_temporal_sampling_run",
    "load_temporal_sampling_config",
    "load_temporal_sampling_source",
    "preflight_temporal_sampling",
]
