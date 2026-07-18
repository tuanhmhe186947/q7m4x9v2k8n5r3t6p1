"""Fail-closed legacy C6 runner for temporal perturbation controls."""

from __future__ import annotations

import copy
import hashlib
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
from pig_behavior.classification_v2.training.legacy_development_l5 import git_state
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
from pig_behavior.classification_v2.training.legacy_development_temporal_base_selection import (
    derive_temporal_base_view,
    load_temporal_base_selection_config,
    load_temporal_base_source,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)
from pig_behavior.classification_v2.training.temporal_perturbation_controls import (
    CONTROLLED_PAIRS,
    MODE_SPECS,
    TemporalControlPlan,
    apply_slot_permutation,
    audit_time_delta_identifiability,
    audit_timing_source_shortcut,
    build_temporal_conclusion_readiness,
    build_temporal_control_plan,
    parameter_control_errors,
)

CONFIG_SCHEMA = "classification_v2.legacy_c6_temporal_controls_config.v1"
PREFLIGHT_SCHEMA = "classification_v2.legacy_c6_temporal_controls_preflight.v1"
RUN_SCHEMA = "classification_v2.legacy_c6_temporal_controls_run.v1"
SHORT_GATE_SCHEMA = "classification_v2.legacy_c6_temporal_controls_short_gate.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
SHORT_SCOPE = "short_repeat_gate"
FULL_SCOPE = "full_development_confirmation"
SELECTED_SKILLS = (
    "safe-refactor-test-guardian",
    "dataset-contract-leakage-guard",
    "experiment-lineage-reproducibility",
    "scientific-ablation-controller",
    "multimodal-sequence-model-builder",
    "grouped-cv-evaluation",
)


@dataclass(frozen=True, slots=True)
class C6TemporalControlConfig:
    """Hash-bound code-ready or authorized temporal-control configuration."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    @property
    def output_root(self) -> Path:
        return self.repo_root / str(self.payload["output"]["root"])

    @property
    def training_scope(self) -> str:
        return str(self.payload["training_scope"])

    def bound_path(self, field: str) -> Path:
        return self.repo_root / str(self.payload[field]["path"])


@dataclass(frozen=True, slots=True)
class DerivedC6TemporalControl:
    """One temporal control over the same C6 native-unit universe."""

    mode_id: str
    view: LegacyL5CachedFeatureView
    plan: TemporalControlPlan
    slot_manifest: pd.DataFrame
    audit: dict[str, Any]


def load_c6_temporal_control_config(path: Path) -> C6TemporalControlConfig:
    """Load an exact config without reading project data."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_config(payload)
    config = C6TemporalControlConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    _verify_file_spec(config.repo_root, payload["source_temporal_config"])
    _verify_file_spec(config.repo_root, payload["implementation"])
    _verify_file_spec(config.repo_root, payload["control_implementation"])
    _verify_file_spec(config.repo_root, payload["launcher"])
    return config


def derive_c6_temporal_control(
    base_view: LegacyL5CachedFeatureView,
    mode_id: str,
    *,
    perturbation_seed: int,
    constant_delta_seconds: float,
) -> DerivedC6TemporalControl:
    """Apply a label-independent control after deriving exact C6 offsets."""

    c6 = derive_temporal_base_view(base_view, "M128").view
    keys = c6.windows["temporal_unit_key"].astype(str).tolist()
    plan = build_temporal_control_plan(
        mode_id=mode_id,
        unit_keys=keys,
        observed_mask=c6.observed_mask,
        real_time_delta=c6.time_delta,
        perturbation_seed=perturbation_seed,
        constant_delta_seconds=constant_delta_seconds,
    )
    feature_rows = apply_slot_permutation(c6.feature_rows, plan)
    windows = c6.windows.copy(deep=True)
    windows["c6_temporal_control_mode_id"] = mode_id
    windows["sequence_control"] = MODE_SPECS[mode_id]["sequence_control"]
    windows["time_delta_control"] = MODE_SPECS[mode_id]["time_delta_control"]
    if windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("C6 temporal control duplicates native units")
    view_name = f"legacy_c6_{mode_id.lower()}_v1"
    audit = {
        "schema_version": "classification_v2.legacy_c6_control_view.v1",
        "mode_id": mode_id,
        "temporal_view_name": view_name,
        "native_frame_offsets": [5, 6, 7, 8, 9, 10],
        "native_units": int(len(windows)),
        "sequence_length": 6,
        "rows_dropped": 0,
        "rows_duplicated": 0,
        "labels_changed": 0,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "one_sequence_per_native_unit": True,
        "plan_audit": copy.deepcopy(plan.audit),
        "human_review_complete": False,
        "legacy_sets_full_data_base": False,
        "errors": [],
        "valid": True,
    }
    inherited = copy.deepcopy(c6.audit)
    inherited["derived_c6_temporal_control"] = audit
    inherited["temporal_view_name"] = view_name
    view = replace(
        c6,
        temporal_view_name=view_name,
        windows=windows,
        feature_rows=feature_rows,
        time_delta=plan.controlled_time_delta,
        sample_weights=np.ones(len(windows), dtype=np.float64),
        audit=inherited,
    )
    slot_manifest = _build_slot_manifest(c6, view, plan)
    return DerivedC6TemporalControl(
        mode_id=mode_id,
        view=view,
        plan=plan,
        slot_manifest=slot_manifest,
        audit=audit,
    )


def static_c6_temporal_control_preflight(
    config: C6TemporalControlConfig,
) -> dict[str, Any]:
    """Validate code/config/model contracts without reading project data."""

    errors: list[str] = []
    parameter_counts: dict[str, int] = {}
    for mode_id in MODE_SPECS:
        model = build_c6_temporal_control_model(config, mode_id, dropout=0.0)
        observed = sum(parameter.numel() for parameter in model.parameters())
        parameter_counts[mode_id] = observed
        expected = int(MODE_SPECS[mode_id]["expected_parameter_count"])
        if observed != expected:
            errors.append(f"{mode_id}:parameter_count={observed},expected={expected}")
    errors.extend(parameter_control_errors(parameter_counts))
    valid = not errors and set(parameter_counts) == set(MODE_SPECS)
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_C6_TEMPORAL_CONTROLS_STATIC"
            if valid
            else "FAIL_C6_TEMPORAL_CONTROLS_STATIC"
        ),
        "config_sha256": config.sha256,
        "selected_skills": list(SELECTED_SKILLS),
        "modes": copy.deepcopy(MODE_SPECS),
        "controlled_pairs": {
            name: {"candidate": pair[0], "control": pair[1]}
            for name, pair in CONTROLLED_PAIRS.items()
        },
        "parameter_counts": parameter_counts,
        "data_run_authorized": bool(
            config.payload["execution"]["data_run_authorized"]
        ),
        "project_data_rows_read": 0,
        "project_data_optimizer_steps": 0,
        "full_development_authorized": False,
        "full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def synthetic_c6_temporal_control_preflight(
    config: C6TemporalControlConfig,
) -> dict[str, Any]:
    """Exercise every transform and model with synthetic variable timing."""

    rng = np.random.default_rng(20260717)
    keys = [f"synthetic-unit-{index}" for index in range(8)]
    mask = np.ones((len(keys), 6), dtype=np.bool_)
    deltas = np.zeros((len(keys), 6), dtype=np.float32)
    deltas[:, 1:] = rng.uniform(0.11, 0.31, size=(len(keys), 5))
    features = rng.normal(size=(len(keys), 6, FEATURE_DIM)).astype(np.float32)
    spatial = rng.normal(size=(len(keys), 6, 7)).astype(np.float32)
    errors: list[str] = []
    modes: dict[str, Any] = {}
    for mode_id in MODE_SPECS:
        plan = build_temporal_control_plan(
            mode_id=mode_id,
            unit_keys=keys,
            observed_mask=mask,
            real_time_delta=deltas,
            perturbation_seed=int(config.payload["controls"]["perturbation_seed"]),
            constant_delta_seconds=float(
                config.payload["controls"]["constant_delta_seconds"]
            ),
        )
        actor = apply_slot_permutation(features, plan)
        aligned_spatial = apply_slot_permutation(spatial, plan)
        if not np.array_equal(
            plan.slot_permutation,
            build_temporal_control_plan(
                mode_id=mode_id,
                unit_keys=keys,
                observed_mask=mask,
                real_time_delta=deltas,
                perturbation_seed=int(
                    config.payload["controls"]["perturbation_seed"]
                ),
                constant_delta_seconds=float(
                    config.payload["controls"]["constant_delta_seconds"]
                ),
            ).slot_permutation,
        ):
            errors.append(f"{mode_id}:nondeterministic_permutation")
        model = build_c6_temporal_control_model(config, mode_id, dropout=0.0)
        model.train()
        input_tensor = torch.from_numpy(actor)
        logits = model(
            input_tensor,
            torch.from_numpy(mask).float(),
            time_delta=torch.from_numpy(plan.controlled_time_delta).float(),
        )
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.arange(len(keys), dtype=torch.long) % len(VALID_BEHAVIORS),
        )
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if list(logits.shape) != [len(keys), len(VALID_BEHAVIORS)]:
            errors.append(f"{mode_id}:forward_shape={list(logits.shape)}")
        if not torch.isfinite(logits).all() or not finite_gradients:
            errors.append(f"{mode_id}:nonfinite_forward_or_backward")
        modes[mode_id] = {
            **copy.deepcopy(plan.audit),
            "actor_shape": list(actor.shape),
            "aligned_spatial_shape": list(aligned_spatial.shape),
            "forward_shape": list(logits.shape),
            "backward_finite": finite_gradients,
        }
    delta_audit = audit_time_delta_identifiability(
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=int(config.payload["controls"]["perturbation_seed"]),
        constant_delta_seconds=float(
            config.payload["controls"]["constant_delta_seconds"]
        ),
        minimum_changed_fraction=float(
            config.payload["controls"]["minimum_changed_fraction"]
        ),
    )
    source_audit = audit_timing_source_shortcut(
        source_types=["legacy", "cvat"] * 4,
        observed_mask=mask,
        real_time_delta=deltas,
    )
    readiness = build_temporal_conclusion_readiness(
        delta_audit=delta_audit,
        source_audit=source_audit,
        short_gate_passed=False,
        paired_native_evidence_passed=False,
        per_source_evidence_passed=False,
        seed_robustness_passed=False,
        mixed_reviewed_lineage=False,
    )
    if readiness["full_data_base_promotion_allowed"]:
        errors.append("synthetic smoke authorized full-data promotion")
    valid = not errors and set(modes) == set(MODE_SPECS)
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_C6_TEMPORAL_CONTROLS_SYNTHETIC"
            if valid
            else "FAIL_C6_TEMPORAL_CONTROLS_SYNTHETIC"
        ),
        "config_sha256": config.sha256,
        "modes": modes,
        "delta_identifiability": delta_audit,
        "timing_source_shortcut": source_audit,
        "conclusion_readiness": readiness,
        "project_data_rows_read": 0,
        "project_data_optimizer_steps": 0,
        "full_development_authorized": False,
        "full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def data_c6_temporal_control_preflight(
    config: C6TemporalControlConfig,
) -> tuple[Any, dict[str, Any]]:
    """Audit an explicitly authorized clean handoff before any optimizer step."""

    _require_data_authorization(config)
    full_gate = _validate_full_development_launch(config)
    source_config = load_temporal_base_selection_config(
        config.bound_path("source_temporal_config")
    )
    source = load_temporal_base_source(source_config)
    base = derive_temporal_base_view(source.base_view, "M128").view
    keys = base.windows["temporal_unit_key"].astype(str).tolist()
    controls = config.payload["controls"]
    delta_audit = audit_time_delta_identifiability(
        unit_keys=keys,
        observed_mask=base.observed_mask,
        real_time_delta=base.time_delta,
        perturbation_seed=int(controls["perturbation_seed"]),
        constant_delta_seconds=float(controls["constant_delta_seconds"]),
        minimum_changed_fraction=float(controls["minimum_changed_fraction"]),
    )
    source_audit = audit_timing_source_shortcut(
        source_types=base.windows["source_type"].astype(str).tolist(),
        observed_mask=base.observed_mask,
        real_time_delta=base.time_delta,
    )
    modes: dict[str, Any] = {}
    native_keys: list[str] | None = None
    errors: list[str] = []
    for mode_id in MODE_SPECS:
        derived = derive_c6_temporal_control(
            source.base_view,
            mode_id,
            perturbation_seed=int(controls["perturbation_seed"]),
            constant_delta_seconds=float(controls["constant_delta_seconds"]),
        )
        current_keys = derived.view.windows["temporal_unit_key"].astype(str).tolist()
        if native_keys is None:
            native_keys = current_keys
        elif current_keys != native_keys:
            errors.append(f"{mode_id}:native_unit_order_drift")
        modes[mode_id] = copy.deepcopy(derived.audit)
    timing_modes_authorized = bool(
        delta_audit["full_real_timing_claim_identifiable"]
    )
    valid = not errors and set(modes) == set(MODE_SPECS)
    return source, {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_C6_TEMPORAL_CONTROLS_DATA_PREFLIGHT"
            if valid
            else "FAIL_C6_TEMPORAL_CONTROLS_DATA_PREFLIGHT"
        ),
        "config_sha256": config.sha256,
        "clean_lineage_handoff_id": config.payload["execution"][
            "clean_lineage_handoff_id"
        ],
        "native_units": len(native_keys or []),
        "modes": modes,
        "delta_identifiability": delta_audit,
        "timing_source_shortcut": source_audit,
        "order_control_short_runs_authorized": valid,
        "timing_control_short_runs_authorized": valid and timing_modes_authorized,
        "full_development_authorized": full_gate is not None,
        "full_development_launch_gate": full_gate,
        "full_oof_authorized": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": valid,
    }


def execute_c6_temporal_control_run(
    config: C6TemporalControlConfig,
    mode_id: str,
    repeat_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Run one bounded mode only after clean-data and short-run authorization."""

    if mode_id not in MODE_SPECS:
        raise ValueError(f"unsupported temporal control mode={mode_id}")
    if not repeat_id or any(character in repeat_id for character in '\\/:*?"<>|'):
        raise ValueError("repeat_id is blank or unsafe")
    full_gate = _validate_full_development_launch(config)
    source, preflight = data_c6_temporal_control_preflight(config)
    timing_sensitive = mode_id in {
        "TR128_CONSTANT_DELTA",
        "TR128_DELTA_SHUFFLED",
    }
    if timing_sensitive and not preflight["timing_control_short_runs_authorized"]:
        raise RuntimeError("timing control is not identifiable on this data lineage")
    run_root = config.output_root / config.training_scope / mode_id / repeat_id
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    controls = config.payload["controls"]
    derived = derive_c6_temporal_control(
        source.base_view,
        mode_id,
        perturbation_seed=int(controls["perturbation_seed"]),
        constant_delta_seconds=float(controls["constant_delta_seconds"]),
    )
    adapter, selection = build_c6_temporal_training_adapter(
        config,
        source,
        derived,
    )
    device = str(config.payload["execution"]["device"])
    outcome = train_legacy_l5_cached_short_core(
        derived.view,
        selection,
        adapter,
        device=device,
    )
    result = _write_run_artifacts(
        config,
        run_root=run_root,
        repeat_id=repeat_id,
        derived=derived,
        selection=selection,
        outcome=outcome,
        preflight=preflight,
        full_gate=full_gate,
        runtime_seconds=time.time() - started,
        device=device,
    )
    return run_root, result


def audit_c6_temporal_short_gate(
    config: C6TemporalControlConfig,
) -> tuple[Path, dict[str, Any]]:
    """Require stable paired short repeats before full-development comparison."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("C6 temporal short gate requires short_repeat_gate config")
    _, data_preflight = data_c6_temporal_control_preflight(config)
    required_modes = list(MODE_SPECS)
    if not data_preflight["timing_control_short_runs_authorized"]:
        required_modes = [
            mode_id
            for mode_id in required_modes
            if mode_id not in {
                "TR128_CONSTANT_DELTA",
                "TR128_DELTA_SHUFFLED",
            }
        ]
    required_repeats = list(
        config.payload["execution"]["required_short_repeats"]
    )
    errors: list[str] = []
    modes: dict[str, Any] = {}
    common_native_hash: str | None = None
    process_ids: list[int] = []
    for mode_id in required_modes:
        packets: list[dict[str, Any]] = []
        for repeat_id in required_repeats:
            path = (
                config.output_root
                / SHORT_SCOPE
                / mode_id
                / repeat_id
                / "run_result.json"
            )
            if not path.is_file():
                errors.append(f"missing_run_result={mode_id}:{repeat_id}")
                continue
            packet = json.loads(path.read_text(encoding="utf-8"))
            packets.append(packet)
            process_ids.append(int(packet["process_id"]))
        if len(packets) != len(required_repeats):
            continue
        exact_fields = (
            "config_sha256",
            "optimizer_steps",
            "parameter_sha256",
            "prediction_content_sha256",
            "native_unit_sha256",
        )
        mismatches = [
            field
            for field in exact_fields
            if len({str(packet[field]) for packet in packets}) != 1
        ]
        if mismatches:
            errors.append(f"{mode_id}:repeat_mismatch={mismatches}")
        native_hash = str(packets[0]["native_unit_sha256"])
        if common_native_hash is None:
            common_native_hash = native_hash
        elif native_hash != common_native_hash:
            errors.append(f"{mode_id}:paired_native_universe_drift")
        modes[mode_id] = {
            "repeat_ids": required_repeats,
            "process_ids": [int(packet["process_id"]) for packet in packets],
            "repeat_mismatches": mismatches,
            "optimizer_steps": int(packets[0]["optimizer_steps"]),
            "metrics": packets[0]["metrics"],
        }
    if len(process_ids) != len(set(process_ids)):
        errors.append("short_repeat_process_ids_are_not_distinct")
    valid = not errors and set(modes) == set(required_modes)
    payload = {
        "schema_version": SHORT_GATE_SCHEMA,
        "status": (
            "PASS_C6_TEMPORAL_CONTROLS_SHORT_GATE"
            if valid
            else "FAIL_C6_TEMPORAL_CONTROLS_SHORT_GATE"
        ),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "clean_lineage_handoff_id": config.payload["execution"][
            "clean_lineage_handoff_id"
        ],
        "required_modes": required_modes,
        "skipped_non_identifiable_timing_modes": sorted(
            set(MODE_SPECS) - set(required_modes)
        ),
        "required_repeats": required_repeats,
        "modes": modes,
        "common_native_unit_sha256": common_native_hash,
        "delta_identifiability": data_preflight["delta_identifiability"],
        "timing_source_shortcut": data_preflight["timing_source_shortcut"],
        "full_development_comparison_authorized": valid,
        "timing_full_development_comparison_authorized": (
            valid and data_preflight["timing_control_short_runs_authorized"]
        ),
        "full_oof_authorized": False,
        "legacy_sets_full_data_base": False,
        "errors": errors,
        "valid": valid,
    }
    output = config.output_root / "c6_temporal_controls_short_gate.json"
    _write_json_exclusive(output, payload)
    return output, payload


def build_c6_temporal_training_adapter(
    config: C6TemporalControlConfig,
    source: Any,
    derived: DerivedC6TemporalControl,
) -> tuple[LegacyL5CachedTrainingConfig, LegacyL5CachedShortSelection]:
    """Bind a controlled view to the canonical cached training core."""

    spec = MODE_SPECS[derived.mode_id]
    optimization = copy.deepcopy(config.payload["optimization"])
    train_count = len(source.selection.train_positions)
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
    adapter = LegacyL5CachedTrainingConfig(
        path=config.path,
        repo_root=config.repo_root,
        payload={
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
                "expected_validation_native_units": len(
                    source.selection.validation_positions
                ),
                "native_prediction_aggregation": "one_sequence_per_native_unit",
            },
            "model": model,
            "optimization": optimization,
        },
    )
    manifest = source.selection.manifest.copy(deep=True)
    manifest["c6_temporal_control_mode_id"] = derived.mode_id
    selection_hash = cached_engine._dataframe_sha256(manifest)
    selection_audit = copy.deepcopy(source.selection.audit)
    selection_audit.update(
        {
            "training_scope": config.training_scope,
            "mode_id": derived.mode_id,
            "selection_content_sha256": selection_hash,
            "outer_holdout_rows": 0,
        }
    )
    selection = LegacyL5CachedShortSelection(
        manifest=manifest,
        train_positions=source.selection.train_positions.copy(),
        validation_positions=source.selection.validation_positions.copy(),
        audit=selection_audit,
    )
    if derived.view.windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("training adapter duplicates native units")
    return adapter, selection


def build_c6_temporal_control_model(
    config: C6TemporalControlConfig,
    mode_id: str,
    *,
    dropout: float | None = None,
) -> LegacyL5CachedFeatureClassifier:
    """Build one declared temporal model with no implicit family changes."""

    try:
        spec = MODE_SPECS[mode_id]
    except KeyError as error:
        raise ValueError(f"unsupported temporal control mode={mode_id}") from error
    return LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(spec["temporal_encoder_name"]),
        hidden_dim=int(spec["hidden_dim"]),
        dropout=(
            float(config.payload["model_common"]["dropout"])
            if dropout is None
            else float(dropout)
        ),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
    )


def _build_slot_manifest(
    original: LegacyL5CachedFeatureView,
    controlled: LegacyL5CachedFeatureView,
    plan: TemporalControlPlan,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_index, key in enumerate(plan.unit_keys):
        for slot_index in range(controlled.sequence_length):
            source_slot = int(plan.slot_permutation[row_index, slot_index])
            records.append(
                {
                    "c6_temporal_control_mode_id": plan.mode_id,
                    "temporal_unit_key": key,
                    "slot_index": slot_index,
                    "source_slot_index": source_slot,
                    "feature_row": int(controlled.feature_rows[row_index, slot_index]),
                    "real_time_delta": float(original.time_delta[row_index, slot_index]),
                    "controlled_time_delta": float(
                        controlled.time_delta[row_index, slot_index]
                    ),
                    "observed_mask": bool(
                        controlled.observed_mask[row_index, slot_index]
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    expected = len(controlled.windows) * controlled.sequence_length
    if len(frame) != expected:
        raise ValueError("temporal control slot manifest row count drift")
    if frame[["temporal_unit_key", "slot_index"]].duplicated().any():
        raise ValueError("temporal control slot manifest duplicates slots")
    return frame


def _write_run_artifacts(
    config: C6TemporalControlConfig,
    *,
    run_root: Path,
    repeat_id: str,
    derived: DerivedC6TemporalControl,
    selection: LegacyL5CachedShortSelection,
    outcome: LegacyL5CachedTrainingOutcome,
    preflight: dict[str, Any],
    full_gate: dict[str, Any] | None,
    runtime_seconds: float,
    device: str,
) -> dict[str, Any]:
    predictions = outcome.predictions.assign(
        c6_temporal_control_mode_id=derived.mode_id,
        repeat_id=repeat_id,
    )
    predictions.to_csv(
        run_root / "validation_native_predictions.csv",
        index=False,
        mode="x",
        lineterminator="\n",
    )
    outcome.per_class_metrics.to_csv(
        run_root / "metrics_per_class.csv",
        index=False,
        mode="x",
        lineterminator="\n",
    )
    outcome.confusion.to_csv(
        run_root / "confusion_matrix.csv",
        index=False,
        mode="x",
        lineterminator="\n",
    )
    outcome.epoch_metrics.to_csv(
        run_root / "epoch_metrics.csv",
        index=False,
        mode="x",
        lineterminator="\n",
    )
    derived.slot_manifest.to_csv(
        run_root / "derived_slot_manifest.csv",
        index=False,
        mode="x",
        lineterminator="\n",
    )
    checkpoint = {
        "schema_version": "classification_v2.c6_temporal_control_checkpoint.v1",
        "config_sha256": config.sha256,
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "model_state": outcome.model_state,
        "optimizer_state": outcome.optimizer_state,
        "best_epoch": outcome.best_epoch,
    }
    checkpoint_path = run_root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        **outcome.metrics,
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "config_sha256": config.sha256,
        "lineage_scope": LINEAGE_SCOPE,
        "legacy_sets_full_data_base": False,
        "full_oof_authorized": False,
    }
    _write_json_exclusive(run_root / "metrics_global.json", metrics)
    checkpoint_manifest = {
        "schema_version": "classification_v2.c6_temporal_checkpoint_manifest.v1",
        "config_sha256": config.sha256,
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "parameter_sha256": outcome.parameter_sha256,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "checkpoint_manifest.json", checkpoint_manifest)
    prediction_path = run_root / "validation_native_predictions.csv"
    prediction_manifest = {
        "schema_version": "classification_v2.c6_temporal_prediction_manifest.v1",
        "config_sha256": config.sha256,
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": file_sha256(prediction_path),
        "prediction_rows": len(predictions),
        "native_unit_rows": predictions["temporal_unit_key"].astype(str).nunique(),
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "prediction_manifest.json", prediction_manifest)
    environment = {
        "schema_version": "classification_v2.c6_temporal_environment.v1",
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": device,
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
    artifact_manifest = {
        "schema_version": "classification_v2.c6_temporal_artifacts.v1",
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
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "run_id": f"{derived.mode_id}-{repeat_id}",
        "experiment_name": "legacy_c6_temporal_perturbation_controls",
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "training_scope": config.training_scope,
        "completed_at_utc": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "code_state": git_state(),
        "clean_lineage_handoff_id": config.payload["execution"][
            "clean_lineage_handoff_id"
        ],
        "input_hashes": config.payload["execution"]["clean_input_hashes"],
        "selection_sha256": selection.audit["selection_content_sha256"],
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "parameter_count": sum(
            parameter.numel()
            for parameter in build_c6_temporal_control_model(
                config,
                derived.mode_id,
            ).parameters()
        ),
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "native_unit_sha256": _string_sequence_sha256(
            predictions["temporal_unit_key"].astype(str).tolist()
        ),
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "preflight": preflight,
        "full_development_launch_gate": full_gate,
        "legacy_sets_full_data_base": False,
        "full_development_authorized": full_gate is not None,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "run_manifest.json", run_manifest)
    result = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "mode_id": derived.mode_id,
        "repeat_id": repeat_id,
        "config_sha256": config.sha256,
        "process_id": os.getpid(),
        "training_scope": config.training_scope,
        "optimizer_steps": outcome.optimizer_steps,
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_content_sha256": outcome.prediction_sha256,
        "native_unit_sha256": _string_sequence_sha256(
            predictions["temporal_unit_key"].astype(str).tolist()
        ),
        "metrics": metrics,
        "run_manifest_sha256": file_sha256(run_root / "run_manifest.json"),
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(run_root / "run_result.json", result)
    return result


def _validate_config(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "experiment_contract",
        "source_temporal_config",
        "implementation",
        "control_implementation",
        "launcher",
        "modes",
        "controlled_pairs",
        "model_common",
        "controls",
        "optimization",
        "evaluation",
        "execution",
        "short_gate",
        "output",
    }
    if set(payload) != required:
        raise ValueError("C6 temporal-control config keys differ")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("C6 temporal-control config schema drift")
    if payload["training_scope"] not in {SHORT_SCOPE, FULL_SCOPE}:
        raise ValueError("C6 temporal-control training scope unsupported")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("C6 temporal-control lineage drift")
    if payload["modes"] != MODE_SPECS:
        raise ValueError("C6 temporal-control mode contract drift")
    expected_pairs = {name: list(pair) for name, pair in CONTROLLED_PAIRS.items()}
    if payload["controlled_pairs"] != expected_pairs:
        raise ValueError("C6 temporal-control pair contract drift")
    contract = payload["experiment_contract"]
    if contract.get("changed_scientific_family") != "temporal_control_only":
        raise ValueError("C6 temporal-control changed-family drift")
    if tuple(contract.get("skills", [])) != SELECTED_SKILLS:
        raise ValueError("C6 temporal-control selected-skill drift")
    for field in (
        "outer_predictions_used_for_model_selection",
        "legacy_sets_final_full_data_base",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "full_oof_authorized",
    ):
        if contract.get(field) is not False:
            raise ValueError(f"C6 temporal-control unsafe contract field={field}")
    controls = payload["controls"]
    if int(controls.get("perturbation_seed", -1)) < 0:
        raise ValueError("C6 temporal-control perturbation seed invalid")
    if float(controls.get("constant_delta_seconds", 0.0)) <= 0.0:
        raise ValueError("C6 temporal-control constant delta invalid")
    if not 0.0 < float(controls.get("minimum_changed_fraction", 0.0)) <= 1.0:
        raise ValueError("C6 temporal-control changed-fraction gate invalid")
    execution = payload["execution"]
    if not isinstance(execution.get("data_run_authorized"), bool):
        raise ValueError("C6 temporal-control data authorization must be boolean")
    if execution.get("device") not in {"cpu", "cuda:0"}:
        raise ValueError("C6 temporal-control device unsupported")
    if execution.get("required_short_repeats") != ["repeat01", "repeat02"]:
        raise ValueError("C6 temporal-control repeat contract drift")
    if execution.get("full_oof_authorized") is not False:
        raise ValueError("C6 temporal-control config cannot authorize full OOF")
    if payload["training_scope"] == SHORT_SCOPE:
        if execution.get("full_development_authorized") is not False:
            raise ValueError("short config cannot authorize full development")
        if payload["short_gate"] is not None:
            raise ValueError("short config cannot bind a short gate")
    else:
        if execution.get("full_development_authorized") is not True:
            raise ValueError("full config requires explicit development authorization")
        _validate_short_gate_spec(payload["short_gate"])
    if execution["data_run_authorized"]:
        _validate_authorized_execution(execution)
    for field in (
        "source_temporal_config",
        "implementation",
        "control_implementation",
        "launcher",
    ):
        spec = payload[field]
        if set(spec) != {"path", "sha256"} or not is_sha256(spec["sha256"]):
            raise ValueError(f"C6 temporal-control invalid file spec={field}")


def _require_data_authorization(config: C6TemporalControlConfig) -> None:
    execution = config.payload["execution"]
    if execution["data_run_authorized"] is not True:
        raise PermissionError("C6 temporal-control project-data run is fail-closed")
    _validate_authorized_execution(execution)


def _validate_full_development_launch(
    config: C6TemporalControlConfig,
) -> dict[str, Any] | None:
    if config.training_scope == SHORT_SCOPE:
        return None
    spec = config.payload["short_gate"]
    _validate_short_gate_spec(spec)
    path = (config.repo_root / str(spec["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"C6 temporal short gate missing: {path}")
    if file_sha256(path) != spec["sha256"]:
        raise ValueError("C6 temporal short gate hash drift")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != spec["status"]:
        raise ValueError("C6 temporal short gate status drift")
    if gate.get("config_sha256") != spec["short_config_sha256"]:
        raise ValueError("C6 temporal short config hash drift")
    if gate.get("full_development_comparison_authorized") is not True:
        raise ValueError("C6 temporal short gate does not authorize full development")
    if gate.get("valid") is not True or gate.get("errors") != []:
        raise ValueError("C6 temporal short gate is invalid")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": str(gate["status"]),
        "short_config_sha256": str(gate["config_sha256"]),
        "full_development_comparison_authorized": True,
        "full_oof_authorized": False,
        "valid": True,
    }


def _validate_short_gate_spec(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("full C6 temporal config requires short_gate object")
    required = {"path", "sha256", "status", "short_config_sha256"}
    if set(value) != required:
        raise ValueError("C6 temporal short_gate fields differ")
    if not is_sha256(value["sha256"]) or not is_sha256(
        value["short_config_sha256"]
    ):
        raise ValueError("C6 temporal short_gate hashes invalid")


def _validate_authorized_execution(execution: dict[str, Any]) -> None:
    if not str(execution.get("clean_lineage_handoff_id", "")).strip():
        raise ValueError("authorized data run requires clean lineage handoff ID")
    hashes = execution.get("clean_input_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("authorized data run requires clean input hashes")
    if any(not is_sha256(value) for value in hashes.values()):
        raise ValueError("authorized data run contains invalid clean input hash")


def _verify_file_spec(root: Path, spec: dict[str, Any]) -> Path:
    path = (root / str(spec["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"C6 temporal-control path escapes root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"C6 temporal-control file hash drift: {path}")
    return path


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _string_sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "C6TemporalControlConfig",
    "DerivedC6TemporalControl",
    "audit_c6_temporal_short_gate",
    "build_c6_temporal_control_model",
    "build_c6_temporal_training_adapter",
    "data_c6_temporal_control_preflight",
    "derive_c6_temporal_control",
    "execute_c6_temporal_control_run",
    "load_c6_temporal_control_config",
    "static_c6_temporal_control_preflight",
    "synthetic_c6_temporal_control_preflight",
]
