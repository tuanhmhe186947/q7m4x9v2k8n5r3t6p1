"""Immutable V1 temporal-length ladder over audited cached frame features."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as frozen_engine,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LegacyL5Config,
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
    build_legacy_l5_cached_feature_view,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
)

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_short_config.v1"
)
FULL_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_full_config.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_preflight.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
SHORT_SCOPE = "short_repeat_gate"
FULL_SCOPE = "full_development_baseline"
MODEL_PARAMETER_COUNT = 68_234
MODEL_VISIBLE_ROLES = ("train", "validation")
RARE_CLASSES = ("fight", "social-nose", "playwithtoy", "move")

CANONICAL_VIEWS: dict[str, dict[str, Any]] = {
    "t6_centered": {
        "temporal_view_name": "legacy_t6_centered_matched_observed_time",
        "sampling_protocol": "one_centered_window_matched",
        "sequence_length": 6,
        "windows_per_native_unit": 1,
        "train_windows_full": 3_652,
        "validation_windows": 245,
        "train_windows_short": 80,
        "optimizer_steps_short": 9,
        "optimizer_steps_full": 345,
    },
    "t8_centered": {
        "temporal_view_name": "legacy_t8_centered_matched_observed_time",
        "sampling_protocol": "one_centered_window_matched",
        "sequence_length": 8,
        "windows_per_native_unit": 1,
        "train_windows_full": 3_652,
        "validation_windows": 245,
        "train_windows_short": 80,
        "optimizer_steps_short": 9,
        "optimizer_steps_full": 345,
    },
    "t12_centered": {
        "temporal_view_name": "legacy_t12_centered_matched_observed_time",
        "sampling_protocol": "one_centered_window_matched",
        "sequence_length": 12,
        "windows_per_native_unit": 1,
        "train_windows_full": 3_652,
        "validation_windows": 245,
        "train_windows_short": 80,
        "optimizer_steps_short": 9,
        "optimizer_steps_full": 345,
    },
    "t16_centered": {
        "temporal_view_name": "legacy_t16_centered_matched_observed_time",
        "sampling_protocol": "one_centered_window_matched",
        "sequence_length": 16,
        "windows_per_native_unit": 1,
        "train_windows_full": 3_652,
        "validation_windows": 245,
        "train_windows_short": 80,
        "optimizer_steps_short": 9,
        "optimizer_steps_full": 345,
    },
    "t6_sliding": {
        "temporal_view_name": "legacy_t6_all_sliding_observed_time",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 6,
        "windows_per_native_unit": 4,
        "train_windows_full": 14_608,
        "validation_windows": 980,
        "train_windows_short": 320,
        "optimizer_steps_short": 30,
        "optimizer_steps_full": 1_371,
    },
    "t8_sliding": {
        "temporal_view_name": "legacy_t8_all_sliding_observed_time",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 8,
        "windows_per_native_unit": 3,
        "train_windows_full": 10_956,
        "validation_windows": 735,
        "train_windows_short": 240,
        "optimizer_steps_short": 24,
        "optimizer_steps_full": 1_029,
    },
    "t12_sliding": {
        "temporal_view_name": "legacy_t12_all_sliding_observed_time",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 12,
        "windows_per_native_unit": 2,
        "train_windows_full": 7_304,
        "validation_windows": 490,
        "train_windows_short": 160,
        "optimizer_steps_short": 15,
        "optimizer_steps_full": 687,
    },
    "t16_sliding": {
        "temporal_view_name": "legacy_t16_all_sliding_observed_time",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 16,
        "windows_per_native_unit": 1,
        "train_windows_full": 3_652,
        "validation_windows": 245,
        "train_windows_short": 80,
        "optimizer_steps_short": 9,
        "optimizer_steps_full": 345,
    },
}


@dataclass(frozen=True, slots=True)
class TemporalLadderConfig:
    """One immutable matrix-wide short or full ladder contract."""

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
    def base_config_path(self) -> Path:
        return self.repo_root / str(self.payload["base_config"]["path"])

    @property
    def feature_result_path(self) -> Path:
        return self.repo_root / str(self.payload["feature_parent"]["result_path"])

    @property
    def output_root(self) -> Path:
        base = load_legacy_l5_config(self.base_config_path)
        return base.primary_root / str(self.payload["output"]["run_root_relative_path"])

    def view_spec(self, view_id: str) -> dict[str, Any]:
        if view_id not in CANONICAL_VIEWS:
            raise ValueError(f"unknown temporal ladder view: {view_id}")
        return _object(self.payload["views"][view_id], f"views.{view_id}")


@dataclass(frozen=True, slots=True)
class TemporalLadderSelection:
    """Native-first selection expanded to exact windows for one view."""

    manifest: pd.DataFrame
    train_positions: np.ndarray
    validation_positions: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalLadderOutcome:
    """Best checkpoint and both window/native validation predictions."""

    epoch_metrics: pd.DataFrame
    window_predictions: pd.DataFrame
    native_predictions: pd.DataFrame
    metrics: dict[str, Any]
    per_class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    best_epoch: int
    optimizer_steps: int
    parameter_sha256: str
    window_prediction_sha256: str
    native_prediction_sha256: str
    epoch_metrics_sha256: str
    maximum_loaded_batch_bytes: int


def load_temporal_ladder_config(path: Path) -> TemporalLadderConfig:
    """Load one matrix config and reject all undeclared semantic drift."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    root = resolved.parents[2]
    config = TemporalLadderConfig(path=resolved, payload=payload, repo_root=root)
    _validate_bound_file(config.base_config_path, payload["base_config"], "base config")
    implementation = _object(payload["implementation"], "implementation")
    for prefix in ("core", "runtime", "frozen_engine"):
        source_path = root / str(implementation[f"{prefix}_path"])
        _validate_bound_file(
            source_path,
            {"sha256": implementation[f"{prefix}_sha256"]},
            prefix,
        )
    feature = _object(payload["feature_parent"], "feature_parent")
    _validate_bound_file(
        config.feature_result_path,
        {"sha256": feature["result_sha256"]},
        "feature result",
    )
    _validate_bound_file(
        root / str(feature["run_manifest_path"]),
        {"sha256": feature["run_manifest_sha256"]},
        "feature run manifest",
    )
    if config.training_scope == FULL_SCOPE:
        _validate_full_authorization(config)
    return config


def load_temporal_ladder_view(
    config: TemporalLadderConfig,
    view_id: str,
) -> tuple[LegacyL5Config, LegacyL5CachedFeatureView, dict[str, Any]]:
    """Load one view after verifying its immutable consumer parent."""

    view_spec = config.view_spec(view_id)
    expected = CANONICAL_VIEWS[view_id]
    parent = _object(view_spec["consumer_parent"], f"{view_id}.consumer_parent")
    base = load_legacy_l5_config(config.base_config_path)
    run_root = base.primary_root / str(parent["run_relative_path"])
    paths = {
        "run_manifest": run_root / "run_manifest.json",
        "cached_data_audit": run_root / "cached_data_audit.json",
        "artifact_manifest": run_root / "artifact_manifest.json",
    }
    for field, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"consumer parent missing {field}: {path}")
        _require_equal(
            file_sha256(path),
            parent[f"{field}_sha256"],
            f"{view_id} {field} hash",
        )
    manifest = _read_json(paths["run_manifest"])
    audit = _read_json(paths["cached_data_audit"])
    artifact_manifest = _read_json(paths["artifact_manifest"])
    manifest_expected = {
        "run_id": parent["run_id"],
        "code_sha": parent["code_sha"],
        "status": "completed",
        "config_hash": config.payload["base_config"]["sha256"],
        "cache_hash": config.payload["feature_parent"]["feature_tensor_sha256"],
        "control_id": "V1",
        "temporal_view_name": expected["temporal_view_name"],
        "sequence_length": expected["sequence_length"],
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "optimizer_steps": 0,
        "peak_vram_bytes": 0,
    }
    _require_mapping(manifest, manifest_expected, f"{view_id} consumer manifest")
    audit_expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_DATA",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "control_id": "V1",
        "temporal_view_name": expected["temporal_view_name"],
        "sequence_length": expected["sequence_length"],
        "feature_dim": FEATURE_DIM,
        "source_media_loads": 0,
        "valid": True,
    }
    _require_mapping(audit, audit_expected, f"{view_id} consumer audit")
    bounded = _object(audit.get("bounded_batch_audit"), "bounded_batch_audit")
    bounded_expected = {
        "outer_holdout_rows_loaded": 0,
        "cuda_runtime_initialized_before": False,
        "cuda_runtime_initialized_after": False,
        "source_media_reads": 0,
        "valid": True,
    }
    _require_mapping(bounded, bounded_expected, f"{view_id} bounded audit")
    _validate_consumer_artifact_links(
        artifact_manifest,
        run_root=run_root,
        expected_run_id=str(parent["run_id"]),
        expected_audit_sha=str(parent["cached_data_audit_sha256"]),
    )
    view = build_legacy_l5_cached_feature_view(
        base,
        feature_result_path=config.feature_result_path,
        temporal_view_name=str(expected["temporal_view_name"]),
    )
    _validate_loaded_view(view, view_id)
    return base, view, {
        "run_root": str(run_root.resolve()),
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "hashes": {name: file_sha256(path) for name, path in paths.items()},
        "run_manifest": manifest,
        "cached_data_audit": audit,
        "errors": [],
        "valid": True,
    }


def build_temporal_ladder_selection(
    view: LegacyL5CachedFeatureView,
    config: TemporalLadderConfig,
    view_id: str,
) -> TemporalLadderSelection:
    """Select train native units first, then include their exact windows."""

    expected = CANONICAL_VIEWS[view_id]
    windows = view.windows.copy().reset_index(drop=True)
    windows["position"] = np.arange(len(windows), dtype=np.int64)
    if not windows["l5_role"].isin(MODEL_VISIBLE_ROLES).all():
        raise ValueError("ladder view exposes a forbidden routing role")
    native = _native_rows(windows)
    train_native = native.loc[native["l5_role"].eq("train")].copy()
    validation_native = native.loc[native["l5_role"].eq("validation")].copy()
    if len(train_native) != 3_652 or len(validation_native) != 245:
        raise ValueError("ladder native train/validation counts drift")
    selection = _object(config.payload["selection"], "selection")
    salt = str(selection["short_native_selection_salt"])
    train_native["selection_score"] = train_native["temporal_unit_key"].map(
        lambda value: _selection_score(salt, str(value))
    )
    train_native = train_native.sort_values(
        ["behavior_label", "selection_score", "temporal_unit_key"],
        kind="mergesort",
    )
    if config.training_scope == SHORT_SCOPE:
        per_class = int(selection["short_train_native_units_per_class"])
        selected_native = train_native.groupby(
            "behavior_label",
            sort=False,
            group_keys=False,
        ).head(per_class)
    else:
        selected_native = train_native
    selected_ids = set(selected_native["temporal_unit_key"].astype(str))
    train_windows = windows.loc[
        windows["l5_role"].eq("train")
        & windows["temporal_unit_key"].astype(str).isin(selected_ids)
    ].copy()
    validation_windows = windows.loc[windows["l5_role"].eq("validation")].copy()
    score_map = selected_native.set_index("temporal_unit_key")["selection_score"]
    train_windows["selection_score"] = train_windows["temporal_unit_key"].map(
        score_map
    )
    validation_windows["selection_score"] = validation_windows[
        "temporal_unit_key"
    ].map(lambda value: _selection_score("all_validation", str(value)))
    train_windows = train_windows.sort_values(
        ["behavior_label", "selection_score", "temporal_unit_key", "window_id"],
        kind="mergesort",
    )
    validation_windows = validation_windows.sort_values(
        ["temporal_unit_key", "window_id"],
        kind="mergesort",
    )
    expected_train_windows = int(
        expected[
            "train_windows_short"
            if config.training_scope == SHORT_SCOPE
            else "train_windows_full"
        ]
    )
    _require_equal(len(train_windows), expected_train_windows, "train window count")
    _require_equal(
        len(validation_windows),
        int(expected["validation_windows"]),
        "validation window count",
    )
    _validate_window_mass(
        view,
        train_windows,
        expected_multiplier=int(expected["windows_per_native_unit"]),
        role="train",
    )
    _validate_window_mass(
        view,
        validation_windows,
        expected_multiplier=int(expected["windows_per_native_unit"]),
        role="validation",
    )
    selected = pd.concat([train_windows, validation_windows], ignore_index=True)
    positions = selected["position"].to_numpy(dtype=np.int64)
    selected["target_index"] = view.targets[positions]
    selected["sample_weight"] = view.sample_weights[positions]
    selected["training_scope"] = config.training_scope
    selected["view_id"] = view_id
    selected["sampling_protocol"] = expected["sampling_protocol"]
    selected["sequence_length"] = int(expected["sequence_length"])
    selected["selection_order"] = np.arange(len(selected), dtype=np.int64)
    manifest_columns = [
        "selection_order",
        "position",
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "target_index",
        "l5_role",
        "selection_score",
        "sample_weight",
        "view_id",
        "sampling_protocol",
        "sequence_length",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
    ]
    manifest = selected[manifest_columns].copy()
    train_count = len(train_windows)
    train_positions = manifest.iloc[:train_count]["position"].to_numpy(dtype=np.int64)
    validation_positions = manifest.iloc[train_count:]["position"].to_numpy(
        dtype=np.int64
    )
    audit = {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_selection.v1"
        ),
        "training_scope": config.training_scope,
        "view_id": view_id,
        "temporal_view_name": expected["temporal_view_name"],
        "sampling_protocol": expected["sampling_protocol"],
        "sequence_length": expected["sequence_length"],
        "windows_per_native_unit": expected["windows_per_native_unit"],
        "train_native_units": int(train_windows["temporal_unit_key"].nunique()),
        "validation_native_units": int(
            validation_windows["temporal_unit_key"].nunique()
        ),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "train_native_unit_sha256": _ordered_hash(
            train_windows["temporal_unit_key"].drop_duplicates()
        ),
        "validation_native_unit_sha256": _ordered_hash(
            validation_windows["temporal_unit_key"].drop_duplicates()
        ),
        "selection_content_sha256": frozen_engine._dataframe_sha256(manifest),
        "outer_holdout_rows": 0,
        "source_media_reads": 0,
        "errors": [],
        "valid": True,
    }
    expected_train_native = 80 if config.training_scope == SHORT_SCOPE else 3_652
    _require_equal(audit["train_native_units"], expected_train_native, "train units")
    _require_equal(audit["validation_native_units"], 245, "validation units")
    return TemporalLadderSelection(
        manifest=manifest,
        train_positions=train_positions,
        validation_positions=validation_positions,
        audit=audit,
    )


def preflight_temporal_ladder_view(
    config: TemporalLadderConfig,
    view_id: str,
) -> dict[str, Any]:
    """Run the exact CPU-only real-parent, selection, shape, and git gate."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    base: LegacyL5Config | None = None
    view: LegacyL5CachedFeatureView | None = None
    parent: dict[str, Any] | None = None
    selection: TemporalLadderSelection | None = None
    output_shape: list[int] | None = None
    parameter_count = 0
    loaded_bytes = 0
    try:
        base, view, parent = load_temporal_ladder_view(config, view_id)
        selection = build_temporal_ladder_selection(view, config, view_id)
        sample_positions = selection.train_positions[:64]
        batch, loaded_bytes = load_temporal_ladder_batch(
            view,
            sample_positions,
            maximum_batch_bytes=int(
                config.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        model = _build_model(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != MODEL_PARAMETER_COUNT:
            errors.append(f"model_parameter_count={parameter_count}")
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(batch["features"]),
                torch.from_numpy(batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(batch["time_delta"]).float(),
            )
        output_shape = list(logits.shape)
        if output_shape != [len(sample_positions), len(VALID_BEHAVIORS)]:
            errors.append(f"cpu_forward_shape={output_shape}")
        del logits, model, batch
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = temporal_ladder_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("CPU preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_PREFLIGHT"
        ),
        "training_scope": config.training_scope,
        "view_id": view_id,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "base_config_sha256": base.sha256 if base is not None else None,
        "consumer_parent_valid": parent is not None,
        "selection_content_sha256": (
            selection.audit["selection_content_sha256"]
            if selection is not None
            else None
        ),
        "train_native_units": (
            selection.audit["train_native_units"] if selection is not None else 0
        ),
        "validation_native_units": (
            selection.audit["validation_native_units"]
            if selection is not None
            else 0
        ),
        "train_windows": (
            selection.audit["train_windows"] if selection is not None else 0
        ),
        "validation_windows": (
            selection.audit["validation_windows"] if selection is not None else 0
        ),
        "maximum_loaded_batch_bytes": loaded_bytes,
        "maximum_loaded_batch_bytes_allowed": int(
            config.payload["optimization"]["maximum_loaded_batch_bytes"]
        ),
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_temporal_ladder_core(
    view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalLadderConfig,
    view_id: str,
    *,
    device: torch.device | str,
) -> TemporalLadderOutcome:
    """Train one exact view and select checkpoints on native-unit metrics."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("temporal ladder requested unavailable CUDA")
    _validate_selection_before_training(view, selection, config, view_id)
    optimization = _object(config.payload["optimization"], "optimization")
    seed = int(optimization["seed"])
    _seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model: LegacyL5CachedFeatureClassifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = _build_model(config).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        optimizer_steps = 0
        maximum_batch_bytes = 0
        best_epoch = 0
        best_score: tuple[float, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_optimizer: dict[str, Any] | None = None
        best_window_predictions: pd.DataFrame | None = None
        best_native_predictions: pd.DataFrame | None = None
        best_metrics: dict[str, Any] | None = None
        best_per_class: pd.DataFrame | None = None
        best_confusion: pd.DataFrame | None = None
        epoch_rows: list[dict[str, Any]] = []
        for epoch in range(1, int(optimization["epochs"]) + 1):
            train_positions = selection.train_positions.copy()
            np.random.default_rng(seed + epoch).shuffle(train_positions)
            loss_mass = 0.0
            weight_mass = 0.0
            model.train()
            for positions in frozen_engine._position_batches(
                train_positions,
                batch_size=int(optimization["batch_size"]),
            ):
                batch, loaded = load_temporal_ladder_batch(
                    view,
                    positions,
                    maximum_batch_bytes=int(
                        optimization["maximum_loaded_batch_bytes"]
                    ),
                )
                maximum_batch_bytes = max(maximum_batch_bytes, loaded)
                loss_value, batch_weight = frozen_engine._cached_training_step(
                    model,
                    optimizer,
                    batch,
                    device=resolved_device,
                    gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                )
                optimizer_steps += 1
                loss_mass += loss_value * batch_weight
                weight_mass += batch_weight
                del batch
            if weight_mass <= 0.0:
                raise RuntimeError("temporal ladder train weight mass is empty")
            evaluation = frozen_engine._evaluate_cached_classifier(
                model,
                view,
                selection.validation_positions,
                batch_size=int(optimization["evaluation_batch_size"]),
                maximum_batch_bytes=int(
                    optimization["maximum_loaded_batch_bytes"]
                ),
                device=resolved_device,
            )
            maximum_batch_bytes = max(
                maximum_batch_bytes,
                int(evaluation["maximum_loaded_batch_bytes"]),
            )
            window_predictions = build_window_prediction_frame(
                view,
                selection.validation_positions,
                probabilities=evaluation["probabilities"],
                targets=evaluation["targets"],
                config=config,
                view_id=view_id,
            )
            native_predictions, metrics, per_class, confusion = (
                aggregate_temporal_ladder_predictions(
                    window_predictions,
                    expected_windows_per_native=int(
                        CANONICAL_VIEWS[view_id]["windows_per_native_unit"]
                    ),
                    training_scope=config.training_scope,
                )
            )
            parameter_sha = frozen_engine._state_dict_sha256(model.state_dict())
            window_sha = frozen_engine._dataframe_sha256(window_predictions)
            native_sha = frozen_engine._dataframe_sha256(native_predictions)
            score = (
                float(metrics["macro_f1_global_10_class"]),
                -float(metrics["nll"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = frozen_engine._clone_state_dict(model.state_dict())
                best_optimizer = frozen_engine._clone_to_cpu(optimizer.state_dict())
                best_window_predictions = window_predictions.copy(deep=True)
                best_native_predictions = native_predictions.copy(deep=True)
                best_metrics = copy.deepcopy(metrics)
                best_per_class = per_class.copy(deep=True)
                best_confusion = confusion.copy(deep=True)
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_steps_cumulative": optimizer_steps,
                    "train_native_units": selection.audit["train_native_units"],
                    "train_windows": selection.audit["train_windows"],
                    "train_loss": loss_mass / weight_mass,
                    "validation_native_units": selection.audit[
                        "validation_native_units"
                    ],
                    "validation_windows": selection.audit["validation_windows"],
                    "validation_macro_f1_global_10_class": metrics[
                        "macro_f1_global_10_class"
                    ],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_nll": metrics["nll"],
                    "parameter_sha256": parameter_sha,
                    "window_prediction_sha256": window_sha,
                    "native_prediction_sha256": native_sha,
                    "selected_checkpoint": False,
                    "training_scope": config.training_scope,
                    "view_id": view_id,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                    "reviewed_or_final_claim_allowed": False,
                    "q2_claim_allowed": False,
                }
            )
        expected_steps = _expected_optimizer_steps(config, view_id)
        _require_equal(optimizer_steps, expected_steps, "optimizer steps")
        if any(
            value is None
            for value in (
                best_state,
                best_optimizer,
                best_window_predictions,
                best_native_predictions,
                best_metrics,
                best_per_class,
                best_confusion,
            )
        ):
            raise RuntimeError("temporal ladder checkpoint selection is empty")
        epoch_rows[best_epoch - 1]["selected_checkpoint"] = True
        epoch_metrics = pd.DataFrame.from_records(epoch_rows)
        assert best_state is not None
        assert best_optimizer is not None
        assert best_window_predictions is not None
        assert best_native_predictions is not None
        assert best_metrics is not None
        assert best_per_class is not None
        assert best_confusion is not None
        return TemporalLadderOutcome(
            epoch_metrics=epoch_metrics,
            window_predictions=best_window_predictions,
            native_predictions=best_native_predictions,
            metrics=best_metrics,
            per_class_metrics=best_per_class,
            confusion=best_confusion,
            model_state=best_state,
            optimizer_state=best_optimizer,
            best_epoch=best_epoch,
            optimizer_steps=optimizer_steps,
            parameter_sha256=frozen_engine._state_dict_sha256(best_state),
            window_prediction_sha256=frozen_engine._dataframe_sha256(
                best_window_predictions
            ),
            native_prediction_sha256=frozen_engine._dataframe_sha256(
                best_native_predictions
            ),
            epoch_metrics_sha256=frozen_engine._dataframe_sha256(epoch_metrics),
            maximum_loaded_batch_bytes=maximum_batch_bytes,
        )
    finally:
        if model is not None:
            model.to("cpu")
        del model, optimizer


def build_window_prediction_frame(
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    config: TemporalLadderConfig,
    view_id: str,
) -> pd.DataFrame:
    """Build exact window predictions before native-unit aggregation."""

    metadata = view.windows.iloc[np.asarray(positions, dtype=np.int64)].reset_index(
        drop=True
    )
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs.shape != (len(metadata), len(VALID_BEHAVIORS)):
        raise ValueError("window probability shape drift")
    if labels.shape != (len(metadata),):
        raise ValueError("window target shape drift")
    predicted = probs.argmax(axis=1).astype(np.int64)
    frame = pd.DataFrame(
        {
            "prediction_order": np.arange(len(metadata), dtype=np.int64),
            "window_id": metadata["window_id"].astype(str),
            "temporal_unit_key": metadata["temporal_unit_key"].astype(str),
            "recording_group_id": metadata["recording_group_id"].astype(str),
            "video_key": metadata["video_key"].astype(str),
            "source_type": metadata["source_type"].astype(str),
            "dataset_id": metadata["dataset_id"].astype(str),
            "behavior_label": metadata["behavior_label"].astype(str),
            "target_index": labels,
            "predicted_index": predicted,
            "predicted_label": [VALID_BEHAVIORS[index] for index in predicted],
            "sample_weight": view.sample_weights[np.asarray(positions, dtype=np.int64)],
        }
    )
    for index, label in enumerate(VALID_BEHAVIORS):
        frame[_probability_column(label)] = probs[:, index]
    frame["training_scope"] = config.training_scope
    frame["view_id"] = view_id
    frame["sampling_protocol"] = CANONICAL_VIEWS[view_id]["sampling_protocol"]
    frame["sequence_length"] = CANONICAL_VIEWS[view_id]["sequence_length"]
    frame["lineage_scope"] = LINEAGE_SCOPE
    frame["human_review_complete"] = False
    frame["reviewed_or_final_claim_allowed"] = False
    frame["q2_claim_allowed"] = False
    return frame


def aggregate_temporal_ladder_predictions(
    window_predictions: pd.DataFrame,
    *,
    expected_windows_per_native: int,
    training_scope: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Mean window probabilities and emit one strict native prediction."""

    probability_columns = [_probability_column(label) for label in VALID_BEHAVIORS]
    required = {
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "target_index",
        "sample_weight",
        *probability_columns,
    }
    missing = sorted(required - set(window_predictions.columns))
    if missing:
        raise ValueError(f"window predictions missing columns: {missing}")
    if window_predictions["window_id"].astype(str).duplicated().any():
        raise ValueError("window predictions contain duplicate window IDs")
    rows: list[dict[str, Any]] = []
    grouped = window_predictions.groupby("temporal_unit_key", sort=True)
    for order, (unit_id, frame) in enumerate(grouped):
        _require_equal(len(frame), expected_windows_per_native, "native window count")
        for column in (
            "recording_group_id",
            "video_key",
            "source_type",
            "dataset_id",
            "behavior_label",
            "target_index",
        ):
            if frame[column].nunique(dropna=False) != 1:
                raise ValueError(f"native aggregation conflict: {column}")
        _require_close(float(frame["sample_weight"].sum()), 1.0, "event mass")
        probabilities = frame[probability_columns].to_numpy(dtype=np.float64).mean(
            axis=0
        )
        _require_probability_mass(float(probabilities.sum()))
        predicted_index = int(probabilities.argmax())
        first = frame.iloc[0]
        row: dict[str, Any] = {
            "prediction_order": order,
            "temporal_unit_key": str(unit_id),
            "recording_group_id": str(first["recording_group_id"]),
            "video_key": str(first["video_key"]),
            "source_type": str(first["source_type"]),
            "dataset_id": str(first["dataset_id"]),
            "behavior_label": str(first["behavior_label"]),
            "target_index": int(first["target_index"]),
            "predicted_index": predicted_index,
            "predicted_label": VALID_BEHAVIORS[predicted_index],
            "aggregated_window_count": len(frame),
        }
        row.update(
            {
                column: float(probabilities[index])
                for index, column in enumerate(probability_columns)
            }
        )
        rows.append(row)
    native = pd.DataFrame.from_records(rows)
    probabilities = native[probability_columns].to_numpy(dtype=np.float64)
    targets = native["target_index"].to_numpy(dtype=np.int64)
    metrics, per_class, confusion = frozen_engine.compute_legacy_l5_native_metrics(
        probabilities,
        targets,
        native["temporal_unit_key"],
    )
    lineage = {
        "training_scope": training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    native = native.assign(**lineage)
    metrics = {
        **metrics,
        **lineage,
        "aggregation": "mean_window_probability_per_native_unit_v1",
        "window_rows": int(len(window_predictions)),
    }
    per_class = per_class.assign(**lineage)
    confusion = confusion.assign(**lineage)
    return native, metrics, per_class, confusion


def load_temporal_ladder_batch(
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    maximum_batch_bytes: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Load one bounded mmap batch and close the mapping before return."""

    values = np.asarray(positions, dtype=np.int64).copy()
    batch = {
        "positions": values,
        "features": view.load_sequences(values),
        "observed_mask": view.observed_mask[values].copy(),
        "time_delta": view.time_delta[values].copy(),
        "targets": view.targets[values].copy(),
        "sample_weights": view.sample_weights[values].copy(),
    }
    loaded_bytes = sum(int(value.nbytes) for value in batch.values())
    if loaded_bytes > maximum_batch_bytes:
        raise MemoryError(
            f"temporal ladder loaded batch={loaded_bytes}>{maximum_batch_bytes}"
        )
    if not np.isfinite(batch["sample_weights"]).all():
        raise ValueError("temporal ladder sample weights are nonfinite")
    if (batch["sample_weights"] <= 0.0).any():
        raise ValueError("temporal ladder sample weights are not positive")
    return batch, loaded_bytes


def temporal_ladder_git_guard(config: TemporalLadderConfig) -> dict[str, Any]:
    """Require committed ladder sources/config and only known user dirt."""

    guard = _object(config.payload["execution_guard"], "execution_guard")
    status = _git(config.repo_root, "status", "--porcelain", "--untracked-files=all")
    entries = [line for line in status.splitlines() if line.strip()]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"])
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(path).replace("\\", "/") for path in guard["required_tracked_paths"]
    ]
    untracked: list[str] = []
    for path in required:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(config.repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                path,
            ],
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
        "code_sha": _git(config.repo_root, "rev-parse", "HEAD").strip(),
        "dirty_entries": entries,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def implementation_hashes(config: TemporalLadderConfig) -> dict[str, str]:
    implementation = _object(config.payload["implementation"], "implementation")
    return {
        name: str(implementation[name])
        for name in (
            "core_sha256",
            "runtime_sha256",
            "frozen_engine_sha256",
        )
    }


def _validate_config_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "experiment_contract",
        "base_config",
        "feature_parent",
        "implementation",
        "views",
        "selection",
        "model",
        "optimization",
        "repeat_gate",
        "execution_guard",
        "output",
    }
    if payload.get("training_scope") == FULL_SCOPE:
        required.add("full_authorization")
    _require_exact_keys(payload, required, "temporal ladder config")
    schema = payload["schema_version"]
    scope = payload["training_scope"]
    if (schema, scope) not in {
        (SHORT_CONFIG_SCHEMA, SHORT_SCOPE),
        (FULL_CONFIG_SCHEMA, FULL_SCOPE),
    }:
        raise ValueError("temporal ladder schema/scope mismatch")
    _require_equal(payload["lineage_scope"], LINEAGE_SCOPE, "lineage scope")
    for field in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    ):
        _require_equal(payload[field], False, field)
    _require_equal(payload["development_metrics_authorized"], True, "metrics claim")
    _validate_experiment_contract(payload["experiment_contract"])
    _validate_bound_spec(payload["base_config"], "base config")
    _validate_feature_parent(payload["feature_parent"])
    _validate_implementation(payload["implementation"])
    views = _object(payload["views"], "views")
    _require_equal(set(views), set(CANONICAL_VIEWS), "view matrix")
    for view_id, expected in CANONICAL_VIEWS.items():
        _validate_view_spec(view_id, views[view_id], expected)
    _validate_selection_contract(payload["selection"])
    _validate_model_contract(payload["model"])
    _validate_optimization_contract(payload["optimization"])
    _validate_repeat_contract(payload["repeat_gate"], scope=str(scope))
    _validate_execution_guard(payload["execution_guard"])
    output = _object(payload["output"], "output")
    _require_exact_keys(
        output,
        {"run_root_relative_path", "matrix_gate_filename"},
        "output",
    )


def _validate_experiment_contract(value: object) -> None:
    payload = _object(value, "experiment_contract")
    expected = {
        "experiment_id": "L5_V1_TEMPORAL_LENGTH_PROTOCOL_LADDER_V1",
        "parent_decision": "RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH",
        "changed_family": "temporal_input_length_and_declared_protocol_matrix",
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(payload, expected, "experiment contract")


def _validate_feature_parent(value: object) -> None:
    payload = _object(value, "feature_parent")
    required = {
        "result_path",
        "result_sha256",
        "run_manifest_path",
        "run_manifest_sha256",
        "feature_tensor_sha256",
        "feature_index_sha256",
    }
    _require_exact_keys(payload, required, "feature_parent")
    for field in required:
        if field.endswith("sha256"):
            _require_sha(payload[field], f"feature_parent.{field}")


def _validate_implementation(value: object) -> None:
    payload = _object(value, "implementation")
    required = {
        "core_path",
        "core_sha256",
        "runtime_path",
        "runtime_sha256",
        "frozen_engine_path",
        "frozen_engine_sha256",
    }
    _require_exact_keys(payload, required, "implementation")
    for field in required:
        if field.endswith("sha256"):
            _require_sha(payload[field], f"implementation.{field}")


def _validate_view_spec(
    view_id: str,
    value: object,
    expected: dict[str, Any],
) -> None:
    payload = _object(value, f"views.{view_id}")
    required = {
        "temporal_view_name",
        "sampling_protocol",
        "sequence_length",
        "windows_per_native_unit",
        "consumer_parent",
    }
    _require_exact_keys(payload, required, f"views.{view_id}")
    for field in required - {"consumer_parent"}:
        _require_equal(payload[field], expected[field], f"{view_id}.{field}")
    parent = _object(payload["consumer_parent"], f"{view_id}.consumer_parent")
    parent_fields = {
        "run_id",
        "code_sha",
        "run_relative_path",
        "run_manifest_sha256",
        "cached_data_audit_sha256",
        "artifact_manifest_sha256",
    }
    _require_exact_keys(parent, parent_fields, f"{view_id}.consumer_parent")
    for field in parent_fields:
        if field.endswith("sha256"):
            _require_sha(parent[field], f"{view_id}.{field}")


def _validate_selection_contract(value: object) -> None:
    payload = _object(value, "selection")
    expected = {
        "native_unit": "complete_legacy_16_frame_burst",
        "short_train_selection_policy": "sha256_rank_per_class_native_first_v1",
        "short_native_selection_salt": "legacy_l5_temporal_ladder_short_v1",
        "short_train_native_units_per_class": 8,
        "short_train_native_units": 80,
        "full_train_native_units": 3_652,
        "validation_native_units": 245,
        "event_mass_per_native_unit": 1.0,
        "window_expansion_after_native_selection": True,
        "validation_selection_policy": "all_validation_native_units_v1",
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    _require_equal(payload, expected, "selection contract")


def _validate_model_contract(value: object) -> None:
    payload = _object(value, "model")
    expected = {
        "architecture": "cached_frame_feature_temporal_classifier_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
    }
    _require_equal(payload, expected, "model contract")


def _validate_optimization_contract(value: object) -> None:
    payload = _object(value, "optimization")
    expected = {
        "seed": 20260714,
        "epochs": 3,
        "batch_size": 32,
        "evaluation_batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "loss": "event_mass_balanced_cross_entropy",
        "sampler": "deterministic_seeded_window_shuffle_after_native_selection",
        "checkpoint_selection": "native_global_10_class_macro_f1_then_nll",
        "precision": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "device": "cuda:0",
        "declared_local_gpu_vram_gib": 4,
        "validated_local_gpu_vram_bytes": 4_294_443_008,
        "maximum_peak_vram_fraction": 0.7,
        "allocator_limit_bytes": 3_006_110_105,
        "oom_retry_allowed": False,
        "maximum_loaded_batch_bytes": 2_103_552,
    }
    _require_equal(payload, expected, "optimization contract")


def _validate_repeat_contract(value: object, *, scope: str) -> None:
    payload = _object(value, "repeat_gate")
    expected = {
        "required_runs": 2 if scope == SHORT_SCOPE else 1,
        "require_fresh_process": True,
        "require_distinct_process_ids": scope == SHORT_SCOPE,
        "require_non_overlapping_execution": scope == SHORT_SCOPE,
        "require_identical_selection_hash": scope == SHORT_SCOPE,
        "require_identical_parameter_hash": scope == SHORT_SCOPE,
        "require_identical_window_prediction_hash": scope == SHORT_SCOPE,
        "require_identical_native_prediction_hash": scope == SHORT_SCOPE,
        "require_identical_epoch_metric_hash": scope == SHORT_SCOPE,
    }
    _require_equal(payload, expected, "repeat contract")


def _validate_execution_guard(value: object) -> None:
    payload = _object(value, "execution_guard")
    _require_exact_keys(
        payload,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )


def _validate_full_authorization(config: TemporalLadderConfig) -> None:
    authorization = _object(config.payload["full_authorization"], "full_authorization")
    required = {
        "short_config_path",
        "short_config_sha256",
        "matrix_gate_path",
        "matrix_gate_sha256",
        "authorized_training_scope",
    }
    _require_exact_keys(authorization, required, "full_authorization")
    _require_equal(
        authorization["authorized_training_scope"],
        FULL_SCOPE,
        "authorized full scope",
    )
    short_path = config.repo_root / str(authorization["short_config_path"])
    gate_path = config.repo_root / str(authorization["matrix_gate_path"])
    _validate_bound_file(
        short_path,
        {"sha256": authorization["short_config_sha256"]},
        "short config authorization",
    )
    _validate_bound_file(
        gate_path,
        {"sha256": authorization["matrix_gate_sha256"]},
        "matrix gate authorization",
    )
    short_config = load_temporal_ladder_config(short_path)
    _validate_full_semantic_binding(config.payload, short_config.payload)
    gate = _read_json(gate_path)
    gate_expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_SHORT_MATRIX",
        "lineage_scope": LINEAGE_SCOPE,
        "short_config_sha256": authorization["short_config_sha256"],
        "full_expansion_authorized": True,
        "valid": True,
    }
    _require_mapping(gate, gate_expected, "short matrix authorization")


def _validate_full_semantic_binding(
    full_payload: dict[str, Any],
    short_payload: dict[str, Any],
) -> None:
    _require_equal(
        (short_payload.get("schema_version"), short_payload.get("training_scope")),
        (SHORT_CONFIG_SCHEMA, SHORT_SCOPE),
        "short authorization schema/scope",
    )
    scientific_fields = (
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "experiment_contract",
        "base_config",
        "feature_parent",
        "implementation",
        "views",
        "selection",
        "model",
        "optimization",
    )
    for field in scientific_fields:
        _require_equal(
            full_payload.get(field),
            short_payload.get(field),
            f"full/short scientific binding.{field}",
        )


def _validate_loaded_view(view: LegacyL5CachedFeatureView, view_id: str) -> None:
    expected = CANONICAL_VIEWS[view_id]
    _require_equal(view.control_id, "V1", "view control")
    _require_equal(view.temporal_view_name, expected["temporal_view_name"], "view name")
    _require_equal(view.sequence_length, expected["sequence_length"], "view length")
    train = view.indices_for_role("train")
    validation = view.indices_for_role("validation")
    _require_equal(len(train), expected["train_windows_full"], "view train windows")
    _require_equal(
        len(validation),
        expected["validation_windows"],
        "view validation windows",
    )


def _validate_consumer_artifact_links(
    manifest: dict[str, Any],
    *,
    run_root: Path,
    expected_run_id: str,
    expected_audit_sha: str,
) -> None:
    _require_equal(manifest.get("run_id"), expected_run_id, "artifact run id")
    _require_equal(manifest.get("status"), "completed", "artifact status")
    _require_equal(manifest.get("valid"), True, "artifact valid")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("consumer artifact rows must be a list")
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    for name in (
        "feature_whitelist",
        "fold_manifest",
        "native_routing_manifest",
        "leakage_audit",
        "temporal_unit_audit",
        "hidden_review_audit",
        "cached_data_audit",
        "environment",
        "checkpoint_manifest",
        "prediction_manifest",
    ):
        if name not in by_name:
            raise ValueError(f"consumer artifact missing: {name}")
        row = by_name[name]
        path = Path(str(row["path"])).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError as error:
            raise ValueError(f"consumer output escapes run root: {name}") from error
        _require_equal(file_sha256(path), row["sha256"], f"consumer artifact {name}")
        _require_equal(path.stat().st_size, row["size_bytes"], f"consumer size {name}")
    _require_equal(
        by_name["cached_data_audit"]["sha256"],
        expected_audit_sha,
        "consumer cached audit link",
    )


def _native_rows(windows: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "l5_role",
        "lineage_scope",
        "human_review_complete",
    ]
    for field in fields:
        if windows.groupby("temporal_unit_key")[field].nunique(dropna=False).gt(1).any():
            raise ValueError(f"native window metadata conflict: {field}")
    return windows[fields].drop_duplicates("temporal_unit_key").reset_index(drop=True)


def _validate_window_mass(
    view: LegacyL5CachedFeatureView,
    windows: pd.DataFrame,
    *,
    expected_multiplier: int,
    role: str,
) -> None:
    positions = windows["position"].to_numpy(dtype=np.int64)
    frame = pd.DataFrame(
        {
            "temporal_unit_key": windows["temporal_unit_key"].astype(str).to_numpy(),
            "sample_weight": view.sample_weights[positions],
        }
    )
    grouped = frame.groupby("temporal_unit_key")["sample_weight"].agg(["size", "sum"])
    if not grouped["size"].eq(expected_multiplier).all():
        raise ValueError(f"{role} windows-per-native drift")
    if not np.allclose(grouped["sum"].to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{role} event mass is not one per native unit")


def _validate_selection_before_training(
    view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalLadderConfig,
    view_id: str,
) -> None:
    _require_equal(selection.audit.get("valid"), True, "selection valid")
    _require_equal(selection.audit.get("view_id"), view_id, "selection view")
    _require_equal(
        selection.audit.get("training_scope"),
        config.training_scope,
        "selection scope",
    )
    _require_equal(selection.audit.get("outer_holdout_rows"), 0, "outer rows")
    _require_equal(
        frozen_engine._dataframe_sha256(selection.manifest),
        selection.audit.get("selection_content_sha256"),
        "selection content hash",
    )
    maximum = len(view.windows) - 1
    for role, positions in (
        ("train", selection.train_positions),
        ("validation", selection.validation_positions),
    ):
        values = np.asarray(positions, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"{role} positions are empty")
        if values.min() < 0 or values.max() > maximum:
            raise ValueError(f"{role} positions are outside the view")
        _require_equal(
            set(view.windows.iloc[values]["l5_role"].astype(str)),
            {role},
            f"{role} routing",
        )


def _build_model(config: TemporalLadderConfig) -> LegacyL5CachedFeatureClassifier:
    model = _object(config.payload["model"], "model")
    return LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )


def _expected_optimizer_steps(config: TemporalLadderConfig, view_id: str) -> int:
    field = (
        "optimizer_steps_short"
        if config.training_scope == SHORT_SCOPE
        else "optimizer_steps_full"
    )
    return int(CANONICAL_VIEWS[view_id][field])


def _seed_all(seed: int, *, seed_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _validate_bound_spec(value: object, name: str) -> None:
    payload = _object(value, name)
    _require_exact_keys(payload, {"path", "sha256"}, name)
    _require_sha(payload["sha256"], name)


def _validate_bound_file(path: Path, spec: dict[str, Any], name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    _require_equal(file_sha256(path), spec["sha256"], f"{name} hash")


def _selection_score(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\x1f{value}".encode()).hexdigest()


def _ordered_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.fillna("").astype(str).tolist()))
    return hashlib.sha256(payload.encode()).hexdigest()


def _probability_column(label: str) -> str:
    return "prob_" + label.replace("-", "_")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"git command failed: {' '.join(arguments)}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"')


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - observed)},"
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
        raise ValueError(f"{name} mismatch: observed={observed!r},expected={expected!r}")


def _require_close(observed: float, expected: float, name: str) -> None:
    if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} mismatch: observed={observed},expected={expected}")


def _require_probability_mass(observed: float) -> None:
    if not np.isclose(observed, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "native probability mass mismatch: "
            f"observed={observed},expected=1.0"
        )


def _require_sha(value: object, name: str) -> None:
    if not is_sha256(str(value)):
        raise ValueError(f"{name} is not a lowercase SHA-256")


__all__ = [
    "CANONICAL_VIEWS",
    "FULL_CONFIG_SCHEMA",
    "FULL_SCOPE",
    "LINEAGE_SCOPE",
    "MODEL_PARAMETER_COUNT",
    "PREFLIGHT_SCHEMA",
    "RARE_CLASSES",
    "SHORT_CONFIG_SCHEMA",
    "SHORT_SCOPE",
    "TemporalLadderConfig",
    "TemporalLadderOutcome",
    "TemporalLadderSelection",
    "aggregate_temporal_ladder_predictions",
    "build_temporal_ladder_selection",
    "build_window_prediction_frame",
    "implementation_hashes",
    "load_temporal_ladder_batch",
    "load_temporal_ladder_config",
    "load_temporal_ladder_view",
    "preflight_temporal_ladder_view",
    "temporal_ladder_git_guard",
    "train_temporal_ladder_core",
]
