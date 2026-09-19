"""Hash-bound short configuration and CPU preflight for legacy L7."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.imbalance_losses import (
    DEFAULT_EFFECTIVE_NUMBER_BETA,
    LOSS_POLICIES,
    weighted_imbalance_loss,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureView,
    cached_feature_whitelist_payload,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    LINEAGE_SCOPE,
    SHORT_SCOPE,
    TemporalLadderConfig,
    TemporalLadderSelection,
    build_temporal_ladder_selection,
    load_temporal_ladder_config,
    load_temporal_ladder_view,
    temporal_ladder_git_guard,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance import (
    EXPECTED_FULL_TRAIN_NATIVE_UNITS,
    EXPECTED_FULL_TRAIN_WINDOWS,
    EXPECTED_PARAMETER_COUNT,
    VIEW_ID,
    build_l7_model,
    fit_full_training_loss,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_short_config.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_preflight.v1"
)
CANONICAL_SOURCE_NAME = "legacy_16f"


@dataclass(frozen=True, slots=True)
class LegacyL7ImbalanceConfig:
    """One immutable three-policy short matrix configuration."""

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
    def output_root(self) -> Path:
        relative = str(self.payload["output"]["root_relative_path"])
        return _resolve_inside(self.repo_root, relative)

    @property
    def parent_path(self) -> Path:
        relative = str(self.payload["temporal_parent"]["path"])
        return _resolve_inside(self.repo_root, relative)


def load_l7_imbalance_config(path: Path) -> LegacyL7ImbalanceConfig:
    """Load one short L7 config and verify every bound dependency."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_payload(payload)
    config = LegacyL7ImbalanceConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    parent_spec = _object(payload["temporal_parent"], "temporal_parent")
    _validate_bound_file(
        config.parent_path,
        str(parent_spec["sha256"]),
        "L7 temporal parent",
    )
    decision = _object(payload["decision_parent"], "decision_parent")
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(decision["path"])),
        str(decision["sha256"]),
        "L7 decision parent",
    )
    implementation = _object(payload["implementation"], "implementation")
    for name, spec_value in implementation.items():
        spec = _object(spec_value, f"implementation.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"L7 implementation {name}",
        )
    parent = load_temporal_ladder_config(config.parent_path)
    _validate_parent_semantics(config, parent)
    return config


def load_l7_imbalance_inputs(
    config: LegacyL7ImbalanceConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    TemporalLadderSelection,
]:
    """Load the unchanged L5 T6 view and native-first short selection."""

    parent = load_temporal_ladder_config(config.parent_path)
    _, view, _ = load_temporal_ladder_view(parent, VIEW_ID)
    selection = build_temporal_ladder_selection(view, parent, VIEW_ID)
    return parent, view, selection


def preflight_l7_imbalance_policy(
    config: LegacyL7ImbalanceConfig,
    policy: str,
) -> dict[str, Any]:
    """Run real-data CPU-only loss-fit, shape, gradient and Git gates."""

    if policy not in LOSS_POLICIES:
        raise ValueError(f"unknown L7 loss policy={policy}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    loss_fit: Any | None = None
    parameter_count = 0
    output_shape: list[int] = []
    batch_bytes = 0
    finite_loss = False
    finite_gradients = False
    short_native_class_counts: dict[str, int] = {}
    try:
        parent, view, selection = load_l7_imbalance_inputs(config)
        loss_fit = fit_full_training_loss(
            view,
            policy=policy,
            effective_number_beta=float(
                config.payload["loss_fit"]["effective_number_beta"]
            ),
        )
        short_native = (
            selection.manifest.loc[
                selection.manifest["l5_role"].astype(str).eq("train")
            ]
            .drop_duplicates("temporal_unit_key")
            ["behavior_label"]
            .astype(str)
            .value_counts()
            .reindex(VALID_BEHAVIORS, fill_value=0)
        )
        short_native_class_counts = {
            label: int(short_native[label]) for label in VALID_BEHAVIORS
        }
        if set(short_native_class_counts.values()) != {8}:
            errors.append("L7 short optimizer subset is not eight per class")
        sample = selection.train_positions[:32]
        batch, batch_bytes = _load_batch(
            view,
            sample,
            maximum_batch_bytes=int(
                parent.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        model = build_l7_model(parent)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model.train()
        logits = model(
            torch.from_numpy(batch["features"]),
            torch.from_numpy(batch["observed_mask"]).float(),
            time_delta=torch.from_numpy(batch["time_delta"]).float(),
        )
        output_shape = list(logits.shape)
        loss, _ = weighted_imbalance_loss(
            logits,
            torch.from_numpy(batch["targets"]).long(),
            torch.from_numpy(batch["sample_weights"]).float(),
            loss_fit.state,
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        finite_loss = bool(torch.isfinite(loss))
        finite_gradients = bool(gradients) and all(
            gradient is not None
            and torch.isfinite(gradient).all()
            and float(gradient.abs().sum()) > 0.0
            for gradient in gradients
        )
        if output_shape != [len(sample), len(VALID_BEHAVIORS)]:
            errors.append(f"L7 CPU output shape drift={output_shape}")
        if not finite_loss:
            errors.append("L7 CPU loss is nonfinite")
        if not finite_gradients:
            errors.append("L7 CPU gradients are invalid")
        del model, logits, loss, gradients, batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = l7_imbalance_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("L7 CPU preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L7_IMBALANCE_PREFLIGHT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "training_scope": config.training_scope,
        "view_id": VIEW_ID,
        "loss_policy": policy,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "selection_content_sha256": (
            selection.audit["selection_content_sha256"]
            if selection is not None
            else None
        ),
        "loss_fit": loss_fit.to_payload() if loss_fit is not None else None,
        "short_optimizer_native_class_counts": short_native_class_counts,
        "short_optimizer_subset_used_for_loss_fit": False,
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "one_batch_finite_loss": finite_loss,
        "one_batch_finite_nonzero_gradients": finite_gradients,
        "maximum_loaded_batch_bytes": batch_bytes,
        "feature_whitelist": cached_feature_whitelist_payload(),
        "sampler": config.payload["experiment_contract"]["fixed_sampler"],
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "validation_rows_read_for_loss_fit": 0,
        "outer_holdout_rows_read_for_loss_fit": 0,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def l7_imbalance_git_guard(
    config: LegacyL7ImbalanceConfig,
) -> dict[str, Any]:
    """Require committed L7 sources while preserving declared user dirt."""

    parent = load_temporal_ladder_config(config.parent_path)
    payload = copy.deepcopy(parent.payload)
    payload["execution_guard"] = copy.deepcopy(
        config.payload["execution_guard"]
    )
    guard_config = TemporalLadderConfig(
        path=parent.path,
        payload=payload,
        repo_root=parent.repo_root,
    )
    return temporal_ladder_git_guard(guard_config)


def l7_implementation_hashes(
    config: LegacyL7ImbalanceConfig,
) -> dict[str, str]:
    """Return every config-bound implementation hash by declared name."""

    return {
        name: str(_object(spec, name)["sha256"])
        for name, spec in config.payload["implementation"].items()
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "canonical_source_name",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "experiment_contract",
        "temporal_parent",
        "decision_parent",
        "loss_fit",
        "implementation",
        "repeat_gate",
        "execution_guard",
        "output",
    }
    _require_exact_keys(payload, required, "L7 config")
    identity = {
        "schema_version": SHORT_CONFIG_SCHEMA,
        "training_scope": SHORT_SCOPE,
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
    }
    for field, value in identity.items():
        _require_equal(payload.get(field), value, field)
    experiment = {
        "experiment_id": "L7_T6_IMBALANCE_POLICY_ABLATION_V1",
        "parent_decision": "L6_PASS_RETAIN_T6_ACTOR_ONLY_BASE",
        "changed_family": "loss_policy_only",
        "loss_policies": list(LOSS_POLICIES),
        "view_id": VIEW_ID,
        "model_parameter_count": EXPECTED_PARAMETER_COUNT,
        "fixed_sampler": (
            "deterministic_seeded_window_shuffle_after_native_selection"
        ),
        "short_optimizer_native_units": 80,
        "short_optimizer_native_units_per_class": 8,
        "loss_fit_native_units": EXPECTED_FULL_TRAIN_NATIVE_UNITS,
        "loss_fit_windows": EXPECTED_FULL_TRAIN_WINDOWS,
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(payload["experiment_contract"], experiment, "experiment")
    for name in ("temporal_parent", "decision_parent"):
        _validate_bound_spec(payload[name], name)
    loss_fit = {
        "fit_role": "training_native_event_mass",
        "fit_scope": "complete_3652_native_training_role_for_all_policies",
        "effective_number_beta": DEFAULT_EFFECTIVE_NUMBER_BETA,
        "class_order": list(VALID_BEHAVIORS),
        "short_optimizer_subset_used_for_fit": False,
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
    }
    _require_equal(payload["loss_fit"], loss_fit, "loss_fit")
    implementation = _object(payload["implementation"], "implementation")
    expected_implementations = {
        "core",
        "loss_engine",
        "config_engine",
        "runtime",
        "synthetic_gate",
        "frozen_training_engine",
        "frozen_temporal_engine",
    }
    _require_exact_keys(
        implementation,
        expected_implementations,
        "implementation",
    )
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    repeat = {
        "required_runs_per_policy": 2,
        "require_fresh_process": True,
        "require_distinct_process_ids": True,
        "require_non_overlapping_execution": True,
        "require_identical_loss_fit_hash": True,
        "require_identical_parameter_hash": True,
        "require_identical_window_prediction_hash": True,
        "require_identical_native_prediction_hash": True,
        "require_identical_epoch_metric_hash": True,
    }
    _require_equal(payload["repeat_gate"], repeat, "repeat_gate")
    execution = _object(payload["execution_guard"], "execution_guard")
    _require_exact_keys(
        execution,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(payload["output"], "output")
    _require_exact_keys(
        output,
        {"root_relative_path", "matrix_gate_filename"},
        "output",
    )
    _validate_relative_path(output["root_relative_path"], "output root")
    if Path(str(output["matrix_gate_filename"])).name != str(
        output["matrix_gate_filename"]
    ):
        raise ValueError("L7 matrix gate is not a filename")


def _validate_parent_semantics(
    config: LegacyL7ImbalanceConfig,
    parent: TemporalLadderConfig,
) -> None:
    _require_equal(parent.training_scope, SHORT_SCOPE, "parent scope")
    _require_equal(parent.payload["lineage_scope"], LINEAGE_SCOPE, "parent lane")
    model = parent.payload["model"]
    expected_model = {
        "architecture": "cached_frame_feature_temporal_classifier_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
    }
    _require_equal(model, expected_model, "L7 retained model")
    optimization = parent.payload["optimization"]
    fixed = {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "sampler": config.payload["experiment_contract"]["fixed_sampler"],
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
    }
    for field, value in fixed.items():
        _require_equal(optimization[field], value, f"parent optimization.{field}")


def _load_batch(
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    maximum_batch_bytes: int,
) -> tuple[dict[str, np.ndarray], int]:
    values = np.asarray(positions, dtype=np.int64)
    batch = {
        "features": view.load_sequences(values),
        "observed_mask": view.observed_mask[values].copy(),
        "time_delta": view.time_delta[values].copy(),
        "targets": view.targets[values].copy(),
        "sample_weights": view.sample_weights[values].copy(),
    }
    loaded = sum(int(value.nbytes) for value in batch.values())
    if loaded > maximum_batch_bytes:
        raise MemoryError(f"L7 preflight batch={loaded}>{maximum_batch_bytes}")
    return batch, loaded


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    if len(str(spec["sha256"])) != 64:
        raise ValueError(f"{name} SHA256 length drift")


def _validate_bound_file(
    path: Path,
    expected_sha256: str,
    name: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    _require_equal(file_sha256(path), expected_sha256, f"{name} hash")


def _validate_relative_path(value: object, name: str) -> None:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"L7 path escapes repository: {value}") from error
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


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
            f"{name} keys differ: missing={sorted(expected - observed)},"
            f"extra={sorted(observed - expected)}"
        )


def _require_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{name} mismatch: observed={observed!r},expected={expected!r}"
        )


def tracked_paths(config: LegacyL7ImbalanceConfig) -> list[str]:
    """Return paths the Git guard requires as committed inputs."""

    paths = [str(value) for value in config.payload["execution_guard"][
        "required_tracked_paths"
    ]]
    return sorted(path.replace("\\", "/") for path in paths)


def is_path_tracked(config: LegacyL7ImbalanceConfig, path: str) -> bool:
    """Expose the exact non-mutating Git membership check for tests."""

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
    return completed.returncode == 0
