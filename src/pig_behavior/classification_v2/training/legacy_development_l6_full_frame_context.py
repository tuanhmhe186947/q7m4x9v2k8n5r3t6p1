"""Bounded full-frame context classifier controls for legacy L6."""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as frozen_engine,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM as ACTOR_FEATURE_DIM,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    SHORT_SCOPE,
    TemporalLadderConfig,
    TemporalLadderSelection,
    build_temporal_ladder_selection,
    load_temporal_ladder_config,
    load_temporal_ladder_view,
)
from pig_behavior.classification_v2.training.legacy_development_l6_cached_modality import (
    CachedModalityNormalizationState,
    LegacyL6CachedModalityClassifier,
    LegacyL6CachedModalityView,
    build_cached_modality_view,
    cached_modality_feature_whitelist,
    fit_cached_modality_normalization,
)
from pig_behavior.classification_v2.training.legacy_development_l6_full_frame_context_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    FEATURE_DIM,
    FEATURE_NAMES,
    LINEAGE_SCOPE,
    MODALITY_NAME,
    SEQUENCE_LENGTH,
    SOURCE_TYPE,
    load_full_frame_context_cache,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    _evaluate_geometry_view,
    _object,
    _read_json,
    _require_equal,
    _require_exact_keys,
    _resolve_inside,
    _train_geometry_epochs,
    _validate_bound_file,
    _validate_cache_view_alignment,
    _validate_selection_for_training,
    geometry_training_git_guard,
    implementation_hashes,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

VIEW_ID = "t6_sliding"
MODES = ("parameter_matched_zero", "availability_only", MODALITY_NAME)
AUXILIARY_DIM = FEATURE_DIM + 1
MODEL_INPUT_DIM = ACTOR_FEATURE_DIM + AUXILIARY_DIM
EXPECTED_PARAMETER_COUNT = 134_924
EXPECTED_SHORT_TRAIN_WINDOWS = 320
EXPECTED_VALIDATION_WINDOWS = 980
EXPECTED_SHORT_OPTIMIZER_STEPS = 30

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_context_short_config.v1"
)
NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_context_normalization.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_context_preflight.v1"
)
WHITELIST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_context_whitelist.v1"
)


@dataclass(frozen=True, slots=True)
class LegacyL6FullFrameContextConfig:
    """Hash-bound short full-frame context experiment."""

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
        relative = str(self.payload["output"]["run_root_relative_path"])
        return _resolve_inside(self.repo_root, relative)

    def bound_path(self, section: str, name: str | None = None) -> Path:
        value: Any = self.payload[section]
        if name is not None:
            value = value[name]
        spec = _object(value, f"{section}.{name}" if name else section)
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6FullFrameContextOutcome:
    """Selected-checkpoint outputs for one full-frame context mode."""

    epoch_metrics: pd.DataFrame
    window_predictions: pd.DataFrame
    native_predictions: pd.DataFrame
    validation_metrics: dict[str, Any]
    per_class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    confusion_groups: pd.DataFrame
    missing_window_predictions: pd.DataFrame
    missing_native_predictions: pd.DataFrame
    missing_validation_metrics: dict[str, Any]
    missing_confusion_groups: pd.DataFrame
    normalization: CachedModalityNormalizationState
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    best_epoch: int
    optimizer_steps: int
    parameter_sha256: str
    window_prediction_sha256: str
    native_prediction_sha256: str
    epoch_metrics_sha256: str
    missing_native_prediction_sha256: str
    maximum_loaded_batch_bytes: int


def load_full_frame_context_training_config(
    path: Path,
) -> LegacyL6FullFrameContextConfig:
    """Load and verify the immutable full-frame experiment contract."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6FullFrameContextConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for section in ("parents", "inputs", "implementation"):
        for name, value in _object(payload[section], section).items():
            spec = _object(value, f"{section}.{name}")
            _validate_bound_file(
                _resolve_inside(config.repo_root, str(spec["path"])),
                str(spec["sha256"]),
                f"full-frame context {section}.{name}",
            )
    return config


def fit_full_frame_context_normalization(
    cache: Any,
    selection: TemporalLadderSelection,
) -> CachedModalityNormalizationState:
    """Fit feature statistics from unique training scene frames only."""

    return fit_cached_modality_normalization(
        cache,
        selection,
        modality_name=MODALITY_NAME,
        feature_names=FEATURE_NAMES,
        identity_field="scene_frame_uid",
        schema_version=NORMALIZATION_SCHEMA,
    )


def build_full_frame_context_view(
    base: LegacyL5CachedFeatureView,
    cache: Any,
    *,
    mode: str,
    normalization: CachedModalityNormalizationState,
) -> LegacyL6CachedModalityView:
    """Align full-frame features and availability to the frozen T6 view."""

    _validate_cache_view_alignment(base, cache)
    return build_cached_modality_view(
        base,
        cache,
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=FEATURE_NAMES,
        sequence_length=SEQUENCE_LENGTH,
        normalization=normalization,
    )


def full_frame_context_feature_whitelist(mode: str) -> dict[str, Any]:
    """Return the explicit 1,025-wide full-frame model-X contract."""

    return cached_modality_feature_whitelist(
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=FEATURE_NAMES,
        schema_version=WHITELIST_SCHEMA,
    )


def build_full_frame_context_model(
    config: LegacyL6FullFrameContextConfig,
) -> LegacyL6CachedModalityClassifier:
    """Build one fixed-width classifier shared by all three controls."""

    model = _object(config.payload["model"], "model")
    classifier = LegacyL6CachedModalityClassifier(
        input_dim=MODEL_INPUT_DIM,
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )
    observed = sum(parameter.numel() for parameter in classifier.parameters())
    if observed != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"L6 full-frame parameters={observed}!="
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def preflight_full_frame_context_mode(
    config: LegacyL6FullFrameContextConfig,
    mode: str,
) -> dict[str, Any]:
    """Run CPU shape, cache, mask, missingness, and Git gates."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 full-frame context mode={mode}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    normalization: CachedModalityNormalizationState | None = None
    parameter_count = 0
    output_shape: list[int] = []
    missing_output_shape: list[int] = []
    loaded_bytes = 0
    cache_content: dict[str, Any] = {}
    try:
        _, base, cache, selection = load_full_frame_context_training_inputs(
            config
        )
        normalization = fit_full_frame_context_normalization(cache, selection)
        view = build_full_frame_context_view(
            base,
            cache,
            mode=mode,
            normalization=normalization,
        )
        sample = selection.train_positions[:64]
        batch, loaded_bytes = frozen_engine._load_selected_batch(
            view,
            sample,
            maximum_batch_bytes=int(
                config.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        model = build_full_frame_context_model(config)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(batch["features"]),
                torch.from_numpy(batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(batch["time_delta"]).float(),
            )
            missing_batch, _ = frozen_engine._load_selected_batch(
                view.with_missing_modality(),
                sample,
                maximum_batch_bytes=int(
                    config.payload["optimization"][
                        "maximum_loaded_batch_bytes"
                    ]
                ),
            )
            missing_logits = model(
                torch.from_numpy(missing_batch["features"]),
                torch.from_numpy(missing_batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(
                    missing_batch["time_delta"]
                ).float(),
            )
        output_shape = list(logits.shape)
        missing_output_shape = list(missing_logits.shape)
        expected_shape = [len(sample), len(VALID_BEHAVIORS)]
        if output_shape != expected_shape:
            errors.append("L6 full-frame CPU forward shape drift")
        if missing_output_shape != expected_shape:
            errors.append("L6 full-frame missing shape drift")
        cache_content = copy.deepcopy(
            _object(cache.manifest["content_audit"], "content_audit")
        )
        expected_patterns = [
            {"pattern": [0] * SEQUENCE_LENGTH, "windows": 14_288},
            {"pattern": [1] * SEQUENCE_LENGTH, "windows": 1_300},
        ]
        if cache_content.get("availability_patterns") != expected_patterns:
            errors.append("L6 full-frame availability pattern drift")
        expected_features = {
            "full_frame_context_only": True,
            "scene_full_frame_only": True,
            "source_scene_ids_in_model_x": False,
            "union_context_values_in_model_x": False,
            "geometry_values_in_model_x": False,
            "motion_values_in_model_x": False,
            "roi_values_in_model_x": False,
            "social_values_in_model_x": False,
            "unit_aggregate_features_in_model_x": False,
            "labels_paths_ids_folds_review_fields_in_model_x": False,
            "availability_is_behavior_evidence": False,
        }
        observed_features = {
            name: cache_content.get(name) for name in expected_features
        }
        if observed_features != expected_features:
            errors.append(
                f"L6 full-frame feature contract={observed_features}"
            )
        del logits, missing_logits, model, batch, missing_batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = full_frame_context_training_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("L6 full-frame preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_CONTEXT_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_FULL_FRAME_CONTEXT_PREFLIGHT"
        ),
        "training_scope": config.training_scope,
        "mode": mode,
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "selection_content_sha256": (
            selection.audit["selection_content_sha256"]
            if selection is not None
            else None
        ),
        "normalization_state_sha256": (
            normalization.state_sha256 if normalization is not None else None
        ),
        "normalization_unique_scene_frames": (
            normalization.unique_train_identity_rows
            if normalization is not None
            else 0
        ),
        "train_windows": (
            selection.audit["train_windows"] if selection is not None else 0
        ),
        "validation_windows": (
            selection.audit["validation_windows"]
            if selection is not None
            else 0
        ),
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "missing_modality_output_shape": missing_output_shape,
        "maximum_loaded_batch_bytes": loaded_bytes,
        "availability_patterns": cache_content.get(
            "availability_patterns", []
        ),
        "cache_feature_contract": cache_content,
        "feature_whitelist": full_frame_context_feature_whitelist(mode),
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "source_scene_ids_in_model_x": False,
        "union_context_values_in_model_x": False,
        "geometry_values_in_model_x": False,
        "motion_values_in_model_x": False,
        "roi_values_in_model_x": False,
        "social_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_full_frame_context_core(
    base: LegacyL5CachedFeatureView,
    cache: Any,
    selection: TemporalLadderSelection,
    config: LegacyL6FullFrameContextConfig,
    mode: str,
    *,
    device: torch.device | str,
) -> LegacyL6FullFrameContextOutcome:
    """Train one full-frame control through the shared L6 loop."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 full-frame context mode={mode}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L6 full-frame requested unavailable CUDA")
    _validate_selection_for_training(base, cache, selection, config)
    normalization = fit_full_frame_context_normalization(cache, selection)
    view = build_full_frame_context_view(
        base,
        cache,
        mode=mode,
        normalization=normalization,
    )
    optimization = _object(config.payload["optimization"], "optimization")
    seed = int(optimization["seed"])
    frozen_engine._seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model: LegacyL6CachedModalityClassifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = build_full_frame_context_model(config).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        best = _train_geometry_epochs(
            model,
            optimizer,
            view,
            selection,
            config,
            mode=mode,
            device=resolved_device,
        )
        _rename_geometry_surfaces(best)
        model.load_state_dict(best["model_state"])
        missing = _evaluate_geometry_view(
            model,
            view.with_missing_modality(),
            selection,
            config,
            device=resolved_device,
        )
        _rename_geometry_surfaces(missing)
        return LegacyL6FullFrameContextOutcome(
            epoch_metrics=best["epoch_metrics"],
            window_predictions=best["window_predictions"],
            native_predictions=best["native_predictions"],
            validation_metrics=best["validation_metrics"],
            per_class_metrics=best["per_class_metrics"],
            confusion=best["confusion"],
            confusion_groups=best["confusion_groups"],
            missing_window_predictions=missing["window_predictions"],
            missing_native_predictions=missing["native_predictions"],
            missing_validation_metrics=missing["validation_metrics"],
            missing_confusion_groups=missing["confusion_groups"],
            normalization=normalization,
            model_state=best["model_state"],
            optimizer_state=best["optimizer_state"],
            best_epoch=int(best["best_epoch"]),
            optimizer_steps=int(best["optimizer_steps"]),
            parameter_sha256=frozen_engine._state_dict_sha256(
                best["model_state"]
            ),
            window_prediction_sha256=frozen_engine._dataframe_sha256(
                best["window_predictions"]
            ),
            native_prediction_sha256=frozen_engine._dataframe_sha256(
                best["native_predictions"]
            ),
            epoch_metrics_sha256=frozen_engine._dataframe_sha256(
                best["epoch_metrics"]
            ),
            missing_native_prediction_sha256=(
                frozen_engine._dataframe_sha256(
                    missing["native_predictions"]
                )
            ),
            maximum_loaded_batch_bytes=max(
                int(best["maximum_loaded_batch_bytes"]),
                int(missing["maximum_loaded_batch_bytes"]),
            ),
        )
    finally:
        if model is not None:
            model.to("cpu")
        del model, optimizer
        gc.collect()


def load_full_frame_context_training_inputs(
    config: LegacyL6FullFrameContextConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    Any,
    TemporalLadderSelection,
]:
    """Load the frozen T6 actor view and full-frame feature mapping."""

    ladder_path = config.bound_path("parents", "temporal_ladder_config")
    ladder = load_temporal_ladder_config(ladder_path)
    _, base, _ = load_temporal_ladder_view(ladder, VIEW_ID)
    selection_payload = copy.deepcopy(ladder.payload)
    selection_payload["training_scope"] = config.training_scope
    selection_config = TemporalLadderConfig(
        path=ladder.path,
        payload=selection_payload,
        repo_root=ladder.repo_root,
    )
    selection = build_temporal_ladder_selection(
        base,
        selection_config,
        VIEW_ID,
    )
    expected_selection = config.payload["selection"]
    _require_equal(
        selection.audit["selection_content_sha256"],
        expected_selection["selection_content_sha256"],
        "full-frame selection content hash",
    )
    cache = load_full_frame_context_cache(config, base_windows=base.windows)
    _validate_cache_view_alignment(base, cache)
    _validate_selection_for_training(base, cache, selection, config)
    return ladder, base, cache, selection


def full_frame_context_training_git_guard(
    config: LegacyL6FullFrameContextConfig,
) -> dict[str, Any]:
    """Use the shared committed-source and dirty-path guard."""

    return geometry_training_git_guard(config)


def full_frame_context_implementation_hashes(
    config: LegacyL6FullFrameContextConfig,
) -> dict[str, str]:
    """Hash every declared training and reusable runtime dependency."""

    return implementation_hashes(config)


def _rename_geometry_surfaces(result: dict[str, Any]) -> None:
    for value in result.values():
        if isinstance(value, pd.DataFrame) and "geometry_mode" in value:
            value.rename(
                columns={"geometry_mode": "full_frame_context_mode"},
                inplace=True,
            )
        elif isinstance(value, dict) and "geometry_mode" in value:
            value["full_frame_context_mode"] = value.pop("geometry_mode")


def _validate_config_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "canonical_source_name",
        "source_type",
        "dataset_id",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "experiment_contract",
        "parents",
        "inputs",
        "implementation",
        "selection",
        "model",
        "optimization",
        "repeat_gate",
        "execution_guard",
        "output",
    }
    _require_exact_keys(payload, required, "L6 full-frame context config")
    _require_equal(payload["schema_version"], SHORT_CONFIG_SCHEMA, "schema")
    _require_equal(payload["training_scope"], SHORT_SCOPE, "scope")
    identity = {
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
    }
    for field, expected in identity.items():
        _require_equal(payload[field], expected, field)
    expected_contract = {
        "experiment_id": "L6_V1_T6_FULL_FRAME_CONTEXT_S4_V1",
        "parent_decision": (
            "DO_NOT_EXPAND_UNION_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "baseline_source": "parameter_matched_zero_without_rejected_values",
        "changed_family": "full_frame_context_only",
        "modes": list(MODES),
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "parameter_matched": True,
        "availability_only_is_diagnostic": True,
        "availability_is_behavior_evidence": False,
        "full_frame_context_only": True,
        "source_scene_ids_in_model_x": False,
        "union_context_values_in_model_x": False,
        "geometry_values_in_model_x": False,
        "motion_values_in_model_x": False,
        "roi_values_in_model_x": False,
        "social_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "missing_modality_inference_required": True,
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(
        _object(payload["experiment_contract"], "experiment_contract"),
        expected_contract,
        "experiment contract",
    )
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {
            "temporal_ladder_config",
            "l5_decision",
            "l6_union_decision",
            "full_frame_cache_config",
            "full_frame_feature_config",
        },
        "parents",
    )
    for name, value in parents.items():
        _validate_bound_spec(value, f"parents.{name}")
    inputs = _object(payload["inputs"], "inputs")
    expected_inputs = {
        "image_window_context_manifest",
        "window_subset_audit",
        "feature_audit",
        "feature_index",
        "feature_tensor",
        "cache_repeat_gate",
        "feature_repeat_gate",
    }
    _require_exact_keys(inputs, expected_inputs, "inputs")
    for name, value in inputs.items():
        _validate_bound_spec(value, f"inputs.{name}")
    implementation = _object(payload["implementation"], "implementation")
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    selection = _object(payload["selection"], "selection")
    expected_selection = {
        "view_id": VIEW_ID,
        "native_unit": "complete_legacy_16_frame_burst",
        "windows_per_native_unit": 4,
        "short_train_native_units": 80,
        "short_train_windows": EXPECTED_SHORT_TRAIN_WINDOWS,
        "validation_native_units": 245,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "selected_windows": 1_300,
        "selected_slots": 7_800,
        "selected_scene_frames": 1_545,
        "selection_content_sha256": (
            "a91bcd684d927423461338f90b303544d0c4a149db8dd612f06f390cca40f070"
        ),
        "event_mass_per_native_unit": 1.0,
        "normalization_fit_scope": "unique_train_scene_frame_uid_only",
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    _require_equal(selection, expected_selection, "selection")
    model = _object(payload["model"], "model")
    expected_model = {
        "architecture": (
            "cached_visual_full_frame_context_temporal_classifier_v1"
        ),
        "feature_control_id": "V1",
        "actor_backbone_name": "resnet18",
        "context_backbone_name": "resnet18",
        "input_resolution": 224,
        "visual_feature_dim": ACTOR_FEATURE_DIM,
        "full_frame_context_feature_dim": FEATURE_DIM,
        "availability_feature_dim": 1,
        "model_input_dim": MODEL_INPUT_DIM,
        "full_frame_context_modes": list(MODES),
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
        "missing_modality_policy": (
            "zero_full_frame_context_and_availability_v1"
        ),
    }
    _require_equal(model, expected_model, "model")
    optimization = _object(payload["optimization"], "optimization")
    expected_optimization = {
        "seed": 20260714,
        "epochs": 3,
        "batch_size": 32,
        "evaluation_batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "loss": "event_mass_balanced_cross_entropy",
        "sampler": (
            "deterministic_seeded_window_shuffle_after_native_selection"
        ),
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
        "maximum_peak_vram_fraction": 0.7,
        "oom_retry_allowed": False,
        "one_run_per_fresh_process": True,
        "maximum_loaded_batch_bytes": 2_200_000,
    }
    _require_equal(optimization, expected_optimization, "optimization")
    repeat = _object(payload["repeat_gate"], "repeat_gate")
    expected_repeat = {
        "required_runs_per_mode": 2,
        "require_fresh_process": True,
        "require_distinct_process_ids": True,
        "require_non_overlapping_execution": True,
        "require_identical_selection_hash": True,
        "require_identical_normalization_hash": True,
        "require_identical_parameter_hash": True,
        "require_identical_window_prediction_hash": True,
        "require_identical_native_prediction_hash": True,
        "require_identical_epoch_metric_hash": True,
    }
    _require_equal(repeat, expected_repeat, "repeat gate")
    guard = _object(payload["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(payload["output"], "output")
    _require_exact_keys(
        output,
        {"run_root_relative_path", "matrix_gate_filename"},
        "output",
    )
    if Path(str(output["matrix_gate_filename"])).name != str(
        output["matrix_gate_filename"]
    ):
        raise ValueError("full-frame matrix gate must be a filename")


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_sha(spec["sha256"], f"{name}.sha256")


def _validate_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} is not lowercase SHA256")
