"""Parameter-matched numeric-social controls for legacy L6 development."""

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
    FEATURE_DIM,
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
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    _evaluate_geometry_view,
    _object,
    _read_json,
    _require_equal,
    _require_exact_keys,
    _require_sha,
    _resolve_inside,
    _train_geometry_epochs,
    _validate_bound_file,
    _validate_cache_view_alignment,
    _validate_selection_for_training,
    geometry_training_git_guard,
    implementation_hashes,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
    SOCIAL_RELATION_DIM,
    SOCIAL_RELATION_FEATURE_NAMES,
    SOURCE_TYPE,
    VIEW_ID,
    LegacyL6SocialRelationCache,
    load_social_relation_cache,
    load_social_relation_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

MODALITY_NAME = "social_relation"
MODES = ("parameter_matched_zero", "availability_only", MODALITY_NAME)
AUXILIARY_DIM = SOCIAL_RELATION_DIM + 1
MODEL_INPUT_DIM = FEATURE_DIM + AUXILIARY_DIM
EXPECTED_PARAMETER_COUNT = 69_664
EXPECTED_SHORT_TRAIN_WINDOWS = 320
EXPECTED_VALIDATION_WINDOWS = 980
EXPECTED_SHORT_OPTIMIZER_STEPS = 30

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_short_config.v1"
)
NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_normalization.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_preflight.v1"
)
WHITELIST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_whitelist.v1"
)


@dataclass(frozen=True, slots=True)
class LegacyL6SocialRelationConfig:
    """Hash-bound short numeric-social experiment matrix."""

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
class LegacyL6SocialRelationOutcome:
    """Selected-checkpoint outputs for one numeric-social mode."""

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


def load_social_relation_training_config(
    path: Path,
) -> LegacyL6SocialRelationConfig:
    """Load a short social matrix and verify immutable dependencies."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6SocialRelationConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for section in ("parents", "implementation"):
        for name, value in _object(payload[section], section).items():
            spec = _object(value, f"{section}.{name}")
            _validate_bound_file(
                _resolve_inside(config.repo_root, str(spec["path"])),
                str(spec["sha256"]),
                f"social relation {section}.{name}",
            )
    cache = _object(payload["cache"], "cache")
    for name in ("config", "manifest", "repeat_gate"):
        spec = _object(cache[name], f"cache.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"social relation cache {name}",
        )
    _validate_bound_cache(config)
    return config


def fit_social_relation_normalization(
    cache: LegacyL6SocialRelationCache,
    selection: TemporalLadderSelection,
) -> CachedModalityNormalizationState:
    """Fit numeric-social statistics from available train slots only."""

    return fit_cached_modality_normalization(
        cache,
        selection,
        modality_name=MODALITY_NAME,
        feature_names=SOCIAL_RELATION_FEATURE_NAMES,
        identity_field="social_relation_window_slot_uid",
        schema_version=NORMALIZATION_SCHEMA,
    )


def build_social_relation_view(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6SocialRelationCache,
    *,
    mode: str,
    normalization: CachedModalityNormalizationState,
) -> LegacyL6CachedModalityView:
    """Align numeric-social values to the frozen L5 T6 view."""

    _validate_cache_view_alignment(base, cache)
    return build_cached_modality_view(
        base,
        cache,
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=SOCIAL_RELATION_FEATURE_NAMES,
        sequence_length=SEQUENCE_LENGTH,
        normalization=normalization,
    )


def l6_social_relation_feature_whitelist(mode: str) -> dict[str, Any]:
    """Return the explicit 523-wide numeric-social model-X contract."""

    return cached_modality_feature_whitelist(
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=SOCIAL_RELATION_FEATURE_NAMES,
        schema_version=WHITELIST_SCHEMA,
    )


def build_social_relation_model(
    config: LegacyL6SocialRelationConfig,
) -> LegacyL6CachedModalityClassifier:
    """Build the same-width classifier for all three controls."""

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
            "L6 social relation model parameters="
            f"{observed}!={EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def preflight_social_relation_mode(
    config: LegacyL6SocialRelationConfig,
    mode: str,
) -> dict[str, Any]:
    """Run CPU cache, shape, mask, missingness, leakage, and Git gates."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 social relation mode={mode}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    normalization: CachedModalityNormalizationState | None = None
    parameter_count = 0
    output_shape: list[int] = []
    missing_output_shape: list[int] = []
    loaded_bytes = 0
    source_probe: dict[str, Any] = {}
    availability_patterns: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    try:
        _, base, cache, selection = load_social_relation_training_inputs(config)
        normalization = fit_social_relation_normalization(cache, selection)
        view = build_social_relation_view(
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
        model = build_social_relation_model(config)
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
                time_delta=torch.from_numpy(missing_batch["time_delta"]).float(),
            )
        output_shape = list(logits.shape)
        missing_output_shape = list(missing_logits.shape)
        expected_shape = [len(sample), len(VALID_BEHAVIORS)]
        if output_shape != expected_shape or missing_output_shape != expected_shape:
            errors.append("L6 social relation CPU forward shape drift")
        content = _object(cache.manifest["content_audit"], "content_audit")
        source_probe = copy.deepcopy(_object(content["source_probe"], "probe"))
        if source_probe.get("status") != "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE":
            errors.append("L6 social relation source probe status drift")
        availability_patterns = copy.deepcopy(content["availability_patterns"])
        expected_patterns = [
            {"pattern": [0, 0, 0, 0, 0, 0], "windows": 144},
            {"pattern": [1, 1, 1, 1, 1, 1], "windows": 15_444},
        ]
        if availability_patterns != expected_patterns:
            errors.append(
                f"L6 social availability patterns={availability_patterns}"
            )
        feature_contract = {
            name: content.get(name)
            for name in (
                "numeric_social_only",
                "partner_identity_values_used",
                "top_k_partner_features_used",
                "unit_aggregate_features_used",
                "geometry_values_in_model_x",
                "motion_values_in_model_x",
                "roi_values_in_model_x",
            )
        }
        expected_features = {
            "numeric_social_only": True,
            "partner_identity_values_used": False,
            "top_k_partner_features_used": False,
            "unit_aggregate_features_used": False,
            "geometry_values_in_model_x": False,
            "motion_values_in_model_x": False,
            "roi_values_in_model_x": False,
        }
        if feature_contract != expected_features:
            errors.append(f"L6 social feature contract={feature_contract}")
        del logits, missing_logits, model, batch, missing_batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = social_relation_training_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("L6 social relation preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_PREFLIGHT"
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
        "availability_patterns": availability_patterns,
        "source_probe": source_probe,
        "cache_feature_contract": feature_contract,
        "feature_whitelist": l6_social_relation_feature_whitelist(mode),
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "partner_identity_values_in_model_x": False,
        "top_k_partner_features_in_model_x": False,
        "geometry_feature_values_in_model_x": False,
        "motion_feature_values_in_model_x": False,
        "roi_feature_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_social_relation_core(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6SocialRelationCache,
    selection: TemporalLadderSelection,
    config: LegacyL6SocialRelationConfig,
    mode: str,
    *,
    device: torch.device | str,
) -> LegacyL6SocialRelationOutcome:
    """Train one numeric-social control through the cached training loop."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 social relation mode={mode}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L6 social relation requested unavailable CUDA")
    _validate_selection_for_training(base, cache, selection, config)
    normalization = fit_social_relation_normalization(cache, selection)
    view = build_social_relation_view(
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
        model = build_social_relation_model(config).to(resolved_device)
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
        epoch_metrics = best["epoch_metrics"]
        native_predictions = best["native_predictions"]
        return LegacyL6SocialRelationOutcome(
            epoch_metrics=epoch_metrics,
            window_predictions=best["window_predictions"],
            native_predictions=native_predictions,
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
                native_predictions
            ),
            epoch_metrics_sha256=frozen_engine._dataframe_sha256(epoch_metrics),
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


def load_social_relation_training_inputs(
    config: LegacyL6SocialRelationConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    LegacyL6SocialRelationCache,
    TemporalLadderSelection,
]:
    """Load the frozen T6 view, social cache, and short selection."""

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
    cache_payload = _object(config.payload["cache"], "cache")
    cache_config = load_social_relation_cache_config(
        config.bound_path("cache", "config")
    )
    cache_root = _resolve_inside(
        config.repo_root,
        str(cache_payload["root_relative_path"]),
    )
    cache = load_social_relation_cache(cache_config, cache_root=cache_root)
    _validate_cache_view_alignment(base, cache)
    _validate_selection_for_training(base, cache, selection, config)
    return ladder, base, cache, selection


def social_relation_training_git_guard(
    config: LegacyL6SocialRelationConfig,
) -> dict[str, Any]:
    """Use the shared committed-source and dirty-path guard."""

    return geometry_training_git_guard(config)


def social_relation_implementation_hashes(
    config: LegacyL6SocialRelationConfig,
) -> dict[str, str]:
    """Hash every declared training and reusable runtime dependency."""

    return implementation_hashes(config)


def _rename_geometry_surfaces(result: dict[str, Any]) -> None:
    for value in result.values():
        if isinstance(value, pd.DataFrame) and "geometry_mode" in value.columns:
            value.rename(
                columns={"geometry_mode": "social_relation_mode"},
                inplace=True,
            )
        elif isinstance(value, dict) and "geometry_mode" in value:
            value["social_relation_mode"] = value.pop("geometry_mode")


def _validate_bound_cache(config: LegacyL6SocialRelationConfig) -> None:
    cache_payload = _object(config.payload["cache"], "cache")
    cache_config = load_social_relation_cache_config(
        config.bound_path("cache", "config")
    )
    root = _resolve_inside(
        config.repo_root,
        str(cache_payload["root_relative_path"]),
    )
    _require_equal(cache_config.output_root, root, "social relation cache root")
    cache = load_social_relation_cache(cache_config, cache_root=root)
    manifest_path = config.bound_path("cache", "manifest")
    _require_equal(
        manifest_path,
        root / "social_relation_cache_manifest.json",
        "social relation cache manifest path",
    )
    _require_equal(
        cache.audit.get("manifest_sha256"),
        file_sha256(manifest_path),
        "social relation cache manifest audit hash",
    )
    repeat_gate = _read_json(config.bound_path("cache", "repeat_gate"))
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_REPEAT",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "valid": True,
    }
    for field, value in expected.items():
        _require_equal(
            repeat_gate.get(field),
            value,
            f"social relation repeat.{field}",
        )
    primary = _object(
        repeat_gate.get("primary"),
        "social relation repeat.primary",
    )
    _require_equal(
        primary.get("manifest_sha256"),
        file_sha256(manifest_path),
        "social relation repeat primary manifest hash",
    )
    artifacts = _object(
        repeat_gate.get("artifact_comparison"),
        "social relation repeat artifacts",
    )
    _require_equal(
        artifacts.get("all_artifact_sha256_equal"),
        True,
        "social relation repeat artifact equality",
    )
    content = _object(
        repeat_gate.get("content_comparison"),
        "social relation repeat content",
    )
    _require_equal(content.get("valid"), True, "social repeat content")
    _require_equal(
        content.get("numeric_social_only"),
        True,
        "social repeat numeric-only",
    )
    _require_equal(
        content.get("top_k_partner_features_used"),
        False,
        "social repeat top-K exclusion",
    )


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
        "cache",
        "implementation",
        "selection",
        "model",
        "optimization",
        "repeat_gate",
        "execution_guard",
        "output",
    }
    _require_exact_keys(payload, required, "L6 social relation config")
    identity = {
        "schema_version": SHORT_CONFIG_SCHEMA,
        "training_scope": SHORT_SCOPE,
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
    for field, value in identity.items():
        _require_equal(payload[field], value, field)
    _validate_experiment_contract(payload["experiment_contract"])
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision", "l6_roi_full_decision"},
        "parents",
    )
    for name, value in parents.items():
        _validate_bound_spec(value, f"parents.{name}")
    cache = _object(payload["cache"], "cache")
    _require_exact_keys(
        cache,
        {"config", "manifest", "repeat_gate", "root_relative_path"},
        "cache",
    )
    for name in ("config", "manifest", "repeat_gate"):
        _validate_bound_spec(cache[name], f"cache.{name}")
    _validate_relative_path(cache["root_relative_path"], "cache root")
    implementation = _object(payload["implementation"], "implementation")
    _require_exact_keys(
        implementation,
        {
            "core",
            "runtime",
            "frozen_engine",
            "cached_modality_engine",
            "cached_modality_runtime",
            "training_loop_engine",
            "social_cache_reader",
        },
        "implementation",
    )
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    _validate_selection_contract(payload["selection"])
    _validate_model_contract(payload["model"])
    _validate_optimization_contract(payload["optimization"])
    _validate_repeat_contract(payload["repeat_gate"])
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
    _validate_relative_path(output["run_root_relative_path"], "output root")
    filename = str(output["matrix_gate_filename"])
    if Path(filename).name != filename:
        raise ValueError("L6 social relation matrix gate is not a filename")


def _validate_experiment_contract(value: object) -> None:
    expected = {
        "experiment_id": "L6_V1_T6_NUMERIC_SOCIAL_RELATION_ABLATION_V1",
        "parent_decision": (
            "DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "baseline_source": "parameter_matched_zero_without_roi_values",
        "changed_family": "numeric_social_relation_only",
        "modes": list(MODES),
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "parameter_matched": True,
        "availability_only_is_diagnostic": True,
        "availability_is_behavior_evidence": False,
        "numeric_social_only": True,
        "social_window_local_rebase": True,
        "partner_identity_values_in_model_x": False,
        "top_k_partner_features_in_model_x": False,
        "geometry_feature_values_in_model_x": False,
        "motion_feature_values_in_model_x": False,
        "roi_feature_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "missing_modality_inference_required": True,
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(_object(value, "experiment_contract"), expected, "experiment")


def _validate_selection_contract(value: object) -> None:
    expected = {
        "view_id": VIEW_ID,
        "native_unit": "complete_legacy_16_frame_burst",
        "windows_per_native_unit": 4,
        "short_train_native_units": 80,
        "short_train_windows": EXPECTED_SHORT_TRAIN_WINDOWS,
        "validation_native_units": 245,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "event_mass_per_native_unit": 1.0,
        "normalization_fit_scope": (
            "unique_train_social_relation_window_slot_uid_only"
        ),
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    _require_equal(_object(value, "selection"), expected, "selection")


def _validate_model_contract(value: object) -> None:
    expected = {
        "architecture": "cached_visual_social_relation_temporal_classifier_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "visual_feature_dim": FEATURE_DIM,
        "social_relation_feature_dim": SOCIAL_RELATION_DIM,
        "availability_feature_dim": 1,
        "model_input_dim": MODEL_INPUT_DIM,
        "social_relation_modes": list(MODES),
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
        "missing_modality_policy": (
            "zero_social_relation_and_availability_v1"
        ),
    }
    _require_equal(_object(value, "model"), expected, "model")


def _validate_optimization_contract(value: object) -> None:
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
        "maximum_peak_vram_fraction": 0.7,
        "oom_retry_allowed": False,
        "one_run_per_fresh_process": True,
        "maximum_loaded_batch_bytes": 2_200_000,
    }
    _require_equal(_object(value, "optimization"), expected, "optimization")


def _validate_repeat_contract(value: object) -> None:
    expected = {
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
    _require_equal(_object(value, "repeat_gate"), expected, "repeat gate")


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _validate_relative_path(spec["path"], f"{name}.path")
    _require_sha(spec["sha256"], f"{name}.sha256")


def _validate_relative_path(value: object, name: str) -> None:
    text = str(value)
    path = Path(text)
    if not text.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
