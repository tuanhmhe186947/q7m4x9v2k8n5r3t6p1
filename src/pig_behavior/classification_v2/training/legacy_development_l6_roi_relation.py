"""Parameter-matched legacy L6 all-class ROI-relation controls."""

from __future__ import annotations

import copy
import gc
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
    LegacyL6CachedModalityClassifier,
    LegacyL6CachedModalityView,
    _ordered_sha256,
    _payload_sha256,
    _validated_rows,
    build_cached_modality_view,
    cached_modality_feature_whitelist,
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
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    LINEAGE_SCOPE,
    ROI_RELATION_DIM,
    ROI_RELATION_FEATURE_NAMES,
    SEQUENCE_LENGTH,
    SOURCE_TYPE,
    VIEW_ID,
    LegacyL6ROIRelationCache,
    load_roi_relation_cache,
    load_roi_relation_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

MODALITY_NAME = "roi_relation"
MODES = ("parameter_matched_zero", "availability_only", MODALITY_NAME)
AUXILIARY_DIM = ROI_RELATION_DIM + 1
MODEL_INPUT_DIM = FEATURE_DIM + AUXILIARY_DIM
EXPECTED_PARAMETER_COUNT = 70_704
EXPECTED_SHORT_TRAIN_WINDOWS = 320
EXPECTED_FULL_TRAIN_WINDOWS = 14_608
EXPECTED_VALIDATION_WINDOWS = 980
EXPECTED_SHORT_OPTIMIZER_STEPS = 30
EXPECTED_FULL_OPTIMIZER_STEPS = 1_371

FULL_SCOPE = "full_development_baseline"

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_short_config.v1"
)
FULL_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_full_config.v1"
)
FULL_AUTHORIZATION_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "roi_relation_full_authorization_gate.v1"
)
NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_normalization.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_preflight.v1"
)
WHITELIST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_whitelist.v1"
)


@dataclass(frozen=True, slots=True)
class LegacyL6ROIRelationConfig:
    """Hash-bound short or full ROI-relation experiment matrix."""

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
class ROIRelationNormalizationState:
    """Train-frame-only ROI normalization with explicit constant columns."""

    schema_version: str
    modality_name: str
    identity_field: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant_feature_names: tuple[str, ...]
    train_window_rows: int
    train_slot_exposures: int
    unique_train_identity_rows: int
    duplicate_train_slot_exposures: int
    train_window_id_sha256: str
    unique_train_identity_sha256: str
    cache_manifest_sha256: str
    selection_content_sha256: str
    fit_role: str
    validation_rows_read_for_fit: int
    outer_holdout_rows_read_for_fit: int
    state_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "modality_name": self.modality_name,
            "identity_field": self.identity_field,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "constant_feature_names": list(self.constant_feature_names),
            "train_window_rows": self.train_window_rows,
            "train_slot_exposures": self.train_slot_exposures,
            "unique_train_identity_rows": self.unique_train_identity_rows,
            "duplicate_train_slot_exposures": (
                self.duplicate_train_slot_exposures
            ),
            "train_window_id_sha256": self.train_window_id_sha256,
            "unique_train_identity_sha256": (
                self.unique_train_identity_sha256
            ),
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "selection_content_sha256": self.selection_content_sha256,
            "fit_role": self.fit_role,
            "validation_rows_read_for_fit": self.validation_rows_read_for_fit,
            "outer_holdout_rows_read_for_fit": (
                self.outer_holdout_rows_read_for_fit
            ),
            "fit_contract": {
                "unique_available_identity_only": True,
                "population_standard_deviation": True,
                "constant_features_zero_centered_unit_scale": True,
                "missing_modality_after_transform": 0.0,
                "validation_and_outer_excluded": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


@dataclass(frozen=True, slots=True)
class LegacyL6ROIRelationOutcome:
    """Selected-checkpoint outputs for one L6 ROI-relation mode."""

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
    normalization: ROIRelationNormalizationState
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


def load_roi_relation_training_config(path: Path) -> LegacyL6ROIRelationConfig:
    """Load one ROI matrix config and verify immutable dependencies."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6ROIRelationConfig(
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
                f"ROI relation {section}.{name}",
            )
    cache = _object(payload["cache"], "cache")
    for name in ("config", "manifest", "repeat_gate"):
        spec = _object(cache[name], f"cache.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"ROI relation cache {name}",
        )
    _validate_bound_cache(config)
    if config.training_scope == FULL_SCOPE:
        _validate_full_authorization(config)
    return config


def fit_roi_relation_normalization(
    cache: LegacyL6ROIRelationCache,
    selection: TemporalLadderSelection,
) -> ROIRelationNormalizationState:
    """Fit ROI mean/scale from unique available training frames only."""

    rows = _validated_rows(selection.train_positions, len(cache.window_index))
    if set(cache.window_index.iloc[rows]["l5_role"].astype(str)) != {"train"}:
        raise ValueError("ROI normalization includes non-training windows")
    raw = cache.load_roi_relation(rows).astype(np.float64)
    available = cache.load_availability(rows)
    expected = (len(rows), SEQUENCE_LENGTH, ROI_RELATION_DIM)
    if raw.shape != expected or available.shape != expected[:2]:
        raise ValueError("ROI normalization tensor shape drift")
    row_order = {int(value): index for index, value in enumerate(rows)}
    slots = cache.slot_index.loc[
        cache.slot_index["cache_row"].astype(int).isin(row_order)
    ].copy()
    slots["train_row_order"] = slots["cache_row"].astype(int).map(row_order)
    slots = slots.sort_values(
        ["train_row_order", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_slots = len(rows) * SEQUENCE_LENGTH
    if len(slots) != expected_slots or "frame_uid" not in slots.columns:
        raise ValueError("ROI normalization slot identity drift")
    matrix = raw.reshape(expected_slots, ROI_RELATION_DIM)
    available_flat = available.reshape(expected_slots)
    identities = slots["frame_uid"].fillna("").astype(str).to_numpy()
    if not available_flat.all() or np.any(identities == ""):
        raise ValueError("ROI normalization has unavailable or blank frames")
    frame = pd.DataFrame(matrix, columns=ROI_RELATION_FEATURE_NAMES)
    frame.insert(0, "frame_uid", identities)
    conflicts = frame.groupby("frame_uid", sort=False)[
        list(ROI_RELATION_FEATURE_NAMES)
    ].nunique(dropna=False)
    if conflicts.gt(1).any(axis=None):
        raise ValueError("repeated ROI frame_uid has conflicting values")
    unique = frame.drop_duplicates("frame_uid", keep="first").sort_values(
        "frame_uid",
        kind="mergesort",
    )
    unique_matrix = unique[list(ROI_RELATION_FEATURE_NAMES)].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(unique_matrix).all():
        raise ValueError("ROI normalization train matrix is nonfinite")
    mean = unique_matrix.mean(axis=0)
    raw_scale = unique_matrix.std(axis=0, ddof=0)
    constant_mask = raw_scale <= 1e-12
    scale = np.where(constant_mask, 1.0, raw_scale)
    constant_features = tuple(
        ROI_RELATION_FEATURE_NAMES[index]
        for index in np.flatnonzero(constant_mask)
    )
    train_windows = cache.window_index.iloc[rows]["window_id"].astype(str)
    semantic = {
        "schema_version": NORMALIZATION_SCHEMA,
        "modality_name": MODALITY_NAME,
        "identity_field": "frame_uid",
        "feature_names": list(ROI_RELATION_FEATURE_NAMES),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "constant_feature_names": list(constant_features),
        "train_window_rows": int(len(rows)),
        "train_slot_exposures": int(expected_slots),
        "unique_train_identity_rows": int(len(unique)),
        "duplicate_train_slot_exposures": int(expected_slots - len(unique)),
        "train_window_id_sha256": _ordered_sha256(train_windows),
        "unique_train_identity_sha256": _ordered_sha256(unique["frame_uid"]),
        "cache_manifest_sha256": str(cache.audit["manifest_sha256"]),
        "selection_content_sha256": str(
            selection.audit["selection_content_sha256"]
        ),
        "fit_role": "train",
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "fit_contract": {
            "unique_available_identity_only": True,
            "population_standard_deviation": True,
            "constant_features_zero_centered_unit_scale": True,
            "missing_modality_after_transform": 0.0,
            "validation_and_outer_excluded": True,
        },
    }
    return ROIRelationNormalizationState(
        schema_version=NORMALIZATION_SCHEMA,
        modality_name=MODALITY_NAME,
        identity_field="frame_uid",
        feature_names=ROI_RELATION_FEATURE_NAMES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        constant_feature_names=constant_features,
        train_window_rows=int(len(rows)),
        train_slot_exposures=int(expected_slots),
        unique_train_identity_rows=int(len(unique)),
        duplicate_train_slot_exposures=int(expected_slots - len(unique)),
        train_window_id_sha256=semantic["train_window_id_sha256"],
        unique_train_identity_sha256=semantic[
            "unique_train_identity_sha256"
        ],
        cache_manifest_sha256=semantic["cache_manifest_sha256"],
        selection_content_sha256=semantic["selection_content_sha256"],
        fit_role="train",
        validation_rows_read_for_fit=0,
        outer_holdout_rows_read_for_fit=0,
        state_sha256=_payload_sha256(semantic),
    )


def build_roi_relation_view(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6ROIRelationCache,
    *,
    mode: str,
    normalization: ROIRelationNormalizationState,
) -> LegacyL6CachedModalityView:
    """Align the ROI cache to the frozen L5 T6 view."""

    _validate_cache_view_alignment(base, cache)
    return build_cached_modality_view(
        base,
        cache,
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=ROI_RELATION_FEATURE_NAMES,
        sequence_length=SEQUENCE_LENGTH,
        normalization=normalization,
    )


def l6_roi_relation_feature_whitelist(mode: str) -> dict[str, Any]:
    """Return the explicit 531-wide ROI-relation model-X contract."""

    return cached_modality_feature_whitelist(
        mode=mode,
        active_mode=MODALITY_NAME,
        modality_name=MODALITY_NAME,
        feature_names=ROI_RELATION_FEATURE_NAMES,
        schema_version=WHITELIST_SCHEMA,
    )


def build_roi_relation_model(
    config: LegacyL6ROIRelationConfig,
) -> LegacyL6CachedModalityClassifier:
    """Build the fixed-width model shared by all ROI controls."""

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
            "L6 ROI relation model parameters="
            f"{observed}!={EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def preflight_roi_relation_mode(
    config: LegacyL6ROIRelationConfig,
    mode: str,
) -> dict[str, Any]:
    """Run CPU-only cache, normalization, shape, leakage, and Git gates."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 ROI relation mode={mode}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    normalization: ROIRelationNormalizationState | None = None
    parameter_count = 0
    output_shape: list[int] = []
    missing_output_shape: list[int] = []
    loaded_bytes = 0
    source_probe: dict[str, Any] = {}
    availability_pattern: list[int] = []
    feature_contract: dict[str, Any] = {}
    try:
        _, base, cache, selection = load_roi_relation_training_inputs(config)
        normalization = fit_roi_relation_normalization(cache, selection)
        view = build_roi_relation_view(
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
        model = build_roi_relation_model(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
            errors.append("L6 ROI relation CPU forward shape drift")
        content = cache.manifest["content_audit"]
        source_probe = copy.deepcopy(content["source_probe"])
        if source_probe["status"] != "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE":
            errors.append("L6 ROI relation source probe status drift")
        availability_pattern = list(content["availability_pattern"])
        if availability_pattern != [1, 1, 1, 1, 1, 1]:
            errors.append(
                f"L6 ROI relation availability pattern={availability_pattern}"
            )
        feature_contract = copy.deepcopy(cache.manifest["feature_contract"])
        forbidden_flags = {
            "target_selected_roi_fields_used": False,
            "unit_aggregate_features_used": False,
            "geometry_values_used": False,
            "motion_values_used": False,
            "labels_ids_paths_or_folds_in_model_x": False,
        }
        for field, value in forbidden_flags.items():
            if feature_contract.get(field) != value:
                errors.append(f"L6 ROI relation feature contract {field} drift")
        del logits, missing_logits, model, batch, missing_batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = roi_relation_training_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("L6 ROI relation preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_ROI_RELATION_PREFLIGHT"
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
        "normalization_constant_features": (
            list(normalization.constant_feature_names)
            if normalization is not None
            else []
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
        "availability_pattern": availability_pattern,
        "source_probe": source_probe,
        "feature_contract": feature_contract,
        "feature_whitelist": l6_roi_relation_feature_whitelist(mode),
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "target_selected_roi_fields_in_model_x": False,
        "geometry_feature_values_in_model_x": False,
        "motion_feature_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_roi_relation_core(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6ROIRelationCache,
    selection: TemporalLadderSelection,
    config: LegacyL6ROIRelationConfig,
    mode: str,
    *,
    device: torch.device | str,
) -> LegacyL6ROIRelationOutcome:
    """Train one ROI control through the proven cached training loop."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 ROI relation mode={mode}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L6 ROI relation requested unavailable CUDA")
    _validate_selection_for_training(base, cache, selection, config)
    normalization = fit_roi_relation_normalization(cache, selection)
    view = build_roi_relation_view(
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
        model = build_roi_relation_model(config).to(resolved_device)
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
        return LegacyL6ROIRelationOutcome(
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
                frozen_engine._dataframe_sha256(missing["native_predictions"])
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


def load_roi_relation_training_inputs(
    config: LegacyL6ROIRelationConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    LegacyL6ROIRelationCache,
    TemporalLadderSelection,
]:
    """Load the frozen T6 view, audited ROI cache, and selection."""

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
    cache_config = load_roi_relation_cache_config(
        config.bound_path("cache", "config")
    )
    cache_root = _resolve_inside(
        config.repo_root,
        str(cache_payload["root_relative_path"]),
    )
    cache = load_roi_relation_cache(cache_config, cache_root=cache_root)
    _validate_cache_view_alignment(base, cache)
    _validate_selection_for_training(base, cache, selection, config)
    return ladder, base, cache, selection


def roi_relation_training_git_guard(
    config: LegacyL6ROIRelationConfig,
) -> dict[str, Any]:
    """Use the shared L6 committed-source and dirty-path guard."""

    return geometry_training_git_guard(config)


def roi_relation_implementation_hashes(
    config: LegacyL6ROIRelationConfig,
) -> dict[str, str]:
    """Hash all declared core, runtime, frozen, and reusable engines."""

    return implementation_hashes(config)


def _rename_geometry_surfaces(result: dict[str, Any]) -> None:
    for value in result.values():
        if isinstance(value, pd.DataFrame) and "geometry_mode" in value.columns:
            value.rename(
                columns={"geometry_mode": "roi_relation_mode"},
                inplace=True,
            )
        elif isinstance(value, dict) and "geometry_mode" in value:
            value["roi_relation_mode"] = value.pop("geometry_mode")


def _validate_bound_cache(config: LegacyL6ROIRelationConfig) -> None:
    cache_payload = _object(config.payload["cache"], "cache")
    cache_config = load_roi_relation_cache_config(
        config.bound_path("cache", "config")
    )
    root = _resolve_inside(
        config.repo_root,
        str(cache_payload["root_relative_path"]),
    )
    _require_equal(cache_config.output_root, root, "ROI relation cache root")
    cache = load_roi_relation_cache(cache_config, cache_root=root)
    manifest_path = config.bound_path("cache", "manifest")
    _require_equal(
        manifest_path,
        root / "roi_relation_cache_manifest.json",
        "ROI relation cache manifest path",
    )
    _require_equal(
        cache.audit.get("manifest_sha256"),
        file_sha256(manifest_path),
        "ROI relation cache manifest audit hash",
    )
    repeat_gate = _read_json(config.bound_path("cache", "repeat_gate"))
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE_REPEAT",
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
            f"ROI relation repeat.{field}",
        )
    primary = _object(repeat_gate.get("primary"), "ROI relation repeat.primary")
    _require_equal(
        primary.get("manifest_sha256"),
        file_sha256(manifest_path),
        "ROI relation repeat primary manifest hash",
    )
    artifacts = _object(
        repeat_gate.get("artifact_comparison"),
        "ROI relation repeat artifacts",
    )
    _require_equal(
        artifacts.get("all_artifact_sha256_equal"),
        True,
        "ROI relation repeat artifact equality",
    )
    content = _object(
        repeat_gate.get("content_comparison"),
        "ROI relation repeat content",
    )
    _require_equal(content.get("valid"), True, "ROI relation repeat content")


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
    if payload.get("training_scope") == FULL_SCOPE:
        required.add("full_authorization")
    _require_exact_keys(payload, required, "L6 ROI relation config")
    pair = (payload["schema_version"], payload["training_scope"])
    if pair not in {
        (SHORT_CONFIG_SCHEMA, SHORT_SCOPE),
        (FULL_CONFIG_SCHEMA, FULL_SCOPE),
    }:
        raise ValueError("L6 ROI relation schema/scope mismatch")
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
    for field, value in identity.items():
        _require_equal(payload[field], value, field)
    _validate_experiment_contract(payload["experiment_contract"])
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision", "l6_motion_decision"},
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
        {"core", "runtime", "frozen_engine", "cached_modality_engine"},
        "implementation",
    )
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    _validate_selection_contract(
        payload["selection"],
        scope=str(payload["training_scope"]),
    )
    _validate_model_contract(payload["model"])
    _validate_optimization_contract(payload["optimization"])
    _validate_repeat_contract(
        payload["repeat_gate"],
        scope=str(payload["training_scope"]),
    )
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
    if Path(str(output["matrix_gate_filename"])).name != str(
        output["matrix_gate_filename"]
    ):
        raise ValueError("L6 ROI relation matrix gate is not a filename")


def _validate_experiment_contract(value: object) -> None:
    expected = {
        "experiment_id": "L6_V1_T6_ALL_CLASS_ROI_RELATION_ABLATION_V1",
        "parent_decision": (
            "DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE"
        ),
        "baseline_source": "same_width_parameter_matched_zero",
        "changed_family": "all_class_roi_relation_only",
        "modes": list(MODES),
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "parameter_matched": True,
        "availability_only_is_diagnostic": True,
        "availability_is_behavior_evidence": False,
        "all_roi_classes_exposed_independently": True,
        "target_selected_roi_fields_in_model_x": False,
        "geometry_feature_values_in_model_x": False,
        "motion_feature_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "missing_modality_inference_required": True,
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(_object(value, "experiment_contract"), expected, "experiment")


def _validate_selection_contract(value: object, *, scope: str) -> None:
    expected = {
        "view_id": VIEW_ID,
        "native_unit": "complete_legacy_16_frame_burst",
        "windows_per_native_unit": 4,
        "short_train_native_units": 80,
        "short_train_windows": EXPECTED_SHORT_TRAIN_WINDOWS,
        "validation_native_units": 245,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "event_mass_per_native_unit": 1.0,
        "normalization_fit_scope": "unique_train_frame_uid_only",
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    if scope == FULL_SCOPE:
        expected["full_train_native_units"] = 3_652
        expected["full_train_windows"] = EXPECTED_FULL_TRAIN_WINDOWS
    _require_equal(_object(value, "selection"), expected, "selection")


def _validate_model_contract(value: object) -> None:
    expected = {
        "architecture": "cached_visual_roi_relation_temporal_classifier_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "visual_feature_dim": FEATURE_DIM,
        "roi_relation_feature_dim": ROI_RELATION_DIM,
        "availability_feature_dim": 1,
        "model_input_dim": MODEL_INPUT_DIM,
        "roi_relation_modes": list(MODES),
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
        "missing_modality_policy": "zero_roi_relation_and_availability_v1",
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


def _validate_repeat_contract(value: object, *, scope: str) -> None:
    expected = {
        "required_runs_per_mode": 2 if scope == SHORT_SCOPE else 1,
        "require_fresh_process": True,
        "require_distinct_process_ids": scope == SHORT_SCOPE,
        "require_non_overlapping_execution": scope == SHORT_SCOPE,
        "require_identical_selection_hash": scope == SHORT_SCOPE,
        "require_identical_normalization_hash": scope == SHORT_SCOPE,
        "require_identical_parameter_hash": scope == SHORT_SCOPE,
        "require_identical_window_prediction_hash": scope == SHORT_SCOPE,
        "require_identical_native_prediction_hash": scope == SHORT_SCOPE,
        "require_identical_epoch_metric_hash": scope == SHORT_SCOPE,
    }
    _require_equal(_object(value, "repeat_gate"), expected, "repeat gate")


def _validate_full_authorization(config: LegacyL6ROIRelationConfig) -> None:
    authorization = _object(
        config.payload["full_authorization"],
        "full_authorization",
    )
    required = {
        "short_config_path",
        "short_config_sha256",
        "authorization_gate_path",
        "authorization_gate_sha256",
        "authorized_training_scope",
    }
    _require_exact_keys(authorization, required, "full_authorization")
    _require_equal(
        authorization["authorized_training_scope"],
        FULL_SCOPE,
        "authorized training scope",
    )
    short_path = _resolve_inside(
        config.repo_root,
        str(authorization["short_config_path"]),
    )
    gate_path = _resolve_inside(
        config.repo_root,
        str(authorization["authorization_gate_path"]),
    )
    _validate_bound_file(
        short_path,
        str(authorization["short_config_sha256"]),
        "ROI short config authorization",
    )
    _validate_bound_file(
        gate_path,
        str(authorization["authorization_gate_sha256"]),
        "ROI full authorization gate",
    )
    short = load_roi_relation_training_config(short_path)
    for field in (
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
        "model",
        "optimization",
    ):
        _require_equal(
            config.payload[field],
            short.payload[field],
            f"full/short scientific binding.{field}",
        )
    full_selection = copy.deepcopy(
        _object(config.payload["selection"], "full selection")
    )
    full_selection.pop("full_train_native_units")
    full_selection.pop("full_train_windows")
    _require_equal(
        full_selection,
        short.payload["selection"],
        "full/short scientific binding.selection",
    )
    gate = _read_json(gate_path)
    expected = {
        "schema_version": FULL_AUTHORIZATION_GATE_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_DECISION",
        "lineage_scope": LINEAGE_SCOPE,
        "short_config_path": str(authorization["short_config_path"]),
        "short_config_sha256": short.sha256,
        "technical_matrix_status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_MATRIX"
        ),
        "paired_decision_status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_DECISION"
        ),
        "paired_decision": "RETAIN_ROI_RELATION_FOR_FULL_LEGACY_DEVELOPMENT",
        "paired_decision_required": True,
        "authorized_modes": list(MODES),
        "modes": list(MODES),
        "all_mode_repeat_gates_pass": True,
        "full_expansion_authorized": True,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    for field, value in expected.items():
        _require_equal(gate.get(field), value, f"ROI full gate.{field}")
    bound_fields = (
        ("technical_matrix_path", "technical_matrix_sha256"),
        ("paired_decision_config_path", "paired_decision_config_sha256"),
        ("paired_decision_path", "paired_decision_sha256"),
    )
    for path_field, hash_field in bound_fields:
        path = _resolve_inside(config.repo_root, str(gate[path_field]))
        _validate_bound_file(path, str(gate[hash_field]), path_field)
    matrix = _read_json(
        _resolve_inside(config.repo_root, str(gate["technical_matrix_path"]))
    )
    _require_equal(matrix.get("short_config_sha256"), short.sha256, "matrix config")
    _require_equal(matrix.get("full_expansion_authorized"), True, "matrix gate")
    decision = _read_json(
        _resolve_inside(config.repo_root, str(gate["paired_decision_path"]))
    )
    decision_payload = _object(decision.get("decision"), "paired decision")
    _require_equal(
        decision_payload.get("decision"),
        "RETAIN_ROI_RELATION_FOR_FULL_LEGACY_DEVELOPMENT",
        "paired decision value",
    )
    _require_equal(
        decision_payload.get("full_roi_relation_expansion_authorized"),
        True,
        "paired decision authorization",
    )


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
