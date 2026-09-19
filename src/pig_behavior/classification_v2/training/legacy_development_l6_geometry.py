"""Parameter-matched legacy L6 geometry controls over frozen T6 features."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.models.temporal_encoders import (
    build_temporal_encoder,
)
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
    aggregate_temporal_ladder_predictions,
    build_temporal_ladder_selection,
    build_window_prediction_frame,
    load_temporal_ladder_config,
    load_temporal_ladder_view,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    GEOMETRY_DIM,
    GEOMETRY_FEATURE_NAMES,
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
    SOURCE_TYPE,
    VIEW_ID,
    LegacyL6GeometryCache,
    load_geometry_cache,
    load_geometry_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

FULL_SCOPE = "full_development_baseline"
TRAINING_SCOPES = frozenset({SHORT_SCOPE, FULL_SCOPE})
MODES = (
    "parameter_matched_zero",
    "availability_only",
    "geometry",
)
AUXILIARY_DIM = GEOMETRY_DIM + 1
MODEL_INPUT_DIM = FEATURE_DIM + AUXILIARY_DIM
EXPECTED_PARAMETER_COUNT = 69_404
EXPECTED_SHORT_TRAIN_WINDOWS = 320
EXPECTED_FULL_TRAIN_WINDOWS = 14_608
EXPECTED_VALIDATION_WINDOWS = 980
EXPECTED_SHORT_OPTIMIZER_STEPS = 30
EXPECTED_FULL_OPTIMIZER_STEPS = 1_371

SHORT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_short_config.v1"
)
FULL_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_full_config.v1"
)
NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_normalization.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_preflight.v1"
)

CONFUSION_GROUPS = {
    "rare": ("fight", "social-nose", "playwithtoy", "move"),
    "interaction": ("fight", "social-nose", "playwithtoy"),
    "feeding": ("drink", "eat"),
    "posture": ("lying", "stand", "sitting"),
    "locomotion_exploration": ("move", "explore"),
}


@dataclass(frozen=True, slots=True)
class LegacyL6GeometryConfig:
    """Hash-bound short or full geometry experiment matrix."""

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
class GeometryNormalizationState:
    """Train-position-only geometry normalization state."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    train_window_rows: int
    train_slot_exposures: int
    unique_train_frame_rows: int
    duplicate_train_slot_exposures: int
    train_window_id_sha256: str
    unique_train_frame_uid_sha256: str
    cache_manifest_sha256: str
    selection_content_sha256: str
    fit_role: str
    validation_rows_read_for_fit: int
    outer_holdout_rows_read_for_fit: int
    state_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "train_window_rows": self.train_window_rows,
            "train_slot_exposures": self.train_slot_exposures,
            "unique_train_frame_rows": self.unique_train_frame_rows,
            "duplicate_train_slot_exposures": self.duplicate_train_slot_exposures,
            "train_window_id_sha256": self.train_window_id_sha256,
            "unique_train_frame_uid_sha256": self.unique_train_frame_uid_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "selection_content_sha256": self.selection_content_sha256,
            "fit_role": self.fit_role,
            "validation_rows_read_for_fit": self.validation_rows_read_for_fit,
            "outer_holdout_rows_read_for_fit": self.outer_holdout_rows_read_for_fit,
            "fit_contract": {
                "unique_frame_uid_only": True,
                "population_standard_deviation": True,
                "missing_geometry_after_transform": 0.0,
                "validation_and_outer_excluded": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


@dataclass(frozen=True, slots=True)
class LegacyL6GeometryView:
    """Duck-typed cached view with one parameter-matched auxiliary mode."""

    base: LegacyL5CachedFeatureView
    cache: LegacyL6GeometryCache
    mode: str
    normalization: GeometryNormalizationState
    missing_modality: bool = False

    @property
    def windows(self) -> pd.DataFrame:
        return self.base.windows

    @property
    def observed_mask(self) -> np.ndarray:
        return self.base.observed_mask

    @property
    def time_delta(self) -> np.ndarray:
        return self.base.time_delta

    @property
    def targets(self) -> np.ndarray:
        return self.base.targets

    @property
    def sample_weights(self) -> np.ndarray:
        return self.base.sample_weights

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        rows = _validated_rows(positions, len(self.windows))
        visual = self.base.load_sequences(rows)
        auxiliary = self._load_auxiliary(rows)
        combined = np.concatenate([visual, auxiliary], axis=2).astype(
            np.float32,
            copy=False,
        )
        expected_shape = (len(rows), SEQUENCE_LENGTH, MODEL_INPUT_DIM)
        if combined.shape != expected_shape:
            raise ValueError(
                f"L6 combined feature shape={combined.shape}!={expected_shape}"
            )
        return combined

    def with_missing_modality(self) -> LegacyL6GeometryView:
        return replace(self, missing_modality=True)

    def _load_auxiliary(self, rows: np.ndarray) -> np.ndarray:
        available = self.cache.load_availability(rows)
        zeros = np.zeros(
            (len(rows), SEQUENCE_LENGTH, GEOMETRY_DIM),
            dtype=np.float32,
        )
        if self.missing_modality or self.mode == "parameter_matched_zero":
            geometry = zeros
            availability = np.zeros_like(available, dtype=np.float32)
        elif self.mode == "availability_only":
            geometry = zeros
            availability = available.astype(np.float32)
        elif self.mode == "geometry":
            raw = self.cache.load_geometry(rows).astype(np.float64)
            mean = np.asarray(self.normalization.mean, dtype=np.float64)
            scale = np.asarray(self.normalization.scale, dtype=np.float64)
            normalized = (raw - mean) / scale
            geometry = np.where(available[..., None], normalized, 0.0).astype(
                np.float32
            )
            availability = available.astype(np.float32)
        else:
            raise ValueError(f"unknown L6 geometry mode={self.mode}")
        auxiliary = np.concatenate(
            [geometry, availability[..., None]],
            axis=2,
        )
        observed = self.observed_mask[rows]
        auxiliary[~observed] = 0.0
        if not np.isfinite(auxiliary).all():
            raise ValueError("L6 auxiliary features contain nonfinite values")
        return auxiliary


@dataclass(frozen=True, slots=True)
class LegacyL6GeometryOutcome:
    """Selected-checkpoint outputs for one L6 geometry mode."""

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
    normalization: GeometryNormalizationState
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


class LegacyL6GeometryClassifier(nn.Module):
    """One fixed-capacity head over visual, geometry, and availability channels."""

    def __init__(
        self,
        *,
        temporal_encoder_name: str,
        hidden_dim: int,
        dropout: float,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("L6 hidden dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("L6 dropout must be in [0,1)")
        self.input_norm = nn.LayerNorm(MODEL_INPUT_DIM)
        self.projection = nn.Linear(MODEL_INPUT_DIM, hidden_dim)
        self.temporal_encoder = build_temporal_encoder(
            temporal_encoder_name,
            embedding_dim=hidden_dim,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.behavior_head = nn.Linear(hidden_dim, len(VALID_BEHAVIORS))

    def forward(
        self,
        features: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        time_delta: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.ndim != 3
            or observed_mask.ndim != 2
            or features.shape[:2] != observed_mask.shape
            or features.shape[-1] != MODEL_INPUT_DIM
        ):
            raise ValueError("L6 features/mask must be [B,T,521] and [B,T]")
        if not torch.isfinite(observed_mask).all():
            raise ValueError("L6 observed mask contains nonfinite values")
        if not torch.all((observed_mask == 0) | (observed_mask == 1)):
            raise ValueError("L6 observed mask must be binary")
        valid = observed_mask.bool()
        if not torch.isfinite(features[valid]).all():
            raise ValueError("L6 observed features contain nonfinite values")
        clean = torch.where(
            valid.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )
        projected = torch.nn.functional.gelu(
            self.projection(self.input_norm(clean))
        )
        pooled = self.temporal_encoder(
            projected,
            observed_mask,
            time_delta=time_delta,
        )
        return self.behavior_head(self.dropout(self.output_norm(pooled)))


def load_geometry_training_config(path: Path) -> LegacyL6GeometryConfig:
    """Load one L6 matrix config and verify all immutable dependencies."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6GeometryConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for name, spec_value in _object(payload["parents"], "parents").items():
        spec = _object(spec_value, f"parents.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"parent {name}",
        )
    cache = _object(payload["cache"], "cache")
    for name in ("config", "manifest", "repeat_gate"):
        spec = _object(cache[name], f"cache.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"cache {name}",
        )
    implementation = _object(payload["implementation"], "implementation")
    for name in ("core", "runtime", "frozen_engine"):
        spec = _object(implementation[name], f"implementation.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"implementation {name}",
        )
    _validate_bound_cache(config)
    if config.training_scope == FULL_SCOPE:
        _validate_full_authorization(config)
    return config


def fit_geometry_normalization(
    cache: LegacyL6GeometryCache,
    selection: TemporalLadderSelection,
) -> GeometryNormalizationState:
    """Fit geometry mean/scale from unique train frames only."""

    rows = _validated_rows(selection.train_positions, len(cache.window_index))
    if len(rows) == 0:
        raise ValueError("geometry normalization has zero train windows")
    roles = set(cache.window_index.iloc[rows]["l5_role"].astype(str))
    if roles != {"train"}:
        raise ValueError(f"geometry normalization roles={roles}")
    raw = cache.load_geometry(rows).astype(np.float64)
    available = cache.load_availability(rows)
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
    if len(slots) != expected_slots:
        raise ValueError(f"geometry normalization slot rows={len(slots)}")
    values = raw.reshape(expected_slots, GEOMETRY_DIM)
    available_flat = available.reshape(expected_slots)
    if not available_flat.any():
        raise ValueError("geometry normalization has zero available train slots")
    frame = pd.DataFrame(values, columns=GEOMETRY_FEATURE_NAMES)
    frame.insert(0, "frame_uid", slots["frame_uid"].astype(str).to_numpy())
    frame = frame.loc[available_flat].copy()
    conflicts = frame.groupby("frame_uid", sort=False)[
        list(GEOMETRY_FEATURE_NAMES)
    ].nunique(dropna=False)
    if conflicts.gt(1).any(axis=None):
        raise ValueError("repeated train frame_uid has conflicting geometry")
    unique = frame.drop_duplicates("frame_uid", keep="first").sort_values(
        "frame_uid",
        kind="mergesort",
    )
    matrix = unique[list(GEOMETRY_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("geometry normalization train matrix is nonfinite")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("geometry normalization statistics are nonfinite")
    if (scale <= 1e-12).any():
        constant = [
            GEOMETRY_FEATURE_NAMES[index]
            for index in np.flatnonzero(scale <= 1e-12)
        ]
        raise ValueError(f"geometry normalization constant features={constant}")
    train_windows = cache.window_index.iloc[rows]["window_id"].astype(str)
    semantic = {
        "schema_version": NORMALIZATION_SCHEMA,
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "train_window_rows": int(len(rows)),
        "train_slot_exposures": int(available_flat.sum()),
        "unique_train_frame_rows": int(len(unique)),
        "duplicate_train_slot_exposures": int(available_flat.sum() - len(unique)),
        "train_window_id_sha256": _ordered_sha256(train_windows),
        "unique_train_frame_uid_sha256": _ordered_sha256(unique["frame_uid"]),
        "cache_manifest_sha256": str(cache.audit["manifest_sha256"]),
        "selection_content_sha256": str(
            selection.audit["selection_content_sha256"]
        ),
        "fit_role": "train",
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "fit_contract": {
            "unique_frame_uid_only": True,
            "population_standard_deviation": True,
            "missing_geometry_after_transform": 0.0,
            "validation_and_outer_excluded": True,
        },
    }
    state_sha = _payload_sha256(semantic)
    return GeometryNormalizationState(
        feature_names=GEOMETRY_FEATURE_NAMES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        train_window_rows=int(len(rows)),
        train_slot_exposures=int(available_flat.sum()),
        unique_train_frame_rows=int(len(unique)),
        duplicate_train_slot_exposures=int(available_flat.sum() - len(unique)),
        train_window_id_sha256=semantic["train_window_id_sha256"],
        unique_train_frame_uid_sha256=semantic[
            "unique_train_frame_uid_sha256"
        ],
        cache_manifest_sha256=semantic["cache_manifest_sha256"],
        selection_content_sha256=semantic["selection_content_sha256"],
        fit_role="train",
        validation_rows_read_for_fit=0,
        outer_holdout_rows_read_for_fit=0,
        state_sha256=state_sha,
    )


def build_geometry_view(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6GeometryCache,
    *,
    mode: str,
    normalization: GeometryNormalizationState,
) -> LegacyL6GeometryView:
    """Align the cache to the frozen L5 view and construct one mode."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    _validate_cache_view_alignment(base, cache)
    if normalization.feature_names != GEOMETRY_FEATURE_NAMES:
        raise ValueError("L6 geometry normalization feature order drift")
    if normalization.cache_manifest_sha256 != cache.audit["manifest_sha256"]:
        raise ValueError("L6 geometry normalization cache hash drift")
    return LegacyL6GeometryView(
        base=base,
        cache=cache,
        mode=mode,
        normalization=normalization,
    )


def l6_feature_whitelist(mode: str) -> dict[str, Any]:
    """Return the exact fixed-width model-X contract for one control."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    visual = [f"cached_frame_feature_{index:03d}" for index in range(FEATURE_DIM)]
    auxiliary = [f"geometry_{name}" for name in GEOMETRY_FEATURE_NAMES]
    features = [*visual, *auxiliary, "geometry_available"]
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6.geometry_whitelist.v1"
        ),
        "mode": mode,
        "features": features,
        "feature_count": len(features),
        "visual_feature_count": FEATURE_DIM,
        "geometry_feature_count": GEOMETRY_DIM,
        "availability_feature_count": 1,
        "parameter_matched_input_width": MODEL_INPUT_DIM,
        "labels_paths_ids_folds_or_review_fields_in_model_x": False,
        "availability_is_behavior_evidence": False,
        "source_identifier_in_model_x": False,
    }


def preflight_geometry_mode(
    config: LegacyL6GeometryConfig,
    mode: str,
) -> dict[str, Any]:
    """Run exact CPU-only parent, cache, normalization, shape, and Git gates."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    normalization: GeometryNormalizationState | None = None
    parameter_count = 0
    output_shape: list[int] = []
    missing_output_shape: list[int] = []
    loaded_bytes = 0
    source_probe: dict[str, Any] = {}
    try:
        _, base, cache, selection = load_geometry_training_inputs(config)
        normalization = fit_geometry_normalization(cache, selection)
        view = build_geometry_view(
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
        model = build_geometry_model(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            errors.append(f"model_parameter_count={parameter_count}")
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(batch["features"]),
                torch.from_numpy(batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(batch["time_delta"]).float(),
            )
            missing_view = view.with_missing_modality()
            missing_batch, _ = frozen_engine._load_selected_batch(
                missing_view,
                sample,
                maximum_batch_bytes=int(
                    config.payload["optimization"]["maximum_loaded_batch_bytes"]
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
            errors.append("L6 CPU forward shape drift")
        source_probe = copy.deepcopy(
            cache.manifest["content_audit"]["source_probe"]
        )
        if source_probe["status"] != "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE":
            errors.append("L6 source probe status drift")
        del logits, missing_logits, model, batch, missing_batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = geometry_training_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("L6 geometry preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_PREFLIGHT"
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
            selection.audit["validation_windows"] if selection is not None else 0
        ),
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "missing_modality_output_shape": missing_output_shape,
        "maximum_loaded_batch_bytes": loaded_bytes,
        "source_probe": source_probe,
        "feature_whitelist": l6_feature_whitelist(mode),
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_geometry_core(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6GeometryCache,
    selection: TemporalLadderSelection,
    config: LegacyL6GeometryConfig,
    mode: str,
    *,
    device: torch.device | str,
) -> LegacyL6GeometryOutcome:
    """Train one geometry control and evaluate its missing-modality path."""

    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L6 geometry requested unavailable CUDA")
    _validate_selection_for_training(base, cache, selection, config)
    normalization = fit_geometry_normalization(cache, selection)
    view = build_geometry_view(
        base,
        cache,
        mode=mode,
        normalization=normalization,
    )
    optimization = _object(config.payload["optimization"], "optimization")
    seed = int(optimization["seed"])
    frozen_engine._seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model: LegacyL6GeometryClassifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = build_geometry_model(config).to(resolved_device)
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
        model.load_state_dict(best["model_state"])
        missing = _evaluate_geometry_view(
            model,
            view.with_missing_modality(),
            selection,
            config,
            device=resolved_device,
        )
        missing_groups = build_confusion_group_report(
            missing["per_class_metrics"],
            missing["confusion"],
            mode=mode,
            missing_modality=True,
        )
        epoch_metrics = best["epoch_metrics"]
        native_predictions = best["native_predictions"]
        return LegacyL6GeometryOutcome(
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
            missing_confusion_groups=missing_groups,
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


def _train_geometry_epochs(
    model: LegacyL6GeometryClassifier,
    optimizer: torch.optim.Optimizer,
    view: LegacyL6GeometryView,
    selection: TemporalLadderSelection,
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    optimization = _object(config.payload["optimization"], "optimization")
    seed = int(optimization["seed"])
    optimizer_steps = 0
    maximum_batch_bytes = 0
    best_score: tuple[float, float] | None = None
    best: dict[str, Any] | None = None
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(optimization["epochs"]) + 1):
        positions = selection.train_positions.copy()
        np.random.default_rng(seed + epoch).shuffle(positions)
        loss_mass = 0.0
        weight_mass = 0.0
        model.train()
        for batch_positions in frozen_engine._position_batches(
            positions,
            batch_size=int(optimization["batch_size"]),
        ):
            batch, loaded = frozen_engine._load_selected_batch(
                view,
                batch_positions,
                maximum_batch_bytes=int(
                    optimization["maximum_loaded_batch_bytes"]
                ),
            )
            maximum_batch_bytes = max(maximum_batch_bytes, loaded)
            loss_value, batch_weight = frozen_engine._cached_training_step(
                model,
                optimizer,
                batch,
                device=device,
                gradient_clip_norm=float(
                    optimization["gradient_clip_norm"]
                ),
            )
            optimizer_steps += 1
            loss_mass += loss_value * batch_weight
            weight_mass += batch_weight
            del batch
        if weight_mass <= 0.0:
            raise RuntimeError("L6 geometry train weight mass is empty")
        evaluation = _evaluate_geometry_view(
            model,
            view,
            selection,
            config,
            device=device,
        )
        maximum_batch_bytes = max(
            maximum_batch_bytes,
            int(evaluation["maximum_loaded_batch_bytes"]),
        )
        metrics = evaluation["validation_metrics"]
        parameter_sha = frozen_engine._state_dict_sha256(model.state_dict())
        window_sha = frozen_engine._dataframe_sha256(
            evaluation["window_predictions"]
        )
        native_sha = frozen_engine._dataframe_sha256(
            evaluation["native_predictions"]
        )
        score = (
            float(metrics["macro_f1_global_10_class"]),
            -float(metrics["nll"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = {
                **evaluation,
                "model_state": frozen_engine._clone_state_dict(
                    model.state_dict()
                ),
                "optimizer_state": frozen_engine._clone_to_cpu(
                    optimizer.state_dict()
                ),
                "best_epoch": epoch,
            }
        epoch_rows.append(
            _geometry_epoch_row(
                config,
                selection,
                mode=mode,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                train_loss=loss_mass / weight_mass,
                metrics=metrics,
                parameter_sha=parameter_sha,
                window_sha=window_sha,
                native_sha=native_sha,
            )
        )
    expected_steps = _expected_optimizer_steps(config)
    if optimizer_steps != expected_steps:
        raise RuntimeError(
            f"L6 geometry optimizer steps={optimizer_steps}!={expected_steps}"
        )
    if best is None:
        raise RuntimeError("L6 geometry checkpoint selection is empty")
    epoch_rows[int(best["best_epoch"]) - 1]["selected_checkpoint"] = True
    best["epoch_metrics"] = pd.DataFrame.from_records(epoch_rows)
    best["optimizer_steps"] = optimizer_steps
    best["maximum_loaded_batch_bytes"] = maximum_batch_bytes
    return best


def _evaluate_geometry_view(
    model: LegacyL6GeometryClassifier,
    view: LegacyL6GeometryView,
    selection: TemporalLadderSelection,
    config: LegacyL6GeometryConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    optimization = _object(config.payload["optimization"], "optimization")
    evaluation = frozen_engine._evaluate_cached_classifier(
        model,
        view,
        selection.validation_positions,
        batch_size=int(optimization["evaluation_batch_size"]),
        maximum_batch_bytes=int(optimization["maximum_loaded_batch_bytes"]),
        device=device,
    )
    windows = build_window_prediction_frame(
        view,
        selection.validation_positions,
        probabilities=evaluation["probabilities"],
        targets=evaluation["targets"],
        config=config,
        view_id=VIEW_ID,
    )
    native, metrics, per_class, confusion = (
        aggregate_temporal_ladder_predictions(
            windows,
            expected_windows_per_native=4,
            training_scope=config.training_scope,
        )
    )
    mode = view.mode
    missing = view.missing_modality
    for frame in (windows, native, per_class, confusion):
        frame["geometry_mode"] = mode
        frame["missing_modality"] = missing
    metrics = {
        **metrics,
        "geometry_mode": mode,
        "missing_modality": missing,
    }
    groups = build_confusion_group_report(
        per_class,
        confusion,
        mode=mode,
        missing_modality=missing,
    )
    return {
        "window_predictions": windows,
        "native_predictions": native,
        "validation_metrics": metrics,
        "per_class_metrics": per_class,
        "confusion": confusion,
        "confusion_groups": groups,
        "maximum_loaded_batch_bytes": int(
            evaluation["maximum_loaded_batch_bytes"]
        ),
    }


def build_confusion_group_report(
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    *,
    mode: str,
    missing_modality: bool,
) -> pd.DataFrame:
    """Summarize declared behavior groups without tuning from predictions."""

    required = {
        "behavior_label",
        "support",
        "true_positive",
        "f1",
    }
    missing = sorted(required - set(per_class.columns))
    if missing:
        raise ValueError(f"L6 per-class metrics missing columns={missing}")
    if "true_behavior" not in confusion.columns:
        raise ValueError("L6 confusion matrix is missing true_behavior")
    matrix = confusion.set_index("true_behavior")
    rows: list[dict[str, Any]] = []
    for group_name, labels in CONFUSION_GROUPS.items():
        selected = per_class.loc[per_class["behavior_label"].isin(labels)]
        if len(selected) != len(labels):
            raise ValueError(f"L6 confusion group incomplete={group_name}")
        support = int(selected["support"].sum())
        true_positive = int(selected["true_positive"].sum())
        inside = int(matrix.loc[list(labels), list(labels)].to_numpy().sum())
        rows.append(
            {
                "confusion_group": group_name,
                "behavior_labels": "|".join(labels),
                "class_count": len(labels),
                "support": support,
                "true_positive": true_positive,
                "accuracy": true_positive / support if support else 0.0,
                "macro_f1": float(selected["f1"].mean()),
                "predicted_inside_group_rate": (
                    inside / support if support else 0.0
                ),
                "geometry_mode": mode,
                "missing_modality": missing_modality,
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
                "reviewed_or_final_claim_allowed": False,
                "q2_claim_allowed": False,
            }
        )
    return pd.DataFrame.from_records(rows)


def _geometry_epoch_row(
    config: LegacyL6GeometryConfig,
    selection: TemporalLadderSelection,
    *,
    mode: str,
    epoch: int,
    optimizer_steps: int,
    train_loss: float,
    metrics: dict[str, Any],
    parameter_sha: str,
    window_sha: str,
    native_sha: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "optimizer_steps_cumulative": optimizer_steps,
        "train_native_units": selection.audit["train_native_units"],
        "train_windows": selection.audit["train_windows"],
        "train_loss": train_loss,
        "validation_native_units": selection.audit["validation_native_units"],
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
        "geometry_mode": mode,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }


def build_geometry_model(
    config: LegacyL6GeometryConfig,
) -> LegacyL6GeometryClassifier:
    """Build the one fixed-width model shared by all geometry controls."""

    model = _object(config.payload["model"], "model")
    classifier = LegacyL6GeometryClassifier(
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )
    observed = sum(parameter.numel() for parameter in classifier.parameters())
    if observed != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"L6 geometry model parameters={observed}!={EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def load_geometry_training_inputs(
    config: LegacyL6GeometryConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    LegacyL6GeometryCache,
    TemporalLadderSelection,
]:
    """Load the frozen T6 view, audited cache, and native-first selection."""

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
    cache_config = load_geometry_cache_config(
        config.bound_path("cache", "config")
    )
    cache_root = _resolve_inside(
        config.repo_root,
        str(config.payload["cache"]["root_relative_path"]),
    )
    cache = load_geometry_cache(cache_config, cache_root=cache_root)
    _validate_cache_view_alignment(base, cache)
    _validate_selection_for_training(base, cache, selection, config)
    return ladder, base, cache, selection


def _validate_selection_for_training(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6GeometryCache,
    selection: TemporalLadderSelection,
    config: LegacyL6GeometryConfig,
) -> None:
    _validate_cache_view_alignment(base, cache)
    if selection.audit.get("valid") is not True:
        raise ValueError("L6 geometry selection audit is invalid")
    if selection.audit.get("outer_holdout_rows") != 0:
        raise ValueError("L6 geometry selection exposes outer holdout")
    expected_train = (
        EXPECTED_SHORT_TRAIN_WINDOWS
        if config.training_scope == SHORT_SCOPE
        else EXPECTED_FULL_TRAIN_WINDOWS
    )
    expected = {
        "training_scope": config.training_scope,
        "view_id": VIEW_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "train_windows": expected_train,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "validation_native_units": 245,
        "source_media_reads": 0,
    }
    for field, value in expected.items():
        _require_equal(selection.audit.get(field), value, f"selection.{field}")
    expected_native = 80 if config.training_scope == SHORT_SCOPE else 3_652
    _require_equal(
        selection.audit.get("train_native_units"),
        expected_native,
        "selection.train_native_units",
    )
    observed_hash = frozen_engine._dataframe_sha256(selection.manifest)
    _require_equal(
        selection.audit.get("selection_content_sha256"),
        observed_hash,
        "selection content hash",
    )
    train = _validated_rows(selection.train_positions, len(base.windows))
    validation = _validated_rows(
        selection.validation_positions,
        len(base.windows),
    )
    if set(train).intersection(set(validation)):
        raise ValueError("L6 geometry train/validation positions overlap")
    if set(base.windows.iloc[train]["l5_role"].astype(str)) != {"train"}:
        raise ValueError("L6 geometry train role routing drift")
    if set(base.windows.iloc[validation]["l5_role"].astype(str)) != {
        "validation"
    }:
        raise ValueError("L6 geometry validation role routing drift")


def _expected_optimizer_steps(config: LegacyL6GeometryConfig) -> int:
    optimization = _object(config.payload["optimization"], "optimization")
    train_windows = (
        EXPECTED_SHORT_TRAIN_WINDOWS
        if config.training_scope == SHORT_SCOPE
        else EXPECTED_FULL_TRAIN_WINDOWS
    )
    observed = math.ceil(train_windows / int(optimization["batch_size"])) * int(
        optimization["epochs"]
    )
    expected = (
        EXPECTED_SHORT_OPTIMIZER_STEPS
        if config.training_scope == SHORT_SCOPE
        else EXPECTED_FULL_OPTIMIZER_STEPS
    )
    _require_equal(observed, expected, "optimizer-step contract")
    return expected


def geometry_training_git_guard(
    config: LegacyL6GeometryConfig,
) -> dict[str, Any]:
    """Require committed L6 sources/config and only declared user dirt."""

    guard = _object(config.payload["execution_guard"], "execution_guard")
    status = _git(
        config.repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    entries = [line for line in status.splitlines() if line.strip()]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(
        str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"]
    )
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(path).replace("\\", "/")
        for path in guard["required_tracked_paths"]
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


def implementation_hashes(config: LegacyL6GeometryConfig) -> dict[str, str]:
    implementation = _object(config.payload["implementation"], "implementation")
    return {
        name: file_sha256(config.bound_path("implementation", name))
        for name in implementation
    }


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
    _require_exact_keys(payload, required, "L6 geometry config")
    pair = (payload["schema_version"], payload["training_scope"])
    if pair not in {
        (SHORT_CONFIG_SCHEMA, SHORT_SCOPE),
        (FULL_CONFIG_SCHEMA, FULL_SCOPE),
    }:
        raise ValueError("L6 geometry schema/scope mismatch")
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
    _validate_experiment_contract(payload["experiment_contract"])
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision"},
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
        {"core", "runtime", "frozen_engine"},
        "implementation",
    )
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    _validate_selection_contract(payload["selection"])
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
    _validate_relative_path(
        output["run_root_relative_path"],
        "output run root",
    )
    if Path(str(output["matrix_gate_filename"])).name != str(
        output["matrix_gate_filename"]
    ):
        raise ValueError("L6 matrix gate filename is not a filename")


def _validate_experiment_contract(value: object) -> None:
    payload = _object(value, "experiment_contract")
    expected = {
        "experiment_id": "L6_V1_T6_GEOMETRY_ABLATION_V1",
        "parent_decision": "RETAIN_T6_SLIDING_AS_BOUNDED_LEGACY_BASELINE",
        "changed_family": "geometry_only",
        "modes": list(MODES),
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "parameter_matched": True,
        "availability_only_is_diagnostic": True,
        "availability_is_behavior_evidence": False,
        "missing_modality_inference_required": True,
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
    }
    _require_equal(payload, expected, "experiment contract")


def _validate_selection_contract(value: object) -> None:
    expected = {
        "view_id": VIEW_ID,
        "native_unit": "complete_legacy_16_frame_burst",
        "windows_per_native_unit": 4,
        "short_train_native_units": 80,
        "short_train_windows": EXPECTED_SHORT_TRAIN_WINDOWS,
        "full_train_native_units": 3_652,
        "full_train_windows": EXPECTED_FULL_TRAIN_WINDOWS,
        "validation_native_units": 245,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "event_mass_per_native_unit": 1.0,
        "normalization_fit_scope": "unique_train_frame_uid_only",
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    _require_equal(_object(value, "selection"), expected, "selection")


def _validate_model_contract(value: object) -> None:
    expected = {
        "architecture": "cached_visual_geometry_temporal_classifier_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "visual_feature_dim": FEATURE_DIM,
        "geometry_feature_dim": GEOMETRY_DIM,
        "availability_feature_dim": 1,
        "model_input_dim": MODEL_INPUT_DIM,
        "geometry_modes": list(MODES),
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
        "missing_modality_policy": "zero_geometry_and_availability_v1",
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
        "maximum_loaded_batch_bytes": 2_103_552,
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


def _validate_bound_cache(config: LegacyL6GeometryConfig) -> None:
    cache_payload = _object(config.payload["cache"], "cache")
    cache_config = load_geometry_cache_config(
        config.bound_path("cache", "config")
    )
    root = _resolve_inside(
        config.repo_root,
        str(cache_payload["root_relative_path"]),
    )
    _require_equal(cache_config.output_root, root, "geometry cache root")
    cache = load_geometry_cache(cache_config, cache_root=root)
    manifest_path = config.bound_path("cache", "manifest")
    _require_equal(
        manifest_path,
        root / "geometry_cache_manifest.json",
        "geometry cache manifest path",
    )
    _require_equal(
        cache.audit.get("manifest_sha256"),
        file_sha256(manifest_path),
        "geometry cache manifest audit hash",
    )
    repeat_gate = _read_json(config.bound_path("cache", "repeat_gate"))
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_REPEAT",
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
        _require_equal(repeat_gate.get(field), value, f"cache repeat.{field}")
    primary = _object(repeat_gate.get("primary"), "cache repeat.primary")
    _require_equal(
        primary.get("manifest_sha256"),
        file_sha256(manifest_path),
        "cache repeat primary manifest hash",
    )
    artifact = _object(
        repeat_gate.get("artifact_comparison"),
        "cache repeat.artifact_comparison",
    )
    _require_equal(
        artifact.get("all_artifact_sha256_equal"),
        True,
        "cache repeat artifact equality",
    )
    content = _object(
        repeat_gate.get("content_comparison"),
        "cache repeat.content_comparison",
    )
    _require_equal(content.get("valid"), True, "cache repeat content")


def _validate_full_authorization(config: LegacyL6GeometryConfig) -> None:
    authorization = _object(
        config.payload["full_authorization"],
        "full_authorization",
    )
    required = {
        "short_config_path",
        "short_config_sha256",
        "short_matrix_gate_path",
        "short_matrix_gate_sha256",
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
        str(authorization["short_matrix_gate_path"]),
    )
    _validate_bound_file(
        short_path,
        str(authorization["short_config_sha256"]),
        "short config authorization",
    )
    _validate_bound_file(
        gate_path,
        str(authorization["short_matrix_gate_sha256"]),
        "short matrix authorization",
    )
    short = load_geometry_training_config(short_path)
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
        "selection",
        "model",
        "optimization",
    ):
        _require_equal(
            config.payload[field],
            short.payload[field],
            f"full/short scientific binding.{field}",
        )
    gate = _read_json(gate_path)
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_MATRIX",
        "lineage_scope": LINEAGE_SCOPE,
        "short_config_sha256": short.sha256,
        "modes": list(MODES),
        "all_mode_repeat_gates_pass": True,
        "full_expansion_authorized": True,
        "valid": True,
    }
    for field, value in expected.items():
        _require_equal(gate.get(field), value, f"short matrix.{field}")


def _validate_cache_view_alignment(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6GeometryCache,
) -> None:
    windows = base.windows.reset_index(drop=True)
    index = cache.window_index.reset_index(drop=True)
    if len(windows) != len(index):
        raise ValueError("L6 geometry cache/view row count drift")
    expected_shape = (len(windows), SEQUENCE_LENGTH)
    if base.observed_mask.shape != expected_shape:
        raise ValueError("L6 geometry base observed-mask shape drift")
    if base.time_delta.shape != expected_shape:
        raise ValueError("L6 geometry base time-delta shape drift")
    if len(base.targets) != len(windows) or len(base.sample_weights) != len(
        windows
    ):
        raise ValueError("L6 geometry base target/weight row drift")
    pairs = (
        ("window_id", "window_id"),
        ("temporal_unit_key", "temporal_unit_key"),
        ("l5_role", "l5_role"),
        ("source_type", "source_type"),
        ("dataset_id", "dataset_id"),
    )
    for base_field, cache_field in pairs:
        left = windows[base_field].fillna("").astype(str).to_numpy()
        right = index[cache_field].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"L6 geometry cache/view drift={base_field}")
    if set(index["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("L6 geometry cache source_type drift")
    if set(index["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("L6 geometry cache dataset_id drift")
    if set(index["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        raise ValueError("L6 geometry cache lineage drift")
    if index["human_review_complete"].map(_strict_false).ne(True).any():
        raise ValueError("L6 geometry cache review flag drift")
    _require_equal(
        cache.audit.get("outer_holdout_slots_materialized"),
        0,
        "geometry cache outer slots",
    )
    _require_equal(
        cache.audit.get("source_media_reads"),
        0,
        "geometry cache source reads",
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


def _validate_bound_file(
    path: Path,
    expected_sha: str,
    name: str,
) -> None:
    _require_sha(expected_sha, f"{name}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"L6 geometry missing {name}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha:
        raise ValueError(
            f"L6 geometry {name} SHA256={observed}!={expected_sha}"
        )


def _resolve_inside(root: Path, value: str) -> Path:
    root_resolved = root.resolve()
    path = (root_resolved / value).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"L6 geometry path escapes repository: {value}") from error
    return path


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("L6 geometry cache rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("L6 geometry cache rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("L6 geometry cache rows contain duplicates")
    return rows


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.fillna("").astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"L6 geometry git command failed: {message}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", maxsplit=1)[1]
    return value.strip('"')


def _strict_false(value: object) -> bool:
    if value is False or value == 0 or str(value).strip().lower() == "false":
        return True
    raise ValueError(f"L6 geometry expected false flag, observed={value!r}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"L6 geometry invalid JSON={path}") from error
    return _object(payload, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"L6 geometry {name} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"L6 geometry {name} keys missing={missing} extra={extra}"
        )


def _require_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(
            f"L6 geometry {name} drift observed={observed!r} "
            f"expected={expected!r}"
        )


def _require_sha(value: object, name: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"L6 geometry {name} is not lowercase SHA256")
