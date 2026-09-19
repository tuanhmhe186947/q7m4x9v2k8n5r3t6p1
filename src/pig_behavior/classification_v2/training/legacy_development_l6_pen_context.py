"""Parameter-matched pen-context ablation on the frozen legacy T6 lineage."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
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
    CachedModalityNormalizationState,
    LegacyL6CachedModalityClassifier,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    GeometryNormalizationState,
    _evaluate_geometry_view,
    _train_geometry_epochs,
    fit_geometry_normalization,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    GEOMETRY_DIM,
    GEOMETRY_FEATURE_NAMES,
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
    SOURCE_TYPE,
    LegacyL6GeometryCache,
    load_geometry_cache,
    load_geometry_cache_config,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion import (
    fit_motion_normalization,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    MOTION_DIM,
    MOTION_FEATURE_NAMES,
    LegacyL6MotionCache,
    LegacyL6MotionCacheConfig,
    load_motion_cache,
)
from pig_behavior.classification_v2.training.legacy_development_l6_pen_context_cache import (
    PEN_DIM,
    PEN_FEATURE_NAMES,
    PEN_STATIC_FEATURE_COUNT,
    LegacyL6PenContextCache,
    load_pen_context_cache,
    load_pen_context_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

MODALITY_NAME = "pen_context"
MODES = ("parameter_matched_zero", "availability_only", MODALITY_NAME)
MODEL_INPUT_DIM = (
    FEATURE_DIM + GEOMETRY_DIM + 1 + MOTION_DIM + 1 + PEN_DIM + 1
)
EXPECTED_PARAMETER_COUNT = 71_874
EXPECTED_SHORT_TRAIN_WINDOWS = 320
EXPECTED_VALIDATION_WINDOWS = 980
EXPECTED_SHORT_OPTIMIZER_STEPS = 30

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_short_config.v1"
)
PEN_NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_normalization.v1"
)
BUNDLE_NORMALIZATION_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_bundle_normalization.v1"
)
PREFLIGHT_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_preflight.v1"
)
WHITELIST_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_whitelist.v1"
)


@dataclass(frozen=True, slots=True)
class LegacyL6PenContextConfig:
    """Hash-bound short experiment with a predeclared promotion contract."""

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
        value = str(self.payload["output"]["run_root_relative_path"])
        return _resolve_inside(self.repo_root, value)

    def bound_path(self, section: str, name: str | None = None) -> Path:
        value: Any = self.payload[section]
        if name is not None:
            value = _object(value, section)[name]
        spec = _object(value, f"{section}.{name}" if name else section)
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6CompositeCache:
    """Three aligned caches; geometry and motion are always active."""

    geometry: LegacyL6GeometryCache
    motion: LegacyL6MotionCache
    pen: LegacyL6PenContextCache

    @property
    def window_index(self) -> pd.DataFrame:
        return self.pen.window_index

    @property
    def slot_index(self) -> pd.DataFrame:
        return self.pen.slot_index


@dataclass(frozen=True, slots=True)
class PenFeatureNormalizationState:
    """Per-feature train-only state using frame or pair identities."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    identity_kinds: tuple[str, ...]
    unique_identity_rows: tuple[int, ...]
    available_slot_exposures: tuple[int, ...]
    constant_features: tuple[str, ...]
    train_window_rows: int
    train_window_id_sha256: str
    identity_sha256: tuple[str, ...]
    cache_manifest_sha256: str
    selection_content_sha256: str
    fit_role: str
    validation_rows_read_for_fit: int
    outer_holdout_rows_read_for_fit: int
    state_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PEN_NORMALIZATION_SCHEMA,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "identity_kinds": list(self.identity_kinds),
            "unique_identity_rows": list(self.unique_identity_rows),
            "available_slot_exposures": list(
                self.available_slot_exposures
            ),
            "constant_features": list(self.constant_features),
            "train_window_rows": self.train_window_rows,
            "train_window_id_sha256": self.train_window_id_sha256,
            "identity_sha256": list(self.identity_sha256),
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "selection_content_sha256": self.selection_content_sha256,
            "fit_role": self.fit_role,
            "validation_rows_read_for_fit": self.validation_rows_read_for_fit,
            "outer_holdout_rows_read_for_fit": (
                self.outer_holdout_rows_read_for_fit
            ),
            "fit_contract": {
                "static_features_use_unique_frame_uid": True,
                "pair_features_use_unique_pen_pair_uid": True,
                "population_standard_deviation": True,
                "constant_train_feature_scale_falls_back_to_one": True,
                "validation_and_outer_excluded": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


@dataclass(frozen=True, slots=True)
class PenContextNormalizationBundle:
    """One shared normalization state for all parameter-matched modes."""

    geometry: GeometryNormalizationState
    motion: CachedModalityNormalizationState
    pen: PenFeatureNormalizationState
    state_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_NORMALIZATION_SCHEMA,
            "geometry": self.geometry.to_payload(),
            "motion": self.motion.to_payload(),
            "pen": self.pen.to_payload(),
            "fit_role": "train",
            "validation_rows_read_for_fit": 0,
            "outer_holdout_rows_read_for_fit": 0,
            "all_modes_share_exact_state": True,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


@dataclass(frozen=True, slots=True)
class LegacyL6PenContextView:
    """Visual + fixed geometry/motion + optional pen-context view."""

    base: LegacyL5CachedFeatureView
    cache: LegacyL6CompositeCache
    mode: str
    normalization: PenContextNormalizationBundle
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

    @property
    def model_input_dim(self) -> int:
        return MODEL_INPUT_DIM

    def with_missing_modality(self) -> LegacyL6PenContextView:
        """Remove only pen context; fixed geometry and motion remain."""

        return replace(self, missing_modality=True)

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        rows = _validated_rows(positions, len(self.windows))
        visual = self.base.load_sequences(rows)
        geometry = _normalized_modality(
            self.cache.geometry.load_geometry(rows),
            self.cache.geometry.load_availability(rows),
            self.normalization.geometry.mean,
            self.normalization.geometry.scale,
        )
        motion = _normalized_modality(
            self.cache.motion.load_motion(rows),
            self.cache.motion.load_availability(rows),
            self.normalization.motion.mean,
            self.normalization.motion.scale,
        )
        pen = self._load_pen(rows)
        combined = np.concatenate(
            [visual, geometry, motion, pen],
            axis=2,
        ).astype(np.float32, copy=False)
        expected = (len(rows), SEQUENCE_LENGTH, MODEL_INPUT_DIM)
        if combined.shape != expected:
            raise ValueError(f"pen context combined shape={combined.shape}")
        observed = self.observed_mask[rows]
        combined[~observed] = 0.0
        if not np.isfinite(combined).all():
            raise ValueError("pen context model input contains nonfinite values")
        return combined

    def _load_pen(self, rows: np.ndarray) -> np.ndarray:
        branch_available = self.cache.pen.load_availability(rows)
        feature_available = self.cache.pen.load_feature_availability(rows)
        zeros = np.zeros(
            (len(rows), SEQUENCE_LENGTH, PEN_DIM),
            dtype=np.float32,
        )
        if self.missing_modality or self.mode == "parameter_matched_zero":
            values = zeros
            availability = np.zeros_like(branch_available, dtype=np.float32)
        elif self.mode == "availability_only":
            values = zeros
            availability = branch_available.astype(np.float32)
        elif self.mode == MODALITY_NAME:
            raw = self.cache.pen.load_pen(rows).astype(np.float64)
            mean = np.asarray(self.normalization.pen.mean, dtype=np.float64)
            scale = np.asarray(self.normalization.pen.scale, dtype=np.float64)
            normalized = (raw - mean) / scale
            values = np.where(
                feature_available,
                normalized,
                0.0,
            ).astype(np.float32)
            availability = branch_available.astype(np.float32)
        else:
            raise ValueError(f"unknown pen context mode={self.mode}")
        return np.concatenate(
            [values, availability[..., None]],
            axis=2,
        )


@dataclass(frozen=True, slots=True)
class LegacyL6PenContextOutcome:
    """Selected-checkpoint packet consumed by the shared runtime."""

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
    normalization: PenContextNormalizationBundle
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


def load_pen_context_training_config(
    path: Path,
) -> LegacyL6PenContextConfig:
    """Load and hash-verify one short pen-context experiment."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6PenContextConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for section in ("parents", "implementation"):
        for name, value in _object(payload[section], section).items():
            _validate_bound_spec(config, value, f"{section}.{name}")
    for cache_name, value in _object(payload["caches"], "caches").items():
        cache = _object(value, f"caches.{cache_name}")
        for name in ("config", "manifest"):
            _validate_bound_spec(
                config,
                cache[name],
                f"caches.{cache_name}.{name}",
            )
        if cache_name == "pen":
            _validate_bound_spec(
                config,
                cache["repeat_gate"],
                "caches.pen.repeat_gate",
            )
    _validate_bound_caches(config)
    return config


def fit_pen_context_normalization(
    cache: LegacyL6CompositeCache,
    selection: TemporalLadderSelection,
) -> PenContextNormalizationBundle:
    """Fit all transforms from the selected training windows only."""

    geometry = fit_geometry_normalization(cache.geometry, selection)
    motion = fit_motion_normalization(cache.motion, selection)
    pen = fit_pen_feature_normalization(cache.pen, selection)
    semantic = {
        "schema_version": BUNDLE_NORMALIZATION_SCHEMA,
        "geometry": geometry.to_payload(),
        "motion": motion.to_payload(),
        "pen": pen.to_payload(),
        "fit_role": "train",
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "all_modes_share_exact_state": True,
    }
    return PenContextNormalizationBundle(
        geometry=geometry,
        motion=motion,
        pen=pen,
        state_sha256=_payload_sha256(semantic),
    )


def fit_pen_feature_normalization(
    cache: LegacyL6PenContextCache,
    selection: TemporalLadderSelection,
) -> PenFeatureNormalizationState:
    """Deduplicate static values by frame and derivatives by frame pair."""

    rows = _validated_rows(selection.train_positions, len(cache.window_index))
    roles = set(cache.window_index.iloc[rows]["l5_role"].astype(str))
    if roles != {"train"}:
        raise ValueError(f"pen normalization roles={roles}")
    raw = cache.load_pen(rows).astype(np.float64)
    available = cache.load_feature_availability(rows)
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
        raise ValueError(f"pen normalization slot rows={len(slots)}")
    flat = raw.reshape(expected_slots, PEN_DIM)
    flat_available = available.reshape(expected_slots, PEN_DIM)
    means: list[float] = []
    scales: list[float] = []
    identity_kinds: list[str] = []
    unique_counts: list[int] = []
    exposures: list[int] = []
    identity_hashes: list[str] = []
    constants: list[str] = []
    for index, name in enumerate(PEN_FEATURE_NAMES):
        identity_column = (
            "frame_uid" if index < PEN_STATIC_FEATURE_COUNT else "pen_pair_uid"
        )
        mask = flat_available[:, index]
        if not mask.any():
            raise ValueError(f"pen normalization has no available {name}")
        identities = slots[identity_column].fillna("").astype(str).to_numpy()
        if np.any(identities[mask] == ""):
            raise ValueError(f"pen normalization blank identity for {name}")
        frame = pd.DataFrame(
            {"identity": identities[mask], "value": flat[mask, index]}
        )
        conflicts = frame.groupby("identity", sort=False)["value"].nunique(
            dropna=False
        )
        if conflicts.gt(1).any():
            raise ValueError(f"pen repeated identity conflicts for {name}")
        unique = frame.drop_duplicates("identity").sort_values(
            "identity",
            kind="mergesort",
        )
        values = unique["value"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"pen normalization nonfinite feature={name}")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if scale <= 1e-12:
            scale = 1.0
            constants.append(name)
        means.append(mean)
        scales.append(scale)
        identity_kinds.append(identity_column)
        unique_counts.append(len(unique))
        exposures.append(int(mask.sum()))
        identity_hashes.append(_ordered_sha256(unique["identity"]))
    train_windows = cache.window_index.iloc[rows]["window_id"].astype(str)
    semantic = {
        "schema_version": PEN_NORMALIZATION_SCHEMA,
        "feature_names": list(PEN_FEATURE_NAMES),
        "mean": means,
        "scale": scales,
        "identity_kinds": identity_kinds,
        "unique_identity_rows": unique_counts,
        "available_slot_exposures": exposures,
        "constant_features": constants,
        "train_window_rows": len(rows),
        "train_window_id_sha256": _ordered_sha256(train_windows),
        "identity_sha256": identity_hashes,
        "cache_manifest_sha256": str(cache.audit["manifest_sha256"]),
        "selection_content_sha256": str(
            selection.audit["selection_content_sha256"]
        ),
        "fit_role": "train",
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "fit_contract": {
            "static_features_use_unique_frame_uid": True,
            "pair_features_use_unique_pen_pair_uid": True,
            "population_standard_deviation": True,
            "constant_train_feature_scale_falls_back_to_one": True,
            "validation_and_outer_excluded": True,
        },
    }
    return PenFeatureNormalizationState(
        feature_names=PEN_FEATURE_NAMES,
        mean=tuple(means),
        scale=tuple(scales),
        identity_kinds=tuple(identity_kinds),
        unique_identity_rows=tuple(unique_counts),
        available_slot_exposures=tuple(exposures),
        constant_features=tuple(constants),
        train_window_rows=len(rows),
        train_window_id_sha256=semantic["train_window_id_sha256"],
        identity_sha256=tuple(identity_hashes),
        cache_manifest_sha256=semantic["cache_manifest_sha256"],
        selection_content_sha256=semantic["selection_content_sha256"],
        fit_role="train",
        validation_rows_read_for_fit=0,
        outer_holdout_rows_read_for_fit=0,
        state_sha256=_payload_sha256(semantic),
    )


def build_pen_context_view(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6CompositeCache,
    *,
    mode: str,
    normalization: PenContextNormalizationBundle,
) -> LegacyL6PenContextView:
    if mode not in MODES:
        raise ValueError(f"unknown pen context mode={mode}")
    _validate_cache_alignment(base, cache)
    expected_hashes = (
        (
            normalization.geometry.cache_manifest_sha256,
            cache.geometry.audit["manifest_sha256"],
        ),
        (
            normalization.motion.cache_manifest_sha256,
            cache.motion.audit["manifest_sha256"],
        ),
        (
            normalization.pen.cache_manifest_sha256,
            cache.pen.audit["manifest_sha256"],
        ),
    )
    if any(left != right for left, right in expected_hashes):
        raise ValueError("pen context normalization cache hash drift")
    return LegacyL6PenContextView(
        base=base,
        cache=cache,
        mode=mode,
        normalization=normalization,
    )


def pen_context_feature_whitelist(mode: str) -> dict[str, Any]:
    """Return the exact 540-wide model-X contract."""

    if mode not in MODES:
        raise ValueError(f"unknown pen context mode={mode}")
    visual = [f"cached_frame_feature_{index:03d}" for index in range(FEATURE_DIM)]
    geometry = [f"geometry_{name}" for name in GEOMETRY_FEATURE_NAMES]
    motion = [f"motion_{name}" for name in MOTION_FEATURE_NAMES]
    pen = [f"pen_{name}" for name in PEN_FEATURE_NAMES]
    features = [
        *visual,
        *geometry,
        "geometry_available",
        *motion,
        "motion_available",
        *pen,
        "pen_context_available",
    ]
    return {
        "schema_version": WHITELIST_SCHEMA,
        "mode": mode,
        "features": features,
        "feature_count": len(features),
        "parameter_matched_input_width": MODEL_INPUT_DIM,
        "visual_feature_count": FEATURE_DIM,
        "geometry_feature_count": GEOMETRY_DIM,
        "motion_feature_count": MOTION_DIM,
        "pen_feature_count": PEN_DIM,
        "availability_feature_count": 3,
        "geometry_and_motion_fixed_active": True,
        "availability_is_behavior_evidence": False,
        "binary_pen_state_in_model_x": False,
        "labels_paths_ids_folds_review_or_unit_aggregates_in_model_x": False,
        "source_identifier_in_model_x": False,
    }


def build_pen_context_model(
    config: LegacyL6PenContextConfig,
) -> LegacyL6CachedModalityClassifier:
    model = _object(config.payload["model"], "model")
    classifier = LegacyL6CachedModalityClassifier(
        input_dim=MODEL_INPUT_DIM,
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )
    parameters = sum(value.numel() for value in classifier.parameters())
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"pen context parameters={parameters}!={EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def preflight_pen_context_mode(
    config: LegacyL6PenContextConfig,
    mode: str,
) -> dict[str, Any]:
    """CPU-only lineage, normalization, forward/backward, and leakage gate."""

    if mode not in MODES:
        raise ValueError(f"unknown pen context mode={mode}")
    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    selection: TemporalLadderSelection | None = None
    normalization: PenContextNormalizationBundle | None = None
    parameter_count = 0
    output_shape: list[int] = []
    missing_output_shape: list[int] = []
    loaded_bytes = 0
    gradient_groups = 0
    try:
        _, base, cache, selection = load_pen_context_training_inputs(config)
        normalization = fit_pen_context_normalization(cache, selection)
        view = build_pen_context_view(
            base,
            cache,
            mode=mode,
            normalization=normalization,
        )
        sample = selection.train_positions[:32]
        batch, loaded_bytes = frozen_engine._load_selected_batch(
            view,
            sample,
            maximum_batch_bytes=int(
                config.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        model = build_pen_context_model(config)
        parameter_count = sum(value.numel() for value in model.parameters())
        features = torch.from_numpy(batch["features"])
        observed = torch.from_numpy(batch["observed_mask"]).float()
        timing = torch.from_numpy(batch["time_delta"]).float()
        logits = model(features, observed, time_delta=timing)
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.from_numpy(batch["targets"]).long(),
        )
        loss.backward()
        gradients = [
            value.grad
            for value in model.parameters()
            if value.requires_grad and value.grad is not None
        ]
        gradient_groups = len(gradients)
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            errors.append("pen context one-batch gradients are invalid")
        missing_view = view.with_missing_modality()
        missing_batch, _ = frozen_engine._load_selected_batch(
            missing_view,
            sample,
            maximum_batch_bytes=int(
                config.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        with torch.inference_mode():
            missing_logits = model(
                torch.from_numpy(missing_batch["features"]),
                torch.from_numpy(missing_batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(missing_batch["time_delta"]).float(),
            )
        output_shape = list(logits.shape)
        missing_output_shape = list(missing_logits.shape)
        expected_shape = [len(sample), len(VALID_BEHAVIORS)]
        if output_shape != expected_shape or missing_output_shape != expected_shape:
            errors.append("pen context CPU forward shape drift")
        fixed_width = FEATURE_DIM + GEOMETRY_DIM + 1 + MOTION_DIM + 1
        if not np.array_equal(
            view.load_sequences(sample)[..., :fixed_width],
            missing_view.load_sequences(sample)[..., :fixed_width],
        ):
            errors.append("missing pen modality changed fixed geometry/motion")
        del logits, missing_logits, model, batch, missing_batch
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = pen_context_training_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("pen context preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_PREFLIGHT"
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
        "normalization_fit_role": "train",
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "train_windows": (
            selection.audit["train_windows"] if selection is not None else 0
        ),
        "validation_windows": (
            selection.audit["validation_windows"] if selection is not None else 0
        ),
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "missing_modality_output_shape": missing_output_shape,
        "one_batch_gradient_tensor_count": gradient_groups,
        "maximum_loaded_batch_bytes": loaded_bytes,
        "feature_whitelist": pen_context_feature_whitelist(mode),
        "source_probe": {
            "status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
            "source_type": SOURCE_TYPE,
        },
        "geometry_and_motion_fixed_active": True,
        "pen_availability_only_diagnostic": True,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def train_pen_context_core(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6CompositeCache,
    selection: TemporalLadderSelection,
    config: LegacyL6PenContextConfig,
    mode: str,
    *,
    device: torch.device | str,
) -> LegacyL6PenContextOutcome:
    """Train one control with the existing deterministic cached engine."""

    if mode not in MODES:
        raise ValueError(f"unknown pen context mode={mode}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pen context requested unavailable CUDA")
    _validate_training_selection(base, cache, selection, config)
    normalization = fit_pen_context_normalization(cache, selection)
    view = build_pen_context_view(
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
        model = build_pen_context_model(config).to(resolved_device)
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
        return LegacyL6PenContextOutcome(
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


def load_pen_context_training_inputs(
    config: LegacyL6PenContextConfig,
) -> tuple[
    TemporalLadderConfig,
    LegacyL5CachedFeatureView,
    LegacyL6CompositeCache,
    TemporalLadderSelection,
]:
    """Load the exact frozen view, caches, and short native-unit selection."""

    ladder = load_temporal_ladder_config(
        config.bound_path("parents", "temporal_ladder_config")
    )
    _, base, _ = load_temporal_ladder_view(ladder, "t6_sliding")
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
        "t6_sliding",
    )
    caches = _object(config.payload["caches"], "caches")
    geometry_spec = _object(caches["geometry"], "caches.geometry")
    geometry_config = load_geometry_cache_config(
        _cache_bound_path(config, "geometry", "config")
    )
    geometry_root = _resolve_inside(
        config.repo_root,
        str(geometry_spec["root_relative_path"]),
    )
    geometry = load_geometry_cache(geometry_config, cache_root=geometry_root)

    motion_spec = _object(caches["motion"], "caches.motion")
    motion_config_path = _cache_bound_path(config, "motion", "config")
    motion_config = LegacyL6MotionCacheConfig(
        path=motion_config_path,
        payload=_read_json(motion_config_path),
        repo_root=config.repo_root,
    )
    motion_root = _resolve_inside(
        config.repo_root,
        str(motion_spec["root_relative_path"]),
    )
    motion = load_motion_cache(motion_config, cache_root=motion_root)

    pen_spec = _object(caches["pen"], "caches.pen")
    pen_config_path = _cache_bound_path(config, "pen", "config")
    pen_config = load_pen_context_cache_config(pen_config_path)
    pen_root = _resolve_inside(
        config.repo_root,
        str(pen_spec["root_relative_path"]),
    )
    pen = load_pen_context_cache(pen_config, cache_root=pen_root)
    cache = LegacyL6CompositeCache(geometry=geometry, motion=motion, pen=pen)
    _validate_cache_alignment(base, cache)
    _validate_training_selection(base, cache, selection, config)
    return ladder, base, cache, selection


def pen_context_training_git_guard(
    config: LegacyL6PenContextConfig,
) -> dict[str, Any]:
    """Record all dirt, but fail only on mutable raw-data involvement."""

    status = _git(
        config.repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    entries = [line for line in status.splitlines() if line.strip()]
    paths = sorted(_status_path(line) for line in entries)
    data_paths = [path for path in paths if path.startswith("data/")]
    classification_prefixes = (
        ".agents/memory/",
        "configs/classification_v2/",
        "docs/CLASSIFICATION_V2_",
        "outputs/classification_v2/",
        "scripts/classification_v2/",
        "src/pig_behavior/classification_v2/",
        "tests/test_classification_v2_",
    )
    classification_paths = [
        path for path in paths if path.startswith(classification_prefixes)
    ]
    unrelated_paths = sorted(set(paths).difference(classification_paths))
    errors = [f"raw_data_worktree_changes={data_paths}"] if data_paths else []
    return {
        "code_sha": _git(config.repo_root, "rev-parse", "HEAD").strip(),
        "dirty_worktree": bool(entries),
        "dirty_path_count": len(paths),
        "dirty_path_sha256": _payload_sha256({"paths": paths}),
        "classification_dirty_paths": classification_paths,
        "unrelated_dirty_path_count": len(unrelated_paths),
        "unrelated_dirty_path_sha256": _payload_sha256(
            {"paths": unrelated_paths}
        ),
        "raw_data_dirty_paths": data_paths,
        "relevant_implementation_hashes": pen_context_implementation_hashes(
            config
        ),
        "unrelated_changes_ignored_as_preexisting": True,
        "errors": errors,
        "valid": not errors,
    }


def pen_context_implementation_hashes(
    config: LegacyL6PenContextConfig,
) -> dict[str, str]:
    return {
        name: file_sha256(
            _resolve_inside(
                config.repo_root,
                str(_object(value, f"implementation.{name}")["path"]),
            )
        )
        for name, value in _object(
            config.payload["implementation"],
            "implementation",
        ).items()
    }


def _normalized_modality(
    raw: np.ndarray,
    available: np.ndarray,
    mean_values: tuple[float, ...],
    scale_values: tuple[float, ...],
) -> np.ndarray:
    mean = np.asarray(mean_values, dtype=np.float64)
    scale = np.asarray(scale_values, dtype=np.float64)
    normalized = (raw.astype(np.float64) - mean) / scale
    values = np.where(available[..., None], normalized, 0.0).astype(np.float32)
    return np.concatenate(
        [values, available.astype(np.float32)[..., None]],
        axis=2,
    )


def _validate_cache_alignment(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6CompositeCache,
) -> None:
    reference = base.windows["window_id"].astype(str).reset_index(drop=True)
    for name, value in (
        ("geometry", cache.geometry),
        ("motion", cache.motion),
        ("pen", cache.pen),
    ):
        observed = value.window_index["window_id"].astype(str).reset_index(
            drop=True
        )
        if not reference.equals(observed):
            raise ValueError(f"{name} cache window IDs differ from T6 base")
    fields = ["window_id", "slot_index", "frame_uid"]
    slot_reference = cache.geometry.slot_index[fields].astype(str).reset_index(
        drop=True
    )
    for name, value in (("motion", cache.motion), ("pen", cache.pen)):
        observed = value.slot_index[fields].astype(str).reset_index(drop=True)
        if not slot_reference.equals(observed):
            raise ValueError(f"{name} cache slot keys differ from geometry")


def _validate_training_selection(
    base: LegacyL5CachedFeatureView,
    cache: LegacyL6CompositeCache,
    selection: TemporalLadderSelection,
    config: LegacyL6PenContextConfig,
) -> None:
    _validate_cache_alignment(base, cache)
    if config.training_scope != SHORT_SCOPE:
        raise ValueError("pen context currently permits only short scope")
    if len(selection.train_positions) != EXPECTED_SHORT_TRAIN_WINDOWS:
        raise ValueError("pen context short train-window count drift")
    if len(selection.validation_positions) != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError("pen context validation-window count drift")
    train_roles = set(
        cache.window_index.iloc[selection.train_positions]["l5_role"].astype(str)
    )
    validation_roles = set(
        cache.window_index.iloc[selection.validation_positions]["l5_role"].astype(str)
    )
    if train_roles != {"train"} or validation_roles != {"validation"}:
        raise ValueError("pen context selection role drift")
    if set(selection.train_positions).intersection(selection.validation_positions):
        raise ValueError("pen context train/validation positions overlap")


def _rename_geometry_surfaces(result: dict[str, Any]) -> None:
    for value in result.values():
        if isinstance(value, pd.DataFrame) and "geometry_mode" in value.columns:
            value.rename(
                columns={"geometry_mode": "pen_context_mode"},
                inplace=True,
            )
        elif isinstance(value, dict) and "geometry_mode" in value:
            value["pen_context_mode"] = value.pop("geometry_mode")


def _validate_bound_caches(config: LegacyL6PenContextConfig) -> None:
    caches = _object(config.payload["caches"], "caches")
    for name, value in caches.items():
        spec = _object(value, f"caches.{name}")
        root = _resolve_inside(config.repo_root, str(spec["root_relative_path"]))
        manifest = root / str(spec["manifest_filename"])
        expected = _object(spec["manifest"], f"caches.{name}.manifest")
        if manifest != _resolve_inside(config.repo_root, str(expected["path"])):
            raise ValueError(f"{name} cache manifest path drift")
        if file_sha256(manifest) != str(expected["sha256"]):
            raise ValueError(f"{name} cache manifest hash drift")
    pen_repeat = _read_json(_cache_bound_path(config, "pen", "repeat_gate"))
    expected_repeat = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_REPEAT",
        "lineage_scope": LINEAGE_SCOPE,
        "all_artifact_sha256_equal": True,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": [],
        "valid": True,
    }
    for field, expected_value in expected_repeat.items():
        if pen_repeat.get(field) != expected_value:
            raise ValueError(f"pen cache repeat {field} drift")


def _cache_bound_path(
    config: LegacyL6PenContextConfig,
    cache_name: str,
    item: str,
) -> Path:
    caches = _object(config.payload["caches"], "caches")
    cache = _object(caches[cache_name], f"caches.{cache_name}")
    spec = _object(cache[item], f"caches.{cache_name}.{item}")
    return _resolve_inside(config.repo_root, str(spec["path"]))


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
        "caches",
        "implementation",
        "selection",
        "model",
        "optimization",
        "repeat_gate",
        "promotion_contract",
        "execution_guard",
        "output",
    }
    if set(payload) != required:
        raise ValueError(
            "pen context config keys differ: "
            f"missing={sorted(required - set(payload))},"
            f"extra={sorted(set(payload) - required)}"
        )
    identity = {
        "schema_version": CONFIG_SCHEMA,
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
    for field, expected in identity.items():
        if payload[field] != expected:
            raise ValueError(f"pen context config {field} drift")
    experiment = _object(payload["experiment_contract"], "experiment_contract")
    expected_experiment = {
        "experiment_id": "L6_V1_T6_PEN_CONTEXT_ABLATION_V1",
        "changed_family": "fixed_camera_pen_boundary_context_only",
        "modes": list(MODES),
        "fixed_modalities": ["actor_visual", "geometry", "motion"],
        "parameter_matched": True,
        "availability_only_is_diagnostic": True,
        "availability_is_behavior_evidence": False,
        "primary_metric": "validation_native_unit_macro_f1_global_10_class",
        "uncertainty_cluster": "video_key",
        "outer_predictions_used_for_model_selection": False,
        "legacy_only_decision": True,
        "merged_reviewed_reassessment_required": True,
    }
    if experiment != expected_experiment:
        raise ValueError("pen context experiment contract drift")
    selection = _object(payload["selection"], "selection")
    expected_selection = {
        "view_id": "t6_sliding",
        "native_unit": "complete_legacy_16_frame_burst",
        "windows_per_native_unit": 4,
        "short_train_native_units": 80,
        "short_train_windows": EXPECTED_SHORT_TRAIN_WINDOWS,
        "validation_native_units": 245,
        "validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "event_mass_per_native_unit": 1.0,
        "normalization_fit_scope": "train_unique_frame_or_pair_only",
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
    }
    if selection != expected_selection:
        raise ValueError("pen context selection contract drift")
    model = _object(payload["model"], "model")
    expected_model = {
        "architecture": "cached_visual_geometry_motion_pen_temporal_v1",
        "feature_control_id": "V1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "visual_feature_dim": FEATURE_DIM,
        "geometry_feature_dim": GEOMETRY_DIM,
        "motion_feature_dim": MOTION_DIM,
        "pen_feature_dim": PEN_DIM,
        "availability_feature_dim": 3,
        "model_input_dim": MODEL_INPUT_DIM,
        "pen_context_modes": list(MODES),
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_probability_aggregation": "mean_window_probability_v1",
        "missing_modality_policy": "zero_pen_only_keep_geometry_motion_v1",
    }
    if model != expected_model:
        raise ValueError("pen context model contract drift")
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
        "maximum_loaded_batch_bytes": 2_300_000,
    }
    if optimization != expected_optimization:
        raise ValueError("pen context optimization contract drift")
    repeat = _object(payload["repeat_gate"], "repeat_gate")
    if repeat != {
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
    }:
        raise ValueError("pen context repeat contract drift")
    promotion = _object(payload["promotion_contract"], "promotion_contract")
    if promotion != {
        "minimum_macro_f1_gain": 0.01,
        "minimum_focus_group_macro_f1_gain": 0.01,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260717,
    }:
        raise ValueError("pen context promotion contract drift")
    caches = _object(payload["caches"], "caches")
    if set(caches) != {"geometry", "motion", "pen"}:
        raise ValueError("pen context cache set drift")
    for name, value in caches.items():
        cache = _object(value, f"caches.{name}")
        expected_keys = {
            "config",
            "manifest",
            "manifest_filename",
            "root_relative_path",
        }
        if name == "pen":
            expected_keys.add("repeat_gate")
        if set(cache) != expected_keys:
            raise ValueError(f"pen context cache keys drift={name}")
        _validate_hash_spec(cache["config"], f"caches.{name}.config")
        _validate_hash_spec(cache["manifest"], f"caches.{name}.manifest")
        if name == "pen":
            _validate_hash_spec(
                cache["repeat_gate"],
                "caches.pen.repeat_gate",
            )
    for section in ("parents", "implementation"):
        for name, value in _object(payload[section], section).items():
            _validate_hash_spec(value, f"{section}.{name}")
    guard = _object(payload["execution_guard"], "execution_guard")
    if guard != {
        "record_dirty_worktree": True,
        "allow_hash_bound_uncommitted_classification_changes": True,
        "forbid_raw_data_worktree_changes": True,
        "classification_only": True,
    }:
        raise ValueError("pen context execution guard drift")
    output = _object(payload["output"], "output")
    if set(output) != {"run_root_relative_path", "matrix_gate_filename"}:
        raise ValueError("pen context output contract keys drift")


def _validate_bound_spec(
    config: LegacyL6PenContextConfig,
    value: object,
    name: str,
) -> None:
    spec = _object(value, name)
    path = _resolve_inside(config.repo_root, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"{name} hash mismatch")


def _validate_hash_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name} keys drift")
    sha = str(spec["sha256"])
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"{name}.sha256 invalid")


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("pen context rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("pen context rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("pen context rows contain duplicates")
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _object(payload, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {path}") from error
    return path


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


__all__ = [
    "EXPECTED_PARAMETER_COUNT",
    "LegacyL6CompositeCache",
    "LegacyL6PenContextConfig",
    "LegacyL6PenContextView",
    "MODES",
    "MODEL_INPUT_DIM",
    "PenContextNormalizationBundle",
    "PenFeatureNormalizationState",
    "build_pen_context_model",
    "build_pen_context_view",
    "fit_pen_context_normalization",
    "fit_pen_feature_normalization",
    "load_pen_context_training_config",
    "load_pen_context_training_inputs",
    "pen_context_feature_whitelist",
    "pen_context_implementation_hashes",
    "pen_context_training_git_guard",
    "preflight_pen_context_mode",
    "train_pen_context_core",
]
