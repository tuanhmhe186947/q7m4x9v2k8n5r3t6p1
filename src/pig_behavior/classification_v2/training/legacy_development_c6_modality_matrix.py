"""Paired C6 modality screening on the frozen legacy 16-frame universe."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.features.pen_context import (
    PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS,
    REQUIRED_PEN_CONTEXT_INPUT_COLUMNS,
    build_pen_context_features,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
    export_spatial_sequences,
)
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as frozen_engine,
)
from pig_behavior.classification_v2.training.legacy_c6_prepared_source import (
    load_legacy_c6_prepared_source,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_training import (
    LegacyL5CachedShortSelection,
    LegacyL5CachedTrainingConfig,
    compute_legacy_l5_native_metrics,
)
from pig_behavior.classification_v2.training.legacy_development_l6_cached_modality import (
    LegacyL6CachedModalityClassifier,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    CONFUSION_GROUPS,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    GEOMETRY_FEATURE_NAMES,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    MOTION_FEATURE_NAMES,
    MOTION_QUALITY_FIELDS,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_cache import (
    ROI_AVAILABILITY_FIELDS,
    ROI_RELATION_FEATURE_NAMES,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    SOCIAL_RELATION_FEATURE_NAMES,
)
from pig_behavior.classification_v2.training.legacy_development_temporal_base_selection import (
    build_training_adapter,
    derive_temporal_base_view,
    load_temporal_base_selection_config,
    load_temporal_base_source,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = "classification_v2.legacy_development.c6_modality_matrix.v1"
CONFIG_SCHEMA_V2 = "classification_v2.legacy_development.c6_modality_matrix.v2"
CONFIG_SCHEMA_V3 = "classification_v2.legacy_development.c6_modality_matrix.v3"
CONFIG_SCHEMA_V4 = "classification_v2.legacy_development.c6_modality_matrix.v4"
CACHE_SCHEMA = "classification_v2.legacy_development.c6_modality_cache.v1"
RUN_SCHEMA = "classification_v2.legacy_development.c6_modality_run.v1"
MATRIX_SCHEMA = "classification_v2.legacy_development.c6_modality_decision.v1"
MATRIX_SCHEMA_V2 = (
    "classification_v2.legacy_development.c6_combined_modality_decision.v2"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
C6_OFFSETS = (5, 6, 7, 8, 9, 10)
SEQUENCE_LENGTH = len(C6_OFFSETS)
CONTROL_MODES = ("parameter_matched_zero", "availability_only", "real")
SINGLE_MODALITY_FAMILY = "single_optional_modality"
COMBINED_ALL7_FAMILY = "combined_all7_modalities"
COMBINED_ALL7_MODE = "combined_all7"
PEN_STATIC_FEATURE_COUNT = 3

MODALITY_FEATURES: dict[str, tuple[str, ...]] = {
    "geometry": tuple(GEOMETRY_FEATURE_NAMES),
    "motion": tuple(MOTION_FEATURE_NAMES),
    "roi": tuple(ROI_RELATION_FEATURE_NAMES),
    "numeric_social": tuple(SOCIAL_RELATION_FEATURE_NAMES),
    "pen_context": tuple(PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS),
    "union_context": tuple(f"union_feature_{index:03d}" for index in range(512)),
    "full_frame_context": tuple(
        f"full_frame_feature_{index:03d}" for index in range(512)
    ),
}

GROUP_LABELS = {
    **CONFUSION_GROUPS,
    "roi_behavior": ("drink", "eat", "playwithtoy"),
}


@dataclass(frozen=True, slots=True)
class C6MatrixConfig:
    """Hash-bound configuration for one short or development matrix."""

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
        return _resolve_inside(
            self.repo_root,
            str(self.payload["output"]["root_relative_path"]),
        )

    @property
    def cache_root(self) -> Path:
        return self.output_root / "cache"

    def bound_path(self, section: str, name: str | None = None) -> Path:
        value: Any = self.payload[section]
        if name is not None:
            value = _object(value, section)[name]
        spec = _object(value, f"{section}.{name}" if name else section)
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class C6ModalityCache:
    """Immutable C6 arrays aligned one-to-one with Stage A native units."""

    root: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    manifest_sha256: str

    def load_values(self, modality: str, rows: np.ndarray) -> np.ndarray:
        return self._load_array(modality, "values", rows, np.float32)

    def load_feature_mask(self, modality: str, rows: np.ndarray) -> np.ndarray:
        return self._load_array(modality, "feature_mask", rows, np.bool_)

    def load_availability(self, modality: str, rows: np.ndarray) -> np.ndarray:
        return self._load_array(modality, "availability", rows, np.bool_)

    def _load_array(
        self,
        modality: str,
        kind: str,
        rows: np.ndarray,
        dtype: np.dtype[Any] | type[np.generic],
    ) -> np.ndarray:
        if modality not in MODALITY_FEATURES:
            raise ValueError(f"unknown C6 modality={modality}")
        indices = _validated_rows(rows, len(self.window_index))
        filename = self.manifest["modalities"][modality]["artifacts"][kind][
            "filename"
        ]
        mapping = np.load(self.root / filename, mmap_mode="r")
        try:
            return np.asarray(mapping[indices], dtype=dtype).copy()
        finally:
            _close_memmap(mapping)


@dataclass(frozen=True, slots=True)
class C6NormalizationState:
    """Per-feature train-only normalization with identity de-duplication."""

    modality: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    unique_identity_rows: tuple[int, ...]
    available_exposures: tuple[int, ...]
    constant_features: tuple[str, ...]
    train_native_units: int
    validation_rows_read_for_fit: int
    outer_rows_read_for_fit: int
    cache_manifest_sha256: str
    state_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "classification_v2.c6_normalization.v1",
            "modality": self.modality,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "unique_identity_rows": list(self.unique_identity_rows),
            "available_exposures": list(self.available_exposures),
            "constant_features": list(self.constant_features),
            "train_native_units": self.train_native_units,
            "validation_rows_read_for_fit": self.validation_rows_read_for_fit,
            "outer_rows_read_for_fit": self.outer_rows_read_for_fit,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class C6MatrixView:
    """Actor A128 inputs with single or combined explicitly masked modalities."""

    base: LegacyL5CachedFeatureView
    cache: C6ModalityCache
    modality: str | None
    control: str
    normalization: C6NormalizationState | None
    combined_modalities: tuple[str, ...] = ()
    combined_normalizations: tuple[C6NormalizationState, ...] = ()
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
    def mode_id(self) -> str:
        if self.combined_modalities:
            return f"{COMBINED_ALL7_MODE}__{self.control}"
        if self.modality is None:
            return "actor_only"
        return f"{self.modality}__{self.control}"

    @property
    def input_dim(self) -> int:
        if self.combined_modalities:
            auxiliary_dim = sum(
                len(MODALITY_FEATURES[name]) + 1
                for name in self.combined_modalities
            )
            return FEATURE_DIM + auxiliary_dim
        if self.modality is None:
            return FEATURE_DIM
        return FEATURE_DIM + len(MODALITY_FEATURES[self.modality]) + 1

    def with_missing_modality(self) -> C6MatrixView:
        return replace(self, missing_modality=True)

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        rows = _validated_rows(positions, len(self.windows))
        actor = self.base.load_sequences(rows)
        if self.combined_modalities:
            if len(self.combined_normalizations) != len(
                self.combined_modalities
            ):
                raise ValueError("C6 combined normalization state drift")
            branches = [
                self._load_modality_branch(modality, state, rows)
                for modality, state in zip(
                    self.combined_modalities,
                    self.combined_normalizations,
                    strict=True,
                )
            ]
            return self._combine_actor_and_auxiliary(actor, branches, rows)
        if self.modality is None:
            return actor
        if self.normalization is None:
            raise ValueError("C6 optional modality lacks normalization state")
        branch = self._load_modality_branch(
            self.modality,
            self.normalization,
            rows,
        )
        return self._combine_actor_and_auxiliary(actor, [branch], rows)

    def _load_modality_branch(
        self,
        modality: str,
        normalization: C6NormalizationState,
        rows: np.ndarray,
    ) -> np.ndarray:
        values = self.cache.load_values(modality, rows)
        feature_mask = self.cache.load_feature_mask(modality, rows)
        availability = self.cache.load_availability(modality, rows)
        zeros = np.zeros_like(values, dtype=np.float32)
        if self.missing_modality or self.control == "parameter_matched_zero":
            normalized = zeros
            branch = np.zeros_like(availability, dtype=np.float32)
        elif self.control == "availability_only":
            normalized = zeros
            branch = availability.astype(np.float32)
        elif self.control == "real":
            mean = np.asarray(normalization.mean, dtype=np.float32)
            scale = np.asarray(normalization.scale, dtype=np.float32)
            normalized = np.where(
                feature_mask,
                (values - mean) / scale,
                0.0,
            ).astype(np.float32)
            branch = availability.astype(np.float32)
        else:
            raise ValueError(f"unknown C6 control={self.control}")
        auxiliary = np.concatenate(
            [normalized, branch[..., None]],
            axis=2,
        )
        return auxiliary

    def _combine_actor_and_auxiliary(
        self,
        actor: np.ndarray,
        branches: list[np.ndarray],
        rows: np.ndarray,
    ) -> np.ndarray:
        auxiliary = np.concatenate(branches, axis=2)
        auxiliary[~self.observed_mask[rows]] = 0.0
        combined = np.concatenate([actor, auxiliary], axis=2)
        expected = (len(rows), SEQUENCE_LENGTH, self.input_dim)
        if combined.shape != expected or not np.isfinite(combined).all():
            raise ValueError(
                f"C6 {self.mode_id} input shape/finite drift={combined.shape}"
            )
        return combined.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class C6TrainingOutcome:
    """Best native-unit checkpoint and its validation evidence."""

    epoch_metrics: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    group_metrics: pd.DataFrame
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    best_epoch: int
    optimizer_steps: int
    parameter_sha256: str
    prediction_sha256: str
    maximum_loaded_batch_bytes: int


def load_c6_matrix_config(path: Path) -> C6MatrixConfig:
    """Load one matrix contract and verify every declared immutable input."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = C6MatrixConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    bound_sections = [_source_spec_name(payload), "implementation"]
    if payload["schema_version"] in {
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA_V4,
    }:
        bound_sections.append("temporal_base_freeze")
    if payload["schema_version"] == CONFIG_SCHEMA_V3:
        bound_sections.append("promotion_freeze")
    if (
        payload["schema_version"] == CONFIG_SCHEMA_V4
        and payload["training_scope"] == "full_development_confirmation"
    ):
        bound_sections.append("short_fusion_gate")
    for section in bound_sections:
        _verify_bound_spec(config.repo_root, _object(payload[section], section))
    for spec in _object(payload["inputs"], "inputs").values():
        _verify_bound_spec(config.repo_root, _object(spec, "inputs item"))
    return config


def build_c6_modality_cache(config: C6MatrixConfig) -> dict[str, Any]:
    """Build exact C6 numeric and cached-visual arrays without media reads."""

    if config.payload["execution"].get("data_run_authorized") is not True:
        raise RuntimeError("C6 cache build is disabled until clean lineage handoff")
    gate_errors = _execution_gate_errors(config)
    if gate_errors:
        raise RuntimeError(f"C6 execution gates failed={gate_errors}")
    root = config.cache_root
    root.mkdir(parents=True, exist_ok=False)
    source, _ = _load_c6_matrix_source(config)
    derived = derive_temporal_base_view(source.base_view, "A128")
    base = derived.view
    frames = _read_required_frames(config.bound_path("inputs", "harmonized_frames"))
    window_index, slot_index, selected_frames = _exact_c6_alignment(
        base.windows,
        frames,
    )
    pen_frames = build_pen_context_features(
        selected_frames,
        mask_path=config.bound_path("inputs", "pen_mask"),
        mask_threshold=int(config.payload["pen_context"]["mask_threshold"]),
        near_boundary_clearance_ratio=float(
            config.payload["pen_context"]["near_boundary_clearance_ratio"]
        ),
        expected_mask_sha256=str(
            config.payload["inputs"]["pen_mask"]["sha256"]
        ),
    )
    exported = export_spatial_sequences(
        window_index,
        pen_frames,
        max_window_length=SEQUENCE_LENGTH,
        feature_schema=LEGACY_SPATIAL_FRAME_FEATURES,
    )
    selected_modalities = _configured_modalities(config.payload)
    numeric_arrays = _numeric_modality_arrays(
        exported,
        pen_frames,
        slot_index,
    )
    arrays = {
        name: values
        for name, values in numeric_arrays.items()
        if name in selected_modalities
    }
    arrays.update(
        _context_modality_arrays(
            config,
            slot_index,
            selected_modalities,
        )
    )
    _validate_all_arrays(arrays, len(window_index))
    window_path = root / "c6_window_index.csv"
    slot_path = root / "c6_slot_index.csv"
    _write_dataframe_exclusive(window_path, window_index)
    _write_dataframe_exclusive(slot_path, slot_index)
    modalities: dict[str, Any] = {}
    for modality, payload in arrays.items():
        artifacts: dict[str, Any] = {}
        for kind in ("values", "feature_mask", "availability"):
            filename = f"{modality}_{kind}.npy"
            path = root / filename
            _write_numpy_exclusive(path, payload[kind])
            artifacts[kind] = _artifact_spec(path)
        modalities[modality] = {
            "feature_names": list(MODALITY_FEATURES[modality]),
            "feature_dim": len(MODALITY_FEATURES[modality]),
            "value_shape": list(payload["values"].shape),
            "feature_available_values": int(payload["feature_mask"].sum()),
            "available_slots": int(payload["availability"].sum()),
            "artifacts": artifacts,
        }
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "status": "PASS_LEGACY_C6_MODALITY_CACHE",
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": config.sha256,
        "source_temporal_config_sha256": file_sha256(
            config.bound_path(_source_spec_name(config.payload))
        ),
        "temporal_base_freeze_sha256": (
            file_sha256(config.bound_path("temporal_base_freeze"))
            if config.payload["schema_version"]
            in {CONFIG_SCHEMA_V2, CONFIG_SCHEMA_V3, CONFIG_SCHEMA_V4}
            else None
        ),
        "temporal_base_mode": "A128",
        "temporal_encoder": "masked_attention",
        "native_frame_offsets": list(C6_OFFSETS),
        "one_sequence_per_native_unit": True,
        "native_units": len(window_index),
        "slot_rows": len(slot_index),
        "train_native_units": int((window_index["l5_role"] == "train").sum()),
        "validation_native_units": int(
            (window_index["l5_role"] == "validation").sum()
        ),
        "window_index": _artifact_spec(window_path),
        "slot_index": _artifact_spec(slot_path),
        "modalities": modalities,
        "spatial_export_audit": exported.audit,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    manifest_path = root / "c6_modality_cache_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {**manifest, "manifest_sha256": file_sha256(manifest_path)}


def load_c6_modality_cache(config: C6MatrixConfig) -> C6ModalityCache:
    """Load and hash-audit a previously built C6 modality cache."""

    root = config.cache_root.resolve()
    manifest_path = root / "c6_modality_cache_manifest.json"
    manifest = _read_json(manifest_path)
    errors: list[str] = []
    expected = {
        "schema_version": CACHE_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": config.sha256,
        "native_frame_offsets": list(C6_OFFSETS),
        "one_sequence_per_native_unit": True,
        "errors": [],
        "valid": True,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            errors.append(f"manifest_{name}_drift")
    specs = [manifest["window_index"], manifest["slot_index"]]
    configured_modalities = _configured_modalities(config.payload)
    if set(manifest.get("modalities", {})) != set(configured_modalities):
        errors.append("manifest_modality_set_drift")
    for modality in configured_modalities:
        item = manifest.get("modalities", {}).get(modality)
        if not isinstance(item, dict):
            errors.append(f"missing_modality={modality}")
            continue
        specs.extend(item["artifacts"].values())
    for spec in specs:
        path = root / str(spec["filename"])
        if not path.is_file() or file_sha256(path) != str(spec["sha256"]):
            errors.append(f"artifact_hash_mismatch={spec['filename']}")
    if errors:
        raise ValueError(f"C6 modality cache audit failed={errors}")
    window_index = pd.read_csv(root / manifest["window_index"]["filename"])
    slot_index = pd.read_csv(root / manifest["slot_index"]["filename"])
    _validate_cache_indexes(window_index, slot_index, manifest)
    return C6ModalityCache(
        root=root,
        window_index=window_index,
        slot_index=slot_index,
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
    )


def c6_mode_ids(
    modalities: tuple[str, ...] | None = None,
    *,
    experiment_family: str = SINGLE_MODALITY_FAMILY,
) -> tuple[str, ...]:
    """Return the exact modes for a single or all-seven fusion contract."""

    modes = ["actor_only"]
    if experiment_family == COMBINED_ALL7_FAMILY:
        selected = modalities or tuple(MODALITY_FEATURES)
        if selected != tuple(MODALITY_FEATURES):
            raise ValueError("C6 combined experiment requires all seven modalities")
        modes.extend(
            f"{COMBINED_ALL7_MODE}__{control}" for control in CONTROL_MODES
        )
        return tuple(modes)
    if experiment_family != SINGLE_MODALITY_FAMILY:
        raise ValueError(f"unknown C6 experiment family={experiment_family}")
    for modality in modalities or tuple(MODALITY_FEATURES):
        modes.extend(f"{modality}__{control}" for control in CONTROL_MODES)
    return tuple(modes)


def fit_c6_normalization(
    cache: C6ModalityCache,
    modality: str,
    train_positions: np.ndarray,
) -> C6NormalizationState:
    """Fit each feature from unique available training identities only."""

    if modality not in MODALITY_FEATURES:
        raise ValueError(f"unknown C6 modality={modality}")
    rows = _validated_rows(train_positions, len(cache.window_index))
    roles = set(cache.window_index.iloc[rows]["l5_role"].astype(str))
    if roles != {"train"}:
        raise ValueError(f"C6 normalization roles={roles}")
    values = cache.load_values(modality, rows).astype(np.float64)
    feature_mask = cache.load_feature_mask(modality, rows)
    slots = _selected_slot_rows(cache.slot_index, rows)
    identities = _normalization_identities(modality, slots)
    means: list[float] = []
    scales: list[float] = []
    unique_counts: list[int] = []
    exposures: list[int] = []
    constants: list[str] = []
    for feature_index, feature_name in enumerate(MODALITY_FEATURES[modality]):
        selected = feature_mask[..., feature_index].reshape(-1)
        feature_values = values[..., feature_index].reshape(-1)[selected]
        feature_ids = identities[:, feature_index][selected]
        if len(feature_values) == 0 or np.any(feature_ids == ""):
            raise ValueError(f"C6 {modality}.{feature_name} has no identities")
        frame = pd.DataFrame({"identity": feature_ids, "value": feature_values})
        conflicts = frame.groupby("identity", sort=False)["value"].nunique()
        if conflicts.gt(1).any():
            raise ValueError(
                f"C6 {modality}.{feature_name} identity value conflict"
            )
        unique = frame.drop_duplicates("identity", keep="first")
        matrix = unique["value"].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"C6 {modality}.{feature_name} is nonfinite")
        mean = float(matrix.mean())
        scale = float(matrix.std(ddof=0))
        if scale <= 1e-12:
            scale = 1.0
            constants.append(feature_name)
        means.append(mean)
        scales.append(scale)
        unique_counts.append(len(unique))
        exposures.append(len(feature_values))
    semantic = {
        "schema_version": "classification_v2.c6_normalization.v1",
        "modality": modality,
        "feature_names": list(MODALITY_FEATURES[modality]),
        "mean": means,
        "scale": scales,
        "unique_identity_rows": unique_counts,
        "available_exposures": exposures,
        "constant_features": constants,
        "train_native_units": len(rows),
        "validation_rows_read_for_fit": 0,
        "outer_rows_read_for_fit": 0,
        "cache_manifest_sha256": cache.manifest_sha256,
    }
    return C6NormalizationState(
        modality=modality,
        feature_names=MODALITY_FEATURES[modality],
        mean=tuple(means),
        scale=tuple(scales),
        unique_identity_rows=tuple(unique_counts),
        available_exposures=tuple(exposures),
        constant_features=tuple(constants),
        train_native_units=len(rows),
        validation_rows_read_for_fit=0,
        outer_rows_read_for_fit=0,
        cache_manifest_sha256=cache.manifest_sha256,
        state_sha256=_payload_sha256(semantic),
    )


def build_c6_view(
    base: LegacyL5CachedFeatureView,
    cache: C6ModalityCache,
    mode_id: str,
    train_positions: np.ndarray,
) -> C6MatrixView:
    """Build one actor or parameter-matched optional-modality view."""

    modality, control = _parse_mode_id(mode_id)
    _validate_base_cache_alignment(base, cache)
    if modality == COMBINED_ALL7_MODE:
        combined_modalities = tuple(MODALITY_FEATURES)
        combined_normalizations = tuple(
            fit_c6_normalization(cache, name, train_positions)
            for name in combined_modalities
        )
        return C6MatrixView(
            base=base,
            cache=cache,
            modality=None,
            control=control,
            normalization=None,
            combined_modalities=combined_modalities,
            combined_normalizations=combined_normalizations,
        )
    normalization = (
        None
        if modality is None
        else fit_c6_normalization(cache, modality, train_positions)
    )
    return C6MatrixView(
        base=base,
        cache=cache,
        modality=modality,
        control=control,
        normalization=normalization,
    )


def build_c6_model(view: C6MatrixView, config: C6MatrixConfig) -> nn.Module:
    """Build the shared masked-attention head at the view's exact width."""

    model = _object(config.payload["model"], "model")
    common = {
        "temporal_encoder_name": "masked_attention",
        "hidden_dim": int(model["hidden_dim"]),
        "dropout": float(model["dropout"]),
        "transformer_layers": 1,
        "transformer_heads": 4,
    }
    if view.input_dim == FEATURE_DIM:
        return LegacyL5CachedFeatureClassifier(**common)
    return LegacyL6CachedModalityClassifier(
        input_dim=view.input_dim,
        **common,
    )


def static_c6_matrix_preflight(config: C6MatrixConfig) -> dict[str, Any]:
    """Validate contracts and paths without reading rows or initializing CUDA."""

    errors: list[str] = []
    warnings: list[str] = []
    cuda_before = torch.cuda.is_initialized()
    expected_modes = _configured_mode_ids(config.payload)
    for path_name, path in _declared_input_paths(config).items():
        if not path.is_file():
            errors.append(f"missing_{path_name}={path}")
    if tuple(config.payload["matrix"]["mode_ids"]) != expected_modes:
        errors.append("mode_matrix_drift")
    if config.payload["temporal_contract"]["native_frame_offsets"] != list(
        C6_OFFSETS
    ):
        errors.append("c6_offsets_drift")
    authorized = bool(
        config.payload["execution"].get("data_run_authorized")
    )
    if config.payload["schema_version"] == CONFIG_SCHEMA and authorized:
        errors.append("dirty_legacy_data_run_must_remain_disabled")
    errors.extend(_execution_gate_errors(config))
    if config.cache_root.exists():
        warnings.append("cache_root_already_exists_not_touched")
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("static_preflight_initialized_cuda")
    return {
        "schema_version": "classification_v2.c6_modality_static_preflight.v1",
        "status": "PASS" if not errors else "FAIL",
        "config_sha256": config.sha256,
        "lineage_scope": LINEAGE_SCOPE,
        "experiment_family": _experiment_family(config.payload),
        "mode_count": len(expected_modes),
        "mode_ids": list(expected_modes),
        "native_frame_offsets": list(C6_OFFSETS),
        "one_sequence_per_native_unit": True,
        "data_rows_read": 0,
        "optimizer_steps": 0,
        "source_media_reads": 0,
        "data_run_authorized": authorized,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def synthetic_c6_functional_preflight(
    *,
    experiment_family: str = SINGLE_MODALITY_FAMILY,
    modalities: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Exercise every input width, mask path, backward, and resume in memory."""

    torch.manual_seed(20260717)
    errors: list[str] = []
    modes: dict[str, Any] = {}
    batch_size = 3
    mask = torch.ones(batch_size, SEQUENCE_LENGTH)
    time_delta = torch.zeros(batch_size, SEQUENCE_LENGTH)
    time_delta[:, 1:] = 1.0 / 6.0
    mode_ids = c6_mode_ids(
        modalities,
        experiment_family=experiment_family,
    )
    for mode_id in mode_ids:
        modality, _ = _parse_mode_id(mode_id)
        if modality == COMBINED_ALL7_MODE:
            width = FEATURE_DIM + sum(
                len(names) + 1 for names in MODALITY_FEATURES.values()
            )
        elif modality is not None:
            width = FEATURE_DIM
            width += len(MODALITY_FEATURES[modality]) + 1
        else:
            width = FEATURE_DIM
        model = _synthetic_model(width)
        features = torch.randn(batch_size, SEQUENCE_LENGTH, width)
        logits = model(features, mask, time_delta=time_delta)
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor([0, 1, 2], dtype=torch.long),
        )
        loss.backward()
        if logits.shape != (batch_size, len(VALID_BEHAVIORS)):
            errors.append(f"{mode_id}:shape={list(logits.shape)}")
        if not torch.isfinite(logits).all():
            errors.append(f"{mode_id}:nonfinite_logits")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        modes[mode_id] = {
            "input_width": width,
            "parameter_count": parameters,
            "forward_shape": list(logits.shape),
            "backward_finite": bool(torch.isfinite(loss)),
        }
    controlled_names = (
        (COMBINED_ALL7_MODE,)
        if experiment_family == COMBINED_ALL7_FAMILY
        else tuple(MODALITY_FEATURES)
    )
    for name in controlled_names:
        counts = {
            modes[f"{name}__{control}"]["parameter_count"]
            for control in CONTROL_MODES
        }
        if len(counts) != 1:
            errors.append(f"{name}:control_parameter_count_drift")
    resume = _synthetic_resume_audit()
    errors.extend(resume["errors"])
    return {
        "schema_version": "classification_v2.c6_modality_synthetic_gate.v1",
        "status": "PASS" if not errors else "FAIL",
        "experiment_family": experiment_family,
        "modes": modes,
        "resume_audit": resume,
        "project_data_rows_read": 0,
        "optimizer_steps_on_project_data": 0,
        "errors": errors,
        "valid": not errors,
    }


def train_c6_mode(
    view: C6MatrixView,
    selection: LegacyL5CachedShortSelection,
    config: C6MatrixConfig,
    *,
    device: torch.device | str,
) -> C6TrainingOutcome:
    """Train one future-authorized mode with native-unit checkpoint selection."""

    if config.payload["execution"].get("data_run_authorized") is not True:
        raise RuntimeError("C6 project-data execution is not authorized")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C6 training requested unavailable CUDA")
    adapter = _training_adapter(config, view, selection)
    optimization = _object(adapter.payload["optimization"], "optimization")
    frozen_engine._validate_training_selection(view, selection, adapter)
    seed = int(optimization["seed"])
    frozen_engine._seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model = build_c6_model(view, config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    best: dict[str, Any] | None = None
    best_score: tuple[float, float] | None = None
    epoch_rows: list[dict[str, Any]] = []
    optimizer_steps = 0
    maximum_loaded = 0
    try:
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
                maximum_loaded = max(maximum_loaded, loaded)
                loss_value, batch_weight = frozen_engine._cached_training_step(
                    model,
                    optimizer,
                    batch,
                    device=resolved_device,
                    gradient_clip_norm=float(
                        optimization["gradient_clip_norm"]
                    ),
                )
                optimizer_steps += 1
                loss_mass += loss_value * batch_weight
                weight_mass += batch_weight
            evaluation = _evaluate_c6_model(
                model,
                view,
                selection,
                adapter,
                device=resolved_device,
            )
            maximum_loaded = max(
                maximum_loaded,
                int(evaluation["maximum_loaded_batch_bytes"]),
            )
            metrics = evaluation["metrics"]
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
                {
                    "epoch": epoch,
                    "optimizer_steps_cumulative": optimizer_steps,
                    "train_native_units": len(selection.train_positions),
                    "train_loss": loss_mass / weight_mass,
                    "validation_native_units": len(
                        selection.validation_positions
                    ),
                    "validation_macro_f1_global_10_class": metrics[
                        "macro_f1_global_10_class"
                    ],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_nll": metrics["nll"],
                    "selected_checkpoint": False,
                    "mode_id": view.mode_id,
                }
            )
        configured_steps = optimization.get("maximum_optimizer_steps")
        expected_steps = (
            int(configured_steps)
            if configured_steps is not None
            else int(optimization["epochs"])
            * int(
                np.ceil(
                    len(selection.train_positions)
                    / int(optimization["batch_size"])
                )
            )
        )
        if optimizer_steps != expected_steps or best is None:
            raise RuntimeError(
                f"C6 optimizer/checkpoint drift={optimizer_steps}/{expected_steps}"
            )
        epoch_rows[int(best["best_epoch"]) - 1]["selected_checkpoint"] = True
        epoch_metrics = pd.DataFrame.from_records(epoch_rows)
        model.load_state_dict(best["model_state"])
        missing = _evaluate_c6_model(
            model,
            view.with_missing_modality(),
            selection,
            adapter,
            device=resolved_device,
        )
        best["metrics"]["missing_modality_macro_f1"] = missing["metrics"][
            "macro_f1_global_10_class"
        ]
        return C6TrainingOutcome(
            epoch_metrics=epoch_metrics,
            predictions=best["predictions"],
            metrics=best["metrics"],
            per_class=best["per_class"],
            confusion=best["confusion"],
            group_metrics=best["group_metrics"],
            model_state=best["model_state"],
            optimizer_state=best["optimizer_state"],
            best_epoch=int(best["best_epoch"]),
            optimizer_steps=optimizer_steps,
            parameter_sha256=frozen_engine._state_dict_sha256(
                best["model_state"]
            ),
            prediction_sha256=frozen_engine._dataframe_sha256(
                best["predictions"]
            ),
            maximum_loaded_batch_bytes=maximum_loaded,
        )
    finally:
        model.to("cpu")
        del model, optimizer
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


def run_c6_repeat(
    config: C6MatrixConfig,
    repeat_id: str,
) -> dict[str, Any]:
    """Run all predeclared modes once after an explicit clean-data authorization."""

    if config.payload["execution"].get("data_run_authorized") is not True:
        raise RuntimeError("C6 repeat is disabled until clean lineage handoff")
    gate_errors = _execution_gate_errors(config)
    if gate_errors:
        raise RuntimeError(f"C6 execution gates failed={gate_errors}")
    cache = load_c6_modality_cache(config)
    source, source_config = _load_c6_matrix_source(config)
    derived = derive_temporal_base_view(source.base_view, "A128")
    if source_config is None:
        selection = replace(
            source.selection,
            audit={
                **source.selection.audit,
                "selection_content_sha256": frozen_engine._dataframe_sha256(
                    source.selection.manifest
                ),
                "training_scope": config.training_scope,
            },
        )
    else:
        _, selection = build_training_adapter(
            source_config,
            source,
            derived,
        )
    repeat_root = config.output_root / "runs" / repeat_id
    repeat_root.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    for mode_id in _configured_mode_ids(config.payload):
        view = build_c6_view(
            derived.view,
            cache,
            mode_id,
            selection.train_positions,
        )
        outcome = train_c6_mode(
            view,
            selection,
            config,
            device=str(config.payload["optimization"]["device"]),
        )
        results[mode_id] = _write_c6_run(
            config,
            cache,
            view,
            selection,
            outcome,
            repeat_root=repeat_root,
            repeat_id=repeat_id,
        )
    summary = {
        "schema_version": "classification_v2.c6_modality_repeat.v1",
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "config_sha256": config.sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "mode_ids": list(results),
        "results": results,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(repeat_root / "repeat_result.json", summary)
    return summary


def evaluate_c6_short_matrix(config: C6MatrixConfig) -> dict[str, Any]:
    """Audit deterministic repeats and paired real-versus-control effects."""

    repeats = [str(value) for value in config.payload["execution"]["repeats"]]
    cache = load_c6_modality_cache(config)
    packets: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    repeat_process_ids: dict[str, int] = {}
    configured_modes = _configured_mode_ids(config.payload)
    for mode_id in configured_modes:
        packets[mode_id] = []
        for repeat_id in repeats:
            path = config.output_root / "runs" / repeat_id / mode_id / "run.json"
            if not path.is_file():
                errors.append(f"missing_run={mode_id}:{repeat_id}")
                continue
            packet = _read_json(path)
            packet_errors = _validate_c6_run_packet(
                packet,
                mode_id=mode_id,
                repeat_id=repeat_id,
                config_sha256=config.sha256,
                cache_manifest_sha256=cache.manifest_sha256,
            )
            errors.extend(packet_errors)
            if packet_errors:
                continue
            process_id = int(packet["process_id"])
            previous = repeat_process_ids.get(repeat_id)
            if previous is not None and previous != process_id:
                errors.append(f"mixed_process={repeat_id}")
            repeat_process_ids[repeat_id] = process_id
            packets[mode_id].append(packet)
    if len(repeat_process_ids) == len(repeats):
        process_ids = list(repeat_process_ids.values())
        if len(set(process_ids)) != len(process_ids):
            errors.append("repeats_not_fresh_processes")
    comparisons: dict[str, Any] = {}
    for mode_id, mode_packets in packets.items():
        if len(mode_packets) != len(repeats):
            continue
        hashes = {packet["prediction_sha256"] for packet in mode_packets}
        if len(hashes) != 1:
            errors.append(f"nondeterministic_repeat={mode_id}")
    experiment_family = _experiment_family(config.payload)
    controlled_names = (
        (COMBINED_ALL7_MODE,)
        if experiment_family == COMBINED_ALL7_FAMILY
        else _configured_modalities(config.payload)
    )
    for name in controlled_names:
        parameter_counts = {
            int(packet["parameter_count"])
            for control in CONTROL_MODES
            for packet in packets.get(f"{name}__{control}", [])
        }
        if len(parameter_counts) != 1:
            errors.append(f"control_parameter_count_drift={name}")
        real = _prediction_path(config, repeats[0], f"{name}__real")
        for control in CONTROL_MODES[:2]:
            baseline = _prediction_path(
                config,
                repeats[0],
                f"{name}__{control}",
            )
            if not real.is_file() or not baseline.is_file():
                continue
            comparisons[f"{name}__real_minus_{control}"] = (
                _paired_prediction_comparison(
                    pd.read_csv(real),
                    pd.read_csv(baseline),
                    iterations=int(config.payload["evaluation"]["bootstrap_draws"]),
                    seed=int(config.payload["evaluation"]["bootstrap_seed"]),
                )
            )
    valid = not errors and len(comparisons) == len(controlled_names) * 2
    payload = {
        "schema_version": (
            MATRIX_SCHEMA_V2
            if experiment_family == COMBINED_ALL7_FAMILY
            else MATRIX_SCHEMA
        ),
        "status": "PASS" if valid else "FAIL",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "experiment_family": experiment_family,
        "config_sha256": config.sha256,
        "mode_count": len(configured_modes),
        "repeat_ids": repeats,
        "repeat_process_ids": repeat_process_ids,
        "comparisons": comparisons,
        "legacy_data_quality_status": (
            "TECHNICALLY_CLEAN_UNREVIEWED_DOUBLE_CHECK_PENDING"
            if config.payload["schema_version"]
            in {CONFIG_SCHEMA_V2, CONFIG_SCHEMA_V3, CONFIG_SCHEMA_V4}
            else "REQUIRES_CLEAN_LINEAGE_HANDOFF"
        ),
        "temporal_base_freeze_sha256": (
            file_sha256(config.bound_path("temporal_base_freeze"))
            if config.payload["schema_version"]
            in {CONFIG_SCHEMA_V2, CONFIG_SCHEMA_V3, CONFIG_SCHEMA_V4}
            else None
        ),
        "full_development_authorized": bool(
            valid
            and experiment_family == COMBINED_ALL7_FAMILY
            and config.training_scope == "short_repeat_gate"
        ),
        "full_development_authorized_modalities": (
            list(MODALITY_FEATURES)
            if valid and experiment_family == COMBINED_ALL7_FAMILY
            else []
        ),
        "full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }
    decision_filename = (
        "c6_combined_short_decision.json"
        if config.training_scope == "short_repeat_gate"
        and experiment_family == COMBINED_ALL7_FAMILY
        else "c6_full_development_decision.json"
        if config.training_scope == "full_development_confirmation"
        and experiment_family == COMBINED_ALL7_FAMILY
        else "c6_short_decision.json"
    )
    _write_json_exclusive(config.output_root / decision_filename, payload)
    return payload


def _validate_c6_run_packet(
    packet: dict[str, Any],
    *,
    mode_id: str,
    repeat_id: str,
    config_sha256: str,
    cache_manifest_sha256: str,
) -> list[str]:
    """Fail closed when a run packet is from the wrong mode or lineage."""

    errors: list[str] = []
    expected = {
        "status": "completed",
        "mode_id": mode_id,
        "repeat_id": repeat_id,
        "config_sha256": config_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "valid": True,
    }
    for field, expected_value in expected.items():
        if packet.get(field) != expected_value:
            errors.append(
                f"packet_mismatch={mode_id}:{repeat_id}:{field}"
            )
    process_id = packet.get("process_id")
    if not isinstance(process_id, int) or process_id <= 0:
        errors.append(f"invalid_process_id={mode_id}:{repeat_id}")
    for field in (
        "selection_sha256",
        "parameter_sha256",
        "prediction_sha256",
        "checkpoint_sha256",
    ):
        value = packet.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"invalid_packet_hash={mode_id}:{repeat_id}:{field}")
    return errors


def _read_required_frames(path: Path) -> pd.DataFrame:
    feature_columns = {
        column
        for columns in LEGACY_SPATIAL_FRAME_FEATURES.values()
        for column in columns
    }
    # The canonical exporter derives this label-independent mask from the
    # nearest partner identifiers; it is not a required source column.
    feature_columns.discard("social_neighbor_available")
    required = {
        "temporal_unit_key",
        "frame_uid",
        "scene_frame_uid",
        "object_track_key",
        "frame_index",
        "source_type",
        "dataset_id",
        "video_key",
        "lineage_scope",
        "human_review_complete",
        "nearest_pig_id",
        "nearest_track_id",
        *GEOMETRY_FEATURE_NAMES,
        *feature_columns,
        *REQUIRED_PEN_CONTEXT_INPUT_COLUMNS,
    }
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"C6 harmonized frames missing columns={missing}")
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    if frame["frame_uid"].fillna("").astype(str).duplicated().any():
        raise ValueError("C6 harmonized frames duplicate frame_uid")
    if set(frame["source_type"].astype(str)) != {"legacy_recovered"}:
        raise ValueError("C6 cache source_type drift")
    if set(frame["dataset_id"].astype(str)) != {"legacy_recovered_16f"}:
        raise ValueError("C6 cache dataset_id drift")
    if set(frame["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        raise ValueError("C6 cache lineage_scope drift")
    if _strict_bool(frame["human_review_complete"]).any():
        raise ValueError("C6 legacy cache incorrectly claims human review")
    return frame


def _exact_c6_alignment(
    base_windows: pd.DataFrame,
    frames: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_windows = {
        "window_id",
        "temporal_unit_key",
        "l5_role",
        "behavior_label",
    }
    missing = sorted(required_windows - set(base_windows.columns))
    if missing:
        raise ValueError(f"C6 base windows missing columns={missing}")
    groups = {
        str(key): group.sort_values("frame_index", kind="mergesort")
        for key, group in frames.groupby("temporal_unit_key", sort=False)
    }
    window_rows: list[dict[str, Any]] = []
    slot_frames: list[pd.DataFrame] = []
    for cache_row, row in base_windows.reset_index(drop=True).iterrows():
        unit_key = str(row["temporal_unit_key"])
        group = groups.get(unit_key)
        if group is None or len(group) != 16:
            count = 0 if group is None else len(group)
            raise ValueError(f"C6 native unit frame count={unit_key}:{count}")
        if group["object_track_key"].astype(str).nunique() != 1:
            raise ValueError(f"C6 native unit crosses object tracks={unit_key}")
        indices = group["frame_index"].to_numpy(dtype=np.int64)
        if not np.all(np.diff(indices) == 1):
            raise ValueError(f"C6 native unit is not contiguous={unit_key}")
        selected = group.iloc[list(C6_OFFSETS)].copy()
        start = int(selected["frame_index"].iloc[0])
        end = int(selected["frame_index"].iloc[-1])
        c6_window_id = f"{row['window_id']}::c6={start}-{end}"
        selected["cache_row"] = cache_row
        selected["slot_index"] = np.arange(SEQUENCE_LENGTH, dtype=np.int64)
        selected["native_frame_offset"] = list(C6_OFFSETS)
        selected["c6_window_id"] = c6_window_id
        slot_frames.append(selected)
        output = row.to_dict()
        output.update(
            {
                "cache_row": cache_row,
                "parent_window_id": str(row["window_id"]),
                "window_id": c6_window_id,
                "object_track_key": str(
                    selected["object_track_key"].iloc[0]
                ),
                "window_start_frame": start,
                "window_end_frame": end,
                "window_length_frames": SEQUENCE_LENGTH,
                "native_frame_offsets_json": json.dumps(
                    list(C6_OFFSETS), separators=(",", ":")
                ),
            }
        )
        window_rows.append(output)
    window_index = pd.DataFrame.from_records(window_rows)
    selected_frames = pd.concat(slot_frames, ignore_index=True)
    selected_frames = selected_frames.sort_values(
        ["cache_row", "slot_index"], kind="mergesort"
    ).reset_index(drop=True)
    slot_index = selected_frames[
        [
            "cache_row",
            "c6_window_id",
            "slot_index",
            "native_frame_offset",
            "temporal_unit_key",
            "frame_uid",
            "scene_frame_uid",
            "object_track_key",
            "frame_index",
            "source_type",
            "dataset_id",
            "video_key",
            "lineage_scope",
            "human_review_complete",
        ]
    ].rename(columns={"c6_window_id": "window_id"})
    slot_index["previous_frame_uid"] = (
        slot_index.groupby("cache_row", sort=False)["frame_uid"]
        .shift(1)
        .fillna("")
        .astype(str)
    )
    slot_index["pair_uid"] = (
        slot_index["previous_frame_uid"]
        + "->"
        + slot_index["frame_uid"].astype(str)
    ).where(slot_index["slot_index"].astype(int).gt(0), "")
    slot_index["window_slot_uid"] = (
        slot_index["window_id"].astype(str)
        + "::slot="
        + slot_index["slot_index"].astype(str)
    )
    if window_index["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("C6 cache duplicates native units")
    if slot_index[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("C6 cache duplicates window slots")
    expected_slots = len(window_index) * SEQUENCE_LENGTH
    if len(slot_index) != expected_slots:
        raise ValueError(f"C6 slot count={len(slot_index)}!={expected_slots}")
    return window_index, slot_index, selected_frames


def _numeric_modality_arrays(
    exported: Any,
    pen_frames: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    observed = np.asarray(exported.arrays["observed_mask"], dtype=np.bool_)
    if not observed.all():
        raise ValueError("C6 spatial export contains missing observed slots")
    frame_index = np.asarray(exported.arrays["frame_index_sequence"])
    if not np.all(np.diff(frame_index, axis=1) == 1):
        raise ValueError("C6 spatial export is not frame-contiguous")
    quality = np.asarray(exported.arrays["quality_mask"], dtype=np.float32)
    quality_names = list(exported.feature_names["quality_mask"])

    ordered_frames = pen_frames.sort_values(
        ["cache_row", "slot_index"], kind="mergesort"
    ).reset_index(drop=True)
    geometry = ordered_frames[list(GEOMETRY_FEATURE_NAMES)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    geometry = geometry.reshape(len(exported.arrays["observed_mask"]), 6, -1)
    geometry_available = observed.copy()
    for name in ("bbox_valid", "actor_bbox_valid", "geometry_feature_valid"):
        geometry_available &= quality[..., quality_names.index(name)] > 0.5
    geometry_mask = np.broadcast_to(
        geometry_available[..., None], geometry.shape
    ).copy()

    motion = np.asarray(exported.arrays["motion_delta"], dtype=np.float32)
    row_valid = observed.copy()
    for name in MOTION_QUALITY_FIELDS:
        row_valid &= quality[..., quality_names.index(name)] > 0.5
    motion_available = np.zeros(observed.shape, dtype=np.bool_)
    motion_available[:, 1:] = (
        row_valid[:, :-1]
        & row_valid[:, 1:]
        & (np.diff(frame_index, axis=1) == 1)
    )
    motion_mask = np.broadcast_to(
        motion_available[..., None], motion.shape
    ).copy()

    roi = np.asarray(exported.arrays["roi_class_relation"], dtype=np.float32)
    roi_available = observed.copy()
    for name in ROI_AVAILABILITY_FIELDS:
        roi_available &= quality[..., quality_names.index(name)] > 0.5
    roi_mask = np.broadcast_to(roi_available[..., None], roi.shape).copy()

    social = np.asarray(exported.arrays["social_relation"], dtype=np.float32)
    social_available = observed & (
        quality[..., quality_names.index("social_neighbor_available")] > 0.5
    )
    social_mask = np.broadcast_to(
        social_available[..., None], social.shape
    ).copy()

    pen = np.asarray(exported.arrays["pen_boundary_context"], dtype=np.float32)
    pen_branch = (
        _strict_bool(ordered_frames["pen_context_available"]).to_numpy()
        & _strict_bool(
            ordered_frames["pen_context_quality_valid"]
        ).to_numpy()
    ).reshape(observed.shape)
    pen_pair = np.zeros_like(pen_branch)
    pen_pair[:, 1:] = pen_branch[:, :-1] & pen_branch[:, 1:]
    pen_mask = np.zeros_like(pen, dtype=np.bool_)
    pen_mask[..., :PEN_STATIC_FEATURE_COUNT] = pen_branch[..., None]
    pen_mask[..., PEN_STATIC_FEATURE_COUNT:] = pen_pair[..., None]
    if len(slot_index) != pen.size // pen.shape[-1]:
        raise ValueError("C6 pen slots differ from slot index")

    return {
        "geometry": _masked_payload(geometry, geometry_mask),
        "motion": _masked_payload(motion, motion_mask),
        "roi": _masked_payload(roi, roi_mask),
        "numeric_social": _masked_payload(social, social_mask),
        "pen_context": _masked_payload(pen, pen_mask),
    }


def _context_modality_arrays(
    config: C6MatrixConfig,
    slot_index: pd.DataFrame,
    selected_modalities: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    context = pd.read_csv(
        config.bound_path("inputs", "image_window_context_manifest"),
        low_memory=False,
    )
    frame_to_context: dict[str, str] = {}
    for row in context.itertuples(index=False):
        frames = str(row.frame_uid_sequence).split("|")
        ids = str(row.image_context_id_sequence).split(";;")
        if len(frames) != len(ids):
            raise ValueError("C6 union context sequence lengths differ")
        for frame_uid, context_id in zip(frames, ids, strict=True):
            previous = frame_to_context.setdefault(frame_uid, context_id)
            if previous != context_id:
                raise ValueError("C6 frame maps to conflicting union contexts")
    output: dict[str, dict[str, np.ndarray]] = {}
    if "union_context" in selected_modalities:
        union_index = pd.read_csv(
            config.bound_path("inputs", "union_feature_index")
        )
        union_keys = (
            slot_index["frame_uid"]
            .astype(str)
            .map(frame_to_context)
            .fillna("")
        )
        slot_index["image_context_id"] = union_keys
        output["union_context"] = _materialize_context_features(
            union_keys,
            union_index,
            key_column="image_context_id",
            tensor_path=config.bound_path("inputs", "union_feature_tensor"),
        )
    if "full_frame_context" in selected_modalities:
        full_index = pd.read_csv(
            config.bound_path("inputs", "full_frame_feature_index")
        )
        output["full_frame_context"] = _materialize_context_features(
            slot_index["scene_frame_uid"].fillna("").astype(str),
            full_index,
            key_column="scene_frame_uid",
            tensor_path=config.bound_path(
                "inputs", "full_frame_feature_tensor"
            ),
        )
    return output


def _materialize_context_features(
    slot_keys: pd.Series,
    feature_index: pd.DataFrame,
    *,
    key_column: str,
    tensor_path: Path,
) -> dict[str, np.ndarray]:
    if feature_index[key_column].astype(str).duplicated().any():
        raise ValueError(f"C6 context index duplicates {key_column}")
    row_by_key = dict(
        zip(
            feature_index[key_column].astype(str),
            feature_index["feature_row"].astype(int),
            strict=True,
        )
    )
    keys = slot_keys.astype(str).to_numpy()
    rows = np.asarray([row_by_key.get(key, -1) for key in keys], dtype=np.int64)
    available = rows >= 0
    values = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    mapping = np.load(tensor_path, mmap_mode="r")
    try:
        if mapping.ndim != 2 or mapping.shape[1] != FEATURE_DIM:
            raise ValueError(f"C6 context tensor shape={mapping.shape}")
        values[available] = np.asarray(mapping[rows[available]], dtype=np.float32)
    finally:
        _close_memmap(mapping)
    unit_rows = len(rows) // SEQUENCE_LENGTH
    values = values.reshape(unit_rows, SEQUENCE_LENGTH, FEATURE_DIM)
    branch = available.reshape(unit_rows, SEQUENCE_LENGTH)
    feature_mask = np.broadcast_to(branch[..., None], values.shape).copy()
    return _masked_payload(values, feature_mask)


def _masked_payload(
    values: np.ndarray,
    feature_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    output = np.asarray(values, dtype=np.float32).copy()
    mask = np.asarray(feature_mask, dtype=np.bool_).copy()
    if output.shape != mask.shape or output.ndim != 3:
        raise ValueError("C6 values and feature mask shapes differ")
    output[~mask] = 0.0
    if not np.isfinite(output).all():
        raise ValueError("C6 modality values contain nonfinite entries")
    availability = mask.any(axis=2)
    return {
        "values": output,
        "feature_mask": mask,
        "availability": availability,
    }


def _validate_all_arrays(
    arrays: dict[str, dict[str, np.ndarray]],
    native_units: int,
) -> None:
    if not set(arrays).issubset(MODALITY_FEATURES):
        raise ValueError("C6 modality set drift")
    for modality, payload in arrays.items():
        feature_names = MODALITY_FEATURES[modality]
        expected = (native_units, SEQUENCE_LENGTH, len(feature_names))
        if payload["values"].shape != expected:
            raise ValueError(
                f"C6 {modality} values shape={payload['values'].shape}"
            )
        if payload["feature_mask"].shape != expected:
            raise ValueError(f"C6 {modality} feature mask shape drift")
        if payload["availability"].shape != expected[:2]:
            raise ValueError(f"C6 {modality} availability shape drift")
        if np.any(payload["values"][~payload["feature_mask"]] != 0.0):
            raise ValueError(f"C6 {modality} unavailable values are nonzero")


def _validate_cache_indexes(
    windows: pd.DataFrame,
    slots: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    native_units = int(manifest["native_units"])
    if len(windows) != native_units:
        raise ValueError("C6 cache window count drift")
    if len(slots) != native_units * SEQUENCE_LENGTH:
        raise ValueError("C6 cache slot count drift")
    if windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("C6 cache duplicate native units")
    if slots[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("C6 cache duplicate slots")
    offsets = slots.groupby("cache_row", sort=False)["native_frame_offset"].apply(
        list
    )
    if any(values != list(C6_OFFSETS) for values in offsets):
        raise ValueError("C6 cache native offsets drift")


def _selected_slot_rows(
    slot_index: pd.DataFrame,
    rows: np.ndarray,
) -> pd.DataFrame:
    order = {int(row): index for index, row in enumerate(rows)}
    selected = slot_index.loc[
        slot_index["cache_row"].astype(int).isin(order)
    ].copy()
    selected["selected_row_order"] = selected["cache_row"].astype(int).map(order)
    selected = selected.sort_values(
        ["selected_row_order", "slot_index"], kind="mergesort"
    ).reset_index(drop=True)
    expected = len(rows) * SEQUENCE_LENGTH
    if len(selected) != expected:
        raise ValueError(f"C6 selected normalization slots={len(selected)}")
    return selected


def _normalization_identities(
    modality: str,
    slots: pd.DataFrame,
) -> np.ndarray:
    frame_ids = slots["frame_uid"].fillna("").astype(str).to_numpy()
    pair_ids = slots["pair_uid"].fillna("").astype(str).to_numpy()
    window_ids = slots["window_slot_uid"].fillna("").astype(str).to_numpy()
    feature_dim = len(MODALITY_FEATURES[modality])
    if modality in {"geometry", "roi"}:
        base = frame_ids
        return np.broadcast_to(base[:, None], (len(base), feature_dim)).copy()
    if modality == "motion":
        return np.broadcast_to(
            pair_ids[:, None], (len(pair_ids), feature_dim)
        ).copy()
    if modality == "numeric_social":
        return np.broadcast_to(
            window_ids[:, None], (len(window_ids), feature_dim)
        ).copy()
    if modality == "pen_context":
        output = np.empty((len(slots), feature_dim), dtype=object)
        output[:, :PEN_STATIC_FEATURE_COUNT] = frame_ids[:, None]
        output[:, PEN_STATIC_FEATURE_COUNT:] = pair_ids[:, None]
        return output.astype(str)
    if modality == "union_context":
        ids = slots.get("image_context_id", pd.Series("", index=slots.index))
        base = ids.fillna("").astype(str).to_numpy()
        return np.broadcast_to(base[:, None], (len(base), feature_dim)).copy()
    if modality == "full_frame_context":
        base = slots["scene_frame_uid"].fillna("").astype(str).to_numpy()
        return np.broadcast_to(base[:, None], (len(base), feature_dim)).copy()
    raise ValueError(f"unknown C6 normalization modality={modality}")


def _validate_base_cache_alignment(
    base: LegacyL5CachedFeatureView,
    cache: C6ModalityCache,
) -> None:
    if base.sequence_length != SEQUENCE_LENGTH:
        raise ValueError("C6 base sequence length drift")
    if len(base.windows) != len(cache.window_index):
        raise ValueError("C6 base/cache native count drift")
    left = base.windows["temporal_unit_key"].fillna("").astype(str).tolist()
    right = (
        cache.window_index["temporal_unit_key"].fillna("").astype(str).tolist()
    )
    if left != right:
        raise ValueError("C6 base/cache native order drift")
    if not base.observed_mask.all():
        raise ValueError("C6 actor base contains unavailable slots")


def _parse_mode_id(mode_id: str) -> tuple[str | None, str]:
    if mode_id == "actor_only":
        return None, "actor_only"
    parts = mode_id.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid C6 mode_id={mode_id}")
    modality, control = parts
    supported = set(MODALITY_FEATURES) | {COMBINED_ALL7_MODE}
    if modality not in supported or control not in CONTROL_MODES:
        raise ValueError(f"unknown C6 mode_id={mode_id}")
    return modality, control


def _synthetic_model(input_width: int) -> nn.Module:
    common = {
        "temporal_encoder_name": "masked_attention",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
    }
    if input_width == FEATURE_DIM:
        return LegacyL5CachedFeatureClassifier(**common)
    return LegacyL6CachedModalityClassifier(input_dim=input_width, **common)


def _synthetic_resume_audit() -> dict[str, Any]:
    errors: list[str] = []
    model = _synthetic_model(FEATURE_DIM + 9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    features = torch.randn(4, SEQUENCE_LENGTH, FEATURE_DIM + 9)
    mask = torch.ones(4, SEQUENCE_LENGTH)
    delta = torch.zeros(4, SEQUENCE_LENGTH)
    logits = model(features, mask, time_delta=delta)
    loss = torch.nn.functional.cross_entropy(
        logits,
        torch.tensor([0, 1, 2, 3]),
    )
    loss.backward()
    optimizer.step()
    expected = frozen_engine._state_dict_sha256(model.state_dict())
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": "classification_v2.c6_checkpoint.v1",
            "config_sha256": "0" * 64,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    resumed = _synthetic_model(FEATURE_DIM + 9)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    resumed.load_state_dict(loaded["model_state"])
    resumed_optimizer.load_state_dict(loaded["optimizer_state"])
    observed = frozen_engine._state_dict_sha256(resumed.state_dict())
    if observed != expected:
        errors.append("resume_parameter_hash_mismatch")
    if loaded.get("config_sha256") != "0" * 64:
        errors.append("resume_config_hash_mismatch")
    return {
        "checkpoint_schema": loaded.get("schema_version"),
        "parameter_sha256_before": expected,
        "parameter_sha256_after": observed,
        "optimizer_state_loaded": bool(resumed_optimizer.state_dict()["state"]),
        "errors": errors,
        "valid": not errors,
    }


def _training_adapter(
    config: C6MatrixConfig,
    view: C6MatrixView,
    selection: LegacyL5CachedShortSelection,
) -> LegacyL5CachedTrainingConfig:
    optimization = copy.deepcopy(config.payload["optimization"])
    batch_size = int(optimization["batch_size"])
    epochs = int(optimization["epochs"])
    steps_per_epoch = (len(selection.train_positions) + batch_size - 1) // batch_size
    optimization["maximum_optimizer_steps"] = steps_per_epoch * epochs
    optimization.pop("device", None)
    payload = {
        "schema_version": CONFIG_SCHEMA,
        "training_scope": config.training_scope,
        "data": {
            "control_id": "V1_C6_MODALITY_MATRIX",
            "temporal_view_name": "legacy_c6_contiguous_centered_a128_v1",
            "sequence_length": SEQUENCE_LENGTH,
            "feature_dim": view.input_dim,
            "expected_train_native_units": len(selection.train_positions),
            "expected_validation_native_units": len(
                selection.validation_positions
            ),
            "native_prediction_aggregation": "one_sequence_per_native_unit",
        },
        "model": copy.deepcopy(config.payload["model"]),
        "optimization": optimization,
    }
    return LegacyL5CachedTrainingConfig(
        path=config.path,
        payload=payload,
        repo_root=config.repo_root,
    )


def _evaluate_c6_model(
    model: nn.Module,
    view: C6MatrixView,
    selection: LegacyL5CachedShortSelection,
    adapter: LegacyL5CachedTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    optimization = adapter.payload["optimization"]
    evaluation = frozen_engine._evaluate_cached_classifier(
        model,
        view,
        selection.validation_positions,
        batch_size=int(optimization["evaluation_batch_size"]),
        maximum_batch_bytes=int(optimization["maximum_loaded_batch_bytes"]),
        device=device,
    )
    predictions = frozen_engine._cached_prediction_frame(
        view,
        selection.validation_positions,
        config=adapter,
        probabilities=evaluation["probabilities"],
        targets=evaluation["targets"],
    )
    metrics, per_class, confusion = compute_legacy_l5_native_metrics(
        evaluation["probabilities"],
        evaluation["targets"],
        predictions["temporal_unit_key"],
    )
    predictions["mode_id"] = view.mode_id
    per_class["mode_id"] = view.mode_id
    confusion["mode_id"] = view.mode_id
    metrics = {
        **metrics,
        "mode_id": view.mode_id,
        "input_dim": view.input_dim,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "per_class": per_class,
        "confusion": confusion,
        "group_metrics": _group_metrics(per_class, view.mode_id),
        "maximum_loaded_batch_bytes": evaluation["maximum_loaded_batch_bytes"],
    }


def _group_metrics(per_class: pd.DataFrame, mode_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, labels in GROUP_LABELS.items():
        selected = per_class.loc[per_class["behavior_label"].isin(labels)]
        if len(selected) != len(labels):
            raise ValueError(f"C6 metric group incomplete={group_name}")
        rows.append(
            {
                "group": group_name,
                "labels": "|".join(labels),
                "support": int(selected["support"].sum()),
                "macro_f1": float(selected["f1"].mean()),
                "macro_recall": float(selected["recall"].mean()),
                "mode_id": mode_id,
            }
        )
    return pd.DataFrame.from_records(rows)


def _view_normalization_payload(view: C6MatrixView) -> dict[str, Any]:
    if view.combined_modalities:
        states = {
            modality: state.to_payload()
            for modality, state in zip(
                view.combined_modalities,
                view.combined_normalizations,
                strict=True,
            )
        }
        return {"states": states}
    state = None if view.normalization is None else view.normalization.to_payload()
    return {"state": state}


def _view_normalization_sha256(view: C6MatrixView) -> str | None:
    if view.combined_modalities:
        return _payload_sha256(_view_normalization_payload(view))
    if view.normalization is None:
        return None
    return view.normalization.state_sha256


def _write_c6_run(
    config: C6MatrixConfig,
    cache: C6ModalityCache,
    view: C6MatrixView,
    selection: LegacyL5CachedShortSelection,
    outcome: C6TrainingOutcome,
    *,
    repeat_root: Path,
    repeat_id: str,
) -> dict[str, Any]:
    root = repeat_root / view.mode_id
    root.mkdir(parents=False, exist_ok=False)
    _write_dataframe_exclusive(root / "native_predictions.csv", outcome.predictions)
    _write_dataframe_exclusive(root / "metrics_per_class.csv", outcome.per_class)
    _write_dataframe_exclusive(root / "confusion_matrix.csv", outcome.confusion)
    _write_dataframe_exclusive(root / "metrics_per_group.csv", outcome.group_metrics)
    _write_dataframe_exclusive(root / "epoch_metrics.csv", outcome.epoch_metrics)
    _write_json_exclusive(root / "metrics_global.json", outcome.metrics)
    normalization = _view_normalization_payload(view)
    _write_json_exclusive(root / "normalization.json", normalization)
    checkpoint = {
        "schema_version": "classification_v2.c6_checkpoint.v1",
        "config_sha256": config.sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "mode_id": view.mode_id,
        "repeat_id": repeat_id,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "normalization": normalization,
        "best_epoch": outcome.best_epoch,
        "model_state": outcome.model_state,
        "optimizer_state": outcome.optimizer_state,
    }
    checkpoint_path = root / "checkpoint.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed = build_c6_model(view, config)
    resumed.load_state_dict(loaded["model_state"])
    resumed_sha = frozen_engine._state_dict_sha256(resumed.state_dict())
    if resumed_sha != outcome.parameter_sha256:
        raise ValueError(f"C6 checkpoint resume hash drift={view.mode_id}")
    run = {
        "schema_version": RUN_SCHEMA,
        "status": "completed",
        "mode_id": view.mode_id,
        "repeat_id": repeat_id,
        "process_id": os.getpid(),
        "config_sha256": config.sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "selection_sha256": selection.audit["selection_content_sha256"],
        "normalization_sha256": _view_normalization_sha256(view),
        "parameter_count": sum(
            parameter.numel() for parameter in resumed.parameters()
        ),
        "parameter_sha256": outcome.parameter_sha256,
        "prediction_sha256": outcome.prediction_sha256,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "metrics": outcome.metrics,
        "environment": _environment_payload(),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(root / "run.json", run)
    return run


def _prediction_path(
    config: C6MatrixConfig,
    repeat_id: str,
    mode_id: str,
) -> Path:
    return config.output_root / "runs" / repeat_id / mode_id / "native_predictions.csv"


def _paired_prediction_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    candidate_metrics, candidate_class = _metrics_from_prediction_frame(candidate)
    baseline_metrics, baseline_class = _metrics_from_prediction_frame(baseline)
    bootstrap = paired_cluster_bootstrap(
        candidate,
        baseline,
        cluster_col="video_key",
        unit_col="temporal_unit_key",
        fold_col="recording_group_id",
        true_col="behavior_label",
        pred_col="predicted_label",
        iterations=iterations,
        seed=seed,
        outer_predictions_used_for_model_selection=False,
    )
    left = candidate_class.set_index("behavior_label")
    right = baseline_class.set_index("behavior_label")
    per_class = {
        label: {
            "support": int(left.loc[label, "support"]),
            "candidate_f1": float(left.loc[label, "f1"]),
            "baseline_f1": float(right.loc[label, "f1"]),
            "f1_delta": float(left.loc[label, "f1"] - right.loc[label, "f1"]),
            "recall_delta": float(
                left.loc[label, "recall"] - right.loc[label, "recall"]
            ),
        }
        for label in VALID_BEHAVIORS
    }
    group_deltas = {}
    for group_name, labels in GROUP_LABELS.items():
        candidate_group = float(left.loc[list(labels), "f1"].mean())
        baseline_group = float(right.loc[list(labels), "f1"].mean())
        group_deltas[group_name] = {
            "candidate_macro_f1": candidate_group,
            "baseline_macro_f1": baseline_group,
            "macro_f1_delta": candidate_group - baseline_group,
        }
    return {
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "macro_f1_delta": (
            candidate_metrics["macro_f1_global_10_class"]
            - baseline_metrics["macro_f1_global_10_class"]
        ),
        "per_class": per_class,
        "group_deltas": group_deltas,
        "video_cluster_bootstrap": bootstrap,
    }


def _metrics_from_prediction_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probability_columns = [f"prob_{label.replace('-', '_')}" for label in VALID_BEHAVIORS]
    missing = sorted(set(probability_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"C6 prediction probabilities missing={missing}")
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    targets = frame["target_index"].to_numpy(dtype=np.int64)
    metrics, per_class, _ = compute_legacy_l5_native_metrics(
        probabilities,
        targets,
        frame["temporal_unit_key"],
    )
    return metrics, per_class


def _execution_gate_errors(config: C6MatrixConfig) -> list[str]:
    schema = config.payload["schema_version"]
    errors: list[str] = []
    if schema in {CONFIG_SCHEMA_V2, CONFIG_SCHEMA_V3, CONFIG_SCHEMA_V4}:
        errors.extend(_temporal_freeze_errors(config))
    if schema == CONFIG_SCHEMA_V3:
        errors.extend(_promotion_freeze_errors(config))
    if (
        schema == CONFIG_SCHEMA_V4
        and config.training_scope == "full_development_confirmation"
    ):
        errors.extend(_combined_short_gate_errors(config))
    return errors


def _combined_short_gate_errors(config: C6MatrixConfig) -> list[str]:
    payload = _read_json(config.bound_path("short_fusion_gate"))
    expected = {
        "schema_version": MATRIX_SCHEMA_V2,
        "status": "PASS",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": "short_repeat_gate",
        "experiment_family": COMBINED_ALL7_FAMILY,
        "full_development_authorized": True,
        "full_development_authorized_modalities": list(MODALITY_FEATURES),
        "full_oof_authorized": False,
        "valid": True,
    }
    errors = [
        f"short_fusion_gate_{key}_drift"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    expected_comparisons = {
        f"{COMBINED_ALL7_MODE}__real_minus_{control}"
        for control in CONTROL_MODES[:2]
    }
    if set(payload.get("comparisons", {})) != expected_comparisons:
        errors.append("short_fusion_gate_comparison_set_drift")
    freeze_sha = file_sha256(config.bound_path("temporal_base_freeze"))
    if payload.get("temporal_base_freeze_sha256") != freeze_sha:
        errors.append("short_fusion_gate_temporal_freeze_drift")
    return errors


def _temporal_freeze_errors(config: C6MatrixConfig) -> list[str]:
    """Require the paired-control freeze before authorized schema-v2 runs."""

    path = config.bound_path("temporal_base_freeze")
    if not path.is_file():
        return [f"missing_temporal_base_freeze={path}"]
    payload = _read_json(path)
    if payload.get("schema_version") == (
        "classification_v2.legacy_c6_temporal_base_freeze.v2"
    ):
        return _evaluated_temporal_freeze_errors(config, payload)
    expected = {
        "schema_version": "classification_v2.legacy_c6_temporal_base_freeze.v1",
        "status": "PASS_C6_TEMPORAL_BASE_FREEZE",
        "decision": "FREEZE_PRIOR_A128_FOR_C6_MODALITY_SCREENING",
        "selected_base_mode": "A128",
        "modality_matrix_authorized": True,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "valid": True,
    }
    return [
        f"temporal_base_freeze_{key}_drift"
        for key, value in expected.items()
        if payload.get(key) != value
    ]


def _evaluated_temporal_freeze_errors(
    config: C6MatrixConfig,
    payload: dict[str, Any],
) -> list[str]:
    expected = {
        "status": "PASS_C6_TEMPORAL_BASE_FREEZE",
        "decision": (
            "FREEZE_EVALUATED_A128_FOR_LEGACY_16F_MODALITY_SCREENING"
        ),
        "selected_base_mode": "A128",
        "selected_base_is_carried_prior_not_tested_in_this_matrix": False,
        "modality_matrix_authorized": True,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "valid": True,
    }
    errors = [
        f"temporal_base_freeze_{key}_drift"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    spec = payload.get("base_selection_decision")
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        errors.append("temporal_base_freeze_decision_spec_drift")
        return errors
    try:
        decision_path = _resolve_inside(config.repo_root, str(spec["path"]))
    except ValueError:
        errors.append("temporal_base_freeze_decision_path_invalid")
        return errors
    if not decision_path.is_file():
        errors.append("temporal_base_freeze_decision_missing")
        return errors
    if file_sha256(decision_path) != str(spec["sha256"]):
        errors.append("temporal_base_freeze_decision_hash_drift")
        return errors
    decision = _read_json(decision_path)
    if decision.get("common_native_universe") != payload.get(
        "common_native_universe"
    ):
        errors.append("temporal_base_freeze_native_universe_drift")
    return errors


def _promotion_freeze_errors(config: C6MatrixConfig) -> list[str]:
    """Require selected full-development modalities to be gate-authorized."""

    payload = _read_json(config.bound_path("promotion_freeze"))
    errors: list[str] = []
    expected = {
        "schema_version": (
            "classification_v2.legacy_c6_modality_promotion_freeze.v1"
        ),
        "status": "PASS_C6_MODALITY_PROMOTION_FREEZE",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "main_branch_promotion_allowed": False,
        "valid": True,
    }
    errors.extend(
        f"promotion_freeze_{key}_drift"
        for key, value in expected.items()
        if payload.get(key) != value
    )
    authorized = set(
        str(value)
        for value in payload.get("full_development_authorized_modalities", [])
    )
    selected = set(_configured_modalities(config.payload))
    if not selected.issubset(authorized):
        errors.append(
            "promotion_freeze_selected_modalities_not_authorized"
        )
    return errors


def _configured_modalities(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return the ordered modality subset declared by one matrix config."""

    matrix = _object(payload.get("matrix"), "matrix")
    values = matrix.get("modalities")
    if values is None:
        return tuple(MODALITY_FEATURES)
    if not isinstance(values, list) or not values:
        raise ValueError("C6 matrix modalities must be a nonempty list")
    if len(set(values)) != len(values):
        raise ValueError("C6 matrix modalities must be unique")
    unknown = set(values).difference(MODALITY_FEATURES)
    if unknown:
        raise ValueError(f"unknown C6 matrix modalities={sorted(unknown)}")
    return tuple(str(value) for value in values)


def _experiment_family(payload: dict[str, Any]) -> str:
    contract = _object(payload.get("experiment_contract"), "experiment_contract")
    value = str(contract.get("changed_scientific_family", ""))
    if value not in {SINGLE_MODALITY_FAMILY, COMBINED_ALL7_FAMILY}:
        raise ValueError(f"unknown C6 experiment family={value}")
    return value


def _configured_mode_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    return c6_mode_ids(
        _configured_modalities(payload),
        experiment_family=_experiment_family(payload),
    )


def _declared_input_paths(config: C6MatrixConfig) -> dict[str, Path]:
    source_field = _source_spec_name(config.payload)
    paths = {source_field: config.bound_path(source_field)}
    paths.update(
        {
            name: config.bound_path("inputs", name)
            for name in config.payload["inputs"]
        }
    )
    paths["implementation"] = config.bound_path("implementation")
    if config.payload["schema_version"] in {
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA_V4,
    }:
        paths["temporal_base_freeze"] = config.bound_path(
            "temporal_base_freeze"
        )
    if config.payload["schema_version"] == CONFIG_SCHEMA_V3:
        paths["promotion_freeze"] = config.bound_path("promotion_freeze")
    if (
        config.payload["schema_version"] == CONFIG_SCHEMA_V4
        and config.training_scope == "full_development_confirmation"
    ):
        paths["short_fusion_gate"] = config.bound_path("short_fusion_gate")
    return paths


def _validate_config_payload(payload: dict[str, Any]) -> None:
    source_field = _source_spec_name(payload)
    required = {
        "schema_version",
        "training_scope",
        "lineage_scope",
        "experiment_contract",
        source_field,
        "inputs",
        "pen_context",
        "temporal_contract",
        "matrix",
        "model",
        "optimization",
        "evaluation",
        "execution",
        "implementation",
        "output",
    }
    if payload["schema_version"] in {
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA_V4,
    }:
        required.add("temporal_base_freeze")
    if payload["schema_version"] == CONFIG_SCHEMA_V3:
        required.add("promotion_freeze")
    if (
        payload["schema_version"] == CONFIG_SCHEMA_V4
        and payload["training_scope"] == "full_development_confirmation"
    ):
        required.add("short_fusion_gate")
    if set(payload) != required:
        raise ValueError(
            "C6 config keys differ: "
            f"missing={sorted(required - set(payload))},"
            f"extra={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] not in {
        CONFIG_SCHEMA,
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA_V4,
    }:
        raise ValueError("C6 config schema drift")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError(f"C6 config lineage_scope={payload['lineage_scope']!r}")
    if payload["training_scope"] not in {
        "short_repeat_gate",
        "full_development_confirmation",
    }:
        raise ValueError("C6 config training_scope is unsupported")
    contract = _object(payload["experiment_contract"], "experiment_contract")
    family = _experiment_family(payload)
    expected_family = (
        COMBINED_ALL7_FAMILY
        if payload["schema_version"] == CONFIG_SCHEMA_V4
        else SINGLE_MODALITY_FAMILY
    )
    if family != expected_family:
        raise ValueError("C6 config changed-family drift")
    if contract.get("outer_predictions_used_for_model_selection") is not False:
        raise ValueError("C6 outer predictions cannot select the architecture")
    if contract.get("full_oof_authorized") is not False:
        raise ValueError("C6 config cannot authorize full OOF")
    temporal = _object(payload["temporal_contract"], "temporal_contract")
    expected_temporal = {
        "base_mode": "A128",
        "native_frame_offsets": list(C6_OFFSETS),
        "sequence_length": SEQUENCE_LENGTH,
        "one_sequence_per_native_unit": True,
        "native_evaluation_unit": "complete_16_frame_burst",
    }
    if temporal != expected_temporal:
        raise ValueError("C6 temporal contract drift")
    matrix = _object(payload["matrix"], "matrix")
    configured_modalities = _configured_modalities(payload)
    if matrix.get("mode_ids") != list(_configured_mode_ids(payload)):
        raise ValueError("C6 matrix mode IDs drift")
    if matrix.get("controls") != list(CONTROL_MODES):
        raise ValueError("C6 matrix controls drift")
    if payload["schema_version"] in {
        CONFIG_SCHEMA,
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V4,
    }:
        if configured_modalities != tuple(MODALITY_FEATURES):
            raise ValueError("C6 matrix modality order drift")
    elif payload["training_scope"] != "full_development_confirmation":
        raise ValueError("C6 v3 is reserved for full development")
    model = _object(payload["model"], "model")
    expected_model = {
        "actor_feature_control": "V1_resnet18_224_imagenet1k_v1",
        "actor_feature_dim": FEATURE_DIM,
        "temporal_encoder_name": "masked_attention",
        "hidden_dim": 128,
        "dropout": 0.1,
        "final_behavior_classes": list(VALID_BEHAVIORS),
    }
    if model != expected_model:
        raise ValueError("C6 model contract drift")
    execution = _object(payload["execution"], "execution")
    if not isinstance(execution.get("data_run_authorized"), bool):
        raise ValueError("C6 data_run_authorized must be explicit boolean")
    if execution.get("data_run_authorized") is True and not execution.get(
        "clean_lineage_handoff_id"
    ):
        raise ValueError("C6 data run lacks clean lineage handoff ID")
    bound_sections = [source_field, "implementation"]
    if payload["schema_version"] in {
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA_V4,
    }:
        bound_sections.append("temporal_base_freeze")
    if payload["schema_version"] == CONFIG_SCHEMA_V3:
        bound_sections.append("promotion_freeze")
    if (
        payload["schema_version"] == CONFIG_SCHEMA_V4
        and payload["training_scope"] == "full_development_confirmation"
    ):
        bound_sections.append("short_fusion_gate")
    for section in bound_sections:
        _validate_hash_spec(_object(payload[section], section), section)
    inputs = _object(payload["inputs"], "inputs")
    required_inputs = {
        "harmonized_frames",
        "pen_mask",
        "image_window_context_manifest",
        "union_feature_tensor",
        "union_feature_index",
    }
    if payload["schema_version"] in {
        CONFIG_SCHEMA,
        CONFIG_SCHEMA_V2,
        CONFIG_SCHEMA_V4,
    }:
        required_inputs.update(
            {"full_frame_feature_tensor", "full_frame_feature_index"}
        )
    if set(inputs) != required_inputs:
        raise ValueError("C6 input set drift")
    for name, value in inputs.items():
        _validate_hash_spec(_object(value, name), f"inputs.{name}")
    output = _object(payload["output"], "output")
    if set(output) != {"root_relative_path"}:
        raise ValueError("C6 output contract drift")
    path = Path(str(output["root_relative_path"]))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("C6 output path is unsafe")


def _source_spec_name(payload: dict[str, Any]) -> str:
    schema = payload.get("schema_version")
    if schema == CONFIG_SCHEMA:
        return "source_temporal_config"
    if schema in {CONFIG_SCHEMA_V2, CONFIG_SCHEMA_V3, CONFIG_SCHEMA_V4}:
        return "prepared_source"
    raise ValueError("C6 config schema drift")


def _load_c6_matrix_source(config: C6MatrixConfig) -> tuple[Any, Any | None]:
    source_field = _source_spec_name(config.payload)
    if source_field == "prepared_source":
        source = load_legacy_c6_prepared_source(
            config.bound_path(source_field),
            repo_root=config.repo_root,
        )
        return source, None
    source_config = load_temporal_base_selection_config(
        config.bound_path(source_field)
    )
    return load_temporal_base_source(source_config), source_config


def _validate_hash_spec(spec: dict[str, Any], name: str) -> None:
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name} hash spec keys drift")
    sha = str(spec["sha256"])
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"{name}.sha256 is invalid")


def _verify_bound_spec(root: Path, spec: dict[str, Any]) -> None:
    path = _resolve_inside(root, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(spec["sha256"]):
        raise ValueError(
            f"C6 input hash mismatch={path}:"
            f"{observed}!={spec['sha256']}"
        )


def _artifact_spec(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _environment_payload() -> dict[str, Any]:
    gpu_name = ""
    gpu_vram = 0
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = str(properties.name)
        gpu_vram = int(properties.total_memory)
    return {
        "os": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_vram,
    }


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError("C6 boolean column contains missing values")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"C6 boolean column values are invalid={unknown}")
    return normalized.map(mapping).astype(bool)


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("C6 rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("C6 rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("C6 rows contain duplicates")
    return rows


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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
        raise ValueError(f"C6 path escapes repository={path}") from error
    return path


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_numpy_exclusive(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


__all__ = [
    "COMBINED_ALL7_FAMILY",
    "COMBINED_ALL7_MODE",
    "CONFIG_SCHEMA_V4",
    "C6MatrixConfig",
    "C6MatrixView",
    "C6ModalityCache",
    "C6NormalizationState",
    "CONTROL_MODES",
    "MODALITY_FEATURES",
    "build_c6_modality_cache",
    "build_c6_model",
    "build_c6_view",
    "c6_mode_ids",
    "evaluate_c6_short_matrix",
    "fit_c6_normalization",
    "load_c6_matrix_config",
    "load_c6_modality_cache",
    "run_c6_repeat",
    "static_c6_matrix_preflight",
    "synthetic_c6_functional_preflight",
    "train_c6_mode",
]
