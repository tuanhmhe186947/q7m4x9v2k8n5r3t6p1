"""Reusable fixed-width cached-modality components for legacy L6 ablations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.models.temporal_encoders import (
    build_temporal_encoder,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)

CONTROL_MODES = ("parameter_matched_zero", "availability_only")


@dataclass(frozen=True, slots=True)
class CachedModalityNormalizationState:
    """Train-pair-only state for one explicit cached modality."""

    schema_version: str
    modality_name: str
    identity_field: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
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
                "missing_modality_after_transform": 0.0,
                "validation_and_outer_excluded": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


@dataclass(frozen=True, slots=True)
class LegacyL6CachedModalityView:
    """Duck-typed L5 view with one fixed-width optional cached modality."""

    base: LegacyL5CachedFeatureView
    cache: Any
    mode: str
    active_mode: str
    modality_name: str
    feature_names: tuple[str, ...]
    sequence_length: int
    normalization: CachedModalityNormalizationState
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
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def model_input_dim(self) -> int:
        return FEATURE_DIM + self.feature_dim + 1

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        rows = _validated_rows(positions, len(self.windows))
        visual = self.base.load_sequences(rows)
        auxiliary = self._load_auxiliary(rows)
        combined = np.concatenate([visual, auxiliary], axis=2).astype(
            np.float32,
            copy=False,
        )
        expected = (len(rows), self.sequence_length, self.model_input_dim)
        if combined.shape != expected:
            raise ValueError(
                f"L6 {self.modality_name} combined shape="
                f"{combined.shape}!={expected}"
            )
        return combined

    def with_missing_modality(self) -> LegacyL6CachedModalityView:
        return replace(self, missing_modality=True)

    def _load_auxiliary(self, rows: np.ndarray) -> np.ndarray:
        available = self.cache.load_availability(rows)
        zeros = np.zeros(
            (len(rows), self.sequence_length, self.feature_dim),
            dtype=np.float32,
        )
        if self.missing_modality or self.mode == "parameter_matched_zero":
            values = zeros
            availability = np.zeros_like(available, dtype=np.float32)
        elif self.mode == "availability_only":
            values = zeros
            availability = available.astype(np.float32)
        elif self.mode == self.active_mode:
            loader = getattr(self.cache, f"load_{self.modality_name}", None)
            if loader is None:
                raise TypeError(
                    f"cache lacks load_{self.modality_name} method"
                )
            raw = loader(rows).astype(np.float64)
            mean = np.asarray(self.normalization.mean, dtype=np.float64)
            scale = np.asarray(self.normalization.scale, dtype=np.float64)
            normalized = (raw - mean) / scale
            values = np.where(
                available[..., None],
                normalized,
                0.0,
            ).astype(np.float32)
            availability = available.astype(np.float32)
        else:
            raise ValueError(
                f"unknown L6 {self.modality_name} mode={self.mode}"
            )
        auxiliary = np.concatenate(
            [values, availability[..., None]],
            axis=2,
        )
        observed = self.observed_mask[rows]
        auxiliary[~observed] = 0.0
        if not np.isfinite(auxiliary).all():
            raise ValueError(
                f"L6 {self.modality_name} features contain nonfinite values"
            )
        return auxiliary


class LegacyL6CachedModalityClassifier(nn.Module):
    """Fixed-capacity temporal head parameterized only by input width."""

    def __init__(
        self,
        *,
        input_dim: int,
        temporal_encoder_name: str,
        hidden_dim: int,
        dropout: float,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("L6 cached-modality dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("L6 cached-modality dropout must be in [0,1)")
        self.input_dim = int(input_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, hidden_dim)
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
            or features.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "L6 cached-modality features/mask shape drift "
                f"input_dim={self.input_dim}"
            )
        if not torch.isfinite(observed_mask).all():
            raise ValueError("L6 cached-modality mask contains nonfinite values")
        if not torch.all((observed_mask == 0) | (observed_mask == 1)):
            raise ValueError("L6 cached-modality mask must be binary")
        valid = observed_mask.bool()
        if not torch.isfinite(features[valid]).all():
            raise ValueError(
                "L6 cached-modality observed features are nonfinite"
            )
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


def fit_cached_modality_normalization(
    cache: Any,
    selection: TemporalLadderSelection,
    *,
    modality_name: str,
    feature_names: tuple[str, ...],
    identity_field: str,
    schema_version: str,
) -> CachedModalityNormalizationState:
    """Fit one modality from unique available training identities only."""

    rows = _validated_rows(selection.train_positions, len(cache.window_index))
    if len(rows) == 0:
        raise ValueError(f"{modality_name} normalization has zero train windows")
    roles = set(cache.window_index.iloc[rows]["l5_role"].astype(str))
    if roles != {"train"}:
        raise ValueError(f"{modality_name} normalization roles={roles}")
    loader = getattr(cache, f"load_{modality_name}", None)
    if loader is None:
        raise TypeError(f"cache lacks load_{modality_name} method")
    raw = loader(rows).astype(np.float64)
    available = cache.load_availability(rows)
    sequence_length = int(raw.shape[1])
    feature_dim = len(feature_names)
    if raw.shape != (len(rows), sequence_length, feature_dim):
        raise ValueError(f"{modality_name} normalization tensor shape drift")
    row_order = {int(value): index for index, value in enumerate(rows)}
    slots = cache.slot_index.loc[
        cache.slot_index["cache_row"].astype(int).isin(row_order)
    ].copy()
    slots["train_row_order"] = slots["cache_row"].astype(int).map(row_order)
    slots = slots.sort_values(
        ["train_row_order", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_slots = len(rows) * sequence_length
    if len(slots) != expected_slots:
        raise ValueError(f"{modality_name} normalization slot rows={len(slots)}")
    if identity_field not in slots.columns:
        if not identity_field.endswith("_window_slot_uid"):
            raise ValueError(
                f"{modality_name} normalization lacks {identity_field}"
            )
        required = {"window_id", "slot_index"}
        if not required.issubset(slots.columns):
            raise ValueError(
                f"{modality_name} normalization lacks window-slot identity"
            )
        slots[identity_field] = (
            slots["window_id"].astype(str)
            + "::slot="
            + slots["slot_index"].astype(str)
        )
    values = raw.reshape(expected_slots, feature_dim)
    available_flat = available.reshape(expected_slots)
    if not available_flat.any():
        raise ValueError(f"{modality_name} has zero available train slots")
    identity = slots[identity_field].fillna("").astype(str).to_numpy()
    if np.any(identity[available_flat] == ""):
        raise ValueError(f"{modality_name} available identities are blank")
    frame = pd.DataFrame(values, columns=feature_names)
    frame.insert(0, identity_field, identity)
    frame = frame.loc[available_flat].copy()
    conflicts = frame.groupby(identity_field, sort=False)[
        list(feature_names)
    ].nunique(dropna=False)
    if conflicts.gt(1).any(axis=None):
        raise ValueError(f"repeated {identity_field} has conflicting values")
    unique = frame.drop_duplicates(identity_field, keep="first").sort_values(
        identity_field,
        kind="mergesort",
    )
    matrix = unique[list(feature_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{modality_name} train matrix is nonfinite")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"{modality_name} statistics are nonfinite")
    if (scale <= 1e-12).any():
        constant = [
            feature_names[index]
            for index in np.flatnonzero(scale <= 1e-12)
        ]
        raise ValueError(f"{modality_name} constant features={constant}")
    train_windows = cache.window_index.iloc[rows]["window_id"].astype(str)
    semantic = {
        "schema_version": schema_version,
        "modality_name": modality_name,
        "identity_field": identity_field,
        "feature_names": list(feature_names),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "train_window_rows": int(len(rows)),
        "train_slot_exposures": int(available_flat.sum()),
        "unique_train_identity_rows": int(len(unique)),
        "duplicate_train_slot_exposures": int(
            available_flat.sum() - len(unique)
        ),
        "train_window_id_sha256": _ordered_sha256(train_windows),
        "unique_train_identity_sha256": _ordered_sha256(
            unique[identity_field]
        ),
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
            "missing_modality_after_transform": 0.0,
            "validation_and_outer_excluded": True,
        },
    }
    state_sha = _payload_sha256(semantic)
    return CachedModalityNormalizationState(
        schema_version=schema_version,
        modality_name=modality_name,
        identity_field=identity_field,
        feature_names=feature_names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        train_window_rows=int(len(rows)),
        train_slot_exposures=int(available_flat.sum()),
        unique_train_identity_rows=int(len(unique)),
        duplicate_train_slot_exposures=int(available_flat.sum() - len(unique)),
        train_window_id_sha256=semantic["train_window_id_sha256"],
        unique_train_identity_sha256=semantic[
            "unique_train_identity_sha256"
        ],
        cache_manifest_sha256=semantic["cache_manifest_sha256"],
        selection_content_sha256=semantic["selection_content_sha256"],
        fit_role="train",
        validation_rows_read_for_fit=0,
        outer_holdout_rows_read_for_fit=0,
        state_sha256=state_sha,
    )


def build_cached_modality_view(
    base: LegacyL5CachedFeatureView,
    cache: Any,
    *,
    mode: str,
    active_mode: str,
    modality_name: str,
    feature_names: tuple[str, ...],
    sequence_length: int,
    normalization: CachedModalityNormalizationState,
) -> LegacyL6CachedModalityView:
    """Construct one fixed-width cached-modality control view."""

    modes = (*CONTROL_MODES, active_mode)
    if mode not in modes:
        raise ValueError(f"unknown L6 {modality_name} mode={mode}")
    if normalization.modality_name != modality_name:
        raise ValueError(f"L6 {modality_name} normalization name drift")
    if normalization.feature_names != feature_names:
        raise ValueError(f"L6 {modality_name} feature order drift")
    if normalization.cache_manifest_sha256 != cache.audit["manifest_sha256"]:
        raise ValueError(f"L6 {modality_name} cache hash drift")
    return LegacyL6CachedModalityView(
        base=base,
        cache=cache,
        mode=mode,
        active_mode=active_mode,
        modality_name=modality_name,
        feature_names=feature_names,
        sequence_length=sequence_length,
        normalization=normalization,
    )


def cached_modality_feature_whitelist(
    *,
    mode: str,
    active_mode: str,
    modality_name: str,
    feature_names: tuple[str, ...],
    schema_version: str,
) -> dict[str, Any]:
    """Return one explicit parameter-matched model-X whitelist."""

    if mode not in (*CONTROL_MODES, active_mode):
        raise ValueError(f"unknown L6 {modality_name} mode={mode}")
    visual = [f"cached_frame_feature_{index:03d}" for index in range(FEATURE_DIM)]
    auxiliary = [f"{modality_name}_{name}" for name in feature_names]
    features = [*visual, *auxiliary, f"{modality_name}_available"]
    return {
        "schema_version": schema_version,
        "mode": mode,
        "modality_name": modality_name,
        "features": features,
        "feature_count": len(features),
        "visual_feature_count": FEATURE_DIM,
        "modality_feature_count": len(feature_names),
        "availability_feature_count": 1,
        "parameter_matched_input_width": len(features),
        "labels_paths_ids_folds_review_or_unit_aggregates_in_model_x": False,
        "availability_is_behavior_evidence": False,
        "source_identifier_in_model_x": False,
    }


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("L6 cached-modality rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("L6 cached-modality rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("L6 cached-modality rows contain duplicates")
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
