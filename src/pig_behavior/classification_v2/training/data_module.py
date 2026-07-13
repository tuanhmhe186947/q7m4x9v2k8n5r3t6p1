"""Strict key-aligned data module for classification_v2 multimodal training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.datasets.visual_interaction_loader import (
    VisualInteractionDatasetConfig,
    VisualInteractionWindowDataset,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.fold_preprocessing import (
    FoldPreprocessingState,
    file_sha256,
    fit_fold_preprocessing,
)
from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    _batch_from_indices,
    _load_bundle,
    _sample_indices,
    _validate_dataset_alignment,
)
from pig_behavior.classification_v2.training.multitask_loss import (
    build_auxiliary_label_maps,
    encode_auxiliary_batch,
)

MODEL_INPUT_KEYS = frozenset(
    {
        "image",
        "length_mask",
        "image_length_mask",
        "image_observed_mask",
        "spatial_features",
        "spatial_length_mask",
        "spatial_observed_mask",
        "interaction_context_features",
        "interaction_context_available_mask",
        "visual_context_image",
        "visual_context_length_mask",
        "visual_context_observed_mask",
    }
)


@dataclass(slots=True)
class StrictTrainingBatch:
    """Keep model X, supervised targets, and audit metadata in separate namespaces."""

    model_inputs: dict[str, Any]
    behavior_target: torch.Tensor
    auxiliary_targets: dict[str, torch.Tensor]
    auxiliary_masks: dict[str, torch.Tensor]
    sample_weight: torch.Tensor
    metadata: dict[str, Any]


class StrictTrainingDataModule:
    """Load immutable artifacts and prove row/key alignment before batching."""

    def __init__(self, config: ClassificationV2TrainingConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.full_config = _to_full_config(config, device)
        self.bundle = _load_bundle(self.full_config)
        self._attach_grouped_roles()
        self.actor_dataset = ClassificationV2ImageSequenceDataset(
            ImageSequenceDatasetConfig(
                frame_context_csv=config.dataset.train_ready_root
                / "image_frame_context_manifest.csv",
                window_context_csv=config.dataset.train_ready_root
                / "image_window_context_manifest.csv",
                packed_image_cache_npy=config.dataset.actor_packed_cache,
                packed_image_cache_index_csv=config.dataset.actor_packed_index,
                image_size=config.model.image_size,
                require_complete=False,
                require_cached_images=config.dataset.strict_packed_cache,
            )
        )
        self.visual_dataset = VisualInteractionWindowDataset(
            VisualInteractionDatasetConfig(
                cache_manifest_csv=config.dataset.visual_cache_manifest,
                window_context_csv=config.dataset.train_ready_root
                / "image_window_context_manifest.csv",
                packed_cache_npy=config.dataset.visual_packed_cache,
                packed_cache_index_csv=config.dataset.visual_packed_index,
                require_packed_cache=config.dataset.strict_packed_cache,
            )
        )
        _validate_dataset_alignment(
            self.actor_dataset,
            self.visual_dataset,
            expected_window_ids=self.bundle.frame["window_id"],
        )
        self.auxiliary = _align_auxiliary(config.dataset.auxiliary_targets_csv, self.bundle.frame)
        self.auxiliary_label_maps = build_auxiliary_label_maps(self.auxiliary)
        self.label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
        self.spatial_audit_path = (
            config.dataset.train_ready_root / "spatial_sequence_audit.json"
        )
        self.spatial_feature_names = _load_spatial_feature_names(
            self.spatial_audit_path,
            self.bundle.arrays,
            config.model.spatial_feature_groups,
        )
        self.fold_preprocessing_state: FoldPreprocessingState | None = None
        self._validate_behavior_target_alignment()

    def close(self) -> None:
        self.actor_dataset.close()

    def __enter__(self) -> StrictTrainingDataModule:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fold_indices(self, *, train: bool) -> np.ndarray:
        """Backward-compatible outer-train/test index access."""

        return self.split_indices("train" if train else "test")

    def split_indices(self, role: str) -> np.ndarray:
        """Return eligible rows for one predeclared inner/outer role."""

        if role not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported grouped split role: {role}")
        mask = self.bundle.frame["eligible"] & self.bundle.frame["grouped_role"].eq(role)
        return np.flatnonzero(mask.to_numpy()).astype(np.int64)

    def balanced_smoke_indices(self, *, train: bool) -> np.ndarray:
        """Select deterministic per-class rows from the requested fold side."""

        return self.balanced_smoke_split("train" if train else "validation")

    def balanced_smoke_split(self, role: str) -> np.ndarray:
        """Select deterministic per-class rows from a declared grouped role."""

        if role not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported grouped smoke role: {role}")
        mask = self.bundle.frame["eligible"] & self.bundle.frame["grouped_role"].eq(role)
        return _sample_indices(
            self.bundle.frame,
            mask=mask,
            per_class=self.config.execution.smoke_per_class,
            seed=self.config.optimization.seed
            + {"train": 0, "validation": 10_000, "test": 20_000}[role],
        )

    def batch(self, indices: np.ndarray) -> StrictTrainingBatch:
        """Build one batch with strict cache access and key-aligned auxiliary targets."""

        raw = _batch_from_indices(
            self.actor_dataset,
            self.visual_dataset,
            self.bundle,
            indices,
            self.label_to_index,
            {label: 1.0 for label in VALID_BEHAVIORS},
            self.full_config,
            self.device,
        )
        self._apply_fold_preprocessing(raw)
        model_inputs = {
            key: (raw["image_length_mask"] if key == "length_mask" else raw[key])
            for key in MODEL_INPUT_KEYS
        }
        validate_model_inputs(model_inputs)
        auxiliary_rows = self.auxiliary.iloc[indices].reset_index(drop=True)
        auxiliary_targets, auxiliary_masks = encode_auxiliary_batch(
            auxiliary_rows,
            self.auxiliary_label_maps,
            device=self.device,
        )
        selected = self.bundle.frame.iloc[indices]
        return StrictTrainingBatch(
            model_inputs=model_inputs,
            behavior_target=raw["target"],
            auxiliary_targets=auxiliary_targets,
            auxiliary_masks=auxiliary_masks,
            sample_weight=raw["training_sample_weight"],
            metadata={
                "row_index": indices.astype(int).tolist(),
                "window_id": selected["window_id"].astype(str).tolist(),
                "temporal_unit_key": selected["temporal_unit_key"].astype(str).tolist(),
                "oof_fold_id": selected["oof_fold_id"].astype(str).tolist(),
                "source_type": selected["source_type"].astype(str).tolist(),
            },
        )

    def audit(self) -> dict[str, Any]:
        """Return key/hash/count evidence without exposing metadata to model X."""

        train_indices = self.fold_indices(train=True)
        validation_indices = self.split_indices("validation")
        test_indices = self.split_indices("test")
        return {
            "schema_version": "classification_v2_strict_data_module_audit_v1",
            "rows": int(len(self.bundle.frame)),
            "eligible_rows": int(self.bundle.frame["eligible"].sum()),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "fold_id": self.config.execution.fold_id,
            "duplicate_window_id": int(self.bundle.frame["window_id"].duplicated().sum()),
            "window_id_sha256": _ids_hash(self.bundle.frame["window_id"]),
            "auxiliary_window_id_sha256": _ids_hash(self.auxiliary["window_id"]),
            "model_input_keys": sorted(MODEL_INPUT_KEYS),
            "metadata_not_model_inputs": [
                "row_index",
                "window_id",
                "temporal_unit_key",
                "oof_fold_id",
                "source_type",
            ],
            "auxiliary_targets_not_model_inputs": True,
            "spatial_normalization": self.spatial_normalizer_audit(),
            "actor_image_load_audit": self.actor_dataset.image_load_audit(),
            "visual_context_load_audit": self.visual_dataset.load_audit(),
        }

    def fit_fold_preprocessor(self) -> FoldPreprocessingState:
        """Fit the complete configured fold from eligible training-role rows."""

        config_payload = json.dumps(
            training_config_to_jsonable(self.config),
            sort_keys=True,
            separators=(",", ":"),
        )
        state = fit_fold_preprocessing(
            self.bundle.frame,
            self.bundle.arrays,
            self.spatial_feature_names,
            fold_id=self.config.execution.fold_id,
            snapshot_sha256=file_sha256(self.config.dataset.snapshot_json),
            config_sha256=hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
            spatial_audit_sha256=file_sha256(self.spatial_audit_path),
            feature_groups=self.config.model.spatial_feature_groups,
            standardized_groups=self.config.model.standardize_spatial_groups,
        )
        self.fold_preprocessing_state = state
        return state

    def fit_spatial_normalizer(self, train_indices: np.ndarray) -> None:
        """Compatibility API that now requires the complete declared train role."""

        expected = self.split_indices("train")
        provided = np.asarray(train_indices, dtype=np.int64)
        if not np.array_equal(provided, expected):
            raise ValueError(
                "spatial preprocessing must fit the complete grouped train role: "
                f"provided={len(provided)}, expected={len(expected)}"
            )
        self.fit_fold_preprocessor()

    def spatial_normalizer_audit(self) -> dict[str, Any]:
        """Serialize fold-local normalization state without exposing it as model X metadata."""

        if self.fold_preprocessing_state is None:
            return {
                "fit_scope": "not_fitted",
                "groups": {},
                "state_sha256": None,
            }
        payload = self.fold_preprocessing_state.to_payload()
        return {
            **payload,
            "fit_scope": "eligible_grouped_train_role_only",
            "groups": payload["statistics"],
        }

    def _apply_fold_preprocessing(self, raw: dict[str, Any]) -> None:
        if self.fold_preprocessing_state is None:
            raise ValueError("fold preprocessing must be fitted before building a batch")
        raw["spatial_features"] = self.fold_preprocessing_state.transform_torch(
            raw["spatial_features"],
            length_mask=raw["spatial_length_mask"],
            observed_mask=raw["spatial_observed_mask"],
        )

    def _validate_behavior_target_alignment(self) -> None:
        auxiliary_behavior = (
            self.auxiliary["behavior_target"].fillna("").astype(str).reset_index(drop=True)
        )
        main_behavior = self.bundle.y.reset_index(drop=True)
        mismatch = auxiliary_behavior.ne(main_behavior)
        if mismatch.any():
            examples = np.flatnonzero(mismatch.to_numpy())[:10].tolist()
            raise ValueError(f"auxiliary/main behavior target mismatch rows: {examples}")

    def _attach_grouped_roles(self) -> None:
        """Join the configured fold's roles by native unit and reject missing lineage."""

        roles = pd.read_csv(
            self.config.dataset.grouped_fold_roles,
            usecols=["temporal_unit_key", "outer_fold_id", "role"],
            low_memory=False,
        )
        roles = roles.loc[
            roles["outer_fold_id"].astype(str).eq(self.config.execution.fold_id)
        ].copy()
        if roles["temporal_unit_key"].duplicated().any():
            raise ValueError("duplicate temporal_unit_key in configured grouped fold roles")
        role_map = roles.set_index("temporal_unit_key")["role"]
        self.bundle.frame["grouped_role"] = self.bundle.frame["temporal_unit_key"].map(role_map)
        missing = self.bundle.frame["grouped_role"].isna()
        eligible_missing = missing & self.bundle.frame["eligible"]
        if eligible_missing.any():
            raise ValueError(
                f"eligible window rows missing grouped role: {int(eligible_missing.sum())}"
            )
        self.bundle.frame.loc[missing, "grouped_role"] = "not_eligible"


def _align_auxiliary(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    auxiliary = pd.read_csv(path, low_memory=False)
    if auxiliary["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in auxiliary targets")
    ordered = frame[["window_id"]].copy()
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    merged = ordered.merge(auxiliary, on="window_id", how="left", validate="one_to_one")
    if merged["behavior_target"].isna().any():
        raise ValueError(
            f"missing auxiliary target rows: {int(merged['behavior_target'].isna().sum())}"
        )
    return merged.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def _to_full_config(
    config: ClassificationV2TrainingConfig, device: torch.device
) -> FullMultimodalOofConfig:
    return FullMultimodalOofConfig(
        root=config.dataset.train_ready_root,
        native_oof_fold_manifest_csv=config.dataset.native_oof_fold_manifest,
        packed_image_cache_npy=config.dataset.actor_packed_cache,
        packed_image_cache_index_csv=config.dataset.actor_packed_index,
        require_cached_images=config.dataset.strict_packed_cache,
        visual_context_cache_manifest_csv=config.dataset.visual_cache_manifest,
        visual_context_packed_cache_npy=config.dataset.visual_packed_cache,
        visual_context_packed_cache_index_csv=config.dataset.visual_packed_index,
        require_packed_visual_context=config.dataset.strict_packed_cache,
        image_size=config.model.image_size,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        device=str(device),
        sample_weight_policy="none",
        ablation_variant="full",
    )


def _ids_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str)).encode("utf-8")).hexdigest()


def _load_spatial_feature_names(
    audit_path: Path,
    arrays: dict[str, np.ndarray],
    feature_groups: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Bind tensor dimensions to the exact exporter feature order."""

    if not audit_path.exists():
        raise FileNotFoundError(
            "spatial feature-order audit is required for training: "
            f"{audit_path}"
        )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    declared = payload.get("feature_names")
    if not isinstance(declared, dict):
        raise ValueError("spatial sequence audit has no feature_names mapping")
    result: dict[str, tuple[str, ...]] = {}
    for group in feature_groups:
        if group not in arrays:
            raise ValueError(f"spatial array missing configured group={group}")
        names = tuple(str(value).strip() for value in declared.get(group, ()))
        dimension = int(np.asarray(arrays[group]).shape[-1])
        if len(names) != dimension or len(set(names)) != len(names):
            raise ValueError(
                f"spatial feature-order mismatch for {group}: "
                f"names={list(names)}, dimension={dimension}"
            )
        result[group] = names
    return result


def validate_model_inputs(model_inputs: dict[str, Any]) -> None:
    """Fail closed if metadata, targets, or undeclared tensors enter model X."""

    observed = set(model_inputs)
    missing = sorted(MODEL_INPUT_KEYS.difference(observed))
    forbidden = sorted(observed.difference(MODEL_INPUT_KEYS))
    if missing or forbidden:
        raise ValueError(f"model input contract mismatch: missing={missing}, forbidden={forbidden}")
