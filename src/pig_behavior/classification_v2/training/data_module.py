"""Strict key-aligned data module for classification_v2 multimodal training."""

from __future__ import annotations

import hashlib
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
from pig_behavior.classification_v2.training.config import ClassificationV2TrainingConfig
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
        self.actor_dataset = ClassificationV2ImageSequenceDataset(
            ImageSequenceDatasetConfig(
                frame_context_csv=config.dataset.train_ready_root / "image_frame_context_manifest.csv",
                window_context_csv=config.dataset.train_ready_root / "image_window_context_manifest.csv",
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
                window_context_csv=config.dataset.train_ready_root / "image_window_context_manifest.csv",
                packed_cache_npy=config.dataset.visual_packed_cache,
                packed_cache_index_csv=config.dataset.visual_packed_index,
                require_packed_cache=config.dataset.strict_packed_cache,
            )
        )
        _validate_dataset_alignment(self.actor_dataset, self.visual_dataset)
        self.auxiliary = _align_auxiliary(config.dataset.auxiliary_targets_csv, self.bundle.frame)
        self.auxiliary_label_maps = build_auxiliary_label_maps(self.auxiliary)
        self.label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
        self._validate_behavior_target_alignment()

    def close(self) -> None:
        self.actor_dataset.close()

    def __enter__(self) -> StrictTrainingDataModule:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fold_indices(self, *, train: bool) -> np.ndarray:
        """Return all eligible rows inside or outside the configured held-out fold."""

        fold = self.bundle.frame["oof_fold_id"].astype(str)
        fold_mask = fold.ne(self.config.execution.fold_id) if train else fold.eq(self.config.execution.fold_id)
        return np.flatnonzero((self.bundle.frame["eligible"] & fold_mask).to_numpy()).astype(np.int64)

    def balanced_smoke_indices(self, *, train: bool) -> np.ndarray:
        """Select deterministic per-class rows from the requested fold side."""

        fold = self.bundle.frame["oof_fold_id"].astype(str)
        mask = self.bundle.frame["eligible"] & (
            fold.ne(self.config.execution.fold_id) if train else fold.eq(self.config.execution.fold_id)
        )
        return _sample_indices(
            self.bundle.frame,
            mask=mask,
            per_class=self.config.execution.smoke_per_class,
            seed=self.config.optimization.seed + (0 if train else 10_000),
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
        model_inputs = {key: raw[key] for key in MODEL_INPUT_KEYS}
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
                "oof_fold_id": selected["oof_fold_id"].astype(str).tolist(),
                "source_type": selected["source_type"].astype(str).tolist(),
            },
        )

    def audit(self) -> dict[str, Any]:
        """Return key/hash/count evidence without exposing metadata to model X."""

        train_indices = self.fold_indices(train=True)
        eval_indices = self.fold_indices(train=False)
        return {
            "schema_version": "classification_v2_strict_data_module_audit_v1",
            "rows": int(len(self.bundle.frame)),
            "eligible_rows": int(self.bundle.frame["eligible"].sum()),
            "train_rows": int(len(train_indices)),
            "eval_rows": int(len(eval_indices)),
            "fold_id": self.config.execution.fold_id,
            "duplicate_window_id": int(self.bundle.frame["window_id"].duplicated().sum()),
            "window_id_sha256": _ids_hash(self.bundle.frame["window_id"]),
            "auxiliary_window_id_sha256": _ids_hash(self.auxiliary["window_id"]),
            "model_input_keys": sorted(MODEL_INPUT_KEYS),
            "metadata_not_model_inputs": ["row_index", "window_id", "oof_fold_id", "source_type"],
            "auxiliary_targets_not_model_inputs": True,
            "actor_image_load_audit": self.actor_dataset.image_load_audit(),
            "visual_context_load_audit": self.visual_dataset.load_audit(),
        }

    def _validate_behavior_target_alignment(self) -> None:
        auxiliary_behavior = self.auxiliary["behavior_target"].fillna("").astype(str).reset_index(drop=True)
        main_behavior = self.bundle.y.reset_index(drop=True)
        mismatch = auxiliary_behavior.ne(main_behavior)
        if mismatch.any():
            examples = np.flatnonzero(mismatch.to_numpy())[:10].tolist()
            raise ValueError(f"auxiliary/main behavior target mismatch rows: {examples}")


def _align_auxiliary(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    auxiliary = pd.read_csv(path, low_memory=False)
    if auxiliary["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in auxiliary targets")
    ordered = frame[["window_id"]].copy()
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    merged = ordered.merge(auxiliary, on="window_id", how="left", validate="one_to_one")
    if merged["behavior_target"].isna().any():
        raise ValueError(f"missing auxiliary target rows: {int(merged['behavior_target'].isna().sum())}")
    return merged.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def _to_full_config(config: ClassificationV2TrainingConfig, device: torch.device) -> FullMultimodalOofConfig:
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
