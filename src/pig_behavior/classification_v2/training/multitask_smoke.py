"""Bounded multitask overfit smoke using the audited multimodal data path."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.datasets.visual_interaction_loader import (
    VisualInteractionDatasetConfig,
    VisualInteractionWindowDataset,
)
from pig_behavior.classification_v2.models.multimodal_fusion import MultimodalFusionConfig
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
    MultitaskFusionClassifier,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    _batch_from_indices,
    _load_bundle,
    _sample_indices,
    _validate_dataset_alignment,
)
from pig_behavior.classification_v2.training.multitask_loss import (
    build_auxiliary_label_maps,
    build_fold_auxiliary_class_weights,
    encode_auxiliary_batch,
    hierarchy_consistency_loss,
    masked_multitask_loss,
)


@dataclass(frozen=True, slots=True)
class MultitaskSmokeConfig:
    root: Path = Path("outputs/classification_v2/train_ready_windows")
    output_dir: Path = Path("outputs/classification_v2/model_smoke/multitask_visual_v3")
    actor_packed_cache: Path = Path("outputs/classification_v2/image_cache_v2_letterbox/packed_rgb_64_letterbox.npy")
    actor_packed_index: Path = Path(
        "outputs/classification_v2/image_cache_v2_letterbox/packed_image_cache_index.csv"
    )
    visual_cache_manifest: Path = Path(
        "outputs/classification_v2/visual_interaction_cache/visual_context_manifest.csv"
    )
    visual_packed_cache: Path = Path(
        "outputs/classification_v2/visual_interaction_cache/packed_rgb_64_letterbox.npy"
    )
    visual_packed_index: Path = Path(
        "outputs/classification_v2/visual_interaction_cache/packed_image_cache_index.csv"
    )
    image_size: int = 64
    hidden_dim: int = 32
    steps: int = 8
    per_class: int = 1
    learning_rate: float = 0.005
    consistency_weight: float = 0.1
    seed: int = 123
    device: str = "auto"


def run_multitask_smoke(config: MultitaskSmokeConfig) -> dict[str, Any]:
    """Overfit a bounded balanced batch and emit per-head optimization evidence."""

    if config.steps <= 0 or config.per_class <= 0:
        raise ValueError("steps and per_class must be positive")
    _set_seed(config.seed)
    resolved_device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if config.device == "auto" and not torch.cuda.is_available():
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    full_config = FullMultimodalOofConfig(
        root=config.root,
        image_size=config.image_size,
        hidden_dim=config.hidden_dim,
        device=str(device),
        sample_weight_policy="none",
        packed_image_cache_npy=config.actor_packed_cache,
        packed_image_cache_index_csv=config.actor_packed_index,
        require_cached_images=True,
        visual_context_cache_manifest_csv=config.visual_cache_manifest,
        visual_context_packed_cache_npy=config.visual_packed_cache,
        visual_context_packed_cache_index_csv=config.visual_packed_index,
        require_packed_visual_context=True,
    )
    bundle = _load_bundle(full_config)
    fold_id = sorted(bundle.frame.loc[bundle.frame["eligible"], "oof_fold_id"].astype(str).unique())[0]
    train_mask = bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).ne(fold_id)
    indices = _sample_indices(bundle.frame, mask=train_mask, per_class=config.per_class, seed=config.seed)
    if len(indices) != len(VALID_BEHAVIORS) * config.per_class:
        raise ValueError(f"balanced smoke selection incomplete: rows={len(indices)}")

    actor_dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=config.root / "image_frame_context_manifest.csv",
            window_context_csv=config.root / "image_window_context_manifest.csv",
            packed_image_cache_npy=config.actor_packed_cache,
            packed_image_cache_index_csv=config.actor_packed_index,
            image_size=config.image_size,
            require_complete=False,
            require_cached_images=True,
        )
    )
    visual_dataset = VisualInteractionWindowDataset(
        VisualInteractionDatasetConfig(
            cache_manifest_csv=config.visual_cache_manifest,
            window_context_csv=config.root / "image_window_context_manifest.csv",
            packed_cache_npy=config.visual_packed_cache,
            packed_cache_index_csv=config.visual_packed_index,
            require_packed_cache=True,
        )
    )
    _validate_dataset_alignment(actor_dataset, visual_dataset)
    label_to_idx = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    try:
        batch = _batch_from_indices(
            actor_dataset,
            visual_dataset,
            bundle,
            indices,
            label_to_idx,
            {label: 1.0 for label in VALID_BEHAVIORS},
            full_config,
            device,
        )
    finally:
        actor_dataset.close()

    auxiliary = _load_aligned_auxiliary(config.root, bundle.frame, indices)
    all_auxiliary = pd.read_csv(config.root / "y_auxiliary_targets.csv", low_memory=False)
    label_maps = build_auxiliary_label_maps(all_auxiliary)
    fold_auxiliary = _load_aligned_auxiliary(
        config.root,
        bundle.frame,
        np.flatnonzero(train_mask.to_numpy()),
    )
    class_weights = build_fold_auxiliary_class_weights(fold_auxiliary, label_maps, device=device)
    auxiliary_targets, auxiliary_masks = encode_auxiliary_batch(auxiliary, label_maps, device=device)
    model = MultitaskFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={name: int(bundle.arrays[name].shape[-1]) for name in batch["spatial_features"]},
            num_classes=len(VALID_BEHAVIORS),
            interaction_context_dim=len(INTERACTION_CONTEXT_FEATURE_COLUMNS),
            image_embedding_dim=config.hidden_dim,
            spatial_embedding_dim=config.hidden_dim,
            interaction_embedding_dim=max(8, config.hidden_dim // 2),
            visual_context_embedding_dim=config.hidden_dim,
            fusion_hidden_dim=config.hidden_dim,
            dropout=0.0,
            enable_visual_context=True,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    behavior_loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    model_inputs = _model_inputs(batch)
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs)
        behavior_loss = behavior_loss_fn(output.behavior, batch["target"])
        auxiliary_loss, auxiliary_audit = masked_multitask_loss(
            output.auxiliary_logits(),
            auxiliary_targets,
            auxiliary_masks,
            class_weights_by_task=class_weights,
        )
        consistency = hierarchy_consistency_loss(output.behavior, output.auxiliary_logits())
        total = behavior_loss + auxiliary_loss + config.consistency_weight * consistency
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "total": float(total.detach().cpu().item()),
                "behavior": float(behavior_loss.detach().cpu().item()),
                "auxiliary": float(auxiliary_loss.detach().cpu().item()),
                "consistency": float(consistency.detach().cpu().item()),
                **{f"aux_{name}": value for name, value in auxiliary_audit["loss_values"].items()},
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "multitask_smoke.pt"
    audit_path = config.output_dir / "multitask_smoke_audit.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _jsonable_config(config),
            "label_order": list(VALID_BEHAVIORS),
            "auxiliary_label_order": label_maps,
        },
        checkpoint_path,
    )
    initial, final = history[0], history[-1]
    errors: list[str] = []
    if final["total"] >= initial["total"]:
        errors.append("total_loss_did_not_decrease")
    if final["behavior"] >= initial["behavior"]:
        errors.append("behavior_loss_did_not_decrease")
    audit = {
        "schema_version": "classification_v2_multitask_smoke_audit_v1",
        "interpretation": "bounded_trainability_smoke_not_model_quality_evidence",
        "model_architecture_version": MULTITASK_ARCHITECTURE_VERSION,
        "config": _jsonable_config(config),
        "fold_id": fold_id,
        "rows": int(len(indices)),
        "label_counts": auxiliary["behavior_target"].value_counts().sort_index().to_dict(),
        "auxiliary_label_order": label_maps,
        "auxiliary_active_rows": {
            name: int(mask.sum().detach().cpu().item()) for name, mask in auxiliary_masks.items()
        },
        "fold_local_class_weights": {
            name: value.detach().cpu().tolist() for name, value in class_weights.items()
        },
        "initial_losses": initial,
        "final_losses": final,
        "loss_history": history,
        "actor_image_load_audit": actor_dataset.image_load_audit(),
        "visual_context_load_audit": visual_dataset.load_audit(),
        "checkpoint_path": str(checkpoint_path),
        "auxiliary_targets_used_as_model_inputs": False,
        "errors": errors,
        "valid": not errors,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"multitask smoke failed: {errors}")
    return audit


def _load_aligned_auxiliary(root: Path, frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Join auxiliary y/masks by window_id instead of trusting row order."""

    auxiliary = pd.read_csv(root / "y_auxiliary_targets.csv", low_memory=False)
    if auxiliary["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in auxiliary targets")
    selected = frame.iloc[indices][["window_id"]].copy()
    selected["_order"] = np.arange(len(selected))
    merged = selected.merge(auxiliary, on="window_id", how="left", validate="one_to_one")
    if merged["behavior_target"].isna().any():
        raise ValueError("selected windows missing auxiliary targets")
    return merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": batch["image"],
        "spatial_features": batch["spatial_features"],
        "length_mask": batch["image_length_mask"],
        "image_length_mask": batch["image_length_mask"],
        "image_observed_mask": batch["image_observed_mask"],
        "spatial_length_mask": batch["spatial_length_mask"],
        "spatial_observed_mask": batch["spatial_observed_mask"],
        "interaction_context_features": batch["interaction_context_features"],
        "interaction_context_available_mask": batch["interaction_context_available_mask"],
        "visual_context_image": batch["visual_context_image"],
        "visual_context_length_mask": batch["visual_context_length_mask"],
        "visual_context_observed_mask": batch["visual_context_observed_mask"],
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: MultitaskSmokeConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
