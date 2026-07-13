"""Learned multimodal native-OOF runner for classification_v2.

The default configuration is a small OOF pilot that exercises the same data
boundaries as the future full evaluation: actor crop images, whitelisted spatial
sequence tensors, interaction context tensors, native temporal fold splits, and
the S16 prediction schema. Full paper-facing evaluation must be launched
explicitly and registered separately; the pilot is engineering evidence only.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
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
    image_sequence_collate,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
    InteractionContextDatasetConfig,
    InteractionContextWindowDataset,
)
from pig_behavior.classification_v2.datasets.visual_interaction_loader import (
    VisualInteractionDatasetConfig,
    VisualInteractionWindowDataset,
    visual_interaction_collate,
)
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import (
    check_prediction_schema,
)
from pig_behavior.classification_v2.evaluation.source_balanced_reporting import (
    build_source_balanced_native_report,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MODEL_ARCHITECTURE_VERSION,
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS

ABLATION_VARIANTS = (
    "full",
    "image_only",
    "spatial_only",
    "no_interaction",
    "no_visual_context",
    "no_roi",
    "no_social",
    "no_motion",
)
SAMPLE_WEIGHT_POLICIES = ("none", "window", "event", "event_class")
PRECISION_POLICIES = ("fp32", "amp")


@dataclass(frozen=True, slots=True)
class FullMultimodalOofConfig:
    """Deterministic options for pilot or full learned native-OOF runs."""

    root: Path = Path("outputs/classification_v2/train_ready_windows")
    sequence_manifest_csv: Path = Path(
        "outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"
    )
    interaction_context_manifest_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/interaction_window_context_manifest.csv"
    )
    visual_context_cache_manifest_csv: Path = Path(
        "outputs/classification_v2/visual_interaction_cache/visual_context_manifest.csv"
    )
    visual_context_packed_cache_npy: Path | None = None
    visual_context_packed_cache_index_csv: Path | None = None
    require_packed_visual_context: bool = False
    native_oof_fold_manifest_csv: Path = Path(
        "outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"
    )
    event_weight_manifest_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/event_weight_manifest.csv"
    )
    output_dir: Path = Path("outputs/classification_v2/model_smoke/full_multimodal_oof_pilot")
    image_cache_manifest_csv: Path | None = None
    packed_image_cache_npy: Path | None = None
    packed_image_cache_index_csv: Path | None = None
    require_cached_images: bool = False
    image_size: int = 32
    hidden_dim: int = 32
    dropout: float = 0.1
    lr: float = 0.003
    weight_decay: float = 0.0
    steps_per_fold: int = 2
    epochs_per_fold: int = 3
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_folds: int | None = 2
    train_per_class_per_fold: int | None = 2
    eval_per_class_per_fold: int | None = 1
    bootstrap_iterations: int = 30
    seed: int = 20260710
    device: str = "auto"
    run_mode: str = "pilot"
    resume: bool = True
    ablation_variant: str = "full"
    sample_weight_policy: str = "event_class"
    class_weight_power: float = 0.5
    class_weight_max: float = 5.0
    precision: str = "fp32"
    checkpoint_every_steps: int = 500
    model_architecture_version: str = MODEL_ARCHITECTURE_VERSION


def run_full_multimodal_oof(config: FullMultimodalOofConfig) -> dict[str, Any]:
    """Run learned multimodal native-OOF pilot/full evaluation and write artifacts."""

    _validate_config(config)
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    if config.precision == "amp" and device.type != "cuda":
        raise ValueError("precision=amp requires a CUDA device; no silent fp32 fallback is allowed")
    bundle = _load_bundle(config)
    label_order = list(VALID_BEHAVIORS)
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    fold_ids = sorted(
        bundle.frame.loc[bundle.frame["eligible"], "oof_fold_id"].astype(str).unique()
    )
    if config.max_folds is not None:
        fold_ids = fold_ids[: int(config.max_folds)]
    if not fold_ids:
        raise ValueError("no eligible OOF folds available")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    fold_artifact_dir = config.output_dir / "fold_artifacts"
    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=config.root / "image_frame_context_manifest.csv",
            window_context_csv=config.root / "image_window_context_manifest.csv",
            image_cache_manifest_csv=config.image_cache_manifest_csv,
            packed_image_cache_npy=config.packed_image_cache_npy,
            packed_image_cache_index_csv=config.packed_image_cache_index_csv,
            image_size=config.image_size,
            require_complete=False,
            require_cached_images=config.require_cached_images,
        )
    )
    visual_context_dataset = VisualInteractionWindowDataset(
        VisualInteractionDatasetConfig(
            cache_manifest_csv=config.visual_context_cache_manifest_csv,
            window_context_csv=config.root / "image_window_context_manifest.csv",
            packed_cache_npy=config.visual_context_packed_cache_npy,
            packed_cache_index_csv=config.visual_context_packed_cache_index_csv,
            require_packed_cache=config.require_packed_visual_context,
        )
    )
    _validate_dataset_alignment(
        dataset,
        visual_context_dataset,
        expected_window_ids=bundle.frame["window_id"],
    )
    predictions: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    try:
        for fold_id in fold_ids:
            fold_predictions, fold_audit = _load_or_run_one_fold(
                dataset,
                visual_context_dataset,
                bundle,
                config,
                fold_id,
                label_order,
                label_to_idx,
                device,
                fold_artifact_dir,
            )
            predictions.append(fold_predictions)
            fold_audits.append(fold_audit)
    finally:
        dataset.close()

    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["oof_fold_id", "window_id"],
        kind="mergesort",
    )
    prediction_schema_audit = check_prediction_schema(prediction_frame)
    native_units, metrics_payload = build_native_temporal_metrics(
        prediction_frame,
        NativeTemporalMetricsConfig(bootstrap_iterations=int(config.bootstrap_iterations)),
    )
    full_oof_verified = _is_full_run(config, bundle) and _fold_training_coverage_complete(
        fold_audits
    )
    source_native_units, source_selection, source_report = build_source_balanced_native_report(
        prediction_frame,
        bundle.frame[["window_id", "source_type"]],
        expected_fold_count=len(fold_ids),
        paper_facing_run_verified=full_oof_verified,
    )
    paper_facing_result = full_oof_verified and bool(source_report.get("paper_facing_ready"))

    predictions_path = config.output_dir / "full_multimodal_oof_predictions.csv"
    native_units_path = config.output_dir / "full_multimodal_oof_unit_predictions.csv"
    metrics_path = config.output_dir / "full_multimodal_oof_metrics.json"
    schema_audit_path = config.output_dir / "full_multimodal_oof_prediction_schema_audit.json"
    audit_path = config.output_dir / "full_multimodal_oof_audit.json"
    source_native_units_path = config.output_dir / "source_balanced_native_units.csv"
    source_selection_path = config.output_dir / "source_balanced_selection.csv"
    source_report_path = config.output_dir / "source_balanced_report.json"
    prediction_frame.to_csv(predictions_path, index=False)
    native_units.to_csv(native_units_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    schema_audit_path.write_text(json.dumps(prediction_schema_audit, indent=2), encoding="utf-8")
    source_native_units.to_csv(source_native_units_path, index=False)
    source_selection.to_csv(source_selection_path, index=False)
    source_report_path.write_text(json.dumps(source_report, indent=2), encoding="utf-8")

    image_load_audit = dataset.image_load_audit()
    git_state = _git_state()
    audit = {
        "schema_version": "classification_v2_full_multimodal_oof_audit_v1",
        "run_mode": config.run_mode,
        "paper_facing_result": paper_facing_result,
        "full_oof_training_verified": full_oof_verified,
        "config": _jsonable_config(config),
        "ablation_settings": _ablation_settings(config.ablation_variant),
        "device": str(device),
        "git_commit": git_state["commit"],
        "git_dirty": git_state["dirty"],
        "label_order": label_order,
        "load_audit": bundle.load_audit,
        "image_load_audit": image_load_audit,
        "visual_context_load_audit": visual_context_dataset.load_audit(),
        "fold_artifact_dir": str(fold_artifact_dir),
        "fold_audits": fold_audits,
        "prediction_rows": int(len(prediction_frame)),
        "native_temporal_rows": int(
            metrics_payload.get("native_temporal_prediction_audit", {}).get(
                "native_temporal_unit_rows", 0
            )
        ),
        "metrics_json": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "prediction_schema_audit_json": str(schema_audit_path),
        "prediction_schema_valid": bool(prediction_schema_audit.get("valid")),
        "source_balanced_report_json": str(source_report_path),
        "source_balanced_native_units_csv": str(source_native_units_path),
        "source_balanced_selection_csv": str(source_selection_path),
        "source_balanced_report_valid": bool(source_report.get("valid")),
        "source_balanced_paper_facing_ready": bool(source_report.get("paper_facing_ready")),
        "errors": [],
        "warnings": _mode_warnings(config, bundle),
        "valid": bool(not prediction_frame.empty and prediction_schema_audit.get("valid") is True),
    }
    if prediction_schema_audit.get("errors"):
        audit["errors"].append(f"prediction_schema_errors={prediction_schema_audit.get('errors')}")
    if metrics_payload.get("native_temporal_prediction_audit", {}).get("errors"):
        audit["errors"].append(
            "native_temporal_prediction_errors="
            f"{metrics_payload.get('native_temporal_prediction_audit', {}).get('errors')}"
        )
    if source_report.get("errors"):
        message = f"source_balanced_report_errors={source_report.get('errors')}"
        if full_oof_verified:
            audit["errors"].append(message)
        else:
            audit["warnings"].append(message)
    if config.require_cached_images and (
        image_load_audit["disk_image_cache_misses"] > 0
        or image_load_audit["source_image_loads"] > 0
    ):
        audit["errors"].append(f"strict_image_cache_violation={image_load_audit}")
    audit["valid"] = audit["valid"] and not audit["errors"]
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"full multimodal OOF run failed: {audit['errors']}")
    return {
        "audit_json": str(audit_path),
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "metrics_json": str(metrics_path),
        "prediction_schema_audit_json": str(schema_audit_path),
        "source_balanced_report_json": str(source_report_path),
        "audit": audit,
    }


def build_full_multimodal_oof_run_plan(config: FullMultimodalOofConfig) -> dict[str, Any]:
    """Summarize full/pilot OOF workload before loading images or training."""

    _validate_config(config)
    bundle = _load_bundle(config)
    fold_ids = sorted(
        bundle.frame.loc[bundle.frame["eligible"], "oof_fold_id"].astype(str).unique()
    )
    selected_fold_ids = fold_ids if config.max_folds is None else fold_ids[: int(config.max_folds)]
    fold_rows: list[dict[str, Any]] = []
    total_eval_rows = 0
    total_train_steps = 0
    for fold_id in selected_fold_ids:
        train_mask = bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).ne(
            str(fold_id)
        )
        eval_mask = bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).eq(
            str(fold_id)
        )
        train_indices = _sample_indices(
            bundle.frame,
            mask=train_mask,
            per_class=config.train_per_class_per_fold,
            seed=config.seed + int(_stable_fold_offset(fold_id)),
        )
        eval_indices = _sample_indices(
            bundle.frame,
            mask=eval_mask,
            per_class=config.eval_per_class_per_fold,
            seed=config.seed + 10_000 + int(_stable_fold_offset(fold_id)),
        )
        eval_batches = _ceil_div(len(eval_indices), int(config.eval_batch_size))
        fold_rows.append(
            {
                "oof_fold_id": str(fold_id),
                "train_rows": int(len(train_indices)),
                "eval_rows": int(len(eval_indices)),
                "steps_per_fold": int(config.steps_per_fold),
                "epochs_per_fold": int(config.epochs_per_fold),
                "effective_training_steps": int(
                    _effective_training_step_count(config, len(train_indices))
                ),
                "train_batch_size": int(config.train_batch_size),
                "eval_batch_size": int(config.eval_batch_size),
                "eval_batches": int(eval_batches),
                "train_label_counts": _label_counts(bundle.frame, train_indices),
                "eval_label_counts": _label_counts(bundle.frame, eval_indices),
            }
        )
        total_eval_rows += int(len(eval_indices))
        total_train_steps += _effective_training_step_count(config, len(train_indices))
    full_like = _is_full_run(config, bundle)
    warnings = []
    if not full_like:
        warnings.append("bounded_or_pilot_plan_not_valid_for_full_paper_record")
    if len(selected_fold_ids) != bundle.load_audit.get("eligible_fold_count"):
        warnings.append("selected_fold_count_less_than_available_fold_count")
    return {
        "schema_version": "classification_v2_full_multimodal_oof_run_plan_v1",
        "config": _jsonable_config(config),
        "config_sha256": full_run_config_fingerprint(config),
        "run_mode": config.run_mode,
        "paper_facing_candidate_plan": full_like,
        "load_audit": bundle.load_audit,
        "available_fold_count": int(bundle.load_audit.get("eligible_fold_count", 0)),
        "selected_fold_count": int(len(selected_fold_ids)),
        "total_eval_rows": int(total_eval_rows),
        "total_train_steps": int(total_train_steps),
        "folds": fold_rows,
        "warnings": warnings,
        "errors": [],
        "valid": bool(selected_fold_ids and total_eval_rows > 0),
    }


def full_run_config_fingerprint(config: FullMultimodalOofConfig) -> str:
    """Hash the exact JSON-safe run config bound to a no-training preflight."""

    payload = json.dumps(_jsonable_config(config), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class _OofBundle:
    """Aligned tensors and metadata needed by the learned OOF runner."""

    arrays: dict[str, np.ndarray]
    interaction_context_features: np.ndarray
    interaction_context_available_mask: np.ndarray
    event_sample_weights: np.ndarray
    y: pd.Series
    frame: pd.DataFrame
    load_audit: dict[str, Any]


def _run_one_fold(
    dataset: ClassificationV2ImageSequenceDataset,
    visual_context_dataset: VisualInteractionWindowDataset,
    bundle: _OofBundle,
    config: FullMultimodalOofConfig,
    fold_id: str,
    label_order: list[str],
    label_to_idx: dict[str, int],
    device: torch.device,
    fold_work_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on all non-held-out native folds and predict the held-out fold."""

    train_indices = _sample_indices(
        bundle.frame,
        mask=bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).ne(str(fold_id)),
        per_class=config.train_per_class_per_fold,
        seed=config.seed + int(_stable_fold_offset(fold_id)),
    )
    eval_indices = _sample_indices(
        bundle.frame,
        mask=bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).eq(str(fold_id)),
        per_class=config.eval_per_class_per_fold,
        seed=config.seed + 10_000 + int(_stable_fold_offset(fold_id)),
    )
    if len(train_indices) < 2 or len(eval_indices) < 1:
        raise ValueError(f"fold {fold_id} has insufficient train/eval rows")
    class_weights = _fold_local_class_weights(bundle, train_indices, config)
    train_sample_weights = _training_sample_weights(bundle, train_indices, class_weights, config)
    model = MultimodalFusionClassifier(
        # Variant settings alter instantiated branches and spatial input dims,
        # so ablation outputs cannot receive signal from disabled tensors.
        MultimodalFusionConfig(
            spatial_input_dims={
                name: int(bundle.arrays[name].shape[-1])
                for name in _ablation_settings(config.ablation_variant)["spatial_groups"]
            },
            num_classes=len(label_order),
            interaction_context_dim=(
                len(INTERACTION_CONTEXT_FEATURE_COLUMNS)
                if _ablation_settings(config.ablation_variant)["enable_interaction"]
                else None
            ),
            image_embedding_dim=config.hidden_dim,
            spatial_embedding_dim=config.hidden_dim,
            interaction_embedding_dim=max(8, config.hidden_dim // 2),
            visual_context_embedding_dim=config.hidden_dim,
            fusion_hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            enable_image=bool(_ablation_settings(config.ablation_variant)["enable_image"]),
            enable_spatial=bool(_ablation_settings(config.ablation_variant)["enable_spatial"]),
            enable_interaction_context=bool(
                _ablation_settings(config.ablation_variant)["enable_interaction"]
            ),
            enable_visual_context=bool(
                _ablation_settings(config.ablation_variant)["enable_visual_context"]
            ),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    amp_enabled = config.precision == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    rng = np.random.default_rng(config.seed + int(_stable_fold_offset(fold_id)))
    losses: list[float] = []
    seen_train_indices: set[int] = set()
    resumed_training = False
    training_elapsed_sec = 0.0
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    training_signature = _fold_training_signature(config, fold_id, train_indices, eval_indices)
    training_audit_path = (
        fold_work_dir / "training_audit.json" if fold_work_dir is not None else None
    )
    model_state_path = fold_work_dir / "trained_model.pt" if fold_work_dir is not None else None
    completed_training_steps = 0
    if fold_work_dir is not None:
        fold_work_dir.mkdir(parents=True, exist_ok=True)
    if (
        config.resume
        and training_audit_path is not None
        and model_state_path is not None
        and training_audit_path.exists()
        and model_state_path.exists()
    ):
        training_audit = json.loads(training_audit_path.read_text(encoding="utf-8"))
        if training_audit.get("training_signature") != training_signature:
            raise ValueError(f"stale fold training checkpoint signature for fold {fold_id}")
        losses = [float(value) for value in training_audit.get("losses", [])]
        seen_train_indices = {int(value) for value in training_audit.get("seen_train_indices", [])}
        completed_training_steps = int(training_audit.get("completed_training_steps", len(losses)))
        training_elapsed_sec = float(training_audit.get("training_elapsed_sec", 0.0))
        peak_allocated_mb = float(training_audit.get("cuda_peak_memory_allocated_mb", 0.0))
        peak_reserved_mb = float(training_audit.get("cuda_peak_memory_reserved_mb", 0.0))
        checkpoint = torch.load(model_state_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            if checkpoint.get("training_signature") != training_signature:
                raise ValueError(f"stale fold checkpoint payload signature for fold {fold_id}")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scaler_state_dict"):
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            losses = [float(value) for value in checkpoint.get("losses", [])]
            seen_train_indices = {int(value) for value in checkpoint.get("seen_train_indices", [])}
            completed_training_steps = int(checkpoint.get("completed_training_steps", len(losses)))
            training_elapsed_sec = float(checkpoint.get("training_elapsed_sec", 0.0))
            peak_allocated_mb = float(checkpoint.get("cuda_peak_memory_allocated_mb", 0.0))
            peak_reserved_mb = float(checkpoint.get("cuda_peak_memory_reserved_mb", 0.0))
        else:
            # Version-1 completed checkpoints stored only the model state dict.
            model.load_state_dict(checkpoint)
        resumed_training = True
    expected_training_steps = _effective_training_step_count(config, len(train_indices))
    if completed_training_steps > expected_training_steps:
        raise ValueError(
            f"fold {fold_id} checkpoint exceeds planned steps: "
            f"completed={completed_training_steps}, expected={expected_training_steps}"
        )
    if completed_training_steps < expected_training_steps:
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        training_started = time.perf_counter()
        for step_index, batch_indices in enumerate(_training_batches(config, train_indices, rng)):
            if step_index < completed_training_steps:
                continue
            seen_train_indices.update(int(value) for value in batch_indices)
            train_batch = _batch_from_indices(
                dataset,
                visual_context_dataset,
                bundle,
                batch_indices,
                label_to_idx,
                class_weights,
                config,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward_model(model, train_batch)
                per_row_loss = loss_fn(logits, train_batch["target"])
                batch_weights = train_batch["training_sample_weight"]
                if float(batch_weights.sum().detach().cpu().item()) <= 0.0:
                    raise ValueError(f"fold {fold_id} produced an all-zero training-weight batch")
                loss = (per_row_loss * batch_weights).sum() / batch_weights.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
            completed_training_steps += 1
            if (
                training_audit_path is not None
                and model_state_path is not None
                and completed_training_steps % int(config.checkpoint_every_steps) == 0
            ):
                elapsed = training_elapsed_sec + float(time.perf_counter() - training_started)
                _save_training_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    model_state_path,
                    training_audit_path,
                    training_signature=training_signature,
                    losses=losses,
                    seen_train_indices=seen_train_indices,
                    completed_training_steps=completed_training_steps,
                    training_elapsed_sec=elapsed,
                    peak_allocated_mb=peak_allocated_mb,
                    peak_reserved_mb=peak_reserved_mb,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
            peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024**2))
        training_elapsed_sec += float(time.perf_counter() - training_started)
        if training_audit_path is not None and model_state_path is not None:
            _save_training_checkpoint(
                model,
                optimizer,
                scaler,
                model_state_path,
                training_audit_path,
                training_signature=training_signature,
                losses=losses,
                seen_train_indices=seen_train_indices,
                completed_training_steps=completed_training_steps,
                training_elapsed_sec=training_elapsed_sec,
                peak_allocated_mb=peak_allocated_mb,
                peak_reserved_mb=peak_reserved_mb,
            )
    if not losses:
        raise ValueError(f"fold {fold_id} has no recorded training losses")
    chunk_dir = fold_work_dir / "prediction_chunks" if fold_work_dir is not None else None
    predictions = _predict_in_batches(
        dataset,
        visual_context_dataset,
        model,
        bundle,
        eval_indices,
        label_to_idx,
        label_order,
        fold_id,
        config,
        device,
        chunk_dir,
        class_weights,
    )
    prediction_chunk_count = (
        len(list(chunk_dir.glob("*.csv"))) if chunk_dir is not None and chunk_dir.exists() else 0
    )
    audit = {
        "oof_fold_id": str(fold_id),
        "ablation_variant": config.ablation_variant,
        "ablation_settings": _ablation_settings(config.ablation_variant),
        "instantiated_branches": {
            "image": model.image_encoder is not None,
            "spatial": model.spatial_encoder is not None,
            "interaction": model.interaction_context_encoder is not None,
            "visual_context": model.visual_context_encoder is not None,
        },
        "spatial_branch_order": (
            list(model.spatial_encoder.branch_order) if model.spatial_encoder is not None else []
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "train_indices_sha256": training_signature["train_indices_sha256"],
        "eval_indices_sha256": training_signature["eval_indices_sha256"],
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "train_batch_size": int(config.train_batch_size),
        "eval_batch_size": int(config.eval_batch_size),
        "train_label_counts": bundle.frame.iloc[train_indices]["behavior_true"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "eval_label_counts": bundle.frame.iloc[eval_indices]["behavior_true"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_reduction": float(losses[0] - losses[-1]),
        "training_steps_completed": int(completed_training_steps),
        "expected_training_steps": int(expected_training_steps),
        "unique_train_rows_seen": int(len(seen_train_indices)),
        "train_row_coverage_ratio": float(len(seen_train_indices) / len(train_indices)),
        "sample_weight_policy": config.sample_weight_policy,
        "fold_local_class_weights": class_weights,
        "training_weight_min": float(train_sample_weights.min()),
        "training_weight_max": float(train_sample_weights.max()),
        "training_weight_mean": float(train_sample_weights.mean()),
        "training_zero_weight_rows": int(np.count_nonzero(train_sample_weights == 0.0)),
        "precision": config.precision,
        "amp_enabled": bool(amp_enabled),
        "training_elapsed_sec": training_elapsed_sec,
        "optimizer_steps_per_sec": float(len(losses) / training_elapsed_sec)
        if training_elapsed_sec > 0.0
        else 0.0,
        "training_rows_per_sec": (
            float(len(losses) * config.train_batch_size / training_elapsed_sec)
            if training_elapsed_sec > 0.0
            else 0.0
        ),
        "cuda_peak_memory_allocated_mb": peak_allocated_mb,
        "cuda_peak_memory_reserved_mb": peak_reserved_mb,
        "resumed_training_checkpoint": bool(resumed_training),
        "prediction_chunk_count": int(prediction_chunk_count),
        "fold_work_dir": str(fold_work_dir) if fold_work_dir is not None else None,
    }
    return predictions, audit


def _load_or_run_one_fold(
    dataset: ClassificationV2ImageSequenceDataset,
    visual_context_dataset: VisualInteractionWindowDataset,
    bundle: _OofBundle,
    config: FullMultimodalOofConfig,
    fold_id: str,
    label_order: list[str],
    label_to_idx: dict[str, int],
    device: torch.device,
    fold_artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resume a completed fold or train/predict it and save fold artifacts."""

    safe_fold_id = _safe_fold_id(fold_id)
    predictions_path = fold_artifact_dir / f"{safe_fold_id}_predictions.csv"
    audit_path = fold_artifact_dir / f"{safe_fold_id}_audit.json"
    if config.resume and predictions_path.exists() and audit_path.exists():
        predictions = pd.read_csv(predictions_path, low_memory=False)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["resumed_from_artifact"] = True
        return predictions, audit
    predictions, audit = _run_one_fold(
        dataset,
        visual_context_dataset,
        bundle,
        config,
        fold_id,
        label_order,
        label_to_idx,
        device,
        fold_artifact_dir / f"{safe_fold_id}_work",
    )
    audit["resumed_from_artifact"] = False
    predictions.to_csv(predictions_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return predictions, audit


def _load_bundle(config: FullMultimodalOofConfig) -> _OofBundle:
    """Load train-ready rows and keep identity/source columns as metadata only."""

    arrays = {
        name: value for name, value in np.load(config.root / "X_spatial_sequences.npz").items()
    }
    missing_arrays = [
        name for name in [*MODEL_GROUPS, "length_mask", "observed_mask"] if name not in arrays
    ]
    if missing_arrays:
        raise ValueError(f"missing spatial arrays: {missing_arrays}")
    y = pd.read_csv(config.root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(config.root / "train_mask.csv")
    split = pd.read_csv(config.root / "split_manifest.csv", low_memory=False)
    image_windows = pd.read_csv(config.root / "image_window_context_manifest.csv", low_memory=False)
    sequence = pd.read_csv(
        config.sequence_manifest_csv,
        usecols=[
            "window_id",
            "temporal_unit_keys_window",
            "num_temporal_units_window",
            "window_valid_for_main_train",
            "window_sample_weight",
        ],
        low_memory=False,
    )
    folds = pd.read_csv(
        config.native_oof_fold_manifest_csv,
        usecols=["temporal_unit_key", "oof_fold_id", "native_unit_valid_for_main_eval"],
        low_memory=False,
    )
    event_weights = pd.read_csv(
        config.event_weight_manifest_csv,
        usecols=["window_id", "event_balanced_sample_weight", "window_valid_for_event_weight"],
        low_memory=False,
    )
    _validate_event_weight_alignment(split, event_weights)
    interaction = InteractionContextWindowDataset(
        InteractionContextDatasetConfig(manifest_csv=config.interaction_context_manifest_csv)
    ).manifest
    window_alignment = _require_ordered_window_ids(
        "split",
        split["window_id"],
        {
            "sequence": sequence["window_id"],
            "image_context": image_windows["window_id"],
            "interaction_context": interaction["window_id"],
        },
    )
    expected = int(len(y))
    row_counts = {
        "y": int(len(y)),
        "train_mask": int(len(train_mask)),
        "split": int(len(split)),
        "image_windows": int(len(image_windows)),
        "interaction": int(len(interaction)),
        "event_weights": int(len(event_weights)),
    }
    row_counts.update({name: int(arr.shape[0]) for name, arr in arrays.items()})
    mismatched = {name: count for name, count in row_counts.items() if count != expected}
    if mismatched:
        raise ValueError(f"row count mismatch against y={expected}: {mismatched}")

    required_split_metadata = {"window_id", "split", "split_group_key", "source_type"}
    missing_split_metadata = sorted(required_split_metadata.difference(split.columns))
    if missing_split_metadata:
        raise ValueError(f"split manifest missing evaluation metadata: {missing_split_metadata}")
    frame = split[["window_id", "split", "split_group_key", "source_type"]].copy()
    frame["behavior_true"] = y
    frame["train_mask"] = train_mask
    frame = frame.merge(
        sequence,
        on="window_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"temporal_unit_keys_window": "temporal_unit_key"})
    frame = frame.merge(folds, on="temporal_unit_key", how="left")
    frame = frame.merge(event_weights, on="window_id", how="left", validate="one_to_one")
    frame["window_image_context_complete"] = _to_bool(
        image_windows["window_image_context_complete"]
    )
    frame["window_valid_for_main_train"] = _to_bool(frame["window_valid_for_main_train"])
    frame["native_unit_valid_for_main_eval"] = _to_bool(frame["native_unit_valid_for_main_eval"])
    frame["window_valid_for_event_weight"] = _to_bool(frame["window_valid_for_event_weight"])
    frame["event_balanced_sample_weight"] = pd.to_numeric(
        frame["event_balanced_sample_weight"], errors="coerce"
    ).fillna(0.0)
    frame["num_temporal_units_window"] = pd.to_numeric(
        frame["num_temporal_units_window"], errors="coerce"
    )
    frame["eligible"] = (
        frame["train_mask"]
        & frame["window_valid_for_main_train"]
        & frame["native_unit_valid_for_main_eval"]
        & frame["window_image_context_complete"]
        & frame["num_temporal_units_window"].eq(1)
        & frame["behavior_true"].isin(VALID_BEHAVIORS)
        & frame["oof_fold_id"].fillna("").astype(str).ne("")
    )
    if config.sample_weight_policy in {"event", "event_class"}:
        invalid_event_training_rows = frame["eligible"] & (
            ~frame["window_valid_for_event_weight"]
            | ~np.isfinite(frame["event_balanced_sample_weight"])
            | frame["event_balanced_sample_weight"].le(0.0)
        )
        if invalid_event_training_rows.any():
            examples = (
                frame.loc[invalid_event_training_rows, "window_id"].astype(str).head(10).tolist()
            )
            raise ValueError(
                "event weighting is invalid for training-eligible rows: "
                f"count={int(invalid_event_training_rows.sum())}, examples={examples}"
            )
    interaction_features = (
        interaction[list(INTERACTION_CONTEXT_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    interaction_available = _to_bool(interaction["scene_partner_context_ready"]).to_numpy(
        dtype=np.float32
    )
    event_sample_weights = frame["event_balanced_sample_weight"].to_numpy(dtype=np.float32)
    load_audit = {
        "row_counts": row_counts,
        "eligible_rows": int(frame["eligible"].sum()),
        "eligible_fold_count": int(frame.loc[frame["eligible"], "oof_fold_id"].nunique()),
        "eligible_label_counts": frame.loc[frame["eligible"], "behavior_true"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "complete_image_context_rows": int(frame["window_image_context_complete"].sum()),
        "interaction_context_ready_rows": int(interaction_available.sum()),
        "valid_event_weight_rows": int(frame["window_valid_for_event_weight"].sum()),
        "zero_event_weight_rows": int(np.count_nonzero(event_sample_weights == 0.0)),
        "window_alignment": window_alignment,
    }
    return _OofBundle(
        arrays=arrays,
        interaction_context_features=interaction_features,
        interaction_context_available_mask=interaction_available,
        event_sample_weights=event_sample_weights,
        y=y,
        frame=frame,
        load_audit=load_audit,
    )


def _require_ordered_window_ids(
    reference_name: str,
    reference: pd.Series,
    candidates: dict[str, pd.Series],
) -> dict[str, Any]:
    """Prove every positional artifact uses one ordered window-key lineage."""

    reference_ids = _clean_window_ids(reference)
    errors = _window_key_errors(reference_ids, reference_name)
    comparisons: dict[str, dict[str, Any]] = {}
    reference_set = set(reference_ids)
    for name, values in candidates.items():
        candidate_ids = _clean_window_ids(values)
        candidate_errors = _window_key_errors(candidate_ids, name)
        missing = sorted(reference_set.difference(candidate_ids))
        extra = sorted(set(candidate_ids).difference(reference_set))
        order_mismatch = _ordered_mismatch_count(
            reference_ids,
            candidate_ids,
        )
        candidate_errors.extend(
            [
                *([f"missing_window_ids={len(missing)}"] if missing else []),
                *([f"extra_window_ids={len(extra)}"] if extra else []),
                *([f"window_order_mismatch_rows={order_mismatch}"] if order_mismatch else []),
            ]
        )
        comparisons[name] = {
            "rows": int(len(candidate_ids)),
            "ordered_window_id_sha256": _ordered_window_id_sha256(candidate_ids),
            "missing_count": int(len(missing)),
            "extra_count": int(len(extra)),
            "order_mismatch_rows": int(order_mismatch),
            "errors": candidate_errors,
        }
        errors.extend(f"{name}:{error}" for error in candidate_errors)

    audit = {
        "reference": reference_name,
        "reference_rows": int(len(reference_ids)),
        "reference_ordered_window_id_sha256": _ordered_window_id_sha256(reference_ids),
        "comparisons": comparisons,
        "errors": errors,
        "valid": not errors,
    }
    if errors:
        raise ValueError(f"ordered window alignment failed: {errors}")
    return audit


def _clean_window_ids(values: pd.Series) -> pd.Series:
    """Normalize keys without making missing values appear valid."""

    return values.fillna("").astype(str).str.strip().reset_index(drop=True)


def _window_key_errors(values: pd.Series, name: str) -> list[str]:
    """Return blank and duplicate violations for one positional artifact."""

    errors: list[str] = []
    blank = int(values.eq("").sum())
    duplicate = int(values.duplicated(keep=False).sum())
    if blank:
        errors.append(f"blank_{name}_window_ids={blank}")
    if duplicate:
        errors.append(f"duplicate_{name}_window_id_rows={duplicate}")
    return errors


def _ordered_mismatch_count(
    reference: pd.Series,
    candidate: pd.Series,
) -> int:
    """Count positional mismatches, including rows absent from either side."""

    size = max(len(reference), len(candidate))
    left = reference.reindex(range(size), fill_value="")
    right = candidate.reindex(range(size), fill_value="")
    return int(left.ne(right).sum())


def _ordered_window_id_sha256(values: pd.Series) -> str:
    """Hash ordered keys so run manifests can prove row alignment cheaply."""

    payload = "\n".join(values.astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _batch_from_indices(
    dataset: ClassificationV2ImageSequenceDataset,
    visual_context_dataset: VisualInteractionWindowDataset,
    bundle: _OofBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    class_weights: dict[str, float],
    config: FullMultimodalOofConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Build one multimodal batch from aligned global row indices."""

    settings = _ablation_settings(config.ablation_variant)
    if settings["enable_image"]:
        image_batch = image_sequence_collate([dataset[int(index)] for index in indices])
        image_errors = [err for item_errors in image_batch["errors"] for err in item_errors]
        if image_errors:
            raise ValueError(f"image load errors: {image_errors[:10]}")
        image = image_batch["image"].float().to(device)
        image_length_mask = image_batch["length_mask"].float().to(device)
        image_observed_mask = image_batch["observed_mask"].float().to(device)
    else:
        # Disabled image branches receive shape-valid placeholders and perform
        # no cache/source I/O; the model never consumes these tensors.
        batch_size = int(len(indices))
        image = torch.zeros((batch_size, 1, 3, 1, 1), dtype=torch.float32, device=device)
        image_length_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        image_observed_mask = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    if settings["enable_visual_context"]:
        visual_batch = visual_interaction_collate(
            [visual_context_dataset[int(index)] for index in indices]
        )
        visual_errors = [error for item_errors in visual_batch["errors"] for error in item_errors]
        if visual_errors:
            raise ValueError(f"visual context load errors: {visual_errors[:10]}")
        visual_context_image = visual_batch["visual_context_image"].float().to(device)
        visual_context_length_mask = visual_batch["visual_context_length_mask"].float().to(device)
        visual_context_observed_mask = (
            visual_batch["visual_context_observed_mask"].float().to(device)
        )
    else:
        batch_size = int(len(indices))
        visual_context_image = torch.zeros(
            (batch_size, 1, 3, 1, 1), dtype=torch.float32, device=device
        )
        visual_context_length_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        visual_context_observed_mask = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )
    target_labels = bundle.frame.iloc[indices]["behavior_true"].astype(str).tolist()
    training_sample_weight = _training_sample_weights(bundle, indices, class_weights, config)
    return {
        "image": image,
        "image_length_mask": image_length_mask,
        "image_observed_mask": image_observed_mask,
        "spatial_features": {
            name: torch.from_numpy(bundle.arrays[name][indices]).float().to(device)
            for name in settings["spatial_groups"]
        },
        "spatial_length_mask": torch.from_numpy(bundle.arrays["length_mask"][indices])
        .float()
        .to(device),
        "spatial_observed_mask": torch.from_numpy(bundle.arrays["observed_mask"][indices])
        .float()
        .to(device),
        "interaction_context_features": torch.from_numpy(
            bundle.interaction_context_features[indices]
        )
        .float()
        .to(device),
        "interaction_context_available_mask": torch.from_numpy(
            bundle.interaction_context_available_mask[indices]
        )
        .float()
        .to(device),
        "visual_context_image": visual_context_image,
        "visual_context_length_mask": visual_context_length_mask,
        "visual_context_observed_mask": visual_context_observed_mask,
        "target": torch.tensor(
            [label_to_idx[label] for label in target_labels], dtype=torch.long
        ).to(device),
        "training_sample_weight": torch.from_numpy(training_sample_weight).float().to(device),
    }


def _forward_model(model: MultimodalFusionClassifier, batch: dict[str, Any]) -> torch.Tensor:
    """Run the multimodal forward path with branch-specific masks."""

    return model(
        image=batch["image"],
        spatial_features=batch["spatial_features"],
        length_mask=batch["image_length_mask"],
        image_length_mask=batch["image_length_mask"],
        image_observed_mask=batch["image_observed_mask"],
        spatial_length_mask=batch["spatial_length_mask"],
        spatial_observed_mask=batch["spatial_observed_mask"],
        interaction_context_features=batch["interaction_context_features"],
        interaction_context_available_mask=batch["interaction_context_available_mask"],
        visual_context_image=batch["visual_context_image"],
        visual_context_length_mask=batch["visual_context_length_mask"],
        visual_context_observed_mask=batch["visual_context_observed_mask"],
    )


def _predict(
    model: MultimodalFusionClassifier,
    bundle: _OofBundle,
    indices: np.ndarray,
    batch: dict[str, Any],
    label_order: list[str],
    fold_id: str,
    experiment_role: str,
) -> pd.DataFrame:
    """Emit S16-compatible window predictions for held-out native units."""

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(_forward_model(model, batch), dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    rows = bundle.frame.iloc[indices].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "temporal_unit_key": rows["temporal_unit_key"].astype(str),
            "window_id": rows["window_id"].astype(str),
            "behavior_true": rows["behavior_true"].astype(str),
            "behavior_pred": [label_order[index] for index in pred_idx],
            "window_sample_weight": pd.to_numeric(
                rows["window_sample_weight"], errors="coerce"
            ).fillna(1.0),
            "window_valid_for_main_train": rows["window_valid_for_main_train"].astype(bool),
            "oof_fold_id": str(fold_id),
            "experiment_role": experiment_role,
        }
    )
    for class_index, label in enumerate(label_order):
        out[f"prob_{label}"] = probs[:, class_index]
    return out


def _predict_in_batches(
    dataset: ClassificationV2ImageSequenceDataset,
    visual_context_dataset: VisualInteractionWindowDataset,
    model: MultimodalFusionClassifier,
    bundle: _OofBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    label_order: list[str],
    fold_id: str,
    config: FullMultimodalOofConfig,
    device: torch.device,
    chunk_dir: Path | None = None,
    class_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict a held-out fold in resumable chunks so long folds survive timeouts."""

    chunks: list[pd.DataFrame] = []
    role_base = (
        "full_multimodal_oof" if _is_full_run(config, bundle) else "full_multimodal_oof_pilot"
    )
    role = f"{role_base}_{config.ablation_variant}"
    if chunk_dir is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(indices), int(config.eval_batch_size)):
        end = min(start + int(config.eval_batch_size), len(indices))
        chunk_path = (
            chunk_dir / f"chunk_{start:08d}_{end:08d}.csv" if chunk_dir is not None else None
        )
        if config.resume and chunk_path is not None and chunk_path.exists():
            chunks.append(pd.read_csv(chunk_path, low_memory=False))
            continue
        chunk_indices = indices[start : start + int(config.eval_batch_size)]
        batch = _batch_from_indices(
            dataset,
            visual_context_dataset,
            bundle,
            chunk_indices,
            label_to_idx,
            class_weights or {label: 1.0 for label in VALID_BEHAVIORS},
            config,
            device,
        )
        chunk_predictions = _predict(
            model, bundle, chunk_indices, batch, label_order, fold_id, role
        )
        if chunk_path is not None:
            chunk_predictions.to_csv(chunk_path, index=False)
        chunks.append(chunk_predictions)
    return pd.concat(chunks, ignore_index=True)


def _step_train_indices(
    train_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    step_index: int,
) -> np.ndarray:
    """Select one deterministic mini-batch, cycling all rows when batch covers the fold."""

    if len(train_indices) <= batch_size:
        return train_indices
    replace = len(train_indices) < batch_size
    if not replace:
        # Mix a deterministic offset with RNG sampling so short pilot runs still
        # see different rows across steps without relying on global state.
        shifted = np.roll(train_indices, step_index * batch_size)
        candidate_pool = shifted[: max(batch_size * 4, batch_size)]
        if len(candidate_pool) >= batch_size:
            return np.sort(rng.choice(candidate_pool, size=batch_size, replace=False))
    return np.sort(rng.choice(train_indices, size=batch_size, replace=replace))


def _sample_indices(
    frame: pd.DataFrame, *, mask: pd.Series, per_class: int | None, seed: int
) -> np.ndarray:
    """Return deterministic per-class sample indices or all rows when per_class is None."""

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in VALID_BEHAVIORS:
        label_indices = np.flatnonzero((mask & frame["behavior_true"].eq(label)).to_numpy())
        if per_class is None or len(label_indices) <= per_class:
            chosen = label_indices
        else:
            chosen = np.sort(rng.choice(label_indices, size=int(per_class), replace=False))
        selected.extend(int(index) for index in chosen)
    return np.array(sorted(selected), dtype=np.int64)


def _validate_config(config: FullMultimodalOofConfig) -> None:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.steps_per_fold <= 0:
        raise ValueError("steps_per_fold must be positive")
    if config.epochs_per_fold <= 0:
        raise ValueError("epochs_per_fold must be positive")
    if config.sample_weight_policy not in SAMPLE_WEIGHT_POLICIES:
        raise ValueError(f"unsupported sample_weight_policy={config.sample_weight_policy}")
    if config.class_weight_power < 0.0 or config.class_weight_max <= 0.0:
        raise ValueError("class_weight_power must be non-negative and class_weight_max positive")
    if config.precision not in PRECISION_POLICIES:
        raise ValueError(f"unsupported precision={config.precision}")
    if config.checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    if config.model_architecture_version != MODEL_ARCHITECTURE_VERSION:
        raise ValueError(
            "model_architecture_version does not match implemented model: "
            f"config={config.model_architecture_version}, implemented={MODEL_ARCHITECTURE_VERSION}"
        )
    if config.train_batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.max_folds is not None and config.max_folds <= 0:
        raise ValueError("max_folds must be positive when provided")
    if config.run_mode not in {"pilot", "full"}:
        raise ValueError("run_mode must be pilot or full")
    if config.ablation_variant not in ABLATION_VARIANTS:
        raise ValueError(f"unsupported ablation_variant={config.ablation_variant}")
    if config.run_mode == "pilot" and (
        config.max_folds is None
        or config.train_per_class_per_fold is None
        or config.eval_per_class_per_fold is None
    ):
        raise ValueError("pilot mode requires bounded folds and per-class sample caps")
    if config.require_cached_images and not (
        config.image_cache_manifest_csv is not None or config.packed_image_cache_npy is not None
    ):
        raise ValueError("require_cached_images needs an individual or packed image cache")
    if _ablation_settings(config.ablation_variant)["enable_visual_context"]:
        if not config.visual_context_cache_manifest_csv.exists():
            raise ValueError(
                "visual context branch requires an existing cache manifest: "
                f"{config.visual_context_cache_manifest_csv}"
            )
        if config.require_packed_visual_context and (
            config.visual_context_packed_cache_npy is None
            or config.visual_context_packed_cache_index_csv is None
        ):
            raise ValueError("require_packed_visual_context needs packed tensor and index paths")


def _validate_dataset_alignment(
    actor_dataset: ClassificationV2ImageSequenceDataset,
    visual_dataset: VisualInteractionWindowDataset,
    *,
    expected_window_ids: pd.Series | None = None,
) -> None:
    """Prove image branches match each other and the supervised row order."""

    actor_ids = actor_dataset.windows["window_id"].astype(str).reset_index(drop=True)
    visual_ids = visual_dataset.windows["window_id"].astype(str).reset_index(drop=True)
    if expected_window_ids is None:
        _require_ordered_window_ids(
            "actor_image",
            actor_ids,
            {"visual_context": visual_ids},
        )
        return
    _require_ordered_window_ids(
        "supervised_bundle",
        expected_window_ids,
        {
            "actor_image": actor_ids,
            "visual_context": visual_ids,
        },
    )


def _is_full_run(config: FullMultimodalOofConfig, bundle: _OofBundle) -> bool:
    """Return whether the config is allowed to be interpreted as full OOF evidence."""

    return (
        config.run_mode == "full"
        and config.max_folds is None
        and config.train_per_class_per_fold is None
        and config.eval_per_class_per_fold is None
        and config.epochs_per_fold >= 1
        and config.sample_weight_policy in {"event", "event_class"}
        and bundle.load_audit.get("eligible_fold_count", 0) >= 2
    )


def _mode_warnings(config: FullMultimodalOofConfig, bundle: _OofBundle) -> list[str]:
    warnings: list[str] = []
    if not _is_full_run(config, bundle):
        warnings.append(
            "bounded pilot run; do not register as full_multimodal_oof_record "
            "or cite as paper metric"
        )
    warnings.append(
        "full learned OOF claim also requires source-balanced reporting and ablation report review"
    )
    return warnings


def _fold_training_coverage_complete(fold_audits: list[dict[str, Any]]) -> bool:
    """Require every full fold to complete its declared epoch coverage."""

    return bool(fold_audits) and all(
        int(fold.get("training_steps_completed", -1))
        == int(fold.get("expected_training_steps", -2))
        and float(fold.get("train_row_coverage_ratio", 0.0)) >= 1.0
        for fold in fold_audits
    )


def _stable_fold_offset(fold_id: str) -> int:
    return sum(ord(char) for char in str(fold_id))


def _fold_training_signature(
    config: FullMultimodalOofConfig,
    fold_id: str,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    """Identify the exact trained fold so prediction chunks cannot mix configs."""

    return {
        "oof_fold_id": str(fold_id),
        "seed": int(config.seed),
        "image_size": int(config.image_size),
        "hidden_dim": int(config.hidden_dim),
        "steps_per_fold": int(config.steps_per_fold),
        "epochs_per_fold": int(config.epochs_per_fold),
        "effective_training_steps": int(_effective_training_step_count(config, len(train_indices))),
        "sample_weight_policy": config.sample_weight_policy,
        "class_weight_power": float(config.class_weight_power),
        "class_weight_max": float(config.class_weight_max),
        "precision": config.precision,
        "checkpoint_every_steps": int(config.checkpoint_every_steps),
        "model_architecture_version": config.model_architecture_version,
        "train_batch_size": int(config.train_batch_size),
        "eval_batch_size": int(config.eval_batch_size),
        "train_per_class_per_fold": config.train_per_class_per_fold,
        "eval_per_class_per_fold": config.eval_per_class_per_fold,
        "ablation_variant": config.ablation_variant,
        "train_indices_sha256": _indices_checksum(train_indices),
        "eval_indices_sha256": _indices_checksum(eval_indices),
    }


def _indices_checksum(indices: np.ndarray) -> str:
    """Hash ordered row indices used for training/evaluation artifact lineage."""

    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _training_batches(
    config: FullMultimodalOofConfig,
    train_indices: np.ndarray,
    rng: np.random.Generator,
):
    """Yield bounded pilot batches or complete deterministic epochs for full runs."""

    if config.run_mode == "full":
        for _ in range(int(config.epochs_per_fold)):
            shuffled = rng.permutation(train_indices)
            for start in range(0, len(shuffled), int(config.train_batch_size)):
                yield shuffled[start : start + int(config.train_batch_size)]
        return
    for step_index in range(int(config.steps_per_fold)):
        yield _step_train_indices(train_indices, config.train_batch_size, rng, step_index)


def _save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_state_path: Path,
    training_audit_path: Path,
    *,
    training_signature: dict[str, Any],
    losses: list[float],
    seen_train_indices: set[int],
    completed_training_steps: int,
    training_elapsed_sec: float,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
) -> None:
    """Atomically persist resumable optimizer state and deterministic progress metadata."""

    checkpoint_path = model_state_path.with_suffix(model_state_path.suffix + ".tmp")
    audit_path = training_audit_path.with_suffix(training_audit_path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "classification_v2_training_checkpoint_v2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "training_signature": training_signature,
            "losses": losses,
            "seen_train_indices": sorted(seen_train_indices),
            "completed_training_steps": int(completed_training_steps),
            "training_elapsed_sec": float(training_elapsed_sec),
            "cuda_peak_memory_allocated_mb": float(peak_allocated_mb),
            "cuda_peak_memory_reserved_mb": float(peak_reserved_mb),
        },
        checkpoint_path,
    )
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2_training_progress_v2",
                "training_signature": training_signature,
                "losses": losses,
                "seen_train_indices": sorted(seen_train_indices),
                "completed_training_steps": int(completed_training_steps),
                "training_elapsed_sec": float(training_elapsed_sec),
                "cuda_peak_memory_allocated_mb": float(peak_allocated_mb),
                "cuda_peak_memory_reserved_mb": float(peak_reserved_mb),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint_path.replace(model_state_path)
    audit_path.replace(training_audit_path)


def _fold_local_class_weights(
    bundle: _OofBundle,
    train_indices: np.ndarray,
    config: FullMultimodalOofConfig,
) -> dict[str, float]:
    """Balance effective event mass using labels from this fold's train partition only."""

    if config.sample_weight_policy != "event_class":
        return {label: 1.0 for label in VALID_BEHAVIORS}
    labels = bundle.frame.iloc[train_indices]["behavior_true"].astype(str).to_numpy()
    event_weights = bundle.event_sample_weights[train_indices].astype(np.float64, copy=False)
    class_mass = (
        pd.Series(event_weights)
        .groupby(pd.Series(labels), sort=False)
        .sum()
        .reindex(VALID_BEHAVIORS, fill_value=0.0)
    )
    positive_mass = class_mass[class_mass > 0.0]
    if positive_mass.empty:
        raise ValueError("cannot compute fold-local class weights without train labels")
    median_mass = float(positive_mass.median())
    weights: dict[str, float] = {}
    for label in VALID_BEHAVIORS:
        mass = float(class_mass[label])
        if mass <= 0.0:
            weights[label] = 0.0
            continue
        raw = (median_mass / mass) ** float(config.class_weight_power)
        weights[label] = float(min(float(config.class_weight_max), raw))
    return weights


def _validate_event_weight_alignment(split: pd.DataFrame, event_weights: pd.DataFrame) -> None:
    """Require a one-to-one window identity match before weights affect training loss."""

    split_ids = split["window_id"].fillna("").astype(str)
    weight_ids = event_weights["window_id"].fillna("").astype(str)
    duplicate_split = int(split_ids.duplicated(keep=False).sum())
    duplicate_weights = int(weight_ids.duplicated(keep=False).sum())
    missing = sorted(set(split_ids) - set(weight_ids))
    extra = sorted(set(weight_ids) - set(split_ids))
    if duplicate_split or duplicate_weights or missing or extra:
        raise ValueError(
            "event weight window_id alignment failed: "
            f"duplicate_split_rows={duplicate_split}, duplicate_weight_rows={duplicate_weights}, "
            f"missing_count={len(missing)}, extra_count={len(extra)}, "
            f"missing_examples={missing[:10]}, extra_examples={extra[:10]}"
        )


def _training_sample_weights(
    bundle: _OofBundle,
    indices: np.ndarray,
    class_weights: dict[str, float],
    config: FullMultimodalOofConfig,
) -> np.ndarray:
    """Compose review/window, event, and fold-local class weights without IDs in X."""

    if config.sample_weight_policy == "none":
        weights = np.ones(len(indices), dtype=np.float32)
    elif config.sample_weight_policy == "window":
        weights = (
            pd.to_numeric(bundle.frame.iloc[indices]["window_sample_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    else:
        weights = bundle.event_sample_weights[indices].astype(np.float32, copy=True)
    if config.sample_weight_policy == "event_class":
        labels = bundle.frame.iloc[indices]["behavior_true"].astype(str).tolist()
        weights *= np.asarray([class_weights.get(label, 0.0) for label in labels], dtype=np.float32)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("training sample weights must be finite and non-negative")
    return weights


def _effective_training_step_count(config: FullMultimodalOofConfig, train_rows: int) -> int:
    """Return optimizer steps implied by bounded pilot or full epoch coverage."""

    if config.run_mode == "full":
        return _ceil_div(train_rows, int(config.train_batch_size)) * int(config.epochs_per_fold)
    return int(config.steps_per_fold)


def _ablation_settings(variant: str) -> dict[str, Any]:
    """Map a predeclared variant to branches and exact spatial feature groups."""

    groups = list(MODEL_GROUPS)
    settings: dict[str, Any] = {
        "enable_image": True,
        "enable_spatial": True,
        "enable_interaction": True,
        "enable_visual_context": True,
        "spatial_groups": groups,
    }
    if variant == "image_only":
        settings.update(
            enable_spatial=False,
            enable_interaction=False,
            enable_visual_context=False,
            spatial_groups=[],
        )
    elif variant == "spatial_only":
        settings.update(enable_image=False, enable_interaction=False, enable_visual_context=False)
    elif variant == "no_interaction":
        settings["enable_interaction"] = False
    elif variant == "no_visual_context":
        settings["enable_visual_context"] = False
    elif variant == "no_roi":
        settings["spatial_groups"] = [name for name in groups if name != "roi_class_relation"]
    elif variant == "no_social":
        settings["spatial_groups"] = [name for name in groups if name != "social_relation"]
    elif variant == "no_motion":
        settings["spatial_groups"] = [name for name in groups if name != "motion_delta"]
    return settings


def _safe_fold_id(fold_id: str) -> str:
    """Make fold IDs safe for deterministic per-fold artifact filenames."""

    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(fold_id))


def _label_counts(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    return frame.iloc[indices]["behavior_true"].value_counts().sort_index().to_dict()


def _ceil_div(value: int, divisor: int) -> int:
    return int((int(value) + int(divisor) - 1) // int(divisor))


def _git_state() -> dict[str, Any]:
    """Bind learned artifacts to the exact source revision used for execution."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def _read_bool(path: Path) -> pd.Series:
    return _to_bool(pd.read_csv(path).iloc[:, 0])


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: FullMultimodalOofConfig) -> dict[str, Any]:
    out = asdict(config)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out
