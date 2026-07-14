from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    training_config_sha256,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    load_training_config,
)
from pig_behavior.classification_v2.training.run_identity import (
    RUN_IDENTITY_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    VISUAL_FREEZE_CONTRACT_VERSION,
    build_visual_optimizer_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check atomic classification_v2 checkpoint/resume contract."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/multimodal_context_multitask.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/checkpoint_contract"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    _seed_all(123)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    optimizer_groups, _ = build_visual_optimizer_groups(
        model,
        learning_rate=config.optimization.learning_rate,
        backbone_lr_multiplier=config.model.visual_backbone_lr_multiplier,
        weight_decay=config.optimization.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    x = torch.randn(5, 4)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint_path = args.output_dir / "checkpoint.pt"
    result_path = args.output_dir / "checkpoint_contract_audit.json"
    require_output_paths_available(
        [
            checkpoint_path,
            checkpoint_path.with_suffix(".pt.audit.json"),
            result_path,
        ],
        overwrite=args.overwrite,
    )
    run_identity = _synthetic_run_identity(config)
    save_audit = save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        run_identity=run_identity,
        preprocessing_sha256="fixture-preprocessing-sha256",
        train_window_id_sha256="fixture-train-window-id-sha256",
        epoch=2,
        global_step=17,
        metrics={"loss": float(loss.detach().item())},
    )
    expected_rng = _draw_rng()
    for parameter in model.parameters():
        parameter.data.zero_()
    resumed = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        run_identity=run_identity,
        preprocessing_sha256="fixture-preprocessing-sha256",
        train_window_id_sha256="fixture-train-window-id-sha256",
        restore_rng=True,
    )
    resumed_rng = _draw_rng()
    mismatch_rejected = False
    stale_config = replace(
        config,
        execution=replace(
            config.execution,
            fold_id="native_oof_stale",
        ),
    )
    try:
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=stale_config,
            run_identity=run_identity,
            preprocessing_sha256="fixture-preprocessing-sha256",
            train_window_id_sha256="fixture-train-window-id-sha256",
            restore_rng=False,
        )
    except ValueError:
        mismatch_rejected = True
    errors: list[str] = []
    if expected_rng != resumed_rng:
        errors.append("rng_continuation_mismatch")
    if resumed["epoch"] != 2 or resumed["global_step"] != 17:
        errors.append(f"resume_position_mismatch={resumed}")
    if not mismatch_rejected:
        errors.append("stale_lineage_not_rejected")
    if checkpoint_path.with_suffix(".pt.tmp").exists():
        errors.append("atomic_checkpoint_temp_file_leftover")
    result = {
        "schema_version": "classification_v2_checkpoint_contract_audit_v1",
        "checkpoint_path": str(checkpoint_path),
        "save_audit": save_audit,
        "rng_continuation_match": expected_rng == resumed_rng,
        "resumed_epoch": resumed["epoch"],
        "resumed_global_step": resumed["global_step"],
        "stale_lineage_rejected": mismatch_rejected,
        "errors": errors,
        "valid": not errors,
    }
    _write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _draw_rng() -> dict[str, float]:
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(1).item()),
    }


def _synthetic_run_identity(
    config: ClassificationV2TrainingConfig,
) -> dict[str, object]:
    """Build explicit fixture lineage without claiming a real dataset run."""

    modalities = [
        name
        for name, enabled in {
            "actor_rgb": config.model.enable_image,
            "spatial": config.model.enable_spatial,
            "interaction_numeric": config.model.enable_interaction_context,
            "partner_visual": config.model.enable_visual_context,
            "auxiliary_heads": config.model.enable_multitask,
        }.items()
        if enabled
    ]
    return {
        "identity_schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "run_id": "synthetic-checkpoint-contract",
        "experiment_name": "checkpoint-contract",
        "execution_profile": "local_smoke",
        "code_sha": "0" * 40,
        "dirty_worktree": True,
        "worktree_state_sha256": "0" * 64,
        "config_sha256": training_config_sha256(config),
        "dataset_snapshot_id": "synthetic-snapshot",
        "dataset_snapshot_sha256": "1" * 64,
        "cache_sha256": "2" * 64,
        "fold_manifest_sha256": "3" * 64,
        "feature_whitelist_sha256": "4" * 64,
        "temporal_view_selection_sha256": "5" * 64,
        "temporal_view_manifest_sha256": "7" * 64,
        "fold_event_weight_sha256": "6" * 64,
        "fold_id": config.execution.fold_id,
        "architecture_version": config.model.architecture_version,
        "model_mode": config.model.model_mode,
        "backbone_name": config.model.backbone_name,
        "pretrained_weight_enum": config.model.pretrained_weight_enum,
        "resolution": config.model.image_size,
        "visual_freeze_contract_version": VISUAL_FREEZE_CONTRACT_VERSION,
        "visual_freeze_policy": config.model.visual_freeze_policy,
        "visual_frozen_warmup_epochs": (
            config.model.visual_frozen_warmup_epochs
        ),
        "visual_layer4_only_epochs": config.model.visual_layer4_only_epochs,
        "visual_backbone_lr_multiplier": (
            config.model.visual_backbone_lr_multiplier
        ),
        "temporal_view": config.model.temporal_view,
        "temporal_encoder_name": config.model.temporal_encoder_name,
        "modalities": modalities,
        "loss_name": f"cross_entropy+{config.loss.sample_weight_policy}",
        "sampler_policy": config.loss.sampler_policy,
        "optimizer_name": config.optimization.optimizer,
        "precision": config.optimization.precision,
        "augmentation_policy": config.dataset.augmentation_policy,
    }


if __name__ == "__main__":
    main()
