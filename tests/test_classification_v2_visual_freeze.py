from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from pig_behavior.classification_v2.models.multimodal_fusion import (
    ActorEncoder,
    ImageSequenceEncoderConfig,
    UnionCropEncoder,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
)
from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    training_config_sha256,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    DatasetConfig,
    ExecutionConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
)
from pig_behavior.classification_v2.training.run_identity import (
    RUN_IDENTITY_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    VISUAL_FREEZE_CONTRACT_VERSION,
    build_visual_optimizer_groups,
    configure_visual_train_stage,
    optimizer_group_report,
    visual_freeze_schedule_errors,
    visual_freeze_stage_for_epoch,
)


class TinyResNetFrame(nn.Module):
    """Small ResNet-shaped fixture with BatchNorm outside and inside layer4."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1),
            nn.BatchNorm2d(4),
        )


class TinyVisualModel(nn.Module):
    """Exercise the real sequence-encoder discovery contract without downloads."""

    def __init__(self, *, include_union: bool = False) -> None:
        super().__init__()
        encoder_config = ImageSequenceEncoderConfig(
            backbone_name="smoke_cnn",
            pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
            embedding_dim=4,
            temporal_encoder_name="masked_mean",
        )
        self.actor = ActorEncoder(encoder_config)
        self.actor.frame_encoder = TinyResNetFrame()
        self.union: UnionCropEncoder | None = None
        if include_union:
            self.union = UnionCropEncoder(encoder_config)
            self.union.frame_encoder = TinyResNetFrame()
        self.final_head = nn.Linear(4, 10)


def test_staged_schedule_resolves_all_three_epochs() -> None:
    config = _model_config()

    assert visual_freeze_stage_for_epoch(config, 0) == "frozen"
    assert visual_freeze_stage_for_epoch(config, 1) == "layer4_only"
    assert visual_freeze_stage_for_epoch(config, 2) == "full"
    assert visual_freeze_schedule_errors(config, total_epochs=3) == []


def test_invalid_schedule_that_never_reaches_full_fails_closed() -> None:
    config = replace(_model_config(), visual_layer4_only_epochs=2)

    errors = visual_freeze_schedule_errors(config, total_epochs=3)

    assert "declared_epochs_never_reach_full_stage" in errors


def test_actor_and_union_follow_same_stage_and_batchnorm_policy() -> None:
    model = TinyVisualModel(include_union=True)
    config = _model_config()

    model.train()
    frozen = configure_visual_train_stage(model, config, epoch=0)
    assert frozen["visual_encoder_count"] == 2
    assert frozen["visual_trainable_parameter_count"] == 0
    assert all(not row["training_batch_norm_names"] for row in frozen["encoders"])

    model.train()
    layer4 = configure_visual_train_stage(model, config, epoch=1)
    assert 0 < layer4["visual_trainable_parameter_count"]
    assert layer4["visual_trainable_parameter_count"] < layer4["visual_parameter_count"]
    for row in layer4["encoders"]:
        assert row["training_batch_norm_names"] == ["layer4.1"]

    model.train()
    full = configure_visual_train_stage(model, config, epoch=2)
    assert full["visual_trainable_parameter_count"] == full["visual_parameter_count"]


def test_optimizer_groups_cover_frozen_backbone_and_heads_once() -> None:
    model = TinyVisualModel()
    config = _model_config()
    model.train()
    configure_visual_train_stage(model, config, epoch=0)

    groups, declared = build_visual_optimizer_groups(
        model,
        learning_rate=1e-3,
        backbone_lr_multiplier=config.visual_backbone_lr_multiplier,
        weight_decay=1e-4,
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-4)
    observed = optimizer_group_report(optimizer)

    assert declared["all_parameters_covered_once"] is True
    assert observed["group_order"] == ["visual_backbone", "nonvisual"]
    assert observed["groups"][0]["learning_rate"] == pytest.approx(1e-4)
    assert observed["groups"][1]["learning_rate"] == pytest.approx(1e-3)
    assert sum(row["parameter_count"] for row in observed["groups"]) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_schedule_rejects_an_accidentally_frozen_nonvisual_head() -> None:
    model = TinyVisualModel()
    config = _model_config()
    model.final_head.weight.requires_grad_(False)
    model.train()

    with pytest.raises(ValueError, match="keep every nonvisual head trainable"):
        configure_visual_train_stage(model, config, epoch=0)


def test_checkpoint_resume_crosses_frozen_to_layer4_boundary(tmp_path: Path) -> None:
    config = _training_config(tmp_path)
    model = TinyVisualModel()
    model.train()
    frozen = configure_visual_train_stage(model, config.model, epoch=0)
    optimizer = _optimizer(model, config)
    checkpoint = tmp_path / "stage-boundary.pt"
    identity = _run_identity(config)

    saved = save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        run_identity=identity,
        preprocessing_sha256="preprocessing-fixture",
        train_window_id_sha256="train-window-fixture",
        epoch=0,
        global_step=0,
        metrics={},
    )

    resumed_model = TinyVisualModel()
    resumed_optimizer = _optimizer(resumed_model, config)
    resumed = load_training_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=None,
        config=config,
        run_identity=identity,
        preprocessing_sha256="preprocessing-fixture",
        train_window_id_sha256="train-window-fixture",
        restore_rng=False,
    )
    resumed_model.train()
    layer4 = configure_visual_train_stage(resumed_model, config.model, epoch=1)

    assert saved["visual_freeze_state"]["stage"] == "frozen"
    assert resumed["visual_freeze_state"]["stage"] == "frozen"
    assert saved["validation_selection_policy"]["primary_metric"] == (
        "validation_native_unit_macro_f1_supported"
    )
    assert resumed["validation_selection_policy"] == (
        saved["validation_selection_policy"]
    )
    assert frozen["visual_encoder_names"] == layer4["visual_encoder_names"]
    assert layer4["stage"] == "layer4_only"
    assert optimizer_group_report(resumed_optimizer)["group_order"] == [
        "visual_backbone",
        "nonvisual",
    ]

    wrong_model = TinyVisualModel()
    wrong_optimizer = _optimizer(wrong_model, config)
    wrong_optimizer.param_groups[0]["group_name"] = "wrong_visual_group"
    with pytest.raises(ValueError, match="optimizer-group contract mismatch before load"):
        load_training_checkpoint(
            checkpoint,
            model=wrong_model,
            optimizer=wrong_optimizer,
            scaler=None,
            config=config,
            run_identity=identity,
            preprocessing_sha256="preprocessing-fixture",
            train_window_id_sha256="train-window-fixture",
            restore_rng=False,
        )


def _model_config() -> ModelConfig:
    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode="actor_temporal",
        backbone_name="resnet18",
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        temporal_encoder_name="masked_tcn",
        hidden_dim=8,
        visual_freeze_policy="frozen_then_layer4_then_full",
        visual_frozen_warmup_epochs=1,
        visual_layer4_only_epochs=1,
        visual_backbone_lr_multiplier=0.1,
        spatial_feature_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    )


def _training_config(tmp_path: Path) -> ClassificationV2TrainingConfig:
    dataset = DatasetConfig(
        snapshot_json=tmp_path / "snapshot.json",
        trainer_contract_json=tmp_path / "trainer.json",
        train_ready_root=tmp_path,
        actor_packed_cache=tmp_path / "actor.npy",
        actor_packed_index=tmp_path / "actor.csv",
        visual_cache_manifest=tmp_path / "visual.json",
        visual_packed_cache=tmp_path / "visual.npy",
        visual_packed_index=tmp_path / "visual.csv",
        native_oof_fold_manifest=tmp_path / "native.csv",
        grouped_fold_roles=tmp_path / "roles.csv",
        temporal_view_selection_manifest=tmp_path / "selection.csv",
        temporal_view_manifest=tmp_path / "temporal.csv",
        auxiliary_targets_csv=tmp_path / "auxiliary.csv",
    )
    return ClassificationV2TrainingConfig(
        version="classification_v2_training_config_v1",
        dataset=dataset,
        model=_model_config(),
        optimization=OptimizationConfig(
            epochs=3,
            precision="fp32",
        ),
        loss=LossConfig(sample_weight_policy="uniform"),
        execution=ExecutionConfig(),
    )


def _optimizer(
    model: nn.Module,
    config: ClassificationV2TrainingConfig,
) -> torch.optim.AdamW:
    groups, _ = build_visual_optimizer_groups(
        model,
        learning_rate=config.optimization.learning_rate,
        backbone_lr_multiplier=config.model.visual_backbone_lr_multiplier,
        weight_decay=config.optimization.weight_decay,
    )
    return torch.optim.AdamW(
        groups,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )


def _run_identity(
    config: ClassificationV2TrainingConfig,
) -> dict[str, object]:
    return {
        "identity_schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "run_id": "visual-freeze-resume-test",
        "experiment_name": "visual-freeze-resume-test",
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
        "temporal_view_manifest_sha256": "6" * 64,
        "fold_event_weight_sha256": "7" * 64,
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
        "early_stopping_contract_version": (
            config.optimization.early_stopping_contract_version
        ),
        "early_stopping_metric": config.optimization.early_stopping_metric,
        "early_stopping_tiebreaker": (
            config.optimization.early_stopping_tiebreaker
        ),
        "early_stopping_tie_tolerance": (
            config.optimization.early_stopping_tie_tolerance
        ),
        "early_stopping_min_supported_classes": (
            config.optimization.early_stopping_min_supported_classes
        ),
        "temporal_view": config.model.temporal_view,
        "temporal_encoder_name": config.model.temporal_encoder_name,
        "modalities": ["actor_rgb"],
        "loss_name": f"cross_entropy+{config.loss.sample_weight_policy}",
        "sampler_policy": config.loss.sampler_policy,
        "optimizer_name": config.optimization.optimizer,
        "precision": config.optimization.precision,
        "augmentation_policy": config.dataset.augmentation_policy,
    }
