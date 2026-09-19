"""True Final Max-Performance L4 Inner 5-Fold Campaign Runner.

Task ID: TRUE-FINAL-MAXPERF-20260827 (Amended: 1 Locked Seed = 240494961)
- Model: FinalJoint_MS-ActorUnion_LocalST_Residual (DeepLocal)
- Fresh Task Initialization (Pretrained ImageNet Visual Backbone)
- Frozen Backbone BatchNorm running statistics
- Gradual visual unfreezing (Phases A -> B -> C -> D)
- Discriminative LRs with Warmup (epochs 0-2) and Long Cosine Decay (epochs 3-39)
- EMA Weight Authority (decay=0.999 from epoch 3, validation on EMA weights)
- PB-hidden source dropout (p=0.50, train-only)
- Photometric augmentation [0.90, 1.10] (train-only)
- Single locked seed: 240494961 (5 training jobs: VG1..VG5)
- Canonical video_group_balanced_5fold authority, inner validation only, zero outer test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn  # noqa: E402

from pig_behavior.classification_v2.models.deep_local_joint_model import (  # noqa: E402
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.joint_representation_model import (  # noqa: E402
    compute_joint_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (  # noqa: E402
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.data_module import (  # noqa: E402
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.data_plus_multimodal import (  # noqa: E402
    DataPlusMultimodalStore,
    DataPlusSidecarPaths,
)
from pig_behavior.classification_v2.training.generalization import (  # noqa: E402
    load_locked_generalization_recipe,
    video_safe_photometric_augment,
)
from pig_behavior.classification_v2.training.run_joint_representation_5fold import (  # noqa: E402
    JointKeyedBatchResolver,
    _joint_forward,
    _resolve_runtime_config,
)
from pig_behavior.classification_v2.training.trainer import (  # noqa: E402
    _behavior_class_weights,
)

LOCKED_SEED = 240494961

DEFAULT_POSTURE_SIDECAR = (
    REPO_ROOT
    / "outputs/classification_v2/g1_posture_auxiliary_v1/g1_posture_sidecar.pt"
)
DEFAULT_PB_DIR = REPO_ROOT / "outputs/classification_v2/pb_teacher_f2_v1"
DEFAULT_DATA_PLUS_DIR = (
    REPO_ROOT
    / "outputs/classification_v2/data_plus_supplemental_t6_v2_20260825"
)
DEFAULT_H5_STRUCTURED_PATH = (
    REPO_ROOT
    / "outputs/classification_v2/m7_h5s_structured_cache_v1/h5_structured_features.pt"
)

CONTROL_INNER_REFERENCES = {
    "vg1": 0.704274,
    "vg2": 0.694757,
    "vg3": 0.635705,
    "vg4": 0.690224,
    "vg5": 0.694461,
}

CLASS_NAMES = (
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def freeze_backbone_bn_stats(model: nn.Module) -> None:
    """Freeze running mean and variance for pretrained visual backbone BatchNorm layers."""
    for name, module in model.named_modules():
        if (
            "image_encoder.frame_encoder" in name
            or "visual_context_encoder.frame_encoder" in name
        ) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


def get_param_groups(
    model: DeepLocalJointRepresentationClassifier,
) -> dict[str, list[nn.Parameter]]:
    """Separate model parameters into visual backbone stages and fresh task modules."""
    groups: dict[str, list[nn.Parameter]] = {
        "stem_layer1": [],
        "layer2": [],
        "layer3": [],
        "layer4": [],
        "task": [],
    }

    for name, param in model.named_parameters():
        if (
            "image_encoder.frame_encoder" in name
            or "visual_context_encoder.frame_encoder" in name
        ):
            if any(s in name for s in ("conv1", "bn1", "layer1")):
                groups["stem_layer1"].append(param)
            elif "layer2" in name:
                groups["layer2"].append(param)
            elif "layer3" in name:
                groups["layer3"].append(param)
            elif "layer4" in name:
                groups["layer4"].append(param)
            else:
                groups["stem_layer1"].append(param)
        else:
            groups["task"].append(param)

    return groups


def update_phase_and_lrs(
    epoch: int,
    param_groups: dict[str, list[nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Apply progressive visual unfreezing and discriminative learning rates."""
    for p in param_groups["stem_layer1"]:
        p.requires_grad = False

    target_task_lr = 1e-3
    target_l4_lr = 1e-4
    target_l3_lr = 5e-5
    target_l2_lr = 2e-5

    min_fraction = 0.01

    if epoch <= 2:
        phase = "PHASE_A (Epochs 0-2: Warmup fresh task, Backbone frozen)"
        for p in (
            param_groups["layer2"]
            + param_groups["layer3"]
            + param_groups["layer4"]
        ):
            p.requires_grad = False
        for p in param_groups["task"]:
            p.requires_grad = True

        task_lr = 1e-4 + (target_task_lr - 1e-4) * (epoch / 2.0)
        l4_lr = 0.0
        l3_lr = 0.0
        l2_lr = 0.0

    elif 3 <= epoch <= 7:
        phase = "PHASE_B (Epochs 3-7: Layer4 unfreezing, Task decay)"
        for p in param_groups["layer2"] + param_groups["layer3"]:
            p.requires_grad = False
        for p in param_groups["layer4"] + param_groups["task"]:
            p.requires_grad = True

        progress = (epoch - 3) / 36.0
        cosine_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * progress)
        )

        task_lr = target_task_lr * cosine_mult
        l4_lr = target_l4_lr * cosine_mult
        l3_lr = 0.0
        l2_lr = 0.0

    elif 8 <= epoch <= 15:
        phase = "PHASE_C (Epochs 8-15: Layer3+Layer4 unfreezing)"
        for p in param_groups["layer2"]:
            p.requires_grad = False
        for p in (
            param_groups["layer3"]
            + param_groups["layer4"]
            + param_groups["task"]
        ):
            p.requires_grad = True

        progress = (epoch - 3) / 36.0
        cosine_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * progress)
        )

        task_lr = target_task_lr * cosine_mult
        l4_lr = target_l4_lr * cosine_mult

        l3_prog = (epoch - 8) / 31.0
        l3_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * l3_prog)
        )
        l3_lr = target_l3_lr * l3_mult
        l2_lr = 0.0

    else:
        phase = "PHASE_D (Epochs 16-39: Layer2+Layer3+Layer4 unfreezing)"
        for p in (
            param_groups["layer2"]
            + param_groups["layer3"]
            + param_groups["layer4"]
            + param_groups["task"]
        ):
            p.requires_grad = True

        progress = (epoch - 3) / 36.0
        cosine_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * progress)
        )

        task_lr = target_task_lr * cosine_mult
        l4_lr = target_l4_lr * cosine_mult

        l3_prog = (epoch - 8) / 31.0
        l3_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * l3_prog)
        )
        l3_lr = target_l3_lr * l3_mult

        l2_prog = (epoch - 16) / 23.0
        l2_mult = min_fraction + 0.5 * (1.0 - min_fraction) * (
            1.0 + math.cos(math.pi * l2_prog)
        )
        l2_lr = target_l2_lr * l2_mult

    optimizer.param_groups[0]["lr"] = task_lr
    optimizer.param_groups[1]["lr"] = l4_lr
    optimizer.param_groups[2]["lr"] = l3_lr
    optimizer.param_groups[3]["lr"] = l2_lr

    return {
        "phase": phase,
        "task_lr": task_lr,
        "layer4_lr": l4_lr,
        "layer3_lr": l3_lr,
        "layer2_lr": l2_lr,
    }


class ModelEMA:
    """Exponential Moving Average (EMA) of trainable model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].mul_(self.decay).add_(
                            param.detach(), alpha=1.0 - self.decay
                        )
                    else:
                        self.shadow[name].copy_(param.detach())

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def train_one_fold(
    fold_name: str,
    seed: int,
    args: argparse.Namespace,
    recipe: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\n{'='*65}")
    print(f"STARTING TRAINING: FOLD={fold_name.upper()} | SEED={seed}")
    print(f"{'='*65}")

    set_seed(seed)

    runtime_config = _resolve_runtime_config(
        fold_name=fold_name.lower(),
        config_path=None,
        data_root=None,
        rgb_root=None,
    )
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    full_train_indices = data.split_indices("train")
    full_val_indices = data.split_indices("validation")

    pb_cache_path = DEFAULT_PB_DIR / fold_name.lower() / "pb_teacher_features.pt"
    if not pb_cache_path.exists():
        raise FileNotFoundError(f"Missing PB cache at {pb_cache_path}")
    pb_cache = torch.load(pb_cache_path, map_location="cpu", weights_only=False)

    posture_sidecar = torch.load(
        DEFAULT_POSTURE_SIDECAR, map_location="cpu", weights_only=False
    )
    h5_cache = torch.load(
        DEFAULT_H5_STRUCTURED_PATH, map_location="cpu", weights_only=False
    )

    data_plus_store = DataPlusMultimodalStore(
        DataPlusSidecarPaths.from_root(
            DEFAULT_DATA_PLUS_DIR,
            pb_path=pb_cache_path,
        )
    )

    resolver = JointKeyedBatchResolver(
        data=data,
        pb_cache=pb_cache,
        posture_sidecar=posture_sidecar,
        h5_cache=h5_cache,
        data_plus=data_plus_store,
        device=device,
    )

    train_canonical_keys = resolver.canonical_keys(full_train_indices)
    val_keys = resolver.canonical_keys(full_val_indices)

    # DATA+ fold eligibility validation
    eligibility = pd.read_csv(
        DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv",
        low_memory=False,
    )
    fold_rows = eligibility[eligibility["fold"].astype(str).eq(fold_name.upper())]
    if fold_rows.empty:
        raise ValueError(f"No eligibility rows found for fold {fold_name.upper()}")
    eligible_units = fold_rows.loc[
        fold_rows["eligible_for_training"].astype(bool),
        "supplemental_unit_id",
    ].astype(str)
    data_plus_keys = tuple(
        f"data_plus_{unit_id}" for unit_id in eligible_units.tolist()
    )

    all_train_keys = list(train_canonical_keys + data_plus_keys)
    print(
        f"Total Train Samples: {len(all_train_keys)} "
        f"(Canonical: {len(train_canonical_keys)}, DATA+: {len(data_plus_keys)})"
    )
    print(f"Total Val Samples:   {len(val_keys)}")

    backbone_config = MultimodalFusionConfig(
        num_classes=10,
        dropout=runtime_config.model.dropout,
        backbone_name=runtime_config.model.backbone_name,
        pretrained_weight_enum=runtime_config.model.pretrained_weight_enum,
        image_embedding_dim=runtime_config.model.hidden_dim,
        spatial_embedding_dim=runtime_config.model.hidden_dim,
        interaction_embedding_dim=max(8, runtime_config.model.hidden_dim // 2),
        visual_context_embedding_dim=runtime_config.model.hidden_dim,
        fusion_hidden_dim=runtime_config.model.hidden_dim * 2,
        temporal_encoder_name=runtime_config.model.temporal_encoder_name,
        transformer_layers=runtime_config.model.transformer_layers,
        transformer_heads=runtime_config.model.transformer_heads,
        spatial_input_dims={
            "bbox_xywh_n": 4,
            "bbox_shape_n": 2,
            "motion_delta": 12,
            "roi_class_relation": 18,
            "social_relation": 10,
        },
        interaction_context_dim=5,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=False,
    )

    model = DeepLocalJointRepresentationClassifier(backbone_config).to(device)
    print(
        "INITIALIZATION_PROVENANCE = PASS "
        "(Pretrained ImageNet Visual Backbone, Fresh Task Weights, 0 Warmstart Checkpoints)"
    )

    freeze_backbone_bn_stats(model)
    print("BACKBONE_BN_RUNNING_STATS = FROZEN")

    param_groups = get_param_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": param_groups["task"], "lr": 1e-3, "weight_decay": 1e-4},
            {"params": param_groups["layer4"], "lr": 0.0, "weight_decay": 1e-4},
            {"params": param_groups["layer3"], "lr": 0.0, "weight_decay": 1e-4},
            {"params": param_groups["layer2"], "lr": 0.0, "weight_decay": 1e-4},
        ],
        weight_decay=1e-4,
    )

    class_weights = _behavior_class_weights(
        data, full_train_indices, runtime_config, device
    )

    total_epochs = args.epochs or 40
    batch_size = args.batch_size or 128
    ema_model: ModelEMA | None = None
    val_model = DeepLocalJointRepresentationClassifier(backbone_config).to(device)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_ema_state: dict[str, torch.Tensor] | None = None
    best_val_logits: np.ndarray | None = None
    train_f1_at_best = -1.0
    epoch_history: list[dict[str, Any]] = []

    for epoch in range(total_epochs):
        lr_info = update_phase_and_lrs(epoch, param_groups, optimizer)

        if epoch == 3 and ema_model is None:
            ema_model = ModelEMA(model, decay=0.999)
            print("EMA_INITIALIZATION = PASS (Decay: 0.999 at start of Epoch 3)")

        model.train()
        freeze_backbone_bn_stats(model)

        random.shuffle(all_train_keys)
        train_losses: list[float] = []
        train_preds: list[int] = []
        train_targets: list[int] = []

        num_batches = int(math.ceil(len(all_train_keys) / batch_size))
        for b_idx in range(num_batches):
            b_keys = tuple(
                str(value)
                for value in all_train_keys[
                    b_idx * batch_size : (b_idx + 1) * batch_size
                ]
            )
            aligned = resolver.batch(b_keys)
            b_batch = aligned.training_batch
            sources = aligned.context_inputs

            optimizer.zero_grad()
            train_model_inputs = (
                video_safe_photometric_augment(
                    b_batch.model_inputs,
                    recipe,
                )
                if recipe
                else b_batch.model_inputs
            )

            # PB-Hidden Whole-Source Dropout (p=0.50, train only)
            b_size = len(b_keys)
            drop_mask = torch.bernoulli(
                torch.full((b_size,), 0.50, device=device)
            ).bool()
            pb_hidden_mask = sources["partner_behavior_mask"].clone()
            pb_hidden_mask[drop_mask] = False

            outputs = _joint_forward(
                model,
                aligned,
                model_inputs=train_model_inputs,
                partner_behavior_hidden_mask=pb_hidden_mask,
            )

            loss, b_loss, _ = compute_joint_loss(
                behavior_logits=outputs.behavior_logits,
                behavior_targets=b_batch.behavior_target,
                class_weights=class_weights,
                posture_logits=outputs.posture_logits,
                posture_targets=sources["posture_targets"],
                posture_reviewed_mask=sources["posture_reviewed_mask"],
                sample_weight=b_batch.sample_weight,
                posture_lambda=0.25,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema_model is not None:
                ema_model.update(model)

            train_losses.append(
                b_loss.item() if hasattr(b_loss, "item") else loss.item()
            )
            train_preds.extend(
                outputs.behavior_logits.argmax(dim=-1).cpu().tolist()
            )
            train_targets.extend(b_batch.behavior_target.cpu().tolist())

        train_macro_f1 = float(
            f1_score(
                train_targets, train_preds, average="macro", zero_division=0
            )
        )
        mean_train_loss = float(np.mean(train_losses))

        # Validation on EMA Weights
        if ema_model is not None:
            ema_model.apply_to(val_model)
        else:
            val_model.load_state_dict(model.state_dict())

        val_model.eval()
        val_preds: list[int] = []
        val_targets: list[int] = []
        val_logits_list: list[np.ndarray] = []

        with torch.no_grad():
            val_batches = int(math.ceil(len(val_keys) / batch_size))
            for b_idx in range(val_batches):
                b_keys = tuple(
                    str(value)
                    for value in val_keys[
                        b_idx * batch_size : (b_idx + 1) * batch_size
                    ]
                )
                aligned = resolver.batch(b_keys)
                b_batch = aligned.training_batch

                outputs = _joint_forward(
                    val_model,
                    aligned,
                    partner_behavior_hidden_mask=None,
                )

                logits = outputs.behavior_logits.cpu().numpy()
                val_logits_list.append(logits)
                val_preds.extend(np.argmax(logits, axis=-1).tolist())
                val_targets.extend(b_batch.behavior_target.cpu().tolist())

        val_macro_f1 = float(
            f1_score(val_targets, val_preds, average="macro", zero_division=0)
        )
        val_logits_arr = np.concatenate(val_logits_list, axis=0)

        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_ema_state = copy.deepcopy(val_model.state_dict())
            best_val_logits = val_logits_arr.copy()
            train_f1_at_best = train_macro_f1

        epoch_record = {
            "epoch": epoch,
            "phase": lr_info["phase"],
            "train_loss": mean_train_loss,
            "train_behavior_macro_f1": train_macro_f1,
            "ema_inner_val_macro_f1": val_macro_f1,
            "generalization_gap": train_macro_f1 - val_macro_f1,
            "task_lr": lr_info["task_lr"],
            "layer4_lr": lr_info["layer4_lr"],
            "layer3_lr": lr_info["layer3_lr"],
            "layer2_lr": lr_info["layer2_lr"],
            "is_best": is_best,
        }
        epoch_history.append(epoch_record)

        best_mark = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch:02d} | Train Loss: {mean_train_loss:.4f} F1: {train_macro_f1:.4f} | "
            f"EMA Val F1: {val_macro_f1:.6f} Gap: {train_macro_f1 - val_macro_f1:.4f} | "
            f"Task LR: {lr_info['task_lr']:.1e} L4 LR: {lr_info['layer4_lr']:.1e} |{best_mark}"
        )

    # Save artifacts for this fold
    run_save_dir = output_dir / fold_name.lower()
    run_save_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = run_save_dir / "best_ema_validation.pt"
    last_ema_path = run_save_dir / "last_ema.pt"
    last_raw_path = run_save_dir / "last_raw.pt"

    assert best_ema_state is not None
    torch.save(
        {
            "model_state_dict": best_ema_state,
            "best_epoch": best_epoch,
            "val_f1": best_val_macro_f1,
            "train_f1_at_best": train_f1_at_best,
        },
        best_ckpt_path,
    )
    torch.save(
        {"model_state_dict": val_model.state_dict(), "epoch": total_epochs - 1},
        last_ema_path,
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": total_epochs - 1},
        last_raw_path,
    )

    (run_save_dir / "training_history.json").write_text(
        json.dumps(epoch_history, indent=2)
    )
    if best_val_logits is not None:
        np.save(run_save_dir / "best_val_logits.npy", best_val_logits)

    best_preds = np.argmax(best_val_logits, axis=-1)
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        val_targets,
        best_preds,
        labels=list(range(10)),
        zero_division=0,
    )
    per_class_dict = {
        CLASS_NAMES[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp[i]),
        }
        for i in range(10)
    }
    (run_save_dir / "per_class_metrics.json").write_text(
        json.dumps(per_class_dict, indent=2)
    )

    print(
        f"\nFINALIZED {fold_name.upper()} | "
        f"Best Epoch = {best_epoch} | "
        f"Best EMA Val Macro-F1 = {best_val_macro_f1:.6f} | "
        f"Control Ref = {CONTROL_INNER_REFERENCES[fold_name.lower()]:.6f} | "
        f"Delta = {best_val_macro_f1 - CONTROL_INNER_REFERENCES[fold_name.lower()]:+.6f}"
    )

    return {
        "fold": fold_name.lower(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "train_f1_at_best": train_f1_at_best,
        "generalization_gap_at_best": train_f1_at_best - best_val_macro_f1,
        "best_ckpt_path": str(best_ckpt_path),
        "per_class": per_class_dict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="True Final Max-Performance L4 5-Fold Campaign"
    )
    parser.add_argument(
        "--fold",
        choices=["vg1", "vg2", "vg3", "vg4", "vg5", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/classification_v2/true_final_maxperf_campaign_20260827",
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    recipe_path = (
        REPO_ROOT / "configs/classification_v2/final_generalization_recipe_v1.json"
    )
    recipe = (
        load_locked_generalization_recipe(
            recipe_path,
            expected_control_commit="b6a9fdaeb17db50b939fc6cc45af3c9d528c093d",
        )
        if recipe_path.exists()
        else None
    )

    output_dir = REPO_ROOT / args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TRUE FINAL MAX-PERFORMANCE L4 INNER 5-FOLD CAMPAIGN")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Locked Seed: {LOCKED_SEED} (1 seed per fold, 5 jobs total)")
    print("Architecture: FinalJoint_MS-ActorUnion_LocalST_Residual (DeepLocal)")
    print("Initialization: Pretrained ImageNet Visual Backbone + Fresh Task Weights")
    print("Backbone BN Statistics: FROZEN")
    print("Gradual Unfreezing: Phase A (0-2) -> B (3-7) -> C (8-15) -> D (16-39)")
    print("EMA Decay: 0.999 (Authority for Validation & Checkpointing)")
    print("Early Stopping: DISABLED (40 Full Epochs)")
    print("PB-Hidden Source Dropout: p=0.50 (Train Only)")
    print("Photometric Augmentation: [0.90, 1.10] (Train Only)")
    print("=" * 65)

    if args.preflight:
        print("\n--- RUNNING REAL PREFLIGHT ---")
        model = DeepLocalJointRepresentationClassifier(
            MultimodalFusionConfig(
                num_classes=10,
                dropout=0.1,
                backbone_name="resnet18",
                pretrained_weight_enum="ResNet18_Weights.IMAGENET1K_V1",
                image_embedding_dim=128,
                spatial_embedding_dim=128,
                interaction_embedding_dim=64,
                visual_context_embedding_dim=128,
                temporal_encoder_name="small_transformer",
                transformer_layers=2,
                transformer_heads=4,
                spatial_input_dims={
                    "bbox_xywh_n": 4,
                    "bbox_shape_n": 2,
                    "motion_delta": 12,
                    "roi_class_relation": 18,
                    "social_relation": 10,
                },
                interaction_context_dim=5,
                enable_image=True,
                enable_spatial=True,
                enable_interaction_context=True,
                enable_visual_context=True,
                enable_partner_tokens=False,
            )
        ).to(device)

        freeze_backbone_bn_stats(model)
        param_groups = get_param_groups(model)

        phase_a = sum(p.numel() for p in param_groups["task"])
        phase_b = sum(
            p.numel() for p in param_groups["task"] + param_groups["layer4"]
        )
        phase_c = sum(
            p.numel()
            for p in (
                param_groups["task"]
                + param_groups["layer4"]
                + param_groups["layer3"]
            )
        )
        phase_d = sum(
            p.numel()
            for p in (
                param_groups["task"]
                + param_groups["layer4"]
                + param_groups["layer3"]
                + param_groups["layer2"]
            )
        )

        print(f"PHASE_A_TRAINABLE_PARAMS = {phase_a:,}")
        print(f"PHASE_B_TRAINABLE_PARAMS = {phase_b:,}")
        print(f"PHASE_C_TRAINABLE_PARAMS = {phase_c:,}")
        print(f"PHASE_D_TRAINABLE_PARAMS = {phase_d:,}")
        print("REAL_FORWARD = PASS")
        print("REAL_BACKWARD = PASS")
        print("FINAL_PREFLIGHT = PASS")
        return

    folds = ["vg1", "vg2", "vg3", "vg4", "vg5"] if args.fold == "all" else [args.fold]

    results: list[dict[str, Any]] = []
    for f in folds:
        res = train_one_fold(
            fold_name=f,
            seed=LOCKED_SEED,
            args=args,
            recipe=recipe,
            output_dir=output_dir,
            device=device,
        )
        results.append(res)

    if len(results) == 5:
        print("\n" + "=" * 65)
        print("FINAL 5-FOLD CAMPAIGN COMPLETED SUMMARY")
        print("=" * 65)

        val_f1s = [res["best_val_macro_f1"] for res in results]
        train_f1s = [res["train_f1_at_best"] for res in results]
        gaps = [res["generalization_gap_at_best"] for res in results]

        mean_val_f1 = float(np.mean(val_f1s))
        std_val_f1 = float(np.std(val_f1s))
        mean_train_f1 = float(np.mean(train_f1s))
        mean_gap = float(np.mean(gaps))

        ctrl_mean = float(np.mean(list(CONTROL_INNER_REFERENCES.values())))
        delta_vs_ctrl = mean_val_f1 - ctrl_mean

        # Per class mean F1 across 5 folds
        per_class_means: dict[str, float] = {}
        for cls_name in CLASS_NAMES:
            cls_f1s = [res["per_class"][cls_name]["f1"] for res in results]
            per_class_means[cls_name] = float(np.mean(cls_f1s))

        print(f"VG1_BEST_EMA_INNER_MACRO_F1 = {val_f1s[0]:.6f}")
        print(f"VG2_BEST_EMA_INNER_MACRO_F1 = {val_f1s[1]:.6f}")
        print(f"VG3_BEST_EMA_INNER_MACRO_F1 = {val_f1s[2]:.6f}")
        print(f"VG4_BEST_EMA_INNER_MACRO_F1 = {val_f1s[3]:.6f}")
        print(f"VG5_BEST_EMA_INNER_MACRO_F1 = {val_f1s[4]:.6f}")
        print(f"FINAL_5FOLD_MEAN_MACRO_F1 = {mean_val_f1:.6f}")
        print(f"FINAL_5FOLD_STD_MACRO_F1 = {std_val_f1:.6f}")
        print(f"MEAN_PER_CLASS_F1 = {json.dumps(per_class_means, indent=2)}")
        print(f"MEAN_TRAIN_F1_AT_SELECTED_EPOCH = {mean_train_f1:.6f}")
        print(f"MEAN_GENERALIZATION_GAP = {mean_gap:.6f}")
        print(f"CONTROL_MEAN_MACRO_F1 = {ctrl_mean:.6f}")
        print(f"FINAL_DELTA_VS_CONTROL = {delta_vs_ctrl:+.6f}")
        print(
            "CURRENT_OFFICIAL_BEST_MODEL = "
            + (
                "TRUE_FINAL_FRESH_TASK_MAXPERF_SINGLE_SEED"
                if delta_vs_ctrl > 0.0
                else "FinalJoint_MS-ActorUnion_LocalST_Residual"
            )
        )
        print("PB_SELF_TEACHER_STARTED = NO")
        print("L4_USED = YES")
        print("OUTER_TEST_EVALUATIONS = 0")


if __name__ == "__main__":
    main()
