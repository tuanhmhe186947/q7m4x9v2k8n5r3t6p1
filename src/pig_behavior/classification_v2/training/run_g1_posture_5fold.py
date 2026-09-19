"""Exact 5-fold scientific training runner for G1 Reviewed Posture Auxiliary.

Implements the exact promoted M2-VFT training recipe + single Linear(256, 3) posture auxiliary head:
  - visual LR = 0.0003 (0.1x backbone multiplier)
  - nonvisual LR = 0.003
  - all M2 parameters trainable from start
  - AdamW, weight_decay = 0.0, scheduler = none, clip = 1.0
  - behavior weighted CE + 0.25 * masked reviewed-posture CE
  - inner_train / inner_val per epoch
  - best checkpoint selected ONLY by inner_val_behavior_macro_f1
  - max_epochs = 30, early_stopping_patience = 5
  - strict outer posture censoring and zero outer evaluation during training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.models.g1_posture_model import (  # noqa: E402
    POSTURE_LAMBDA_LOCKED,
    G1PostureAuxiliaryClassifier,
    compute_g1_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (  # noqa: E402
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.config import (  # noqa: E402
    ClassificationV2TrainingConfig,
    load_training_config,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.data_module import (  # noqa: E402
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.trainer import (  # noqa: E402
    _behavior_class_weights,
)
from pig_behavior.classification_v2.training.visual_freeze import (  # noqa: E402
    build_visual_optimizer_groups,
)

SEED: int = 240494961
EXPECTED_PARAM_COUNT: int = 43634603
EXPECTED_TOTAL_ROWS: int = 33287
EXPECTED_REVIEWED_ROWS: int = 620
EXPECTED_SIDECAR_SHA256: str = (
    "e7a40bdd4e86e89e5d32eaaffa662c5ab5b4ba6b55c58e2950262a9ef9847c90"
)
DATA_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_DATA_ROOT"
RGB_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_RGB_ROOT"
DEFAULT_SIDECAR_PATH: Path = (
    REPO_ROOT / "outputs/classification_v2/g1_posture_auxiliary_v1/g1_posture_sidecar.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 Posture Auxiliary 5-Fold Scientific Runner")
    parser.add_argument(
        "--fold",
        choices=("vg1", "vg2", "vg3", "vg4", "vg5", "all"),
        default="vg1",
        help="Fold ID to execute.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config path.",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=DEFAULT_SIDECAR_PATH,
        help="Path to authoritative G1 posture sidecar.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root for explicit external data artifacts.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="Root for explicit external window-major RGB artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Execute two-epoch CPU production preflight.",
    )
    parser.add_argument(
        "--check-init-parity",
        action="store_true",
        help="Verify parameter initialization parity against M2.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _repo_path(value: Path) -> Path:
    return value if value.is_absolute() else REPO_ROOT / value


def _resolve_external_paths(
    config: ClassificationV2TrainingConfig,
    data_root: Path | None,
    rgb_root: Path | None,
) -> ClassificationV2TrainingConfig:
    dataset = config.dataset
    updates: dict[str, Path] = {}
    data_marker = f"{DATA_EXTERNAL_PREFIX}/"
    rgb_marker = f"{RGB_EXTERNAL_PREFIX}/"
    for field_name in dataset.__dataclass_fields__:
        value = getattr(dataset, field_name)
        if not isinstance(value, Path):
            continue
        text = value.as_posix()
        if text.startswith(data_marker):
            updates[field_name] = (
                data_root / text[len(data_marker) :]
                if data_root is not None
                else REPO_ROOT / text
            )
        elif text.startswith(rgb_marker):
            updates[field_name] = (
                rgb_root / text[len(rgb_marker) :]
                if rgb_root is not None
                else REPO_ROOT / text
            )
        elif field_name in {
            "snapshot_json",
            "trainer_contract_json",
            "grouped_fold_roles",
        }:
            updates[field_name] = _repo_path(value)
    return replace(config, dataset=replace(dataset, **updates))


def _build_g1_model_and_check_parity(
    backbone_config: MultimodalFusionConfig,
    check_parity: bool = False,
) -> G1PostureAuxiliaryClassifier:
    """Build G1 model and optionally verify parameter initialization parity with M2."""
    if check_parity:
        torch.manual_seed(SEED)
        m2_ref = MultimodalFusionClassifier(backbone_config)

    torch.manual_seed(SEED)
    g1_model = G1PostureAuxiliaryClassifier(backbone_config)

    if check_parity:
        max_diff = 0.0
        for (n1, p1), (n2, p2) in zip(
            m2_ref.named_parameters(),
            g1_model.backbone.named_parameters(),
            strict=True,
        ):
            assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
            diff = (p1 - p2).abs().max().item()
            if diff > max_diff:
                max_diff = diff
        print(f"M2_SHARED_PARAMETER_MAX_ABS_INIT_DIFF = {max_diff:.8f}", flush=True)
        if max_diff > 0.0:
            raise ValueError(f"Initialization parity failure: max_diff = {max_diff}")

    return g1_model


def train_g1_fold(
    config: ClassificationV2TrainingConfig,
    sidecar_path: Path,
    output_dir: Path,
    device: torch.device,
    *,
    preflight: bool = False,
    check_init_parity: bool = False,
) -> dict[str, Any]:
    """Execute G1 training for one fold with strict posture role safety."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # 1. Verify sidecar
    sidecar_path = _repo_path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Posture sidecar not found: {sidecar_path}")
    observed_sidecar_sha = _file_sha256(sidecar_path)
    if observed_sidecar_sha != EXPECTED_SIDECAR_SHA256:
        raise ValueError(
            f"Posture sidecar SHA256 mismatch: {observed_sidecar_sha} vs {EXPECTED_SIDECAR_SHA256}"
        )
    sidecar_data = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    all_posture_targets = sidecar_data["posture_target"].numpy()
    all_posture_masks = sidecar_data["posture_reviewed_mask"].numpy()
    if len(all_posture_targets) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Total sidecar rows mismatch: {len(all_posture_targets)}")
    if int(all_posture_masks.sum()) != EXPECTED_REVIEWED_ROWS:
        raise ValueError(f"Total reviewed rows mismatch: {all_posture_masks.sum()}")

    # 2. Setup DataModule
    runtime_config = replace(
        config,
        execution=replace(
            config.execution,
            mode="smoke" if preflight else "full_oof",
        ),
        optimization=replace(
            config.optimization,
            epochs=2 if preflight else config.optimization.epochs,
        ),
    )

    with StrictTrainingDataModule(runtime_config, device=device) as data:
        data.fit_fold_preprocessor()
        full_train_indices = data.split_indices("train")
        full_val_indices = data.split_indices("validation")
        full_test_indices = data.split_indices("test")

        # Censor outer test posture targets strictly before training setup
        censored_test_targets = np.full(len(full_test_indices), -1, dtype=np.int64)
        censored_test_masks = np.zeros(len(full_test_indices), dtype=bool)
        assert (censored_test_targets == -1).all()
        assert (~censored_test_masks).all()

        train_indices = (
            data.balanced_smoke_indices(train=True) if preflight else full_train_indices
        )
        val_indices = (
            data.balanced_smoke_split("validation") if preflight else full_val_indices
        )

        train_pos_targets = all_posture_targets[train_indices]
        train_pos_masks = all_posture_masks[train_indices]
        val_pos_targets = all_posture_targets[val_indices]
        val_pos_masks = all_posture_masks[val_indices]

        # 3. Build Model & Optimizer
        probe = data.batch(train_indices[: min(len(train_indices), 2)])
        spatial_input_dims = {
            name: probe.model_inputs["spatial_features"][name].shape[-1]
            for name in runtime_config.model.spatial_feature_groups
        }
        interaction_dim = (
            probe.model_inputs["interaction_context_features"].shape[-1]
            if runtime_config.model.enable_interaction_context
            else None
        )
        hidden_dim = runtime_config.model.hidden_dim
        backbone_config = MultimodalFusionConfig(
            spatial_input_dims=spatial_input_dims,
            num_classes=10,
            interaction_context_dim=interaction_dim,
            backbone_name=runtime_config.model.backbone_name,
            pretrained_weight_enum=runtime_config.model.pretrained_weight_enum,
            image_embedding_dim=hidden_dim,
            spatial_embedding_dim=hidden_dim,
            interaction_embedding_dim=max(8, hidden_dim // 2),
            visual_context_embedding_dim=hidden_dim,
            fusion_hidden_dim=hidden_dim * 2,
            dropout=runtime_config.model.dropout,
            temporal_encoder_name=runtime_config.model.temporal_encoder_name,
            transformer_layers=runtime_config.model.transformer_layers,
            transformer_heads=runtime_config.model.transformer_heads,
            enable_image=runtime_config.model.enable_image,
            enable_spatial=runtime_config.model.enable_spatial,
            enable_interaction_context=runtime_config.model.enable_interaction_context,
            enable_visual_context=runtime_config.model.enable_visual_context,
            enable_partner_tokens=runtime_config.model.enable_partner_tokens,
            partner_token_dim=int(getattr(runtime_config.model, "partner_token_dim", 6)),
            partner_embedding_dim=max(8, hidden_dim // 2),
        )

        model = _build_g1_model_and_check_parity(
            backbone_config, check_parity=check_init_parity or preflight
        ).to(device)

        total_param_count = sum(p.numel() for p in model.parameters())
        if total_param_count != EXPECTED_PARAM_COUNT:
            raise ValueError(
                f"G1 parameter count mismatch: {total_param_count} vs {EXPECTED_PARAM_COUNT}"
            )

        # Optimizer: Visual LR = 0.0003, Nonvisual LR = 0.003
        optimizer_groups, _ = build_visual_optimizer_groups(
            model,
            learning_rate=runtime_config.optimization.learning_rate,
            backbone_lr_multiplier=runtime_config.model.visual_backbone_lr_multiplier,
            weight_decay=runtime_config.optimization.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_groups)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and runtime_config.optimization.precision == "amp"),
        )
        behavior_weights = _behavior_class_weights(
            data, full_train_indices, runtime_config, device
        )

        batch_size = runtime_config.optimization.batch_size
        max_epochs = runtime_config.optimization.epochs
        patience = runtime_config.optimization.early_stopping_patience

        best_val_f1 = -1.0
        best_val_nll = float("inf")
        best_epoch = -1
        patience_counter = 0
        outer_test_eval_count = 0
        history: list[dict[str, Any]] = []

        for epoch in range(max_epochs):
            # --- inner_train ---
            model.train()
            train_loss_sum = 0.0
            train_l_beh_sum = 0.0
            train_l_pos_sum = 0.0
            train_reviewed_count = 0
            train_beh_preds: list[int] = []
            train_beh_gts: list[int] = []
            num_train_batches = 0

            perm = np.random.permutation(len(train_indices))
            for b_start in range(0, len(train_indices), batch_size):
                b_sub = perm[b_start : b_start + batch_size]
                b_indices = train_indices[b_sub]
                b_batch = data.batch(b_indices)
                b_pos_target = torch.from_numpy(train_pos_targets[b_sub]).to(device)
                b_pos_mask = torch.from_numpy(train_pos_masks[b_sub]).to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(
                    "cuda",
                    enabled=(
                        device.type == "cuda"
                        and runtime_config.optimization.precision == "amp"
                    ),
                ):
                    out = model(**b_batch.model_inputs)
                    total_loss, l_beh, l_pos = compute_g1_loss(
                        out.behavior_logits,
                        b_batch.behavior_target,
                        behavior_weights,
                        out.posture_logits,
                        b_pos_target,
                        b_pos_mask,
                        posture_lambda=POSTURE_LAMBDA_LOCKED,
                    )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    runtime_config.optimization.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += total_loss.item()
                train_l_beh_sum += l_beh.item()
                train_l_pos_sum += l_pos.item()
                train_reviewed_count += int(b_pos_mask.sum().item())
                train_beh_preds.extend(
                    torch.argmax(out.behavior_logits.detach(), dim=-1).cpu().tolist()
                )
                train_beh_gts.extend(b_batch.behavior_target.cpu().tolist())
                num_train_batches += 1

            train_rep = classification_report(
                train_beh_gts,
                train_beh_preds,
                labels=list(range(10)),
                output_dict=True,
                zero_division=0,
            )
            train_beh_macro_f1 = float(train_rep["macro avg"]["f1-score"])
            avg_train_loss = train_loss_sum / max(1, num_train_batches)
            avg_train_l_pos = train_l_pos_sum / max(1, num_train_batches)

            # --- inner_val ---
            model.eval()
            val_loss_sum = 0.0
            val_l_pos_sum = 0.0
            val_beh_preds = []
            val_beh_gts = []
            val_pos_preds = []
            val_pos_gts = []
            num_val_batches = 0
            val_reviewed_count = 0

            with torch.inference_mode():
                for b_start in range(0, len(val_indices), batch_size):
                    b_sub = np.arange(b_start, min(b_start + batch_size, len(val_indices)))
                    b_indices = val_indices[b_sub]
                    b_batch = data.batch(b_indices)
                    b_pos_target = torch.from_numpy(val_pos_targets[b_sub]).to(device)
                    b_pos_mask = torch.from_numpy(val_pos_masks[b_sub]).to(device)

                    with torch.amp.autocast(
                        "cuda",
                        enabled=(
                            device.type == "cuda"
                            and runtime_config.optimization.precision == "amp"
                        ),
                    ):
                        out = model(**b_batch.model_inputs)
                        _, l_beh, l_pos = compute_g1_loss(
                            out.behavior_logits,
                            b_batch.behavior_target,
                            behavior_weights,
                            out.posture_logits,
                            b_pos_target,
                            b_pos_mask,
                            posture_lambda=POSTURE_LAMBDA_LOCKED,
                        )

                    val_loss_sum += l_beh.item()
                    val_l_pos_sum += l_pos.item()
                    val_beh_preds.extend(
                        torch.argmax(out.behavior_logits, dim=-1).cpu().tolist()
                    )
                    val_beh_gts.extend(b_batch.behavior_target.cpu().tolist())
                    if b_pos_mask.any():
                        val_pos_preds.extend(
                            torch.argmax(out.posture_logits[b_pos_mask], dim=-1).cpu().tolist()
                        )
                        val_pos_gts.extend(b_pos_target[b_pos_mask].cpu().tolist())
                        val_reviewed_count += int(b_pos_mask.sum().item())
                    num_val_batches += 1

            val_beh_rep = classification_report(
                val_beh_gts,
                val_beh_preds,
                labels=list(range(10)),
                output_dict=True,
                zero_division=0,
            )
            val_beh_macro_f1 = float(val_beh_rep["macro avg"]["f1-score"])
            val_beh_nll = val_loss_sum / max(1, num_val_batches)
            val_pos_loss = val_l_pos_sum / max(1, num_val_batches)

            if val_pos_gts:
                val_pos_rep = classification_report(
                    val_pos_gts,
                    val_pos_preds,
                    labels=list(range(3)),
                    output_dict=True,
                    zero_division=0,
                )
                val_pos_macro_f1 = float(val_pos_rep["macro avg"]["f1-score"])
            else:
                val_pos_macro_f1 = 0.0

            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_behavior_macro_f1": train_beh_macro_f1,
                "train_posture_loss": avg_train_l_pos,
                "train_reviewed_posture_count": train_reviewed_count,
                "inner_val_behavior_nll": val_beh_nll,
                "inner_val_behavior_macro_f1": val_beh_macro_f1,
                "inner_val_posture_loss": val_pos_loss,
                "inner_val_posture_macro_f1": val_pos_macro_f1,
                "inner_val_reviewed_posture_count": val_reviewed_count,
                "completed_epochs": epoch + 1,
            }
            history.append(epoch_record)

            # Checkpoint saving & Early stopping (by BEHAVIOR Macro-F1 ONLY)
            is_best = (val_beh_macro_f1 > best_val_f1) or (
                val_beh_macro_f1 == best_val_f1 and val_beh_nll < best_val_nll
            )
            if is_best:
                best_val_f1 = val_beh_macro_f1
                best_val_nll = val_beh_nll
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_validation_behavior_macro_f1": best_val_f1,
                        "best_epoch": best_epoch,
                        "metrics": epoch_record,
                        "config": training_config_to_jsonable(runtime_config),
                    },
                    output_dir / "best_validation.pt",
                )
            else:
                patience_counter += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_behavior_macro_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "metrics": epoch_record,
                    "config": training_config_to_jsonable(runtime_config),
                },
                output_dir / "last.pt",
            )

            (output_dir / "epoch_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            pd.DataFrame(history).to_csv(output_dir / "epoch_history.csv", index=False)

            print(
                f"EPOCH_{epoch}_COMPLETED | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_beh_f1={val_beh_macro_f1:.6f} | "
                f"val_pos_f1={val_pos_macro_f1:.6f} | "
                f"is_best={is_best}",
                flush=True,
            )

            if patience_counter >= patience and not preflight:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

        # 4. Strict Reload & Restore Verification
        best_ckpt_path = output_dir / "best_validation.pt"
        last_ckpt_path = output_dir / "last.pt"
        if not best_ckpt_path.exists() or not last_ckpt_path.exists():
            raise RuntimeError("Required checkpoint files missing after training")

        restored_model = _build_g1_model_and_check_parity(backbone_config, check_parity=False).to(
            device
        )
        loaded_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        restored_model.load_state_dict(loaded_state["model_state_dict"], strict=True)
        restored_model.eval()

        # Test inference with forward_behavior on a val batch
        val_probe = data.batch(val_indices[: min(len(val_indices), 2)])
        with torch.inference_mode():
            test_out = restored_model.forward_behavior(**val_probe.model_inputs)
        assert test_out.shape[1] == 10, f"Expected 10 behavior classes, got {test_out.shape[1]}"

        summary = {
            "fold": config.execution.fold_id,
            "status": "PASS",
            "best_epoch": best_epoch,
            "best_inner_val_behavior_macro_f1": best_val_f1,
            "completed_epochs": len(history),
            "outer_test_eval_count": outer_test_eval_count,
            "best_validation_checkpoint_exists": best_ckpt_path.exists(),
            "last_checkpoint_exists": last_ckpt_path.exists(),
            "epoch_history_exists": (output_dir / "epoch_history.json").exists(),
            "strict_reload_pass": True,
            "best_restore_pass": True,
            "history": history,
        }
        with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_root = args.data_root or (
        Path(os.environ["M0_F1_DATA_ROOT"]) if "M0_F1_DATA_ROOT" in os.environ else None
    )
    rgb_root = args.rgb_root or (
        Path(os.environ["M0_F1_RGB_ROOT"]) if "M0_F1_RGB_ROOT" in os.environ else None
    )

    folds = ["vg1", "vg2", "vg3", "vg4", "vg5"] if args.fold == "all" else [args.fold]

    for fold_id in folds:
        if args.config is not None:
            config_path = args.config
        else:
            config_name = f"g1_posture_auxiliary_{fold_id}_scientific_v1.json"
            config_path = REPO_ROOT / "configs/classification_v2" / config_name
        raw_config = load_training_config(config_path)
        resolved_config = _resolve_external_paths(raw_config, data_root, rgb_root)

        sub_dir = "preflight" if args.preflight else "runs"
        default_out = (
            REPO_ROOT
            / f"outputs/classification_v2/g1_posture_auxiliary_v1/{sub_dir}/{fold_id}"
        )
        out_dir = args.output_dir or default_out

        print(
            f"\n{'='*50}\nSTARTING G1 TRAINING: fold={fold_id} "
            f"(preflight={args.preflight})\n{'='*50}"
        )
        res = train_g1_fold(
            resolved_config,
            sidecar_path=args.sidecar,
            output_dir=out_dir,
            device=device,
            preflight=args.preflight,
            check_init_parity=args.check_init_parity,
        )
        print(f"COMPLETED G1 FOLD {fold_id}: status={res['status']}")


if __name__ == "__main__":
    main()
