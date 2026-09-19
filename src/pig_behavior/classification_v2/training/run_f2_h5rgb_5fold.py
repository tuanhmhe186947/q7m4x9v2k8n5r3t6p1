"""Exact 5-fold scientific training runner for F2 Causal Pre-target Actor RGB H5.

Implements the promoted M2 visual-fine-tuning recipe + pre-target actor RGB H5:
  - visual LR = 0.0003 (0.1x backbone multiplier)
  - nonvisual + adapter LR = 0.003
  - shared actor ImageSequenceEncoder for T6 actor and H5 history actor
  - Linear(256, 128) adapter with exact Step-0 identity/zero initialization
  - AdamW, weight_decay = 0.0, scheduler = none, clip = 1.0
  - behavior weighted CE
  - inner_train / inner_val per epoch
  - best checkpoint selected ONLY by inner_val_behavior_macro_f1
  - max_epochs = 30, early_stopping_patience = 5
  - zero outer evaluation during training
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
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.models.f2_h5_rgb_model import (  # noqa: E402
    F2H5RgbClassifier,
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
EXPECTED_PARAM_COUNT: int = 43666728
EXPECTED_TOTAL_ROWS: int = 33287
EXPECTED_FULL_H5_ROWS: int = 33191
EXPECTED_NO_H5_ROWS: int = 96
DEFAULT_H5_DIR: Path = REPO_ROOT / "outputs/classification_v2/h5_actor_rgb_v1"
DATA_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_DATA_ROOT"
RGB_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_RGB_ROOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F2 H5-RGB 5-Fold Scientific Runner")
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
        "--h5-dir",
        type=Path,
        default=DEFAULT_H5_DIR,
        help="Path to authoritative H5 RGB sidecar directory.",
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
        help="Root for explicit external RGB cache artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/classification_v2/f2_h5_actor_rgb_v1",
        help="Root output directory for run artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs (default from config or 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default from config).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run bounded CPU preflight (2 epochs on single fold) and exit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit compute device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _resolve_runtime_config(
    fold_name: str,
    base_config_path: Path | None,
    data_root: Path | None,
    rgb_root: Path | None,
) -> ClassificationV2TrainingConfig:
    fold_num = fold_name.replace("vg", "").replace("VG", "")
    if base_config_path is None:
        configs_dir = REPO_ROOT / "configs/classification_v2"
        base_config_path = configs_dir / f"f2_h5rgb_earlystop_vg{fold_num}_scientific_v1.json"

    if not base_config_path.exists():
        raise FileNotFoundError(f"Authority config missing: {base_config_path}")

    config = load_training_config(base_config_path)
    env_data_root = os.environ.get(DATA_EXTERNAL_PREFIX)
    env_rgb_root = os.environ.get(RGB_EXTERNAL_PREFIX)
    effective_data_root = data_root or (Path(env_data_root) if env_data_root else None)
    effective_rgb_root = rgb_root or (Path(env_rgb_root) if env_rgb_root else None)

    default_data_root = (
        REPO_ROOT / "outputs/classification_v2/full_t6_canonical_46d_20260816"
    )
    default_rgb_root = REPO_ROOT / "outputs/classification_v2/m0_window_major_r128_t6"

    eff_data = effective_data_root or default_data_root
    eff_rgb = effective_rgb_root or default_rgb_root

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
            updates[field_name] = eff_data / text[len(data_marker) :]
        elif text.startswith(rgb_marker):
            updates[field_name] = eff_rgb / text[len(rgb_marker) :]
        elif field_name in {
            "snapshot_json",
            "trainer_contract_json",
            "grouped_fold_roles",
        }:
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    return replace(config, dataset=replace(dataset, **updates))


def verify_f2_step0_parity(
    model: F2H5RgbClassifier,
    m2_model: MultimodalFusionClassifier,
    data: StrictTrainingDataModule,
    h5_rgb_tensor: np.ndarray,
    h5_masks_arr: np.ndarray,
    device: torch.device,
) -> float:
    """Verify exact step-0 logit parity between M2 and F2 (max abs diff <= 1e-7)."""
    model.eval()
    m2_model.eval()

    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    b_batch = data.batch(sample_indices)

    h5_u8 = h5_rgb_tensor[sample_indices]
    h5_float = (
        torch.from_numpy(h5_u8).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
    ).to(device)
    h5_mask = torch.from_numpy(h5_masks_arr[sample_indices]).to(device)
    h5_len_mask = h5_mask.float()
    h5_obs_mask = h5_mask.float()

    with torch.inference_mode():
        m2_logits = m2_model(**b_batch.model_inputs)
        f2_logits = model(
            **b_batch.model_inputs,
            history_image=h5_float,
            history_length_mask=h5_len_mask,
            history_observed_mask=h5_obs_mask,
            history_available_mask=h5_mask,
        )

    max_diff = (m2_logits - f2_logits).abs().max().item()
    if max_diff > 1e-7:
        raise ValueError(f"F2 step-0 parity check failed! Max logit diff: {max_diff}")
    return max_diff


def run_f2_training(
    fold_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(SEED)
    fold_upper = fold_name.upper()
    run_dir = args.output_dir / "runs" / fold_name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"STARTING F2 H5-RGB SCIENTIFIC RUN: {fold_upper}")
    print("=======================================================")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Compute Device: {device}")

    # 1. Verify H5 Sidecar
    h5_dir = Path(args.h5_dir)
    h5_npy_path = h5_dir / "h5_actor_rgb_u8.npy"
    h5_masks_path = h5_dir / "h5_actor_rgb_masks.npz"
    h5_manifest_path = h5_dir / "h5_actor_rgb_manifest.json"

    if (
        not h5_npy_path.exists()
        or not h5_masks_path.exists()
        or not h5_manifest_path.exists()
    ):
        raise FileNotFoundError(f"H5 sidecar artifacts missing under {h5_dir}")

    h5_rgb_mmap = np.load(h5_npy_path, mmap_mode="r")
    h5_masks_data = np.load(h5_masks_path)
    h5_avail_mask = h5_masks_data["history_available_mask"]

    if h5_rgb_mmap.shape != (EXPECTED_TOTAL_ROWS, 5, 128, 128, 3):
        raise ValueError(f"Unexpected H5 RGB shape: {h5_rgb_mmap.shape}")
    if h5_avail_mask.shape != (EXPECTED_TOTAL_ROWS, 5):
        raise ValueError(f"Unexpected H5 mask shape: {h5_avail_mask.shape}")

    # 2. Resolve Config & Data
    runtime_config = _resolve_runtime_config(
        fold_name, args.config, args.data_root, args.rgb_root
    )
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    # Resolve splits
    full_train_indices = data.split_indices("train")
    full_val_indices = data.split_indices("validation")
    test_indices = data.split_indices("test")

    train_indices = full_train_indices[:128] if args.preflight else full_train_indices
    val_indices = full_val_indices[:64] if args.preflight else full_val_indices

    print(
        f"Rows: Train={len(train_indices)} (full={len(full_train_indices)}), "
        f"Val={len(val_indices)} (full={len(full_val_indices)}), Test={len(test_indices)}"
    )

    # 3. Model Architecture
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

    model = F2H5RgbClassifier(backbone_config).to(device)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Total Parameters: {total_param_count} (Expected: {EXPECTED_PARAM_COUNT})")
    if total_param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(
            f"F2 parameter count mismatch: {total_param_count} vs {EXPECTED_PARAM_COUNT}"
        )

    # 4. Parity Check
    set_seed(SEED)
    m2_ref_model = MultimodalFusionClassifier(backbone_config).to(device)
    model.backbone.load_state_dict(m2_ref_model.state_dict())
    step0_diff = verify_f2_step0_parity(
        model, m2_ref_model, data, h5_rgb_mmap, h5_avail_mask, device
    )
    print(f"Verified Step-0 M2 Parity: Max Logit Diff = {step0_diff:.10e} (PASS <= 1e-7)")
    del m2_ref_model

    # 5. Optimizer Setup
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

    full_train_indices = data.split_indices("train")
    behavior_weights = _behavior_class_weights(
        data, full_train_indices, runtime_config, device
    )

    batch_size = args.batch_size or runtime_config.optimization.batch_size
    max_epochs = args.epochs or (2 if args.preflight else runtime_config.optimization.epochs)
    patience = runtime_config.optimization.early_stopping_patience

    best_val_f1 = -1.0
    best_val_nll = float("inf")
    best_epoch = -1
    patience_counter = 0
    outer_test_eval_count = 0
    history: list[dict[str, Any]] = []

    best_ckpt_path = run_dir / "best_validation.pt"
    last_ckpt_path = run_dir / "last.pt"
    history_path = run_dir / "epoch_history.json"

    print(f"Training Plan: Max Epochs={max_epochs}, Batch Size={batch_size}, Patience={patience}")

    for epoch in range(max_epochs):
        # --- inner_train ---
        model.train()
        train_loss_sum = 0.0
        train_preds: list[int] = []
        train_gts: list[int] = []
        num_train_batches = 0

        perm = np.random.permutation(len(train_indices))
        for b_start in range(0, len(train_indices), batch_size):
            b_sub = perm[b_start : b_start + batch_size]
            b_indices = train_indices[b_sub]
            b_batch = data.batch(b_indices)

            # Load H5 sidecar slice
            h5_u8 = h5_rgb_mmap[b_indices]
            h5_float = (
                torch.from_numpy(h5_u8).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
            ).to(device)
            h5_mask = torch.from_numpy(h5_avail_mask[b_indices]).to(device)
            h5_len_mask = h5_mask.float()
            h5_obs_mask = h5_mask.float()

            optimizer.zero_grad()
            with torch.amp.autocast(
                "cuda",
                enabled=(
                    device.type == "cuda"
                    and runtime_config.optimization.precision == "amp"
                ),
            ):
                logits = model(
                    **b_batch.model_inputs,
                    history_image=h5_float,
                    history_length_mask=h5_len_mask,
                    history_observed_mask=h5_obs_mask,
                    history_available_mask=h5_mask,
                )
                loss = F.cross_entropy(
                    logits,
                    b_batch.behavior_target,
                    weight=behavior_weights,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                runtime_config.optimization.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_preds.extend(torch.argmax(logits.detach(), dim=-1).cpu().tolist())
            train_gts.extend(b_batch.behavior_target.cpu().tolist())
            num_train_batches += 1

        train_rep = classification_report(
            train_gts,
            train_preds,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )
        train_macro_f1 = float(train_rep["macro avg"]["f1-score"])
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # --- inner_val ---
        model.eval()
        val_loss_sum = 0.0
        val_preds: list[int] = []
        val_gts: list[int] = []
        num_val_batches = 0

        with torch.inference_mode():
            for b_start in range(0, len(val_indices), batch_size):
                b_sub = np.arange(b_start, min(b_start + batch_size, len(val_indices)))
                b_indices = val_indices[b_sub]
                b_batch = data.batch(b_indices)

                h5_u8 = h5_rgb_mmap[b_indices]
                h5_float = (
                    torch.from_numpy(h5_u8).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
                ).to(device)
                h5_mask = torch.from_numpy(h5_avail_mask[b_indices]).to(device)
                h5_len_mask = h5_mask.float()
                h5_obs_mask = h5_mask.float()

                with torch.amp.autocast(
                    "cuda",
                    enabled=(
                        device.type == "cuda"
                        and runtime_config.optimization.precision == "amp"
                    ),
                ):
                    logits = model(
                        **b_batch.model_inputs,
                        history_image=h5_float,
                        history_length_mask=h5_len_mask,
                        history_observed_mask=h5_obs_mask,
                        history_available_mask=h5_mask,
                    )
                    v_loss = F.cross_entropy(
                        logits,
                        b_batch.behavior_target,
                        weight=behavior_weights,
                    )

                val_loss_sum += v_loss.item()
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                val_gts.extend(b_batch.behavior_target.cpu().tolist())
                num_val_batches += 1

        val_rep = classification_report(
            val_gts,
            val_preds,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )
        val_macro_f1 = float(val_rep["macro avg"]["f1-score"])
        avg_val_loss = val_loss_sum / max(1, num_val_batches)

        # STRICT ZERO OUTER TEST EVALUATION DURING TRAINING
        assert outer_test_eval_count == 0, "Outer test evaluation occurred during training!"

        is_best = (val_macro_f1 > best_val_f1) or (
            abs(val_macro_f1 - best_val_f1) < 1e-6 and avg_val_loss < best_val_nll
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_behavior_macro_f1": train_macro_f1,
            "inner_val_loss": avg_val_loss,
            "inner_val_behavior_macro_f1": val_macro_f1,
            "is_best": is_best,
        }
        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} F1: {train_macro_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} F1: {val_macro_f1:.4f} | {'*BEST*' if is_best else ''}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch if not is_best else epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": training_config_to_jsonable(runtime_config),
            "metrics": epoch_record,
        }
        torch.save(ckpt_payload, last_ckpt_path)

        if is_best:
            best_val_f1 = val_macro_f1
            best_val_nll = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(ckpt_payload, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and not args.preflight:
                print(f"Early stopping triggered at Epoch {epoch:02d}.")
                break

    # Reload best checkpoint to verify strict restorable state
    print(f"\nReloading best checkpoint from {best_ckpt_path} (Best Epoch: {best_epoch})...")
    saved_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(saved_ckpt["model_state_dict"])
    model.eval()

    best_sha = sha256_file(best_ckpt_path)
    last_sha = sha256_file(last_ckpt_path)
    history_sha = sha256_file(history_path)

    print(f"BEST_CHECKPOINT_SHA256: {best_sha}")
    print(f"LAST_CHECKPOINT_SHA256: {last_sha}")
    print(f"EPOCH_HISTORY_SHA256:   {history_sha}")
    print(f"OUTER_TEST_EVAL_COUNT:  {outer_test_eval_count}")

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_val_loss": best_val_nll,
        "best_checkpoint_sha256": best_sha,
        "last_checkpoint_sha256": last_sha,
        "history_sha256": history_sha,
        "outer_test_eval_count": outer_test_eval_count,
        "history": history,
    }


def main():
    args = parse_args()
    if args.fold == "all":
        folds = ["vg1", "vg2", "vg3", "vg4", "vg5"]
    else:
        folds = [args.fold]

    results = {}
    for fold in folds:
        res = run_f2_training(fold, args)
        results[fold] = res

    print("\n=======================================================")
    print("ALL REQUESTED FOLDS COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
