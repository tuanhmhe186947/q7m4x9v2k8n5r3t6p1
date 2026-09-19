"""Model J 5-Fold Training Runner with PB_NEW (E_J Teacher).

Task ID: FINAL-PBNEW-EJ-TEACHER-20260916
Execution Agent: Gemini

Architecture:
  - Frozen Model A (DeepLocalJointRepresentationClassifier)
  - Pre-GAP Local Spatial Attention (ActorLocal, UnionLocal)
  - Class-Aware Gate (12 multimodal sources)
  - Zero-init residual delta logits

PB Input:
  - PB_NEW E_J teacher partner probabilities (outputs/classification_v2/pb_teacher_ej_fixed50_v1)
  - Preserved PB F2 partner hidden [N, 2, 256]
  - Preserved canonical PB F2 partner mask [N, 2]

Recipe:
  - Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
  - Cosine scheduler, final_lr_fraction=0.10
  - 10 epochs max, early stopping patience 3
  - Seed: 240494961
  - Step-0 Parity: max |J_PBNEW_STEP0 - A_PBNEW| <= 1e-7, argmax parity = 100%
  - Per-epoch atomic persistence: last_training.pt and best_validation.pt
  - Physical disk reload authority & SHA-256 verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.near_final_model_j import (
    MODEL_J_ARCHITECTURE_VERSION,
    NearFinalModelJ,
)
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.data_plus_multimodal import (
    DataPlusMultimodalStore,
    DataPlusSidecarPaths,
)
from pig_behavior.classification_v2.training.run_joint_representation_5fold import (
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_H5_STRUCTURED_PATH,
    DEFAULT_POSTURE_SIDECAR,
    JointKeyedBatchResolver,
    _resolve_runtime_config,
)

SEED = 240494961
FOLDS = ["vg1", "vg2", "vg3", "vg4", "vg5"]

EXPECTED_CONTROL_HASHES = {
    "vg1": "4cfa96122769cd44428b8dd4732138344633246602aeafaf34c3862a327a9c95",
    "vg2": "a3bec7187bfaa65c09355b19b3f09b3c875269373e170d2ad0b4982f89afe659",
    "vg3": "39ad0847c6391829c00876cb17b0d2d603a376f372438087c83fca15af953154",
    "vg4": "c3c8fc458155e33aa882a6dab33b3337030dc9ac973491d6e3d1efe826e43412",
    "vg5": "b4046a241ae2a5f391b378e6874c19147436544e4a28517fd86c5ff6a0090650",
}

CANONICAL_BEHAVIORS = [
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
]

DEFAULT_PBNEW_DIR = Path("outputs/classification_v2/pb_teacher_ej_fixed50_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/classification_v2/final_model_j_pbnew_ejteacher_v1/runs")
DEFAULT_BASE_MODEL_A_DIR = Path("outputs/classification_v2/final_high_ceiling_v1/runs")

HISTORICAL_J_F2_SCORES = {
    "vg1": 0.709400,
    "vg2": 0.694903,
    "vg3": 0.633957,
    "vg4": 0.692737,
    "vg5": 0.692233,
}
J_F2_MEAN = 0.684646

A_F2_SCORES = {
    "vg1": 0.700872,
    "vg2": 0.694903,
    "vg3": 0.632632,
    "vg4": 0.680758,
    "vg5": 0.679783,
}


def log(msg: str) -> None:
    print(msg, flush=True)
    with open("live_progress_pbnew.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_base_model(
    fold_name: str,
    data: StrictTrainingDataModule,
    runtime_config: Any,
    device: torch.device,
    base_model_dir: Path = DEFAULT_BASE_MODEL_A_DIR,
) -> tuple[DeepLocalJointRepresentationClassifier, str]:
    full_train_indices = data.split_indices("train")
    probe = data.batch(full_train_indices[: min(len(full_train_indices), 2)])
    spatial_dims = {
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
        spatial_input_dims=spatial_dims,
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
        enable_partner_tokens=False,
    )

    base_model = DeepLocalJointRepresentationClassifier(backbone_config).to(device)
    ckpt_path = base_model_dir / fold_name / "best_validation.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Base Model A checkpoint: {ckpt_path}")

    ckpt_sha = sha256_file(ckpt_path)
    expected_sha = EXPECTED_CONTROL_HASHES[fold_name]
    if ckpt_sha != expected_sha:
        raise ValueError(
            f"Base Model A SHA256 mismatch for {fold_name}: {ckpt_sha} != {expected_sha}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    for p in base_model.parameters():
        p.requires_grad = False

    return base_model, ckpt_sha


def get_model_hash_dict(model: nn.Module) -> dict[str, str]:
    hashes = {}
    for name, param in model.named_parameters():
        h = hashlib.sha256(param.data.cpu().numpy().tobytes()).hexdigest()
        hashes[name] = h
    return hashes


def get_bn_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    bn_states = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_states[f"{name}.running_mean"] = m.running_mean.clone()
            bn_states[f"{name}.running_var"] = m.running_var.clone()
    return bn_states


def verify_bn_parity(s1: dict[str, torch.Tensor], s2: dict[str, torch.Tensor]) -> bool:
    if set(s1.keys()) != set(s2.keys()):
        return False
    for k in s1:
        if not torch.equal(s1[k], s2[k]):
            return False
    return True


def evaluate_model_j(
    model: NearFinalModelJ,
    resolver: JointKeyedBatchResolver,
    val_keys: list[str],
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    all_final_logits = []
    all_base_logits = []
    all_source_weights = []
    all_targets = []

    with torch.inference_mode():
        num_batches = int(np.ceil(len(val_keys) / batch_size))
        for b_idx in range(num_batches):
            b_keys = tuple(val_keys[b_idx * batch_size : (b_idx + 1) * batch_size])
            aligned = resolver.batch(b_keys)

            final_logits, base_logits, weights, _, _ = model(aligned)

            all_final_logits.append(final_logits.cpu().numpy())
            all_base_logits.append(base_logits.cpu().numpy())
            all_source_weights.append(weights.cpu().numpy())
            all_targets.append(aligned.training_batch.behavior_target.cpu().numpy())

    final_logits = np.concatenate(all_final_logits, axis=0)
    base_logits = np.concatenate(all_base_logits, axis=0)
    weights = np.concatenate(all_source_weights, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    final_preds = final_logits.argmax(axis=-1)
    base_preds = base_logits.argmax(axis=-1)

    final_macro_f1 = float(f1_score(targets, final_preds, average="macro", zero_division=0))
    base_macro_f1 = float(f1_score(targets, base_preds, average="macro", zero_division=0))

    per_class_j = precision_recall_fscore_support(
        targets, final_preds, labels=list(range(10)), average=None, zero_division=0
    )[2]
    per_class_a = precision_recall_fscore_support(
        targets, base_preds, labels=list(range(10)), average=None, zero_division=0
    )[2]

    return {
        "final_f1": final_macro_f1,
        "base_f1": base_macro_f1,
        "final_logits": final_logits,
        "base_logits": base_logits,
        "weights": weights,
        "targets": targets,
        "per_class_j": per_class_j,
        "per_class_a": per_class_a,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model J 5-Fold Training with PB_NEW.")
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--pb-dir", type=Path, default=DEFAULT_PBNEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_A_DIR)
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 60)
    log("CAMPAIGN: MODEL J 5-FOLD TRAINING WITH PB_NEW (E_J TEACHER)")
    log("TASK_ID: FINAL-PBNEW-EJ-TEACHER-20260916")
    log("=" * 60)
    log(f"Compute Device: {device}")
    if device.type == "cuda":
        log(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    log(f"MODEL_J_ARCHITECTURE_VERSION = {MODEL_J_ARCHITECTURE_VERSION}")
    log("N_SOURCES = 12")
    log("TEMPORAL = T6")
    log(f"SEED = {SEED}")
    log("OUTER_TEST_EVALUATIONS = 0")
    log(f"PB_SOURCE = {args.pb_dir}")

    results_summary: dict[str, Any] = {}

    for fold_name in args.folds:
        log("\n" + "=" * 56)
        log(f"STARTING MODEL J (PB_NEW): FOLD {fold_name.upper()}")
        log("=" * 56)

        fold_dir = args.output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed and verified
        if (
            (fold_dir / "best_validation.pt").exists()
            and (fold_dir / "fold_manifest.json").exists()
            and (fold_dir / "metrics.json").exists()
        ):
            try:
                with open(fold_dir / "fold_manifest.json", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("reloaded_parity") == "PASS":
                    log(f"Fold {fold_name.upper()} already completed and verified. Loading cached.")
                    results_summary[fold_name] = cached
                    continue
            except Exception as e:
                log(f"Cached manifest read error: {e}. Re-running fold.")

        runtime_config = _resolve_runtime_config(fold_name, None, None, None)
        data = StrictTrainingDataModule(runtime_config, device=device)
        data.fit_fold_preprocessor()

        train_indices = data.split_indices("train")
        val_indices = data.split_indices("validation")

        pb_cache_path = args.pb_dir / fold_name.lower() / "pb_teacher_features.pt"
        if not pb_cache_path.exists():
            raise FileNotFoundError(f"Missing PB_NEW cache: {pb_cache_path}")
        pb_cache = torch.load(pb_cache_path, map_location="cpu", weights_only=False)

        posture_sidecar = torch.load(
            DEFAULT_POSTURE_SIDECAR, map_location="cpu", weights_only=False
        )
        h5_cache = torch.load(
            DEFAULT_H5_STRUCTURED_PATH, map_location="cpu", weights_only=False
        )
        data_plus_store = DataPlusMultimodalStore(
            DataPlusSidecarPaths.from_root(DEFAULT_DATA_PLUS_DIR, pb_path=pb_cache_path)
        )

        resolver = JointKeyedBatchResolver(
            data, pb_cache, posture_sidecar, h5_cache, data_plus_store, device
        )

        train_keys = list(resolver.canonical_keys(train_indices))
        supp_manifest = DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv"
        if supp_manifest.exists():
            eligibility = pd.read_csv(supp_manifest, low_memory=False)
            fold_rows = eligibility[eligibility["fold"].astype(str).eq(fold_name.upper())]
            mask_elig = fold_rows["eligible_for_training"].astype(bool)
            eligible_units = fold_rows.loc[mask_elig, "supplemental_unit_id"].astype(str)
            data_plus_keys = [f"data_plus_{unit_id}" for unit_id in eligible_units.tolist()]
            train_keys.extend(data_plus_keys)
            log(f"DATA+ supplemental keys added: {len(data_plus_keys)}")

        val_keys = list(resolver.canonical_keys(val_indices))
        log(f"Train keys total: {len(train_keys)}, Val keys total: {len(val_keys)}")

        # Build Model J
        base_model, base_sha = _load_base_model(
            fold_name, data, runtime_config, device, base_model_dir=args.base_model_dir
        )
        model_j = NearFinalModelJ(base_model, token_dim=32, num_classes=10).to(device)

        base_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in model_j.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_j.parameters())
        param_pct = (trainable_params / base_params) * 100.0

        log(f"BASE_PARAMETER_COUNT = {base_params:,}")
        log(f"TRAINABLE_PARAMETER_COUNT = {trainable_params:,}")
        log(f"TOTAL_PARAMETER_COUNT = {total_params:,}")
        log(f"TRAINABLE_PERCENT = {param_pct:.4f}%")
        assert param_pct < 1.0, f"Parameter count exceeded 1% cap: {param_pct:.4f}%"

        # Step-0 Parity Audit (J_PBNEW_STEP0 == A_PBNEW)
        step0_eval = evaluate_model_j(model_j, resolver, val_keys)
        a_pbnew_f1 = step0_eval["base_f1"]
        max_logit_diff = float(
            np.max(np.abs(step0_eval["final_logits"] - step0_eval["base_logits"]))
        )
        j_step0_pred = step0_eval["final_logits"].argmax(axis=-1)
        a_step0_pred = step0_eval["base_logits"].argmax(axis=-1)
        argmax_parity = float((j_step0_pred == a_step0_pred).mean() * 100.0)

        log(f"[{fold_name.upper()}] A_F2_HISTORICAL = {A_F2_SCORES[fold_name]:.6f}")
        delta_a = a_pbnew_f1 - A_F2_SCORES[fold_name]
        log(f"[{fold_name.upper()}] A_PBNEW_STEP0   = {a_pbnew_f1:.6f} (DeltaVsAF2={delta_a:+.6f})")
        log(f"[{fold_name.upper()}] MAX_ABS_LOGIT_DIFF = {max_logit_diff:.2e}")
        log(f"[{fold_name.upper()}] ARGMAX_PARITY = {argmax_parity:.2f}%")
        assert max_logit_diff <= 1e-7, f"Step-0 logit diff exceeds 1e-7: {max_logit_diff}"
        assert argmax_parity == 100.0, f"Step-0 argmax parity is not 100%: {argmax_parity}"
        log("STEP0_LOGIT_PARITY = PASS")

        initial_base_hashes = get_model_hash_dict(base_model)
        initial_bn_states = get_bn_state_dict(base_model)

        # Setup Optimizer & Scheduler
        trainable_named_params = [
            (n, p) for n, p in model_j.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [p for _, p in trainable_named_params],
            lr=3e-4,
            weight_decay=1e-4,
        )

        epochs = 10
        patience = 3
        patience_counter = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=3e-4 * 0.10
        )
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = a_pbnew_f1
        best_epoch = -1
        replaces_step0 = False

        # Save Step-0 fallback
        torch.save(
            {
                "model_state_dict": model_j.state_dict(),
                "best_epoch": -1,
                "best_f1": a_pbnew_f1,
                "fold": fold_name,
                "architecture_version": MODEL_J_ARCHITECTURE_VERSION,
                "replaces_step0": False,
            },
            fold_dir / "best_validation.tmp.pt",
        )
        os.replace(fold_dir / "best_validation.tmp.pt", fold_dir / "best_validation.pt")

        for epoch in range(epochs):
            t0 = time.time()
            model_j.train()

            rng = np.random.RandomState(SEED + epoch * 1000)
            shuffled_keys = list(train_keys)
            rng.shuffle(shuffled_keys)

            train_loss_sum = 0.0
            train_batches = 0
            batch_size = 64
            num_train_batches = int(np.ceil(len(shuffled_keys) / batch_size))

            for b_idx in range(num_train_batches):
                b_keys = tuple(
                    shuffled_keys[b_idx * batch_size : (b_idx + 1) * batch_size]
                )
                aligned = resolver.batch(b_keys)

                optimizer.zero_grad()
                final_logits, _, _, _, _ = model_j(aligned)
                loss = criterion(final_logits, aligned.training_batch.behavior_target)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            scheduler.step()
            train_loss = train_loss_sum / max(1, train_batches)

            val_res = evaluate_model_j(model_j, resolver, val_keys)
            val_f1 = val_res["final_f1"]
            elapsed = time.time() - t0
            lr_cur = scheduler.get_last_lr()[0]

            # Atomic save of last_training.pt
            torch.save(
                {
                    "model_state_dict": model_j.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "best_f1": best_val_f1,
                    "seed": SEED,
                    "fold": fold_name,
                    "architecture_version": MODEL_J_ARCHITECTURE_VERSION,
                },
                fold_dir / "last_training.pt",
            )

            # Atomic save of best_validation.pt on improvement
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                replaces_step0 = True

                torch.save(
                    {
                        "model_state_dict": model_j.state_dict(),
                        "best_epoch": best_epoch,
                        "best_f1": best_val_f1,
                        "fold": fold_name,
                        "architecture_version": MODEL_J_ARCHITECTURE_VERSION,
                        "replaces_step0": replaces_step0,
                    },
                    fold_dir / "best_validation.tmp.pt",
                )
                os.replace(
                    fold_dir / "best_validation.tmp.pt", fold_dir / "best_validation.pt"
                )
                assert (fold_dir / "best_validation.pt").stat().st_size > 0

                loaded_test = torch.load(
                    fold_dir / "best_validation.pt",
                    map_location=device,
                    weights_only=False,
                )
                assert "model_state_dict" in loaded_test
            else:
                patience_counter += 1

            log(
                f"[{fold_name.upper()}] Epoch {epoch+1:02d}/{epochs:02d} | "
                f"TrainLoss={train_loss:.4f} | ValF1={val_f1:.6f} | "
                f"LR={lr_cur:.6f}, Time={elapsed:.1f}s | "
                f"Best={best_val_f1:.6f} (Ep{best_epoch}), ReplacesStep0={replaces_step0}"
            )

            if patience_counter >= patience:
                log(f"[{fold_name.upper()}] Early stopping triggered after {epoch+1} epochs.")
                break

        # Parity Verification on Frozen Base
        final_base_hashes = get_model_hash_dict(base_model)
        final_bn_states = get_bn_state_dict(base_model)
        weight_parity = initial_base_hashes == final_base_hashes
        bn_parity = verify_bn_parity(initial_bn_states, final_bn_states)

        log(f"[{fold_name.upper()}] BASE_WEIGHT_PARITY = {'PASS' if weight_parity else 'FAIL'}")
        log(f"[{fold_name.upper()}] BASE_BN_PARITY = {'PASS' if bn_parity else 'FAIL'}")
        assert weight_parity, f"Base weight parity failed for fold {fold_name}!"
        assert bn_parity, f"Base BN parity failed for fold {fold_name}!"

        # Reload from Physical Disk for Official Authority
        log(f"[{fold_name.upper()}] Reloading best_validation.pt fresh from disk...")
        reloaded_ckpt = torch.load(
            fold_dir / "best_validation.pt", map_location=device, weights_only=False
        )
        model_j.load_state_dict(reloaded_ckpt["model_state_dict"])
        model_j.eval()

        reloaded_eval = evaluate_model_j(model_j, resolver, val_keys)
        j_reloaded_f1 = reloaded_eval["final_f1"]
        delta_vs_a_pbnew = j_reloaded_f1 - a_pbnew_f1
        delta_vs_j_f2 = j_reloaded_f1 - HISTORICAL_J_F2_SCORES[fold_name]

        assert abs(j_reloaded_f1 - best_val_f1) < 1e-5, (
            f"Reload F1 {j_reloaded_f1} != Best Val F1 {best_val_f1}!"
        )
        best_ckpt_sha = sha256_file(fold_dir / "best_validation.pt")
        log(f"[{fold_name.upper()}] CHECKPOINT_RELOAD_PARITY = PASS")
        log(f"[{fold_name.upper()}] CHECKPOINT_SHA256 = {best_ckpt_sha}")

        manifest_data = {
            "fold": fold_name,
            "architecture": MODEL_J_ARCHITECTURE_VERSION,
            "teacher": "E_J_FIXED50",
            "base_checkpoint_sha256": base_sha,
            "best_validation_sha256": best_ckpt_sha,
            "best_epoch": best_epoch,
            "replaces_step0": replaces_step0,
            "a_f2": A_F2_SCORES[fold_name],
            "a_pbnew": a_pbnew_f1,
            "j_f2": HISTORICAL_J_F2_SCORES[fold_name],
            "j_pbnew": j_reloaded_f1,
            "delta_vs_a_pbnew": delta_vs_a_pbnew,
            "delta_vs_j_f2": delta_vs_j_f2,
            "base_weight_parity": "PASS" if weight_parity else "FAIL",
            "base_bn_parity": "PASS" if bn_parity else "FAIL",
            "reloaded_parity": "PASS",
            "per_class_f1": {
                name: float(reloaded_eval["per_class_j"][i])
                for i, name in enumerate(CANONICAL_BEHAVIORS)
            },
        }

        with open(fold_dir / "fold_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        results_summary[fold_name] = manifest_data
        log(
            f"[{fold_name.upper()}] COMPLETE: J_PBNEW={j_reloaded_f1:.6f} "
            f"(DeltaVsJF2={delta_vs_j_f2:+.6f}, DeltaVsAPBNew={delta_vs_a_pbnew:+.6f})"
        )

    # 5-Fold Summary
    log("\n" + "=" * 60)
    log("ALL 5 FOLDS COMPLETED — GENERATING AGGREGATE SUMMARY")
    log("=" * 60)

    j_pbnew_scores = [results_summary[f]["j_pbnew"] for f in FOLDS]
    a_pbnew_scores = [results_summary[f]["a_pbnew"] for f in FOLDS]

    j_pbnew_mean = float(np.mean(j_pbnew_scores))
    j_pbnew_std = float(np.std(j_pbnew_scores))
    a_pbnew_mean = float(np.mean(a_pbnew_scores))
    delta_j_pbnew_vs_jf2 = j_pbnew_mean - J_F2_MEAN

    positive_or_tied = sum(
        1 for f in FOLDS if results_summary[f]["j_pbnew"] >= HISTORICAL_J_F2_SCORES[f]
    )
    promotion_pass = j_pbnew_mean > J_F2_MEAN and positive_or_tied >= 3
    promotion_status = "PASS" if promotion_pass else "FAIL"

    log(f"J_F2_MEAN              = {J_F2_MEAN:.6f}")
    log(f"A_PBNEW_5FOLD_MEAN     = {a_pbnew_mean:.6f}")
    log(f"J_PBNEW_5FOLD_MEAN     = {j_pbnew_mean:.6f}")
    log(f"J_PBNEW_5FOLD_STD      = {j_pbnew_std:.6f}")
    log(f"DELTA_J_PBNEW_VS_J_F2  = {delta_j_pbnew_vs_jf2:+.6f}")
    log(f"POSITIVE_OR_TIED_FOLDS = {positive_or_tied}/5")
    log(f"PB_NEW_PROMOTION       = {promotion_status}")

    best_single_model = "J_PBNEW" if promotion_pass else "J_F2"
    best_single_f1 = j_pbnew_mean if promotion_pass else J_F2_MEAN
    log(f"BEST_SINGLE_MODEL      = {best_single_model}")
    log(f"BEST_SINGLE_MODEL_F1   = {best_single_f1:.6f}")

    # Summary JSON
    summary_file = args.output_dir.parent / "summary.json"
    summary_data = {
        "task_id": "FINAL-PBNEW-EJ-TEACHER-20260916",
        "j_f2_mean": J_F2_MEAN,
        "a_pbnew_5fold_mean": a_pbnew_mean,
        "j_pbnew_5fold_mean": j_pbnew_mean,
        "j_pbnew_5fold_std": j_pbnew_std,
        "delta_j_pbnew_vs_j_f2": delta_j_pbnew_vs_jf2,
        "positive_or_tied_folds": positive_or_tied,
        "pb_new_promotion": promotion_status,
        "best_single_model": best_single_model,
        "best_single_model_f1": best_single_f1,
        "current_best_system": "E_J_FIXED50",
        "current_best_system_f1": 0.693905,
        "outer_test_evaluations": 0,
        "folds": results_summary,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log(f"\nSummary saved to: {summary_file}")
    log("MODEL J PB_NEW 5-FOLD CAMPAIGN COMPLETE.")


if __name__ == "__main__":
    main()
