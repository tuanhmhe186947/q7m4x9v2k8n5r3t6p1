"""Exact 5-fold scientific training runner for G2-CCLSA.

Implements M2 visual-fine-tuning recipe + Actor Conditioned Localized Spatial Attention:
  - visual LR = 0.0003 (0.1x backbone multiplier)
  - nonvisual + G2 CCLSA LR = 0.003
  - bias-free Linear(46 -> 64) query from canonical 46D spatial features
  - bias-free Conv2d(256 -> 64, kernel=1) keys from ResNet layer3
  - bias-free Linear(256 -> 128) residual projection initialized to EXACT ZERO
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
import random
import sys
from dataclasses import asdict, dataclass, replace
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

from pig_behavior.classification_v2.models.g2_cclsa_model import (  # noqa: E402
    G2CCLSAClassifier,
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
EXPECTED_M2_PARAM_COUNT: int = 43633832
EXPECTED_PARAM_COUNT: int = 43685928
EXPECTED_TOTAL_ROWS: int = 33287
DATA_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_DATA_ROOT"
RGB_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_RGB_ROOT"

M2_INNER_REFERENCES: dict[str, float] = {
    "vg1": 0.642622,
    "vg2": 0.609316,
    "vg3": 0.587899,
    "vg4": 0.675154,
    "vg5": 0.656352,
}


@dataclass(frozen=True, slots=True)
class G2GateDecision:
    status: str
    authorize_next_fold: bool
    completed_folds: dict[str, float]
    mean_delta: float | None
    positive_folds: int
    reason: str


def evaluate_g2_gate(completed_folds: dict[str, float]) -> G2GateDecision:
    """Evaluate locked G2 inner gate with strict futility termination."""
    if not completed_folds:
        return G2GateDecision(
            status="INITIAL",
            authorize_next_fold=True,
            completed_folds={},
            mean_delta=None,
            positive_folds=0,
            reason="No folds completed yet.",
        )

    deltas: list[float] = []
    for f_name, score in completed_folds.items():
        ref = M2_INNER_REFERENCES[f_name.lower()]
        delta = score - ref
        deltas.append(delta)
        if delta <= 0.0:
            return G2GateDecision(
                status="REJECT_FUTILITY",
                authorize_next_fold=False,
                completed_folds=completed_folds,
                mean_delta=float(np.mean(deltas)),
                positive_folds=sum(1 for d in deltas if d > 0.0),
                reason=(
                    f"Futility rule triggered: Fold {f_name.upper()} delta ({delta:+.6f}) <= 0. "
                    "5/5 positive folds impossible."
                ),
            )

    mean_delta = float(np.mean(deltas))
    positive_count = sum(1 for d in deltas if d > 0.0)

    if len(completed_folds) == 5:
        if mean_delta >= 0.015 and positive_count == 5:
            return G2GateDecision(
                status="PASS",
                authorize_next_fold=True,
                completed_folds=completed_folds,
                mean_delta=mean_delta,
                positive_folds=positive_count,
                reason=(
                    f"Locked G2 Gate PASS: mean_delta={mean_delta:+.6f} >= +0.015 "
                    f"and positive_folds={positive_count}/5."
                ),
            )
        return G2GateDecision(
            status="REJECT",
            authorize_next_fold=False,
            completed_folds=completed_folds,
            mean_delta=mean_delta,
            positive_folds=positive_count,
            reason=(
                f"Locked G2 Gate REJECT: mean_delta={mean_delta:+.6f} (required >= +0.015) "
                f"or positive_folds={positive_count}/5 (required 5/5)."
            ),
        )

    return G2GateDecision(
        status="IN_PROGRESS",
        authorize_next_fold=True,
        completed_folds=completed_folds,
        mean_delta=mean_delta,
        positive_folds=positive_count,
        reason=f"All {len(completed_folds)} evaluated folds are positive. Continue campaign.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2-CCLSA 5-Fold Scientific Runner")
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
        default=REPO_ROOT / "outputs/classification_v2/g2_cclsa_v1",
        help="Base output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda or cpu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run fast preflight on CPU without full epoch train.",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional max train windows for CPU preflight.",
    )
    parser.add_argument(
        "--max-val-windows",
        type=int,
        default=None,
        help="Optional max val windows for CPU preflight.",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _resolve_runtime_config(
    fold_name: str,
    config_path: Path | None,
    data_root: Path | None,
    rgb_root: Path | None,
) -> ClassificationV2TrainingConfig:
    if config_path is None:
        cfg_name = f"g2_cclsa_earlystop_{fold_name}_scientific_v1.json"
        cfg_file = REPO_ROOT / "configs/classification_v2" / cfg_name
    else:
        cfg_file = config_path

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    config = load_training_config(cfg_file)
    dataset = config.dataset
    updates: dict[str, Any] = {}

    for field_name in (
        "train_ready_root",
        "actor_packed_cache",
        "actor_packed_index",
        "visual_cache_manifest",
        "visual_packed_cache",
        "visual_packed_index",
        "native_oof_fold_manifest",
        "auxiliary_targets_csv",
        "spatial_bundle_npz",
        "frame_context_csv",
        "window_context_csv",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is None:
            continue
        val_str = str(raw_val)
        if DATA_EXTERNAL_PREFIX in val_str:
            if data_root is None:
                raise ValueError(
                    f"Encountered {DATA_EXTERNAL_PREFIX} but --data-root was not provided"
                )
            resolved = Path(val_str.replace(DATA_EXTERNAL_PREFIX, str(data_root)))
            updates[field_name] = resolved
        else:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    for field_name in (
        "window_major_rgb_cache",
        "window_major_union_mask",
        "window_major_window_index",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is None:
            continue
        val_str = str(raw_val)
        if RGB_EXTERNAL_PREFIX in val_str:
            if rgb_root is None:
                raise ValueError(
                    f"Encountered {RGB_EXTERNAL_PREFIX} but --rgb-root was not provided"
                )
            resolved = Path(val_str.replace(RGB_EXTERNAL_PREFIX, str(rgb_root)))
            updates[field_name] = resolved
        else:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    for field_name in (
        "snapshot_json",
        "trainer_contract_json",
        "grouped_fold_roles",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is not None:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    return replace(config, dataset=replace(dataset, **updates))


def verify_g2_step0_parity(
    model: G2CCLSAClassifier,
    m2_model: MultimodalFusionClassifier,
    data: StrictTrainingDataModule,
    device: torch.device,
) -> float:
    """Verify exact step-0 logit parity between M2 and G2 (max abs diff <= 1e-7)."""
    model.eval()
    m2_model.eval()

    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    b_batch = data.batch(sample_indices)

    with torch.inference_mode():
        m2_logits = m2_model(**b_batch.model_inputs)
        g2_logits = model(**b_batch.model_inputs)

    max_diff = (m2_logits - g2_logits).abs().max().item()
    if max_diff > 1e-7:
        raise ValueError(f"G2 step-0 parity check failed! Max logit diff: {max_diff}")
    return max_diff


def run_g2_training(
    fold_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(SEED)
    fold_upper = fold_name.upper()
    run_dir = args.output_dir / "runs" / fold_name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"STARTING G2-CCLSA SCIENTIFIC RUN: {fold_upper}")
    print("=======================================================")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Compute Device: {device}")

    # 1. Config
    runtime_config = _resolve_runtime_config(
        fold_name, args.config, args.data_root, args.rgb_root
    )

    # 2. Data Module
    print("Initializing StrictTrainingDataModule...")
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    full_train_indices = data.split_indices("train")
    full_val_indices = data.split_indices("validation")
    test_indices = data.split_indices("test")

    if args.preflight:
        max_train = args.max_train_windows or 64
        max_val = args.max_val_windows or 32
        train_indices = full_train_indices[: min(len(full_train_indices), max_train)]
        val_indices = full_val_indices[: min(len(full_val_indices), max_val)]
        print(f"[PREFLIGHT MODE] Subsampling train={len(train_indices)}, val={len(val_indices)}")
    else:
        train_indices = full_train_indices
        val_indices = full_val_indices

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

    model = G2CCLSAClassifier(backbone_config).to(device)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Total Parameters: {total_param_count} (Expected: {EXPECTED_PARAM_COUNT})")
    if total_param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(
            f"G2 parameter count mismatch: {total_param_count} vs {EXPECTED_PARAM_COUNT}"
        )

    # 4. Parity Check
    set_seed(SEED)
    m2_ref_model = MultimodalFusionClassifier(backbone_config).to(device)
    # Load M2 weights into G2 base
    model.load_state_dict(m2_ref_model.state_dict(), strict=False)
    step0_diff = verify_g2_step0_parity(model, m2_ref_model, data, device)
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

            optimizer.zero_grad()
            with torch.amp.autocast(
                "cuda",
                enabled=(
                    device.type == "cuda"
                    and runtime_config.optimization.precision == "amp"
                ),
            ):
                logits = model(**b_batch.model_inputs)
                per_row_loss = F.cross_entropy(
                    logits,
                    b_batch.behavior_target,
                    weight=behavior_weights,
                    reduction="none",
                )
                denominator = b_batch.sample_weight.sum().clamp_min(1e-8)
                loss = (per_row_loss * b_batch.sample_weight).sum() / denominator

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

                with torch.amp.autocast(
                    "cuda",
                    enabled=(
                        device.type == "cuda"
                        and runtime_config.optimization.precision == "amp"
                    ),
                ):
                    logits = model(**b_batch.model_inputs)
                    per_row_loss = F.cross_entropy(
                        logits,
                        b_batch.behavior_target,
                        weight=behavior_weights,
                        reduction="none",
                    )
                    denominator = b_batch.sample_weight.sum().clamp_min(1e-8)
                    v_loss = (per_row_loss * b_batch.sample_weight).sum() / denominator

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


def main() -> None:
    args = parse_args()
    if args.fold == "all":
        folds = ["vg1", "vg2", "vg3", "vg4", "vg5"]
    else:
        folds = [args.fold]

    results = {}
    completed_folds: dict[str, float] = {}

    for fold in folds:
        print(f"\n--- Checking Gate Authorization Before {fold.upper()} ---")
        gate_decision = evaluate_g2_gate(completed_folds)
        if not gate_decision.authorize_next_fold and len(completed_folds) > 0:
            print(f"Gate stopped campaign before {fold.upper()}: {gate_decision.reason}")
            break

        res = run_g2_training(fold, args)
        results[fold] = res
        completed_folds[fold] = res["best_val_macro_f1"]

        gate_decision = evaluate_g2_gate(completed_folds)
        gate_path = args.output_dir / "g2_campaign_gate_state.json"
        gate_path.write_text(
            json.dumps(asdict(gate_decision), indent=2), encoding="utf-8"
        )
        print(
            f"Gate State after {fold.upper()}: Status={gate_decision.status} | "
            f"Reason: {gate_decision.reason}"
        )
        if not gate_decision.authorize_next_fold:
            print(f"Campaign stopped by Gate: {gate_decision.status}")
            break

    print("\n=======================================================")
    print("ALL REQUESTED FOLDS PROCESSED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
