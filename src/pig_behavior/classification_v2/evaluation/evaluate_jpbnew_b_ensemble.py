"""Evaluation runner for Task FINAL-EJ-PBNEW-FIXED50-20260916.

Evaluates the newly trained physical J_PBNEW single model on inner validation folds,
verifies single-model reload parity, aligns samples strictly against the authoritative
saved Model B sample_ids.npy and targets.npy, and computes the fixed 50/50 ensemble:
    E_J_PBNEW = 0.50 * J_PBNEW_logits + 0.50 * B_authority_logits
    prediction = argmax(E_J_PBNEW_logits)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

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

FOLDS = ["vg1", "vg2", "vg3", "vg4", "vg5"]

EXPECTED_J_PBNEW_HASHES = {
    "vg1": "fb80ef77aa6bc1f09614143756cbb10680612ffaeae91cac91fc8952e3610db2",
    "vg2": "8a238f94297980e3a85f6beb9d3930bd0a709f9cfdd81249ba33ebbe5ada3059",
    "vg3": "2a13c34c3dd6c0071c9014323e671aca7ff260f32399326514cb3dbf7b9f9053",
    "vg4": "67c67826be2383469125977e26d32165a2c21ae394fc5917be2ca0da44a8ce9a",
    "vg5": "32ee81a904f7eaeb194b25fe0b98c6d5be6703bd3c5a7a54b65d0802558b8e18",
}

EXPECTED_J_PBNEW_F1 = {
    "vg1": 0.709683,
    "vg2": 0.700285,
    "vg3": 0.644606,
    "vg4": 0.678462,
    "vg5": 0.690816,
}
J_PBNEW_EXPECTED_MEAN = 0.684770

OLD_EJ_SCORES = {
    "vg1": 0.719035,
    "vg2": 0.714796,
    "vg3": 0.627833,
    "vg4": 0.699707,
    "vg5": 0.708154,
}
OLD_EJ_MEAN = 0.693905

PINNED_B_HASHES = {
    "vg1": {
        "sample_ids.npy": "ef0739a4737343d50329f016fb5934ff289938ad7f977b73aeef47ac5931b36f",
        "targets.npy": "eb624777c9cd4f2d2a7d335e4ec5668782d8b780914d3a6bc2a32b5125986f87",
        "b_logits.npy": "29d7e439fd7832a1577e9024ad634124137fc415229dc1c9bd7c2724629a88a9",
    },
    "vg2": {
        "sample_ids.npy": "4549943bed294389e14a4dc460a63d7a95a3cb5f661864f89482c8b52ca4c012",
        "targets.npy": "310d706b276c7edd682578a4af2fa458a1be169e81071e0494d2fb09b989d767",
        "b_logits.npy": "185923504ec325e6068e2dae98a906359239bec4ee37a22989a3ad5cc0d5f618",
    },
    "vg3": {
        "sample_ids.npy": "966c6df80409c117227fb26b10c16a2c41cbff0a429391279233f37d26e87752",
        "targets.npy": "de7d822deb10aa2a53070fb072fe11a03bcddf75ed97b8cbdc7124050825ce64",
        "b_logits.npy": "a072c046abd189331219a8f13fa56abb06bd61a9d1b75029b4f6217bc0cebc65",
    },
    "vg4": {
        "sample_ids.npy": "bac4c664e3df64c98cbb71b24803805c256d0a17c655f10cca991767ae89d579",
        "targets.npy": "65a79c927a5e62abedf275c621ab95e7cabd274914c88b09282c2cfdbc397889",
        "b_logits.npy": "8b32ea3675c5cea8f0a34c39df87cb65951dc2b378a8d1e3f7f1e11d9a84b94f",
    },
    "vg5": {
        "sample_ids.npy": "81c21ffb3be7cf5955387e38bed48fa43f3a876757850ae65014b92877d85dc7",
        "targets.npy": "f7ec5706b9070118c298f5543162690eb1bdf75e63cf09eb0cc6d7d7e68fb094",
        "b_logits.npy": "6ae05547cd0caacb8860d969cf2f6d37f85b803538555e6de034cc835eb23191",
    },
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

DEFAULT_J_RUNS_DIR = Path(
    "outputs/classification_v2/final_model_j_pbnew_ejteacher_v1/runs"
)
DEFAULT_PB_DIR = Path("outputs/classification_v2/pb_teacher_ej_fixed50_v1")
DEFAULT_B_AUTHORITY_DIR = Path(
    "outputs/classification_v2/final_model_j_replay_v1/ensemble_with_b_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/classification_v2/final_model_j_pbnew_ejteacher_v1/ensemble_with_b_fixed50_v1"
)
DEFAULT_BASE_MODEL_A_DIR = Path(
    "outputs/classification_v2/final_high_ceiling_v1/runs"
)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open("live_progress_jpbnew_ensemble.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_base_model(
    fold_name: str,
    data: StrictTrainingDataModule,
    runtime_config: Any,
    device: torch.device,
    base_model_dir: Path = DEFAULT_BASE_MODEL_A_DIR,
) -> DeepLocalJointRepresentationClassifier:
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
    base_ckpt_path = base_model_dir / fold_name / "best_validation.pt"
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    return base_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate J_PBNEW + Model B Fixed 50/50 Ensemble."
    )
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--j-runs-dir", type=Path, default=DEFAULT_J_RUNS_DIR)
    parser.add_argument("--pb-dir", type=Path, default=DEFAULT_PB_DIR)
    parser.add_argument(
        "--b-authority-dir", type=Path, default=DEFAULT_B_AUTHORITY_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_A_DIR
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 60)
    log("TASK: FINAL FIXED ENSEMBLE - J_PBNEW + EXISTING B AUTHORITY LOGITS")
    log("TASK_ID: FINAL-EJ-PBNEW-FIXED50-20260916")
    log("=" * 60)
    log(f"Compute Device: {device}")
    if device.type == "cuda":
        log(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    log("B_LOGIT_AUTHORITY = EXISTING_EJ_FIXED50_ARTIFACTS")
    log("OUTER_TEST_EVALUATIONS = 0")
    log(f"J_RUNS_DIR = {args.j_runs_dir}")
    log(f"PB_DIR = {args.pb_dir}")
    log(f"B_AUTHORITY_DIR = {args.b_authority_dir}")
    log(f"OUTPUT_DIR = {args.output_dir}")

    # 1. VERIFY J_PBNEW CHECKPOINT HASHES
    log("\n[STAGE 1] Verifying physical J_PBNEW checkpoints...")
    for fold in args.folds:
        ckpt_path = args.j_runs_dir / fold / "best_validation.pt"
        assert ckpt_path.is_file(), f"Missing J_PBNEW checkpoint: {ckpt_path}"
        sha = sha256_file(ckpt_path)
        expected = EXPECTED_J_PBNEW_HASHES[fold]
        log(f"[{fold.upper()}] J_PBNEW SHA256 = {sha}")
        assert sha == expected, (
            f"SHA256 mismatch for {fold}: {sha} != {expected}"
        )
    log("ALL 5 J_PBNEW CHECKPOINTS VERIFIED: PASS")

    # 2. VERIFY B-AUTHORITY ARTIFACT HASHES
    log("\n[STAGE 2] Verifying pinned B-authority artifacts (Correction 1)...")
    for fold in args.folds:
        b_fold_dir = args.b_authority_dir / fold
        for fname in ["sample_ids.npy", "targets.npy", "b_logits.npy"]:
            fpath = b_fold_dir / fname
            assert fpath.is_file(), f"Missing B-authority file: {fpath}"
            f_sha = sha256_file(fpath)
            expected_f_sha = PINNED_B_HASHES[fold][fname]
            log(f"[{fold.upper()}] {fname} SHA256 = {f_sha}")
            assert f_sha == expected_f_sha, (
                f"B-authority SHA256 mismatch on {fold} {fname}: "
                f"{f_sha} != {expected_f_sha}"
            )
    log("ALL PINNED B-AUTHORITY ARTIFACTS VERIFIED: PASS")

    # Storage for fold evaluation
    fold_results: dict[str, dict[str, Any]] = {}

    # 3. EVALUATION ACROSS FOLDS
    log("\n" + "=" * 60)
    log("STAGE 3: EVALUATING J_PBNEW & COMPUTING FIXED ENSEMBLE WITH B")
    log("=" * 60)

    for fold_name in args.folds:
        log(f"\n--- Processing Fold {fold_name.upper()} ---")
        runtime_config = _resolve_runtime_config(fold_name, None, None, None)
        data = StrictTrainingDataModule(runtime_config, device=device)
        data.fit_fold_preprocessor()

        full_val_indices = data.split_indices("validation")
        pb_cache_path = (
            args.pb_dir / fold_name.lower() / "pb_teacher_features.pt"
        )
        assert pb_cache_path.exists(), (
            f"Missing PB_NEW cache: {pb_cache_path}"
        )
        pb_cache = torch.load(
            pb_cache_path, map_location="cpu", weights_only=False
        )

        posture_sidecar = torch.load(
            DEFAULT_POSTURE_SIDECAR, map_location="cpu", weights_only=False
        )
        h5_cache = torch.load(
            DEFAULT_H5_STRUCTURED_PATH, map_location="cpu", weights_only=False
        )
        data_plus_store = DataPlusMultimodalStore(
            DataPlusSidecarPaths.from_root(
                DEFAULT_DATA_PLUS_DIR, pb_path=pb_cache_path
            )
        )
        resolver = JointKeyedBatchResolver(
            data, pb_cache, posture_sidecar, h5_cache, data_plus_store, device
        )
        val_keys = list(resolver.canonical_keys(full_val_indices))
        log(
            f"[{fold_name.upper()}] Extracted {len(val_keys)} validation keys"
        )

        # Load authoritative B artifacts
        b_fold_dir = args.b_authority_dir / fold_name
        b_sample_ids = np.load(b_fold_dir / "sample_ids.npy")
        b_targets = np.load(b_fold_dir / "targets.npy")
        b_logits = np.load(b_fold_dir / "b_logits.npy")

        log(
            f"[{fold_name.upper()}] Authoritative samples count = "
            f"{len(b_sample_ids)}"
        )
        assert len(val_keys) == len(b_sample_ids), (
            f"Sample count mismatch: {len(val_keys)} vs {len(b_sample_ids)}"
        )

        # Load physical Model J checkpoint
        base_model = _load_base_model(
            fold_name,
            data,
            runtime_config,
            device,
            base_model_dir=args.base_model_dir,
        )
        model_j = NearFinalModelJ(
            base_model, token_dim=32, num_classes=10
        ).to(device)
        j_ckpt_path = args.j_runs_dir / fold_name / "best_validation.pt"
        j_ckpt = torch.load(
            j_ckpt_path, map_location=device, weights_only=False
        )
        model_j.load_state_dict(j_ckpt["model_state_dict"])
        model_j.eval()

        # Batch evaluation
        all_j_logits = []
        all_targets = []
        all_sample_ids = []

        num_batches = int(np.ceil(len(val_keys) / args.batch_size))
        with torch.inference_mode():
            for b_idx in range(num_batches):
                b_keys = tuple(
                    val_keys[
                        b_idx
                        * args.batch_size : (b_idx + 1)
                        * args.batch_size
                    ]
                )
                aligned = resolver.batch(b_keys)

                final_logits, _, _, _, _ = model_j(aligned)
                all_j_logits.append(final_logits.cpu().numpy())
                all_targets.append(
                    aligned.training_batch.behavior_target.cpu().numpy()
                )
                all_sample_ids.extend(list(b_keys))

        raw_j_logits = np.concatenate(all_j_logits, axis=0)
        raw_targets = np.concatenate(all_targets, axis=0)
        raw_sample_ids = np.array(all_sample_ids)

        # Align strictly by explicit sample ID
        j_dict = {
            sid: (raw_j_logits[i], raw_targets[i])
            for i, sid in enumerate(raw_sample_ids)
        }
        aligned_j_logits = []
        aligned_targets = []
        for sid in b_sample_ids:
            assert sid in j_dict, f"Sample ID {sid} missing from evaluation!"
            lgt, tgt = j_dict[sid]
            aligned_j_logits.append(lgt)
            aligned_targets.append(tgt)

        j_logits = np.array(aligned_j_logits, dtype=np.float32)
        targets = np.array(aligned_targets, dtype=np.int64)

        # Target parity verification
        targets_match = np.array_equal(targets, b_targets)
        log(
            f"[{fold_name.upper()}] TARGETS_EXACT_MATCH = "
            f"{'PASS' if targets_match else 'FAIL'}"
        )
        assert targets_match, (
            f"Targets mismatch between evaluation and saved B targets on {fold_name}!"
        )

        # Single-model reload parity verification
        j_preds = j_logits.argmax(axis=-1)
        j_f1 = float(
            f1_score(targets, j_preds, average="macro", zero_division=0)
        )
        expected_f1 = EXPECTED_J_PBNEW_F1[fold_name]
        f1_diff = abs(j_f1 - expected_f1)
        log(
            f"[{fold_name.upper()}] J_PBNEW_F1 = {j_f1:.6f} "
            f"(Expected={expected_f1:.6f}, Diff={f1_diff:.2e})"
        )
        assert f1_diff < 1e-5, (
            f"J_PBNEW reload parity failed on {fold_name}: {j_f1} vs {expected_f1}"
        )

        # Model B single model score check
        b_preds = b_logits.argmax(axis=-1)
        b_f1 = float(
            f1_score(targets, b_preds, average="macro", zero_division=0)
        )
        log(f"[{fold_name.upper()}] B_AUTHORITY_F1 = {b_f1:.6f}")

        # Fixed 50/50 Ensemble: E_J_PBNEW = 0.50 * J_PBNEW + 0.50 * B_authority
        ej_pbnew_logits = 0.50 * j_logits + 0.50 * b_logits
        ej_pbnew_preds = ej_pbnew_logits.argmax(axis=-1)
        ej_pbnew_f1 = float(
            f1_score(targets, ej_pbnew_preds, average="macro", zero_division=0)
        )
        ej_pbnew_acc = float(accuracy_score(targets, ej_pbnew_preds))

        old_ej_score = OLD_EJ_SCORES[fold_name]
        delta_vs_old_ej = ej_pbnew_f1 - old_ej_score
        delta_vs_j_pbnew = ej_pbnew_f1 - j_f1

        log(
            f"[{fold_name.upper()}] E_J_PBNEW_F1 = {ej_pbnew_f1:.6f} | "
            f"OLD_EJ_F1 = {old_ej_score:.6f} | "
            f"DeltaVsOldEJ = {delta_vs_old_ej:+.6f} | "
            f"DeltaVsJPBNew = {delta_vs_j_pbnew:+.6f}"
        )

        # Complementarity Analysis relative to J_PBNEW
        j_correct = j_preds == targets
        b_correct = b_preds == targets
        ens_correct = ej_pbnew_preds == targets

        fixed_j_errors = int(np.sum((~j_correct) & ens_correct))
        broken_j_correct = int(np.sum(j_correct & (~ens_correct)))
        net_corrections_vs_j = fixed_j_errors - broken_j_correct

        j_c_b_c = int(np.sum(j_correct & b_correct))
        j_c_b_w = int(np.sum(j_correct & (~b_correct)))
        j_w_b_c = int(np.sum((~j_correct) & b_correct))
        j_w_b_w = int(np.sum((~j_correct) & (~b_correct)))

        log(
            f"[{fold_name.upper()}] Complementarity: FixedErrors={fixed_j_errors}, "
            f"BrokenCorrect={broken_j_correct}, NetCorrections={net_corrections_vs_j:+d}"
        )

        # Per-class F1
        _, _, j_per_class, _ = precision_recall_fscore_support(
            targets, j_preds, labels=list(range(10)), zero_division=0
        )
        _, _, b_per_class, _ = precision_recall_fscore_support(
            targets, b_preds, labels=list(range(10)), zero_division=0
        )
        _, _, ens_per_class, _ = precision_recall_fscore_support(
            targets, ej_pbnew_preds, labels=list(range(10)), zero_division=0
        )

        per_class_dict = {}
        for c_idx, c_name in enumerate(CANONICAL_BEHAVIORS):
            c_mask = targets == c_idx
            c_fixed = int(np.sum((~j_correct) & ens_correct & c_mask))
            c_broken = int(np.sum(j_correct & (~ens_correct) & c_mask))
            per_class_dict[c_name] = {
                "fixed": c_fixed,
                "broken": c_broken,
                "net": c_fixed - c_broken,
                "j_pbnew_f1": float(j_per_class[c_idx]),
                "b_authority_f1": float(b_per_class[c_idx]),
                "e_j_pbnew_f1": float(ens_per_class[c_idx]),
                "delta_vs_j_pbnew": float(
                    ens_per_class[c_idx] - j_per_class[c_idx]
                ),
            }

        # Save Fold Artifacts
        fold_out_dir = args.output_dir / fold_name
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        np.save(fold_out_dir / "sample_ids.npy", b_sample_ids)
        np.save(fold_out_dir / "targets.npy", targets)
        np.save(fold_out_dir / "j_pbnew_logits.npy", j_logits)
        np.save(fold_out_dir / "b_authority_logits.npy", b_logits)
        np.save(fold_out_dir / "ensemble_logits.npy", ej_pbnew_logits)

        pd.DataFrame(
            {"sample_id": b_sample_ids, "prediction": ej_pbnew_preds}
        ).to_csv(fold_out_dir / "predictions.csv", index=False)

        fold_metrics = {
            "fold": fold_name,
            "sample_count": len(b_sample_ids),
            "j_pbnew_f1": j_f1,
            "b_authority_f1": b_f1,
            "e_j_pbnew_f1": ej_pbnew_f1,
            "e_j_pbnew_acc": ej_pbnew_acc,
            "old_ej_f1": old_ej_score,
            "delta_vs_old_ej": delta_vs_old_ej,
            "delta_vs_j_pbnew": delta_vs_j_pbnew,
            "j_correct_b_correct": j_c_b_c,
            "j_correct_b_wrong": j_c_b_w,
            "j_wrong_b_correct": j_w_b_c,
            "j_wrong_b_wrong": j_w_b_w,
            "fixed_errors": fixed_j_errors,
            "broken_correct": broken_j_correct,
            "net_corrections": net_corrections_vs_j,
            "per_class": per_class_dict,
        }
        with open(fold_out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2)

        fold_results[fold_name] = fold_metrics

    # 4. AGGREGATE SUMMARY
    log("\n" + "=" * 60)
    log("ALL 5 FOLDS COMPLETE — AGGREGATING 5-FOLD ENSEMBLE METRICS")
    log("=" * 60)

    ej_pbnew_scores = [fold_results[f]["e_j_pbnew_f1"] for f in FOLDS]
    j_pbnew_scores = [fold_results[f]["j_pbnew_f1"] for f in FOLDS]
    b_scores = [fold_results[f]["b_authority_f1"] for f in FOLDS]

    ej_pbnew_mean = float(np.mean(ej_pbnew_scores))
    ej_pbnew_std = float(np.std(ej_pbnew_scores, ddof=0))
    j_pbnew_mean = float(np.mean(j_pbnew_scores))
    b_mean = float(np.mean(b_scores))

    delta_vs_old_ej = ej_pbnew_mean - OLD_EJ_MEAN
    positive_or_tied = sum(
        1
        for f in FOLDS
        if fold_results[f]["e_j_pbnew_f1"] >= OLD_EJ_SCORES[f]
    )

    total_fixed = sum(fold_results[f]["fixed_errors"] for f in FOLDS)
    total_broken = sum(fold_results[f]["broken_correct"] for f in FOLDS)
    total_net = total_fixed - total_broken

    promotion_pass = (
        ej_pbnew_mean > OLD_EJ_MEAN and positive_or_tied >= 3
    )
    promotion_status = "PASS" if promotion_pass else "FAIL"

    best_overall_system = "E_J_PBNEW" if promotion_pass else "E_J_FIXED50"
    best_overall_f1 = ej_pbnew_mean if promotion_pass else OLD_EJ_MEAN

    log(f"J_PBNEW_MEAN            = {j_pbnew_mean:.6f}")
    log(f"B_AUTHORITY_MEAN        = {b_mean:.6f}")
    log(f"OLD_EJ_MEAN             = {OLD_EJ_MEAN:.6f}")
    log(f"E_J_PBNEW_MEAN          = {ej_pbnew_mean:.6f}")
    log(f"E_J_PBNEW_STD           = {ej_pbnew_std:.6f}")
    log(f"DELTA_VS_OLD_EJ         = {delta_vs_old_ej:+.6f}")
    log(f"POSITIVE_OR_TIED_FOLDS  = {positive_or_tied}/5")
    log(f"TOTAL_FIXED_ERRORS      = {total_fixed}")
    log(f"TOTAL_BROKEN_CORRECT    = {total_broken}")
    log(f"TOTAL_NET_CORRECTIONS   = {total_net:+d}")
    log(f"E_J_PBNEW_PROMOTION     = {promotion_status}")
    log(f"BEST_OVERALL_SYSTEM     = {best_overall_system}")
    log(f"BEST_OVERALL_F1         = {best_overall_f1:.6f}")

    # Save Tables
    # Complementarity by Fold
    comp_fold_rows = []
    for f in FOLDS:
        m = fold_results[f]
        comp_fold_rows.append({
            "fold": f,
            "sample_count": m["sample_count"],
            "j_pbnew_f1": m["j_pbnew_f1"],
            "b_authority_f1": m["b_authority_f1"],
            "e_j_pbnew_f1": m["e_j_pbnew_f1"],
            "old_ej_f1": m["old_ej_f1"],
            "delta_vs_old_ej": m["delta_vs_old_ej"],
            "fixed_errors": m["fixed_errors"],
            "broken_correct": m["broken_correct"],
            "net_corrections": m["net_corrections"],
        })
    pd.DataFrame(comp_fold_rows).to_csv(
        args.output_dir / "complementarity_by_fold.csv", index=False
    )

    # Per-Class F1 Table & Complementarity
    per_class_rows = []
    for c_idx, c_name in enumerate(CANONICAL_BEHAVIORS):
        j_cls = [
            fold_results[f]["per_class"][c_name]["j_pbnew_f1"]
            for f in FOLDS
        ]
        b_cls = [
            fold_results[f]["per_class"][c_name]["b_authority_f1"]
            for f in FOLDS
        ]
        ens_cls = [
            fold_results[f]["per_class"][c_name]["e_j_pbnew_f1"]
            for f in FOLDS
        ]
        fixed_cls = [
            fold_results[f]["per_class"][c_name]["fixed"] for f in FOLDS
        ]
        broken_cls = [
            fold_results[f]["per_class"][c_name]["broken"] for f in FOLDS
        ]

        j_m = float(np.mean(j_cls))
        b_m = float(np.mean(b_cls))
        ens_m = float(np.mean(ens_cls))

        per_class_rows.append({
            "class_index": c_idx,
            "behavior": c_name,
            "j_pbnew_f1": j_m,
            "b_authority_f1": b_m,
            "e_j_pbnew_f1": ens_m,
            "delta_vs_j_pbnew": ens_m - j_m,
            "total_fixed": sum(fixed_cls),
            "total_broken": sum(broken_cls),
            "net_corrections": sum(fixed_cls) - sum(broken_cls),
        })
    pd.DataFrame(per_class_rows).to_csv(
        args.output_dir / "per_class_f1.csv", index=False
    )
    pd.DataFrame(per_class_rows).to_csv(
        args.output_dir / "complementarity_by_class.csv", index=False
    )

    # Root Summary JSON
    summary_data = {
        "task_id": "FINAL-EJ-PBNEW-FIXED50-20260916",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture": MODEL_J_ARCHITECTURE_VERSION,
        "b_logit_authority": "EXISTING_EJ_FIXED50_ARTIFACTS",
        "sample_alignment": "PASS",
        "j_pbnew_reload_parity": "PASS",
        "outer_test_evaluations": 0,
        "old_ej_mean": OLD_EJ_MEAN,
        "j_pbnew_mean": j_pbnew_mean,
        "b_authority_mean": b_mean,
        "e_j_pbnew_mean": ej_pbnew_mean,
        "e_j_pbnew_std": ej_pbnew_std,
        "delta_vs_old_ej": delta_vs_old_ej,
        "positive_or_tied_folds": positive_or_tied,
        "total_fixed_errors": total_fixed,
        "total_broken_correct": total_broken,
        "total_net_corrections": total_net,
        "e_j_pbnew_promotion": promotion_status,
        "best_single": "J_PBNEW",
        "best_single_f1": j_pbnew_mean,
        "best_overall_system": best_overall_system,
        "best_overall_f1": best_overall_f1,
        "ensemble_weight_search": "CLOSED",
        "folds": fold_results,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log(f"\nSummary successfully written to: {args.output_dir / 'summary.json'}")
    log("EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
