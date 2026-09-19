"""5-Fold Factor Ablation Runner for Model J (Task FINAL-J-LOCAL-GATE-ABLATION-20260916).

Decomposes Model J via a 2x2 architectural factor matrix + neutral scaffold control:
- Config N: Model A + 10 Sources + Fixed Neutral Gate + Residual Head (Neutral Scaffold Control)
- Config L: Model A + 12 Sources + Fixed Neutral Gate + Residual Head (Local-Spatial Only)
- Config G: Model A + 10 Sources + Learned Class-Aware Gate + Residual Head (Gate Only)
- Config J: Full Physical Model J (Historical Replay Reference: Mean 0.684646)
- Config A: Canonical Model A (External Baseline: Mean 0.677789)
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from pig_behavior.classification_v2.models.model_j_ablation import (
    NearFinalModelG,
    NearFinalModelJ_Factorial,
    NearFinalModelL,
    NearFinalModelN,
    compute_module_subhashes,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
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
    DEFAULT_PB_DIR,
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

A_AUTHORITY_SCORES = {
    "vg1": 0.700872,
    "vg2": 0.694903,
    "vg3": 0.632632,
    "vg4": 0.680758,
    "vg5": 0.679783,
}
A_MEAN = 0.677789

J_AUTHORITY_SCORES = {
    "vg1": 0.709400,
    "vg2": 0.694903,
    "vg3": 0.633957,
    "vg4": 0.692737,
    "vg5": 0.692233,
}
J_MEAN = 0.684646

K_REFERENCE_SCORES = {
    "vg1": 0.709503,
    "vg2": 0.695357,
    "vg3": 0.638355,
    "vg4": 0.680758,
    "vg5": 0.687785,
}
K_MEAN = 0.682351

DEFAULT_BASE_MODEL_A_DIR = Path("outputs/classification_v2/final_high_ceiling_v1/runs")


def log(msg: str) -> None:
    print(msg, flush=True)
    with open("live_progress_ablation.txt", "a", encoding="utf-8") as f:
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


def get_model_hash_dict(model: nn.Module) -> dict[str, str]:
    hashes = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        hashes[name] = hashlib.sha256(arr.tobytes()).hexdigest()
    return hashes


def get_bn_state_dict(model: nn.Module) -> dict[str, str]:
    hashes = {}
    for name, buf in model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            arr = buf.detach().cpu().numpy()
            hashes[name] = hashlib.sha256(arr.tobytes()).hexdigest()
    return hashes


def build_ablation_model(
    config_name: str,
    base_model: DeepLocalJointRepresentationClassifier,
    token_dim: int = 32,
    num_classes: int = 10,
    base_seed: int = 240494961,
) -> nn.Module:
    """Instantiate the designated ablation model configuration."""
    cfg = config_name.upper()
    if cfg == "N":
        return NearFinalModelN(
            base_model,
            token_dim=token_dim,
            num_classes=num_classes,
            base_seed=base_seed,
        )
    elif cfg == "G":
        return NearFinalModelG(
            base_model,
            token_dim=token_dim,
            num_classes=num_classes,
            base_seed=base_seed,
        )
    elif cfg == "L":
        return NearFinalModelL(
            base_model,
            token_dim=token_dim,
            num_classes=num_classes,
            base_seed=base_seed,
        )
    elif cfg == "J":
        return NearFinalModelJ_Factorial(
            base_model,
            token_dim=token_dim,
            num_classes=num_classes,
            base_seed=base_seed,
        )
    else:
        raise ValueError(f"Unknown ablation configuration: {config_name}")


def _load_base_model(
    fold_name: str,
    data: StrictTrainingDataModule,
    runtime_config: Any,
    device: torch.device,
    base_model_dir: Path = DEFAULT_BASE_MODEL_A_DIR,
) -> tuple[DeepLocalJointRepresentationClassifier, str]:
    ckpt_path = base_model_dir / fold_name.lower() / "best_validation.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing base Model A checkpoint at {ckpt_path}")

    actual_sha = sha256_file(ckpt_path)
    expected_sha = EXPECTED_CONTROL_HASHES[fold_name.lower()]
    assert actual_sha == expected_sha, (
        f"Base Model A SHA mismatch on {fold_name}: {actual_sha} != {expected_sha}"
    )

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

    model = DeepLocalJointRepresentationClassifier(backbone_config).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    fixed_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    load_res = model.load_state_dict(fixed_state, strict=True)
    assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, actual_sha


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    resolver: JointKeyedBatchResolver,
    val_keys: list[str],
    key_to_video: dict[str, str],
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate ablation model, returning predictions, logits, targets, and video keys."""
    model.eval()
    all_final_logits = []
    all_base_logits = []
    all_targets = []
    all_sample_ids = []
    all_video_keys = []

    for i in range(0, len(val_keys), batch_size):
        b_keys = val_keys[i : i + batch_size]
        aligned = resolver.batch(b_keys)
        out = model(aligned)
        final_logits = out[0]
        base_logits = out[1]

        all_final_logits.append(final_logits.cpu().numpy())
        all_base_logits.append(base_logits.cpu().numpy())
        all_targets.append(aligned.training_batch.behavior_target.cpu().numpy())
        all_sample_ids.extend(b_keys)
        all_video_keys.extend([key_to_video.get(k, "UNKNOWN_VIDEO") for k in b_keys])

    final_logits_arr = np.concatenate(all_final_logits, axis=0)
    base_logits_arr = np.concatenate(all_base_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    sample_ids_arr = np.array(all_sample_ids)
    video_keys_arr = np.array(all_video_keys)

    final_preds = final_logits_arr.argmax(axis=-1)
    base_preds = base_logits_arr.argmax(axis=-1)

    macro_f1 = float(f1_score(targets_arr, final_preds, average="macro", zero_division=0))
    base_macro_f1 = float(f1_score(targets_arr, base_preds, average="macro", zero_division=0))
    accuracy = float((final_preds == targets_arr).mean())

    per_class_res = precision_recall_fscore_support(
        targets_arr, final_preds, labels=list(range(10)), zero_division=0
    )
    per_class_f1 = {
        CANONICAL_BEHAVIORS[c]: float(per_class_res[2][c]) for c in range(10)
    }

    # Group/Video level metrics
    df_eval = pd.DataFrame({
        "sample_id": sample_ids_arr,
        "video_key": video_keys_arr,
        "target": targets_arr,
        "prediction": final_preds,
    })
    group_rows = []
    for vk, group in df_eval.groupby("video_key"):
        g_acc = float((group["prediction"] == group["target"]).mean())
        g_f1 = float(
            f1_score(group["target"], group["prediction"], average="macro", zero_division=0)
        )
        group_rows.append({
            "video_key": vk,
            "sample_count": len(group),
            "accuracy": g_acc,
            "macro_f1": g_f1,
        })
    df_group = pd.DataFrame(group_rows)

    return {
        "macro_f1": macro_f1,
        "base_f1": base_macro_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "final_logits": final_logits_arr,
        "base_logits": base_logits_arr,
        "targets": targets_arr,
        "sample_ids": sample_ids_arr,
        "video_keys": video_keys_arr,
        "group_metrics_df": df_group,
    }


def verify_common_initialization_parity() -> bool:
    """Verify bit-for-bit initialization parity across N, G, L, and J submodules."""
    log("=== AUDITING COMMON SUBMODULE INITIALIZATION PARITY ===")
    from unittest.mock import MagicMock

    dummy_base = MagicMock()
    dummy_base.parameters.return_value = []

    m_n = NearFinalModelN(dummy_base, base_seed=SEED)
    m_g = NearFinalModelG(dummy_base, base_seed=SEED)
    m_l = NearFinalModelL(dummy_base, base_seed=SEED)
    m_j = NearFinalModelJ_Factorial(dummy_base, base_seed=SEED)

    hn = compute_module_subhashes(m_n.gate.source_projections)
    hg = compute_module_subhashes(m_g.gate.source_projections)
    hl = compute_module_subhashes(m_l.gate.source_projections)
    hj = compute_module_subhashes(m_j.gate.source_projections)

    n_vs_g = all(hn[k] == hg[k] for k in hn)
    n_vs_l = all(hn[k] == hl[k] for k in hn)
    l_vs_j = all(hl[k] == hj[k] for k in hl)
    g_vs_j = all(hg[k] == hj[k] for k in hg)

    hn_res = compute_module_subhashes(m_n.gate.residual_head)
    hg_res = compute_module_subhashes(m_g.gate.residual_head)
    hl_res = compute_module_subhashes(m_l.gate.residual_head)
    hj_res = compute_module_subhashes(m_j.gate.residual_head)
    res_match = (hn_res == hg_res == hl_res == hj_res)

    hl_lsa_act = compute_module_subhashes(m_l.actor_spatial_attn)
    hj_lsa_act = compute_module_subhashes(m_j.actor_spatial_attn)
    hl_lsa_uni = compute_module_subhashes(m_l.union_spatial_attn)
    hj_lsa_uni = compute_module_subhashes(m_j.union_spatial_attn)
    lsa_match = (hl_lsa_act == hj_lsa_act and hl_lsa_uni == hj_lsa_uni)

    log(f"  N vs G 10 source projections: {'PASS' if n_vs_g else 'FAIL'}")
    log(f"  N vs L 10 source projections: {'PASS' if n_vs_l else 'FAIL'}")
    log(f"  L vs J 12 source projections: {'PASS' if l_vs_j else 'FAIL'}")
    log(f"  G vs J 10 source projections: {'PASS' if g_vs_j else 'FAIL'}")
    log(f"  Residual heads match:         {'PASS' if res_match else 'FAIL'}")
    log(f"  L vs J LSA modules match:     {'PASS' if lsa_match else 'FAIL'}")

    all_pass = n_vs_g and n_vs_l and l_vs_j and g_vs_j and res_match and lsa_match
    log(f"COMMON_INIT_PARITY = {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def run_preflight_canary(
    device: torch.device,
    base_model_dir: Path,
    pb_dir: Path,
) -> bool:
    """Run preflight canary check for N, G, and L on fold VG1."""
    log("=== RUNNING PREFLIGHT CPU CANARY AUDIT ===")
    fold_name = "vg1"
    runtime_config = _resolve_runtime_config(fold_name, None, None, None)
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    val_indices = data.split_indices("validation")
    pb_cache_path = pb_dir / fold_name.lower() / "pb_teacher_features.pt"
    pb_cache = torch.load(pb_cache_path, map_location="cpu", weights_only=False)
    posture_sidecar = torch.load(DEFAULT_POSTURE_SIDECAR, map_location="cpu", weights_only=False)
    h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu", weights_only=False)
    data_plus_store = DataPlusMultimodalStore(
        DataPlusSidecarPaths.from_root(DEFAULT_DATA_PLUS_DIR, pb_path=pb_cache_path)
    )
    resolver = JointKeyedBatchResolver(
        data, pb_cache, posture_sidecar, h5_cache, data_plus_store, device
    )
    val_keys = list(resolver.canonical_keys(val_indices))[:64]
    key_to_video = {
        k: data.bundle.frame.iloc[idx]["video_key"]
        for idx, k in zip(val_indices[:64], val_keys, strict=False)
    }

    base_model, _ = _load_base_model(
        fold_name, data, runtime_config, device, base_model_dir=base_model_dir
    )

    configs_to_check = ["N", "G", "L"]
    canary_passed = True

    for cfg in configs_to_check:
        seed_everything(SEED)
        model = build_ablation_model(cfg, base_model, base_seed=SEED).to(device)

        base_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_pct = (trainable_params / base_params) * 100.0
        log(f"[{cfg}] Trainable params: {trainable_params:,} ({param_pct:.4f}%)")

        eval_res = evaluate_model(model, resolver, val_keys, key_to_video, batch_size=32)
        diff = float(np.max(np.abs(eval_res["final_logits"] - eval_res["base_logits"])))
        argmax_match = float(
            (eval_res["final_logits"].argmax(axis=-1) == eval_res["base_logits"].argmax(axis=-1))
            .mean() * 100.0
        )
        log(f"[{cfg}] Step-0 max abs diff: {diff:.2e}, argmax parity: {argmax_match:.2f}%")

        if diff > 1e-7 or argmax_match != 100.0:
            log(f"[{cfg}] Step-0 parity FAIL!")
            canary_passed = False
        else:
            log(f"[{cfg}] Step-0 parity PASS!")

    return canary_passed


def train_and_eval_fold(
    config_name: str,
    fold_name: str,
    device: torch.device,
    base_model_dir: Path,
    pb_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate an ablation configuration on one fold."""
    log("\n==================================================")
    log(f"STARTING CONFIG {config_name.upper()} - FOLD {fold_name.upper()}")
    log("==================================================")

    fold_dir = output_dir / fold_name.lower()
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Check cached completed fold
    manifest_file = fold_dir / "fold_manifest.json"
    if manifest_file.exists() and (fold_dir / "best_validation.pt").exists():
        try:
            with open(manifest_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("reloaded_parity") == "PASS":
                log(f"[{config_name.upper()}-{fold_name.upper()}] Completed. Loading cached.")
                return cached
        except Exception as e:
            log(f"Error reading manifest: {e}. Re-running.")

    runtime_config = _resolve_runtime_config(fold_name, None, None, None)
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    train_indices = data.split_indices("train")
    val_indices = data.split_indices("validation")

    pb_cache_path = pb_dir / fold_name.lower() / "pb_teacher_features.pt"
    if not pb_cache_path.exists():
        raise FileNotFoundError(f"Missing PB cache: {pb_cache_path}")
    pb_cache = torch.load(pb_cache_path, map_location="cpu", weights_only=False)

    posture_sidecar = torch.load(DEFAULT_POSTURE_SIDECAR, map_location="cpu", weights_only=False)
    h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu", weights_only=False)
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
        data_plus_keys = [f"data_plus_{u}" for u in eligible_units.tolist()]
        train_keys.extend(data_plus_keys)

    val_keys = list(resolver.canonical_keys(val_indices))
    key_to_video = {
        k: data.bundle.frame.iloc[idx]["video_key"]
        for idx, k in zip(val_indices, val_keys, strict=False)
    }

    base_model, base_sha = _load_base_model(
        fold_name, data, runtime_config, device, base_model_dir=base_model_dir
    )

    seed_everything(SEED)
    model = build_ablation_model(config_name, base_model, base_seed=SEED).to(device)

    # Runtime tap provenance
    actor_tap_name = "layer3"
    actor_tap_shape = "[B*T, 256, 8, 8]"
    union_tap_name = "layer3"
    union_tap_shape = "[B*T, 256, 8, 8]"

    base_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_pct = (trainable_params / base_params) * 100.0

    log(
        f"BASE_PARAMS = {base_params:,}, "
        f"TRAINABLE_PARAMS = {trainable_params:,} ({param_pct:.4f}%)"
    )

    # Step-0 Parity Check
    step0_eval = evaluate_model(model, resolver, val_keys, key_to_video)
    max_logit_diff = float(
        np.max(np.abs(step0_eval["final_logits"] - step0_eval["base_logits"]))
    )
    argmax_parity = float(
        (step0_eval["final_logits"].argmax(axis=-1) == step0_eval["base_logits"].argmax(axis=-1))
        .mean() * 100.0
    )
    a_f2_ref = A_AUTHORITY_SCORES[fold_name.lower()]
    f1_diff = abs(step0_eval["base_f1"] - a_f2_ref)

    log(f"[{fold_name.upper()}] A_AUTHORITY_F1 = {a_f2_ref:.6f}")
    log(f"[{fold_name.upper()}] STEP0_BASE_F1   = {step0_eval['base_f1']:.6f} (Diff={f1_diff:.2e})")
    log(f"[{fold_name.upper()}] MAX_ABS_LOGIT_DIFF = {max_logit_diff:.2e}")
    log(f"[{fold_name.upper()}] ARGMAX_PARITY = {argmax_parity:.2f}%")
    assert max_logit_diff <= 1e-7, f"Step-0 logit diff exceeds 1e-7: {max_logit_diff}"
    assert argmax_parity == 100.0, f"Step-0 argmax parity not 100%: {argmax_parity}"
    assert f1_diff < 1e-5, f"Step-0 base F1 does not match Model A authority: {f1_diff}"
    log("STEP0_LOGIT_PARITY = PASS")

    initial_base_hashes = get_model_hash_dict(base_model)
    initial_bn_states = get_bn_state_dict(base_model)

    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
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

    best_val_f1 = step0_eval["base_f1"]
    best_epoch = -1
    replaces_step0 = False

    # Save Step-0 fallback checkpoint
    ckpt_path = fold_dir / "best_validation.pt"
    tmp_ckpt_path = fold_dir / "best_validation.pt.tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": -1,
            "best_f1": best_val_f1,
            "config_name": config_name,
            "fold_name": fold_name,
        },
        tmp_ckpt_path,
    )
    tmp_ckpt_path.replace(ckpt_path)

    batch_size = 64
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        random.shuffle(train_keys)
        total_loss = 0.0
        batch_count = 0

        for i in range(0, len(train_keys), batch_size):
            b_keys = train_keys[i : i + batch_size]
            aligned = resolver.batch(b_keys)
            targets = aligned.training_batch.behavior_target

            optimizer.zero_grad()
            out = model(aligned)
            final_logits = out[0]
            loss = criterion(final_logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = total_loss / max(batch_count, 1)

        # Evaluate validation set
        val_eval = evaluate_model(model, resolver, val_keys, key_to_video)
        val_f1 = val_eval["macro_f1"]
        val_acc = val_eval["accuracy"]

        log(
            f"Epoch {epoch:02d} | TrainLoss={avg_loss:.4f} | "
            f"ValF1={val_f1:.6f} | ValAcc={val_acc:.4f} | Time={epoch_time:.1f}s"
        )

        # Save last_training.pt atomically
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_f1": val_f1,
            },
            fold_dir / "last_training.pt",
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            replaces_step0 = True

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_f1": best_val_f1,
                    "config_name": config_name,
                    "fold_name": fold_name,
                },
                tmp_ckpt_path,
            )
            tmp_ckpt_path.replace(ckpt_path)
            log(f"  -> New best validation Macro-F1: {best_val_f1:.6f} (saved best_validation.pt)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping triggered at epoch {epoch}.")
                break

    log(f"Training finished. Best Epoch: {best_epoch}, Best F1: {best_val_f1:.6f}")

    # Base Freeze & BN Parity Audits
    final_base_hashes = get_model_hash_dict(base_model)
    final_bn_states = get_bn_state_dict(base_model)
    base_weight_parity = "PASS" if initial_base_hashes == final_base_hashes else "FAIL"
    base_bn_parity = "PASS" if initial_bn_states == final_bn_states else "FAIL"
    assert base_weight_parity == "PASS", "Base Model A weights changed during training!"
    assert base_bn_parity == "PASS", "Base Model A BatchNorm buffers drifted during training!"
    log(f"BASE_WEIGHT_PARITY = {base_weight_parity}")
    log(f"BASE_BN_PARITY     = {base_bn_parity}")

    # Physical Reload Authority Verification
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 0
    final_sha256 = sha256_file(ckpt_path)
    reloaded_model = build_ablation_model(config_name, base_model, base_seed=SEED).to(device)
    saved_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(saved_state["model_state_dict"], strict=True)
    reloaded_model.eval()

    final_eval = evaluate_model(reloaded_model, resolver, val_keys, key_to_video)
    reloaded_f1 = final_eval["macro_f1"]
    reloaded_diff = abs(reloaded_f1 - best_val_f1)
    reloaded_parity = "PASS" if reloaded_diff < 1e-6 else "FAIL"
    log(
        f"PHYSICAL_RELOAD_AUTHORITY: ReloadedF1={reloaded_f1:.6f} "
        f"vs BestF1={best_val_f1:.6f} -> {reloaded_parity}"
    )
    assert reloaded_parity == "PASS", "Physical reload parity failed!"

    # Save raw validation arrays
    np.save(fold_dir / "sample_ids.npy", final_eval["sample_ids"])
    np.save(fold_dir / "video_keys.npy", final_eval["video_keys"])
    np.save(fold_dir / "targets.npy", final_eval["targets"])
    np.save(fold_dir / "logits.npy", final_eval["final_logits"])

    df_preds = pd.DataFrame({
        "sample_id": final_eval["sample_ids"],
        "video_key": final_eval["video_keys"],
        "target": final_eval["targets"],
        "prediction": final_eval["final_logits"].argmax(axis=-1),
    })
    df_preds.to_csv(fold_dir / "predictions.csv", index=False)
    final_eval["group_metrics_df"].to_csv(fold_dir / "group_level_metrics.csv", index=False)

    metrics_record = {
        "config_name": config_name,
        "fold": fold_name,
        "best_epoch": best_epoch,
        "replaces_step0": replaces_step0,
        "step0_f1": step0_eval["base_f1"],
        "reloaded_f1": reloaded_f1,
        "accuracy": final_eval["accuracy"],
        "delta_vs_a": reloaded_f1 - A_AUTHORITY_SCORES[fold_name.lower()],
        "per_class_f1": final_eval["per_class_f1"],
        "checkpoint_sha256": final_sha256,
        "actor_local_tap_name": actor_tap_name,
        "actor_local_tap_shape": actor_tap_shape,
        "union_local_tap_name": union_tap_name,
        "union_local_tap_shape": union_tap_shape,
    }
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    fold_manifest = {
        "task_id": "FINAL-J-LOCAL-GATE-ABLATION-20260916",
        "config": config_name,
        "fold": fold_name,
        "reloaded_parity": reloaded_parity,
        "base_weight_parity": base_weight_parity,
        "base_bn_parity": base_bn_parity,
        "step0_parity": "PASS",
        "checkpoint_sha256": final_sha256,
        "trainable_parameters": trainable_params,
        "trainable_percent": param_pct,
        "reloaded_f1": reloaded_f1,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(fold_manifest, f, indent=2)

    return fold_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["N", "G", "L", "ALL"], default="ALL")
    parser.add_argument(
        "--fold", choices=["vg1", "vg2", "vg3", "vg4", "vg5", "all"], default="all"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--pb_dir", type=Path, default=DEFAULT_PB_DIR)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE_MODEL_A_DIR)
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        default=Path("outputs/classification_v2/final_j_ablation_v1"),
    )
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"Executing Factor Ablation on device: {device}")

    # 1. Common Initialization Parity Audit
    init_parity = verify_common_initialization_parity()
    assert init_parity, "Common initialization parity failed!"

    # 2. Preflight Canary
    if args.preflight_only or args.device == "cpu":
        canary_pass = run_preflight_canary(torch.device("cpu"), args.base_model_dir, args.pb_dir)
        assert canary_pass, "Preflight canary failed!"
        log("PREFLIGHT CANARY = PASS")
        if args.preflight_only:
            return

    configs_to_run = ["N", "L", "G"] if args.config == "ALL" else [args.config]
    folds_to_run = FOLDS if args.fold == "all" else [args.fold]

    results_by_config: dict[str, dict[str, Any]] = {c: {} for c in configs_to_run}

    for cfg in configs_to_run:
        cfg_dir = args.output_base_dir / f"config_{cfg.lower()}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for fld in folds_to_run:
            res = train_and_eval_fold(
                cfg, fld, device, args.base_model_dir, args.pb_dir, cfg_dir
            )
            results_by_config[cfg][fld] = res

    # Generate summary if all folds and configs were executed
    log("\n=== FACTORIAL ABLATION EXECUTION COMPLETE ===")

    summary_data: dict[str, Any] = {
        "task_id": "FINAL-J-LOCAL-GATE-ABLATION-20260916",
        "reference_authorities": {
            "A": {"folds": A_AUTHORITY_SCORES, "mean": A_MEAN},
            "J": {"folds": J_AUTHORITY_SCORES, "mean": J_MEAN},
            "K": {"folds": K_REFERENCE_SCORES, "mean": K_MEAN},
        },
        "results_by_config": {},
    }

    for cfg, fold_dict in results_by_config.items():
        if len(fold_dict) == len(FOLDS):
            scores = [fold_dict[fld]["reloaded_f1"] for fld in FOLDS]
            m = float(np.mean(scores))
            std0 = float(np.std(scores, ddof=0))
            std1 = float(np.std(scores, ddof=1))
            summary_data["results_by_config"][cfg] = {
                "folds": {fld: fold_dict[fld]["reloaded_f1"] for fld in FOLDS},
                "mean": m,
                "std_ddof0": std0,
                "std_ddof1": std1,
                "delta_vs_a": m - A_MEAN,
                "delta_vs_j": m - J_MEAN,
            }
            log(f"[{cfg}] 5-Fold Mean: {m:.6f} +/- {std0:.6f} (vs A: {m - A_MEAN:+.6f})")

    # If N, L, G all executed 5 folds, compute factorial effects
    if all(cfg in summary_data["results_by_config"] for cfg in ["N", "L", "G"]):
        mean_n = summary_data["results_by_config"]["N"]["mean"]
        mean_l = summary_data["results_by_config"]["L"]["mean"]
        mean_g = summary_data["results_by_config"]["G"]["mean"]
        mean_j = J_MEAN
        mean_a = A_MEAN

        scaffold_effect = mean_n - mean_a
        local_effect_neutral = mean_l - mean_n
        gate_effect_no_local = mean_g - mean_n
        full_effect_vs_scaffold = mean_j - mean_n
        interaction = mean_j - mean_l - mean_g + mean_n

        summary_data["factor_decompositions"] = {
            "scaffold_effect": scaffold_effect,
            "local_effect_neutral": local_effect_neutral,
            "gate_effect_no_local": gate_effect_no_local,
            "full_effect_vs_scaffold": full_effect_vs_scaffold,
            "descriptive_local_gate_interaction": interaction,
        }
        log("\n--- FACTORIAL DECOMPOSITION SUMMARY ---")
        log(f"SCAFFOLD_EFFECT (N - A):             {scaffold_effect:+.6f}")
        log(f"LOCAL_EFFECT_NEUTRAL (L - N):        {local_effect_neutral:+.6f}")
        log(f"GATE_EFFECT_NO_LOCAL (G - N):        {gate_effect_no_local:+.6f}")
        log(f"FULL_EFFECT_VS_SCAFFOLD (J - N):     {full_effect_vs_scaffold:+.6f}")
        log(f"DESCRIPTIVE_INTERACTION (J-L-G+N):   {interaction:+.6f}")

    args.output_base_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_base_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
