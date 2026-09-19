"""Two-epoch CPU lifecycle smoke runner for G1 Reviewed Posture Auxiliary treatment.

Demonstrates:
  1. inner_train optimization with L_behavior + 0.25 * L_posture
  2. inner_val execution after every epoch
  3. behavior Macro-F1 logged each epoch
  4. posture secondary metrics logged
  5. best checkpoint saved strictly by BEHAVIOR Macro-F1
  6. last checkpoint saved
  7. checkpoint restoration verified
  8. outer_test never evaluated (count = 0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report

from pig_behavior.classification_v2.models.g1_posture_model import (
    POSTURE_LAMBDA_LOCKED,
    G1PostureAuxiliaryClassifier,
    compute_g1_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

CLASSES = list(VALID_BEHAVIORS)


def run_g1_cpu_smoke(
    out_dir: Path,
    num_epochs: int = 2,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Execute a self-contained 2-epoch CPU smoke run for G1 posture auxiliary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Instantiate model
    backbone_config = MultimodalFusionConfig(
        enable_image=False,
        enable_spatial=True,
        spatial_input_dims={"bbox_xywh_n": 4},
        spatial_embedding_dim=256,
        fusion_hidden_dim=256,
        num_classes=10,
    )
    model = G1PostureAuxiliaryClassifier(backbone_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 2. Synthetic tiny inner_train (24 samples) and inner_val (12 samples)
    n_train = 24
    n_val = 12
    t_len = 6

    train_spatial = {"bbox_xywh_n": torch.randn(n_train, t_len, 4)}
    train_mask = torch.ones(n_train, t_len, dtype=torch.bool)
    train_beh_targets = torch.randint(0, 10, (n_train,))
    # Some reviewed rows, some unreviewed rows (-1 and False)
    train_pos_targets = torch.tensor([0, 1, 2, -1, -1, 0, 1, -1] * 3)
    train_pos_mask = train_pos_targets >= 0

    val_spatial = {"bbox_xywh_n": torch.randn(n_val, t_len, 4)}
    val_mask = torch.ones(n_val, t_len, dtype=torch.bool)
    val_beh_targets = torch.randint(0, 10, (n_val,))
    val_pos_targets = torch.tensor([0, 1, 2, -1] * 3)
    val_pos_mask = val_pos_targets >= 0

    best_beh_f1 = -1.0
    best_epoch = -1
    epoch_logs = []
    outer_test_eval_count = 0

    for epoch in range(num_epochs):
        # --- inner_train ---
        model.train()
        epoch_loss = 0.0
        epoch_l_beh = 0.0
        epoch_l_pos = 0.0
        num_batches = 0

        perm = torch.randperm(n_train)
        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            optimizer.zero_grad()

            b_spatial = {"bbox_xywh_n": train_spatial["bbox_xywh_n"][b_idx]}
            b_mask = train_mask[b_idx]
            b_beh_target = train_beh_targets[b_idx]
            b_pos_target = train_pos_targets[b_idx]
            b_pos_mask = train_pos_mask[b_idx]

            out = model(spatial_features=b_spatial, length_mask=b_mask)
            total_loss, l_beh, l_pos = compute_g1_loss(
                out.behavior_logits,
                b_beh_target,
                None,
                out.posture_logits,
                b_pos_target,
                b_pos_mask,
                posture_lambda=POSTURE_LAMBDA_LOCKED,
            )
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_l_beh += l_beh.item()
            epoch_l_pos += l_pos.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_l_beh = epoch_l_beh / max(1, num_batches)
        avg_l_pos = epoch_l_pos / max(1, num_batches)

        # --- inner_val ---
        model.eval()
        with torch.inference_mode():
            val_out = model(spatial_features=val_spatial, length_mask=val_mask)
            val_beh_preds = torch.argmax(val_out.behavior_logits, dim=-1).numpy()
            val_beh_rep = classification_report(
                val_beh_targets.numpy(),
                val_beh_preds,
                labels=list(range(10)),
                output_dict=True,
                zero_division=0,
            )
            val_beh_macro_f1 = float(val_beh_rep["macro avg"]["f1-score"])

            # Secondary posture metrics on reviewed val subset
            if val_pos_mask.any():
                val_pos_preds = torch.argmax(val_out.posture_logits[val_pos_mask], dim=-1).numpy()
                val_pos_rep = classification_report(
                    val_pos_targets[val_pos_mask].numpy(),
                    val_pos_preds,
                    labels=list(range(3)),
                    output_dict=True,
                    zero_division=0,
                )
                val_pos_macro_f1 = float(val_pos_rep["macro avg"]["f1-score"])
            else:
                val_pos_macro_f1 = 0.0

        # Save checkpoint based on BEHAVIOR Macro-F1
        if val_beh_macro_f1 > best_beh_f1:
            best_beh_f1 = val_beh_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")

        torch.save(model.state_dict(), out_dir / "last_checkpoint.pt")

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_l_behavior": avg_l_beh,
            "train_l_posture": avg_l_pos,
            "inner_val_behavior_macro_f1": val_beh_macro_f1,
            "inner_val_posture_macro_f1": val_pos_macro_f1,
            "best_epoch": best_epoch,
            "best_behavior_macro_f1": best_beh_f1,
        }
        epoch_logs.append(log_entry)

    # 3. Verify checkpoint restoration
    restored_model = G1PostureAuxiliaryClassifier(backbone_config)
    restored_state = torch.load(out_dir / "best_checkpoint.pt", weights_only=True)
    restored_model.load_state_dict(restored_state, strict=True)
    restored_model.eval()

    with torch.inference_mode():
        restored_out = restored_model.forward_behavior(
            spatial_features=val_spatial,
            length_mask=val_mask,
        )
    assert restored_out.shape == (n_val, 10)

    summary = {
        "two_epoch_cpu_lifecycle": "PASS",
        "inner_val_behavior_f1_logged": True,
        "best_checkpoint_by_behavior_f1": True,
        "checkpoint_restore_success": True,
        "outer_test_used": False,
        "outer_test_eval_count": outer_test_eval_count,
        "num_epochs_completed": num_epochs,
        "best_epoch": best_epoch,
        "epoch_logs": epoch_logs,
    }
    with open(out_dir / "cpu_smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    smoke_out = Path("outputs/classification_v2/g1_posture_auxiliary_v1/cpu_smoke")
    res = run_g1_cpu_smoke(smoke_out)
    print("Smoke Result:", json.dumps(res, indent=2))
