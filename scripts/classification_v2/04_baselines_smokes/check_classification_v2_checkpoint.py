from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from pig_behavior.classification_v2.training.config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check atomic classification_v2 checkpoint/resume contract.")
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
    args = parser.parse_args()
    config = load_training_config(args.config)
    _seed_all(123)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    x = torch.randn(5, 4)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint_path = args.output_dir / "checkpoint.pt"
    save_audit = save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
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
        restore_rng=True,
    )
    resumed_rng = _draw_rng()
    mismatch_rejected = False
    stale_config = replace(config, execution=replace(config.execution, fold_id="native_oof_stale"))
    try:
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=stale_config,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_contract_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw_rng() -> dict[str, float]:
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(1).item()),
    }


if __name__ == "__main__":
    main()
