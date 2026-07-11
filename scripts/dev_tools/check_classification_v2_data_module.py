from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.data_module import (
    MODEL_INPUT_KEYS,
    StrictTrainingDataModule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict classification_v2 data module.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/multimodal_context_multitask.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/data_module_audit.json"),
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    errors: list[str] = []
    with StrictTrainingDataModule(config, device=torch.device("cpu")) as data:
        train_indices = data.balanced_smoke_indices(train=True)
        eval_indices = data.balanced_smoke_indices(train=False)
        train_batch = data.batch(train_indices)
        audit = data.audit()
        if set(train_batch.model_inputs) != set(MODEL_INPUT_KEYS):
            errors.append("model_input_key_contract_mismatch")
        forbidden_tokens = ["window", "source", "behavior", "target", "review", "manual", "path"]
        leaked = [
            key
            for key in train_batch.model_inputs
            if any(token in key.lower() for token in forbidden_tokens)
        ]
        if leaked:
            errors.append(f"forbidden_model_input_keys={leaked}")
        if len(train_indices) != len(eval_indices) or len(train_indices) == 0:
            errors.append(f"balanced_smoke_size_mismatch=train:{len(train_indices)}:eval:{len(eval_indices)}")
        if audit["duplicate_window_id"]:
            errors.append(f"duplicate_window_id={audit['duplicate_window_id']}")
        if audit["window_id_sha256"] != audit["auxiliary_window_id_sha256"]:
            errors.append("auxiliary_window_alignment_hash_mismatch")
        actor = data.actor_dataset.image_load_audit()
        visual = data.visual_dataset.load_audit()
        if actor["disk_image_cache_misses"] or actor["source_image_loads"]:
            errors.append("actor_strict_cache_violation")
        if visual["packed_cache_misses"] or visual["individual_cache_loads"]:
            errors.append("visual_strict_cache_violation")
    result = {
        **audit,
        "balanced_train_smoke_rows": int(len(train_indices)),
        "balanced_eval_smoke_rows": int(len(eval_indices)),
        "batch_model_input_keys": sorted(train_batch.model_inputs),
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
