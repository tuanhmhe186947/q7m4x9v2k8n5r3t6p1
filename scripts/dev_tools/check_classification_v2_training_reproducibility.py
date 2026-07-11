"""Run two bounded strict-training smokes and compare deterministic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.data_module import MODEL_INPUT_KEYS, validate_model_inputs
from pig_behavior.classification_v2.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict trainer reproducibility without full OOF.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/multimodal_context_multitask.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/training_reproducibility"),
    )
    args = parser.parse_args()
    base = load_training_config(args.config)
    audits = []
    prediction_hashes = []
    for run_name in ("run_a", "run_b"):
        run_dir = args.output_dir / run_name
        config = replace(
            base,
            optimization=replace(base.optimization, batch_size=10, eval_batch_size=10),
            execution=replace(base.execution, mode="smoke", smoke_steps=2, output_dir=run_dir, resume=False),
        )
        audits.append(run_training(config))
        prediction_hashes.append(_prediction_digest(run_dir / "validation_predictions.csv"))
    forbidden_rejected = False
    try:
        validate_model_inputs({**{key: None for key in MODEL_INPUT_KEYS}, "review_sample_weight": None})
    except ValueError:
        forbidden_rejected = True
    comparable_history = [_history_signature(audit) for audit in audits]
    errors: list[str] = []
    if audits[0]["train_selected_window_id_sha256"] != audits[1]["train_selected_window_id_sha256"]:
        errors.append("train_selection_hash_mismatch")
    if audits[0]["validation_selected_window_id_sha256"] != audits[1]["validation_selected_window_id_sha256"]:
        errors.append("validation_selection_hash_mismatch")
    if comparable_history[0] != comparable_history[1]:
        errors.append(f"metric_history_mismatch={comparable_history}")
    if prediction_hashes[0] != prediction_hashes[1]:
        errors.append("prediction_digest_mismatch")
    if not forbidden_rejected:
        errors.append("forbidden_model_input_not_rejected")
    result = {
        "schema_version": "classification_v2_training_reproducibility_audit_v1",
        "run_directories": [str(args.output_dir / name) for name in ("run_a", "run_b")],
        "train_selection_sha256": [audit["train_selected_window_id_sha256"] for audit in audits],
        "validation_selection_sha256": [audit["validation_selected_window_id_sha256"] for audit in audits],
        "history_signatures": comparable_history,
        "prediction_sha256": prediction_hashes,
        "forbidden_model_input_rejected": forbidden_rejected,
        "tolerance": 0.0,
        "errors": errors,
        "valid": not errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reproducibility_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _history_signature(audit: dict[str, object]) -> list[dict[str, object]]:
    keys = ["train_loss_mean", "train_loss_first", "train_loss_last", "validation_window_macro_f1"]
    return [{key: row[key] for key in keys} for row in audit["history"]]


def _prediction_digest(path: Path) -> str:
    frame = pd.read_csv(path).sort_values("window_id").reset_index(drop=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
