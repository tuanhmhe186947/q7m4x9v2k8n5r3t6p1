"""Run one-step behavior-only B2/B3 smokes through the strict Q2 trainer."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Check behavior-only strict baseline training paths.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/s6_behavior_only_baselines"),
    )
    args = parser.parse_args()
    configs = {
        "B2_spatial_tcn": Path("configs/classification_v2/baseline_spatial_tcn.json"),
        "B3_actor_image": Path("configs/classification_v2/baseline_actor_image.json"),
    }
    runs = {}
    errors: list[str] = []
    for name, path in configs.items():
        config = load_training_config(path)
        config = replace(
            config,
            optimization=replace(config.optimization, batch_size=10, eval_batch_size=10),
            execution=replace(
                config.execution,
                smoke_steps=1,
                output_dir=args.output_dir / name,
                resume=False,
            ),
        )
        audit = run_training(config)
        runs[name] = {
            "model_parameter_count": audit["model_parameter_count"],
            "train_steps": audit["history"][0]["train_steps"],
            "validation_rows": audit["validation_rows"],
            "test_rows": audit["test_rows"],
            "auxiliary_class_weights": audit["auxiliary_class_weights_train_fold_only"],
            "oof_test_predictions_exists": (args.output_dir / name / "oof_test_predictions.csv").exists(),
        }
        if config.model.enable_multitask:
            errors.append(f"baseline_multitask_enabled={name}")
        if audit["auxiliary_class_weights_train_fold_only"]:
            errors.append(f"baseline_auxiliary_weights_present={name}")
        if audit["history"][0]["train_steps"] != 1:
            errors.append(f"baseline_train_step_count={name}")
    result = {
        "schema_version": "classification_v2_s6_behavior_only_baseline_check_v1",
        "runs": runs,
        "errors": errors,
        "valid": not errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavior_only_baseline_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
