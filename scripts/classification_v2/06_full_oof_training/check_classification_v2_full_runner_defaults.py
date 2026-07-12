from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
)
from pig_behavior.classification_v2.training.full_run_contract import (
    full_runner_default_config as _full_runner_default_config,
)


def main() -> None:
    """Audit that --full runner defaults match the reviewed full OOF plan."""

    parser = argparse.ArgumentParser(description="Check classification_v2 full runner default config contract.")
    parser.add_argument(
        "--run-plan-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_multimodal_oof_run_plan.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_runner_default_config_audit.json"),
    )
    args = parser.parse_args()
    audit = check_runner_defaults(args.run_plan_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_runner_defaults(run_plan_json: Path) -> dict[str, Any]:
    """Compare no-training runner defaults against the reviewed workload plan."""

    plan = json.loads(run_plan_json.read_text(encoding="utf-8"))
    config = _full_runner_default_config()
    config_hash = full_run_config_fingerprint(config)
    plan_config = plan.get("config") or {}
    errors: list[str] = []
    if config_hash != plan.get("config_sha256"):
        errors.append(f"runner_default_fingerprint_mismatch=runner:{config_hash},plan:{plan.get('config_sha256')}")
    for key, expected in _expected_config_values(config).items():
        observed = plan_config.get(key)
        if observed != expected:
            errors.append(f"plan_config_mismatch={key}:expected:{expected},actual:{observed}")
    for key in _required_existing_path_keys():
        path = plan_config.get(key)
        if not path or not Path(str(path)).exists():
            errors.append(f"missing_runner_default_path={key}:{path}")
    return {
        "schema_version": "classification_v2_full_runner_default_config_audit_v1",
        "run_plan_json": str(run_plan_json),
        "runner_config_sha256": config_hash,
        "plan_config_sha256": plan.get("config_sha256"),
        "runner_config": _expected_config_values(config),
        "checked_existing_path_keys": list(_required_existing_path_keys()),
        "errors": errors,
        "valid": not errors,
    }


def _expected_config_values(config: FullMultimodalOofConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "packed_image_cache_npy": str(config.packed_image_cache_npy),
        "packed_image_cache_index_csv": str(config.packed_image_cache_index_csv),
        "visual_context_cache_manifest_csv": str(config.visual_context_cache_manifest_csv),
        "visual_context_packed_cache_npy": str(config.visual_context_packed_cache_npy),
        "visual_context_packed_cache_index_csv": str(config.visual_context_packed_cache_index_csv),
        "require_cached_images": config.require_cached_images,
        "require_packed_visual_context": config.require_packed_visual_context,
        "image_size": config.image_size,
        "hidden_dim": config.hidden_dim,
        "steps_per_fold": config.steps_per_fold,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "bootstrap_iterations": config.bootstrap_iterations,
        "device": config.device,
        "precision": config.precision,
        "max_folds": config.max_folds,
        "train_per_class_per_fold": config.train_per_class_per_fold,
        "eval_per_class_per_fold": config.eval_per_class_per_fold,
        "run_mode": config.run_mode,
    }


def _required_existing_path_keys() -> tuple[str, ...]:
    return (
        "packed_image_cache_npy",
        "packed_image_cache_index_csv",
        "visual_context_cache_manifest_csv",
        "visual_context_packed_cache_npy",
        "visual_context_packed_cache_index_csv",
    )


if __name__ == "__main__":
    main()
