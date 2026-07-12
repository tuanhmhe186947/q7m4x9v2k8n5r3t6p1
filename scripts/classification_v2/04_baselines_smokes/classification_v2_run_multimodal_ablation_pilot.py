from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    ABLATION_VARIANTS,
    FullMultimodalOofConfig,
    run_full_multimodal_oof,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched bounded CUDA branch ablations for engineering validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_ablation_pilot"),
    )
    parser.add_argument(
        "--image-cache-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument("--variants", default=",".join(ABLATION_VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--steps-per-fold", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--train-per-class", type=int, default=1)
    parser.add_argument("--eval-per-class", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unsupported = sorted(set(variants).difference(ABLATION_VARIANTS))
    if unsupported:
        raise ValueError(f"unsupported ablation variants: {unsupported}")
    if len(variants) != len(set(variants)):
        raise ValueError("ablation variants must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in variants:
        result = run_full_multimodal_oof(
            FullMultimodalOofConfig(
                output_dir=args.output_dir / variant,
                image_cache_manifest_csv=args.image_cache_manifest,
                require_cached_images=True,
                image_size=args.image_size,
                hidden_dim=args.hidden_dim,
                steps_per_fold=args.steps_per_fold,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                max_folds=1,
                train_per_class_per_fold=args.train_per_class,
                eval_per_class_per_fold=args.eval_per_class,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
                device=args.device,
                run_mode="pilot",
                resume=False,
                ablation_variant=variant,
            )
        )
        records.append(_record_from_result(variant, result))

    train_hashes = sorted({record["train_indices_sha256"] for record in records})
    eval_hashes = sorted({record["eval_indices_sha256"] for record in records})
    errors = []
    if len(train_hashes) != 1:
        errors.append(f"train_index_mismatch_across_variants={len(train_hashes)}")
    if len(eval_hashes) != 1:
        errors.append(f"eval_index_mismatch_across_variants={len(eval_hashes)}")
    invalid = [record["variant"] for record in records if not record["valid"]]
    if invalid:
        errors.append(f"invalid_variants={invalid}")
    cache_violations = [
        record["variant"]
        for record in records
        if record["image_load_audit"].get("disk_image_cache_misses", 0)
        or record["image_load_audit"].get("source_image_loads", 0)
    ]
    if cache_violations:
        errors.append(f"cache_only_violations={cache_violations}")
    audit = {
        "schema_version": "classification_v2_multimodal_ablation_pilot_audit_v1",
        "evidence_level": "engineering_only_bounded_one_fold",
        "paper_facing_result": False,
        "external_generalization_claim": False,
        "variants": variants,
        "variant_count": int(len(records)),
        "matched_train_indices": len(train_hashes) == 1,
        "matched_eval_indices": len(eval_hashes) == 1,
        "records": records,
        "errors": errors,
        "warnings": [
            "bounded one-fold ablation; do not interpret metric deltas as paper evidence",
            "confirmatory ablations require all native OOF folds and multiple seeds",
        ],
        "valid": not errors,
    }
    output_path = args.output_dir / "multimodal_ablation_pilot_audit.json"
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(2)


def _record_from_result(variant: str, result: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable engineering fields without promoting pilot metrics."""

    audit = result["audit"]
    fold = audit["fold_audits"][0]
    metrics = json.loads(Path(result["metrics_json"]).read_text(encoding="utf-8"))
    native = metrics.get("native_temporal_metrics", {})
    return {
        "variant": variant,
        "valid": bool(audit.get("valid")),
        "device": audit.get("device"),
        "ablation_settings": audit.get("ablation_settings"),
        "instantiated_branches": fold.get("instantiated_branches"),
        "spatial_branch_order": fold.get("spatial_branch_order"),
        "trainable_parameter_count": fold.get("trainable_parameter_count"),
        "train_indices_sha256": fold.get("train_indices_sha256"),
        "eval_indices_sha256": fold.get("eval_indices_sha256"),
        "prediction_rows": audit.get("prediction_rows"),
        "native_temporal_rows": audit.get("native_temporal_rows"),
        "macro_f1_supported": native.get("macro_f1_supported"),
        "accuracy": native.get("accuracy"),
        "prediction_schema_valid": audit.get("prediction_schema_valid"),
        "image_load_audit": audit.get("image_load_audit", {}),
        "audit_json": result["audit_json"],
        "metrics_json": result["metrics_json"],
    }


if __name__ == "__main__":
    main()
