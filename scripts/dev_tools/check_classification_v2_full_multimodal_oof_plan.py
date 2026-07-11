from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    ABLATION_VARIANTS,
    PRECISION_POLICIES,
    SAMPLE_WEIGHT_POLICIES,
    FullMultimodalOofConfig,
    build_full_multimodal_oof_run_plan,
)


def main() -> None:
    """Write a no-training workload plan for learned multimodal native OOF."""

    parser = argparse.ArgumentParser(description="Plan classification_v2 learned multimodal OOF workload.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_multimodal_oof_run_plan.json"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--steps-per-fold", type=int, default=6)
    parser.add_argument("--epochs-per-fold", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--train-per-class-per-fold", type=int, default=None)
    parser.add_argument("--eval-per-class-per-fold", type=int, default=None)
    parser.add_argument("--pilot", action="store_true", help="Plan bounded pilot settings instead of full OOF.")
    parser.add_argument("--ablation-variant", choices=ABLATION_VARIANTS, default="full")
    parser.add_argument("--image-cache-manifest", type=Path, default=None)
    parser.add_argument("--packed-image-cache", type=Path, default=None)
    parser.add_argument("--packed-image-cache-index", type=Path, default=None)
    parser.add_argument("--sample-weight-policy", choices=SAMPLE_WEIGHT_POLICIES, default="event_class")
    parser.add_argument("--precision", choices=PRECISION_POLICIES, default="amp")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = FullMultimodalOofConfig(
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        steps_per_fold=args.steps_per_fold,
        epochs_per_fold=args.epochs_per_fold,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_folds=args.max_folds if args.pilot else None,
        train_per_class_per_fold=args.train_per_class_per_fold if args.pilot else None,
        eval_per_class_per_fold=args.eval_per_class_per_fold if args.pilot else None,
        run_mode="pilot" if args.pilot else "full",
        ablation_variant=args.ablation_variant,
        image_cache_manifest_csv=args.image_cache_manifest,
        packed_image_cache_npy=args.packed_image_cache,
        packed_image_cache_index_csv=args.packed_image_cache_index,
        require_cached_images=args.image_cache_manifest is not None or args.packed_image_cache is not None,
        sample_weight_policy=args.sample_weight_policy,
        precision=args.precision,
        bootstrap_iterations=args.bootstrap_iterations,
        device=args.device,
    )
    plan = build_full_multimodal_oof_run_plan(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if plan["errors"] or not plan["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
