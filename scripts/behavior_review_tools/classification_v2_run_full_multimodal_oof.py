from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    ABLATION_VARIANTS,
    PRECISION_POLICIES,
    SAMPLE_WEIGHT_POLICIES,
    FullMultimodalOofConfig,
    run_full_multimodal_oof,
)


def main() -> None:
    """Run a bounded pilot or explicit full learned multimodal native-OOF evaluation."""

    parser = argparse.ArgumentParser(description="Run classification_v2 learned multimodal native-OOF evaluation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/full_multimodal_oof_pilot"),
    )
    parser.add_argument("--image-cache-manifest", type=Path, default=None)
    parser.add_argument("--packed-image-cache", type=Path, default=None)
    parser.add_argument("--packed-image-cache-index", type=Path, default=None)
    parser.add_argument(
        "--allow-image-source-fallback",
        action="store_true",
        help="Allow missing cache entries to fall back to legacy crops/CVAT videos.",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps-per-fold", type=int, default=2)
    parser.add_argument("--epochs-per-fold", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--train-per-class-per-fold", type=int, default=2)
    parser.add_argument("--eval-per-class-per-fold", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablation-variant", choices=ABLATION_VARIANTS, default="full")
    parser.add_argument("--sample-weight-policy", choices=SAMPLE_WEIGHT_POLICIES, default="event_class")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=5.0)
    parser.add_argument("--precision", choices=PRECISION_POLICIES, default="fp32")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing per-fold artifacts and recompute.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all folds/all eligible rows. This can be slow and is required before paper-facing registration.",
    )
    args = parser.parse_args()
    config = FullMultimodalOofConfig(
        output_dir=args.output_dir,
        image_cache_manifest_csv=args.image_cache_manifest,
        packed_image_cache_npy=args.packed_image_cache,
        packed_image_cache_index_csv=args.packed_image_cache_index,
        require_cached_images=bool(
            (args.image_cache_manifest is not None or args.packed_image_cache is not None)
            and not args.allow_image_source_fallback
        ),
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        steps_per_fold=args.steps_per_fold,
        epochs_per_fold=args.epochs_per_fold,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_folds=None if args.full else args.max_folds,
        train_per_class_per_fold=None if args.full else args.train_per_class_per_fold,
        eval_per_class_per_fold=None if args.full else args.eval_per_class_per_fold,
        bootstrap_iterations=args.bootstrap_iterations,
        device=args.device,
        run_mode="full" if args.full else "pilot",
        resume=not args.no_resume,
        ablation_variant=args.ablation_variant,
        sample_weight_policy=args.sample_weight_policy,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
        precision=args.precision,
    )
    result = run_full_multimodal_oof(config)
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
