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

DEFAULT_FULL_OUTPUT_DIR = Path(
    "outputs/classification_v2/model_full/full_multimodal_oof"
)
DEFAULT_PILOT_OUTPUT_DIR = Path(
    "outputs/classification_v2/model_smoke/full_multimodal_oof_pilot"
)
DEFAULT_ACTOR_CACHE_ROOT = Path("outputs/classification_v2/image_cache_v2_letterbox")
DEFAULT_VISUAL_CACHE_ROOT = Path(
    "outputs/classification_v2/visual_interaction_cache"
)


def main() -> None:
    """Write a no-training workload plan for learned multimodal native OOF."""

    parser = argparse.ArgumentParser(
        description="Plan classification_v2 learned multimodal OOF workload."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/full_multimodal_oof_run_plan.json"
        ),
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=None,
        help=(
            "Planned model artifact directory. Defaults to model_full for full "
            "plans and model_smoke for pilot plans."
        ),
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
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Plan bounded pilot settings instead of full OOF.",
    )
    parser.add_argument("--ablation-variant", choices=ABLATION_VARIANTS, default="full")
    parser.add_argument("--image-cache-manifest", type=Path, default=None)
    parser.add_argument("--packed-image-cache", type=Path, default=None)
    parser.add_argument("--packed-image-cache-index", type=Path, default=None)
    parser.add_argument("--visual-context-cache-manifest", type=Path, default=None)
    parser.add_argument("--visual-context-packed-cache", type=Path, default=None)
    parser.add_argument("--visual-context-packed-cache-index", type=Path, default=None)
    parser.add_argument(
        "--sample-weight-policy",
        choices=SAMPLE_WEIGHT_POLICIES,
        default="event_class",
    )
    parser.add_argument("--precision", choices=PRECISION_POLICIES, default="amp")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    args = parser.parse_args()
    run_output_dir = args.run_output_dir or (
        DEFAULT_PILOT_OUTPUT_DIR if args.pilot else DEFAULT_FULL_OUTPUT_DIR
    )
    actor_manifest = args.image_cache_manifest
    actor_tensor = args.packed_image_cache
    actor_index = args.packed_image_cache_index
    visual_manifest = args.visual_context_cache_manifest
    visual_tensor = args.visual_context_packed_cache
    visual_index = args.visual_context_packed_cache_index
    if not args.pilot:
        actor_tensor = actor_tensor or _actor_packed_tensor(args.image_size)
        actor_index = actor_index or DEFAULT_ACTOR_CACHE_ROOT / "packed_image_cache_index.csv"
        visual_manifest = visual_manifest or DEFAULT_VISUAL_CACHE_ROOT / (
            "visual_context_manifest.csv"
        )
        visual_tensor = visual_tensor or _visual_packed_tensor(args.image_size)
        visual_index = visual_index or DEFAULT_VISUAL_CACHE_ROOT / (
            "packed_image_cache_index.csv"
        )
    config = FullMultimodalOofConfig(
        output_dir=run_output_dir,
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
        image_cache_manifest_csv=actor_manifest,
        packed_image_cache_npy=actor_tensor,
        packed_image_cache_index_csv=actor_index,
        visual_context_cache_manifest_csv=visual_manifest,
        visual_context_packed_cache_npy=visual_tensor,
        visual_context_packed_cache_index_csv=visual_index,
        require_packed_visual_context=not args.pilot and visual_tensor is not None,
        require_cached_images=(
            actor_manifest is not None or actor_tensor is not None
        ),
        sample_weight_policy=args.sample_weight_policy,
        precision=args.precision,
        bootstrap_iterations=args.bootstrap_iterations,
        device=args.device,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    plan = build_full_multimodal_oof_run_plan(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if plan["errors"] or not plan["valid"]:
        raise SystemExit(1)


def _actor_packed_tensor(image_size: int) -> Path:
    return DEFAULT_ACTOR_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


def _visual_packed_tensor(image_size: int) -> Path:
    return DEFAULT_VISUAL_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


if __name__ == "__main__":
    main()
