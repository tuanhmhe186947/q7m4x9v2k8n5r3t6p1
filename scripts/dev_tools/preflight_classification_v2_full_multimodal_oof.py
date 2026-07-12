from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import FullMultimodalOofConfig
from pig_behavior.classification_v2.training.full_run_preflight import build_full_run_preflight
from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (
    DEFAULT_ACTOR_CACHE_ROOT,
    DEFAULT_FULL_OUTPUT_DIR,
    DEFAULT_VISUAL_CACHE_ROOT,
    DEFAULT_VISUAL_CONTEXT_MANIFEST,
    FULL_DEFAULTS,
)


def main() -> None:
    """Write a full OOF preflight artifact without loading images or training."""

    parser = argparse.ArgumentParser(description="Preflight classification_v2 full multimodal OOF.")
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--runtime-benchmark-audit-json", type=Path, required=True)
    parser.add_argument(
        "--feature-whitelist-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_feature_whitelist_audit.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_multimodal_oof_preflight.json"
        ),
    )
    parser.add_argument("--model-output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--packed-image-cache", type=Path, default=None)
    parser.add_argument("--packed-image-cache-index", type=Path, default=None)
    parser.add_argument(
        "--visual-context-cache-manifest",
        type=Path,
        default=DEFAULT_VISUAL_CONTEXT_MANIFEST,
    )
    parser.add_argument("--visual-context-packed-cache", type=Path, default=None)
    parser.add_argument("--visual-context-packed-cache-index", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=FULL_DEFAULTS["image_size"])
    parser.add_argument("--hidden-dim", type=int, default=FULL_DEFAULTS["hidden_dim"])
    parser.add_argument(
        "--steps-per-fold",
        type=int,
        default=FULL_DEFAULTS["steps_per_fold"],
    )
    parser.add_argument("--epochs-per-fold", type=int, default=3)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=FULL_DEFAULTS["train_batch_size"],
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=FULL_DEFAULTS["eval_batch_size"],
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=FULL_DEFAULTS["bootstrap_iterations"],
    )
    parser.add_argument("--device", default=FULL_DEFAULTS["device"])
    parser.add_argument(
        "--precision",
        choices=["fp32", "amp"],
        default=FULL_DEFAULTS["precision"],
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    args = parser.parse_args()
    actor_tensor = args.packed_image_cache or _actor_packed_tensor(args.image_size)
    actor_index = args.packed_image_cache_index or (
        DEFAULT_ACTOR_CACHE_ROOT / "packed_image_cache_index.csv"
    )
    visual_tensor = args.visual_context_packed_cache or _visual_packed_tensor(
        args.image_size
    )
    visual_index = args.visual_context_packed_cache_index or (
        DEFAULT_VISUAL_CACHE_ROOT / "packed_image_cache_index.csv"
    )

    config = FullMultimodalOofConfig(
        output_dir=args.model_output_dir,
        packed_image_cache_npy=actor_tensor,
        packed_image_cache_index_csv=actor_index,
        require_cached_images=True,
        visual_context_cache_manifest_csv=args.visual_context_cache_manifest,
        visual_context_packed_cache_npy=visual_tensor,
        visual_context_packed_cache_index_csv=visual_index,
        require_packed_visual_context=True,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        steps_per_fold=args.steps_per_fold,
        epochs_per_fold=args.epochs_per_fold,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_folds=None,
        train_per_class_per_fold=None,
        eval_per_class_per_fold=None,
        bootstrap_iterations=args.bootstrap_iterations,
        device=args.device,
        run_mode="full",
        resume=True,
        sample_weight_policy="event_class",
        precision=args.precision,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    result = build_full_run_preflight(
        config,
        snapshot_json=args.snapshot_json,
        runtime_benchmark_audit_json=args.runtime_benchmark_audit_json,
        feature_whitelist_audit_json=args.feature_whitelist_audit_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def _actor_packed_tensor(image_size: int) -> Path:
    return DEFAULT_ACTOR_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


def _visual_packed_tensor(image_size: int) -> Path:
    return DEFAULT_VISUAL_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


if __name__ == "__main__":
    main()
