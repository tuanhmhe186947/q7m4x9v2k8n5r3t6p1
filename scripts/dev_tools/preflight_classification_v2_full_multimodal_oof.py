from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import FullMultimodalOofConfig
from pig_behavior.classification_v2.training.full_run_preflight import build_full_run_preflight


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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, required=True)
    parser.add_argument("--packed-image-cache", type=Path, required=True)
    parser.add_argument("--packed-image-cache-index", type=Path, required=True)
    parser.add_argument("--visual-context-cache-manifest", type=Path, required=True)
    parser.add_argument("--visual-context-packed-cache", type=Path, required=True)
    parser.add_argument("--visual-context-packed-cache-index", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs-per-fold", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    args = parser.parse_args()

    config = FullMultimodalOofConfig(
        output_dir=args.model_output_dir,
        packed_image_cache_npy=args.packed_image_cache,
        packed_image_cache_index_csv=args.packed_image_cache_index,
        require_cached_images=True,
        visual_context_cache_manifest_csv=args.visual_context_cache_manifest,
        visual_context_packed_cache_npy=args.visual_context_packed_cache,
        visual_context_packed_cache_index_csv=args.visual_context_packed_cache_index,
        require_packed_visual_context=True,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
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


if __name__ == "__main__":
    main()
