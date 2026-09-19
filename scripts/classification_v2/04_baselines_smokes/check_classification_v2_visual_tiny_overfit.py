"""Run the bounded synthetic visual one-batch, overfit, and resume gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pig_behavior.classification_v2.training.visual_backbone_smoke import (
    SyntheticVisualSmokeConfig,
    run_synthetic_visual_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a data-free ResNet18 actor-temporal correctness gate."
    )
    parser.add_argument("--backbone-name", default="resnet18")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--events-per-class", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-norm-recalibration-passes", type=int, default=20)
    parser.add_argument("--minimum-accuracy", type=float, default=0.95)
    parser.add_argument("--maximum-loss-ratio", type=float, default=0.25)
    parser.add_argument("--repeatability-runs", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "visual_tiny_overfit_audit.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite explicitly: {args.output_json}"
        )
    if args.repeatability_runs <= 0:
        raise ValueError("--repeatability-runs must be positive")
    config = SyntheticVisualSmokeConfig(
        backbone_name=args.backbone_name,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
        events_per_class=args.events_per_class,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        batch_norm_recalibration_passes=(
            args.batch_norm_recalibration_passes
        ),
        minimum_accuracy=args.minimum_accuracy,
        maximum_loss_ratio=args.maximum_loss_ratio,
    )
    results = [
        run_synthetic_visual_smoke(config)
        for _ in range(args.repeatability_runs)
    ]
    result = results[0]
    signatures = [_repeatability_sha256(row) for row in results]
    signatures_match = len(set(signatures)) == 1
    result["repeatability_audit"] = {
        "runs": args.repeatability_runs,
        "semantic_sha256": signatures,
        "signatures_match": signatures_match,
        "valid": signatures_match,
    }
    if not signatures_match:
        result["errors"].append("repeatability_signature_mismatch")
        result["valid"] = False
    if not args.dry_run:
        _write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    """Write only the small audit; no checkpoint or training artifact is emitted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _repeatability_sha256(result: dict[str, object]) -> str:
    """Hash deterministic evidence while excluding runtime and memory noise."""

    fields = {
        key: result[key]
        for key in (
            "initial_loss",
            "final_loss",
            "loss_ratio",
            "final_accuracy",
            "losses",
            "gradient_audit",
            "resume_audit",
            "batch_norm_audit",
            "parameters",
            "errors",
            "valid",
        )
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
