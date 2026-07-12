from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    ABLATION_VARIANTS,
    PRECISION_POLICIES,
    SAMPLE_WEIGHT_POLICIES,
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
    run_full_multimodal_oof,
)
from pig_behavior.classification_v2.training.full_run_preflight import (
    validate_preflight_for_execution,
)

FULL_RUN_AUTHORIZATION_PURPOSE = "classification_v2_full_multimodal_oof"
FULL_RUN_AUTHORIZATION_SCHEMA_VERSION = (
    "classification_v2_full_oof_authorization_v1"
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
DEFAULT_VISUAL_CONTEXT_MANIFEST = Path(
    "outputs/classification_v2/visual_interaction_cache/visual_context_manifest.csv"
)
FULL_DEFAULTS = {
    "image_size": 64,
    "hidden_dim": 48,
    "steps_per_fold": 6,
    "train_batch_size": 128,
    "eval_batch_size": 128,
    "bootstrap_iterations": 2000,
    "device": "cuda",
    "precision": "amp",
}
PILOT_DEFAULTS = {
    "image_size": 32,
    "hidden_dim": 32,
    "steps_per_fold": 2,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "bootstrap_iterations": 30,
    "device": "auto",
    "precision": "fp32",
}


def main() -> None:
    """Run a bounded pilot or explicit full learned multimodal native-OOF evaluation."""

    parser = argparse.ArgumentParser(
        description="Run classification_v2 learned multimodal native-OOF evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Model artifact directory. Defaults to model_full for --full and "
            "model_smoke for bounded pilot runs."
        ),
    )
    parser.add_argument("--image-cache-manifest", type=Path, default=None)
    parser.add_argument("--packed-image-cache", type=Path, default=None)
    parser.add_argument("--packed-image-cache-index", type=Path, default=None)
    parser.add_argument(
        "--visual-context-cache-manifest",
        type=Path,
        default=DEFAULT_VISUAL_CONTEXT_MANIFEST,
    )
    parser.add_argument("--visual-context-packed-cache", type=Path, default=None)
    parser.add_argument("--visual-context-packed-cache-index", type=Path, default=None)
    parser.add_argument("--require-packed-visual-context", action="store_true")
    parser.add_argument(
        "--allow-image-source-fallback",
        action="store_true",
        help="Allow missing cache entries to fall back to legacy crops/CVAT videos.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--steps-per-fold", type=int, default=None)
    parser.add_argument("--epochs-per-fold", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--train-per-class-per-fold", type=int, default=2)
    parser.add_argument("--eval-per-class-per-fold", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ablation-variant",
        choices=ABLATION_VARIANTS,
        default="full",
    )
    parser.add_argument(
        "--sample-weight-policy",
        choices=SAMPLE_WEIGHT_POLICIES,
        default="event_class",
    )
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=5.0)
    parser.add_argument("--precision", choices=PRECISION_POLICIES, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument(
        "--authorization-json",
        type=Path,
        default=None,
        help="Required with --full; records explicit approval for this exact preflight.",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help=(
            "Required with --full after a matching clean preflight; "
            "prevents accidental long runs."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing per-fold artifacts and recompute.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run all folds/all eligible rows. This can be slow and is "
            "required before paper-facing registration."
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_FULL_OUTPUT_DIR if args.full else DEFAULT_PILOT_OUTPUT_DIR
    )
    mode_defaults = FULL_DEFAULTS if args.full else PILOT_DEFAULTS
    image_size = args.image_size or int(mode_defaults["image_size"])
    hidden_dim = args.hidden_dim or int(mode_defaults["hidden_dim"])
    steps_per_fold = args.steps_per_fold or int(mode_defaults["steps_per_fold"])
    train_batch_size = args.train_batch_size or int(mode_defaults["train_batch_size"])
    eval_batch_size = args.eval_batch_size or int(mode_defaults["eval_batch_size"])
    bootstrap_iterations = args.bootstrap_iterations or int(
        mode_defaults["bootstrap_iterations"]
    )
    device = args.device or str(mode_defaults["device"])
    precision = args.precision or str(mode_defaults["precision"])
    actor_manifest = args.image_cache_manifest
    actor_tensor = args.packed_image_cache
    actor_index = args.packed_image_cache_index
    visual_manifest = args.visual_context_cache_manifest
    visual_tensor = args.visual_context_packed_cache
    visual_index = args.visual_context_packed_cache_index
    if args.full and not args.allow_image_source_fallback:
        actor_tensor = actor_tensor or _actor_packed_tensor(image_size)
        actor_index = actor_index or DEFAULT_ACTOR_CACHE_ROOT / "packed_image_cache_index.csv"
        visual_manifest = visual_manifest or DEFAULT_VISUAL_CONTEXT_MANIFEST
        visual_tensor = visual_tensor or _visual_packed_tensor(image_size)
        visual_index = visual_index or DEFAULT_VISUAL_CACHE_ROOT / (
            "packed_image_cache_index.csv"
        )
    config = FullMultimodalOofConfig(
        output_dir=output_dir,
        image_cache_manifest_csv=actor_manifest,
        packed_image_cache_npy=actor_tensor,
        packed_image_cache_index_csv=actor_index,
        visual_context_cache_manifest_csv=visual_manifest,
        visual_context_packed_cache_npy=visual_tensor,
        visual_context_packed_cache_index_csv=visual_index,
        require_packed_visual_context=(
            args.require_packed_visual_context
            or bool(args.full and visual_tensor is not None)
        ),
        require_cached_images=bool(
            (actor_manifest is not None or actor_tensor is not None)
            and not args.allow_image_source_fallback
        ),
        image_size=image_size,
        hidden_dim=hidden_dim,
        steps_per_fold=steps_per_fold,
        epochs_per_fold=args.epochs_per_fold,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        max_folds=None if args.full else args.max_folds,
        train_per_class_per_fold=None if args.full else args.train_per_class_per_fold,
        eval_per_class_per_fold=None if args.full else args.eval_per_class_per_fold,
        bootstrap_iterations=bootstrap_iterations,
        device=device,
        run_mode="full" if args.full else "pilot",
        resume=not args.no_resume,
        ablation_variant=args.ablation_variant,
        sample_weight_policy=args.sample_weight_policy,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
        precision=precision,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    if args.full:
        _validate_full_execution_confirmation(
            config,
            args.preflight_json,
            args.authorization_json,
            args.confirm_full_run,
        )
    result = run_full_multimodal_oof(config)
    print(json.dumps(result["audit"], indent=2))


def _actor_packed_tensor(image_size: int) -> Path:
    return DEFAULT_ACTOR_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


def _visual_packed_tensor(image_size: int) -> Path:
    return DEFAULT_VISUAL_CACHE_ROOT / f"packed_rgb_{int(image_size)}_letterbox.npy"


def _validate_full_execution_confirmation(
    config: FullMultimodalOofConfig,
    preflight_json: Path | None,
    authorization_json: Path | None,
    confirmed: bool,
) -> None:
    """Require matching preflight and approval before full OOF training starts."""

    if not confirmed:
        raise ValueError("--full requires --confirm-full-run after reviewing the workload plan")
    if preflight_json is None or not preflight_json.exists():
        raise ValueError("--full requires an existing --preflight-json")
    if authorization_json is None or not authorization_json.exists():
        raise ValueError("--full requires an existing --authorization-json")
    preflight = json.loads(preflight_json.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_json.read_text(encoding="utf-8"))
    errors = validate_preflight_for_execution(config, preflight)
    errors.extend(_validate_full_run_authorization(config, preflight, authorization))
    if errors:
        raise ValueError(f"full-run preflight execution gate failed: {errors}")


def _validate_full_run_authorization(
    config: FullMultimodalOofConfig,
    preflight: dict[str, object],
    authorization: dict[str, object],
) -> list[str]:
    """Bind human approval to the same config and commit as the clean preflight."""

    errors: list[str] = []
    if authorization.get("schema_version") != FULL_RUN_AUTHORIZATION_SCHEMA_VERSION:
        errors.append(
            "full_run_authorization_schema_version_mismatch="
            f"{authorization.get('schema_version')}"
        )
    if authorization.get("authorized") is not True:
        errors.append("full_run_authorization_requires_authorized_true")
    if authorization.get("purpose") != FULL_RUN_AUTHORIZATION_PURPOSE:
        errors.append(
            "full_run_authorization_purpose_mismatch="
            f"{authorization.get('purpose')}"
        )
    if authorization.get("acknowledges_long_run") is not True:
        errors.append("full_run_authorization_must_acknowledge_long_run")
    if authorization.get("acknowledges_no_q2_claim_until_verified") is not True:
        errors.append("full_run_authorization_must_acknowledge_no_q2_claim")
    if not str(authorization.get("reviewer") or "").strip():
        errors.append("full_run_authorization_requires_reviewer")
    if not str(authorization.get("reviewed_at") or "").strip():
        errors.append("full_run_authorization_requires_reviewed_at")

    expected_hash = full_run_config_fingerprint(config)
    preflight_hash = preflight.get("config_sha256")
    authorized_hash = authorization.get("preflight_config_sha256")
    if authorized_hash != preflight_hash:
        errors.append(
            "full_run_authorization_preflight_hash_mismatch="
            f"preflight:{preflight_hash},authorization:{authorized_hash}"
        )
    if authorized_hash != expected_hash:
        errors.append(
            "full_run_authorization_config_hash_mismatch="
            f"expected:{expected_hash},authorization:{authorized_hash}"
        )

    expected_commit = preflight.get("git_commit")
    authorized_commit = authorization.get("git_commit")
    if authorized_commit != expected_commit:
        errors.append(
            "full_run_authorization_git_commit_mismatch="
            f"expected:{expected_commit},authorization:{authorized_commit}"
        )
    return errors


if __name__ == "__main__":
    main()
