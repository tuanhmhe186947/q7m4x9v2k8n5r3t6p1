"""Shared defaults and authorization policy for full multimodal OOF runs."""

from __future__ import annotations

import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
)
from pig_behavior.classification_v2.training.full_run_preflight import (
    validate_preflight_for_execution,
)

FULL_RUN_AUTHORIZATION_PURPOSE = "classification_v2_full_multimodal_oof"
FULL_RUN_AUTHORIZATION_SCHEMA_VERSION = "classification_v2_full_oof_authorization_v1"
DEFAULT_FULL_OUTPUT_DIR = Path("outputs/classification_v2/model_full/full_multimodal_oof")
DEFAULT_PILOT_OUTPUT_DIR = Path("outputs/classification_v2/model_smoke/full_multimodal_oof_pilot")
DEFAULT_ACTOR_CACHE_ROOT = Path("outputs/classification_v2/image_cache_v2_letterbox")
DEFAULT_VISUAL_CACHE_ROOT = Path("outputs/classification_v2/visual_interaction_cache")
DEFAULT_VISUAL_CONTEXT_MANIFEST = DEFAULT_VISUAL_CACHE_ROOT / "visual_context_manifest.csv"
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


def actor_packed_tensor(image_size: int) -> Path:
    """Return the canonical letterboxed actor tensor for an image size."""

    return DEFAULT_ACTOR_CACHE_ROOT / f"packed_rgb_{image_size}_letterbox.npy"


def visual_packed_tensor(image_size: int) -> Path:
    """Return the canonical letterboxed visual-context tensor path."""

    return DEFAULT_VISUAL_CACHE_ROOT / f"packed_rgb_{image_size}_letterbox.npy"


def full_runner_default_config() -> FullMultimodalOofConfig:
    """Build the exact reviewed full-run configuration used by gate checks."""

    image_size = int(FULL_DEFAULTS["image_size"])
    return FullMultimodalOofConfig(
        output_dir=DEFAULT_FULL_OUTPUT_DIR,
        packed_image_cache_npy=actor_packed_tensor(image_size),
        packed_image_cache_index_csv=(DEFAULT_ACTOR_CACHE_ROOT / "packed_image_cache_index.csv"),
        visual_context_cache_manifest_csv=DEFAULT_VISUAL_CONTEXT_MANIFEST,
        visual_context_packed_cache_npy=visual_packed_tensor(image_size),
        visual_context_packed_cache_index_csv=(DEFAULT_VISUAL_CACHE_ROOT / "packed_image_cache_index.csv"),
        require_cached_images=True,
        require_packed_visual_context=True,
        image_size=image_size,
        hidden_dim=int(FULL_DEFAULTS["hidden_dim"]),
        steps_per_fold=int(FULL_DEFAULTS["steps_per_fold"]),
        train_batch_size=int(FULL_DEFAULTS["train_batch_size"]),
        eval_batch_size=int(FULL_DEFAULTS["eval_batch_size"]),
        bootstrap_iterations=int(FULL_DEFAULTS["bootstrap_iterations"]),
        device=str(FULL_DEFAULTS["device"]),
        precision=str(FULL_DEFAULTS["precision"]),
        max_folds=None,
        train_per_class_per_fold=None,
        eval_per_class_per_fold=None,
        run_mode="full",
    )


def validate_full_execution_confirmation(
    config: FullMultimodalOofConfig,
    preflight_json: Path | None,
    authorization_json: Path | None,
    confirmed: bool,
) -> None:
    """Require matching clean preflight and explicit human authorization."""

    if not confirmed:
        raise ValueError("--full requires --confirm-full-run after reviewing the workload plan")
    if preflight_json is None or not preflight_json.exists():
        raise ValueError("--full requires an existing --preflight-json")
    if authorization_json is None or not authorization_json.exists():
        raise ValueError("--full requires an existing --authorization-json")
    preflight = json.loads(preflight_json.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_json.read_text(encoding="utf-8"))
    errors = validate_preflight_for_execution(config, preflight)
    errors.extend(validate_full_run_authorization(config, preflight, authorization))
    if errors:
        raise ValueError(f"full-run preflight execution gate failed: {errors}")


def validate_full_run_authorization(
    config: FullMultimodalOofConfig,
    preflight: dict[str, object],
    authorization: dict[str, object],
) -> list[str]:
    """Bind human approval to the same config and commit as the preflight."""

    errors: list[str] = []
    if authorization.get("schema_version") != FULL_RUN_AUTHORIZATION_SCHEMA_VERSION:
        errors.append(f"full_run_authorization_schema_version_mismatch={authorization.get('schema_version')}")
    if authorization.get("authorized") is not True:
        errors.append("full_run_authorization_requires_authorized_true")
    if authorization.get("purpose") != FULL_RUN_AUTHORIZATION_PURPOSE:
        errors.append(f"full_run_authorization_purpose_mismatch={authorization.get('purpose')}")
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
            f"full_run_authorization_preflight_hash_mismatch=preflight:{preflight_hash},authorization:{authorized_hash}"
        )
    if authorized_hash != expected_hash:
        errors.append(
            f"full_run_authorization_config_hash_mismatch=expected:{expected_hash},authorization:{authorized_hash}"
        )
    expected_commit = preflight.get("git_commit")
    authorized_commit = authorization.get("git_commit")
    if authorized_commit != expected_commit:
        errors.append(
            f"full_run_authorization_git_commit_mismatch=expected:{expected_commit},authorization:{authorized_commit}"
        )
    return errors
