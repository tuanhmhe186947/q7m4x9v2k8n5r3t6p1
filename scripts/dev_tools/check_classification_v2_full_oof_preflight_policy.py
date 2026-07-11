from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.full_multimodal_oof import FullMultimodalOofConfig
from pig_behavior.classification_v2.training.full_run_preflight import _canonical_full_run_path_errors


def main() -> None:
    """Check fail-closed path policy for full OOF preflight without training."""

    parser = argparse.ArgumentParser(description="Check classification_v2 full OOF preflight path policy.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_preflight_policy_audit.json"),
    )
    args = parser.parse_args()
    good_config = _config(
        output_dir=Path("outputs/classification_v2/model_full/full_multimodal_oof"),
        actor_root=Path("outputs/classification_v2/image_cache_v2_letterbox"),
        visual_root=Path("outputs/classification_v2/visual_interaction_cache"),
    )
    bad_config = _config(
        output_dir=Path("outputs/classification_v2/model_smoke/full_multimodal_oof_resume_smoke"),
        actor_root=Path("outputs/classification_v2/image_cache_v2_resume_smoke"),
        visual_root=Path("outputs/classification_v2/visual_interaction_cache_smoke"),
    )
    good_errors = _canonical_full_run_path_errors(good_config)
    bad_errors = _canonical_full_run_path_errors(bad_config)
    required_bad_tokens = [
        "full_run_output_dir_must_not_be_smoke_or_pilot",
        "packed_actor_cache_must_use_canonical_letterbox_tensor",
        "packed_actor_cache_index_must_use_canonical_letterbox_index",
        "visual_context_manifest_must_use_canonical_cache",
        "packed_visual_context_must_use_canonical_letterbox_tensor",
        "packed_visual_context_index_must_use_canonical_letterbox_index",
    ]
    missing_bad_tokens = [token for token in required_bad_tokens if not any(token in err for err in bad_errors)]
    errors: list[str] = []
    if good_errors:
        errors.append(f"canonical_config_rejected={good_errors}")
    if missing_bad_tokens:
        errors.append(f"ad_hoc_config_not_rejected={missing_bad_tokens}")
    audit = {
        "schema_version": "classification_v2_full_oof_preflight_policy_audit_v1",
        "canonical_config_errors": good_errors,
        "ad_hoc_config_errors": bad_errors,
        "required_bad_token_count": len(required_bad_tokens),
        "missing_bad_tokens": missing_bad_tokens,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _config(*, output_dir: Path, actor_root: Path, visual_root: Path) -> FullMultimodalOofConfig:
    return FullMultimodalOofConfig(
        output_dir=output_dir,
        packed_image_cache_npy=actor_root / "packed_rgb_64_letterbox.npy",
        packed_image_cache_index_csv=actor_root / "packed_image_cache_index.csv",
        require_cached_images=True,
        visual_context_cache_manifest_csv=visual_root / "visual_context_manifest.csv",
        visual_context_packed_cache_npy=visual_root / "packed_rgb_64_letterbox.npy",
        visual_context_packed_cache_index_csv=visual_root / "packed_image_cache_index.csv",
        require_packed_visual_context=True,
        image_size=64,
        hidden_dim=48,
        run_mode="full",
        device="cuda",
        precision="amp",
    )


if __name__ == "__main__":
    main()
