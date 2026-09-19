"""Freeze the immutable legacy-only L3 development input snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.legacy_development_l3 import (
    LEGACY_L3_SCHEMA_VERSION,
    MASK_ONLY_SPATIAL_GROUPS,
    PREDICTIVE_SPATIAL_GROUPS,
    audit_legacy_feature_contract,
    audit_legacy_shortcuts,
    build_legacy_artifact_manifest,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--feature-contract-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "feature_audit": output_dir / "legacy_feature_contract_audit.json",
        "shortcut_audit": output_dir / "legacy_shortcut_audit.json",
        "artifact_manifest": output_dir
        / "legacy_development_lineage_manifest_v1.csv",
        "snapshot": output_dir / "legacy_development_input_snapshot_v1.json",
    }


def freeze_legacy_development_inputs(
    *,
    primary_root: Path,
    repeat_root: Path,
    feature_contract_json: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Write non-cyclic audits, hashes, and one immutable snapshot."""

    outputs = _output_paths(output_dir)
    require_output_paths_available(outputs.values(), overwrite=overwrite)
    paths = _input_paths(primary_root, repeat_root, feature_contract_json)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing L3 freeze inputs={missing}")

    contract = _read_json(feature_contract_json)
    enhanced = pd.read_csv(paths["enhanced_frames"], low_memory=False)
    feature_audit = audit_legacy_feature_contract(
        contract,
        available_frame_columns=enhanced.columns.tolist(),
    )
    if not feature_audit["valid"]:
        raise ValueError(
            "legacy feature contract failed: "
            + "; ".join(feature_audit["errors"])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(outputs["feature_audit"], feature_audit)

    temporal_views = {
        view_name: pd.read_csv(paths[f"temporal_view:{view_name}"], low_memory=False)
        for view_name in LEGACY_TEMPORAL_MODEL_VIEW_SPECS
    }
    shortcut_audit = audit_legacy_shortcuts(
        native_units=pd.read_csv(paths["native_units"], low_memory=False),
        temporal_selection=pd.read_csv(
            paths["temporal_selection"],
            low_memory=False,
        ),
        temporal_views=temporal_views,
        image_frames=pd.read_csv(paths["image_frames"], low_memory=False),
        enhanced_frames=enhanced,
        feature_contract_audit=feature_audit,
    )
    if not shortcut_audit["valid"]:
        raise ValueError(
            "legacy shortcut audit failed: "
            + "; ".join(shortcut_audit["errors"])
        )
    _write_json(outputs["shortcut_audit"], shortcut_audit)

    frozen_paths = {
        **paths,
        "feature_audit": outputs["feature_audit"],
        "shortcut_audit": outputs["shortcut_audit"],
    }
    artifacts = {
        name: (_artifact_kind(name), path)
        for name, path in frozen_paths.items()
    }
    manifest = build_legacy_artifact_manifest(artifacts)
    manifest.to_csv(outputs["artifact_manifest"], index=False)

    git_state = _git_state()
    snapshot = {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "status": "FROZEN_LEGACY_DEVELOPMENT_INPUTS_PRE_L3_GATE",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "model_training_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "git_state": git_state,
        "primary_root": primary_root.as_posix(),
        "repeat_root": repeat_root.as_posix(),
        "artifact_manifest_path": outputs["artifact_manifest"].as_posix(),
        "artifact_manifest_sha256": file_sha256(outputs["artifact_manifest"]),
        "feature_contract_path": feature_contract_json.as_posix(),
        "feature_contract_sha256": file_sha256(feature_contract_json),
        "feature_audit_path": outputs["feature_audit"].as_posix(),
        "feature_audit_sha256": file_sha256(outputs["feature_audit"]),
        "shortcut_audit_path": outputs["shortcut_audit"].as_posix(),
        "shortcut_audit_sha256": file_sha256(outputs["shortcut_audit"]),
        "frozen_contract": {
            "image_size": 160,
            "resize_policy": (
                "letterbox_preserve_aspect_rgb_pad_black_v1"
            ),
            "packed_cache_required": True,
            "source_media_fallback_allowed": False,
            "temporal_views": list(LEGACY_TEMPORAL_MODEL_VIEW_SPECS),
            "predictive_spatial_groups": list(PREDICTIVE_SPATIAL_GROUPS),
            "mask_only_spatial_groups": list(MASK_ONLY_SPATIAL_GROUPS),
            "predictive_whitelist_sha256": feature_audit[
                "predictive_whitelist_sha256"
            ],
            "mask_only_whitelist_sha256": feature_audit[
                "mask_only_whitelist_sha256"
            ],
            "blacklist_sha256": feature_audit["blacklist_sha256"],
            "fold_group_level": "recording_date",
            "source_probe_status": shortcut_audit["source_probe"]["status"],
            "availability_is_predictive_x": False,
        },
        "artifact_count": int(len(manifest)),
        "errors": [],
        "valid": True,
    }
    snapshot["snapshot_id"] = payload_sha256(snapshot)
    _write_json(outputs["snapshot"], snapshot)
    return snapshot


def _input_paths(
    primary_root: Path,
    repeat_root: Path,
    feature_contract_json: Path,
) -> dict[str, Path]:
    tier_root = primary_root / "06_temporal_tier_contract"
    paths = {
        "feature_contract": feature_contract_json,
        "l2_audit": primary_root
        / "08_l2_audit"
        / "legacy_development_l2_audit.json",
        "enhanced_frames": primary_root
        / "04_enhanced"
        / "spatiotemporal_frame_features_enhanced.csv",
        "window_features": primary_root
        / "05_windows"
        / "sequence_window_features.csv",
        "native_units": tier_root / "native_temporal_unit_manifest.csv",
        "temporal_selection": tier_root / "temporal_tier_selection_manifest.csv",
        "image_frames": primary_root
        / "09_image_context"
        / "image_frame_context_manifest.csv",
        "image_windows": primary_root
        / "09_image_context"
        / "image_window_context_manifest.csv",
        "image_context_audit": primary_root
        / "09_image_context"
        / "image_context_index_audit.json",
        "cache_manifest": primary_root / "10_actor_cache_160" / "manifest.csv",
        "cache_audit": primary_root / "10_actor_cache_160" / "cache_audit.json",
        "cache_policy_audit": primary_root
        / "10_actor_cache_160"
        / "cache_letterbox_policy_audit.json",
        "packed_tensor": primary_root
        / "10_actor_cache_160"
        / "packed_rgb_160_letterbox.npy",
        "packed_index": primary_root
        / "10_actor_cache_160"
        / "packed_image_cache_index.csv",
        "packed_audit": primary_root
        / "10_actor_cache_160"
        / "packed_image_cache_audit.json",
        "repeat_image_frames": repeat_root
        / "09_image_context"
        / "image_frame_context_manifest.csv",
        "repeat_image_windows": repeat_root
        / "09_image_context"
        / "image_window_context_manifest.csv",
        "repeat_image_context_audit": repeat_root
        / "09_image_context"
        / "image_context_index_audit.json",
    }
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        paths[f"temporal_view:{view_name}"] = tier_root / str(
            spec["slot_manifest_filename"]
        )
    for prefix, root in (("fold", primary_root), ("repeat_fold", repeat_root)):
        fold_root = root / "11_folds"
        paths[f"{prefix}:recording_groups"] = (
            fold_root / "recording_group_manifest.csv"
        )
        paths[f"{prefix}:native"] = fold_root / "native_oof_fold_manifest.csv"
        paths[f"{prefix}:window"] = fold_root / "window_oof_fold_manifest.csv"
        paths[f"{prefix}:class_support"] = fold_root / "class_by_fold_support.csv"
        paths[f"{prefix}:source_support"] = fold_root / "source_by_fold_support.csv"
        paths[f"{prefix}:audit"] = (
            fold_root / "legacy_development_l1_fold_audit.json"
        )
    return paths


def _artifact_kind(name: str) -> str:
    if name.startswith("temporal_view:"):
        return "temporal_view"
    if name.startswith(("fold:", "repeat_fold:")):
        return "fold"
    if name.startswith(("cache_", "packed_")):
        return "cache"
    if name in {"feature_contract", "feature_audit"}:
        return "feature_contract"
    if name == "shortcut_audit":
        return "shortcut_audit"
    return "lineage_input"


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_entries": dirty}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    snapshot = freeze_legacy_development_inputs(
        primary_root=args.primary_root,
        repeat_root=args.repeat_root,
        feature_contract_json=args.feature_contract_json,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
