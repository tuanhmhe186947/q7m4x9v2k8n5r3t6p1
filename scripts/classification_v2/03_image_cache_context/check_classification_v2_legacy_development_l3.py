"""Run the full immutable-input gate for legacy-only L3 development."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.legacy_development_l1 import (
    audit_legacy_l1_tables,
)
from pig_behavior.classification_v2.contracts.legacy_development_l3 import (
    LEGACY_L3_SCHEMA_VERSION,
    audit_legacy_feature_contract,
    audit_legacy_shortcuts,
    verify_legacy_artifact_manifest,
    verify_legacy_snapshot,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_SOURCE,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

EXPECTED_FRAME_ROWS = 72_864
EXPECTED_WINDOW_ROWS = 45_540
EXPECTED_NATIVE_UNITS = 4_554
EXPECTED_IMAGE_SLOTS = 400_752
EXPECTED_FOLDS = 12
EXPECTED_PREVIEWS = 64
EXPECTED_SOURCE_EQUIVALENCE_ROWS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--feature-contract-json", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reuse-pixel-evidence-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_legacy_development_l3_audit(
    *,
    primary_root: Path,
    repeat_root: Path,
    feature_contract_json: Path,
    freeze_dir: Path,
    reuse_pixel_evidence_json: Path | None = None,
) -> dict[str, Any]:
    """Independently verify all full L3 relationships, pixels, and hashes."""

    paths = _paths(
        primary_root,
        repeat_root,
        feature_contract_json,
        freeze_dir,
    )
    errors: list[str] = []
    missing = [
        name
        for name, path in paths.items()
        if (
            not path.is_dir()
            if name == "preview_root"
            else not path.is_file()
        )
    ]
    if missing:
        errors.append(f"missing_l3_artifacts={missing}")
        return _final_audit(
            primary_root=primary_root,
            repeat_root=repeat_root,
            paths=paths,
            errors=errors,
        )

    tables = {
        name: pd.read_csv(paths[name], low_memory=False)
        for name in (
            "native_units",
            "selection",
            "image_frames",
            "image_windows",
            "enhanced_frames",
            "cache_manifest",
            "packed_index",
            "recording_groups",
            "native_folds",
            "window_folds",
            "class_support",
            "source_support",
            "artifact_manifest",
        )
    }
    temporal_views = {
        view_name: pd.read_csv(paths[f"temporal_view:{view_name}"], low_memory=False)
        for view_name in LEGACY_TEMPORAL_MODEL_VIEW_SPECS
    }

    relational = _audit_relational(tables)
    errors.extend(relational["errors"])
    contract = _read_json(paths["feature_contract"])
    feature_audit = audit_legacy_feature_contract(
        contract,
        available_frame_columns=tables["enhanced_frames"].columns.tolist(),
    )
    errors.extend(feature_audit["errors"])
    stored_feature_audit = _read_json(paths["feature_audit"])
    if payload_sha256(feature_audit) != payload_sha256(stored_feature_audit):
        errors.append("stored_feature_audit_does_not_match_recomputation")

    shortcut_audit = audit_legacy_shortcuts(
        native_units=tables["native_units"],
        temporal_selection=tables["selection"],
        temporal_views=temporal_views,
        image_frames=tables["image_frames"],
        enhanced_frames=tables["enhanced_frames"],
        feature_contract_audit=feature_audit,
    )
    errors.extend(shortcut_audit["errors"])
    stored_shortcut_audit = _read_json(paths["shortcut_audit"])
    if payload_sha256(shortcut_audit) != payload_sha256(stored_shortcut_audit):
        errors.append("stored_shortcut_audit_does_not_match_recomputation")

    cache_contract = _audit_cache_jsons_and_previews(paths)
    errors.extend(cache_contract["errors"])
    if reuse_pixel_evidence_json is None:
        packed_loader = _audit_all_packed_pixels_and_loader(
            frame_context_csv=paths["image_frames"],
            window_context_csv=paths["image_windows"],
            cache_manifest=tables["cache_manifest"],
            packed_index=tables["packed_index"],
            packed_tensor=paths["packed_tensor"],
            image_size=160,
        )
        packed_loader["reused"] = False
    else:
        packed_loader = _reuse_packed_pixel_evidence(
            reuse_pixel_evidence_json,
            paths,
        )
    errors.extend(packed_loader["errors"])

    artifact_manifest = verify_legacy_artifact_manifest(
        tables["artifact_manifest"]
    )
    errors.extend(artifact_manifest["errors"])
    snapshot = _read_json(paths["snapshot"])
    snapshot_audit = verify_legacy_snapshot(
        snapshot,
        artifact_manifest_path=paths["artifact_manifest"],
        feature_contract_path=paths["feature_contract"],
        feature_audit_path=paths["feature_audit"],
        shortcut_audit_path=paths["shortcut_audit"],
    )
    errors.extend(snapshot_audit["errors"])
    repeat_inputs = _audit_repeat_input_hashes(paths)
    errors.extend(repeat_inputs["errors"])
    l2_boundary = _audit_l2_boundary(paths)
    errors.extend(l2_boundary["errors"])

    return _final_audit(
        primary_root=primary_root,
        repeat_root=repeat_root,
        paths=paths,
        errors=errors,
        relational=relational,
        feature_contract_audit=feature_audit,
        shortcut_audit=shortcut_audit,
        cache_contract_audit=cache_contract,
        packed_pixel_and_loader_audit=packed_loader,
        artifact_manifest_audit=artifact_manifest,
        snapshot_audit=snapshot_audit,
        repeat_input_hash_audit=repeat_inputs,
        l2_boundary_audit=l2_boundary,
    )


def _audit_relational(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        audit = audit_legacy_l1_tables(
            temporal_selection=tables["selection"],
            image_frames=tables["image_frames"],
            image_windows=tables["image_windows"],
            cache_manifest=tables["cache_manifest"],
            packed_index=tables["packed_index"],
            recording_groups=tables["recording_groups"],
            native_folds=tables["native_folds"],
            window_folds=tables["window_folds"],
            class_support=tables["class_support"],
            source_support=tables["source_support"],
            image_size=160,
        )
    except Exception as exc:
        return {
            "errors": [f"relational_audit_error:{type(exc).__name__}:{exc}"],
            "valid": False,
        }
    errors.extend(str(value) for value in audit.get("errors", []))
    expected = {
        "image_frame_rows": EXPECTED_FRAME_ROWS,
        "selection_rows": EXPECTED_WINDOW_ROWS,
        "image_window_rows": EXPECTED_WINDOW_ROWS,
        "native_fold_rows": EXPECTED_NATIVE_UNITS,
        "window_fold_rows": EXPECTED_WINDOW_ROWS,
        "total_selected_image_slots": EXPECTED_IMAGE_SLOTS,
        "fold_count": EXPECTED_FOLDS,
    }
    for name, value in expected.items():
        if audit.get(name) != value:
            errors.append(
                f"{name}_mismatch=expected:{value},observed:{audit.get(name)}"
            )
    if audit.get("native_behavior_labels") != sorted(VALID_BEHAVIORS):
        errors.append("native_behavior_labels_mismatch")
    if audit.get("native_source_types") != [LEGACY_SOURCE]:
        errors.append("native_source_types_mismatch")
    return {**audit, "errors": errors, "valid": not errors}


def _audit_cache_jsons_and_previews(paths: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    payloads: dict[str, Any] = {}
    for name in ("cache_audit", "packed_audit", "image_context_audit"):
        payload = _read_json(paths[name])
        payloads[name] = payload
        if payload.get("lineage_scope") != LEGACY_DEVELOPMENT_SCOPE:
            errors.append(f"{name}_lineage_scope_mismatch")
        if payload.get("human_review_complete") is not False:
            errors.append(f"{name}_must_remain_unreviewed")
        if payload.get("failed_rows", 0) != 0:
            errors.append(f"{name}_failed_rows={payload.get('failed_rows')}")
        if payload.get("valid", True) is False:
            errors.append(f"{name}_invalid")
    policy = _read_json(paths["cache_policy_audit"])
    payloads["cache_policy_audit"] = policy
    if policy.get("valid") is not True or policy.get("errors"):
        errors.append("cache_policy_audit_invalid")
    if policy.get("manifest_rows") != EXPECTED_FRAME_ROWS:
        errors.append(
            f"cache_policy_manifest_rows={policy.get('manifest_rows')}"
        )
    if (
        policy.get("source_equivalence_checked")
        != EXPECTED_SOURCE_EQUIVALENCE_ROWS
    ):
        errors.append(
            "cache_policy_source_equivalence_rows="
            f"{policy.get('source_equivalence_checked')}"
        )
    if policy.get("source_equivalence_mismatches") != 0:
        errors.append(
            "cache_policy_source_equivalence_mismatches="
            f"{policy.get('source_equivalence_mismatches')}"
        )
    if policy.get("letterbox_geometry_invalid_rows") != 0:
        errors.append(
            "cache_policy_letterbox_invalid_rows="
            f"{policy.get('letterbox_geometry_invalid_rows')}"
        )
    cache = payloads["cache_audit"]
    if cache.get("manifest_rows") != EXPECTED_FRAME_ROWS:
        errors.append(f"cache_manifest_rows={cache.get('manifest_rows')}")
    if cache.get("image_size") != 160:
        errors.append(f"cache_image_size={cache.get('image_size')}")
    if cache.get("cache_format") != "npy_uint8_rgb_hwc":
        errors.append(f"cache_format={cache.get('cache_format')}")
    if cache.get("resize_policy") != (
        "letterbox_preserve_aspect_rgb_pad_black_v1"
    ):
        errors.append(f"cache_resize_policy={cache.get('resize_policy')}")
    packed = payloads["packed_audit"]
    if packed.get("shape") != [EXPECTED_FRAME_ROWS, 160, 160, 3]:
        errors.append(f"packed_audit_shape={packed.get('shape')}")
    if packed.get("dtype") != "uint8":
        errors.append(f"packed_audit_dtype={packed.get('dtype')}")
    if packed.get("verification_mismatches") != 0:
        errors.append(
            "packed_audit_verification_mismatches="
            f"{packed.get('verification_mismatches')}"
        )
    preview_paths = _preview_jpg_paths(paths["preview_root"])
    if len(preview_paths) < EXPECTED_PREVIEWS:
        errors.append(f"inspectable_preview_count={len(preview_paths)}")
    return {
        "payloads": payloads,
        "preview_count": len(preview_paths),
        "preview_sample": [path.as_posix() for path in preview_paths[:8]],
        "errors": errors,
        "valid": not errors,
    }


def _preview_jpg_paths(root: Path) -> list[Path]:
    """Find inspectable previews under the cache's source/video/pig shards."""

    return sorted(path for path in root.rglob("*.jpg") if path.is_file())


def _audit_all_packed_pixels_and_loader(
    *,
    frame_context_csv: Path,
    window_context_csv: Path,
    cache_manifest: pd.DataFrame,
    packed_index: pd.DataFrame,
    packed_tensor: Path,
    image_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    tensor = np.load(packed_tensor, mmap_mode="r")
    expected_shape = (EXPECTED_FRAME_ROWS, image_size, image_size, 3)
    if tensor.shape != expected_shape:
        errors.append(f"packed_tensor_shape={tensor.shape}")
    if tensor.dtype != np.uint8:
        errors.append(f"packed_tensor_dtype={tensor.dtype}")
    manifest = cache_manifest.set_index("image_context_id", drop=False)
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            packed_image_cache_npy=packed_tensor,
            packed_image_cache_index_csv=packed_tensor.parent
            / "packed_image_cache_index.csv",
            image_size=image_size,
            require_complete=True,
            require_cached_images=True,
            image_cache_size=0,
        )
    )
    pixel_mismatches = 0
    invalid_source_tensors = 0
    loader_failures = 0
    checked_rows = 0
    try:
        for row in packed_index.itertuples(index=False):
            context_id = str(row.image_context_id)
            if context_id not in manifest.index:
                pixel_mismatches += 1
                continue
            frame = dataset.frame_by_context_id.get(context_id)
            loaded = (
                dataset._load_context_image(context_id, frame)
                if frame is not None
                else None
            )
            if loaded is None:
                loader_failures += 1
                continue
            source_path = Path(str(manifest.loc[context_id, "cache_path"]))
            if not source_path.is_absolute():
                source_path = packed_tensor.parent / source_path
            try:
                source = np.load(source_path)
            except Exception:
                invalid_source_tensors += 1
                continue
            if source.shape != (image_size, image_size, 3):
                invalid_source_tensors += 1
                continue
            if source.dtype != np.uint8:
                invalid_source_tensors += 1
                continue
            expected = np.transpose(
                source.astype(np.float32) / 255.0,
                (2, 0, 1),
            )
            if not np.array_equal(loaded, expected):
                pixel_mismatches += 1
            checked_rows += 1
            if checked_rows % 10_000 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "packed_pixel_loader_audit",
                            "checked_rows": checked_rows,
                            "expected_rows": EXPECTED_FRAME_ROWS,
                            "pixel_mismatches": pixel_mismatches,
                            "loader_failures": loader_failures,
                        }
                    ),
                    flush=True,
                )
        load_counters = dataset.image_load_audit()
    finally:
        dataset.close()
    if checked_rows != EXPECTED_FRAME_ROWS:
        errors.append(f"all_pixel_checked_rows={checked_rows}")
    if invalid_source_tensors:
        errors.append(f"invalid_source_cache_tensors={invalid_source_tensors}")
    if loader_failures:
        errors.append(f"packed_loader_failures={loader_failures}")
    if pixel_mismatches:
        errors.append(f"packed_pixel_mismatches={pixel_mismatches}")
    if load_counters["packed_image_cache_hits"] != EXPECTED_FRAME_ROWS:
        errors.append(
            "packed_loader_hits="
            f"{load_counters['packed_image_cache_hits']}"
        )
    if load_counters["source_image_loads"] != 0:
        errors.append(
            f"packed_loader_source_reads={load_counters['source_image_loads']}"
        )
    if load_counters["disk_image_cache_misses"] != 0:
        errors.append(
            "packed_loader_cache_misses="
            f"{load_counters['disk_image_cache_misses']}"
        )
    return {
        "tensor_shape": [int(value) for value in tensor.shape],
        "tensor_dtype": str(tensor.dtype),
        "all_pixel_checked_rows": checked_rows,
        "invalid_source_cache_tensors": invalid_source_tensors,
        "packed_loader_failures": loader_failures,
        "packed_pixel_mismatches": pixel_mismatches,
        "image_load_audit": load_counters,
        "source_media_fallback_reads": load_counters["source_image_loads"],
        "errors": errors,
        "valid": not errors,
    }


def _reuse_packed_pixel_evidence(
    evidence_json: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Reuse a full pixel pass only under exact code and artifact bindings."""

    prior = _read_json(evidence_json)
    section = dict(prior.get("packed_pixel_and_loader_audit", {}))
    errors: list[str] = []
    if prior.get("schema_version") != LEGACY_L3_SCHEMA_VERSION:
        errors.append("reused_pixel_evidence_schema_mismatch")
    if prior.get("status") != "PASS_LEGACY_DEVELOPMENT_L3":
        errors.append("reused_pixel_evidence_is_not_l3_pass")
    if prior.get("valid") is not True:
        errors.append("reused_pixel_evidence_invalid")
    checker_sha = file_sha256(Path(__file__))
    if prior.get("checker_source_sha256") != checker_sha:
        errors.append("reused_pixel_checker_source_sha_mismatch")
    current_packed_sha = file_sha256(paths["packed_tensor"])
    if prior.get("packed_tensor_sha256") != current_packed_sha:
        errors.append("reused_pixel_packed_tensor_sha_mismatch")
    prior_hashes = prior.get("artifact_hashes", {})
    binding_names = (
        "cache_manifest",
        "packed_index",
        "packed_audit",
        "image_frames",
        "image_windows",
    )
    binding_matches: dict[str, bool] = {}
    for name in binding_names:
        current_sha = file_sha256(paths[name])
        matches = prior_hashes.get(name) == current_sha
        binding_matches[name] = matches
        if not matches:
            errors.append(f"reused_pixel_artifact_sha_mismatch={name}")
    expected = {
        "all_pixel_checked_rows": EXPECTED_FRAME_ROWS,
        "invalid_source_cache_tensors": 0,
        "packed_loader_failures": 0,
        "packed_pixel_mismatches": 0,
        "source_media_fallback_reads": 0,
    }
    for name, value in expected.items():
        if section.get(name) != value:
            errors.append(
                f"reused_pixel_field_mismatch={name}:"
                f"{section.get(name)}!={value}"
            )
    counters = section.get("image_load_audit", {})
    if counters.get("packed_image_cache_hits") != EXPECTED_FRAME_ROWS:
        errors.append("reused_pixel_packed_hit_count_mismatch")
    if counters.get("disk_image_cache_misses") != 0:
        errors.append("reused_pixel_cache_misses_nonzero")
    if counters.get("source_image_loads") != 0:
        errors.append("reused_pixel_source_reads_nonzero")
    if section.get("valid") is not True or section.get("errors"):
        errors.append("reused_pixel_section_invalid")
    return {
        **section,
        "reused": True,
        "reused_evidence_path": evidence_json.as_posix(),
        "reused_evidence_sha256": file_sha256(evidence_json),
        "checker_source_sha256_match": (
            prior.get("checker_source_sha256") == checker_sha
        ),
        "artifact_sha_matches": binding_matches,
        "errors": errors,
        "valid": not errors,
    }


def _audit_repeat_input_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    pairs = {
        "image_frames": ("image_frames", "repeat_image_frames"),
        "image_windows": ("image_windows", "repeat_image_windows"),
        "recording_groups": (
            "recording_groups",
            "repeat_recording_groups",
        ),
        "native_folds": ("native_folds", "repeat_native_folds"),
        "window_folds": ("window_folds", "repeat_window_folds"),
        "class_support": ("class_support", "repeat_class_support"),
        "source_support": ("source_support", "repeat_source_support"),
    }
    errors: list[str] = []
    results: dict[str, Any] = {}
    for name, (primary_name, repeat_name) in pairs.items():
        primary_sha = file_sha256(paths[primary_name])
        repeat_sha = file_sha256(paths[repeat_name])
        equal = primary_sha == repeat_sha
        if not equal:
            errors.append(f"repeat_fold_hash_mismatch={name}")
        results[name] = {
            "primary_sha256": primary_sha,
            "repeat_sha256": repeat_sha,
            "byte_identical": equal,
        }
    return {"pairs": results, "errors": errors, "valid": not errors}


def _audit_l2_boundary(paths: dict[str, Path]) -> dict[str, Any]:
    payload = _read_json(paths["l2_audit"])
    errors: list[str] = []
    if payload.get("valid") is not True:
        errors.append("l2_boundary_invalid")
    if payload.get("lineage_scope") != LEGACY_DEVELOPMENT_SCOPE:
        errors.append("l2_boundary_scope_mismatch")
    if payload.get("human_review_complete") is not False:
        errors.append("l2_boundary_must_remain_unreviewed")
    if payload.get("model_training_authorized") is not False:
        errors.append("l2_boundary_must_not_authorize_training")
    return {
        "path": str(paths["l2_audit"]),
        "sha256": file_sha256(paths["l2_audit"]),
        "status": payload.get("status"),
        "errors": errors,
        "valid": not errors,
    }


def _paths(
    primary_root: Path,
    repeat_root: Path,
    feature_contract_json: Path,
    freeze_dir: Path,
) -> dict[str, Path]:
    tier_root = primary_root / "06_temporal_tier_contract"
    cache_root = primary_root / "10_actor_cache_160"
    folds = primary_root / "11_folds"
    repeat_folds = repeat_root / "11_folds"
    paths = {
        "feature_contract": feature_contract_json,
        "feature_audit": freeze_dir / "legacy_feature_contract_audit.json",
        "shortcut_audit": freeze_dir / "legacy_shortcut_audit.json",
        "artifact_manifest": freeze_dir
        / "legacy_development_lineage_manifest_v1.csv",
        "snapshot": freeze_dir / "legacy_development_input_snapshot_v1.json",
        "l2_audit": primary_root
        / "08_l2_audit"
        / "legacy_development_l2_audit.json",
        "native_units": tier_root / "native_temporal_unit_manifest.csv",
        "selection": tier_root / "temporal_tier_selection_manifest.csv",
        "enhanced_frames": primary_root
        / "04_enhanced"
        / "spatiotemporal_frame_features_enhanced.csv",
        "image_frames": primary_root
        / "09_image_context"
        / "image_frame_context_manifest.csv",
        "image_windows": primary_root
        / "09_image_context"
        / "image_window_context_manifest.csv",
        "image_context_audit": primary_root
        / "09_image_context"
        / "image_context_index_audit.json",
        "cache_manifest": cache_root / "manifest.csv",
        "cache_audit": cache_root / "cache_audit.json",
        "cache_policy_audit": cache_root / "cache_letterbox_policy_audit.json",
        "packed_tensor": cache_root / "packed_rgb_160_letterbox.npy",
        "packed_index": cache_root / "packed_image_cache_index.csv",
        "packed_audit": cache_root / "packed_image_cache_audit.json",
        "preview_root": cache_root / "preview_jpg_160_letterbox",
        "recording_groups": folds / "recording_group_manifest.csv",
        "native_folds": folds / "native_oof_fold_manifest.csv",
        "window_folds": folds / "window_oof_fold_manifest.csv",
        "class_support": folds / "class_by_fold_support.csv",
        "source_support": folds / "source_by_fold_support.csv",
        "fold_audit": folds / "legacy_development_l1_fold_audit.json",
        "repeat_recording_groups": repeat_folds
        / "recording_group_manifest.csv",
        "repeat_native_folds": repeat_folds / "native_oof_fold_manifest.csv",
        "repeat_window_folds": repeat_folds / "window_oof_fold_manifest.csv",
        "repeat_class_support": repeat_folds / "class_by_fold_support.csv",
        "repeat_source_support": repeat_folds / "source_by_fold_support.csv",
        "repeat_fold_audit": repeat_folds
        / "legacy_development_l1_fold_audit.json",
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
    return paths


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


def _final_audit(
    *,
    primary_root: Path,
    repeat_root: Path,
    paths: dict[str, Path],
    errors: list[str],
    **sections: Any,
) -> dict[str, Any]:
    valid = not errors
    return {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L3" if valid else "FAIL_L3",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "l4_model_correctness_authorized": valid,
        "bounded_model_correctness_training_authorized": valid,
        "primary_root": primary_root.as_posix(),
        "repeat_root": repeat_root.as_posix(),
        "git_state": _git_state(),
        "checker_source_sha256": file_sha256(Path(__file__)),
        "artifact_hashes": {
            name: file_sha256(path)
            for name, path in paths.items()
            if path.is_file() and name != "packed_tensor"
        },
        "packed_tensor_sha256": (
            file_sha256(paths["packed_tensor"])
            if paths["packed_tensor"].is_file()
            else None
        ),
        **sections,
        "errors": errors,
        "valid": valid,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    audit = run_legacy_development_l3_audit(
        primary_root=args.primary_root,
        repeat_root=args.repeat_root,
        feature_contract_json=args.feature_contract_json,
        freeze_dir=args.freeze_dir,
        reuse_pixel_evidence_json=args.reuse_pixel_evidence_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
