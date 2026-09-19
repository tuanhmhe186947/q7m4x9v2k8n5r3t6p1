"""Run the exact cache, slot, fold, and repeatability gate for legacy L1."""

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
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
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
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

EXPECTED_FRAME_ROWS = 496
EXPECTED_WINDOW_ROWS = 310
EXPECTED_NATIVE_UNITS = 31
EXPECTED_IMAGE_SLOTS = 2728
EXPECTED_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _paths(packet_root: Path, image_size: int) -> dict[str, Path]:
    primary_context = packet_root / "08_image_context"
    repeat_context = packet_root / "08_image_context_repeat"
    primary_cache = packet_root / f"09_actor_cache_{image_size}"
    repeat_cache = packet_root / f"09_actor_cache_{image_size}_repeat"
    primary_folds = packet_root / "10_folds"
    repeat_folds = packet_root / "10_folds_repeat"
    tensor_name = f"packed_rgb_{image_size}_letterbox.npy"
    return {
        "selection": packet_root
        / "06_temporal_tier_contract"
        / "temporal_tier_selection_manifest.csv",
        "image_frames": primary_context / "image_frame_context_manifest.csv",
        "image_windows": primary_context / "image_window_context_manifest.csv",
        "image_audit": primary_context / "image_context_index_audit.json",
        "cache_manifest": primary_cache / "manifest.csv",
        "cache_audit": primary_cache / "cache_audit.json",
        "packed_index": primary_cache / "packed_image_cache_index.csv",
        "packed_tensor": primary_cache / tensor_name,
        "packed_audit": primary_cache / "packed_image_cache_audit.json",
        "recording_groups": primary_folds / "recording_group_manifest.csv",
        "native_folds": primary_folds / "native_oof_fold_manifest.csv",
        "window_folds": primary_folds / "window_oof_fold_manifest.csv",
        "class_support": primary_folds / "class_by_fold_support.csv",
        "source_support": primary_folds / "source_by_fold_support.csv",
        "fold_audit": primary_folds / "legacy_development_l1_fold_audit.json",
        "repeat_image_frames": repeat_context
        / "image_frame_context_manifest.csv",
        "repeat_image_windows": repeat_context
        / "image_window_context_manifest.csv",
        "repeat_image_audit": repeat_context
        / "image_context_index_audit.json",
        "repeat_cache_manifest": repeat_cache / "manifest.csv",
        "repeat_cache_audit": repeat_cache / "cache_audit.json",
        "repeat_packed_index": repeat_cache
        / "packed_image_cache_index.csv",
        "repeat_packed_tensor": repeat_cache / tensor_name,
        "repeat_packed_audit": repeat_cache
        / "packed_image_cache_audit.json",
        "repeat_recording_groups": repeat_folds
        / "recording_group_manifest.csv",
        "repeat_native_folds": repeat_folds / "native_oof_fold_manifest.csv",
        "repeat_window_folds": repeat_folds / "window_oof_fold_manifest.csv",
        "repeat_class_support": repeat_folds / "class_by_fold_support.csv",
        "repeat_source_support": repeat_folds / "source_by_fold_support.csv",
        "repeat_fold_audit": repeat_folds
        / "legacy_development_l1_fold_audit.json",
    }


def run_legacy_development_l1_audit(
    packet_root: Path,
    image_size: int,
) -> dict[str, Any]:
    paths = _paths(packet_root, image_size)
    errors: list[str] = []
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        errors.append(f"missing_artifacts={missing}")
        return _final_audit(
            packet_root=packet_root,
            image_size=image_size,
            paths=paths,
            errors=errors,
        )

    tables = {
        name: pd.read_csv(paths[name], low_memory=False)
        for name in (
            "selection",
            "image_frames",
            "image_windows",
            "cache_manifest",
            "packed_index",
            "recording_groups",
            "native_folds",
            "window_folds",
            "class_support",
            "source_support",
        )
    }
    try:
        relational = audit_legacy_l1_tables(
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
            image_size=image_size,
        )
    except Exception as exc:
        errors.append(f"relational_audit_error:{type(exc).__name__}:{exc}")
        relational = {}
    errors.extend(str(error) for error in relational.get("errors", []))
    errors.extend(_exact_count_errors(relational))

    tensor_audit = _audit_tensor_and_cache_pixels(
        cache_root=paths["cache_manifest"].parent,
        cache_manifest=tables["cache_manifest"],
        packed_index=tables["packed_index"],
        tensor_path=paths["packed_tensor"],
        image_size=image_size,
    )
    errors.extend(tensor_audit["errors"])
    loader_audit = _audit_strict_packed_loader(
        frame_context_csv=paths["image_frames"],
        window_context_csv=paths["image_windows"],
        packed_tensor=paths["packed_tensor"],
        packed_index=paths["packed_index"],
        image_size=image_size,
    )
    errors.extend(loader_audit["errors"])
    json_claim_audit = _audit_json_claims(paths)
    errors.extend(json_claim_audit["errors"])
    repeat_hash_audit = _audit_repeat_hashes(paths)
    errors.extend(repeat_hash_audit["errors"])
    return _final_audit(
        packet_root=packet_root,
        image_size=image_size,
        paths=paths,
        errors=errors,
        relational=relational,
        tensor_audit=tensor_audit,
        loader_audit=loader_audit,
        json_claim_audit=json_claim_audit,
        repeat_hash_audit=repeat_hash_audit,
    )


def _exact_count_errors(relational: dict[str, Any]) -> list[str]:
    expected = {
        "image_frame_rows": EXPECTED_FRAME_ROWS,
        "selection_rows": EXPECTED_WINDOW_ROWS,
        "image_window_rows": EXPECTED_WINDOW_ROWS,
        "native_fold_rows": EXPECTED_NATIVE_UNITS,
        "window_fold_rows": EXPECTED_WINDOW_ROWS,
        "total_selected_image_slots": EXPECTED_IMAGE_SLOTS,
        "fold_count": EXPECTED_FOLDS,
    }
    errors = [
        f"{name}_mismatch=expected:{value},observed:{relational.get(name)}"
        for name, value in expected.items()
        if relational.get(name) != value
    ]
    expected_labels = sorted(VALID_BEHAVIORS)
    if relational.get("native_behavior_labels") != expected_labels:
        errors.append(
            "native_behavior_labels_mismatch="
            f"{relational.get('native_behavior_labels')}"
        )
    if relational.get("native_source_types") != [LEGACY_SOURCE]:
        errors.append(
            "native_source_types_mismatch="
            f"{relational.get('native_source_types')}"
        )
    return errors


def _audit_tensor_and_cache_pixels(
    *,
    cache_root: Path,
    cache_manifest: pd.DataFrame,
    packed_index: pd.DataFrame,
    tensor_path: Path,
    image_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    tensor = np.load(tensor_path, mmap_mode="r")
    expected_shape = (EXPECTED_FRAME_ROWS, image_size, image_size, 3)
    if tensor.shape != expected_shape:
        errors.append(
            f"packed_tensor_shape_mismatch={tensor.shape}!={expected_shape}"
        )
    if tensor.dtype != np.uint8:
        errors.append(f"packed_tensor_dtype_mismatch={tensor.dtype}")
    manifest = cache_manifest.set_index("image_context_id", drop=False)
    pixel_mismatches = 0
    invalid_source_tensors = 0
    for row in packed_index.itertuples(index=False):
        context_id = str(row.image_context_id)
        if context_id not in manifest.index:
            pixel_mismatches += 1
            continue
        source_path = Path(str(manifest.loc[context_id, "cache_path"]))
        if not source_path.is_absolute():
            source_path = cache_root / source_path
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
        packed_row = int(row.packed_row)
        if packed_row >= len(tensor):
            pixel_mismatches += 1
        elif not np.array_equal(source, np.asarray(tensor[packed_row])):
            pixel_mismatches += 1
    if invalid_source_tensors:
        errors.append(f"invalid_source_cache_tensors={invalid_source_tensors}")
    if pixel_mismatches:
        errors.append(f"packed_pixel_mismatches={pixel_mismatches}")
    return {
        "tensor_shape": [int(value) for value in tensor.shape],
        "tensor_dtype": str(tensor.dtype),
        "all_cache_rows_checked": int(len(packed_index)),
        "invalid_source_cache_tensors": invalid_source_tensors,
        "packed_pixel_mismatches": pixel_mismatches,
        "errors": errors,
        "valid": not errors,
    }


def _audit_strict_packed_loader(
    *,
    frame_context_csv: Path,
    window_context_csv: Path,
    packed_tensor: Path,
    packed_index: Path,
    image_size: int,
) -> dict[str, Any]:
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            packed_image_cache_npy=packed_tensor,
            packed_image_cache_index_csv=packed_index,
            image_size=image_size,
            require_complete=True,
            require_cached_images=True,
            image_cache_size=0,
        )
    )
    errors: list[str] = []
    observed_slots = 0
    requested_slots = 0
    item_error_count = 0
    try:
        for index in range(len(dataset)):
            item = dataset[index]
            requested_slots += len(item["image_context_ids"])
            observed_slots += int(item["observed_mask"].sum().item())
            item_error_count += len(item["errors"])
        load_counters = dataset.image_load_audit()
    finally:
        dataset.close()
    if len(dataset) != EXPECTED_WINDOW_ROWS:
        errors.append(f"strict_loader_window_rows={len(dataset)}")
    if requested_slots != EXPECTED_IMAGE_SLOTS:
        errors.append(f"strict_loader_requested_slots={requested_slots}")
    if observed_slots != requested_slots:
        errors.append(
            f"strict_loader_observed_slots={observed_slots}/{requested_slots}"
        )
    if item_error_count:
        errors.append(f"strict_loader_item_errors={item_error_count}")
    if load_counters["source_image_loads"] != 0:
        errors.append(
            f"strict_loader_source_reads={load_counters['source_image_loads']}"
        )
    if load_counters["disk_image_cache_misses"] != 0:
        errors.append(
            "strict_loader_cache_misses="
            f"{load_counters['disk_image_cache_misses']}"
        )
    if load_counters["packed_image_cache_hits"] != requested_slots:
        errors.append(
            "strict_loader_packed_hits="
            f"{load_counters['packed_image_cache_hits']}/{requested_slots}"
        )
    return {
        "window_rows": int(len(dataset)),
        "requested_slots": requested_slots,
        "observed_slots": observed_slots,
        "item_error_count": item_error_count,
        "image_load_audit": load_counters,
        "errors": errors,
        "valid": not errors,
    }


def _audit_json_claims(paths: dict[str, Path]) -> dict[str, Any]:
    names = (
        "image_audit",
        "cache_audit",
        "packed_audit",
        "fold_audit",
        "repeat_image_audit",
        "repeat_cache_audit",
        "repeat_packed_audit",
        "repeat_fold_audit",
    )
    errors: list[str] = []
    rows: dict[str, Any] = {}
    for name in names:
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        scope_ok = payload.get("lineage_scope") == LEGACY_DEVELOPMENT_SCOPE
        reviewed_ok = payload.get("human_review_complete") is False
        artifact_errors = payload.get("errors", [])
        valid_ok = payload.get("valid", True) is not False
        if not scope_ok or not reviewed_ok or artifact_errors or not valid_ok:
            errors.append(f"invalid_json_claim_or_status={name}")
        rows[name] = {
            "sha256": file_sha256(paths[name]),
            "scope_ok": scope_ok,
            "human_review_complete_false": reviewed_ok,
            "upstream_errors": artifact_errors,
            "upstream_valid": valid_ok,
        }
    return {"artifacts": rows, "errors": errors, "valid": not errors}


def _audit_repeat_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    pairs = {
        "image_frames": ("image_frames", "repeat_image_frames"),
        "image_windows": ("image_windows", "repeat_image_windows"),
        "cache_manifest": ("cache_manifest", "repeat_cache_manifest"),
        "packed_index": ("packed_index", "repeat_packed_index"),
        "packed_tensor": ("packed_tensor", "repeat_packed_tensor"),
        "recording_groups": ("recording_groups", "repeat_recording_groups"),
        "native_folds": ("native_folds", "repeat_native_folds"),
        "window_folds": ("window_folds", "repeat_window_folds"),
        "class_support": ("class_support", "repeat_class_support"),
        "source_support": ("source_support", "repeat_source_support"),
    }
    errors: list[str] = []
    results: dict[str, Any] = {}
    for name, (primary_name, repeat_name) in pairs.items():
        primary_hash = file_sha256(paths[primary_name])
        repeat_hash = file_sha256(paths[repeat_name])
        equal = primary_hash == repeat_hash
        if not equal:
            errors.append(f"repeat_hash_mismatch={name}")
        results[name] = {
            "primary_sha256": primary_hash,
            "repeat_sha256": repeat_hash,
            "byte_identical": equal,
        }
    return {"pairs": results, "errors": errors, "valid": not errors}


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_entries": status}


def _final_audit(
    *,
    packet_root: Path,
    image_size: int,
    paths: dict[str, Path],
    errors: list[str],
    **sections: Any,
) -> dict[str, Any]:
    valid = not errors
    existing_hashes = {
        name: file_sha256(path)
        for name, path in paths.items()
        if path.exists()
    }
    return {
        "schema_version": "classification_v2.legacy_development_l1_gate.v1",
        "status": "PASS_LEGACY_DEVELOPMENT_L1" if valid else "FAIL_L1",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "model_training_authorized": False,
        "full_legacy_l2_data_build_authorized": valid,
        "packet_root": str(packet_root),
        "image_size": image_size,
        "git_state": _git_state(),
        "artifact_hashes": existing_hashes,
        **sections,
        "errors": errors,
        "valid": valid,
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json or (
        args.packet_root / "11_l1_audit" / "legacy_development_l1_audit.json"
    )
    require_output_paths_available([output_json], overwrite=args.overwrite)
    audit = run_legacy_development_l1_audit(
        args.packet_root,
        args.image_size,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
