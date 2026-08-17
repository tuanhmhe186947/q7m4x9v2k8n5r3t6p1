"""Materialize the local FULL-T6 canonical 46D binding.

This is an explicit, fail-closed task runner rather than a training script.
It copies the existing CVAT rows from the immutable 18377 bundle and invokes
the current executable spatial producer only for the missing legacy rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.features.motion_schema import (  # noqa: E402
    motion_schema_metadata,
)
from pig_behavior.classification_v2.features.spatial_schema import (  # noqa: E402
    spatial_schema_hash,
    spatial_schema_metadata,
)
from pig_behavior.classification_v2.spatial_sequence_export import (  # noqa: E402
    export_spatial_sequences,
)

SCHEMA_SHA256 = (
    "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
)
TEMPORAL_ROOT = (
    PROJECT_ROOT
    / ".codex_worktrees"
    / "temporal_semantics_rebuild_20260813"
    / "outputs"
    / "classification_v2"
    / "temporal_semantics_rebuild_v2_full_chunk1024"
)
TEMPORAL_MANIFEST = TEMPORAL_ROOT / "full_temporal_window_manifest_release.csv"
TEMPORAL_AUTHORITY = TEMPORAL_ROOT / "temporal_semantics_authority_v2.json"
TEMPORAL_HASH_MANIFEST = (
    TEMPORAL_ROOT / "temporal_v2_artifact_hash_manifest.json"
)
SOURCE_FEATURES = Path(
    "C:/pig_runs/classification_v2_reviewed_rebuild_20260802_v1/"
    "candidates/behavior_decision_apply_af95297/reviewed_frame_features.csv"
)
OLD_BUNDLE = Path(
    "C:/pig_runs/classification_v2_reviewed_rebuild_20260802_v1/"
    "candidates/train_ready/spatial_memmap_bundle"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/classification_v2/full_t6_canonical_46d_20260816"
AUTHORITY_PATH = PROJECT_ROOT / (
    "docs/classification_v2/full_t6_46d_final_authority_20260817.json"
)

EXPECTED_TARGETS = 33_287
EXPECTED_CVAT = 28_748
EXPECTED_LEGACY = 4_539
EXPECTED_T6_LENGTH = 6

PREDICTIVE_NAMES = [
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "vx_n_per_second",
    "vy_n_per_second",
    "bw_rate_n_per_second",
    "bh_rate_n_per_second",
    "area_rate_n_per_second",
    "aspect_ratio_rate_per_second",
    "speed_n_per_second",
    "direction_change_rad",
    "tangential_acceleration_n_per_second2",
    "ax_n_per_second2",
    "ay_n_per_second2",
    "acceleration_vector_magnitude_n_per_second2",
    "roi_feeder_min_dist_n",
    "roi_feeder_max_overlap_ratio",
    "roi_feeder_max_iou",
    "roi_feeder_center_inside",
    "roi_feeder_near",
    "roi_feeder_contact",
    "roi_drinker_min_dist_n",
    "roi_drinker_max_overlap_ratio",
    "roi_drinker_max_iou",
    "roi_drinker_center_inside",
    "roi_drinker_near",
    "roi_drinker_contact",
    "roi_toy_min_dist_n",
    "roi_toy_max_overlap_ratio",
    "roi_toy_max_iou",
    "roi_toy_center_inside",
    "roi_toy_near",
    "roi_toy_contact",
    "nearest_dist_n",
    "nearest_pair_iou",
    "nearest_pair_overlap_ratio",
    "social_density_near_count",
    "social_contact_count",
    "partner_distance_delta_n",
    "approach_speed_n_per_second",
    "retreat_speed_n_per_second",
    "pair_contact_with_nearest",
    "aggression_score_proxy_per_second",
]

DERIVATION_NAMES = [
    "timestamp_sec",
    "image_width",
    "image_height",
    "pen_boundary_inward_normal_x",
    "pen_boundary_inward_normal_y",
    "pen_context_available",
    "pen_center_inside",
    "bbox_valid",
    "actor_bbox_valid",
    "geometry_feature_valid",
    "spatiotemporal_feature_valid",
    "roi_feeder_available",
    "roi_drinker_available",
    "roi_toy_available",
    "social_neighbor_available",
    "valid_motion_pair",
    "velocity_valid",
    "bbox_rate_valid",
    "direction_valid",
    "direction_change_valid",
    "tangential_acceleration_valid",
    "vector_acceleration_valid",
    "motion_feature_available",
    "velocity_sample_time_sec",
    "acceleration_delta_t_sec",
    "social_context_valid",
]
SOURCE_USECOLS = [
    "source_type",
    "object_track_key",
    "frame_index",
    "temporal_unit_key",
    "nearest_partner_key",
    *PREDICTIVE_NAMES,
    *DERIVATION_NAMES,
]

ARRAY_NAMES = [
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
    "social_relation",
    "length_mask",
    "observed_mask",
    "spatial_quality_mask",
    "roi_validity_mask",
    "social_validity_mask",
    "social_feature_validity_mask",
    "motion_feature_validity_mask",
    "pen_validity_mask",
    "adjacent_motion_pair_mask",
    "sparse_velocity_pair_mask",
    "valid_motion_pair_mask",
    "velocity_valid_mask",
    "bbox_rate_valid_mask",
    "direction_valid_mask",
    "direction_change_valid_mask",
    "tangential_acceleration_valid_mask",
    "vector_acceleration_valid_mask",
    "motion_feature_available_mask",
    "frame_index_sequence",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_list(value: str) -> list[Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
    return parsed


def _target_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row["source_type"],
        row["dataset_id"],
        row["video_key"],
        row["object_track_key"],
        tuple(int(value) for value in _json_list(row["selected_frame_indices"])),
    )


def _read_t6_targets() -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    with TEMPORAL_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["view_id"] != "T6":
                continue
            target_id = row["target_id"]
            if target_id in seen:
                raise ValueError(f"duplicate current T6 target={target_id}")
            seen.add(target_id)
            frames = tuple(
                int(value) for value in _json_list(row["selected_frame_indices"])
            )
            if len(frames) != EXPECTED_T6_LENGTH:
                raise ValueError(f"non-T6 target length={target_id}:{frames}")
            record = dict(row)
            record["frames"] = frames
            record["key"] = _target_key(row)
            targets.append(record)
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError(f"FULL-T6 target count={len(targets)}")
    counts = Counter(row["source_type"] for row in targets)
    if counts != Counter(
        {"cvat_tracking_xml": EXPECTED_CVAT, "legacy_recovered": EXPECTED_LEGACY}
    ):
        raise ValueError(f"FULL-T6 source counts={counts}")
    return targets


def _read_split_roles(targets: list[dict[str, Any]]) -> None:
    path = TEMPORAL_ROOT / "target_split_roles_release.csv"
    roles: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            roles[row["target_id"]] = (row["outer_fold_id"], row["split"])
    missing = [row["target_id"] for row in targets if row["target_id"] not in roles]
    if missing:
        raise ValueError(f"missing FULL-T6 split roles={missing[:5]}")
    for row in targets:
        row["outer_fold_id"], row["split"] = roles[row["target_id"]]


def _required_legacy_keys(
    targets: list[dict[str, Any]],
) -> set[tuple[str, int]]:
    return {
        (str(row["object_track_key"]), frame)
        for row in targets
        if row["source_type"] == "legacy_recovered"
        for frame in row["frames"]
    }


def _load_legacy_source(
    required: set[tuple[str, int]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SOURCE_FEATURES.is_file():
        raise FileNotFoundError(SOURCE_FEATURES)
    chunks: list[pd.DataFrame] = []
    matched: set[tuple[str, int]] = set()
    total_rows = 0
    usecols = list(dict.fromkeys(SOURCE_USECOLS))
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            SOURCE_FEATURES,
            usecols=usecols,
            dtype=str,
            chunksize=100_000,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)
        chunk = chunk.loc[chunk["source_type"] == "legacy_recovered"].copy()
        chunk["frame_index"] = pd.to_numeric(
            chunk["frame_index"], errors="coerce"
        )
        chunk = chunk.loc[chunk["frame_index"].notna()].copy()
        chunk["frame_index"] = chunk["frame_index"].astype(int)
        keys = list(
            zip(
                chunk["object_track_key"].astype(str),
                chunk["frame_index"],
                strict=True,
            )
        )
        keep = [key in required for key in keys]
        if any(keep):
            selected = chunk.loc[keep].copy()
            selected_keys = [
                key for key, flag in zip(keys, keep, strict=True) if flag
            ]
            overlap = matched.intersection(selected_keys)
            if overlap:
                raise ValueError(
                    f"duplicate legacy source keys={sorted(overlap)[:5]}"
                )
            matched.update(selected_keys)
            chunks.append(selected)
        if chunk_number % 10 == 0:
            print(
                f"SOURCE_SCAN_CHUNK={chunk_number} "
                f"ROWS_SCANNED={total_rows} MATCHED={len(matched)}",
                flush=True,
            )
    missing = required - matched
    if missing:
        raise ValueError(f"unresolved legacy source keys={sorted(missing)[:10]}")
    frames = pd.concat(chunks, ignore_index=True)
    if len(frames) != len(required):
        raise ValueError(
            f"legacy source row count={len(frames)} expected={len(required)}"
        )
    frames = _coerce_string_boolean_columns(frames)
    frames["frame_index"] = frames["frame_index"].astype(int)
    return frames, {
        "required_frame_rows": len(required),
        "resolved_frame_rows": len(frames),
        "missing_frame_rows": len(missing),
        "duplicate_source_keys": False,
        "source_rows_scanned": total_rows,
    }


def _coerce_string_boolean_columns(frames: pd.DataFrame) -> pd.DataFrame:
    """Keep CSV boolean flags boolean for the canonical exporter.

    ``read_csv(dtype=str)`` yields a pandas string dtype on the current
    runtime.  The shared exporter handles numeric and bool dtypes, but its
    legacy object-dtype compatibility path does not see this dtype.  Coerce
    only columns whose non-missing values are exactly True/False; predictive
    numeric strings and identifiers remain unchanged.
    """
    result = frames.copy()
    missing_tokens = {"", "nan", "none", "<na>"}
    for column in result.columns:
        normalized = result[column].astype(str).str.strip().str.lower()
        present = normalized[~normalized.isin(missing_tokens)]
        if not present.empty and set(present.unique()).issubset(
            {"true", "false"}
        ):
            result[column] = normalized.map({"true": True, "false": False})
    return result


def _build_windows(
    targets: list[dict[str, Any]],
    frames: pd.DataFrame,
) -> pd.DataFrame:
    frame_times = {
        (str(row["object_track_key"]), int(row["frame_index"])): row[
            "timestamp_sec"
        ]
        for _, row in frames.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for record in targets:
        if record["source_type"] != "legacy_recovered":
            continue
        key = str(record["object_track_key"])
        frames_for_target = record["frames"]
        timestamps: list[float] = []
        for frame in frames_for_target:
            value = pd.to_numeric(
                frame_times.get((key, frame)), errors="coerce"
            )
            if pd.isna(value):
                raise ValueError(
                    f"missing legacy timestamp={record['target_id']}:{frame}"
                )
            timestamps.append(float(value))
        deltas = [
            current - previous
            for previous, current in zip(
                timestamps, timestamps[1:], strict=True
            )
        ]
        rows.append(
            {
                "window_id": record["target_id"],
                "object_track_key": key,
                "window_start_frame": frames_for_target[0],
                "window_end_frame": frames_for_target[-1],
                "window_length_frames": EXPECTED_T6_LENGTH,
                "feature_computation_grain": "FINAL_VIEW_FEATURES",
                "pair_scope_key": record["target_id"],
                "view_type": "T6_contiguous",
                "sampling_pattern": "contiguous",
                "selected_frame_offsets": json.dumps(
                    [value - frames_for_target[0] for value in frames_for_target]
                ),
                "selected_frame_indices": json.dumps(list(frames_for_target)),
                "selected_timestamps_seconds": json.dumps(timestamps),
                "pair_delta_frames": json.dumps([1] * (EXPECTED_T6_LENGTH - 1)),
                "pair_delta_seconds": json.dumps(deltas),
                "pair_recomputed_for_view": "True",
                "aggregate_recomputed_for_view": "True",
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_LEGACY:
        raise ValueError(f"legacy windows={len(result)}")
    return result


def _read_old_rows() -> dict[tuple[Any, ...], int]:
    path = OLD_BUNDLE / "spatial_memmap_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("spatial_schema_hash") != SCHEMA_SHA256:
        raise ValueError("old CVAT bundle schema is not executable 18377")
    window_path = Path(str(manifest["source_audit"]))
    audit = json.loads(window_path.read_text(encoding="utf-8"))
    manifest_csv = Path(str(audit["window_manifest_csv"]))
    index: dict[tuple[Any, ...], int] = {}
    with manifest_csv.open(encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            key = _target_key(
                {
                    "source_type": row["source_type"],
                    "dataset_id": row["dataset_id"],
                    "video_key": row["video_key"],
                    "object_track_key": row["object_track_key"],
                    "selected_frame_indices": row["selected_frame_indices"],
                }
            )
            if key in index:
                raise ValueError(f"duplicate old spatial key={key}")
            index[key] = row_index
    if len(index) != int(manifest["row_count"]):
        raise ValueError(
            f"old row index={len(index)} expected={manifest['row_count']}"
        )
    return index


def _load_old_arrays() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(
        (OLD_BUNDLE / "spatial_memmap_manifest.json").read_text(encoding="utf-8")
    )
    if list(manifest["ordered_array_names"]) != ARRAY_NAMES:
        raise ValueError("old spatial array order drift")
    arrays = {
        name: np.load(OLD_BUNDLE / "arrays" / f"{name}.npy", mmap_mode="r")
        for name in ARRAY_NAMES
    }
    for name, array in arrays.items():
        if int(array.shape[0]) != int(manifest["row_count"]):
            raise ValueError(f"old array row count drift={name}")
    return arrays, manifest


def _write_row_manifest(path: Path, targets: list[dict[str, Any]]) -> None:
    fields = [
        "row_index",
        "target_id",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "behavior",
        "outer_fold_id",
        "split",
        "native_unit_id",
        "matched_support_id",
        "physical_frame_ids_json",
        "observed_mask_json",
        "source_row_kind",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, row in enumerate(targets):
            observed = row.get("observed_mask", [True] * EXPECTED_T6_LENGTH)
            writer.writerow(
                {
                    "row_index": row_index,
                    "target_id": row["target_id"],
                    "source_type": row["source_type"],
                    "dataset_id": row["dataset_id"],
                    "video_key": row["video_key"],
                    "object_track_key": row["object_track_key"],
                    "behavior": row["behavior"],
                    "outer_fold_id": row["outer_fold_id"],
                    "split": row["split"],
                    "native_unit_id": row.get("native_unit_id", ""),
                    "matched_support_id": row.get("matched_support_id", ""),
                    "physical_frame_ids_json": json.dumps(list(row["frames"])),
                    "observed_mask_json": json.dumps(list(observed)),
                    "source_row_kind": row["source_row_kind"],
                }
            )


def _array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    value = np.ascontiguousarray(np.asarray(array))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _materialize() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output root={OUTPUT_ROOT}"
        )
    for path in [TEMPORAL_MANIFEST, TEMPORAL_AUTHORITY, TEMPORAL_HASH_MANIFEST]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if spatial_schema_hash() != SCHEMA_SHA256:
        raise ValueError("active executable spatial schema hash drift")

    print("RESOLVE_TARGETS=START", flush=True)
    targets = _read_t6_targets()
    _read_split_roles(targets)
    source_counts = Counter(row["source_type"] for row in targets)
    print(
        f"RESOLVE_TARGETS=PASS COUNT={len(targets)} "
        f"CVAT={source_counts['cvat_tracking_xml']} "
        f"LEGACY={source_counts['legacy_recovered']}",
        flush=True,
    )

    required_legacy_keys = _required_legacy_keys(targets)
    print(
        f"LEGACY_SOURCE_RESOLVE=START REQUIRED_FRAME_KEYS={len(required_legacy_keys)}",
        flush=True,
    )
    legacy_frames, source_audit = _load_legacy_source(required_legacy_keys)
    print(
        f"LEGACY_SOURCE_RESOLVE=PASS FRAME_ROWS={len(legacy_frames)}",
        flush=True,
    )

    windows = _build_windows(targets, legacy_frames)
    print("LEGACY_PRODUCER=START ROWS=4539", flush=True)
    legacy_export = export_spatial_sequences(
        windows,
        legacy_frames,
        max_window_length=EXPECTED_T6_LENGTH,
        motion_schema_manifest=motion_schema_metadata(),
        spatial_schema_manifest=spatial_schema_metadata(),
    )
    if legacy_export.audit.get("missing_frame_slots") != 0:
        raise ValueError(
            f"legacy producer missing slots={legacy_export.audit.get('missing_frame_slots')}"
        )
    if legacy_export.audit.get("spatial_schema_hash") != SCHEMA_SHA256:
        raise ValueError("legacy producer schema hash drift")
    print("LEGACY_PRODUCER=PASS ROWS=4539 WIDTH=46", flush=True)

    old_index = _read_old_rows()
    old_arrays, old_manifest = _load_old_arrays()
    cvat_targets = [
        row for row in targets if row["source_type"] == "cvat_tracking_xml"
    ]
    legacy_targets = [
        row for row in targets if row["source_type"] == "legacy_recovered"
    ]
    missing_cvat = [row["target_id"] for row in cvat_targets if row["key"] not in old_index]
    if missing_cvat:
        raise ValueError(f"CVAT rows missing from immutable bundle={missing_cvat[:5]}")
    legacy_index = {row["target_id"]: index for index, row in enumerate(legacy_targets)}

    legacy_arrays = legacy_export.arrays
    if set(legacy_arrays) != set(ARRAY_NAMES):
        raise ValueError(
            f"legacy array names drift={sorted(set(legacy_arrays) ^ set(ARRAY_NAMES))}"
        )
    merged: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        old_array = old_arrays[name]
        tail_shape = tuple(int(value) for value in old_array.shape[2:])
        merged[name] = np.empty(
            (EXPECTED_TARGETS, EXPECTED_T6_LENGTH, *tail_shape),
            dtype=old_array.dtype,
        )

    old_indices: list[int] = []
    cvat_positions: list[int] = []
    for row_index, row in enumerate(targets):
        if row["source_type"] == "cvat_tracking_xml":
            old_row = old_index[row["key"]]
            old_indices.append(old_row)
            cvat_positions.append(row_index)
            old_frames = np.asarray(old_arrays["frame_index_sequence"][old_row, :6])
            if not np.array_equal(old_frames, np.asarray(row["frames"], dtype=np.int32)):
                raise ValueError(f"CVAT frame-order mismatch={row['target_id']}")
            row["observed_mask"] = [
                bool(value)
                for value in old_arrays["observed_mask"][old_row, :6].tolist()
            ]
            row["source_row_kind"] = "immutable_existing_cvat"
            for name in ARRAY_NAMES:
                merged[name][row_index] = old_arrays[name][old_row, :6]
        else:
            legacy_row = legacy_index[row["target_id"]]
            row["observed_mask"] = [
                bool(value)
                for value in legacy_arrays["observed_mask"][legacy_row].tolist()
            ]
            row["source_row_kind"] = "computed_legacy_18377"
            for name in ARRAY_NAMES:
                merged[name][row_index] = legacy_arrays[name][legacy_row]
    if len(old_indices) != EXPECTED_CVAT:
        raise ValueError(f"mapped CVAT rows={len(old_indices)}")

    cvat_value_parity: dict[str, bool] = {}
    for name in ARRAY_NAMES:
        before = np.asarray(old_arrays[name][old_indices, :6])
        after = np.asarray(merged[name][cvat_positions])
        cvat_value_parity[name] = bool(np.array_equal(before, after))
    if not all(cvat_value_parity.values()):
        raise ValueError(f"existing CVAT value parity failed={cvat_value_parity}")

    target_ids = [row["target_id"] for row in targets]
    if len(set(target_ids)) != EXPECTED_TARGETS:
        raise ValueError("duplicate final target IDs")
    if not np.array_equal(
        merged["frame_index_sequence"],
        np.asarray([row["frames"] for row in targets], dtype=np.int32),
    ):
        raise ValueError("final frame-order parity failed")
    if not np.isfinite(merged["bbox_xywh_n"]).all():
        raise ValueError("non-finite final geometry tensor")
    if not np.isfinite(merged["motion_delta"]).all():
        raise ValueError("non-finite final motion tensor")
    if not np.isfinite(merged["roi_class_relation"]).all():
        raise ValueError("non-finite final ROI tensor")
    if not np.isfinite(merged["social_relation"]).all():
        raise ValueError("non-finite final social tensor")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".full_t6_canonical_46d_",
            dir=str(OUTPUT_ROOT.parent),
        )
    )
    try:
        np.savez_compressed(
            temporary_root / "legacy_4539_spatial_sequences.npz",
            **{name: legacy_arrays[name] for name in ARRAY_NAMES},
        )
        np.savez_compressed(
            temporary_root / "full_t6_canonical_46d.npz",
            **merged,
        )
        _write_row_manifest(
            temporary_root / "full_t6_row_manifest.csv",
            targets,
        )
        _write_row_manifest(
            temporary_root / "legacy_4539_row_manifest.csv",
            legacy_targets,
        )
        temporal_hash_manifest = json.loads(
            TEMPORAL_HASH_MANIFEST.read_text(encoding="utf-8")
        )
        temporal_authority_sha = _sha256_file(TEMPORAL_AUTHORITY)
        temporal_manifest_sha = _sha256_file(TEMPORAL_MANIFEST)
        expected_authority_sha = (
            "c9f4ba4ffa6ebae7405d13eaccc481097f9810d7be2385c1c9715aea04524681"
        )
        if temporal_authority_sha != expected_authority_sha:
            raise ValueError("Temporal-v2 authority hash drift")
        expected_temporal_manifest_sha = (
            temporal_hash_manifest["artifacts"][
                "full_temporal_window_manifest_release.csv"
            ]["sha256"]
        )
        if temporal_manifest_sha != expected_temporal_manifest_sha:
            raise ValueError("Temporal-v2 full manifest hash drift")
        final_npz = temporary_root / "full_t6_canonical_46d.npz"
        legacy_npz = temporary_root / "legacy_4539_spatial_sequences.npz"
        row_manifest = temporary_root / "full_t6_row_manifest.csv"
        legacy_manifest = temporary_root / "legacy_4539_row_manifest.csv"
        build_evidence = {
            "schema_version": "pig.classification_v2.full_t6_canonical_46d.v1",
            "status": "PASS",
            "active_schema_sha256": SCHEMA_SHA256,
            "d890_revoked": True,
            "target_counts": {
                "total": EXPECTED_TARGETS,
                "cvat_tracking_xml": EXPECTED_CVAT,
                "legacy_recovered": EXPECTED_LEGACY,
            },
            "legacy_source_audit": source_audit,
            "legacy_export_audit": legacy_export.audit,
            "existing_cvat_recomputed": False,
            "existing_cvat_value_parity": cvat_value_parity,
            "existing_cvat_row_count": len(cvat_positions),
            "legacy_computed_row_count": len(legacy_targets),
            "legacy_unresolved_row_count": 0,
            "target_id_sha256": _sha256_json(target_ids),
            "frame_order_parity": True,
            "mask_alignment": True,
            "feature_width": 46,
            "group_order": [
                ["bbox_xywh_n", 4],
                ["bbox_shape_n", 2],
                ["motion_delta", 12],
                ["roi_class_relation", 18],
                ["social_relation", 10],
            ],
            "temporal_v2": {
                "authority_path": str(TEMPORAL_AUTHORITY),
                "authority_sha256": temporal_authority_sha,
                "full_manifest_path": str(TEMPORAL_MANIFEST),
                "full_manifest_sha256": temporal_manifest_sha,
                "full_t6_count": EXPECTED_TARGETS,
                "population": "FULL_NONOVERLAP_VIEW_POOL",
            },
            "source_features": {
                "path": str(SOURCE_FEATURES),
                "size_bytes": SOURCE_FEATURES.stat().st_size,
                "sha256": _sha256_file(SOURCE_FEATURES),
            },
            "immutable_cvat_bundle": {
                "manifest_path": str(OLD_BUNDLE / "spatial_memmap_manifest.json"),
                "manifest_sha256": _sha256_file(
                    OLD_BUNDLE / "spatial_memmap_manifest.json"
                ),
                "tensor_content_hash": old_manifest["spatial_tensor_content_hash"],
            },
            "artifacts": {
                "final_npz": {
                    "path": str(OUTPUT_ROOT / final_npz.name),
                    "sha256": _sha256_file(final_npz),
                    "size_bytes": final_npz.stat().st_size,
                },
                "legacy_npz": {
                    "path": str(OUTPUT_ROOT / legacy_npz.name),
                    "sha256": _sha256_file(legacy_npz),
                    "size_bytes": legacy_npz.stat().st_size,
                },
                "row_manifest": {
                    "path": str(OUTPUT_ROOT / row_manifest.name),
                    "sha256": _sha256_file(row_manifest),
                    "size_bytes": row_manifest.stat().st_size,
                },
                "legacy_manifest": {
                    "path": str(OUTPUT_ROOT / legacy_manifest.name),
                    "sha256": _sha256_file(legacy_manifest),
                    "size_bytes": legacy_manifest.stat().st_size,
                },
            },
            "output_root": str(OUTPUT_ROOT),
            "runtime": {
                "gpu_used": False,
                "studio_started": False,
                "model_training_runs": 0,
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=PROJECT_ROOT,
                    text=True,
                ).strip(),
                "producer_path": "src/pig_behavior/classification_v2/spatial_sequence_export.py",
            },
        }
        evidence_path = temporary_root / "build_evidence.json"
        evidence_path.write_text(
            json.dumps(build_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_root / "legacy_export_audit.json").write_text(
            json.dumps(legacy_export.audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return build_evidence


def _write_final_authority(evidence: dict[str, Any]) -> str:
    if AUTHORITY_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite existing authority={AUTHORITY_PATH}"
        )
    AUTHORITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    authority = {
        "schema_version": "pig.classification_v2.full_t6_46d_authority.v1",
        "authority_id": "FULL-T6-CANONICAL-46D-20260817",
        "status": "PASS",
        "active_executable_schema_sha256": SCHEMA_SHA256,
        "d890_revoked": True,
        "population": {
            "name": "FULL-T6",
            "pool": "FULL_NONOVERLAP_VIEW_POOL",
            "total_targets": EXPECTED_TARGETS,
            "cvat_count": EXPECTED_CVAT,
            "legacy_recovered_count": EXPECTED_LEGACY,
        },
        "artifact_root": evidence["output_root"],
        "final_artifact": evidence["artifacts"]["final_npz"],
        "row_manifest": evidence["artifacts"]["row_manifest"],
        "legacy_artifact": evidence["artifacts"]["legacy_npz"],
        "legacy_manifest": evidence["artifacts"]["legacy_manifest"],
        "build_evidence": {
            "path": str(OUTPUT_ROOT / "build_evidence.json"),
            "sha256": _sha256_file(OUTPUT_ROOT / "build_evidence.json"),
        },
        "parity": {
            "missing_targets": 0,
            "duplicate_targets": 0,
            "extra_targets": 0,
            "target_id_parity": "PASS",
            "t6_frame_order_parity": "PASS",
            "mask_alignment": "PASS",
            "source_count_parity": "PASS",
            "existing_cvat_value_parity": "PASS",
            "feature_width": 46,
            "group_order": "6/12/18/10",
            "group_order_detail": "4/2/12/18/10",
        },
        "scientific_boundaries": {
            "existing_cvat_recomputed": False,
            "legacy_rows_zero_filled": False,
            "temporal_v2_mutated": False,
            "labels_or_membership_changed": False,
            "gpu_used": False,
            "studio_started": False,
            "model_training_runs": 0,
        },
        "producer": {
            "module": "pig_behavior.classification_v2.spatial_sequence_export",
            "function": "export_spatial_sequences",
            "producer_path": evidence["runtime"]["producer_path"],
            "git_head": evidence["runtime"]["git_head"],
        },
    }
    AUTHORITY_PATH.write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256_file(AUTHORITY_PATH)


def main() -> int:
    evidence = _materialize()
    authority_sha = _write_final_authority(evidence)
    print("FULL_T6_CANONICAL_46D_READY=YES", flush=True)
    print(f"FINAL_46D_ARTIFACT_SHA256={evidence['artifacts']['final_npz']['sha256']}")
    print(f"FINAL_46D_AUTHORITY_SHA256={authority_sha}")
    print("GPU_USED=NO")
    print("STUDIO_STARTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
