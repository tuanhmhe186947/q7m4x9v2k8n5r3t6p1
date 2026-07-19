"""Build versioned Pig-STRENet and causal-history artifacts.

The command is an artifact canary, not a trainer launcher.  It reads one
immutable frame-feature CSV and writes a fresh output directory containing
pair/slot manifests, safe numeric history features, all-class ROI dynamics,
geometry-selected top-K social edges, optional stabilized RGB differences, and
lineage hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.pig_strenet_media import (
    FrameMediaResolver,
    crop_rgb_box,
)
from pig_behavior.classification_v2.features.pig_strenet_artifacts import (
    availability_columns,
    build_pig_strenet_artifacts,
    compute_stabilized_difference_maps,
    model_x_columns,
)

SCHEMA_VERSION = "classification_v2.pig_strenet_artifact_run.v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--target-length", type=int, default=6)
    parser.add_argument("--legacy-target-starts", default="6")
    parser.add_argument("--top-k-neighbors", type=int, default=3)
    parser.add_argument("--roi-coco", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument("--legacy-crop-root", type=Path, default=Path("."))
    parser.add_argument("--max-open-videos", type=int, default=2)
    parser.add_argument("--max-cached-frames", type=int, default=32)
    parser.add_argument("--max-native-events", type=int, default=None)
    parser.add_argument("--difference-size", type=int, default=64)
    parser.add_argument("--visual-size", type=int, default=64)
    parser.add_argument("--skip-difference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output exists; use a fresh root or --overwrite: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True)
    if args.difference_size <= 0:
        raise ValueError("--difference-size must be positive")
    if args.visual_size <= 0:
        raise ValueError("--visual-size must be positive")

    frames = pd.read_csv(args.input_csv, low_memory=False)
    target_unit_keys = _select_target_unit_keys(frames, args.max_native_events)

    starts = tuple(
        int(value.strip())
        for value in args.legacy_target_starts.split(",")
        if value.strip()
    )
    if not starts:
        raise ValueError("--legacy-target-starts must not be empty")

    artifacts = build_pig_strenet_artifacts(
        frames,
        history_length=args.history_length,
        target_length=args.target_length,
        legacy_target_starts=starts,
        top_k_neighbors=args.top_k_neighbors,
        target_unit_keys=target_unit_keys,
        roi_coco_path=args.roi_coco,
    )
    _write_csv(artifacts.pair_manifest, args.output_dir / "pair_manifest.csv")
    _write_csv(artifacts.slot_manifest, args.output_dir / "slot_manifest.csv")
    _write_csv(artifacts.history_features, args.output_dir / "history_features.csv")
    _write_csv(artifacts.roi_dynamics, args.output_dir / "roi_dynamics.csv")
    _write_csv(
        artifacts.roi_visual_selection,
        args.output_dir / "roi_visual_selection.csv",
    )
    _write_csv(artifacts.social_nodes, args.output_dir / "social_nodes.csv")
    _write_csv(artifacts.social_edges, args.output_dir / "social_edges.csv")
    _write_csv(artifacts.control_matrix, args.output_dir / "history_control_matrix.csv")
    with FrameMediaResolver(
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
        max_open_videos=args.max_open_videos,
        max_cached_frames=args.max_cached_frames,
    ) as media:
        tensor_audit = _write_packed_tensors(
            artifacts,
            frames,
            args.output_dir,
            media=media,
            visual_size=args.visual_size,
        )
    (args.output_dir / "feature_whitelist.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "model_x_columns": model_x_columns(artifacts.history_features),
                "availability_only_columns": availability_columns(
                    artifacts.history_features
                ),
                "audit_only_columns": [
                    "history_frame_count",
                    "history_expected_frame_count",
                    "history_available_ratio",
                    "history_complete",
                    "history_gap_count",
                    "history_duration_sec",
                    "target_duration_sec",
                    "history_target_gap_sec",
                ],
                "forbidden_inputs": [
                    "labels",
                    "review_metadata",
                    "paths",
                    "ids",
                    "source_identity",
                    "target_selected_roi",
                    "future_frames",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    difference_audit: dict[str, Any]
    if args.skip_difference:
        difference_audit = {
            "status": "SKIPPED_EXPLICITLY",
            "maps_path": None,
            "summary_path": None,
        }
    else:
        difference_audit = _write_difference_artifacts(
            artifacts.slot_manifest,
            args.output_dir,
            frames=frames,
            media=media,
            image_size=args.difference_size,
        )
    media_audit = media.write_manifest(args.output_dir / "media_manifest.json")

    audit = dict(artifacts.audit)
    audit["difference"] = difference_audit
    audit["packed_tensors"] = tensor_audit
    audit["media"] = {
        "manifest_path": str(args.output_dir / "media_manifest.json"),
        "manifest_sha256": _sha256(args.output_dir / "media_manifest.json"),
        "source_file_count": media_audit["source_file_count"],
        "runtime_counts": media_audit["runtime_counts"],
        "status_counts": media_audit["status_counts"],
        "valid": media_audit["valid"],
    }
    difference_contract_valid = args.skip_difference or (
        difference_audit.get("status") in {"PASS", "PASS_WITH_NATURAL_SKIPS"}
        and difference_audit.get("missing_available_frame_slots", 0) == 0
    )
    roi_status = tensor_audit["roi_visual_pixels"].get("status")
    roi_contract_valid = roi_status in {
        "PASS",
        "NOT_APPLICABLE_NO_VALID_ROI_GEOMETRY",
    }
    audit["media_contract_valid"] = bool(
        media_audit["valid"] and difference_contract_valid and roi_contract_valid
    )
    audit["valid"] = bool(audit.get("valid")) and bool(
        audit["media_contract_valid"]
    )
    audit["input_csv"] = str(args.input_csv)
    audit["input_sha256"] = _sha256(args.input_csv)
    audit["input_frame_rows"] = int(len(frames))
    audit["target_unit_count"] = (
        None if target_unit_keys is None else len(target_unit_keys)
    )
    audit["parameters"] = {
        "history_length": args.history_length,
        "target_length": args.target_length,
        "legacy_target_starts": list(starts),
        "top_k_neighbors": args.top_k_neighbors,
        "roi_coco": None if args.roi_coco is None else str(args.roi_coco),
        "video_root": str(args.video_root),
        "legacy_crop_root": str(args.legacy_crop_root),
        "max_open_videos": args.max_open_videos,
        "max_cached_frames": args.max_cached_frames,
        "max_native_events": args.max_native_events,
        "difference_size": args.difference_size,
        "visual_size": args.visual_size,
    }
    audit_path = args.output_dir / "pig_strenet_artifact_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = _write_artifact_manifest(args.output_dir)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "bounded_artifact_canary",
        "lineage_scope": _lineage_scope(frames),
        "human_review_complete": _review_claim(frames),
        "input": {"path": str(args.input_csv), "sha256": _sha256(args.input_csv)},
        "output_dir": str(args.output_dir),
        "audit_path": str(audit_path),
        "resolved_config": {
            key: _jsonable(value) for key, value in vars(args).items()
            if key != "overwrite"
        },
        "implementation": _implementation_lineage(),
        "environment": _environment_lineage(),
        "artifact_manifest": artifact_manifest,
        "training_started": False,
        "oof_started": False,
        "data_modified": False,
        "skills": [
            "computer-vision-opencv",
            "dataset-contract-leakage-guard",
            "experiment-lineage-reproducibility",
            "safe-refactor-test-guardian",
        ],
        "valid": bool(audit.get("valid")),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def _select_target_unit_keys(
    frames: pd.DataFrame,
    max_native_events: int | None,
) -> set[str] | None:
    """Select target units without truncating the source frame table."""

    if max_native_events is None:
        return None
    if max_native_events <= 0:
        raise ValueError("--max-native-events must be positive")
    keys = frames["temporal_unit_key"].astype(str).drop_duplicates().head(
        max_native_events
    )
    return set(keys)


def _write_difference_artifacts(
    slots: pd.DataFrame,
    output_dir: Path,
    *,
    frames: pd.DataFrame,
    media: FrameMediaResolver,
    image_size: int,
) -> dict[str, Any]:
    maps: list[np.ndarray] = []
    summaries: list[pd.DataFrame] = []
    pair_ids: list[str] = []
    skipped = 0
    available_frame_slots = 0
    missing_available_frame_slots = 0
    provenance_rows: list[dict[str, Any]] = []
    frame_lookup = {
        str(value): row for value, row in frames.set_index("frame_uid").iterrows()
    }
    for pair_id, group in slots.groupby("pair_id", sort=False):
        ordered = group.sort_values("global_slot_index")
        crops: list[np.ndarray] = []
        valid: list[bool] = []
        for row in ordered.itertuples(index=False):
            source = frame_lookup.get(str(row.frame_uid))
            result = media.read_actor(source, image_size=image_size)
            image = result.image_rgb
            pixel_valid = bool(row.frame_available and result.available)
            if bool(row.frame_available):
                available_frame_slots += 1
                if not pixel_valid:
                    missing_available_frame_slots += 1
            crops.append(
                image
                if image is not None
                else np.zeros((image_size, image_size, 3), dtype=np.uint8)
            )
            valid.append(pixel_valid)
            provenance_rows.append(
                {
                    "pair_id": str(pair_id),
                    "global_slot_index": int(row.global_slot_index),
                    "slot_role": str(row.slot_role),
                    "frame_uid": str(row.frame_uid),
                    "frame_available": bool(row.frame_available),
                    "pixel_available": pixel_valid,
                    **result.provenance(),
                }
            )
        if sum(valid) < 2:
            skipped += 1
            continue
        diff_maps, summary, pair_valid = compute_stabilized_difference_maps(
            np.stack(crops),
            np.asarray(valid, dtype=bool),
        )
        summary.insert(0, "pair_id", str(pair_id))
        summary["pair_valid"] = pair_valid
        maps.append(diff_maps)
        summaries.append(summary)
        pair_ids.append(str(pair_id))
    maps_path = output_dir / "stabilized_difference_maps_f32.npy"
    summary_path = output_dir / "stabilized_difference_summary.csv"
    if maps:
        np.save(maps_path, np.stack(maps).astype(np.float32))
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(summary_path, index=False, lineterminator="\n")
    else:
        np.save(maps_path, np.zeros((0, 0, image_size, image_size), dtype=np.float32))
        pd.DataFrame(
            columns=["pair_id", "pair_slot_index", "pair_valid"]
        ).to_csv(summary_path, index=False, lineterminator="\n")
    provenance_path = output_dir / "difference_pixel_index.csv"
    pd.DataFrame.from_records(provenance_rows).to_csv(
        provenance_path,
        index=False,
        lineterminator="\n",
    )
    if not pair_ids:
        status = "BLOCKED_NO_ACTOR_PIXELS"
    elif missing_available_frame_slots:
        status = "PASS_WITH_MISSING_ACTOR_PIXELS"
    elif skipped:
        status = "PASS_WITH_NATURAL_SKIPS"
    else:
        status = "PASS"
    return {
        "status": status,
        "pairs_written": len(pair_ids),
        "pairs_skipped_missing_crops": skipped,
        "available_frame_slots": available_frame_slots,
        "missing_available_frame_slots": missing_available_frame_slots,
        "maps_shape": list(np.load(maps_path, mmap_mode="r").shape),
        "maps_path": str(maps_path),
        "summary_path": str(summary_path),
        "provenance_path": str(provenance_path),
        "maps_sha256": _sha256(maps_path),
        "summary_sha256": _sha256(summary_path),
        "provenance_sha256": _sha256(provenance_path),
    }


def _write_packed_tensors(
    artifacts: Any,
    frames: pd.DataFrame,
    output_dir: Path,
    *,
    media: FrameMediaResolver,
    visual_size: int,
) -> dict[str, Any]:
    roi_columns = [
        "available",
        "min_dist_n",
        "overlap_ratio",
        "iou",
        "center_inside",
        "near",
        "contact",
        "entry",
        "exit",
        "contact_run_length",
        "motion_inside",
    ]
    roi = artifacts.roi_dynamics.copy()
    roi_values = roi[roi_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    roi_values_path = output_dir / "roi_dynamics_values_f32.npy"
    np.save(roi_values_path, roi_values.to_numpy(dtype=np.float32))
    roi_index = roi[
        ["pair_id", "native_event_id", "slot_index", "slot_role", "roi_class"]
    ].copy()
    roi_index.insert(0, "tensor_row", np.arange(len(roi_index), dtype=np.int64))
    roi_index.to_csv(
        output_dir / "roi_dynamics_index.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "roi_dynamics_feature_names.json").write_text(
        json.dumps(roi_columns, indent=2) + "\n",
        encoding="utf-8",
    )

    social_columns = [
        "distance_n",
        "relative_dx_n",
        "relative_dy_n",
        "relative_speed_n",
        "relative_angle",
        "approach_speed_n",
        "separation_speed_n",
        "pair_iou",
        "pair_overlap_ratio",
        "pair_contact",
        "pair_contact_duration_frames",
        "pair_contact_duration_sec",
        "pair_motion_energy",
        "pair_contact_motion_intensity",
        "partner_available_ratio",
        "partner_persistence_ratio",
        "partner_switch_count",
        "partner_key_consistency",
        "pair_valid_ratio",
    ]
    social = artifacts.social_edges.copy()
    social_values = (
        social[social_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    social_values_path = output_dir / "social_edges_values_f32.npy"
    social_mask_path = output_dir / "social_edges_mask_bool.npy"
    np.save(social_values_path, social_values.to_numpy(dtype=np.float32))
    np.save(
        social_mask_path,
        social["edge_available"].astype(bool).to_numpy(dtype=bool),
    )
    social_index = social[
        [
            "pair_id",
            "native_event_id",
            "slot_index",
            "actor_node_key",
            "neighbor_rank",
            "neighbor_node_key",
        ]
    ].copy()
    social_index.insert(0, "tensor_row", np.arange(len(social_index), dtype=np.int64))
    social_index.to_csv(
        output_dir / "social_edges_index.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "social_edges_feature_names.json").write_text(
        json.dumps(social_columns, indent=2) + "\n",
        encoding="utf-8",
    )
    fixed_k = _fixed_top_k_contract(social)
    visual_audit = _write_roi_visual_pixel_artifacts(
        artifacts.roi_visual_selection,
        frames,
        output_dir,
        media=media,
        image_size=visual_size,
    )
    return {
        "roi_dynamics": {
            "values_path": str(roi_values_path),
            "index_path": str(output_dir / "roi_dynamics_index.csv"),
            "shape": list(roi_values.shape),
            "values_sha256": _sha256(roi_values_path),
        },
        "social_edges": {
            "values_path": str(social_values_path),
            "mask_path": str(social_mask_path),
            "index_path": str(output_dir / "social_edges_index.csv"),
            "shape": list(social_values.shape),
            "fixed_top_k": fixed_k,
            "values_sha256": _sha256(social_values_path),
            "mask_sha256": _sha256(social_mask_path),
        },
        "roi_visual_pixels": visual_audit,
    }


def _fixed_top_k_contract(edges: pd.DataFrame) -> dict[str, Any]:
    if edges.empty:
        return {"valid": False, "top_k": 0, "groups": 0}
    top_k = int(edges["neighbor_rank"].max())
    expected = list(range(1, top_k + 1))
    valid = True
    groups = 0
    for _, group in edges.groupby(
        ["pair_id", "slot_index", "actor_node_key"], sort=False
    ):
        groups += 1
        valid = valid and group["neighbor_rank"].astype(int).tolist() == expected
    return {"valid": bool(valid), "top_k": top_k, "groups": groups}


def _write_roi_visual_pixel_artifacts(
    visual: pd.DataFrame,
    frames: pd.DataFrame,
    output_dir: Path,
    *,
    media: FrameMediaResolver,
    image_size: int,
) -> dict[str, Any]:
    frame_lookup = {
        str(value): row for value, row in frames.set_index("frame_uid").iterrows()
    }
    visual = visual.reset_index(drop=True)
    scene_groups: dict[str, list[int]] = {}
    for row_index, row in visual.iterrows():
        source = frame_lookup.get(str(row.get("frame_uid", "")))
        scene_key = (
            str(source.get("scene_frame_uid", ""))
            if source is not None
            else f"missing::{row_index}"
        )
        scene_groups.setdefault(scene_key, []).append(int(row_index))
    patches: list[np.ndarray] = []
    mask = np.zeros(len(visual), dtype=bool)
    index_rows: list[dict[str, Any]] = []
    for row_indices in scene_groups.values():
        first_row = visual.iloc[row_indices[0]]
        source = frame_lookup.get(str(first_row.get("frame_uid", "")))
        scene_result = media.read_scene(source)
        for row_index in row_indices:
            row = visual.iloc[row_index]
            box = _visual_box(row, "union")
            geometry_expected = bool(row.get("actor_roi_visual_available", False))
            patch = (
                crop_rgb_box(
                    scene_result.image_rgb,
                    box,
                    image_size=image_size,
                )
                if geometry_expected and scene_result.available
                else None
            )
            tensor_row = -1
            if patch is not None:
                tensor_row = len(patches)
                patches.append(np.transpose(patch, (2, 0, 1)))
                mask[row_index] = True
            index_rows.append(
                {
                    "tensor_row": int(tensor_row),
                    "visual_context_id": str(row.get("visual_context_id", "")),
                    "pair_id": str(row.get("pair_id", "")),
                    "slot_index": int(row.get("slot_index", -1)),
                    "roi_class": str(row.get("roi_class", "")),
                    "frame_uid": str(row.get("frame_uid", "")),
                    "scene_frame_uid": (
                        "" if source is None else str(source.get("scene_frame_uid", ""))
                    ),
                    "pixel_geometry_expected": geometry_expected,
                    "pixel_available": bool(mask[row_index]),
                    **scene_result.provenance(),
                }
            )
    patches_path = output_dir / "roi_visual_union_patches_uint8.npy"
    mask_path = output_dir / "roi_visual_union_patch_mask_bool.npy"
    index_path = output_dir / "roi_visual_union_patch_index.csv"
    packed = (
        np.stack(patches).astype(np.uint8)
        if patches
        else np.zeros((0, 3, image_size, image_size), dtype=np.uint8)
    )
    np.save(patches_path, packed)
    np.save(mask_path, mask)
    pd.DataFrame.from_records(index_rows).to_csv(
        index_path,
        index=False,
        lineterminator="\n",
    )
    available = int(mask.sum())
    expected = int(visual.get("actor_roi_visual_available", False).sum())
    if expected == 0:
        status = "NOT_APPLICABLE_NO_VALID_ROI_GEOMETRY"
    elif available == expected:
        status = "PASS"
    elif available == 0:
        status = "BLOCKED_NO_SCENE_FRAME_PIXELS"
    else:
        status = "PASS_WITH_MISSING_SCENE_FRAME_PIXELS"
    return {
        "status": status,
        "rows": int(len(visual)),
        "expected_pixel_rows": expected,
        "available_rows": available,
        "missing_expected_rows": max(0, expected - available),
        "scene_groups": len(scene_groups),
        "patch_shape": list(packed.shape),
        "patches_path": str(patches_path),
        "mask_path": str(mask_path),
        "index_path": str(index_path),
        "patches_sha256": _sha256(patches_path),
        "mask_sha256": _sha256(mask_path),
    }


def _visual_box(row: pd.Series, prefix: str) -> tuple[float, float, float, float] | None:
    values = [
        pd.to_numeric(row.get(f"{prefix}_x1"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_y1"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_x2"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_y2"), errors="coerce"),
    ]
    if not np.isfinite(values).all():
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _implementation_lineage() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    module_path = Path(inspect.getfile(build_pig_strenet_artifacts)).resolve()
    media_module_path = Path(inspect.getfile(FrameMediaResolver)).resolve()
    return {
        "script": str(script_path),
        "script_sha256": _sha256(script_path),
        "module": str(module_path),
        "module_sha256": _sha256(module_path),
        "media_module": str(media_module_path),
        "media_module_sha256": _sha256(media_module_path),
        "git": _git_lineage(),
    }


def _git_lineage() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_status": run("status", "--short"),
        "dirty": run("status", "--porcelain") not in {"", "unavailable"},
    }


def _environment_lineage() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
    }


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in {"artifact_manifest.json", "run_manifest.json"}:
            continue
        files.append(
            {
                "name": path.name,
                "size": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.artifact_manifest",
        "immutable": True,
        "file_count": len(files),
        "files": files,
    }
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "file_count": len(files),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _lineage_scope(frames: pd.DataFrame) -> str:
    values = set(frames.get("lineage_scope", pd.Series(dtype=str)).fillna("").astype(str))
    if len(values) != 1:
        raise ValueError(f"input lineage scope is ambiguous={sorted(values)}")
    return next(iter(values))


def _review_claim(frames: pd.DataFrame) -> bool:
    values = frames.get("human_review_complete", pd.Series([False] * len(frames)))
    normalized = values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    if normalized.nunique() != 1:
        raise ValueError("human_review_complete claim is mixed")
    return bool(normalized.iloc[0])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
