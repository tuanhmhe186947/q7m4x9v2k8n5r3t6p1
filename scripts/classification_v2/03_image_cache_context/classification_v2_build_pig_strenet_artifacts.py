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
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    build_video_index,
)
from pig_behavior.classification_v2.datasets.pig_strenet_media import (
    FrameMediaResolver,
    crop_rgb_box,
)
from pig_behavior.classification_v2.datasets.pig_strenet_publication import (
    checkpointed_sha256,
    recover_media_manifest,
)
from pig_behavior.classification_v2.features.pig_strenet_artifacts import (
    PER_FRAME_AUDIT_ONLY_HISTORY_COLUMNS,
    availability_columns,
    build_pig_strenet_artifacts,
    compute_stabilized_difference_maps,
    model_x_columns,
)
from pig_behavior.classification_v2.features.pig_strenet_checkpoint import (
    PigSTRENetCheckpointStore,
)

SCHEMA_VERSION = "classification_v2.pig_strenet_artifact_run.v3"


class _ProgressReporter:
    """Atomically publish durable, non-semantic full-stage progress."""

    def __init__(
        self,
        path: Path,
        *,
        input_csv: Path,
        run_scope: str,
        resumed: bool = False,
    ) -> None:
        self._path = path
        self._started = monotonic()
        self._phase = "initializing"
        self._phase_started = self._started
        self._phase_initial_completed = 0
        self._payload: dict[str, Any] = {
            "schema_version": "classification_v2.stage_progress.v1",
            "status": "RUNNING",
            "run_scope": run_scope,
            "input_csv": str(input_csv),
            "process_id": os.getpid(),
            "resumed": resumed,
            "phase": "initializing",
            "completed": 0,
            "total": None,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "updated_at_utc": _utc_now(),
        }
        self._write()

    def __call__(
        self,
        phase: str,
        completed: int | None,
        total: int | None,
    ) -> None:
        now = monotonic()
        elapsed = now - self._started
        if phase != self._phase:
            self._phase = phase
            self._phase_started = now
            self._phase_initial_completed = int(completed or 0)
        phase_elapsed = now - self._phase_started
        phase_units = int(completed or 0) - self._phase_initial_completed
        remaining: float | None = None
        if total is not None and completed is not None and phase_units > 0:
            remaining = max(
                0.0,
                (phase_elapsed / phase_units) * (total - completed),
            )
        self._payload.update(
            phase=phase,
            completed=completed,
            total=total,
            elapsed_seconds=round(elapsed, 3),
            estimated_remaining_seconds=(
                None if remaining is None else round(remaining, 3)
            ),
            updated_at_utc=_utc_now(),
        )
        self._write()

    def fail(self, error: BaseException) -> None:
        self._payload.update(
            status="FAILED",
            error_type=type(error).__name__,
            error_message=str(error),
            updated_at_utc=_utc_now(),
        )
        self._write()

    def complete(self) -> None:
        self._payload.update(
            status="COMPUTED",
            phase="publication",
            updated_at_utc=_utc_now(),
        )
        self._write()

    def _write(self) -> None:
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
    parser.add_argument(
        "--run-scope",
        choices=("smoke", "full"),
        required=True,
        help="Declare whether this artifact run is a bounded smoke or full run.",
    )
    parser.add_argument(
        "--lineage-scope",
        default=None,
        help="Explicit lineage claim when the input artifact omits that column.",
    )
    parser.add_argument(
        "--human-review-complete",
        choices=("false", "true"),
        default=None,
        help="Explicit review claim when the input artifact omits that column.",
    )
    parser.add_argument("--skip-difference", action="store_true")
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Atomic heartbeat JSON; defaults inside --output-dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from an exact hash-matching checkpoint.",
    )
    parser.add_argument(
        "--recover-publication",
        action="store_true",
        help=(
            "Validate completed outputs and resume manifest publication only; "
            "never recompute artifacts."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/.checkpoints.",
    )
    parser.add_argument(
        "--social-checkpoint-pairs",
        type=int,
        default=250,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if args.output_dir.exists():
        if not args.overwrite and not args.resume and not args.recover_publication:
            raise FileExistsError(
                f"output exists; use a fresh root or --overwrite: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True)
    if args.difference_size <= 0:
        raise ValueError("--difference-size must be positive")
    if args.visual_size <= 0:
        raise ValueError("--visual-size must be positive")
    if args.social_checkpoint_pairs <= 0:
        raise ValueError("--social-checkpoint-pairs must be positive")
    if args.recover_publication and (args.resume or args.overwrite):
        raise ValueError(
            "--recover-publication is incompatible with --resume/--overwrite"
        )

    progress = _ProgressReporter(
        args.progress_json or args.output_dir / "pig_strenet_progress.json",
        input_csv=args.input_csv,
        run_scope=args.run_scope,
        resumed=args.resume,
    )
    previous_excepthook = sys.excepthook

    def report_unhandled(
        error_type: type[BaseException],
        error: BaseException,
        traceback: Any,
    ) -> None:
        progress.fail(error)
        previous_excepthook(error_type, error, traceback)

    sys.excepthook = report_unhandled
    if args.recover_publication:
        _recover_publication(args, progress)
        return
    progress("read_input", 0, 1)
    frames = pd.read_csv(args.input_csv, low_memory=False)
    progress("read_input", 1, 1)
    target_unit_keys = _select_target_unit_keys(frames, args.max_native_events)

    starts = tuple(
        int(value.strip())
        for value in args.legacy_target_starts.split(",")
        if value.strip()
    )
    if not starts:
        raise ValueError("--legacy-target-starts must not be empty")
    implementation = _implementation_lineage()
    checkpoint_store = PigSTRENetCheckpointStore(
        args.checkpoint_dir or args.output_dir / ".checkpoints",
        identity=_checkpoint_identity(
            args,
            starts=starts,
            implementation=implementation,
        ),
        resume=args.resume,
        social_chunk_pairs=args.social_checkpoint_pairs,
    )

    try:
        artifacts = build_pig_strenet_artifacts(
            frames,
            history_length=args.history_length,
            target_length=args.target_length,
            legacy_target_starts=starts,
            top_k_neighbors=args.top_k_neighbors,
            target_unit_keys=target_unit_keys,
            roi_coco_path=args.roi_coco,
            progress_callback=progress,
            checkpoint_store=checkpoint_store,
        )
    except Exception as error:
        progress.fail(error)
        raise
    progress("write_core_artifacts", 0, 8)
    _write_csv(artifacts.pair_manifest, args.output_dir / "pair_manifest.csv")
    progress("write_core_artifacts", 1, 8)
    _write_csv(artifacts.slot_manifest, args.output_dir / "slot_manifest.csv")
    progress("write_core_artifacts", 2, 8)
    _write_csv(artifacts.history_features, args.output_dir / "history_features.csv")
    progress("write_core_artifacts", 3, 8)
    _write_csv(artifacts.roi_dynamics, args.output_dir / "roi_dynamics.csv")
    progress("write_core_artifacts", 4, 8)
    _write_csv(
        artifacts.roi_visual_selection,
        args.output_dir / "roi_visual_selection.csv",
    )
    progress("write_core_artifacts", 5, 8)
    _write_csv(artifacts.social_nodes, args.output_dir / "social_nodes.csv")
    progress("write_core_artifacts", 6, 8)
    _write_csv(artifacts.social_edges, args.output_dir / "social_edges.csv")
    progress("write_core_artifacts", 7, 8)
    _write_csv(artifacts.control_matrix, args.output_dir / "history_control_matrix.csv")
    progress("write_core_artifacts", 8, 8)
    progress("write_packed_tensors", 0, 1)
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
    progress("write_packed_tensors", 1, 1)
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
        progress("write_difference_artifacts", 0, 1)
        difference_audit = _write_difference_artifacts(
            artifacts.slot_manifest,
            args.output_dir,
            frames=frames,
            media=media,
            image_size=args.difference_size,
        )
        progress("write_difference_artifacts", 1, 1)
    media_audit = media.write_manifest(
        args.output_dir / "media_manifest.json",
        checkpoint_path=(
            args.checkpoint_dir
            or args.output_dir / ".checkpoints"
        )
        / "media_publication.sqlite3",
        progress_callback=progress,
    )

    audit = dict(artifacts.audit)
    audit["difference"] = difference_audit
    audit["packed_tensors"] = tensor_audit
    audit["media"] = {
        "manifest_path": str(args.output_dir / "media_manifest.json"),
        "manifest_sha256": _sha256(args.output_dir / "media_manifest.json"),
        "source_file_count": media_audit["source_file_count"],
        "full_file_sha256_count": media_audit[
            "full_file_sha256_count"
        ],
        "derived_pixel_video_authority_count": media_audit[
            "derived_pixel_video_authority_count"
        ],
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
        "run_scope": args.run_scope,
    }
    audit_path = args.output_dir / "pig_strenet_artifact_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = _write_artifact_manifest(
        args.output_dir,
        progress_callback=progress,
        checkpoint_path=(
            args.checkpoint_dir
            or args.output_dir / ".checkpoints"
        )
        / "media_publication.sqlite3",
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": f"pig_strenet_review_artifacts_{args.run_scope}",
        "run_scope": args.run_scope,
        "lineage_scope": _lineage_scope(
            frames,
            override=args.lineage_scope,
        ),
        "human_review_complete": _review_claim(
            frames,
            override=args.human_review_complete,
        ),
        "input": {"path": str(args.input_csv), "sha256": _sha256(args.input_csv)},
        "output_dir": str(args.output_dir),
        "audit_path": str(audit_path),
        "resolved_config": {
            key: _jsonable(value) for key, value in vars(args).items()
            if key != "overwrite"
        },
        "implementation": implementation,
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
    progress.complete()
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    if not audit.get("valid", False):
        progress.fail(
            RuntimeError("Pig-STRENet artifact/media contract failed")
        )
        raise SystemExit("FAIL: Pig-STRENet artifact/media contract")
    checkpoint_store.cleanup()


def _recover_publication(
    args: argparse.Namespace,
    progress: _ProgressReporter,
) -> None:
    """Recover only publication from completed, checkpoint-bound outputs."""

    progress("validate_publication_recovery_authority", 0, 1)
    starts = tuple(
        int(value.strip())
        for value in args.legacy_target_starts.split(",")
        if value.strip()
    )
    implementation = _implementation_lineage()
    recovery_authority = _validate_publication_recovery_authority(
        args,
        starts=starts,
        implementation=implementation,
    )
    progress("validate_publication_recovery_authority", 1, 1)
    checkpoint_dir = (
        args.checkpoint_dir or args.output_dir / ".checkpoints"
    )
    media_audit = recover_media_manifest(
        args.output_dir / "media_manifest.json",
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
        video_index_aliases=len(build_video_index(args.video_root)),
        provenance_paths=[
            args.output_dir / "difference_pixel_index.csv",
            args.output_dir / "roi_visual_union_patch_index.csv",
        ],
        checkpoint_path=checkpoint_dir / "media_publication.sqlite3",
        progress_callback=progress,
    )
    progress("validate_completed_outputs", 0, 1)
    audit = _audit_completed_outputs(
        args.output_dir,
        input_csv=args.input_csv,
        checkpoint_dir=checkpoint_dir,
        media_audit=media_audit,
    )
    audit["input_csv"] = str(args.input_csv)
    audit["input_sha256"] = _sha256(args.input_csv)
    audit["input_frame_rows"] = _csv_row_count(args.input_csv)
    audit["target_unit_count"] = None
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
        "run_scope": args.run_scope,
    }
    audit["publication_recovery"] = recovery_authority
    audit_path = args.output_dir / "pig_strenet_artifact_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress("validate_completed_outputs", 1, 1)
    if not audit["valid"]:
        progress.fail(RuntimeError("completed Pig-STRENet outputs are invalid"))
        raise SystemExit("FAIL: completed Pig-STRENet outputs are invalid")

    artifact_manifest = _write_artifact_manifest(
        args.output_dir,
        progress_callback=progress,
        checkpoint_path=checkpoint_dir / "media_publication.sqlite3",
    )
    claims = _read_publication_claims(
        args.input_csv,
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_type": f"pig_strenet_review_artifacts_{args.run_scope}",
        "run_scope": args.run_scope,
        "lineage_scope": _lineage_scope(
            claims,
            override=args.lineage_scope,
        ),
        "human_review_complete": _review_claim(
            claims,
            override=args.human_review_complete,
        ),
        "input": {
            "path": str(args.input_csv),
            "sha256": _sha256(args.input_csv),
        },
        "output_dir": str(args.output_dir),
        "audit_path": str(audit_path),
        "resolved_config": {
            key: _jsonable(value)
            for key, value in vars(args).items()
            if key != "overwrite"
        },
        "implementation": implementation,
        "environment": _environment_lineage(),
        "artifact_manifest": artifact_manifest,
        "publication_recovery": recovery_authority,
        "training_started": False,
        "oof_started": False,
        "data_modified": False,
        "skills": [
            "experiment-lineage-reproducibility",
            "safe-refactor-test-guardian",
        ],
        "valid": True,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    progress.complete()
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def _validate_publication_recovery_authority(
    args: argparse.Namespace,
    *,
    starts: tuple[int, ...],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = (
        args.checkpoint_dir or args.output_dir / ".checkpoints"
    )
    identity_path = checkpoint_dir / "checkpoint_identity.json"
    if not identity_path.is_file():
        raise RuntimeError(
            f"PUBLICATION_RECOVERY_IDENTITY_MISSING:{identity_path}"
        )
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    recorded = payload.get("identity")
    if not isinstance(recorded, dict):
        raise RuntimeError("PUBLICATION_RECOVERY_IDENTITY_INVALID")
    current = _checkpoint_identity(
        args,
        starts=starts,
        implementation=implementation,
    )
    for key in ("input_csv", "input_sha256", "parameters", "environment"):
        if recorded.get(key) != current.get(key):
            raise RuntimeError(
                f"PUBLICATION_RECOVERY_AUTHORITY_MISMATCH:{key}"
            )
    recorded_code = recorded.get("implementation", {})
    current_code = current.get("implementation", {})
    if recorded_code.get("module_sha256") != current_code.get(
        "module_sha256"
    ):
        raise RuntimeError(
            "PUBLICATION_RECOVERY_SCIENTIFIC_MODULE_DRIFT"
        )
    return {
        "status": "PASS_PUBLICATION_ONLY_RECOVERY_AUTHORITY",
        "checkpoint_identity_hash": payload.get("identity_hash"),
        "recorded_script_sha256": recorded_code.get("script_sha256"),
        "current_script_sha256": current_code.get("script_sha256"),
        "recorded_media_module_sha256": recorded_code.get(
            "media_module_sha256"
        ),
        "current_media_module_sha256": current_code.get(
            "media_module_sha256"
        ),
        "recorded_publication_module_sha256": recorded_code.get(
            "publication_module_sha256"
        ),
        "current_publication_module_sha256": current_code.get(
            "publication_module_sha256"
        ),
        "scientific_module_sha256": current_code.get("module_sha256"),
        "computation_reexecuted": False,
    }


def _audit_completed_outputs(
    output_dir: Path,
    *,
    input_csv: Path,
    checkpoint_dir: Path,
    media_audit: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    partials = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and ".partial" in path.name
    )
    if partials:
        errors.append(f"PARTIAL_OUTPUTS_PRESENT:{partials}")
    records = _checkpoint_output_records(checkpoint_dir, errors)
    for table_name, filename in _checkpoint_csv_outputs().items():
        record = records.get(table_name)
        path = output_dir / filename
        if record is None:
            errors.append(f"CHECKPOINT_RECORD_MISSING:{table_name}")
            continue
        _validate_csv_output(path, record, errors)

    pair_path = output_dir / "pair_manifest.csv"
    history_path = output_dir / "history_features.csv"
    if errors or not pair_path.is_file() or not history_path.is_file():
        return _failed_recovery_audit(errors)
    pairs = pd.read_csv(pair_path, low_memory=False)
    history = pd.read_csv(history_path, low_memory=False)
    _validate_pair_authority(pairs, history, errors)

    pair_count = int(len(pairs))
    slot_count = int(records["slots"]["rows"])
    roi_count = int(records["roi_dynamics"]["rows"])
    roi_visual_count = int(records["roi_visual_selection"]["rows"])
    social_node_count = int(records["social_nodes"]["rows"])
    social_edge_count = int(records["social_edges"]["rows"])
    tensor_audit = _audit_completed_tensors(
        output_dir,
        pair_count=pair_count,
        slot_count=slot_count,
        roi_count=roi_count,
        roi_visual_count=roi_visual_count,
        social_edge_count=social_edge_count,
        media_audit=media_audit,
        errors=errors,
    )
    control_count = _csv_row_count(
        output_dir / "history_control_matrix.csv"
    )
    whitelist_path = output_dir / "feature_whitelist.json"
    if not whitelist_path.is_file():
        errors.append("FEATURE_WHITELIST_MISSING")
    else:
        whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))
        forbidden = set(whitelist.get("forbidden_inputs", []))
        if not {"labels", "review_metadata", "future_frames"}.issubset(
            forbidden
        ):
            errors.append("FEATURE_WHITELIST_FORBIDDEN_INPUTS_INCOMPLETE")
    event_mass = pairs.groupby("native_event_id")["event_weight"].sum()
    audit = {
        "schema_version": "classification_v2.pig_strenet_artifacts.v3",
        "primary_motion_time_basis": "source_frame_timestamp_seconds",
        "per_frame_motion_columns_audit_only": sorted(
            PER_FRAME_AUDIT_ONLY_HISTORY_COLUMNS
        ),
        "status": (
            "PASS_PIG_STRENET_ARTIFACT_SCHEMA"
            if not errors
            else "FAIL_PIG_STRENET_ARTIFACT_SCHEMA"
        ),
        "native_event_count": int(pairs["native_event_id"].nunique()),
        "pair_count": pair_count,
        "source_counts": _value_counts(pairs, "source_type"),
        "derived_view_counts": _value_counts(pairs, "derived_view"),
        "event_mass_min": float(event_mass.min()),
        "event_mass_max": float(event_mass.max()),
        "slot_count": slot_count,
        "roi_row_count": roi_count,
        "roi_visual_selection_count": roi_visual_count,
        "roi_visual_available_count": int(
            tensor_audit["roi_visual_pixels"]["expected_pixel_rows"]
        ),
        "social_node_count": social_node_count,
        "social_edge_count": social_edge_count,
        "control_count": control_count,
        "model_x_columns": model_x_columns(history),
        "target_selected_roi_used": False,
        "behavior_selected_partner_used": False,
        "future_frame_used": False,
        "difference": tensor_audit["difference"],
        "packed_tensors": {
            "roi_dynamics": tensor_audit["roi_dynamics"],
            "social_edges": tensor_audit["social_edges"],
            "roi_visual_pixels": tensor_audit["roi_visual_pixels"],
        },
        "media": {
            "manifest_path": str(output_dir / "media_manifest.json"),
            "manifest_sha256": _sha256(
                output_dir / "media_manifest.json"
            ),
            "source_file_count": media_audit["source_file_count"],
            "full_file_sha256_count": media_audit[
                "full_file_sha256_count"
            ],
            "derived_pixel_video_authority_count": media_audit[
                "derived_pixel_video_authority_count"
            ],
            "runtime_counts": media_audit["runtime_counts"],
            "status_counts": media_audit["status_counts"],
            "valid": media_audit["valid"],
        },
        "media_contract_valid": bool(
            media_audit["valid"]
            and tensor_audit["difference"]["missing_available_frame_slots"]
            == 0
            and tensor_audit["roi_visual_pixels"]["missing_expected_rows"]
            == 0
        ),
        "recovery_validation": {
            "checkpoint_bound": True,
            "input_csv": str(input_csv),
            "no_partial_outputs": not partials,
            "errors": errors,
        },
        "errors": errors,
        "warnings": [],
        "valid": not errors and bool(media_audit["valid"]),
    }
    return audit


def _checkpoint_output_records(
    checkpoint_dir: Path,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    phases = {
        "pairs_and_slots": {"pairs": "pairs", "slots": "slots"},
        "history_features": {"history_features": "history_features"},
        "roi_dynamics": {"roi_dynamics": "roi_dynamics"},
        "roi_visual_selection": {
            "roi_visual_selection": "roi_visual_selection"
        },
        "social_graph": {
            "nodes": "social_nodes",
            "edges": "social_edges",
        },
    }
    identity = json.loads(
        (checkpoint_dir / "checkpoint_identity.json").read_text(
            encoding="utf-8"
        )
    )
    identity_hash = identity.get("identity_hash")
    records: dict[str, dict[str, Any]] = {}
    for phase, table_map in phases.items():
        path = checkpoint_dir / f"{phase}.checkpoint.json"
        if not path.is_file():
            errors.append(f"CHECKPOINT_MANIFEST_MISSING:{path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity_hash") != identity_hash:
            errors.append(f"CHECKPOINT_IDENTITY_MISMATCH:{path}")
            continue
        for source_name, output_name in table_map.items():
            record = dict(payload.get("tables", {}).get(source_name, {}))
            checkpoint_file = checkpoint_dir / str(record.get("file", ""))
            if not checkpoint_file.is_file():
                errors.append(
                    f"CHECKPOINT_TABLE_MISSING:{checkpoint_file}"
                )
            elif _sha256(checkpoint_file) != record.get("sha256"):
                errors.append(
                    f"CHECKPOINT_TABLE_HASH_MISMATCH:{checkpoint_file}"
                )
            records[output_name] = record
    return records


def _checkpoint_csv_outputs() -> dict[str, str]:
    return {
        "pairs": "pair_manifest.csv",
        "slots": "slot_manifest.csv",
        "history_features": "history_features.csv",
        "roi_dynamics": "roi_dynamics.csv",
        "roi_visual_selection": "roi_visual_selection.csv",
        "social_nodes": "social_nodes.csv",
        "social_edges": "social_edges.csv",
    }


def _validate_csv_output(
    path: Path,
    record: dict[str, Any],
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"COMPLETED_OUTPUT_MISSING:{path}")
        return
    actual_columns = list(pd.read_csv(path, nrows=0).columns)
    expected_columns = list(record.get("columns", []))
    if actual_columns != expected_columns:
        errors.append(f"COMPLETED_OUTPUT_COLUMNS_MISMATCH:{path}")
    if _csv_row_count(path) != int(record.get("rows", -1)):
        errors.append(f"COMPLETED_OUTPUT_ROWS_MISMATCH:{path}")


def _validate_pair_authority(
    pairs: pd.DataFrame,
    history: pd.DataFrame,
    errors: list[str],
) -> None:
    if pairs["pair_id"].astype(str).duplicated().any():
        errors.append("PAIR_ID_DUPLICATE")
    if history["pair_id"].astype(str).duplicated().any():
        errors.append("HISTORY_PAIR_ID_DUPLICATE")
    if set(pairs["pair_id"].astype(str)) != set(
        history["pair_id"].astype(str)
    ):
        errors.append("HISTORY_PAIR_ID_SET_MISMATCH")
    if pairs["temporal_unit_key"].astype(str).eq("").any():
        errors.append("TEMPORAL_UNIT_KEY_MISSING")
    if not pairs["target_complete"].fillna(False).astype(bool).all():
        errors.append("TARGET_INCOMPLETE_PAIR_PRESENT")


def _audit_completed_tensors(
    output_dir: Path,
    *,
    pair_count: int,
    slot_count: int,
    roi_count: int,
    roi_visual_count: int,
    social_edge_count: int,
    media_audit: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    shapes = {
        "roi_values": _npy_shape(
            output_dir / "roi_dynamics_values_f32.npy",
            errors,
        ),
        "social_values": _npy_shape(
            output_dir / "social_edges_values_f32.npy",
            errors,
        ),
        "social_mask": _npy_shape(
            output_dir / "social_edges_mask_bool.npy",
            errors,
        ),
        "roi_patches": _npy_shape(
            output_dir / "roi_visual_union_patches_uint8.npy",
            errors,
        ),
        "roi_mask": _npy_shape(
            output_dir / "roi_visual_union_patch_mask_bool.npy",
            errors,
        ),
        "difference_maps": _npy_shape(
            output_dir / "stabilized_difference_maps_f32.npy",
            errors,
        ),
    }
    _require_first_dimension(shapes["roi_values"], roi_count, "ROI", errors)
    _require_first_dimension(
        shapes["social_values"],
        social_edge_count,
        "SOCIAL_VALUES",
        errors,
    )
    _require_first_dimension(
        shapes["social_mask"],
        social_edge_count,
        "SOCIAL_MASK",
        errors,
    )
    _require_first_dimension(
        shapes["roi_mask"],
        roi_visual_count,
        "ROI_VISUAL_MASK",
        errors,
    )
    _require_first_dimension(
        shapes["difference_maps"],
        pair_count,
        "DIFFERENCE_MAPS",
        errors,
    )
    expected_difference_rows = (
        pair_count * int(shapes["difference_maps"][1])
        if len(shapes["difference_maps"]) >= 2
        else -1
    )
    summary_rows = _csv_row_count(
        output_dir / "stabilized_difference_summary.csv"
    )
    if summary_rows != expected_difference_rows:
        errors.append("DIFFERENCE_SUMMARY_ROW_MISMATCH")
    difference_index_rows = _csv_row_count(
        output_dir / "difference_pixel_index.csv"
    )
    if difference_index_rows != slot_count:
        errors.append("DIFFERENCE_PROVENANCE_ROW_MISMATCH")
    roi_index_rows = _csv_row_count(
        output_dir / "roi_visual_union_patch_index.csv"
    )
    if roi_index_rows != roi_visual_count:
        errors.append("ROI_VISUAL_INDEX_ROW_MISMATCH")
    social_index_rows = _csv_row_count(
        output_dir / "social_edges_index.csv"
    )
    if social_index_rows != social_edge_count:
        errors.append("SOCIAL_INDEX_ROW_MISMATCH")
    provenance = {
        Path(item["path"]).name: item
        for item in media_audit.get("provenance_audit", [])
    }
    difference_provenance = provenance.get("difference_pixel_index.csv", {})
    roi_provenance = provenance.get(
        "roi_visual_union_patch_index.csv",
        {},
    )
    missing_difference = int(
        difference_provenance.get("missing_required_rows", -1)
    )
    missing_roi = int(roi_provenance.get("missing_required_rows", -1))
    if missing_difference != 0:
        errors.append("DIFFERENCE_REQUIRED_MEDIA_MISSING")
    if missing_roi != 0:
        errors.append("ROI_REQUIRED_MEDIA_MISSING")
    expected_roi = int(roi_provenance.get("required_rows", 0))
    available_roi = int(roi_provenance.get("available_rows", 0))
    _require_first_dimension(
        shapes["roi_patches"],
        available_roi,
        "ROI_VISUAL_PATCHES",
        errors,
    )
    return {
        "difference": {
            "status": "PASS" if missing_difference == 0 else "FAIL",
            "pairs_written": pair_count,
            "pairs_skipped_missing_crops": 0,
            "available_frame_slots": int(
                difference_provenance.get("required_rows", 0)
            ),
            "missing_available_frame_slots": missing_difference,
            "maps_shape": list(shapes["difference_maps"]),
            "maps_path": str(
                output_dir / "stabilized_difference_maps_f32.npy"
            ),
            "summary_path": str(
                output_dir / "stabilized_difference_summary.csv"
            ),
            "provenance_path": str(
                output_dir / "difference_pixel_index.csv"
            ),
        },
        "roi_dynamics": {
            "shape": list(shapes["roi_values"]),
            "index_rows": _csv_row_count(
                output_dir / "roi_dynamics_index.csv"
            ),
        },
        "social_edges": {
            "shape": list(shapes["social_values"]),
            "mask_shape": list(shapes["social_mask"]),
            "index_rows": social_index_rows,
        },
        "roi_visual_pixels": {
            "status": "PASS" if missing_roi == 0 else "FAIL",
            "rows": roi_visual_count,
            "expected_pixel_rows": expected_roi,
            "available_rows": available_roi,
            "missing_expected_rows": missing_roi,
            "patch_shape": list(shapes["roi_patches"]),
        },
    }


def _npy_shape(path: Path, errors: list[str]) -> tuple[int, ...]:
    if not path.is_file():
        errors.append(f"NPY_OUTPUT_MISSING:{path}")
        return ()
    try:
        return tuple(int(value) for value in np.load(path, mmap_mode="r").shape)
    except (OSError, ValueError) as error:
        errors.append(f"NPY_OUTPUT_INVALID:{path}:{error}")
        return ()


def _require_first_dimension(
    shape: tuple[int, ...],
    expected: int,
    label: str,
    errors: list[str],
) -> None:
    if not shape or int(shape[0]) != int(expected):
        errors.append(f"{label}_FIRST_DIMENSION_MISMATCH")


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return -1
    rows = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            rows += chunk.count(b"\n")
    return max(0, rows - 1)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts = frame[column].fillna("").astype(str).value_counts()
    return {
        str(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: item[0])
    }


def _failed_recovery_audit(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2.pig_strenet_artifacts.v3",
        "status": "FAIL_PIG_STRENET_ARTIFACT_SCHEMA",
        "media_contract_valid": False,
        "errors": errors,
        "warnings": [],
        "valid": False,
    }


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
    maps_path = output_dir / "stabilized_difference_maps_f32.npy"
    maps_temporary = maps_path.with_name(f"{maps_path.name}.partial.npy")
    summary_path = output_dir / "stabilized_difference_summary.csv"
    summary_temporary = summary_path.with_name(f"{summary_path.name}.partial")
    provenance_path = output_dir / "difference_pixel_index.csv"
    provenance_temporary = provenance_path.with_name(
        f"{provenance_path.name}.partial"
    )
    for path in (maps_temporary, summary_temporary, provenance_temporary):
        path.unlink(missing_ok=True)
    maps_memmap: np.memmap | None = None
    maps_shape: tuple[int, ...] | None = None
    maps_written = 0
    summary_header = True
    provenance_header = True
    pair_ids: list[str] = []
    skipped = 0
    available_frame_slots = 0
    missing_available_frame_slots = 0
    frame_lookup = {
        str(value): row for value, row in frames.set_index("frame_uid").iterrows()
    }
    pair_total = int(slots["pair_id"].nunique())
    for pair_id, group in slots.groupby("pair_id", sort=False):
        ordered = group.sort_values("global_slot_index")
        crops: list[np.ndarray] = []
        valid: list[bool] = []
        provenance_rows: list[dict[str, Any]] = []
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
        provenance_header = _append_csv_records(
            provenance_temporary,
            provenance_rows,
            header=provenance_header,
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
        if maps_memmap is None:
            maps_shape = tuple(int(value) for value in diff_maps.shape)
            maps_memmap = np.lib.format.open_memmap(
                maps_temporary,
                mode="w+",
                dtype=np.float32,
                shape=(pair_total, *maps_shape),
            )
        maps_memmap[maps_written] = diff_maps.astype(np.float32, copy=False)
        maps_written += 1
        summary_header = _append_csv_frame(
            summary_temporary,
            summary,
            header=summary_header,
        )
        pair_ids.append(str(pair_id))
    if maps_memmap is not None and maps_shape is not None:
        maps_memmap.flush()
        del maps_memmap
        _publish_compact_memmap(
            maps_temporary,
            maps_path,
            rows=maps_written,
            row_shape=maps_shape,
            dtype=np.float32,
        )
        summary_temporary.replace(summary_path)
    else:
        _atomic_save_array(
            maps_path,
            np.zeros((0, 0, image_size, image_size), dtype=np.float32),
        )
        _atomic_write_csv(
            pd.DataFrame(
            columns=["pair_id", "pair_slot_index", "pair_valid"]
            ),
            summary_path,
        )
    provenance_temporary.replace(provenance_path)
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
        "pairs_written": maps_written,
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
        "motion_inside_n_per_second",
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
        "relative_speed_n_per_second",
        "relative_angle",
        "approach_speed_n_per_second",
        "separation_speed_n_per_second",
        "pair_iou",
        "pair_overlap_ratio",
        "pair_contact",
        "pair_contact_duration_frames",
        "pair_contact_duration_sec",
        "pair_motion_energy_n_per_second2",
        "pair_contact_motion_intensity_n_per_second2",
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
    patches_path = output_dir / "roi_visual_union_patches_uint8.npy"
    patches_temporary = patches_path.with_name(
        f"{patches_path.name}.partial.npy"
    )
    index_path = output_dir / "roi_visual_union_patch_index.csv"
    index_temporary = index_path.with_name(f"{index_path.name}.partial")
    patches_temporary.unlink(missing_ok=True)
    index_temporary.unlink(missing_ok=True)
    packed: np.memmap | None = None
    packed_rows = 0
    mask = np.zeros(len(visual), dtype=bool)
    index_header = True
    for row_indices in scene_groups.values():
        first_row = visual.iloc[row_indices[0]]
        source = frame_lookup.get(str(first_row.get("frame_uid", "")))
        scene_result = media.read_scene(source)
        index_rows: list[dict[str, Any]] = []
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
                if packed is None:
                    packed = np.lib.format.open_memmap(
                        patches_temporary,
                        mode="w+",
                        dtype=np.uint8,
                        shape=(len(visual), 3, image_size, image_size),
                    )
                tensor_row = packed_rows
                packed[packed_rows] = np.transpose(patch, (2, 0, 1))
                packed_rows += 1
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
        index_header = _append_csv_records(
            index_temporary,
            index_rows,
            header=index_header,
        )
    mask_path = output_dir / "roi_visual_union_patch_mask_bool.npy"
    if packed is not None:
        packed.flush()
        del packed
        _publish_compact_memmap(
            patches_temporary,
            patches_path,
            rows=packed_rows,
            row_shape=(3, image_size, image_size),
            dtype=np.uint8,
        )
    else:
        _atomic_save_array(
            patches_path,
            np.zeros((0, 3, image_size, image_size), dtype=np.uint8),
        )
    _atomic_save_array(mask_path, mask)
    index_temporary.replace(index_path)
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
        "patch_shape": list(np.load(patches_path, mmap_mode="r").shape),
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
    publication_module_path = Path(
        inspect.getfile(recover_media_manifest)
    ).resolve()
    return {
        "script": str(script_path),
        "script_sha256": _sha256(script_path),
        "module": str(module_path),
        "module_sha256": _sha256(module_path),
        "media_module": str(media_module_path),
        "media_module_sha256": _sha256(media_module_path),
        "publication_module": str(publication_module_path),
        "publication_module_sha256": _sha256(publication_module_path),
        "git": _git_lineage(),
    }


def _checkpoint_identity(
    args: argparse.Namespace,
    *,
    starts: tuple[int, ...],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    roi_sha256 = (
        _sha256(args.roi_coco)
        if args.roi_coco is not None and args.roi_coco.is_file()
        else None
    )
    return {
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": _sha256(args.input_csv),
        "implementation": {
            "script_sha256": implementation["script_sha256"],
            "module_sha256": implementation["module_sha256"],
            "media_module_sha256": implementation["media_module_sha256"],
            "publication_module_sha256": implementation[
                "publication_module_sha256"
            ],
        },
        "parameters": {
            "history_length": args.history_length,
            "target_length": args.target_length,
            "legacy_target_starts": list(starts),
            "top_k_neighbors": args.top_k_neighbors,
            "roi_coco": (
                None if args.roi_coco is None else str(args.roi_coco.resolve())
            ),
            "roi_coco_sha256": roi_sha256,
            "video_root": str(args.video_root.resolve()),
            "legacy_crop_root": str(args.legacy_crop_root.resolve()),
            "max_native_events": args.max_native_events,
            "difference_size": args.difference_size,
            "visual_size": args.visual_size,
            "run_scope": args.run_scope,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
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


def _write_artifact_manifest(
    output_dir: Path,
    *,
    progress_callback: _ProgressReporter | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    paths = [
        path
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
        and path.name
        not in {
            "artifact_manifest.json",
            "pig_strenet_progress.json",
            "run_manifest.json",
        }
    ]
    for index, path in enumerate(paths, start=1):
        if checkpoint_path is None:
            digest = _sha256(path)
        else:
            digest = checkpointed_sha256(
                path,
                checkpoint_path=checkpoint_path,
            )
        if progress_callback is not None:
            progress_callback("hash_output_artifacts", index, len(paths))
        if not path.is_file():
            continue
        files.append(
            {
                "name": path.name,
                "size": int(path.stat().st_size),
                "sha256": digest,
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


def _read_publication_claims(path: Path) -> pd.DataFrame:
    columns = set(pd.read_csv(path, nrows=0).columns)
    usecols = sorted(
        columns.intersection({"lineage_scope", "human_review_complete"})
    )
    if not usecols:
        return pd.DataFrame(index=[0])
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _lineage_scope(
    frames: pd.DataFrame,
    *,
    override: str | None = None,
) -> str:
    values = set(
        frames.get("lineage_scope", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
    )
    values.discard("")
    if override is not None:
        if values and values != {override}:
            raise ValueError(
                "input lineage scope conflicts with explicit claim="
                f"{sorted(values)}!={override}"
            )
        return override
    if len(values) != 1:
        raise ValueError(f"input lineage scope is ambiguous={sorted(values)}")
    return next(iter(values))


def _review_claim(
    frames: pd.DataFrame,
    *,
    override: str | None = None,
) -> bool:
    column_present = "human_review_complete" in frames.columns
    values = frames.get(
        "human_review_complete",
        pd.Series([False] * len(frames)),
    )
    normalized = (
        values.fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if override is not None:
        expected = override == "true"
        if column_present and normalized.nunique() != 1:
            raise ValueError("human_review_complete claim is mixed")
        if column_present and bool(normalized.iloc[0]) != expected:
            raise ValueError(
                "input human_review_complete conflicts with explicit claim"
            )
        return expected
    if normalized.nunique() != 1:
        raise ValueError("human_review_complete claim is mixed")
    return bool(normalized.iloc[0])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_csv(frame, path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _append_csv_records(
    path: Path,
    records: list[dict[str, Any]],
    *,
    header: bool,
) -> bool:
    if not records:
        return header
    return _append_csv_frame(
        path,
        pd.DataFrame.from_records(records),
        header=header,
    )


def _append_csv_frame(
    path: Path,
    frame: pd.DataFrame,
    *,
    header: bool,
) -> bool:
    frame.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        lineterminator="\n",
    )
    return False


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.partial.npy")
    np.save(temporary, values)
    temporary.replace(path)


def _publish_compact_memmap(
    source_path: Path,
    output_path: Path,
    *,
    rows: int,
    row_shape: tuple[int, ...],
    dtype: Any,
) -> None:
    source = np.load(source_path, mmap_mode="r")
    if rows == int(source.shape[0]):
        del source
        source_path.replace(output_path)
        return
    compact_path = output_path.with_name(f"{output_path.name}.compact.partial.npy")
    compact_path.unlink(missing_ok=True)
    compact = np.lib.format.open_memmap(
        compact_path,
        mode="w+",
        dtype=dtype,
        shape=(rows, *row_shape),
    )
    for start in range(0, rows, 64):
        stop = min(rows, start + 64)
        compact[start:stop] = source[start:stop]
    compact.flush()
    del compact
    del source
    compact_path.replace(output_path)
    source_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
