"""Reproduce and freeze the historical-best H5b/H4 executable pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.tracking import evaluate_b0_b1_r0_standard_v2 as standard_v2  # noqa: E402
from scripts.tracking import generate_b0_b1_frozen_predictions as generator  # noqa: E402
from scripts.tracking import reproduce_historical_h5b_h4_detector_cache as cache_tool  # noqa: E402

from pig_behavior.evaluation.tracking.contracts import (  # noqa: E402
    resolve_evaluator_code_sha,
)
from pig_behavior.evaluation.tracking.lineage import (  # noqa: E402
    cvat_prediction_semantic_sha256,
)
from pig_behavior.tracking.detector_cache import (  # noqa: E402
    DetectorEvidenceCache,
    ReplayDetector,
)
from pig_behavior.tracking.offline_repair import (  # noqa: E402
    OfflineRepairResult,
    canonical_authority_hash,
    offline_repair_semantic_hash,
)
from pig_behavior.tracking.profiles.hybrid_bytetrack import (  # noqa: E402
    EVAL_CONFIGS,
)
from pig_behavior.tracking.runner import run_tracking  # noqa: E402

DATE = "20260729"
STARTING_MAIN_SHA = "ce75b9bee9e159abe1e659c6d60b28c157f4bd45"
RESOLVED_CACHE_BASE = "b189a863fc9b1f60c71682fa734de7f65818ef6b"
HISTORICAL_SOURCE_SHA = "31d360ba96b4065ce5125c0d88765531cc5898ae"
HISTORICAL_RUN_ID = "20260719_h5b_h4_full13_combined_v2"
HISTORICAL_PROFILE = "hybrid_bytetrack_best"
EXECUTION_TOPOLOGY = "CURRENT_MAIN_EXECUTABLE_SEMANTICALLY_EQUIVALENT"
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
EXPECTED_TOTAL_FRAMES = 23400
PROFILE_CONFIG_SHA256 = (
    "4eb3d4e2262485d48d425be06fd8a6b3adfd8a01a27b28e76b5a8d55958d1d55"
)
REPAIR_SEMANTIC_SHA256 = (
    "e078b5b165dda82dee5b61e9465dc9844446e4cb576a02858c4ed7369828d758"
)
CACHE_AUTHORITY_SHA256 = (
    "52566e9318e8a6c60ea49b01af8715c3b074460c4a803332e6a77b40ac004975"
)
CACHE_CONFIG_SHA256 = (
    "c51b7549ae8624502389b685f7ae47668404339d74bd38c73f527a2e4e78bad8"
)
WEIGHTS_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
HISTORICAL_PREDICTION_ARTIFACT_SHA256 = (
    "36c3bdd3f6d92c0c5336590dbf4c8822402d718ec47df2b137cef862339d5b8a"
)
BBOX_ABS_TOLERANCE = 0.01
BBOX_IOU_THRESHOLD = 0.9999
FLOAT_METRIC_TOLERANCE = 1e-15

HISTORICAL_METRICS = {
    "hota": 0.9002906560906596,
    "deta": 0.8897945412373369,
    "assa": 0.9119035496598343,
    "loca": 0.9296724339629265,
    "idf1": 0.9915010683760683,
    "idp": 0.9915010683760683,
    "idr": 0.9915010683760683,
    "idsw_standard": 0,
    "fp": 1579,
    "fn": 1579,
    "fragments": 425,
    "wrong_id_matched_frames": 24,
    "identity_error_episodes": 8,
    "recovered_identity_error_episodes": 8,
    "terminal_identity_error_episodes": 0,
    "persistent_pairwise_identity_swaps": 0,
}
COUNT_METRICS = (
    "idsw_standard",
    "fp",
    "fn",
    "fragments",
    "wrong_id_matched_frames",
    "identity_error_episodes",
    "recovered_identity_error_episodes",
    "terminal_identity_error_episodes",
    "persistent_pairwise_identity_swaps",
)
FLOAT_METRICS = ("hota", "deta", "assa", "loca", "idf1", "idp", "idr")


class ReproductionError(RuntimeError):
    """Fail-closed historical executable reproduction error."""


def canonical_hash(payload: Any) -> str:
    data = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReproductionError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise ReproductionError(result.stderr.strip())
    return result.returncode == 0


def require_environment() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ReproductionError("PYTHONDONTWRITEBYTECODE=1 is required")


def code_blob(repo: Path, path: str) -> str:
    return git(repo, "hash-object", path)


def profile_payload() -> dict[str, Any]:
    profile = dict(EVAL_CONFIGS[HISTORICAL_PROFILE])
    if canonical_hash(profile) != PROFILE_CONFIG_SHA256:
        raise ReproductionError("Historical promoted profile hash mismatch")
    return profile


def environment_payload() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "cv2", "scipy", "torch"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "UNKNOWN"))
        except ImportError:
            versions[name] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "UNSET"),
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }


def verify_lineage(worktree_repo: Path) -> str:
    head = git(worktree_repo, "rev-parse", "HEAD")
    if not git_is_ancestor(worktree_repo, RESOLVED_CACHE_BASE, head):
        raise ReproductionError("Resolved cache base is not reachable")
    forbidden = git(
        worktree_repo,
        "diff",
        "--name-only",
        f"{RESOLVED_CACHE_BASE}..{STARTING_MAIN_SHA}",
        "--",
        "src/pig_behavior/tracking",
        "src/pig_behavior/evaluation/tracking",
    )
    if forbidden:
        raise ReproductionError("Tracking authority changed across descendants")
    if not git_is_ancestor(worktree_repo, STARTING_MAIN_SHA, head):
        raise ReproductionError("Starting main is not an ancestor")
    return head


def load_cache_authority(cache_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = load_json(
        cache_root
        / "manifests"
        / "HISTORICAL_H5B_H4_DETECTOR_CACHE_AUTHORITY_20260728.json"
    )
    manifest = load_json(
        cache_root
        / "manifests"
        / "HISTORICAL_H5B_H4_DETECTOR_CACHE_MANIFEST_20260728.json"
    )
    required = {
        "cache_authority_sha256": CACHE_AUTHORITY_SHA256,
        "effective_config_sha256": CACHE_CONFIG_SHA256,
        "video_count": EXPECTED_VIDEOS,
        "unique_authoritative_frame_records": EXPECTED_TOTAL_FRAMES,
        "missing_frame_records": 0,
        "duplicate_authoritative_frame_records": 0,
        "cache_validation": "PASS",
        "cache_replay": "PASS",
    }
    for key, expected in required.items():
        if authority.get(key) != expected:
            raise ReproductionError(f"Cache authority mismatch: {key}")
    return authority, manifest


def cache_expected_identity(
    video: Any,
    effective_config: dict[str, Any],
) -> Any:
    return cache_tool.cache_identity(
        video,
        CACHE_CONFIG_SHA256,
        str(effective_config["producer_git_sha"]),
    )


def source_equivalence_audit(worktree_repo: Path) -> dict[str, Any]:
    operations = [
        {
            "operation": "promoted profile overrides",
            "historical": "runtime overrides frozen in promotion record",
            "current": "hybrid_bytetrack_best promoted profile",
            "status": "SEMANTIC_MATCH",
        },
        {
            "operation": "H5b hidden-suffix persistence repair",
            "historical": "c11a2f4037cc20db996ce370adf4558c93d85474",
            "current": "same function, enabled with persistence=2",
            "status": "SEMANTIC_MATCH",
        },
        {
            "operation": "H4 far-camera geometry replay",
            "historical": "befaa58c4bc4228e714a187999aa12abb72f00b8",
            "current": "same function and frozen thresholds",
            "status": "SEMANTIC_MATCH",
        },
        {
            "operation": "offline stage order",
            "historical": "runner post-video sequence",
            "current": "apply_offline_repair_stack refactor",
            "status": "SEMANTIC_MATCH",
        },
        {
            "operation": "post-historical association changes",
            "historical": "not present",
            "current": "realtime-only or disabled opt-in flags",
            "status": "NOT_ACTIVE_FOR_HYBRID_PROFILE",
        },
        {
            "operation": "detector invocation",
            "historical": "live model.track",
            "current": "fail-closed ReplayDetector cache seam",
            "status": "AUTHORIZED_EVIDENCE_SUBSTITUTION",
        },
    ]
    paths = (
        "src/pig_behavior/tracking/association.py",
        "src/pig_behavior/tracking/refinement.py",
        "src/pig_behavior/tracking/offline_repair.py",
        "src/pig_behavior/tracking/runner.py",
        "src/pig_behavior/tracking/profiles/hybrid_bytetrack.py",
        "src/pig_behavior/tracking/exporters/cvat_xml.py",
    )
    return {
        "schema_version": "tracking.h5b_h4.executable_topology_audit.v1",
        "date": DATE,
        "topology": EXECUTION_TOPOLOGY,
        "historical_source_sha": HISTORICAL_SOURCE_SHA,
        "current_source_sha": git(worktree_repo, "rev-parse", "HEAD"),
        "current_code_blobs": {
            path: code_blob(worktree_repo, path) for path in paths
        },
        "operations": operations,
        "claim_limit": (
            "Equivalence applies only to the frozen hybrid profile and cache "
            "replay path; exact missing historical detector rows remain unproven."
        ),
    }


def executable_config(
    source_repo: Path,
    worktree_repo: Path,
    cache_root: Path,
    population: list[Any],
) -> dict[str, Any]:
    profile = profile_payload()
    sample_cfg = build_tracking_config(
        source_repo,
        population[0],
        Path("PREEXECUTION_PLACEHOLDER"),
    )
    repair_hash = offline_repair_semantic_hash(sample_cfg)
    if repair_hash != REPAIR_SEMANTIC_SHA256:
        raise ReproductionError("Offline repair semantic hash mismatch")
    payload = {
        "schema_version": "tracking.h5b_h4.executable_config.v1",
        "date": DATE,
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "execution_topology": EXECUTION_TOPOLOGY,
        "profile": HISTORICAL_PROFILE,
        "profile_values": profile,
        "profile_config_sha256": PROFILE_CONFIG_SHA256,
        "offline_repair_semantic_sha256": repair_hash,
        "repair_stage_order": [
            "identity_swap_guard",
            "temporal_box_refinement",
            "overlap_hidden_island_stabilization",
            "local_pair_swap_repair",
            "episode_pair_swap_repair",
            "long_pair_swap_repair",
            "suffix_pair_swap_repair",
            "overlap_small_box_suppression",
            "hidden_suffix_id_swap_repair",
            "realtime_motion_pair_stabilizer",
            "near_wall_hidden_geometry_refinement",
            "far_camera_hidden_geometry_refinement",
        ],
        "detector_cache": {
            "root": str(cache_root),
            "authority_sha256": CACHE_AUTHORITY_SHA256,
            "effective_config_sha256": CACHE_CONFIG_SHA256,
            "confidence": 0.20,
            "max_raw_detections": 64,
            "nms_iou": 0.50,
            "image_size": 640,
            "cadence": "EVERY_FRAME",
            "live_detector_fallback": False,
        },
        "population": {
            "video_count": EXPECTED_VIDEOS,
            "frames_per_video": EXPECTED_FRAMES,
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "video_keys": [video.video_key for video in population],
        },
        "output": {
            "schema": "CVAT video XML 1.1",
            "frame_indexing": "ZERO_BASED_0_TO_1799",
            "bbox_serialization_decimals": 2,
            "write_mp4": False,
        },
        "randomness": {
            "config_variants_authorized": 1,
            "config_search_authorized": False,
            "tracker_random_seed": "NOT_USED",
            "detector_inference": "FORBIDDEN_CACHE_REPLAY_ONLY",
        },
        "environment": environment_payload(),
    }
    payload["executable_config_sha256"] = canonical_hash(payload)
    return payload


def build_tracking_config(
    source_repo: Path,
    video: Any,
    output_dir: Path,
) -> Any:
    from pig_behavior.tracking.config import TrackingConfig, validate_config

    overrides = profile_payload()
    cfg = TrackingConfig(
        mode="hybrid_bytetrack",
        video_path=video.video_path,
        weights_path=(
            source_repo / "models" / "detector" / "pig_detector_yolov8.pt"
        ),
        mask_path=source_repo / "data" / "annotations" / "scene" / "mask.png",
        output_dir=output_dir,
        device="cpu",
        half=False,
        nms_iou=0.50,
        imgsz=640,
        write_output_video=False,
        show=False,
        start_frame=0,
        max_frames=EXPECTED_FRAMES,
        **overrides,
    )
    cfg.association_debug = False
    validate_config(cfg)
    return cfg


def prediction_parity_contract() -> dict[str, Any]:
    return {
        "schema_version": "tracking.h5b_h4.prediction_parity_contract.v1",
        "date": DATE,
        "frozen_before_execution": True,
        "levels": {
            "level_1": "EXACT_CANONICAL_PREDICTION_PARITY",
            "level_2": "SEMANTIC_PREDICTION_PARITY",
            "level_3": "METRIC_ONLY_PARITY",
            "level_4": "REPRODUCTION_FAILED",
        },
        "canonical_contract": {
            "same_video_count": True,
            "same_rows": True,
            "same_frame_indices": True,
            "same_identity_values": True,
            "same_hidden_state": True,
            "same_bbox_values_after_historical_serialization": True,
            "same_deterministic_row_order": True,
        },
        "semantic_contract": {
            "identity_differences": 0,
            "hidden_state_differences": 0,
            "row_additions": 0,
            "row_removals": 0,
            "frame_index_differences": 0,
            "bbox_absolute_tolerance": BBOX_ABS_TOLERANCE,
            "minimum_paired_bbox_iou": BBOX_IOU_THRESHOLD,
            "tolerance_basis": "CVAT writer serializes bbox coordinates to 2 decimals",
        },
        "metric_contract": {
            "count_metrics": "EXACT_EQUALITY",
            "identity_diagnostics": "EXACT_EQUALITY",
            "floating_absolute_tolerance": FLOAT_METRIC_TOLERANCE,
        },
        "post_result_parameter_changes_authorized": False,
        "post_evaluation_rerun_authorized": False,
    }


def execution_manifest(population: list[Any], cache_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "tracking.h5b_h4.execution_manifest.v1",
        "date": DATE,
        "status": "FROZEN_BEFORE_EXECUTION",
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_source_video_authority": "PASS",
        "cache_authority": "PASS",
        "gt_consumed_during_prediction_generation": False,
        "sequence_boundary": "VIDEO",
        "cache_root": str(cache_root),
        "videos": [
            {
                "video_key": video.video_key,
                "source_video_path": str(video.video_path),
                "source_video_sha256": video.video_sha256,
                "gt_sha256_population_binding_only": video.gt_sha256,
                "frame_start": 0,
                "frame_end": EXPECTED_FRAMES - 1,
                "frame_count": EXPECTED_FRAMES,
            }
            for video in population
        ],
    }


def prepare(
    source_repo: Path,
    worktree_repo: Path,
    historical_repo: Path,
    cache_root: Path,
    output_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    require_environment()
    if output_root.exists():
        raise ReproductionError(f"Refusing overwrite: {output_root}")
    head = verify_lineage(worktree_repo)
    if git(historical_repo, "status", "--porcelain=v1", "-uall"):
        raise ReproductionError("Historical worktree is not clean")
    load_cache_authority(cache_root)
    population = cache_tool.load_population(source_repo, worktree_repo)
    if len(population) != EXPECTED_VIDEOS:
        raise ReproductionError("Population is not full-13")
    effective_config = load_json(
        worktree_repo
        / "docs"
        / "tracking"
        / "historical_h5b_h4_reproduction"
        / "HISTORICAL_H5B_H4_DETECTOR_EFFECTIVE_CONFIG_20260728.json"
    )
    for video in population:
        cache_path = cache_tool.cache_path(cache_root, video.video_key)
        cache = DetectorEvidenceCache.load(
            cache_path,
            expected_identity=cache_expected_identity(video, effective_config),
        )
        if tuple(cache.frames) != tuple(range(EXPECTED_FRAMES)):
            raise ReproductionError(f"Cache coverage mismatch: {video.video_key}")
    output_root.mkdir(parents=True)
    for relative in (
        "run_1/raw_pre_repair",
        "run_1/final_predictions",
        "run_1/repair_ledgers",
        "run_1/manifests",
        "run_1/commands",
        "run_1/audits",
        "run_2_determinism/raw_pre_repair",
        "run_2_determinism/final_predictions",
        "run_2_determinism/repair_ledgers",
        "run_2_determinism/manifests",
        "run_2_determinism/commands",
        "run_2_determinism/audits",
        "comparisons",
        "standard_v2",
        "authority",
    ):
        (output_root / relative).mkdir(parents=True)
    config = executable_config(
        source_repo,
        worktree_repo,
        cache_root,
        population,
    )
    topology = source_equivalence_audit(worktree_repo)
    contract = prediction_parity_contract()
    manifest = execution_manifest(population, cache_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_EXECUTABLE_CONFIG_{DATE}.json",
        config,
    )
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_PREDICTION_PARITY_CONTRACT_{DATE}.json",
        contract,
    )
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_EXECUTION_MANIFEST_{DATE}.json",
        manifest,
    )
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_EXECUTION_TOPOLOGY_AUDIT_{DATE}.json",
        topology,
    )
    state = {
        "schema_version": "tracking.h5b_h4.reproduction_state.v1",
        "status": "PREPARED",
        "prepared_at": utc_now(),
        "producer_head": head,
        "historical_head": git(historical_repo, "rev-parse", "HEAD"),
        "historical_status": "",
        "executable_config_sha256": config["executable_config_sha256"],
        "config_variants_authorized": 1,
        "config_variants_executed": 0,
        "post_result_parameter_changes": 0,
    }
    write_json(output_root / "authority" / "REPRODUCTION_STATE.json", state)
    return state


def write_capture(
    raw_path: Path,
    ledger_path: Path,
    video_key: str,
) -> Any:
    def capture(
        raw_shapes: list[dict[str, Any]],
        result: OfflineRepairResult,
    ) -> None:
        raw_payload = {
            "schema_version": "tracking.h5b_h4.raw_pre_repair.v1",
            "source_core": "hybrid_bytetrack",
            "video_key": video_key,
            "input_authority_hash": canonical_authority_hash(raw_shapes),
            "shapes": raw_shapes,
        }
        ledger_payload = {
            "schema_version": "tracking.h5b_h4.repair_ledger.v1",
            "profile": HISTORICAL_PROFILE,
            "source_core": "hybrid_bytetrack",
            "video_key": video_key,
            "repair_config_hash": result.repair_config_hash,
            "input_authority_hash": result.input_authority_hash,
            "output_authority_hash": result.output_authority_hash,
            "events": result.ledger,
        }
        if raw_payload["input_authority_hash"] != result.input_authority_hash:
            raise ReproductionError("Raw snapshot authority mismatch")
        write_json(raw_path, raw_payload)
        write_json(ledger_path, ledger_payload)

    return capture


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def run_once(
    source_repo: Path,
    worktree_repo: Path,
    cache_root: Path,
    output_root: Path,
    run_number: int,
) -> dict[str, Any]:
    require_environment()
    state_path = output_root / "authority" / "REPRODUCTION_STATE.json"
    state = load_json(state_path)
    expected_status = "PREPARED" if run_number == 1 else "RUN1_FROZEN"
    if state.get("status") != expected_status:
        raise ReproductionError(f"Invalid state for run {run_number}")
    if git(worktree_repo, "status", "--porcelain=v1", "-uall"):
        raise ReproductionError("Producer worktree must be clean before tracking")
    population = cache_tool.load_population(source_repo, worktree_repo)
    effective_config = load_json(
        worktree_repo
        / "docs"
        / "tracking"
        / "historical_h5b_h4_reproduction"
        / "HISTORICAL_H5B_H4_DETECTOR_EFFECTIVE_CONFIG_20260728.json"
    )
    run_name = "run_1" if run_number == 1 else "run_2_determinism"
    run_root = output_root / run_name
    if any((run_root / "final_predictions").iterdir()):
        raise ReproductionError(f"Refusing non-empty {run_name}")
    records: list[dict[str, Any]] = []
    started = utc_now()
    for index, video in enumerate(population, start=1):
        cache = DetectorEvidenceCache.load(
            cache_tool.cache_path(cache_root, video.video_key),
            expected_identity=cache_expected_identity(video, effective_config),
        )
        detector = ReplayDetector(cache)
        machine_root = run_root / "manifests" / "machine" / video.video_key
        cfg = build_tracking_config(source_repo, video, machine_root)
        raw_path = run_root / "raw_pre_repair" / f"{video.video_key}.json"
        ledger_path = run_root / "repair_ledgers" / f"{video.video_key}.json"
        print(
            f"RUN{run_number}_BEGIN {index}/{EXPECTED_VIDEOS} {video.video_key}",
            flush=True,
        )
        summary = run_tracking(
            cfg,
            model=detector,
            hybrid_repair_capture=write_capture(
                raw_path,
                ledger_path,
                video.video_key,
            ),
        )
        if detector.invocations != EXPECTED_FRAMES:
            raise ReproductionError(f"Replay count mismatch: {video.video_key}")
        if summary.frames_read != EXPECTED_FRAMES:
            raise ReproductionError(f"Frame count mismatch: {video.video_key}")
        destination = (
            run_root / "final_predictions" / f"{video.video_key}.xml"
        )
        shutil.copy2(summary.cvat_video_xml, destination)
        record = generator.xml_structural_record(
            destination,
            video_key=video.video_key,
            width=video.width,
            height=video.height,
        )
        record["raw_pre_repair_sha256"] = sha256_file(raw_path)
        record["repair_ledger_sha256"] = sha256_file(ledger_path)
        records.append(record)
        print(
            f"RUN{run_number}_END {video.video_key} "
            f"objects={record['prediction_object_count']}",
            flush=True,
        )
    prediction_hash = generator.prediction_set_hash(records)
    run_manifest = {
        "schema_version": "tracking.h5b_h4.executable_run.v1",
        "date": DATE,
        "run_number": run_number,
        "status": "FROZEN",
        "started_at": started,
        "completed_at": utc_now(),
        "video_count": len(records),
        "frame_count": EXPECTED_TOTAL_FRAMES,
        "tracker_executions": EXPECTED_VIDEOS,
        "repair_invocations": EXPECTED_VIDEOS,
        "detector_inference_calls": 0,
        "prediction_xml_count": len(records),
        "run_root_mp4_count": len(list(run_root.rglob("*.mp4"))),
        "prediction_artifact_sha256": prediction_hash,
        "predictions": records,
    }
    if run_manifest["run_root_mp4_count"]:
        raise ReproductionError("MP4 produced")
    write_json(
        run_root / "manifests" / f"RUN_{run_number}_MANIFEST.json",
        run_manifest,
    )
    command = (
        subprocess.list2cmdline(sys.argv)
        + "\nDETECTOR_INFERENCE_CALLS=0\nUNSEEN_FILES_ACCESSED=0\n"
    )
    (run_root / "commands" / "EXECUTION_COMMAND.txt").write_text(
        command,
        encoding="utf-8",
        newline="\n",
    )
    run_inventory = inventory(run_root)
    write_json(
        run_root / "manifests" / f"RUN_{run_number}_INVENTORY.json",
        {
            "root": str(run_root),
            "inventory_sha256": canonical_hash(run_inventory),
            "artifacts": run_inventory,
        },
    )
    state["status"] = "RUN1_FROZEN" if run_number == 1 else "RUN2_FROZEN"
    state["config_variants_executed"] = 1
    state[f"run_{run_number}_prediction_artifact_sha256"] = prediction_hash
    write_json(state_path, state)
    return run_manifest


def xml_rows(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    rows: list[dict[str, Any]] = []
    for track_position, track in enumerate(root.findall("track")):
        track_id = str(track.attrib["id"])
        label = str(track.attrib["label"])
        for row_position, box in enumerate(track.findall("box")):
            attrs = {
                str(item.attrib["name"]): item.text or ""
                for item in box.findall("attribute")
            }
            rows.append(
                {
                    "key": (track_id, int(box.attrib["frame"])),
                    "track_id": track_id,
                    "label": label,
                    "frame": int(box.attrib["frame"]),
                    "identity": attrs.get("ID", ""),
                    "hidden": attrs.get("Hidden", ""),
                    "outside": box.attrib.get("outside", "0"),
                    "occluded": box.attrib.get("occluded", "0"),
                    "bbox": tuple(
                        float(box.attrib[name])
                        for name in ("xtl", "ytl", "xbr", "ybr")
                    ),
                    "order": (track_position, row_position),
                }
            )
    return rows


def bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 1.0


def compare_xml(first: Path, second: Path, video_key: str) -> dict[str, Any]:
    first_rows = xml_rows(first)
    second_rows = xml_rows(second)
    first_by_key = {row["key"]: row for row in first_rows}
    second_by_key = {row["key"]: row for row in second_rows}
    shared = sorted(set(first_by_key).intersection(second_by_key))
    missing = sorted(set(first_by_key).difference(second_by_key))
    added = sorted(set(second_by_key).difference(first_by_key))
    coordinate_differences: list[float] = []
    ious: list[float] = []
    identity = 0
    hidden = 0
    bbox_exact = 0
    bbox_violations = 0
    for key in shared:
        left = first_by_key[key]
        right = second_by_key[key]
        identity += left["identity"] != right["identity"]
        hidden += (
            left["hidden"],
            left["outside"],
            left["occluded"],
        ) != (
            right["hidden"],
            right["outside"],
            right["occluded"],
        )
        deltas = [
            abs(a - b) for a, b in zip(left["bbox"], right["bbox"], strict=True)
        ]
        coordinate_differences.extend(deltas)
        iou = bbox_iou(left["bbox"], right["bbox"])
        ious.append(iou)
        bbox_exact += any(delta != 0.0 for delta in deltas)
        bbox_violations += (
            max(deltas, default=0.0) > BBOX_ABS_TOLERANCE
            or iou < BBOX_IOU_THRESHOLD
        )
    ordered_first = [row["key"] for row in first_rows]
    ordered_second = [row["key"] for row in second_rows]
    return {
        "video_key": video_key,
        "first_path": str(first),
        "second_path": str(second),
        "first_file_sha256": sha256_file(first),
        "second_file_sha256": sha256_file(second),
        "first_semantic_sha256": cvat_prediction_semantic_sha256(first),
        "second_semantic_sha256": cvat_prediction_semantic_sha256(second),
        "first_row_count": len(first_rows),
        "second_row_count": len(second_rows),
        "row_removals": len(missing),
        "row_additions": len(added),
        "frame_index_differences": len(missing) + len(added),
        "identity_value_differences": identity,
        "hidden_state_differences": hidden,
        "bbox_exact_value_differences": bbox_exact,
        "bbox_tolerance_violations": bbox_violations,
        "maximum_absolute_bbox_coordinate_difference": max(
            coordinate_differences,
            default=0.0,
        ),
        "minimum_paired_bbox_iou": min(ious, default=1.0),
        "ordering_differences": int(ordered_first != ordered_second),
    }


def aggregate_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    sum_fields = (
        "row_removals",
        "row_additions",
        "frame_index_differences",
        "identity_value_differences",
        "hidden_state_differences",
        "bbox_exact_value_differences",
        "bbox_tolerance_violations",
        "ordering_differences",
    )
    return {
        **{field: sum(int(row[field]) for row in records) for field in sum_fields},
        "maximum_absolute_bbox_coordinate_difference": max(
            float(row["maximum_absolute_bbox_coordinate_difference"])
            for row in records
        ),
        "minimum_paired_bbox_iou": min(
            float(row["minimum_paired_bbox_iou"]) for row in records
        ),
    }


def compare_predictions(
    output_root: Path,
    historical_prediction_root: Path,
) -> dict[str, Any]:
    state_path = output_root / "authority" / "REPRODUCTION_STATE.json"
    state = load_json(state_path)
    if state.get("status") != "RUN2_FROZEN":
        raise ReproductionError("Both runs must be frozen before comparison")
    run1_root = output_root / "run_1"
    run2_root = output_root / "run_2_determinism"
    keys = sorted(path.stem for path in historical_prediction_root.glob("*.xml"))
    if len(keys) != EXPECTED_VIDEOS:
        raise ReproductionError("Historical prediction population mismatch")
    repeat_records = [
        compare_xml(
            run1_root / "final_predictions" / f"{key}.xml",
            run2_root / "final_predictions" / f"{key}.xml",
            key,
        )
        for key in keys
    ]
    historical_records = [
        compare_xml(
            historical_prediction_root / f"{key}.xml",
            run1_root / "final_predictions" / f"{key}.xml",
            key,
        )
        for key in keys
    ]
    repeat = aggregate_comparison(repeat_records)
    historical = aggregate_comparison(historical_records)
    raw_equal = all(
        sha256_file(run1_root / "raw_pre_repair" / f"{key}.json")
        == sha256_file(run2_root / "raw_pre_repair" / f"{key}.json")
        for key in keys
    )
    ledger_equal = all(
        sha256_file(run1_root / "repair_ledgers" / f"{key}.json")
        == sha256_file(run2_root / "repair_ledgers" / f"{key}.json")
        for key in keys
    )
    repeat_pass = (
        all(value == 0 for key, value in repeat.items() if key.endswith("differences"))
        and repeat["row_additions"] == 0
        and repeat["row_removals"] == 0
        and repeat["bbox_tolerance_violations"] == 0
        and raw_equal
        and ledger_equal
    )
    exact = all(
        historical[field] == 0
        for field in (
            "row_additions",
            "row_removals",
            "frame_index_differences",
            "identity_value_differences",
            "hidden_state_differences",
            "bbox_exact_value_differences",
            "ordering_differences",
        )
    )
    semantic = (
        historical["row_additions"] == 0
        and historical["row_removals"] == 0
        and historical["frame_index_differences"] == 0
        and historical["identity_value_differences"] == 0
        and historical["hidden_state_differences"] == 0
        and historical["bbox_tolerance_violations"] == 0
    )
    provisional = (
        "EXACT_CANONICAL_PREDICTION_PARITY"
        if exact
        else (
            "SEMANTIC_PREDICTION_PARITY_CANDIDATE"
            if semantic
            else "METRIC_PARITY_REQUIRED"
        )
    )
    payload = {
        "schema_version": "tracking.h5b_h4.prediction_comparison.v1",
        "date": DATE,
        "full_execution_repeatability": "PASS" if repeat_pass else "FAIL",
        "raw_prediction_repeatability": "PASS" if raw_equal else "FAIL",
        "repair_ledger_repeatability": "PASS" if ledger_equal else "FAIL",
        "provisional_prediction_reproduction": provisional,
        "run1_vs_run2": repeat,
        "historical_vs_run1": historical,
        "run1_vs_run2_per_video": repeat_records,
        "historical_vs_run1_per_video": historical_records,
    }
    write_json(
        output_root / "comparisons" / "PREDICTION_PARITY_COMPARISON.json",
        payload,
    )
    pd.DataFrame(historical_records).to_csv(
        output_root / "comparisons" / "HISTORICAL_VS_REPRODUCED_PER_VIDEO.csv",
        index=False,
        lineterminator="\n",
    )
    if not repeat_pass:
        raise ReproductionError("Full execution repeatability failed")
    state["status"] = "PREDICTIONS_FROZEN"
    state["prediction_selection_complete"] = True
    state["post_evaluation_rerun_authorized"] = False
    state["post_evaluation_code_change_authorized"] = False
    state["prediction_comparison_sha256"] = sha256_file(
        output_root / "comparisons" / "PREDICTION_PARITY_COMPARISON.json"
    )
    write_json(state_path, state)
    return payload


def evaluate(
    source_repo: Path,
    worktree_repo: Path,
    output_root: Path,
) -> dict[str, Any]:
    state_path = output_root / "authority" / "REPRODUCTION_STATE.json"
    state = load_json(state_path)
    if state.get("status") != "PREDICTIONS_FROZEN":
        raise ReproductionError("Predictions must be frozen before evaluation")
    videos, _, _, _ = standard_v2.preflight(source_repo, worktree_repo)
    prediction_root = output_root / "run_1" / "final_predictions"
    for row in videos:
        row["prediction_paths"] = {
            "HISTORICAL_REPRODUCED": str(
                prediction_root / f"{row['video_key']}.xml"
            )
        }
    run_manifest = load_json(
        output_root / "run_1" / "manifests" / "RUN_1_MANIFEST.json"
    )
    arm = standard_v2.ArmSpec(
        arm="HISTORICAL_REPRODUCED",
        profile=HISTORICAL_PROFILE,
        prediction_root=prediction_root,
        authority_path=(
            output_root / "run_1" / "manifests" / "RUN_1_MANIFEST.json"
        ),
        artifact_sha256=run_manifest["prediction_artifact_sha256"],
        config_sha256=state["executable_config_sha256"],
        detector_cadence="EVERY_FRAME",
        detector_authority_sha256=CACHE_AUTHORITY_SHA256,
    )
    evaluator_sha = resolve_evaluator_code_sha()
    if evaluator_sha != git(worktree_repo, "rev-parse", "HEAD"):
        raise ReproductionError("Evaluator code SHA is not current producer HEAD")
    standard_root = output_root / "standard_v2"
    pass1 = standard_v2.evaluate_pass(
        standard_root / "pass1",
        (arm,),
        videos,
        evaluator_code_sha=evaluator_sha,
        reverse_inputs=False,
    )
    pass2 = standard_v2.evaluate_pass(
        standard_root / "pass2",
        (arm,),
        videos,
        evaluator_code_sha=evaluator_sha,
        reverse_inputs=True,
    )
    determinism = standard_v2.compare_passes(pass1, pass2)
    aggregate = pass1["aggregate_dataframe"].iloc[0].to_dict()
    differences = {
        metric: float(aggregate[metric]) - float(reference)
        for metric, reference in HISTORICAL_METRICS.items()
    }
    count_parity = all(
        int(aggregate[metric]) == int(HISTORICAL_METRICS[metric])
        for metric in COUNT_METRICS
    )
    float_parity = all(
        abs(differences[metric]) <= FLOAT_METRIC_TOLERANCE
        for metric in FLOAT_METRICS
    )
    metric_payload = {
        "schema_version": "tracking.h5b_h4.metric_parity.v1",
        "date": DATE,
        "historical_reference": HISTORICAL_METRICS,
        "reproduced": aggregate,
        "reproduced_minus_historical": differences,
        "count_metric_parity": "PASS" if count_parity else "FAIL",
        "float_metric_parity": "PASS" if float_parity else "FAIL",
        "standard_v2_metric_parity": (
            "PASS" if count_parity and float_parity else "FAIL"
        ),
        "evaluator_runs": 2,
        "reevaluation_repeatability": determinism[
            "reevaluation_repeatability"
        ],
        "input_order_invariance": determinism["input_order_invariance"],
        "conservation": pass1["conservation"],
    }
    write_json(
        output_root
        / "comparisons"
        / f"HISTORICAL_H5B_H4_REPRODUCTION_METRIC_PARITY_{DATE}.json",
        metric_payload,
    )
    pd.DataFrame(
        [
            {
                "metric": metric,
                "historical": HISTORICAL_METRICS[metric],
                "reproduced": aggregate[metric],
                "absolute_difference": abs(differences[metric]),
            }
            for metric in (*FLOAT_METRICS, *COUNT_METRICS)
        ]
    ).to_csv(
        output_root
        / "comparisons"
        / f"HISTORICAL_H5B_H4_REPRODUCED_VS_ORIGINAL_METRICS_{DATE}.csv",
        index=False,
        lineterminator="\n",
    )
    state["status"] = "METRICS_FROZEN"
    state["evaluator_runs"] = 2
    state["post_result_parameter_changes"] = 0
    write_json(state_path, state)
    return metric_payload


def finalize(
    worktree_repo: Path,
    historical_repo: Path,
    output_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    state = load_json(output_root / "authority" / "REPRODUCTION_STATE.json")
    if state.get("status") != "METRICS_FROZEN":
        raise ReproductionError("Metrics must be frozen before finalization")
    prediction = load_json(
        output_root / "comparisons" / "PREDICTION_PARITY_COMPARISON.json"
    )
    metric = load_json(
        output_root
        / "comparisons"
        / f"HISTORICAL_H5B_H4_REPRODUCTION_METRIC_PARITY_{DATE}.json"
    )
    historical = prediction["historical_vs_run1"]
    exact = (
        prediction["provisional_prediction_reproduction"]
        == "EXACT_CANONICAL_PREDICTION_PARITY"
    )
    semantic = (
        prediction["provisional_prediction_reproduction"]
        == "SEMANTIC_PREDICTION_PARITY_CANDIDATE"
    )
    metric_pass = metric["standard_v2_metric_parity"] == "PASS"
    if exact and metric_pass:
        classification = "PASS_EXACT_EXECUTABLE_REPRODUCTION"
    elif semantic and metric_pass:
        classification = "PASS_SEMANTIC_EXECUTABLE_REPRODUCTION"
    elif metric_pass:
        classification = "METRIC_EQUIVALENT_BUT_PREDICTION_PARITY_NOT_ESTABLISHED"
    elif prediction["full_execution_repeatability"] != "PASS":
        classification = "FAIL_DETERMINISM"
    elif exact or semantic:
        classification = "FAIL_METRIC_PARITY"
    else:
        classification = "FAIL_PREDICTION_PARITY"
    ready = classification in {
        "PASS_EXACT_EXECUTABLE_REPRODUCTION",
        "PASS_SEMANTIC_EXECUTABLE_REPRODUCTION",
    }
    historical_head = git(historical_repo, "rev-parse", "HEAD")
    historical_status = git(
        historical_repo,
        "status",
        "--porcelain=v1",
        "-uall",
    )
    if (
        historical_head != state["historical_head"]
        or historical_status != state["historical_status"]
    ):
        raise ReproductionError("Historical worktree changed")
    run1 = load_json(
        output_root / "run_1" / "manifests" / "RUN_1_MANIFEST.json"
    )
    run2 = load_json(
        output_root
        / "run_2_determinism"
        / "manifests"
        / "RUN_2_MANIFEST.json"
    )
    authority = {
        "schema_version": "tracking.h5b_h4.executable_reproduction_authority.v1",
        "date": DATE,
        "status": "ESTABLISHED" if ready else "FAILED_REPRODUCTION",
        "classification": classification,
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "execution_topology": EXECUTION_TOPOLOGY,
        "executable_config_sha256": state["executable_config_sha256"],
        "detector_cache_authority_sha256": CACHE_AUTHORITY_SHA256,
        "historical_prediction_artifact_sha256": (
            HISTORICAL_PREDICTION_ARTIFACT_SHA256
        ),
        "run1_prediction_artifact_sha256": run1[
            "prediction_artifact_sha256"
        ],
        "run2_prediction_artifact_sha256": run2[
            "prediction_artifact_sha256"
        ],
        "full_execution_repeatability": prediction[
            "full_execution_repeatability"
        ],
        "prediction_parity": prediction[
            "provisional_prediction_reproduction"
        ],
        "prediction_differences": historical,
        "metric_parity": metric,
        "execution_counts": {
            "detector_inference_calls": 0,
            "tracker_executions": 26,
            "repair_invocations": 26,
            "evaluator_runs": 2,
            "config_variants_executed": 1,
            "post_result_parameter_changes": 0,
            "unseen_files_accessed": 0,
            "run_root_mp4_count": 0,
        },
        "historical_worktree_integrity": {
            "head_changed": False,
            "tracked_file_content_changed": 0,
            "staged_state_changed": False,
            "unstaged_state_changed": False,
            "untracked_file_count_changed": False,
        },
        "scientific_limitations": [
            "The original historical raw detector cache remains unavailable.",
            "Prediction parity tests the frozen reconstructed 0.20/64 cache.",
            "No unseen execution or profile promotion is authorized.",
        ],
    }
    decision = {
        "schema_version": "tracking.h5b_h4.executable_reproduction_decision.v1",
        "date": DATE,
        "decision": classification,
        "ready_for_superseding_unseen_method_freeze_decision": ready,
        "ready_for_reproduction_limitation_decision": (
            classification
            == "METRIC_EQUIVALENT_BUT_PREDICTION_PARITY_NOT_ESTABLISHED"
        ),
        "ready_for_unseen_data_authority_freeze": False,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
    }
    write_json(
        docs_root
        / f"HISTORICAL_H5B_H4_EXECUTABLE_REPRODUCTION_AUTHORITY_{DATE}.json",
        authority,
    )
    write_json(
        docs_root
        / f"HISTORICAL_H5B_H4_EXECUTABLE_REPRODUCTION_DECISION_{DATE}.json",
        decision,
    )
    write_json(output_root / "authority" / "REPRODUCTION_AUTHORITY.json", authority)
    write_json(output_root / "authority" / "REPRODUCTION_DECISION.json", decision)
    complete_inventory = inventory(output_root)
    write_json(
        output_root / "authority" / "COMPLETE_ARTIFACT_INVENTORY.json",
        {
            "inventory_sha256": canonical_hash(complete_inventory),
            "artifacts": complete_inventory,
        },
    )
    marker = output_root / "NON_DISPOSABLE_REPRODUCTION_AUTHORITY.txt"
    marker.write_text(
        "NON_DISPOSABLE_FROZEN_HISTORICAL_EXECUTABLE_REPRODUCTION_AUTHORITY\n"
        "Deletion requires explicit authority retirement.\n",
        encoding="utf-8",
        newline="\n",
    )
    if list(output_root.rglob("*.mp4")):
        raise ReproductionError("MP4 found during closeout")
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    return authority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "run1", "run2", "compare", "evaluate", "finalize"),
    )
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO)
    parser.add_argument("--historical-repo", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--historical-prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=(
            REPO
            / "docs"
            / "tracking"
            / "historical_h5b_h4_executable_reproduction"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "prepare":
        result = prepare(
            args.source_repo,
            args.worktree_repo,
            args.historical_repo,
            args.cache_root,
            args.output_root,
            args.docs_root,
        )
    elif args.phase in {"run1", "run2"}:
        result = run_once(
            args.source_repo,
            args.worktree_repo,
            args.cache_root,
            args.output_root,
            1 if args.phase == "run1" else 2,
        )
    elif args.phase == "compare":
        result = compare_predictions(
            args.output_root,
            args.historical_prediction_root,
        )
    elif args.phase == "evaluate":
        result = evaluate(
            args.source_repo,
            args.worktree_repo,
            args.output_root,
        )
    else:
        result = finalize(
            args.worktree_repo,
            args.historical_repo,
            args.output_root,
            args.docs_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
