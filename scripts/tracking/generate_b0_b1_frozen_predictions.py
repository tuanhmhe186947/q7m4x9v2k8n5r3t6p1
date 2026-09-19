"""Generate and freeze B0/B1 predictions from full-frame detector replay.

This tool is intentionally not an evaluator.  It validates frozen authorities,
replays immutable detector evidence, runs one exact active tracking profile at
a time, and freezes structural prediction artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.tracking import (  # noqa: E402
    complete_full_frame_detector_cache as full_cache_tool,
)
from scripts.tracking import (  # noqa: E402
    generate_current_main_baseline_caches as baseline,
)

from pig_behavior.evaluation.tracking.lineage import (  # noqa: E402
    CVAT_PREDICTION_SEMANTIC_HASH_CONTRACT,
    cvat_prediction_semantic_sha256,
)
from pig_behavior.tracking.detector_cache import (  # noqa: E402
    DETECTOR_CACHE_SCHEMA_VERSION,
    DetectorCacheIdentity,
    DetectorEvidenceCache,
    ReplayDetector,
)

STARTING_MAIN_SHA = "8903151809a47557c8c9cfa6116e990eec5fcf92"
DATE_STAMP = "20260728"
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
EXPECTED_TOTAL_FRAMES = 23400

B0_PROFILE = "bytetrack_raw"
B1_PROFILE = "hybrid_bytetrack"
B0_CONFIG_SHA256 = (
    "547ae86e3be26671a9a148cb0e613ea1c602a0ff842a977ce9b7f1d217c10e41"
)
B1_CONFIG_SHA256 = (
    "4eb3d4e2262485d48d425be06fd8a6b3adfd8a01a27b28e76b5a8d55958d1d55"
)
B1_OFFLINE_REPAIR_SEMANTIC_SHA256 = (
    "e078b5b165dda82dee5b61e9465dc9844446e4cb576a02858c4ed7369828d758"
)
FULL_FRAME_CACHE_AUTHORITY_SHA256 = (
    "494d17ddb9d592dcf2105fd89a7204181f99a49353614315216d30ea43716e00"
)
DETECTOR_WEIGHTS_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
DETECTOR_SEMANTIC_CONFIG_SHA256 = (
    "2b50d8afa950626e2bed6b41807cb602a01a90e66baf7529fa08945d3d676ef8"
)
R0_PREDICTION_ARTIFACT_SHA256 = (
    "fd2d4f3dec0710d1c9eecba9308247a7b226dd34a4a02a9cb89f17acb22bbbfe"
)
R0_ARTIFACT_MANIFEST_SHA256 = (
    "461f2300318ab26134bb36beb78957a6642cbd361de9d74492f9cd09db688223"
)
R0_RUN_MANIFEST_SHA256 = (
    "9885138ab701b53fe7d4d1a85ba5a10bf9940ba39266409ef15d4dfba459d966"
)
SOURCE_VIDEO_MANIFEST_SHA256 = (
    "91289c9acb40958e59c17e98872714904f8df7e4c49682a4649e7fcc84bab9be"
)
GT_AUTHORITY_SHA256 = (
    "675cf37c4f924e391ffa457ba6c6e9453b967af318f37ecd2bc1ab1190a1d9dd"
)

ACTIVE_PROFILES = (
    "bytetrack_raw",
    "realtime_fast",
    "hybrid_bytetrack",
    "rf_hybrid_offline",
)
FULL_CACHE_AUTHORITY_FILE = (
    f"FULL_FRAME_DETECTOR_CACHE_AUTHORITY_{DATE_STAMP}.json"
)
FULL_CACHE_DECISION_FILE = (
    f"FULL_FRAME_DETECTOR_CACHE_DECISION_{DATE_STAMP}.json"
)
STATE_FILE = "B0_B1_GENERATION_STATE.json"
FROZEN_MARKER = "FROZEN_SCIENTIFIC_AUTHORITY_DO_NOT_DELETE.txt"
PREDICTION_HASH_CONTRACT = "tracking_prediction_xml_set_sha256_v1"
ROOT_HASH_CONTRACT = "tracking_prediction_root_inventory_sha256_v1"
RETENTION_CLASS = "NON_DISPOSABLE_FROZEN_PREDICTION_AUTHORITY"


class PredictionGenerationError(RuntimeError):
    """Fail-closed prediction generation or authority error."""


def utc_now() -> str:
    """Return one timezone-explicit UTC timestamp."""

    return datetime.now(UTC).isoformat()


def canonical_hash(payload: Any) -> str:
    """Hash a deterministic JSON representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON authority object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionGenerationError(
            f"invalid JSON authority: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PredictionGenerationError(
            f"JSON authority is not an object: {path}"
        )
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON for a generated mechanical artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def git_output(repo: Path, *args: str) -> str:
    """Return one Git command's text output."""

    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
    ).strip()


def require_clean_producer() -> tuple[str, str]:
    """Bind generation to a clean descendant with unchanged tracking source."""

    head = git_output(REPO, "rev-parse", "HEAD")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_MAIN_SHA, head],
        cwd=REPO,
        check=False,
    )
    if result.returncode:
        raise PredictionGenerationError(
            "starting-main authority is not an ancestor"
        )
    starting_tree = git_output(
        REPO,
        "rev-parse",
        f"{STARTING_MAIN_SHA}:src/pig_behavior/tracking",
    )
    current_tree = git_output(
        REPO,
        "rev-parse",
        "HEAD:src/pig_behavior/tracking",
    )
    if current_tree != starting_tree:
        raise PredictionGenerationError(
            "tracking subtree differs from starting main"
        )
    if git_output(REPO, "status", "--short"):
        raise PredictionGenerationError(
            "prediction producer worktree must be clean"
        )
    return head, current_tree


def profile_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact two active profile override payloads."""

    from pig_behavior.tracking.profiles import PRESENTATION_PROFILES
    from pig_behavior.tracking.profiles.bytetrack_raw import (
        EVAL_CONFIGS as RAW_CONFIGS,
    )
    from pig_behavior.tracking.profiles.hybrid_bytetrack import (
        EVAL_CONFIGS as HYBRID_CONFIGS,
    )

    if tuple(PRESENTATION_PROFILES) != ACTIVE_PROFILES:
        raise PredictionGenerationError(
            "active profile registry differs from frozen authority"
        )
    b0 = dict(RAW_CONFIGS[B0_PROFILE])
    b1 = dict(HYBRID_CONFIGS["hybrid_bytetrack_best"])
    if canonical_hash(b0) != B0_CONFIG_SHA256:
        raise PredictionGenerationError("B0 profile hash mismatch")
    if canonical_hash(b1) != B1_CONFIG_SHA256:
        raise PredictionGenerationError("B1 profile hash mismatch")
    return b0, b1


def build_tracking_config(
    source_repo: Path,
    video: Any,
    output_dir: Path,
    profile: str,
) -> Any:
    """Build one exact profile config with no scientific CLI overrides."""

    from pig_behavior.tracking.config import TrackingConfig, validate_config

    b0, b1 = profile_payloads()
    overrides = dict(b0 if profile == B0_PROFILE else b1)
    overrides.pop("mode", None)
    cfg = TrackingConfig(
        mode=profile,
        video_path=video.video_path,
        weights_path=(
            source_repo / "models" / "detector" / "pig_detector_yolov8.pt"
        ),
        mask_path=(
            source_repo / "data" / "annotations" / "scene" / "mask.png"
        ),
        output_dir=output_dir,
        device="cpu",
        half=False,
        write_output_video=False,
        show=False,
        start_frame=0,
        max_frames=EXPECTED_FRAMES,
        **overrides,
    )
    cfg.association_debug = False
    validate_config(cfg)
    return cfg


def verify_offline_repair_hash(
    source_repo: Path,
    video: Any,
    output_dir: Path,
) -> str:
    """Reproduce the frozen B1 offline-repair semantic hash."""

    from pig_behavior.tracking.offline_repair import (
        offline_repair_semantic_hash,
    )

    cfg = build_tracking_config(
        source_repo,
        video,
        output_dir,
        B1_PROFILE,
    )
    actual = offline_repair_semantic_hash(cfg)
    if actual != B1_OFFLINE_REPAIR_SEMANTIC_SHA256:
        raise PredictionGenerationError(
            "B1 offline-repair semantic hash mismatch"
        )
    return actual


def full_cache_path(cache_root: Path, video_key: str) -> Path:
    """Return the canonical full-frame partition path."""

    return (
        cache_root
        / "full"
        / "partitions"
        / video_key
        / "detector_evidence.npz"
    )


def expected_cache_identity(
    video: Any,
    cache_authority: dict[str, Any],
) -> DetectorCacheIdentity:
    """Reconstruct the exact full-frame cache identity."""

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=DETECTOR_WEIGHTS_SHA256,
        detector_semantic_config_sha256=(
            DETECTOR_SEMANTIC_CONFIG_SHA256
        ),
        producer_code_sha=str(cache_authority["producer_code_sha"]),
        creation_authority=full_cache_tool.FULL_CREATION_AUTHORITY,
    )


def verify_cache_authority(
    cache_root: Path,
    videos: list[Any],
    *,
    replay: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Validate authority, identity, hashes, coverage, and optional replay."""

    authority_path = cache_root / FULL_CACHE_AUTHORITY_FILE
    decision_path = cache_root / FULL_CACHE_DECISION_FILE
    authority = load_json(authority_path)
    decision = load_json(decision_path)
    if sha256_file(authority_path) != FULL_FRAME_CACHE_AUTHORITY_SHA256:
        raise PredictionGenerationError(
            "full-frame cache authority hash mismatch"
        )
    if decision.get("authority_sha256") != (
        FULL_FRAME_CACHE_AUTHORITY_SHA256
    ):
        raise PredictionGenerationError(
            "full-frame cache decision binding mismatch"
        )
    required = {
        "status": "ESTABLISHED",
        "full_frame_coverage": "PASS",
        "cache_replay": "PASS",
        "determinism": "PASS",
        "even_subset_parity": "PASS",
        "full_cache_frames": EXPECTED_TOTAL_FRAMES,
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "detector_semantic_config_sha256": (
            DETECTOR_SEMANTIC_CONFIG_SHA256
        ),
    }
    for field, expected in required.items():
        if authority.get(field) != expected:
            raise PredictionGenerationError(
                f"full-frame cache authority mismatch: {field}"
            )
    per_video = {
        str(row["video_key"]): row
        for row in authority.get("per_video", [])
    }
    if set(per_video) != {video.video_key for video in videos}:
        raise PredictionGenerationError(
            "full-frame cache video population mismatch"
        )
    records: list[dict[str, Any]] = []
    replay_count = 0
    for video in videos:
        expected = per_video[video.video_key]
        path = full_cache_path(cache_root, video.video_key)
        actual_sha = sha256_file(path)
        if actual_sha != expected["full_cache_sha256"]:
            raise PredictionGenerationError(
                f"full cache file hash mismatch: {video.video_key}"
            )
        cache = DetectorEvidenceCache.load(
            path,
            expected_identity=expected_cache_identity(
                video,
                authority,
            ),
        )
        if tuple(cache.frames) != tuple(range(EXPECTED_FRAMES)):
            raise PredictionGenerationError(
                f"full cache coverage mismatch: {video.video_key}"
            )
        if replay:
            detector = ReplayDetector(cache)
            for frame_index, entry in cache.frames.items():
                detector.set_frame_context(
                    frame_index,
                    entry["original_frame_dimensions"],
                )
                detector.track()
            if detector.invocations != EXPECTED_FRAMES:
                raise PredictionGenerationError(
                    f"cache replay count mismatch: {video.video_key}"
                )
            replay_count += detector.invocations
        records.append(
            {
                "video_key": video.video_key,
                "cache_path": str(path),
                "cache_sha256": actual_sha,
                "canonical_content_hash": expected[
                    "canonical_content_hash"
                ],
                "detector_record_count": len(cache.frames),
                "cache_schema": DETECTOR_CACHE_SCHEMA_VERSION,
            }
        )
    if replay and replay_count != EXPECTED_TOTAL_FRAMES:
        raise PredictionGenerationError(
            "cache dry-run did not replay all 23,400 records"
        )
    return authority, records, replay_count


def source_and_gt_authorities(
    videos: list[Any],
    source_lineage_sha256: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Build the established source and GT authority records."""

    gt_rows = [
        {
            "video_key": video.video_key,
            "gt_sha256": video.gt_sha256,
            "gt_authority": video.gt_authority,
        }
        for video in videos
    ]
    gt_hash = canonical_hash(gt_rows)
    if source_lineage_sha256 != SOURCE_VIDEO_MANIFEST_SHA256:
        raise PredictionGenerationError(
            "source-video manifest authority mismatch"
        )
    if gt_hash != GT_AUTHORITY_SHA256:
        raise PredictionGenerationError("GT authority hash mismatch")
    return source_lineage_sha256, gt_hash, gt_rows


def verify_r0_authority(source_repo: Path, videos: list[Any]) -> dict[str, Any]:
    """Revalidate the surviving R0 bytes without running R0."""

    r0_root = (
        source_repo / "outputs" / "tracking" / "current_main_baseline_20260728"
    )
    artifact_manifest = r0_root / "ARTIFACT_SHA256.json"
    run_manifest = r0_root / "CURRENT_MAIN_BASELINE_RUN_MANIFEST.json"
    if sha256_file(artifact_manifest) != R0_ARTIFACT_MANIFEST_SHA256:
        raise PredictionGenerationError("R0 artifact manifest hash mismatch")
    if sha256_file(run_manifest) != R0_RUN_MANIFEST_SHA256:
        raise PredictionGenerationError("R0 run manifest hash mismatch")
    inventory = load_json(artifact_manifest)
    xml_records = {
        str(record["relative_path"]): record
        for record in inventory.get("artifacts", [])
        if str(record["relative_path"]).endswith(".xml")
    }
    xml_paths = sorted((r0_root / path) for path in xml_records)
    if len(xml_paths) != EXPECTED_VIDEOS:
        raise PredictionGenerationError("R0 prediction XML count mismatch")
    expected_keys = {video.video_key for video in videos}
    observed_keys: set[str] = set()
    for path in xml_paths:
        relative = path.relative_to(r0_root).as_posix()
        if sha256_file(path) != xml_records[relative]["sha256"]:
            raise PredictionGenerationError(
                f"R0 prediction byte hash mismatch: {relative}"
            )
        matches = expected_keys.intersection(path.parts)
        if len(matches) != 1:
            raise PredictionGenerationError(
                f"R0 prediction video binding mismatch: {relative}"
            )
        observed_keys.update(matches)
    if observed_keys != expected_keys:
        raise PredictionGenerationError("R0 prediction population mismatch")
    return {
        "status": "PASS",
        "prediction_root": str(r0_root / "predictions"),
        "prediction_xml_count": len(xml_paths),
        "prediction_artifact_sha256": R0_PREDICTION_ARTIFACT_SHA256,
        "artifact_manifest_sha256": R0_ARTIFACT_MANIFEST_SHA256,
        "run_manifest_sha256": R0_RUN_MANIFEST_SHA256,
        "tracker_executions": 0,
    }


def population_manifest(
    videos: list[Any],
    cache_records: list[dict[str, Any]],
    *,
    source_lineage_path: Path,
    source_lineage_sha256: str,
    source_authority_sha256: str,
    gt_authority_sha256: str,
) -> dict[str, Any]:
    """Create the locked B0/B1 execution population manifest."""

    caches = {row["video_key"]: row for row in cache_records}
    rows = []
    for video in videos:
        unresolved = "_000216_" in video.video_key
        rows.append(
            {
                "video_key": video.video_key,
                "source_video_path": str(video.video_path),
                "source_video_sha256": video.video_sha256,
                "gt_path": str(video.gt_path),
                "gt_sha256": video.gt_sha256,
                "frame_start": 0,
                "frame_end": EXPECTED_FRAMES - 1,
                "expected_frame_count": EXPECTED_FRAMES,
                "full_frame_cache_path": caches[video.video_key][
                    "cache_path"
                ],
                "full_frame_cache_sha256": caches[video.video_key][
                    "cache_sha256"
                ],
                "detector_record_count": caches[video.video_key][
                    "detector_record_count"
                ],
                "source_video_session_boundary": video.video_key,
                "gt_authority_status": video.gt_authority,
                "aggregate_inclusion_role": (
                    "LOCKED_AGGREGATE_ONLY"
                    if unresolved
                    else "LOCKED_PRIMARY"
                ),
                "mechanism_ranking_eligibility": not unresolved,
            }
        )
    return {
        "schema_version": "tracking.b0_b1.locked_execution_manifest.v1",
        "date": DATE_STAMP,
        "status": "FROZEN_READY_FOR_REPLAY_TRACKING",
        "video_count": len(rows),
        "expected_frames_per_video": EXPECTED_FRAMES,
        "expected_total_frames": EXPECTED_TOTAL_FRAMES,
        "cache_records_per_video": EXPECTED_FRAMES,
        "cache_total_records": EXPECTED_TOTAL_FRAMES,
        "source_lineage_path": str(source_lineage_path),
        "source_lineage_sha256": source_lineage_sha256,
        "source_video_manifest_sha256": source_authority_sha256,
        "gt_authority_sha256": gt_authority_sha256,
        "full_frame_cache_authority_sha256": (
            FULL_FRAME_CACHE_AUTHORITY_SHA256
        ),
        "videos": rows,
    }


def environment_payload(producer_sha: str) -> dict[str, Any]:
    """Record the non-quality execution environment."""

    import cv2
    import numpy
    import torch

    try:
        import ultralytics

        ultralytics_version = str(ultralytics.__version__)
    except ImportError:
        ultralytics_version = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "ultralytics": ultralytics_version,
        "producer_code_sha": producer_sha,
    }


def preflight(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    output_root: Path,
) -> None:
    """Freeze all inputs and dry-run the full replay population."""

    if output_root.exists():
        raise PredictionGenerationError(
            f"refusing existing output root: {output_root}"
        )
    producer_sha, tracking_tree = require_clean_producer()
    b0, b1 = profile_payloads()
    videos, lineage_sha = baseline.load_population(
        source_repo,
        lineage_manifest,
    )
    if len(videos) != EXPECTED_VIDEOS:
        raise PredictionGenerationError("locked population is not 13 videos")
    source_hash, gt_hash, gt_rows = source_and_gt_authorities(
        videos,
        lineage_sha,
    )
    offline_hash = verify_offline_repair_hash(
        source_repo,
        videos[0],
        output_root / "_preflight_unused",
    )
    cache_authority, cache_records, replay_count = verify_cache_authority(
        cache_root,
        videos,
        replay=True,
    )
    r0 = verify_r0_authority(source_repo, videos)
    output_root.mkdir(parents=True)
    for relative in (
        "B0_bytetrack_raw/predictions",
        "B0_bytetrack_raw/machine_readable",
        "B0_bytetrack_raw/manifests",
        "B0_bytetrack_raw/audits",
        "B0_bytetrack_raw/commands",
        "B1_hybrid_bytetrack/predictions",
        "B1_hybrid_bytetrack/machine_readable",
        "B1_hybrid_bytetrack/repair_ledgers",
        "B1_hybrid_bytetrack/manifests",
        "B1_hybrid_bytetrack/audits",
        "B1_hybrid_bytetrack/commands",
    ):
        (output_root / relative).mkdir(parents=True)
    manifest = population_manifest(
        videos,
        cache_records,
        source_lineage_path=lineage_manifest,
        source_lineage_sha256=lineage_sha,
        source_authority_sha256=source_hash,
        gt_authority_sha256=gt_hash,
    )
    manifest_path = (
        output_root
        / f"B0_B1_LOCKED_EXECUTION_MANIFEST_{DATE_STAMP}.json"
    )
    write_json(manifest_path, manifest)
    primary_dirty_paths = [
        line
        for line in git_output(source_repo, "status", "--short").splitlines()
        if line
    ]
    preflight_payload = {
        "schema_version": "tracking.b0_b1.prediction_preflight.v1",
        "date": DATE_STAMP,
        "status": "PASS_READY_FOR_B0",
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": producer_sha,
        "tracking_tree_object": tracking_tree,
        "execution_topology": "SEPARATE_EXACT_PROFILE_EXECUTIONS",
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
        ],
        "b0_profile": B0_PROFILE,
        "b1_profile": B1_PROFILE,
        "b0_config": b0,
        "b0_config_sha256": canonical_hash(b0),
        "b1_config": b1,
        "b1_config_sha256": canonical_hash(b1),
        "b1_offline_repair_semantic_sha256": offline_hash,
        "active_profiles": list(ACTIVE_PROFILES),
        "full_frame_cache_root": str(cache_root),
        "full_frame_cache_authority_sha256": (
            FULL_FRAME_CACHE_AUTHORITY_SHA256
        ),
        "full_frame_cache_authority_status": cache_authority["status"],
        "cache_replay_dry_run": "PASS",
        "cache_replay_records": replay_count,
        "live_detector_fallback_enabled": False,
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "detector_semantic_config_sha256": (
            DETECTOR_SEMANTIC_CONFIG_SHA256
        ),
        "locked_execution_manifest_sha256": sha256_file(manifest_path),
        "source_video_manifest_sha256": source_hash,
        "gt_authority_sha256": gt_hash,
        "gt_authority": gt_rows,
        "r0_reuse_authority": r0,
        "primary_worktree_dirty_paths": primary_dirty_paths,
        "experiment_worktree_clean": True,
        "stale_cache_worktree_reused": False,
        "total_detector_inference_calls": 0,
        "r0_tracker_executions": 0,
        "r1_tracker_executions": 0,
        "standard_v2_metric_runs": 0,
        "legacy_metric_runs": 0,
        "quality_comparisons": 0,
        "unseen_videos_accessed": False,
        "run_root_mp4_count": 0,
        "created_at": utc_now(),
    }
    write_json(output_root / "B0_B1_PREFLIGHT.json", preflight_payload)
    write_json(
        output_root / STATE_FILE,
        {
            "schema_version": "tracking.b0_b1.generation_state.v1",
            "status": "READY",
            "phase": "AWAITING_B0",
            "producer_code_sha": producer_sha,
            "b0_videos_completed": 0,
            "b1_videos_completed": 0,
            "updated_at": utc_now(),
        },
    )


def verify_phase_inputs(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    output_root: Path,
    expected_phase: str,
) -> tuple[list[Any], dict[str, Any], str]:
    """Revalidate immutable authorities before an execution phase."""

    preflight_payload = load_json(output_root / "B0_B1_PREFLIGHT.json")
    state = load_json(output_root / STATE_FILE)
    if state.get("phase") != expected_phase:
        raise PredictionGenerationError(
            f"unexpected generation phase: {state.get('phase')}"
        )
    producer_sha, _ = require_clean_producer()
    if preflight_payload.get("producer_code_sha") != producer_sha:
        raise PredictionGenerationError("producer commit changed after preflight")
    videos, lineage_sha = baseline.load_population(
        source_repo,
        lineage_manifest,
    )
    manifest_path = (
        output_root
        / f"B0_B1_LOCKED_EXECUTION_MANIFEST_{DATE_STAMP}.json"
    )
    if sha256_file(manifest_path) != preflight_payload[
        "locked_execution_manifest_sha256"
    ]:
        raise PredictionGenerationError(
            "locked execution manifest changed after preflight"
        )
    if lineage_sha != load_json(manifest_path)["source_lineage_sha256"]:
        raise PredictionGenerationError("source lineage changed after preflight")
    cache_authority, _, _ = verify_cache_authority(
        cache_root,
        videos,
        replay=False,
    )
    verify_r0_authority(source_repo, videos)
    profile_payloads()
    verify_offline_repair_hash(
        source_repo,
        videos[0],
        output_root / "_phase_unused",
    )
    return videos, cache_authority, producer_sha


def update_state(
    output_root: Path,
    *,
    status: str,
    phase: str,
    b0_completed: int,
    b1_completed: int,
    video_key: str | None = None,
    failure: str | None = None,
) -> None:
    """Update the generated phase checkpoint."""

    preflight_payload = load_json(output_root / "B0_B1_PREFLIGHT.json")
    write_json(
        output_root / STATE_FILE,
        {
            "schema_version": "tracking.b0_b1.generation_state.v1",
            "status": status,
            "phase": phase,
            "producer_code_sha": preflight_payload["producer_code_sha"],
            "b0_videos_completed": b0_completed,
            "b1_videos_completed": b1_completed,
            "current_video_key": video_key,
            "failure": failure,
            "updated_at": utc_now(),
        },
    )


def xml_structural_record(
    path: Path,
    *,
    video_key: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Validate structural prediction content without evaluating quality."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PredictionGenerationError(
            f"invalid prediction XML: {path}"
        ) from exc
    task_name = root.findtext("./meta/task/name")
    task_size = root.findtext("./meta/task/size")
    if task_name != video_key or task_size != str(EXPECTED_FRAMES):
        raise PredictionGenerationError(
            f"prediction XML authority mismatch: {video_key}"
        )
    seen_track_ids: set[int] = set()
    rows: list[tuple[Any, ...]] = []
    hidden_values: set[str] = set()
    minimum_frame: int | None = None
    maximum_frame: int | None = None
    for track in root.findall("./track"):
        try:
            track_id = int(track.attrib["id"])
        except (KeyError, ValueError) as exc:
            raise PredictionGenerationError(
                f"invalid serialized track ID: {video_key}"
            ) from exc
        if track_id < 0 or track_id in seen_track_ids:
            raise PredictionGenerationError(
                f"duplicate or negative track ID: {video_key}"
            )
        seen_track_ids.add(track_id)
        label = str(track.attrib.get("label", ""))
        for box in track.findall("./box"):
            try:
                frame = int(box.attrib["frame"])
                coords = tuple(
                    float(box.attrib[name])
                    for name in ("xtl", "ytl", "xbr", "ybr")
                )
            except (KeyError, ValueError) as exc:
                raise PredictionGenerationError(
                    f"invalid prediction box: {video_key}"
                ) from exc
            if frame < 0 or frame >= EXPECTED_FRAMES:
                raise PredictionGenerationError(
                    f"prediction frame outside authority: {video_key}"
                )
            if not all(math.isfinite(value) for value in coords):
                raise PredictionGenerationError(
                    f"non-finite prediction box: {video_key}"
                )
            xtl, ytl, xbr, ybr = coords
            if not (
                0 <= xtl <= xbr <= width
                and 0 <= ytl <= ybr <= height
            ):
                raise PredictionGenerationError(
                    f"prediction bbox outside frame: {video_key}"
                )
            attributes = tuple(
                sorted(
                    (
                        str(attribute.attrib.get("name", "")),
                        str(attribute.text or ""),
                    )
                    for attribute in box.findall("./attribute")
                )
            )
            for name, value in attributes:
                if name == "Hidden":
                    if value not in {"Yes", "No"}:
                        raise PredictionGenerationError(
                            f"invalid Hidden value: {video_key}"
                        )
                    hidden_values.add(value)
            rows.append(
                (
                    track_id,
                    label,
                    frame,
                    xtl,
                    ytl,
                    xbr,
                    ybr,
                    attributes,
                )
            )
            minimum_frame = (
                frame if minimum_frame is None else min(minimum_frame, frame)
            )
            maximum_frame = (
                frame if maximum_frame is None else max(maximum_frame, frame)
            )
    canonical_rows = sorted(rows)
    canonical_rows_hash = canonical_hash(canonical_rows)
    permuted_rows_hash = canonical_hash(
        sorted(list(reversed(canonical_rows)))
    )
    if canonical_rows_hash != permuted_rows_hash:
        raise PredictionGenerationError(
            f"prediction row canonicalization failed: {video_key}"
        )
    return {
        "video_key": video_key,
        "xml_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "semantic_sha256": cvat_prediction_semantic_sha256(path),
        "semantic_hash_contract": CVAT_PREDICTION_SEMANTIC_HASH_CONTRACT,
        "canonical_row_sha256": canonical_rows_hash,
        "track_count": len(seen_track_ids),
        "prediction_object_count": len(rows),
        "minimum_prediction_frame": minimum_frame,
        "maximum_prediction_frame": maximum_frame,
        "processed_frame_start": 0,
        "processed_frame_end": EXPECTED_FRAMES - 1,
        "processed_frame_count": EXPECTED_FRAMES,
        "hidden_values_present": sorted(hidden_values),
        "bbox_validity": "PASS",
        "identity_serialization": "PASS",
        "deterministic_object_ordering": "PASS_CANONICALIZED",
    }


def prediction_set_hash(records: list[dict[str, Any]]) -> str:
    """Hash the semantic prediction population independent of input order."""

    payload = [
        {
            "video_key": record["video_key"],
            "prediction_xml_sha256": record["sha256"],
            "prediction_semantic_sha256": record["semantic_sha256"],
            "canonical_row_sha256": record["canonical_row_sha256"],
            "prediction_object_count": record["prediction_object_count"],
            "processed_frame_count": record["processed_frame_count"],
        }
        for record in sorted(records, key=lambda item: item["video_key"])
    ]
    return canonical_hash(
        {
            "contract": PREDICTION_HASH_CONTRACT,
            "predictions": payload,
        }
    )


def artifact_inventory(arm_root: Path) -> list[dict[str, Any]]:
    """Inventory non-self-referential retained scientific artifacts."""

    included_roots = [
        arm_root / "predictions",
        arm_root / "machine_readable",
        arm_root / "repair_ledgers",
        arm_root / "commands",
    ]
    paths = [
        path
        for root in included_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    marker = arm_root / FROZEN_MARKER
    if marker.is_file():
        paths.append(marker)
    records = []
    for path in sorted(set(paths)):
        relative = path.relative_to(arm_root).as_posix()
        role = relative.split("/", maxsplit=1)[0]
        records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_role": role,
            }
        )
    return records


def arm_paths(output_root: Path, profile: str) -> tuple[str, Path]:
    """Return arm label and stable root for one authorized profile."""

    if profile == B0_PROFILE:
        return "B0", output_root / "B0_bytetrack_raw"
    if profile == B1_PROFILE:
        return "B1", output_root / "B1_hybrid_bytetrack"
    raise PredictionGenerationError(f"unauthorized profile: {profile}")


def copy_machine_xml(
    summary: Any,
    prediction_root: Path,
    video_key: str,
) -> Path:
    """Copy one completed XML into the canonical one-file-per-video root."""

    destination = prediction_root / f"{video_key}.xml"
    if destination.exists():
        raise PredictionGenerationError(
            f"refusing prediction overwrite: {destination}"
        )
    shutil.copy2(Path(summary.cvat_video_xml), destination)
    return destination


def marker_text(profile: str, producer_sha: str) -> str:
    """Return the visible non-disposable authority marker."""

    return (
        "FROZEN SCIENTIFIC PREDICTION AUTHORITY\n"
        f"profile={profile}\n"
        f"producer_code_sha={producer_sha}\n"
        f"retention_class={RETENTION_CLASS}\n"
        "deletion_allowed=NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT\n"
        "This directory is not temporary output. Do not modify or delete it.\n"
    )


def freeze_files(root: Path) -> None:
    """Mark retained files read-only where the platform permits."""

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            os.chmod(path, stat.S_IREAD)
        except OSError as exc:
            raise PredictionGenerationError(
                f"could not freeze artifact read-only: {path}"
            ) from exc


def assert_arm_immutable(
    arm_root: Path,
    expected_inventory_hash: str,
) -> None:
    """Rehash a frozen arm and reject any post-freeze mutation."""

    actual = canonical_hash(artifact_inventory(arm_root))
    if actual != expected_inventory_hash:
        raise PredictionGenerationError(
            f"post-freeze artifact mutation: {arm_root.name}"
        )


def run_arm(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    output_root: Path,
    profile: str,
) -> None:
    """Run and freeze one exact profile over all 13 locked videos."""

    from pig_behavior.tracking.runner import run_tracking

    arm, arm_root = arm_paths(output_root, profile)
    expected_phase = "AWAITING_B0" if arm == "B0" else "AWAITING_B1"
    videos, cache_authority, producer_sha = verify_phase_inputs(
        source_repo,
        lineage_manifest,
        cache_root,
        output_root,
        expected_phase,
    )
    prediction_root = arm_root / "predictions"
    machine_root = arm_root / "machine_readable"
    if any(prediction_root.iterdir()) or any(machine_root.iterdir()):
        raise PredictionGenerationError(
            f"refusing non-empty {arm} output directories"
        )
    if arm == "B1":
        b0_manifest = load_json(
            output_root
            / "B0_bytetrack_raw"
            / "manifests"
            / f"B0_BYTETRACK_RAW_PREDICTION_ARTIFACT_MANIFEST_{DATE_STAMP}.json"
        )
        assert_arm_immutable(
            output_root / "B0_bytetrack_raw",
            b0_manifest["recursive_artifact_inventory_sha256"],
        )
    records: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    completed = 0
    replay_calls = 0
    for video in videos:
        update_state(
            output_root,
            status="RUNNING",
            phase=f"RUNNING_{arm}",
            b0_completed=completed if arm == "B0" else EXPECTED_VIDEOS,
            b1_completed=completed if arm == "B1" else 0,
            video_key=video.video_key,
        )
        started_at = utc_now()
        cache_path = full_cache_path(cache_root, video.video_key)
        cache = DetectorEvidenceCache.load(
            cache_path,
            expected_identity=expected_cache_identity(
                video,
                cache_authority,
            ),
        )
        detector = ReplayDetector(cache)
        video_output = machine_root / video.video_key
        cfg = build_tracking_config(
            source_repo,
            video,
            video_output,
            profile,
        )
        print(f"{arm}_BEGIN {video.video_key}", flush=True)
        summary = run_tracking(cfg, model=detector)
        if detector.invocations != EXPECTED_FRAMES:
            raise PredictionGenerationError(
                f"{arm} replay count mismatch: {video.video_key}"
            )
        if summary.frames_read != EXPECTED_FRAMES:
            raise PredictionGenerationError(
                f"{arm} frame coverage mismatch: {video.video_key}"
            )
        if summary.raw_annotations_json is not None:
            raise PredictionGenerationError(
                f"{arm} unexpectedly entered RF raw-output path"
            )
        if summary.repair_ledger_json is not None:
            raise PredictionGenerationError(
                f"{arm} unexpectedly entered RF repair-ledger path"
            )
        prediction_xml = copy_machine_xml(
            summary,
            prediction_root,
            video.video_key,
        )
        record = xml_structural_record(
            prediction_xml,
            video_key=video.video_key,
            width=video.width,
            height=video.height,
        )
        records.append(record)
        replay_calls += detector.invocations
        completed += 1
        execution_rows.append(
            {
                "video_key": video.video_key,
                "started_at": started_at,
                "completed_at": utc_now(),
                "frames_processed": summary.frames_read,
                "cache_records_consumed": detector.invocations,
                "prediction_object_count": summary.shape_count,
                "prediction_xml": str(prediction_xml),
                "prediction_xml_sha256": record["sha256"],
                "detector_inference_calls": 0,
                "mp4_count": 0,
                "status": "PASS",
            }
        )
        print(
            f"{arm}_END {video.video_key} "
            f"objects={record['prediction_object_count']}",
            flush=True,
        )
    if completed != EXPECTED_VIDEOS:
        raise PredictionGenerationError(f"{arm} did not complete 13 videos")
    if replay_calls != EXPECTED_TOTAL_FRAMES:
        raise PredictionGenerationError(
            f"{arm} did not consume 23,400 cache records"
        )
    xml_paths = sorted(prediction_root.glob("*.xml"))
    if len(xml_paths) != EXPECTED_VIDEOS:
        raise PredictionGenerationError(
            f"{arm} prediction XML count is not 13"
        )
    mp4_count = len(list(arm_root.rglob("*.mp4")))
    if mp4_count:
        raise PredictionGenerationError(f"{arm} produced MP4 files")
    if arm == "B1":
        write_json(
            arm_root
            / "repair_ledgers"
            / "B1_REPAIR_LEDGER_CONTRACT.json",
            {
                "schema_version": "tracking.b1.repair_ledger_contract.v1",
                "status": "NOT_EXPOSED_BY_ACTIVE_CONTRACT",
                "profile": B1_PROFILE,
                "rf_adapter_used": False,
                "rf_hybrid_offline_code_path_used": False,
                "offline_repair_semantic_sha256": (
                    B1_OFFLINE_REPAIR_SEMANTIC_SHA256
                ),
                "note": (
                    "The active hybrid_bytetrack runner exposes a final "
                    "post-repair XML but no separate repair-ledger output."
                ),
            },
        )
    command_path = arm_root / "commands" / f"{arm}_EXECUTION_COMMAND.txt"
    command_path.write_text(
        subprocess.list2cmdline(sys.argv)
        + "\nTOTAL_DETECTOR_INFERENCE_CALLS=0\n"
        + "STANDARD_V2_METRIC_RUNS=0\n"
        + "LEGACY_METRIC_RUNS=0\n"
        + "QUALITY_COMPARISONS=0\n"
        + "RUN_ROOT_MP4_COUNT=0\n",
        encoding="utf-8",
    )
    marker = arm_root / FROZEN_MARKER
    marker.write_text(
        marker_text(profile, producer_sha),
        encoding="utf-8",
    )
    prediction_hash = prediction_set_hash(records)
    conservation = {
        "schema_version": f"tracking.{arm.lower()}.prediction_conservation.v1",
        "date": DATE_STAMP,
        "status": "PASS",
        "profile": profile,
        "video_count": completed,
        "prediction_xml_count": len(xml_paths),
        "prediction_object_count": sum(
            int(record["prediction_object_count"]) for record in records
        ),
        "processed_frames": EXPECTED_TOTAL_FRAMES,
        "cache_records_consumed": replay_calls,
        "prediction_set_sha256": prediction_hash,
        "prediction_hash_contract": PREDICTION_HASH_CONTRACT,
        "locked_video_key_set": "PASS",
        "frame_range_coverage": "PASS",
        "bbox_validity": "PASS",
        "finite_numeric_values": "PASS",
        "identity_serialization": "PASS",
        "no_cross_video_identity_state": "PASS_SEPARATE_RUN_TRACKER_STATE",
        "deterministic_object_ordering": "PASS_CANONICALIZED",
        "hidden_value_policy": "PRESERVED_WHEN_PRESENT_NOT_INFERRED",
        "evaluator_produced_modifications": 0,
        "detector_inference_calls": 0,
        "mp4_count": mp4_count,
        "per_video": records,
    }
    write_json(
        arm_root
        / "audits"
        / f"{arm}_PREDICTION_CONSERVATION_{DATE_STAMP}.json",
        conservation,
    )
    write_json(
        arm_root
        / "audits"
        / f"{arm}_EXECUTION_AUDIT_{DATE_STAMP}.json",
        {
            "schema_version": f"tracking.{arm.lower()}.execution_audit.v1",
            "profile": profile,
            "execution_topology": "SEPARATE_EXACT_PROFILE_EXECUTIONS",
            "producer_code_sha": producer_sha,
            "environment": environment_payload(producer_sha),
            "videos": execution_rows,
            "tracker_executions": completed,
            "cache_records_consumed": replay_calls,
            "detector_inference_calls": 0,
            "standard_v2_metric_runs": 0,
            "legacy_metric_runs": 0,
            "quality_comparisons": 0,
            "hota_values_generated": 0,
            "idf1_values_generated": 0,
            "idsw_values_generated": 0,
            "mp4_count": 0,
            "unseen_videos_accessed": False,
        },
    )
    inventory = artifact_inventory(arm_root)
    inventory_hash = canonical_hash(inventory)
    manifest_name = (
        f"{arm}_BYTETRACK_RAW_PREDICTION_ARTIFACT_MANIFEST_{DATE_STAMP}.json"
        if arm == "B0"
        else (
            f"{arm}_HYBRID_BYTETRACK_PREDICTION_ARTIFACT_MANIFEST_"
            f"{DATE_STAMP}.json"
        )
    )
    manifest = {
        "schema_version": f"tracking.{arm.lower()}.prediction_manifest.v1",
        "date": DATE_STAMP,
        "status": "FROZEN",
        "profile": profile,
        "retention_class": RETENTION_CLASS,
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "output_root": str(arm_root),
        "recursive_artifact_inventory_sha256": inventory_hash,
        "root_hash_contract": ROOT_HASH_CONTRACT,
        "canonical_prediction_content_sha256": prediction_hash,
        "prediction_hash_contract": PREDICTION_HASH_CONTRACT,
        "profile_config_sha256": (
            B0_CONFIG_SHA256 if arm == "B0" else B1_CONFIG_SHA256
        ),
        "offline_repair_semantic_sha256": (
            "NOT_APPLICABLE"
            if arm == "B0"
            else B1_OFFLINE_REPAIR_SEMANTIC_SHA256
        ),
        "full_frame_cache_authority_sha256": (
            FULL_FRAME_CACHE_AUTHORITY_SHA256
        ),
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "detector_semantic_config_sha256": (
            DETECTOR_SEMANTIC_CONFIG_SHA256
        ),
        "producer_code_sha": producer_sha,
        "video_count": completed,
        "processed_frame_count": EXPECTED_TOTAL_FRAMES,
        "prediction_object_count": conservation[
            "prediction_object_count"
        ],
        "artifacts": inventory,
        "predictions": records,
    }
    write_json(arm_root / "manifests" / manifest_name, manifest)
    freeze_files(arm_root)
    assert_arm_immutable(arm_root, inventory_hash)
    update_state(
        output_root,
        status="READY",
        phase="AWAITING_B1" if arm == "B0" else "AWAITING_FINALIZE",
        b0_completed=EXPECTED_VIDEOS,
        b1_completed=0 if arm == "B0" else EXPECTED_VIDEOS,
    )


def read_arm_manifest(output_root: Path, arm: str) -> dict[str, Any]:
    """Read one completed immutable arm manifest."""

    if arm == "B0":
        path = (
            output_root
            / "B0_bytetrack_raw"
            / "manifests"
            / f"B0_BYTETRACK_RAW_PREDICTION_ARTIFACT_MANIFEST_{DATE_STAMP}.json"
        )
    else:
        path = (
            output_root
            / "B1_hybrid_bytetrack"
            / "manifests"
            / (
                "B1_HYBRID_BYTETRACK_PREDICTION_ARTIFACT_MANIFEST_"
                f"{DATE_STAMP}.json"
            )
        )
    return load_json(path)


def authority_payload(
    arm: str,
    manifest: dict[str, Any],
    preflight_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build one small Git-suitable prediction authority."""

    profile = B0_PROFILE if arm == "B0" else B1_PROFILE
    prediction_root = Path(str(manifest["output_root"])) / "predictions"
    return {
        "schema_version": f"tracking.{arm.lower()}.prediction_authority.v1",
        "date": DATE_STAMP,
        "arm": arm,
        "profile": profile,
        "status": "ESTABLISHED",
        "output_root": manifest["output_root"],
        "prediction_root": str(prediction_root),
        "recursive_artifact_inventory_sha256": manifest[
            "recursive_artifact_inventory_sha256"
        ],
        "canonical_prediction_content_sha256": manifest[
            "canonical_prediction_content_sha256"
        ],
        "prediction_hash_contract": manifest["prediction_hash_contract"],
        "per_video_prediction_hashes": {
            row["video_key"]: {
                "file_sha256": row["sha256"],
                "semantic_sha256": row["semantic_sha256"],
                "canonical_row_sha256": row["canonical_row_sha256"],
            }
            for row in manifest["predictions"]
        },
        "source_video_manifest_sha256": preflight_payload[
            "source_video_manifest_sha256"
        ],
        "gt_authority_sha256": preflight_payload["gt_authority_sha256"],
        "full_frame_cache_authority_sha256": (
            FULL_FRAME_CACHE_AUTHORITY_SHA256
        ),
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "detector_semantic_config_sha256": (
            DETECTOR_SEMANTIC_CONFIG_SHA256
        ),
        "profile_config_sha256": manifest["profile_config_sha256"],
        "offline_repair_semantic_sha256": manifest[
            "offline_repair_semantic_sha256"
        ],
        "producer_code_sha": manifest["producer_code_sha"],
        "execution_topology": "SEPARATE_EXACT_PROFILE_EXECUTIONS",
        "video_count": manifest["video_count"],
        "processed_frame_count": manifest["processed_frame_count"],
        "prediction_object_count": manifest["prediction_object_count"],
        "detector_inference_calls": 0,
        "mp4_count": 0,
        "retention_class": RETENTION_CLASS,
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "immutable": True,
        "execution_repeatability": (
            "NOT_RUN_NOT_REQUIRED_BY_FROZEN_POLICY"
        ),
        "authority_limitations": [
            "Prediction authority only; no tracking metric was calculated.",
            (
                "GT Hidden policy is deferred to the later Standard-V2 "
                "evaluation."
            ),
            (
                "B1 final repaired XML is retained; the active profile does "
                "not expose a separate repair ledger."
                if arm == "B1"
                else "B0 is the active raw ByteTrack presentation profile."
            ),
        ],
    }


def finalize(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    output_root: Path,
) -> None:
    """Revalidate both frozen arms and write small authority records."""

    videos, _, producer_sha = verify_phase_inputs(
        source_repo,
        lineage_manifest,
        cache_root,
        output_root,
        "AWAITING_FINALIZE",
    )
    if len(videos) != EXPECTED_VIDEOS:
        raise PredictionGenerationError("final population changed")
    b0_manifest = read_arm_manifest(output_root, "B0")
    b1_manifest = read_arm_manifest(output_root, "B1")
    assert_arm_immutable(
        output_root / "B0_bytetrack_raw",
        b0_manifest["recursive_artifact_inventory_sha256"],
    )
    assert_arm_immutable(
        output_root / "B1_hybrid_bytetrack",
        b1_manifest["recursive_artifact_inventory_sha256"],
    )
    if (
        b0_manifest["full_frame_cache_authority_sha256"]
        != b1_manifest["full_frame_cache_authority_sha256"]
        or b0_manifest["processed_frame_count"] != EXPECTED_TOTAL_FRAMES
        or b1_manifest["processed_frame_count"] != EXPECTED_TOTAL_FRAMES
    ):
        raise PredictionGenerationError("B0/B1 cache authority mismatch")
    preflight_payload = load_json(output_root / "B0_B1_PREFLIGHT.json")
    r0 = verify_r0_authority(source_repo, videos)
    docs_root = REPO / "docs" / "tracking" / "b0_b1_prediction_authority"
    docs_root.mkdir(parents=True, exist_ok=True)
    locked_manifest_source = (
        output_root
        / f"B0_B1_LOCKED_EXECUTION_MANIFEST_{DATE_STAMP}.json"
    )
    shutil.copy2(
        locked_manifest_source,
        docs_root / locked_manifest_source.name,
    )
    b0_authority = authority_payload(
        "B0",
        b0_manifest,
        preflight_payload,
    )
    b1_authority = authority_payload(
        "B1",
        b1_manifest,
        preflight_payload,
    )
    b0_authority_path = (
        docs_root
        / f"B0_BYTETRACK_RAW_PREDICTION_AUTHORITY_{DATE_STAMP}.json"
    )
    b1_authority_path = (
        docs_root
        / f"B1_HYBRID_BYTETRACK_PREDICTION_AUTHORITY_{DATE_STAMP}.json"
    )
    write_json(b0_authority_path, b0_authority)
    write_json(b1_authority_path, b1_authority)
    fairness = {
        "schema_version": "tracking.b0_b1_r0.prediction_fairness.v2",
        "date": DATE_STAMP,
        "status": "PASS",
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_gt_authority": "PASS",
        "common_source_video_authority": "PASS",
        "common_detector_model_authority": "PASS",
        "common_detector_config_authority": "PASS",
        "common_sequence_boundary_authority": "PASS",
        "b0_b1_full_cache_authority_match": "PASS",
        "r0_even_subset_authority_preserved": "PASS",
        "all_arms_identical_detector_row_count_required": False,
        "b0_detector_cadence": "EVERY_FRAME",
        "b1_detector_cadence": "EVERY_FRAME",
        "r0_detector_cadence": "EVERY_2_FRAMES",
        "b1_minus_b0_detector_cadence_matched": True,
        "future_r1_minus_r0_detector_cadence_matched": True,
        "cross_core_comparison_scope": (
            "WHOLE_PIPELINE_EFFECT_INCLUDING_DETECTOR_CADENCE"
        ),
        "pure_association_core_effect_claim_authorized": False,
        "r0_prediction_artifact_sha256": r0[
            "prediction_artifact_sha256"
        ],
        "quality_metrics_calculated": 0,
    }
    fairness_path = (
        docs_root
        / f"B0_B1_R0_PREDICTION_AUTHORITY_FAIRNESS_{DATE_STAMP}.json"
    )
    write_json(fairness_path, fairness)
    decision = {
        "schema_version": "tracking.b0_b1.prediction_generation_decision.v1",
        "date": DATE_STAMP,
        "decision": "PASS_B0_B1_PREDICTIONS_FROZEN",
        "b0_prediction_authority": "ESTABLISHED",
        "b1_prediction_authority": "ESTABLISHED",
        "b0_prediction_artifact_sha256": b0_manifest[
            "canonical_prediction_content_sha256"
        ],
        "b1_prediction_artifact_sha256": b1_manifest[
            "canonical_prediction_content_sha256"
        ],
        "r0_prediction_artifact_sha256": R0_PREDICTION_ARTIFACT_SHA256,
        "b0_authority_sha256": sha256_file(b0_authority_path),
        "b1_authority_sha256": sha256_file(b1_authority_path),
        "fairness_audit_sha256": sha256_file(fairness_path),
        "execution_topology": "SEPARATE_EXACT_PROFILE_EXECUTIONS",
        "producer_code_sha": producer_sha,
        "total_detector_inference_calls": 0,
        "r0_tracker_executions": 0,
        "r1_tracker_executions": 0,
        "standard_v2_metric_runs": 0,
        "legacy_metric_runs": 0,
        "quality_comparisons": 0,
        "hota_values_generated": 0,
        "idf1_values_generated": 0,
        "idsw_values_generated": 0,
        "run_root_mp4_count": 0,
        "unseen_videos_accessed": False,
        "ready_for_b0_b1_r0_standard_v2_reevaluation": True,
        "ready_for_development_2x2_evaluation": False,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
        "blockers": [
            (
                "Standard-V2 B0/B1/R0 quality re-evaluation remains a "
                "separate task."
            ),
            "R1 prediction generation and the 2x2 campaign were not run.",
        ],
    }
    write_json(
        docs_root
        / f"B0_B1_PREDICTION_GENERATION_DECISION_{DATE_STAMP}.json",
        decision,
    )
    write_json(
        output_root
        / f"B0_B1_PREDICTION_GENERATION_DECISION_{DATE_STAMP}.json",
        decision,
    )
    update_state(
        output_root,
        status="PASS",
        phase="COMPLETE",
        b0_completed=EXPECTED_VIDEOS,
        b1_completed=EXPECTED_VIDEOS,
    )


def parse_args() -> argparse.Namespace:
    """Parse the deliberately narrow phase CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("preflight", "run-b0", "run-b1", "finalize-docs"),
    )
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Execute one authorized phase, recording failures without recovery."""

    args = parse_args()
    source_repo = args.source_repo.resolve()
    lineage_manifest = args.lineage_manifest.resolve()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    try:
        if args.phase == "preflight":
            preflight(
                source_repo,
                lineage_manifest,
                cache_root,
                output_root,
            )
        elif args.phase == "run-b0":
            run_arm(
                source_repo,
                lineage_manifest,
                cache_root,
                output_root,
                B0_PROFILE,
            )
        elif args.phase == "run-b1":
            run_arm(
                source_repo,
                lineage_manifest,
                cache_root,
                output_root,
                B1_PROFILE,
            )
        else:
            finalize(
                source_repo,
                lineage_manifest,
                cache_root,
                output_root,
            )
    except Exception as exc:
        if output_root.exists() and (
            output_root / "B0_B1_PREFLIGHT.json"
        ).is_file():
            try:
                state = load_json(output_root / STATE_FILE)
                update_state(
                    output_root,
                    status="FAIL",
                    phase=f"FAILED_{args.phase.upper().replace('-', '_')}",
                    b0_completed=int(state.get("b0_videos_completed", 0)),
                    b1_completed=int(state.get("b1_videos_completed", 0)),
                    video_key=state.get("current_video_key"),
                    failure=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
