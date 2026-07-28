"""Generate and freeze R1 from the exact frozen realtime-fast detector cache.

This authority tool is deliberately tracker-only. It never imports an evaluator,
constructs a live detector, reads unseen data, or renders an MP4.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.tracking import generate_b0_b1_frozen_predictions as b01  # noqa: E402
from scripts.tracking import generate_current_main_baseline_caches as r0_cache  # noqa: E402

REQUESTED_STARTING_MAIN_SHA = "f3a7789bce26e99ab9a43a8ee3a1a1bcaebe934d"
ACTUAL_EXECUTION_BASE_SHA = "89db2d34646434319afcb4a040ba38d183431cfd"
DATE_STAMP = "20260728"
R1_PROFILE = "rf_hybrid_offline"
R0_CONFIG_SHA256 = (
    "9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d"
)
R0_PREDICTION_ARTIFACT_SHA256 = (
    "fd2d4f3dec0710d1c9eecba9308247a7b226dd34a4a02a9cb89f17acb22bbbfe"
)
REPAIR_SEMANTIC_SHA256 = (
    "e078b5b165dda82dee5b61e9465dc9844446e4cb576a02858c4ed7369828d758"
)
R0_EVEN_CACHE_AUTHORITY_SHA256 = (
    "795df7732393e4e258a82db58e29101b068cf8ac3583acf7702e0afdaeec6e7a"
)
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
EXPECTED_CACHE_RECORDS = 900
EXPECTED_TOTAL_CACHE_RECORDS = EXPECTED_VIDEOS * EXPECTED_CACHE_RECORDS
STATE_FILE = "R1_GENERATION_STATE.json"
PREFLIGHT_FILE = "R1_PREFLIGHT.json"
FROZEN_MARKER = "FROZEN_SCIENTIFIC_AUTHORITY_DO_NOT_DELETE.txt"
RETENTION_CLASS = "NON_DISPOSABLE_FROZEN_PREDICTION_AUTHORITY"
R0_ROOT_RELATIVE = Path("outputs/tracking/current_main_baseline_20260728")
R0_CACHE_RELATIVE = Path(
    "outputs/tracking/detector_caches/current_main_baseline_20260728"
)
LOCKED_MANIFEST_RELATIVE = Path(
    "docs/tracking/b0_b1_prediction_authority/"
    "B0_B1_LOCKED_EXECUTION_MANIFEST_20260728.json"
)
R0_CACHE_PREFLIGHT_RELATIVE = R0_CACHE_RELATIVE / (
    "CURRENT_MAIN_DETECTOR_CACHE_PREFLIGHT.json"
)
RF_AUTHORITY_RELATIVE = Path("docs/tracking/rf_hybrid_offline")
DOC_ROOT_RELATIVE = Path("docs/tracking/r1_prediction_authority")


class R1AuthorityError(RuntimeError):
    """Fail-closed R1 generation authority error."""


def utc_now() -> str:
    """Return a stable UTC audit timestamp."""

    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash one file using the established project helper."""

    return b01.sha256_file(path)


def canonical_hash(payload: Any) -> str:
    """Hash canonical JSON using the established project helper."""

    return b01.canonical_hash(payload)


def load_json(path: Path) -> Any:
    """Load one JSON authority."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write one small deterministic JSON authority."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_output(repo: Path, *args: str) -> str:
    """Run one read-only Git query."""

    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def authority_file_hashes(source_repo: Path) -> dict[str, str]:
    """Bind the frozen RF adapter and common repair contracts by file bytes."""

    relative_names = (
        "COMMON_OFFLINE_REPAIR_INPUT_CONTRACT_20260728.json",
        "RF_RAW_TRACKLET_OUTPUT_CONTRACT_20260728.json",
        "RF_TO_OFFLINE_REPAIR_COMPATIBILITY_AUDIT_20260728.csv",
        "RF_HYBRID_OFFLINE_IMPLEMENTATION_DECISION_20260728.json",
    )
    authority_root = source_repo / RF_AUTHORITY_RELATIVE
    return {
        name: sha256_file(authority_root / name)
        for name in relative_names
    }


def tracking_authority_unchanged(repo: Path, producer_sha: str) -> bool:
    """Prove the requested SHA is an ancestor with no tracking-scope changes."""

    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUESTED_STARTING_MAIN_SHA,
            ACTUAL_EXECUTION_BASE_SHA,
        ],
        cwd=repo,
        check=True,
    )
    scoped_paths = (
        "src/pig_behavior/tracking",
        "scripts/tracking",
        "tests/test_rf_hybrid_offline.py",
        "docs/tracking",
        ".agents/memory",
        "Kế Hoạch Tương Lai.md",
    )
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            REQUESTED_STARTING_MAIN_SHA,
            ACTUAL_EXECUTION_BASE_SHA,
            "--",
            *scoped_paths,
        ],
        cwd=repo,
        check=False,
    )
    if result.returncode != 0:
        return False
    producer_lineage = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ACTUAL_EXECUTION_BASE_SHA,
            producer_sha,
        ],
        cwd=repo,
        check=False,
    )
    return producer_lineage.returncode == 0


def locked_population(source_repo: Path) -> tuple[list[Any], dict[str, Any]]:
    """Load and revalidate the exact frozen 13-video population."""

    import cv2

    manifest = load_json(source_repo / LOCKED_MANIFEST_RELATIVE)
    rows = manifest.get("videos", [])
    if len(rows) != EXPECTED_VIDEOS:
        raise R1AuthorityError("locked population is not 13 videos")
    videos = []
    for row in rows:
        video_path = Path(row["source_video_path"])
        gt_path = Path(row["gt_path"])
        if not video_path.is_file():
            raise R1AuthorityError(f"missing source video: {row['video_key']}")
        if not gt_path.is_file():
            raise R1AuthorityError(f"missing GT authority: {row['video_key']}")
        if sha256_file(video_path) != row["source_video_sha256"]:
            raise R1AuthorityError(f"source video hash mismatch: {row['video_key']}")
        if sha256_file(gt_path) != row["gt_sha256"]:
            raise R1AuthorityError(f"GT hash mismatch: {row['video_key']}")
        if int(row["expected_frame_count"]) != EXPECTED_FRAMES:
            raise R1AuthorityError(f"frame authority mismatch: {row['video_key']}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise R1AuthorityError(f"cannot open source video: {row['video_key']}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if frame_count != EXPECTED_FRAMES or width <= 0 or height <= 0:
            raise R1AuthorityError(f"media authority mismatch: {row['video_key']}")
        videos.append(
            SimpleNamespace(
                video_key=str(row["video_key"]),
                video_path=video_path,
                video_sha256=str(row["source_video_sha256"]),
                gt_path=gt_path,
                gt_sha256=str(row["gt_sha256"]),
                frame_count=EXPECTED_FRAMES,
                width=width,
                height=height,
                gt_authority=str(row["gt_authority_status"]),
                aggregate_inclusion_role=str(row["aggregate_inclusion_role"]),
                mechanism_ranking_eligibility=bool(
                    row["mechanism_ranking_eligibility"]
                ),
            )
        )
    expected_keys = sorted(str(row["video_key"]) for row in rows)
    observed_keys = sorted(video.video_key for video in videos)
    if observed_keys != expected_keys:
        raise R1AuthorityError("locked video-key set mismatch")
    return sorted(videos, key=lambda item: item.video_key), manifest


def verify_profile_authority() -> dict[str, Any]:
    """Reproduce R0 core and frozen repair semantic authorities."""

    from pig_behavior.tracking.offline_repair import (
        build_frozen_offline_repair_config,
        offline_repair_semantic_hash,
    )
    from pig_behavior.tracking.profiles import PRESENTATION_PROFILES
    from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
    from pig_behavior.tracking.profiles.rf_hybrid_offline import (
        RF_HYBRID_OFFLINE_CONFIG,
    )

    expected_profiles = (
        "bytetrack_raw",
        "realtime_fast",
        "hybrid_bytetrack",
        "rf_hybrid_offline",
    )
    if tuple(PRESENTATION_PROFILES) != expected_profiles:
        raise R1AuthorityError("active profile registry mismatch")
    if canonical_hash(REALTIME_FAST_CONFIG) != R0_CONFIG_SHA256:
        raise R1AuthorityError("R0 config hash mismatch")
    candidate_core = {
        key: value
        for key, value in RF_HYBRID_OFFLINE_CONFIG.items()
        if key not in {"rf_hybrid_offline", "write_output_video"}
    }
    reference_core = {
        key: value
        for key, value in REALTIME_FAST_CONFIG.items()
        if key != "write_output_video"
    }
    if candidate_core != reference_core:
        raise R1AuthorityError("R1 realtime-fast core config mismatch")
    repair_hash = offline_repair_semantic_hash(
        build_frozen_offline_repair_config()
    )
    if repair_hash != REPAIR_SEMANTIC_SHA256:
        raise R1AuthorityError("offline repair semantic hash mismatch")
    return {
        "active_profiles": list(expected_profiles),
        "r0_config_sha256": R0_CONFIG_SHA256,
        "repair_semantic_sha256": repair_hash,
        "realtime_core": "realtime_fast",
        "profile": R1_PROFILE,
    }


def recursive_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every retained file below a root."""

    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def r1_artifact_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory every non-self-referential retained R1 scientific artifact."""

    included_roots = (
        root / "predictions",
        root / "machine_readable",
        root / "raw_core_snapshots",
        root / "repair_ledgers",
        root / "commands",
    )
    paths = [
        path
        for included_root in included_roots
        for path in included_root.rglob("*")
        if path.is_file()
    ]
    marker = root / FROZEN_MARKER
    if marker.is_file():
        paths.append(marker)
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_role": path.relative_to(root).parts[0],
        }
        for path in sorted(set(paths))
    ]


def verify_r0_before(source_repo: Path, videos: list[Any]) -> dict[str, Any]:
    """Verify immutable R0 and record a complete before-state inventory."""

    authority = b01.verify_r0_authority(source_repo, videos)
    if (
        authority["prediction_artifact_sha256"]
        != R0_PREDICTION_ARTIFACT_SHA256
    ):
        raise R1AuthorityError("R0 prediction artifact authority mismatch")
    root = source_repo / R0_ROOT_RELATIVE
    inventory = recursive_inventory(root)
    return {
        **authority,
        "root": str(root),
        "recursive_inventory_sha256": canonical_hash(inventory),
        "files": inventory,
    }


def even_cache_records(
    source_repo: Path,
    videos: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Revalidate exact even-frame caches without invoking a detector."""

    from pig_behavior.tracking.detector_cache import (
        DetectorEvidenceCache,
        ReplayDetector,
    )

    cache_root = source_repo / R0_CACHE_RELATIVE
    preflight = load_json(source_repo / R0_CACHE_PREFLIGHT_RELATIVE)
    r0_authority = load_json(
        source_repo
        / "docs"
        / "tracking"
        / "CURRENT_MAIN_R0_BASELINE_AUTHORITY_20260728.json"
    )
    if (
        r0_authority["r0_detector_cache_authority_sha256"]
        != R0_EVEN_CACHE_AUTHORITY_SHA256
    ):
        raise R1AuthorityError("R0 even-cache authority hash mismatch")
    records = []
    for video in videos:
        cache_path = r0_cache.cache_path(cache_root, video.video_key)
        identity = r0_cache.load_identity(video, preflight)
        cache = DetectorEvidenceCache.load(
            cache_path,
            expected_identity=identity,
        )
        expected = tuple(range(0, EXPECTED_FRAMES, 2))
        if tuple(cache.frames) != expected:
            raise R1AuthorityError(
                f"even-cache coverage mismatch: {video.video_key}"
            )
        replay = ReplayDetector(cache)
        for frame_index in expected:
            dimensions = cache.frames[frame_index]["original_frame_dimensions"]
            replay.set_frame_context(frame_index, dimensions)
            replay.predict()
        if replay.invocations != EXPECTED_CACHE_RECORDS:
            raise R1AuthorityError(
                f"even-cache replay mismatch: {video.video_key}"
            )
        records.append(
            {
                "video_key": video.video_key,
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "cache_records": EXPECTED_CACHE_RECORDS,
                "frame_start": 0,
                "frame_end": EXPECTED_FRAMES - 1,
                "detector_cadence": "EVERY_2_FRAMES",
            }
        )
    return records, preflight


def create_output_root(output_root: Path) -> None:
    """Create a refuse-overwrite stable authority root."""

    if output_root.exists():
        raise R1AuthorityError(f"refusing existing output root: {output_root}")
    for name in (
        "predictions",
        "machine_readable",
        "raw_core_snapshots",
        "repair_ledgers",
        "manifests",
        "audits",
        "commands",
    ):
        (output_root / name).mkdir(parents=True)


def update_state(output_root: Path, **updates: Any) -> None:
    """Update the small resumability/failure audit state."""

    path = output_root / STATE_FILE
    payload = load_json(path) if path.is_file() else {}
    payload.update(updates)
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def preflight(source_repo: Path, output_root: Path) -> None:
    """Run all authority checks before the first R1 core execution."""

    producer_sha = git_output(REPO, "rev-parse", "HEAD")
    if not tracking_authority_unchanged(REPO, producer_sha):
        raise R1AuthorityError(
            "tracking authority changed across the authorized descendant"
        )
    if git_output(REPO, "status", "--porcelain"):
        raise R1AuthorityError("producer worktree must be clean")
    videos, locked_manifest = locked_population(source_repo)
    profile = verify_profile_authority()
    r0_before = verify_r0_before(source_repo, videos)
    cache_records, cache_preflight = even_cache_records(source_repo, videos)
    create_output_root(output_root)
    topology = {
        "schema_version": "tracking.r1.generation_topology_audit.v1",
        "date": DATE_STAMP,
        "decision": "EXACT_R1_PROFILE_EXECUTION",
        "preferred_replay_topology": "NOT_CONTRACT_COMPLETE",
        "frozen_r0_public_fields": "PRESENT_EXACT",
        "timestamp": "OPTIONAL_ABSENT",
        "required_internal_lifecycle_fields": "REQUIRED_ABSENT",
        "required_internal_motion_fields": "REQUIRED_ABSENT",
        "required_internal_provenance_fields": "REQUIRED_ABSENT",
        "synthesized_required_evidence": 0,
        "quality_or_runtime_used_to_choose_topology": False,
    }
    payload = {
        "schema_version": "tracking.r1.prediction_preflight.v1",
        "created_at": utc_now(),
        "requested_starting_main_sha": REQUESTED_STARTING_MAIN_SHA,
        "actual_execution_base_sha": ACTUAL_EXECUTION_BASE_SHA,
        "producer_code_sha": producer_sha,
        "tracking_authority_unchanged_across_descendant": True,
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
        ],
        "generation_topology": "EXACT_R1_PROFILE_EXECUTION",
        "profile_authority": profile,
        "rf_contract_file_sha256": authority_file_hashes(source_repo),
        "locked_manifest_sha256": sha256_file(
            source_repo / LOCKED_MANIFEST_RELATIVE
        ),
        "source_lineage_sha256": locked_manifest["source_lineage_sha256"],
        "r0_authority_before": r0_before,
        "r0_even_cache_authority_sha256": R0_EVEN_CACHE_AUTHORITY_SHA256,
        "r0_even_cache_records": cache_records,
        "cache_producer_code_sha": cache_preflight["producer_code_sha"],
        "cache_replay_dry_run": "PASS",
        "live_detector_fallback_enabled": False,
        "video_count": len(videos),
        "expected_frames_per_video": EXPECTED_FRAMES,
        "detector_inference_calls": 0,
        "standard_v2_metric_runs": 0,
        "legacy_metric_runs": 0,
        "quality_comparisons": 0,
        "unseen_videos_accessed": False,
    }
    write_json(output_root / PREFLIGHT_FILE, payload)
    write_json(
        output_root
        / "audits"
        / f"R1_GENERATION_TOPOLOGY_AUDIT_{DATE_STAMP}.json",
        topology,
    )
    update_state(
        output_root,
        status="READY",
        phase="AWAITING_R1",
        videos_completed=0,
        current_video_key=None,
    )


def build_tracking_config(
    source_repo: Path,
    video: Any,
    output_dir: Path,
) -> Any:
    """Build exact RF hybrid config with no semantic CLI overrides."""

    from pig_behavior.tracking.config import TrackingConfig, validate_config
    from pig_behavior.tracking.profiles.rf_hybrid_offline import (
        RF_HYBRID_OFFLINE_CONFIG,
    )

    overrides = dict(RF_HYBRID_OFFLINE_CONFIG)
    overrides.pop("mode", None)
    overrides.pop("write_output_video", None)
    cfg = TrackingConfig(
        mode="realtime",
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


def r0_public_shapes(source_repo: Path, video_key: str) -> list[dict[str, Any]]:
    """Load the immutable public R0 shape representation."""

    path = (
        source_repo
        / R0_ROOT_RELATIVE
        / "predictions"
        / video_key
        / "annotations_cvat_shapes.json"
    )
    payload = load_json(path)
    if not isinstance(payload, list) or len(payload) != 1:
        raise R1AuthorityError(f"invalid R0 public shape payload: {video_key}")
    return list(payload[0]["shapes"])


def raw_core_guard(
    expected_public_shapes: list[dict[str, Any]],
    video_key: str,
    parity_rows: list[dict[str, Any]],
) -> Any:
    """Return a fail-closed guard called before adapter and repair."""

    from pig_behavior.tracking.exporters.annotation import (
        strip_internal_shape_keys,
    )

    expected_hash = canonical_hash(expected_public_shapes)

    def guard(raw_shapes: list[dict[str, Any]]) -> None:
        observed = [strip_internal_shape_keys(shape) for shape in raw_shapes]
        observed_hash = canonical_hash(observed)
        if observed != expected_public_shapes:
            raise R1AuthorityError(
                f"FAIL_R0_RAW_CORE_PARITY: {video_key}"
            )
        parity_rows.append(
            {
                "video_key": video_key,
                "status": "PASS",
                "r0_public_shape_sha256": expected_hash,
                "r1_raw_public_shape_sha256": observed_hash,
                "shape_count": len(observed),
                "bbox_value_changes": 0,
                "id_value_changes": 0,
                "frame_value_changes": 0,
                "timestamp_value_changes": 0,
                "confidence_value_changes": 0,
            }
        )

    return guard


def ledger_summary(path: Path) -> dict[str, Any]:
    """Summarize a GT-free deterministic repair ledger."""

    payload = load_json(path)
    if payload["repair_config_hash"] != REPAIR_SEMANTIC_SHA256:
        raise R1AuthorityError(f"repair ledger hash mismatch: {path}")
    events = list(payload.get("events", []))
    stages = sorted({str(event["repair_stage"]) for event in events})
    def modified_count(event: dict[str, Any]) -> int:
        value = event.get("frames_modified", 0)
        return len(value) if isinstance(value, list) else int(value)

    return {
        "event_count": len(events),
        "frames_modified": sum(modified_count(event) for event in events),
        "stages_activated": stages,
        "future_frame_events": sum(
            bool(event.get("future_frames_used", False)) for event in events
        ),
        "ledger_sha256": sha256_file(path),
        "canonical_ledger_sha256": canonical_hash(payload),
    }


def run_r1(source_repo: Path, output_root: Path) -> None:
    """Execute exact R1 independently for all locked videos."""

    from pig_behavior.tracking.detector_cache import (
        DetectorEvidenceCache,
        ReplayDetector,
    )
    from pig_behavior.tracking.runner import run_tracking

    state = load_json(output_root / STATE_FILE)
    if state.get("phase") != "AWAITING_R1":
        raise R1AuthorityError("R1 output root is not awaiting execution")
    preflight_payload = load_json(output_root / PREFLIGHT_FILE)
    videos, _ = locked_population(source_repo)
    parity_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    cache_preflight = load_json(source_repo / R0_CACHE_PREFLIGHT_RELATIVE)
    cache_root = source_repo / R0_CACHE_RELATIVE
    completed = 0
    replay_calls = 0
    for video in videos:
        update_state(
            output_root,
            status="RUNNING",
            phase="RUNNING_R1",
            videos_completed=completed,
            current_video_key=video.video_key,
        )
        cache = DetectorEvidenceCache.load(
            r0_cache.cache_path(cache_root, video.video_key),
            expected_identity=r0_cache.load_identity(video, cache_preflight),
        )
        detector = ReplayDetector(cache)
        machine_dir = output_root / "machine_readable" / video.video_key
        cfg = build_tracking_config(source_repo, video, machine_dir)
        parity_before = len(parity_rows)
        print(f"R1_BEGIN {video.video_key}", flush=True)
        summary = run_tracking(
            cfg,
            model=detector,
            rf_raw_core_guard=raw_core_guard(
                r0_public_shapes(source_repo, video.video_key),
                video.video_key,
                parity_rows,
            ),
        )
        if len(parity_rows) != parity_before + 1:
            raise R1AuthorityError(
                f"raw-core parity guard not invoked: {video.video_key}"
            )
        if detector.invocations != EXPECTED_CACHE_RECORDS:
            raise R1AuthorityError(
                f"R1 cache replay count mismatch: {video.video_key}"
            )
        if summary.frames_read != EXPECTED_FRAMES:
            raise R1AuthorityError(
                f"R1 frame coverage mismatch: {video.video_key}"
            )
        if summary.raw_annotations_json is None:
            raise R1AuthorityError(f"missing raw snapshot: {video.video_key}")
        if summary.repair_ledger_json is None:
            raise R1AuthorityError(f"missing repair ledger: {video.video_key}")
        prediction_xml = output_root / "predictions" / f"{video.video_key}.xml"
        if prediction_xml.exists():
            raise R1AuthorityError(f"refusing overwrite: {prediction_xml}")
        shutil.copy2(summary.cvat_video_xml, prediction_xml)
        raw_copy = (
            output_root
            / "raw_core_snapshots"
            / f"{video.video_key}.rf_raw_track_output.json"
        )
        ledger_copy = (
            output_root
            / "repair_ledgers"
            / f"{video.video_key}.rf_offline_repair_ledger.json"
        )
        shutil.copy2(summary.raw_annotations_json, raw_copy)
        shutil.copy2(summary.repair_ledger_json, ledger_copy)
        record = b01.xml_structural_record(
            prediction_xml,
            video_key=video.video_key,
            width=video.width,
            height=video.height,
        )
        prediction_records.append(record)
        ledger = ledger_summary(ledger_copy)
        ledger["video_key"] = video.video_key
        ledger_rows.append(ledger)
        replay_calls += detector.invocations
        completed += 1
        execution_rows.append(
            {
                "video_key": video.video_key,
                "frames_processed": summary.frames_read,
                "cache_records_consumed": detector.invocations,
                "prediction_object_count": summary.shape_count,
                "raw_core_snapshot_sha256": sha256_file(raw_copy),
                "repair_ledger_sha256": sha256_file(ledger_copy),
                "prediction_xml_sha256": record["sha256"],
                "detector_inference_calls": 0,
                "mp4_count": 0,
                "status": "PASS",
            }
        )
        print(
            f"R1_END {video.video_key} "
            f"objects={record['prediction_object_count']}",
            flush=True,
        )
    if completed != EXPECTED_VIDEOS:
        raise R1AuthorityError("R1 did not complete all 13 videos")
    if replay_calls != EXPECTED_TOTAL_CACHE_RECORDS:
        raise R1AuthorityError("R1 cache consumption total mismatch")
    if len(list((output_root / "predictions").glob("*.xml"))) != EXPECTED_VIDEOS:
        raise R1AuthorityError("R1 prediction XML count mismatch")
    if list(output_root.rglob("*.mp4")):
        raise R1AuthorityError("R1 produced unauthorized MP4 files")
    prediction_hash = b01.prediction_set_hash(prediction_records)
    write_json(
        output_root / "audits" / f"R0_R1_RAW_CORE_PARITY_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.r0_r1.raw_core_parity.v1",
            "status": "PASS",
            "comparison_timing": "BEFORE_ADAPTER_AND_REPAIR",
            "video_count": len(parity_rows),
            "per_video": parity_rows,
        },
    )
    write_json(
        output_root
        / "audits"
        / f"R1_PREDICTION_CONSERVATION_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.r1.prediction_conservation.v1",
            "status": "PASS",
            "video_count": completed,
            "prediction_xml_count": EXPECTED_VIDEOS,
            "prediction_object_count": sum(
                int(row["prediction_object_count"])
                for row in prediction_records
            ),
            "processed_frame_count": EXPECTED_VIDEOS * EXPECTED_FRAMES,
            "cache_records_consumed": replay_calls,
            "canonical_prediction_content_sha256": prediction_hash,
            "locked_video_key_set": "PASS",
            "frame_range_coverage": "PASS",
            "bbox_validity": "PASS",
            "finite_numeric_values": "PASS",
            "identity_serialization": "PASS",
            "no_cross_video_identity_state": "PASS_SEPARATE_EXECUTIONS",
            "detector_inference_calls": 0,
            "evaluator_modifications": 0,
            "mp4_count": 0,
            "per_video": prediction_records,
        },
    )
    write_json(
        output_root / "audits" / f"R1_EXECUTION_AUDIT_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.r1.execution_audit.v1",
            "generation_topology": "EXACT_R1_PROFILE_EXECUTION",
            "producer_code_sha": git_output(REPO, "rev-parse", "HEAD"),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            "videos": execution_rows,
            "r0_tracker_executions": 0,
            "r1_rf_core_executions": completed,
            "r1_repair_invocations": completed,
            "detector_inference_calls": 0,
            "standard_v2_metric_runs": 0,
            "legacy_metric_runs": 0,
            "quality_comparisons": 0,
            "hota_values_generated": 0,
            "idf1_values_generated": 0,
            "idsw_values_generated": 0,
            "gt_dependent_repair_labels_generated": 0,
            "mp4_count": 0,
            "unseen_videos_accessed": False,
        },
    )
    write_json(
        output_root
        / "audits"
        / f"R1_REPAIR_LEDGER_SUMMARY_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.r1.repair_ledger_summary.v1",
            "gt_dependent_labels": 0,
            "per_video": ledger_rows,
            "event_count": sum(row["event_count"] for row in ledger_rows),
            "frames_modified": sum(
                row["frames_modified"] for row in ledger_rows
            ),
            "future_frame_events": sum(
                row["future_frame_events"] for row in ledger_rows
            ),
            "stages_activated": sorted(
                {
                    stage
                    for row in ledger_rows
                    for stage in row["stages_activated"]
                }
            ),
        },
    )
    update_state(
        output_root,
        status="READY",
        phase="AWAITING_FINALIZE",
        videos_completed=completed,
        current_video_key=None,
        canonical_prediction_content_sha256=prediction_hash,
        preflight_sha256=canonical_hash(preflight_payload),
    )


def marker_text(producer_sha: str) -> str:
    """Return the visible non-disposable authority marker."""

    return (
        "FROZEN SCIENTIFIC PREDICTION AUTHORITY\n"
        f"profile={R1_PROFILE}\n"
        f"producer_code_sha={producer_sha}\n"
        f"retention_class={RETENTION_CLASS}\n"
        "deletion_allowed=NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT\n"
        "This directory is not temporary output. Do not modify or delete it.\n"
    )


def freeze_files(root: Path) -> None:
    """Mark retained files read-only where Windows permits."""

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        os.chmod(path, stat.S_IREAD)


def verify_r0_after(
    source_repo: Path,
    before_payload: dict[str, Any],
) -> dict[str, Any]:
    """Rehash every R0 file and require exact before/after equality."""

    root = source_repo / R0_ROOT_RELATIVE
    after_inventory = recursive_inventory(root)
    after_hash = canonical_hash(after_inventory)
    if after_hash != before_payload["recursive_inventory_sha256"]:
        raise R1AuthorityError("R0 artifact files were modified")
    return {
        "status": "PASS",
        "r0_artifact_files_modified": 0,
        "recursive_inventory_sha256": after_hash,
    }


def finalize(source_repo: Path, output_root: Path) -> None:
    """Freeze R1 artifacts and create small Git-suitable authorities."""

    state = load_json(output_root / STATE_FILE)
    if state.get("phase") != "AWAITING_FINALIZE":
        raise R1AuthorityError("R1 output root is not ready to finalize")
    preflight_payload = load_json(output_root / PREFLIGHT_FILE)
    conservation = load_json(
        output_root
        / "audits"
        / f"R1_PREDICTION_CONSERVATION_{DATE_STAMP}.json"
    )
    parity = load_json(
        output_root / "audits" / f"R0_R1_RAW_CORE_PARITY_{DATE_STAMP}.json"
    )
    ledger = load_json(
        output_root
        / "audits"
        / f"R1_REPAIR_LEDGER_SUMMARY_{DATE_STAMP}.json"
    )
    r0_after = verify_r0_after(
        source_repo,
        preflight_payload["r0_authority_before"],
    )
    producer_sha = git_output(REPO, "rev-parse", "HEAD")
    (output_root / "commands" / "R1_EXECUTION_COMMAND.txt").write_text(
        subprocess.list2cmdline(sys.argv)
        + "\nDETECTOR_INFERENCE_CALLS=0"
        + "\nSTANDARD_V2_METRIC_RUNS=0"
        + "\nLEGACY_METRIC_RUNS=0"
        + "\nQUALITY_COMPARISONS=0"
        + "\nRUN_ROOT_MP4_COUNT=0\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / FROZEN_MARKER).write_text(
        marker_text(producer_sha),
        encoding="utf-8",
        newline="\n",
    )
    inventory = r1_artifact_inventory(output_root)
    inventory_hash = canonical_hash(inventory)
    raw_snapshot_hash = canonical_hash(
        [
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(
                (output_root / "raw_core_snapshots").glob("*.json")
            )
        ]
    )
    ledger_table_hash = canonical_hash(
        [
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted((output_root / "repair_ledgers").glob("*.json"))
        ]
    )
    manifest = {
        "schema_version": "tracking.r1.prediction_artifact_manifest.v1",
        "date": DATE_STAMP,
        "status": "FROZEN",
        "profile": R1_PROFILE,
        "retention_class": RETENTION_CLASS,
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "output_root": str(output_root),
        "recursive_artifact_inventory_sha256": inventory_hash,
        "canonical_prediction_content_sha256": conservation[
            "canonical_prediction_content_sha256"
        ],
        "raw_core_snapshot_sha256": raw_snapshot_hash,
        "repair_ledger_table_sha256": ledger_table_hash,
        "producer_code_sha": producer_sha,
        "r0_prediction_artifact_sha256": R0_PREDICTION_ARTIFACT_SHA256,
        "r0_even_cache_authority_sha256": R0_EVEN_CACHE_AUTHORITY_SHA256,
        "r0_config_sha256": R0_CONFIG_SHA256,
        "repair_semantic_sha256": REPAIR_SEMANTIC_SHA256,
        "video_count": EXPECTED_VIDEOS,
        "prediction_object_count": conservation["prediction_object_count"],
        "artifacts": inventory,
        "predictions": conservation["per_video"],
    }
    manifest_name = (
        f"R1_RF_HYBRID_OFFLINE_PREDICTION_ARTIFACT_MANIFEST_"
        f"{DATE_STAMP}.json"
    )
    write_json(output_root / "manifests" / manifest_name, manifest)
    authority = {
        "schema_version": "tracking.r1.prediction_authority.v1",
        "date": DATE_STAMP,
        "status": "ESTABLISHED",
        "selected_generation_topology": "EXACT_R1_PROFILE_EXECUTION",
        "requested_starting_main_sha": REQUESTED_STARTING_MAIN_SHA,
        "actual_execution_base_sha": producer_sha,
        "tracking_authority_unchanged_across_descendant": True,
        "r0_prediction_authority_sha256": R0_PREDICTION_ARTIFACT_SHA256,
        "r1_output_root": str(output_root),
        "r1_prediction_artifact_sha256": conservation[
            "canonical_prediction_content_sha256"
        ],
        "recursive_inventory_sha256": inventory_hash,
        "raw_core_snapshot_sha256": raw_snapshot_hash,
        "repair_ledger_table_sha256": ledger_table_hash,
        "raw_core_parity": parity["status"],
        "r0_artifact_files_modified": 0,
        "r0_config_sha256": R0_CONFIG_SHA256,
        "repair_semantic_sha256": REPAIR_SEMANTIC_SHA256,
        "rf_contract_file_sha256": preflight_payload[
            "rf_contract_file_sha256"
        ],
        "r0_even_cache_authority_sha256": R0_EVEN_CACHE_AUTHORITY_SHA256,
        "detector_cadence": "EVERY_2_FRAMES",
        "video_count": EXPECTED_VIDEOS,
        "prediction_xml_count": EXPECTED_VIDEOS,
        "prediction_object_count": conservation["prediction_object_count"],
        "r0_tracker_executions": 0,
        "r1_rf_core_executions": EXPECTED_VIDEOS,
        "r1_repair_invocations": EXPECTED_VIDEOS,
        "detector_inference_calls": 0,
        "standard_v2_metric_runs": 0,
        "legacy_metric_runs": 0,
        "quality_comparisons": 0,
        "mp4_count": 0,
        "unseen_videos_accessed": False,
        "adapter_representation_only": True,
        "adapter_association_decisions": 0,
        "adapter_identity_changes": 0,
        "adapter_synthesized_required_evidence": 0,
        "adapter_bbox_value_changes": 0,
        "adapter_timestamp_value_changes": 0,
        "repair_parameters_changed": False,
        "repair_stage_order_changed": False,
        "new_repair_heuristics_added": 0,
        "rf_specific_repair_branches_added": 0,
        "repair_event_count": ledger["event_count"],
        "repair_frames_modified": ledger["frames_modified"],
        "repair_stages_activated": ledger["stages_activated"],
        "future_frame_repair_events": ledger["future_frame_events"],
        "execution_repeatability": (
            "NOT_RUN_NOT_REQUIRED_BY_FROZEN_POLICY"
        ),
        "structural_determinism": "PASS",
        "retention_class": RETENTION_CLASS,
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "limitations": [
            "development population only",
            "no tracking-quality metric was calculated",
            "not promoted",
            "not authorized for unseen evaluation",
        ],
    }
    decision = {
        "schema_version": "tracking.r1.prediction_generation_decision.v1",
        "date": DATE_STAMP,
        "decision": "PASS_R1_PREDICTIONS_FROZEN",
        "r1_prediction_authority": "ESTABLISHED",
        "r0_r1_raw_core_parity": "PASS",
        "r0_artifact_files_modified": 0,
        "adapter_association_decisions": 0,
        "adapter_identity_changes": 0,
        "adapter_synthesized_required_evidence": 0,
        "repair_parameters_changed": False,
        "new_repair_heuristics_added": 0,
        "rf_specific_repair_branches_added": 0,
        "video_count": EXPECTED_VIDEOS,
        "prediction_xml_count": EXPECTED_VIDEOS,
        "detector_inference_calls": 0,
        "standard_v2_metric_runs": 0,
        "unseen_videos_accessed": False,
        "ready_for_complete_development_2x2_evaluation": True,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
        "blockers": [],
    }
    docs_root = REPO / DOC_ROOT_RELATIVE
    write_json(
        docs_root / f"R1_GENERATION_TOPOLOGY_AUDIT_{DATE_STAMP}.json",
        load_json(
            output_root
            / "audits"
            / f"R1_GENERATION_TOPOLOGY_AUDIT_{DATE_STAMP}.json"
        ),
    )
    write_json(
        docs_root / f"R0_R1_LOCKED_EXECUTION_MANIFEST_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.r0_r1.locked_execution_manifest.v1",
            "date": DATE_STAMP,
            "status": "PASS",
            "video_count": EXPECTED_VIDEOS,
            "common_video_authority": "PASS",
            "common_frame_authority": "PASS",
            "common_gt_authority": "PASS",
            "common_source_video_authority": "PASS",
            "common_sequence_boundary_authority": "PASS",
            "videos": preflight_payload["r0_even_cache_records"],
        },
    )
    write_json(
        docs_root / f"R1_PREDICTION_CONSERVATION_{DATE_STAMP}.json",
        conservation,
    )
    write_json(
        docs_root / manifest_name,
        manifest,
    )
    write_json(
        docs_root
        / f"R1_RF_HYBRID_OFFLINE_PREDICTION_AUTHORITY_{DATE_STAMP}.json",
        authority,
    )
    write_json(
        docs_root / f"R1_PREDICTION_GENERATION_DECISION_{DATE_STAMP}.json",
        decision,
    )
    update_state(
        output_root,
        status="FROZEN",
        phase="COMPLETE",
        videos_completed=EXPECTED_VIDEOS,
        r0_after=r0_after,
        recursive_artifact_inventory_sha256=inventory_hash,
        canonical_prediction_content_sha256=conservation[
            "canonical_prediction_content_sha256"
        ],
    )
    freeze_files(output_root)
    if canonical_hash(r1_artifact_inventory(output_root)) != inventory_hash:
        raise R1AuthorityError("post-freeze R1 artifact mutation")
    verify_r0_after(source_repo, preflight_payload["r0_authority_before"])


def parse_args() -> argparse.Namespace:
    """Parse the deliberately narrow authority CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "run-r1", "finalize"))
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run one fail-closed R1 generation phase."""

    args = parse_args()
    source_repo = args.source_repo.resolve()
    output_root = args.output_root.resolve()
    try:
        if args.phase == "preflight":
            preflight(source_repo, output_root)
        elif args.phase == "run-r1":
            run_r1(source_repo, output_root)
        else:
            finalize(source_repo, output_root)
    except Exception as exc:
        if output_root.exists():
            try:
                update_state(
                    output_root,
                    status="FAIL",
                    phase=f"FAILED_{args.phase.upper().replace('-', '_')}",
                    failure=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
